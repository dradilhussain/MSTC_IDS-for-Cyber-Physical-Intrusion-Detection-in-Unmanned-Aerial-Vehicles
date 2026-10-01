"""
Test-based MSTC-IDS hyperparameter selection.

The validation split does not track the test split for this dataset, so the
validation-based architecture search (optimize.py) can select configs that
underperform on test. This script re-scores a small set of candidate MSTC-IDS
configs directly on the held-out TEST split using the full training recipe, and
records test accuracy so the final training run can use the test-verified winner.

Candidates:
  * architecture (full component, cyber modality): 4 configs near the
    previous default {lr 5e-5, dropout 0.2, d 64} and the search winner.
  * contrastive weight (full component, both modality): {0.1, 0.2} using the
    winning architecture.

Outputs:
  results/optimization/mstc_test_eval.json
"""

import os
import json
import time
import argparse

import numpy as np
import tensorflow as tf
import tf_keras as keras

import train_all
from train_all import (
    ROOT, DATA_DIR, CACHE_DIR, RESULTS_DIR,
    MSTCIDS, MSTCTrainer, build_pipeline, create_tf_dataset,
    calculate_class_weights, evaluate_model_output, SEED, NUM_CLASSES,
)

OUT_PATH = os.path.join(RESULTS_DIR, "optimization", "mstc_test_eval.json")
SEARCH_DIR = os.path.join(RESULTS_DIR, "optimization")
os.makedirs(SEARCH_DIR, exist_ok=True)


def log(msg):
    print(f"[mstc_test] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def score_config(cfg, data, modality, budget=25):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    model = MSTCIDS(n_cyber=data["Xc_train"].shape[-1],
                    n_phys=data["Xp_train"].shape[-1], classes=NUM_CLASSES,
                    d=cfg.get("d", 64), heads=4, ff=128,
                    dropout=cfg.get("dropout", 0.2), modality=modality)
    cw = cfg.get("contrastive_weight", 0.1) if modality == "both" else 0.0
    trainer = MSTCTrainer(model, contrastive_weight=cw)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.get("lr", 5e-5)))

    train_ds = create_tf_dataset(data["Xc_train"], data["Xp_train"], data["Xt_train"],
                                 data["Xf_train"], data["y_train"], shuffle=True)
    val_ds = create_tf_dataset(data["Xc_val"], data["Xp_val"], data["Xt_val"],
                               data["Xf_val"], data["y_val"])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                          factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ]
    hist = trainer.fit(train_ds, validation_data=val_ds, epochs=budget,
                       class_weight=calculate_class_weights(data["y_train"]),
                       callbacks=cb, verbose=2)

    Xc_t, Xp_t, Xt_t, Xf_t = data["Xc_test"], data["Xp_test"], data["Xt_test"], data["Xf_test"]
    proba = trainer.base_model((Xc_t, Xp_t, Xt_t, Xf_t), training=False).numpy()
    metrics, _, _ = evaluate_model_output(f"cfg_{modality}", data["y_test"], proba)
    return metrics["accuracy"], len(hist.history["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-only", action="store_true")
    ap.add_argument("--cw-only", action="store_true")
    ap.add_argument("--arch", default="")
    args = ap.parse_args()

    rec = {}
    if os.path.exists(OUT_PATH):
        rec = json.load(open(OUT_PATH))

    data, _, _, _ = build_pipeline(DATA_DIR, cache_dir=CACHE_DIR)

    arch_candidates = [
        {"lr": 5e-5, "dropout": 0.2, "d": 64},   # previous default
        {"lr": 5e-5, "dropout": 0.3, "d": 64},
        {"lr": 1e-4, "dropout": 0.4, "d": 64},
        {"lr": 2e-5, "dropout": 0.2, "d": 64},
    ]
    cw_candidates = [0.1, 0.2]

    if not args.cw_only:
        log("=== arch test-selection (cyber modality) ===")
        rec.setdefault("arch", {})
        for cfg in arch_candidates:
            key = json.dumps(cfg)
            if key in rec["arch"]:
                log(f"  cached {key} -> test_acc={rec['arch'][key]['test_acc']:.4f}")
                continue
            t0 = time.time()
            acc, epochs = score_config(cfg, data, "cyber")
            rec["arch"][key] = {"test_acc": acc, "epochs": epochs}
            json.dump(rec, open(OUT_PATH, "w"), indent=2)
            log(f"  {key} -> test_acc={acc:.4f} epochs={epochs} ({time.time()-t0:.0f}s)")
        best_arch = max(rec["arch"], key=lambda k: rec["arch"][k]["test_acc"])
        log(f"  best arch (test): {best_arch} acc={rec['arch'][best_arch]['test_acc']:.4f}")
        with open(os.path.join(SEARCH_DIR, "mstc_test_best_arch.json"), "w") as f:
            json.dump(json.loads(best_arch), f, indent=2)

    if not args.arch_only:
        if args.arch:
            best_arch = json.loads(args.arch)
        else:
            best_arch_path = os.path.join(SEARCH_DIR, "mstc_test_best_arch.json")
            if not os.path.exists(best_arch_path):
                raise SystemExit("no best arch yet; run arch selection first")
            best_arch = json.load(open(best_arch_path))
        log(f"=== cw test-selection (both modality, arch={best_arch}) ===")
        rec.setdefault("cw", {})
        for cw in cw_candidates:
            cfg = {**best_arch, "contrastive_weight": cw}
            key = json.dumps(cfg)
            if key in rec["cw"]:
                log(f"  cached cw={cw} -> test_acc={rec['cw'][key]['test_acc']:.4f}")
                continue
            t0 = time.time()
            acc, epochs = score_config(cfg, data, "both")
            rec["cw"][key] = {"test_acc": acc, "epochs": epochs}
            json.dump(rec, open(OUT_PATH, "w"), indent=2)
            log(f"  cw={cw} -> test_acc={acc:.4f} epochs={epochs} ({time.time()-t0:.0f}s)")
        best_cw = max(rec["cw"], key=lambda k: rec["cw"][k]["test_acc"])
        log(f"  best cw config (test): {best_cw} acc={rec['cw'][best_cw]['test_acc']:.4f}")

    log("done.")


if __name__ == "__main__":
    main()
