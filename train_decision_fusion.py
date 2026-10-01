"""Train the gated decision-fusion (late-fusion) variant of MSTC-IDS.

Motivation: the early (cross-attention + concat) fusion scores ~0.788 on the
cyber-physical test set, i.e. *below* its own physical-only branch (0.871).
Decision-level fusion of the two branches reaches ~0.87. This script trains a
single model that has one classification head per branch plus a learned
per-sample gate, with modality dropout, so the fused model can never be dragged
below the best branch.

Checkpoint selection uses val accuracy (same as the rest of the study). We run
several seeds because the pipeline has a ~1-2% run-to-run noise floor and report
mean +/- std.
"""

import os
import sys
import json
import time
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_pipeline
from mstc_ids_model import MSTCIDS
from losses import MSTCTrainer
import train_all as TA
from tune_proposed import ValMonitorCheckpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("MSTC_CACHE", os.path.join(ROOT, "data"))
OUT_JSON = os.environ.get("MSTC_DF_JSON",
                          os.path.join(ROOT, "results", "optimization", "decision_fusion.json"))
CKPT_DIR = os.environ.get("MSTC_DF_CKPT",
                          os.path.join(ROOT, "models", "decision_fusion"))
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = int(os.environ.get("MSTC_EPOCHS", "50"))
BATCH_SIZE = 128
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 6
MIN_LR = 1e-6
NUM_CLASSES = 5

BASE = dict(d=64, heads=4, ff=128, dropout=0.4, lr=5e-5, wd=1e-4, cw=0.0,
            use_cross_attention=False, use_decision_fusion=True)


def log(msg):
    print(f"[df] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_out():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {}


def save_out(d):
    with open(OUT_JSON, "w") as f:
        json.dump(d, f, indent=2)


def train_one(tag, seed, modality_dropout, data, class_weights, cfg=BASE):
    log(f"=== {tag} (seed={seed}, md={modality_dropout}) ===")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = MSTCIDS(
        n_cyber=data["Xc_train"].shape[-1],
        n_phys=data["Xp_train"].shape[-1],
        classes=NUM_CLASSES,
        d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], dropout=cfg["dropout"],
        modality="both",
        use_cross_attention=cfg["use_cross_attention"],
        use_decision_fusion=cfg["use_decision_fusion"],
        modality_dropout=modality_dropout,
    )
    trainer = MSTCTrainer(model, contrastive_weight=cfg["cw"])
    opt = (keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["wd"])
           if cfg["wd"] > 0 else keras.optimizers.Adam(learning_rate=cfg["lr"]))
    trainer.compile(optimizer=opt)

    train_ds = TA.create_tf_dataset(data["Xc_train"], data["Xp_train"],
                                    data["Xt_train"], data["Xf_train"],
                                    data["y_train"], shuffle=True, seed=seed)
    val_ds = TA.create_tf_dataset(data["Xc_val"], data["Xp_val"],
                                  data["Xt_val"], data["Xf_val"], data["y_val"])
    val_inputs = (data["Xc_val"], data["Xp_val"], data["Xt_val"], data["Xf_val"])

    ckpt = os.path.join(CKPT_DIR, f"{tag}.keras")
    callbacks = [
        ValMonitorCheckpoint(val_inputs, data["y_val"], ckpt,
                             monitor="val_accuracy"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=False, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                          factor=0.5, patience=REDUCE_LR_PATIENCE,
                                          min_lr=MIN_LR, verbose=1),
    ]
    t0 = time.time()
    hist = trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                       class_weight=class_weights, callbacks=callbacks, verbose=2)

    best = keras.models.load_model(ckpt, compile=False)
    xte = (data["Xc_test"], data["Xp_test"], data["Xt_test"], data["Xf_test"])
    pred = best(xte, training=False).numpy()
    yp = pred.argmax(1)
    rec = {
        "tag": tag, "seed": seed, "modality_dropout": modality_dropout,
        "best_val_accuracy": float(max(hist.history["val_accuracy"])),
        "test_accuracy": float(accuracy_score(data["y_test"], yp)),
        "test_f1_weighted": float(f1_score(data["y_test"], yp,
                                           average="weighted", zero_division=0)),
        "test_f1_macro": float(f1_score(data["y_test"], yp,
                                        average="macro", zero_division=0)),
        "epochs": len(hist.history["loss"]),
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  {tag}: valAcc={rec['best_val_accuracy']:.4f} "
        f"testAcc={rec['test_accuracy']:.4f} testF1={rec['test_f1_weighted']:.4f} "
        f"epochs={rec['epochs']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--md", type=float, default=0.2,
                    help="modality dropout probability")
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()

    if args.select:
        out = load_out()
        vals = [r["test_accuracy"] for r in out.values()]
        f1s = [r["test_f1_weighted"] for r in out.values()]
        log(f"n={len(vals)} testAcc={np.mean(vals):.4f}+/-{np.std(vals):.4f} "
            f"testF1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")
        return

    log(f"Loading pipeline from {CACHE_DIR} ...")
    data, _, _, _ = build_pipeline(TA.DATA_DIR, cache_dir=CACHE_DIR)
    class_weights = TA.calculate_class_weights(data["y_train"])

    out = load_out()
    for seed in [int(s) for s in args.seeds.split(",")]:
        tag = f"df_md{args.md}_s{seed}"
        if tag in out:
            log(f"skip {tag}")
            continue
        out[tag] = train_one(tag, seed, args.md, data, class_weights)
        save_out(out)
    vals = [r["test_accuracy"] for r in out.values()]
    f1s = [r["test_f1_weighted"] for r in out.values()]
    log(f"summary n={len(vals)} testAcc={np.mean(vals):.4f}+/-{np.std(vals):.4f} "
        f"testF1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
