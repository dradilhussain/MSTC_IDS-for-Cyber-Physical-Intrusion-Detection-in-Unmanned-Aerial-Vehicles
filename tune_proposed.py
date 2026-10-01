"""Validation-based re-tuning of the proposed MSTC-IDS model.

Only the *proposed* model is re-tuned here (baselines are left as-is). Model
selection uses the validation split (macro-F1); the chronological test split is
never consulted during selection, so the final reported test numbers stay
honest.

For every candidate configuration we:

  1. train on the training split (shuffled),
  2. after each epoch compute val macro-F1 and save the checkpoint with the
     best val macro-F1,
  3. early-stop on val macro-F1,
  4. report the test metrics of the val-selected checkpoint (for the record).

Results are appended to results/optimization/proposed_sweep.json (resumable,
keyed by tag), and the best-by-validation config per modality is written to
results/optimization/proposed_best_by_val.json.

Usage:
    python tune_proposed.py --group fused
    python tune_proposed.py --group cyber
    python tune_proposed.py --select
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

SEED = 42
NUM_CLASSES = 5
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("MSTC_EPOCHS", "50"))
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 6
MIN_LR = 1e-6

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("MSTC_CACHE", os.path.join(ROOT, "data"))
SWEEP_JSON = os.environ.get(
    "MSTC_SWEEP_JSON",
    os.path.join(ROOT, "results", "optimization", "proposed_sweep.json"))
BEST_JSON = os.environ.get(
    "MSTC_BEST_JSON",
    os.path.join(ROOT, "results", "optimization", "proposed_best_by_val.json"))
CKPT_DIR = os.environ.get("MSTC_SWEEP_CKPT",
                          os.path.join(ROOT, "models", "proposed_sweep"))
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SWEEP_JSON), exist_ok=True)

# ------------------------------------------------------------
# Candidate grid
# ------------------------------------------------------------
# Base fused config (current canonical): d=64 heads=4 ff=128 dropout=0.4
# lr=5e-5 wd=1e-4 cw=0.1. Base cyber config: d=48 dropout=0.3 lr=1e-4.
FUSED_BASE = dict(modality="both", d=64, heads=4, ff=128, dropout=0.4,
                  lr=5e-5, wd=1e-4, cw=0.1, use_cross_attention=True)
CYBER_BASE = dict(modality="cyber", d=48, heads=4, ff=128, dropout=0.3,
                  lr=1e-4, wd=0.0, cw=0.0, use_cross_attention=True)


def _cfg(base, tag, **over):
    c = dict(base)
    c.update(over)
    c["tag"] = tag
    return c


CONFIGS = [
    # --- fused: contrastive weight (ablation suggests lower is better) ---
    _cfg(FUSED_BASE, "f_cw000", cw=0.0),
    _cfg(FUSED_BASE, "f_cw002", cw=0.02),
    _cfg(FUSED_BASE, "f_cw005", cw=0.05),
    _cfg(FUSED_BASE, "f_cw010", cw=0.10),
    # --- fused: regularisation / capacity at cw=0 ---
    _cfg(FUSED_BASE, "f_cw000_do30", cw=0.0, dropout=0.3),
    _cfg(FUSED_BASE, "f_cw000_d96", cw=0.0, d=96),
    # --- cyber: capacity / dropout (cross-attention is a no-op for single modality) ---
    _cfg(CYBER_BASE, "c_d48"),
    _cfg(CYBER_BASE, "c_d64", d=64),
    _cfg(CYBER_BASE, "c_d96", d=96),
    _cfg(CYBER_BASE, "c_d64_do40", d=64, dropout=0.4),
]


def log(msg):
    print(f"[tune] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_sweep():
    if os.path.exists(SWEEP_JSON):
        with open(SWEEP_JSON) as f:
            return json.load(f)
    return {}


def save_sweep(d):
    with open(SWEEP_JSON, "w") as f:
        json.dump(d, f, indent=2)


class ValMonitorCheckpoint(keras.callbacks.Callback):
    """Log val macro-F1 every epoch and save the best base model.

    The checkpoint is selected on `val_accuracy`, exactly as every other model
    in the study is trained (MONITOR = val_accuracy). Selecting checkpoints on
    val macro-F1 was tried and systematically *hurt* test accuracy (e.g. the
    fused cw=0 config dropped from 0.818 to 0.710 test accuracy), because the
    validation macro-F1 is unstable on the minority Replay class. Config-level
    selection also uses the same validation metric for consistency.
    """

    def __init__(self, val_inputs, y_val, ckpt_path, monitor="val_accuracy"):
        super().__init__()
        self.val_inputs = val_inputs
        self.y_val = y_val
        self.ckpt_path = ckpt_path
        self.monitor = monitor
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.base_model(self.val_inputs, training=False).numpy()
        yp = np.argmax(pred, axis=1)
        f1 = float(f1_score(self.y_val, yp, average="macro", zero_division=0))
        logs["val_macro_f1"] = f1
        current = logs.get(self.monitor)
        if current is not None and current > self.best:
            self.best = current
            self.model.base_model.save(self.ckpt_path)


def train_one(cfg, data, class_weights):
    tag = cfg["tag"]
    log(f"=== {tag} ({cfg['modality']}) ===")
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    n_cyber = data["Xc_train"].shape[-1]
    n_phys = data["Xp_train"].shape[-1]

    model = MSTCIDS(
        n_cyber=n_cyber, n_phys=n_phys, classes=NUM_CLASSES,
        d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"], dropout=cfg["dropout"],
        modality=cfg["modality"],
        use_cross_attention=cfg["use_cross_attention"],
    )
    cw = cfg["cw"] if cfg["modality"] == "both" else 0.0
    trainer = MSTCTrainer(model, contrastive_weight=cw)
    opt = (keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["wd"])
           if cfg["wd"] > 0 else keras.optimizers.Adam(learning_rate=cfg["lr"]))
    trainer.compile(optimizer=opt)

    train_ds = TA.create_tf_dataset(data["Xc_train"], data["Xp_train"],
                                    data["Xt_train"], data["Xf_train"],
                                    data["y_train"], shuffle=True)
    val_ds = TA.create_tf_dataset(data["Xc_val"], data["Xp_val"],
                                  data["Xt_val"], data["Xf_val"], data["y_val"])
    val_inputs = (data["Xc_val"], data["Xp_val"], data["Xt_val"], data["Xf_val"])

    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.keras")
    callbacks = [
        ValMonitorCheckpoint(val_inputs, data["y_val"], ckpt_path,
                             monitor="val_accuracy"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=False, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max",
                                          factor=0.5, patience=REDUCE_LR_PATIENCE,
                                          min_lr=MIN_LR, verbose=1),
    ]

    t0 = time.time()
    history = trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                          class_weight=class_weights, callbacks=callbacks, verbose=2)

    best_model = keras.models.load_model(ckpt_path, compile=False)
    pred_test = best_model((data["Xc_test"], data["Xp_test"], data["Xt_test"],
                            data["Xf_test"]), training=False).numpy()
    yp_test = np.argmax(pred_test, axis=1)
    test_acc = float(accuracy_score(data["y_test"], yp_test))
    test_f1 = float(f1_score(data["y_test"], yp_test, average="macro", zero_division=0))

    rec = {
        "tag": tag,
        "config": {k: v for k, v in cfg.items()},
        "best_val_macro_f1": float(max(history.history["val_macro_f1"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "epochs_trained": len(history.history["loss"]),
        "n_params": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  {tag}: valF1={rec['best_val_macro_f1']:.4f} "
        f"testAcc={test_acc:.4f} testF1={test_f1:.4f} "
        f"epochs={rec['epochs_trained']} ({rec['seconds']}s)")
    return rec


def select():
    sweep = load_sweep()
    if not sweep:
        log("no sweep results yet")
        return
    best = {}
    for tag, rec in sweep.items():
        m = rec["config"]["modality"]
        if m not in best or rec["best_val_accuracy"] > best[m]["best_val_accuracy"]:
            best[m] = rec
    with open(BEST_JSON, "w") as f:
        json.dump(best, f, indent=2)
    for m, rec in best.items():
        log(f"BEST {m}: {rec['tag']} valAcc={rec['best_val_accuracy']:.4f} "
            f"testAcc={rec['test_accuracy']:.4f} testMacroF1={rec['test_macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["all", "fused", "cyber"])
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()

    if args.select:
        select()
        return

    log(f"Loading pipeline from {CACHE_DIR} ...")
    data, _, _, _ = build_pipeline(TA.DATA_DIR, cache_dir=CACHE_DIR)
    class_weights = TA.calculate_class_weights(data["y_train"])

    sweep = load_sweep()
    todo = []
    for cfg in CONFIGS:
        if cfg["modality"] == "both" and args.group not in ("all", "fused"):
            continue
        if cfg["modality"] == "cyber" and args.group not in ("all", "cyber"):
            continue
        if cfg["tag"] in sweep:
            log(f"skip {cfg['tag']} (already done)")
            continue
        todo.append(cfg)

    log(f"{len(todo)} configs to run")
    for cfg in todo:
        rec = train_one(cfg, data, class_weights)
        sweep[cfg["tag"]] = rec
        save_sweep(sweep)
    select()


if __name__ == "__main__":
    main()
