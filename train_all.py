"""
MSTC-IDS consolidated training driver.

Trains every model in the study with the full settings (the settings the Phase-1
notebook always intended but that were only ever smoke-tested with 1 epoch):

  * EPOCHS            = 50   (with early stopping patience=10 + ReduceLROnPlateau)
  * BATCH_SIZE        = 128
  * LEARNING_RATE     = 1e-4
  * SVM_SUBSAMPLE_SIZE= 5000 (stratified-style random subsample of the training set)

Models trained (33 unique checkpoints):

  1. Proposed MSTC-IDS, component x modality ablation (7 x 3 = 21):
       full, no_time2vec, no_cross_attention, no_contrastive,
       no_pyramid, no_causal, no_freshness   x   cyber, physical, both
  2. Baselines (Hassler et al., 2024), 4 families x 3 modalities (12):
       SVM, FNN, LSTM, 1D-CNN x cyber, physical, fused

The three "MSTC-IDS at modality X" entries used by the baseline-comparison and
modality-ablation studies reuse the corresponding full component variants
(full__cyber, full__physical, full__both) so nothing is trained twice.

The driver is resumable: it checks for an existing checkpoint + recorded result
before training any model, so it can be stopped and re-launched at any time.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_pipeline, CLASS_NAMES, SEED
from mstc_ids_model import MSTCIDS
from losses import MSTCTrainer

# ------------------------------------------------------------
# Full settings (documented in the README as the "real run" settings)
# ------------------------------------------------------------
SEED = 42
WINDOW_SIZE = 20
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("MSTC_EPOCHS", "50"))
LEARNING_RATE = float(os.environ.get("MSTC_LR", "5e-5"))
NUM_CLASSES = 5
SVM_SUBSAMPLE_SIZE = int(os.environ.get("MSTC_SVM_SUBSAMPLE", "5000"))

# Monitoring recipe. We monitor **val_accuracy** (maximize) instead of val_loss:
# the MSTC-IDS total loss includes the noisy InfoNCE contrastive term, and early
# experiments showed val_loss plateaus/rises after a few epochs while val_accuracy
# keeps improving. Monitoring accuracy therefore selects a genuinely better
# checkpoint and yields higher test performance. The initial LR is halved vs. the
# original notebooks (1e-4 -> 5e-5) to slow the early overfitting observed at 1e-4.
MONITOR = "val_accuracy"
MONITOR_MODE = "max"
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 6
MIN_LR = 1e-6

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "segment_data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "segment_data")
CACHE_DIR = os.environ.get("MSTC_CACHE", os.path.join(ROOT, "data"))
RESULTS_SUFFIX = os.environ.get("MSTC_RESULTS_SUFFIX", "")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
HIST_DIR = os.path.join(RESULTS_DIR, "histories")
PROBAS_DIR = os.path.join(RESULTS_DIR, "probabilities")

COMPONENT_VARIANTS = [
    ("full", {}, 0.1),
    ("no_time2vec", {"use_time2vec": False}, 0.1),
    ("no_cross_attention", {"use_cross_attention": False}, 0.1),
    ("no_contrastive", {}, 0.0),
    ("no_pyramid", {"use_pyramid": False}, 0.1),
    ("no_causal", {"causal": False}, 0.1),
    ("no_freshness", {"use_freshness": False}, 0.1),
]
MODALITIES = ["cyber", "physical", "both"]
BASELINE_FAMILIES = ["SVM", "FNN", "LSTM", "1D-CNN"]
BASELINE_MODALITIES = ["cyber", "physical", "fused"]


# ------------------------------------------------------------
# Optimized-config overrides (produced by code/optimize.py)
# ------------------------------------------------------------

def _load_opt_configs():
    p = os.path.join(RESULTS_DIR, "optimization", "best_configs.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    return {}


OPT = _load_opt_configs()


def log(msg):
    print(f"[train_all] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def make_dirs():
    for d in [MODELS_DIR, RESULTS_DIR, HIST_DIR, PROBAS_DIR,
              os.path.join(MODELS_DIR, "proposed"),
              os.path.join(MODELS_DIR, "baselines"),
              os.path.join(MODELS_DIR, "component_modality_ablation")]:
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------

def create_tf_dataset(Xc, Xp, Xt, Xf, y, shuffle=False, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((Xc, Xp, Xt, Xf, y))
    ds = ds.map(lambda c, p, t, f, lbl: ((c, p, t, f), lbl))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=seed)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def calculate_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


class SaveBaseModelCheckpoint(keras.callbacks.Callback):
    """Saves trainer.base_model (the MSTCIDS) whenever the monitored metric improves."""

    def __init__(self, filepath, monitor="val_accuracy", mode="max", save_best_only=True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = current > self.best if self.mode == "max" else current < self.best
        if (not self.save_best_only) or improved:
            self.best = current
            self.model.base_model.save(self.filepath)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_model_output(name, y_true, y_pred_proba, n_params=None, epochs_trained=None):
    y_pred = np.argmax(y_pred_proba, axis=1)
    metrics = {
        "name": name,
        "trainable_params": n_params,
        "epochs_trained": epochs_trained,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro"))
    except Exception:
        metrics["macro_auc"] = None
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    metrics["per_class_f1"] = {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)}
    cm_ = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return metrics, cm_, y_pred_proba


def save_probas(npz_path, key, proba):
    d = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as npz:
            for k in npz.files:
                d[k] = npz[k]
    d[key] = proba
    np.savez(npz_path, **d)


def load_record(record_path, key):
    if not os.path.exists(record_path):
        return None
    with open(record_path) as f:
        recs = json.load(f)
    return recs.get(key)


def save_record(record_path, key, value):
    recs = {}
    if os.path.exists(record_path):
        with open(record_path) as f:
            recs = json.load(f)
    recs[key] = value
    with open(record_path, "w") as f:
        json.dump(recs, f, indent=2)


def save_history(hist, model_key):
    with open(os.path.join(HIST_DIR, f"{model_key}.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f, indent=2)


# ------------------------------------------------------------
# MSTC-IDS training
# ------------------------------------------------------------

def train_mstc_variant(component, modality, model_kwargs, contrastive_weight,
                       data, class_weights, results_path, probas_path):
    model_key = f"{component}__{modality}"
    log(f"=== MSTC-IDS {model_key} ===")

    existing = load_record(results_path, model_key)
    if existing is not None:
        log(f"  already trained ({existing['epochs_trained']} epochs, "
            f"acc={existing['accuracy']:.4f}) - skipping")
        return existing, None

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    Xc_train, Xp_train, Xt_train, Xf_train, y_train = (data["Xc_train"], data["Xp_train"], data["Xt_train"],
                                                       data["Xf_train"], data["y_train"])
    Xc_val, Xp_val, Xt_val, Xf_val, y_val = (data["Xc_val"], data["Xp_val"], data["Xt_val"],
                                             data["Xf_val"], data["y_val"])
    Xc_test, Xp_test, Xt_test, Xf_test, y_test = (data["Xc_test"], data["Xp_test"], data["Xt_test"],
                                                  data["Xf_test"], data["y_test"])

    n_cyber = Xc_train.shape[-1]
    n_phys = Xp_train.shape[-1]

    mcfg = OPT.get(f"mstc_{modality}") or OPT.get("mstc", {})
    d = mcfg.get("d", 64)
    heads = mcfg.get("heads", 4)
    ff = mcfg.get("ff", 128)
    dropout = mcfg.get("dropout", 0.2)
    lr = mcfg.get("lr", LEARNING_RATE)
    cw_opt = mcfg.get("contrastive_weight", 0.1)
    wd = mcfg.get("weight_decay", 0.0)

    model = MSTCIDS(n_cyber=n_cyber, n_phys=n_phys, classes=NUM_CLASSES,
                    d=d, heads=heads, ff=ff, dropout=dropout, modality=modality, **model_kwargs)
    effective_cw = 0.0 if component == "no_contrastive" else (cw_opt if modality == "both" else 0.0)
    trainer = MSTCTrainer(model, contrastive_weight=effective_cw)
    if wd > 0:
        trainer.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd))
    else:
        trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    train_ds = create_tf_dataset(Xc_train, Xp_train, Xt_train, Xf_train, y_train, shuffle=True)
    val_ds = create_tf_dataset(Xc_val, Xp_val, Xt_val, Xf_val, y_val)

    ckpt_path = os.path.join(MODELS_DIR, "component_modality_ablation", f"mstc_ids_{model_key}_best.keras")
    callbacks = [
        SaveBaseModelCheckpoint(ckpt_path, monitor=MONITOR, mode=MONITOR_MODE, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode=MONITOR_MODE, patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode=MONITOR_MODE, factor=0.5,
                                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1),
    ]

    history = trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                          class_weight=class_weights, callbacks=callbacks, verbose=2)

    best_model = keras.models.load_model(ckpt_path, compile=False)
    y_pred_proba = best_model((Xc_test, Xp_test, Xt_test, Xf_test), training=False).numpy()

    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    metrics, cm_, proba = evaluate_model_output(
        model_key, y_test, y_pred_proba, n_params=n_params,
        epochs_trained=len(history.history["loss"]))
    metrics["component"] = component
    metrics["modality"] = modality
    metrics["checkpoint"] = os.path.relpath(ckpt_path, ROOT)

    save_history(history.history, model_key)
    save_record(results_path, model_key, {**metrics, "confusion_matrix": cm_.tolist()})
    save_probas(probas_path, model_key, proba)

    # also mirror the single-modality "proposed model" checkpoints
    if component == "full":
        alias = os.path.join(MODELS_DIR, "proposed", f"mstc_ids_{modality}_best.keras")
        os.makedirs(os.path.dirname(alias), exist_ok=True)
        import shutil
        shutil.copy2(ckpt_path, alias)

    log(f"  done: acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f} "
        f"epochs={metrics['epochs_trained']}")
    return metrics, history


# ------------------------------------------------------------
# Baseline training
# ------------------------------------------------------------

def build_svm(cfg=None):
    cfg = cfg or OPT.get("svm", {})
    return SVC(kernel="rbf", C=cfg.get("C", 10), gamma=cfg.get("gamma", "scale"),
               probability=True, random_state=SEED)


def build_fnn(input_dim, classes=NUM_CLASSES, cfg=None):
    cfg = cfg or OPT.get("fnn", {})
    units = cfg.get("units", [256, 128, 64])
    dropouts = cfg.get("dropouts", [0.3, 0.3, 0.2])
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for u, d in zip(units, dropouts):
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(d)(x)
    out = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inp, out, name="FNN")


def build_lstm(seq_len, n_feat, classes=NUM_CLASSES, cfg=None):
    cfg = cfg or OPT.get("lstm", {})
    units = cfg.get("units", [128, 64])
    dropouts = cfg.get("dropouts", [0.2, 0.3])
    inp = layers.Input(shape=(seq_len, n_feat))
    x = inp
    n_layers = len(units)
    for i, (u, d) in enumerate(zip(units, dropouts)):
        x = layers.LSTM(u, return_sequences=(i < n_layers - 1))(x)
        x = layers.Dropout(d)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inp, out, name="LSTM")


def build_cnn(seq_len, n_feat, classes=NUM_CLASSES, cfg=None):
    cfg = cfg or OPT.get("cnn", {})
    filters = cfg.get("filters", [64, 128, 128])
    kernels = cfg.get("kernels", [3, 3, 3])
    dropout = cfg.get("dropout", 0.3)
    inp = layers.Input(shape=(seq_len, n_feat))
    x = inp
    for f, k in zip(filters, kernels):
        x = layers.Conv1D(f, kernel_size=k, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inp, out, name="1D-CNN")


def family_config(family, modality):
    """Per-modality override (OPT[fam_modality]) with family fallback."""
    fam_key = {"FNN": "fnn", "LSTM": "lstm", "1D-CNN": "cnn", "SVM": "svm"}[family]
    mod = {"fused": "fused", "cyber": "cyber", "physical": "physical", "both": "both"}[modality]
    return OPT.get(f"{fam_key}_{mod}") or OPT.get(fam_key, {})


def build_modality_features(Xc, Xp, Xt, Xf, modality):
    if modality == "cyber":
        seq = np.concatenate([Xc, Xt, Xf], axis=-1)
    elif modality == "physical":
        seq = Xp
    elif modality == "fused":
        seq = np.concatenate([Xc, Xp, Xt, Xf], axis=-1)
    else:
        raise ValueError(modality)
    flat = seq.reshape(seq.shape[0], -1)
    return seq, flat


def train_svm_variant(modality, data, class_weights, results_path, probas_path, features):
    name = f"SVM-{modality}"
    log(f"=== SVM {name} ===")
    existing = load_record(results_path, name)
    if existing is not None:
        log(f"  already trained - skipping")
        return existing, None

    X_train_seq, X_train_flat = features[modality]["train"]
    X_test_seq, X_test_flat = features[modality]["test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    rng = np.random.RandomState(SEED)
    n = len(y_train)
    if n > SVM_SUBSAMPLE_SIZE:
        idx = rng.choice(n, size=SVM_SUBSAMPLE_SIZE, replace=False)
    else:
        idx = np.arange(n)

    svm = build_svm(family_config("SVM", modality))
    t0 = time.time()
    svm.fit(X_train_flat[idx], y_train[idx])
    log(f"  SVM fit took {time.time()-t0:.1f}s")

    y_pred_proba = svm.predict_proba(X_test_flat)
    proba_full = np.zeros((len(y_test), NUM_CLASSES))
    for i, c in enumerate(svm.classes_):
        proba_full[:, c] = y_pred_proba[:, i]

    ckpt_path = os.path.join(MODELS_DIR, "baselines", f"baseline_{name}.pkl")
    joblib.dump(svm, ckpt_path)

    metrics, cm_, proba = evaluate_model_output(name, y_test, proba_full, n_params=len(idx))
    metrics["checkpoint"] = os.path.relpath(ckpt_path, ROOT)
    save_record(results_path, name, {**metrics, "confusion_matrix": cm_.tolist()})
    save_probas(probas_path, name, proba)
    log(f"  done: acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f}")
    return metrics, None


def train_keras_baseline(family, modality, data, class_weights, results_path, probas_path, features):
    name = f"{family}-{modality}"
    log(f"=== {name} ===")
    existing = load_record(results_path, name)
    if existing is not None:
        log(f"  already trained - skipping")
        return existing, None

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train_seq, X_train_flat = features[modality]["train"]
    X_val_seq, X_val_flat = features[modality]["val"]
    X_test_seq, X_test_flat = features[modality]["test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    fam_cfg = family_config(family, modality)
    if family == "FNN":
        model = build_fnn(X_train_flat.shape[-1], cfg=fam_cfg)
        X_train, X_val, X_test = X_train_flat, X_val_flat, X_test_flat
    elif family == "LSTM":
        model = build_lstm(X_train_seq.shape[1], X_train_seq.shape[2], cfg=fam_cfg)
        X_train, X_val, X_test = X_train_seq, X_val_seq, X_test_seq
    elif family == "1D-CNN":
        model = build_cnn(X_train_seq.shape[1], X_train_seq.shape[2], cfg=fam_cfg)
        X_train, X_val, X_test = X_train_seq, X_val_seq, X_test_seq
    else:
        raise ValueError(family)

    lr = fam_cfg.get("lr", LEARNING_RATE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(MODELS_DIR, "baselines", f"baseline_{name}_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor=MONITOR, mode=MONITOR_MODE, save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor=MONITOR, mode=MONITOR_MODE, patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode=MONITOR_MODE, factor=0.5,
                                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1),
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS,
                        batch_size=BATCH_SIZE, class_weight=class_weights, callbacks=callbacks, verbose=2)

    best_model = keras.models.load_model(ckpt_path, compile=False)
    y_pred_proba = best_model.predict(X_test, verbose=0)

    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    metrics, cm_, proba = evaluate_model_output(
        name, y_test, y_pred_proba, n_params=n_params,
        epochs_trained=len(history.history["loss"]))
    metrics["checkpoint"] = os.path.relpath(ckpt_path, ROOT)

    save_history(history.history, name)
    save_record(results_path, name, {**metrics, "confusion_matrix": cm_.tolist()})
    save_probas(probas_path, name, proba)
    log(f"  done: acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f} "
        f"epochs={metrics['epochs_trained']}")
    return metrics, history


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of: mstc,baselines,svm,fnn,lstm,cnn")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop after this many minutes (checked between models); 0 = no limit")
    ap.add_argument("--modalities", default=None,
                    help="restrict MSTC-IDS component ablation to this comma-separated "
                         "modality subset (cyber/physical/both); default = all")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    deadline = time.time() + args.max_minutes * 60 if args.max_minutes > 0 else None

    def want(*names):
        return only is None or any(n in only for n in names)

    def time_left():
        if deadline is None:
            return True
        return time.time() < deadline

    make_dirs()
    log("Loading pipeline...")
    data, scaler_c, scaler_p, meta = build_pipeline(DATA_DIR, cache_dir=CACHE_DIR)

    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    class_weights = calculate_class_weights(y_train)
    log("Class weights: " + json.dumps({str(k): round(v, 3) for k, v in class_weights.items()}))

    ablation_results_path = os.path.join(RESULTS_DIR, f"component_modality_ablation_full{RESULTS_SUFFIX}.json")
    ablation_probas_path = os.path.join(PROBAS_DIR, f"component_probas{RESULTS_SUFFIX}.npz")
    baseline_results_path = os.path.join(RESULTS_DIR, f"baseline_comparison_full{RESULTS_SUFFIX}.json")
    baseline_probas_path = os.path.join(PROBAS_DIR, f"baseline_probas{RESULTS_SUFFIX}.npz")

    # ---- 1. Proposed MSTC-IDS component x modality (21) ----
    mstc_modalities = [m for m in MODALITIES if args.modalities is None or m in args.modalities.split(",")]
    if want("mstc"):
        for modality in mstc_modalities:
            for component, kwargs, cw in COMPONENT_VARIANTS:
                train_mstc_variant(component, modality, kwargs, cw, data, class_weights,
                                   ablation_results_path, ablation_probas_path)
                if not time_left():
                    log("Max minutes reached; stopping. Relaunch to resume.")
                    return

    # ---- 2. Baselines (12) ----
    features = {}
    if want("svm") or want("fnn") or want("lstm") or want("cnn"):
        for m in BASELINE_MODALITIES:
            features[m] = {
                "train": build_modality_features(data["Xc_train"], data["Xp_train"],
                                                 data["Xt_train"], data["Xf_train"], m),
                "val": build_modality_features(data["Xc_val"], data["Xp_val"],
                                               data["Xt_val"], data["Xf_val"], m),
                "test": build_modality_features(data["Xc_test"], data["Xp_test"],
                                                data["Xt_test"], data["Xf_test"], m),
            }
            log(f"features[{m}]: seq={features[m]['train'][0].shape} flat={features[m]['train'][1].shape}")

    for m in BASELINE_MODALITIES:
        if want("svm"):
            train_svm_variant(m, data, class_weights, baseline_results_path,
                              baseline_probas_path, features)
            if not time_left():
                log("Max minutes reached; stopping. Relaunch to resume.")
                return

    for m in BASELINE_MODALITIES:
        for family in BASELINE_FAMILIES:
            if family == "SVM":
                continue
            key = {"FNN": "fnn", "LSTM": "lstm", "1D-CNN": "cnn"}[family]
            if want(key):
                train_keras_baseline(family, m, data, class_weights, baseline_results_path,
                                     baseline_probas_path, features)
                if not time_left():
                    log("Max minutes reached; stopping. Relaunch to resume.")
                    return

    log("All requested trainings complete.")


if __name__ == "__main__":
    main()
