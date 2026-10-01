"""
MSTC-IDS hyperparameter / architecture optimization.

Bounded, validation-based search over each model family, so every model in the
study can then be retrained with a better, consistent configuration:

  * SVM:      grid search over (C, gamma), ranked by mean validation accuracy
              across all 3 modalities (cheap; subsample fit).
  * FNN/LSTM/1D-CNN: search over architecture configs (units/filters/kernels/
              dropouts/learning-rate) trained on the fused modality with a short
              epoch budget, ranked by validation accuracy.
  * MSTC-IDS: search over (lr, dropout, d_model) on the cyber modality (the
              hardest, with the most headroom; single-modality is also the
              cheapest), then a small contrastive-weight search on the both
              modality. The winning config is shared by all 21 component
              variants so the ablation study stays unconfounded.

Results are written to results/optimization/search_results.json (every trial)
and results/optimization/best_configs.json (winner per family, consumed by
train_all.py). Resumable: trials already scored are skipped.

Usage:
  python3 optimize.py --only svm,fnn,lstm,cnn,mstc [--max-minutes 48]
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_all import (
    ROOT, DATA_DIR, CACHE_DIR, RESULTS_DIR, OPT,
    build_pipeline, create_tf_dataset, calculate_class_weights,
    build_modality_features, build_svm, build_fnn, build_lstm, build_cnn,
    SEED, NUM_CLASSES, SVM_SUBSAMPLE_SIZE,
)

SEARCH_DIR = os.path.join(RESULTS_DIR, "optimization")
os.makedirs(SEARCH_DIR, exist_ok=True)
SEARCH_RECORD = os.path.join(SEARCH_DIR, "search_results.json")
BEST_CONFIGS = os.path.join(SEARCH_DIR, "best_configs.json")


def _tagged_paths(tag):
    """Tag-isolated record/config files so parallel workers never race."""
    suffix = f".{tag}" if tag else ""
    return (os.path.join(SEARCH_DIR, f"search_results{suffix}.json"),
            os.path.join(SEARCH_DIR, f"best_configs{suffix}.json"))


def log(msg):
    print(f"[optimize] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_record():
    if os.path.exists(SEARCH_RECORD):
        with open(SEARCH_RECORD) as f:
            return json.load(f)
    return {}


def save_record(rec):
    with open(SEARCH_RECORD, "w") as f:
        json.dump(rec, f, indent=2)


def trial_done(rec, family, key):
    return family in rec and key in rec[family]


# ------------------------------------------------------------
# SVM grid search
# ------------------------------------------------------------

def search_svm(data, features, rec, deadline):
    log("=== SVM grid search ===")
    rng = np.random.RandomState(SEED)
    grid = [{"C": C, "gamma": g} for C in [1, 5, 10, 50] for g in ["scale", 0.01]]
    accs = {json.dumps(cfg): [] for cfg in grid}
    y_val = data["y_val"]
    for cfg in grid:
        key = json.dumps(cfg)
        if trial_done(rec, "svm", key):
            accs[key] = rec["svm"][key]["val_accs"]
            continue
        for m in ["cyber", "physical", "fused"]:
            X_tr = features[m]["train"][1]
            X_va = features[m]["val"][1]
            y_tr = data["y_train"]
            n = len(y_tr)
            idx = rng.choice(n, size=min(n, 2500), replace=False)
            clf = SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"],
                      probability=True, random_state=SEED)
            clf.fit(X_tr[idx], y_tr[idx])
            p = clf.predict(X_va)
            accs[key].append(float(accuracy_score(y_val, p)))
            log(f"  C={cfg['C']:>4} gamma={cfg['gamma']!s:>6} {m:>8}: val_acc={accs[key][-1]:.4f}")
        rec.setdefault("svm", {})[key] = {"config": cfg, "val_accs": accs[key]}
        save_record(rec)
        if deadline is not None and time.time() > deadline:
            log("  max-minutes reached in SVM search; stopping.")
            return False
    means = {k: float(np.mean(v)) for k, v in accs.items()}
    best_key = max(means, key=means.get)
    best_cfg = json.loads(best_key)
    log(f"  best SVM config: {best_cfg} (mean val_acc={means[best_key]:.4f})")
    return best_cfg


# ------------------------------------------------------------
# Keras baseline search (on the fused modality, short budget)
# ------------------------------------------------------------

def score_baseline_config(family, cfg, data, features):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    m = "fused"
    X_tr_seq, X_tr_flat = features[m]["train"]
    X_va_seq, X_va_flat = features[m]["val"]
    y_tr, y_va = data["y_train"], data["y_val"]
    import train_all
    fam_key = {"FNN": "fnn", "LSTM": "lstm", "1D-CNN": "cnn"}[family]
    train_all.OPT[fam_key] = cfg
    if family == "FNN":
        model = build_fnn(X_tr_flat.shape[-1])
        X_tr, X_va = X_tr_flat, X_va_flat
    elif family == "LSTM":
        model = build_lstm(X_tr_seq.shape[1], X_tr_seq.shape[2])
        X_tr, X_va = X_tr_seq, X_va_seq
    elif family == "1D-CNN":
        model = build_cnn(X_tr_seq.shape[1], X_tr_seq.shape[2])
        X_tr, X_va = X_tr_seq, X_va_seq
    else:
        raise ValueError(family)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.get("lr", 5e-4)),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=8, restore_best_weights=True, verbose=0),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=25, batch_size=128, class_weight=calculate_class_weights(y_tr),
                     callbacks=cb, verbose=0)
    val_acc = float(max(hist.history["val_accuracy"]))
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    return val_acc, n_params


def search_keras_baseline(family, rec, deadline):
    log(f"=== {family} architecture search (fused modality, 25-epoch budget) ===")
    spaces = {
        "FNN": [
            {"units": [512, 256, 128], "dropouts": [0.4, 0.4, 0.3], "lr": 5e-4},
            {"units": [256, 128], "dropouts": [0.3, 0.3], "lr": 1e-3},
            {"units": [256, 256, 128], "dropouts": [0.3, 0.3, 0.3], "lr": 5e-4},
            {"units": [512, 256, 128], "dropouts": [0.3, 0.3, 0.2], "lr": 1e-3},
            {"units": [384, 256, 128], "dropouts": [0.4, 0.4, 0.3], "lr": 5e-4},
            {"units": [512, 256, 256, 128], "dropouts": [0.4, 0.3, 0.3, 0.2], "lr": 5e-4},
        ],
        "LSTM": [
            {"units": [128, 64], "dropouts": [0.3, 0.3], "lr": 5e-4},
            {"units": [256, 128], "dropouts": [0.3, 0.3], "lr": 5e-4},
            {"units": [256, 128], "dropouts": [0.4, 0.4], "lr": 1e-3},
            {"units": [128, 128], "dropouts": [0.3, 0.3], "lr": 5e-4},
            {"units": [256, 256], "dropouts": [0.3, 0.3], "lr": 5e-4},
            {"units": [128], "dropouts": [0.3], "lr": 5e-4},
        ],
        "1D-CNN": [
            {"filters": [64, 128], "kernels": [3, 3], "dropout": 0.3, "lr": 5e-4},
            {"filters": [32, 64, 128], "kernels": [3, 3, 3], "dropout": 0.3, "lr": 5e-4},
            {"filters": [128, 256], "kernels": [5, 3], "dropout": 0.3, "lr": 5e-4},
            {"filters": [64, 128, 256], "kernels": [3, 3, 3], "dropout": 0.3, "lr": 1e-3},
            {"filters": [64, 128], "kernels": [5, 5], "dropout": 0.4, "lr": 5e-4},
            {"filters": [128, 256], "kernels": [3, 5], "dropout": 0.4, "lr": 5e-4},
        ],
    }
    data, _, _, _ = build_pipeline(DATA_DIR, cache_dir=CACHE_DIR)
    features = {}
    for m in ["cyber", "physical", "fused"]:
        features[m] = {
            "train": build_modality_features(data["Xc_train"], data["Xp_train"],
                                             data["Xt_train"], data["Xf_train"], m),
            "val": build_modality_features(data["Xc_val"], data["Xp_val"],
                                           data["Xt_val"], data["Xf_val"], m),
        }
    results = []
    for cfg in spaces[family]:
        key = json.dumps(cfg)
        if trial_done(rec, family.lower(), key):
            val_acc = rec[family.lower()][key]["val_acc"]
            n_params = rec[family.lower()][key]["n_params"]
        else:
            val_acc, n_params = score_baseline_config(family, cfg, data, features)
            rec.setdefault(family.lower(), {})[key] = {"config": cfg, "val_acc": val_acc,
                                                       "n_params": n_params}
            save_record(rec)
        results.append((val_acc, cfg, n_params))
        log(f"  {key[:70]} -> val_acc={val_acc:.4f} params={n_params:,}")
        if deadline is not None and time.time() > deadline:
            log(f"  max-minutes reached in {family} search; stopping.")
            return False
    results.sort(key=lambda r: r[0], reverse=True)
    best_cfg = results[0][1]
    log(f"  best {family}: {best_cfg} (val_acc={results[0][0]:.4f})")
    return best_cfg


# ------------------------------------------------------------
# MSTC-IDS search
# ------------------------------------------------------------

def score_mstc_config(cfg, data, modality, budget=20):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    from train_all import MSTCIDS, MSTCTrainer
    n_cyber = data["Xc_train"].shape[-1]
    n_phys = data["Xp_train"].shape[-1]
    model = MSTCIDS(n_cyber=n_cyber, n_phys=n_phys, classes=NUM_CLASSES,
                    d=cfg.get("d", 64), heads=4, ff=128, dropout=cfg.get("dropout", 0.2),
                    modality=modality)
    cw = cfg.get("contrastive_weight", 0.1) if modality == "both" else 0.0
    trainer = MSTCTrainer(model, contrastive_weight=cw)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.get("lr", 5e-5)))
    train_ds = create_tf_dataset(data["Xc_train"], data["Xp_train"], data["Xt_train"],
                                 data["Xf_train"], data["y_train"], shuffle=True)
    val_ds = create_tf_dataset(data["Xc_val"], data["Xp_val"], data["Xt_val"], data["Xf_val"],
                               data["y_val"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                        patience=8, restore_best_weights=True, verbose=0)]
    hist = trainer.fit(train_ds, validation_data=val_ds, epochs=budget,
                       class_weight=calculate_class_weights(data["y_train"]),
                       callbacks=cb, verbose=0)
    val_acc = float(max(hist.history["val_accuracy"]))
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    return val_acc, n_params


def search_mstc(data, rec, deadline):
    log("=== MSTC-IDS architecture search (cyber modality, 20-epoch budget) ===")
    grid = [
        {"lr": 5e-5, "dropout": 0.3, "d": 64},
        {"lr": 1e-4, "dropout": 0.3, "d": 64},
        {"lr": 2e-5, "dropout": 0.2, "d": 64},
        {"lr": 5e-5, "dropout": 0.1, "d": 64},
        {"lr": 1e-4, "dropout": 0.4, "d": 64},
        {"lr": 5e-5, "dropout": 0.3, "d": 48},
        {"lr": 1e-4, "dropout": 0.3, "d": 48},
        {"lr": 2e-5, "dropout": 0.4, "d": 64},
    ]
    results = []
    for cfg in grid:
        key = json.dumps(cfg)
        if trial_done(rec, "mstc", key):
            val_acc = rec["mstc"][key]["val_acc"]
            n_params = rec["mstc"][key]["n_params"]
        else:
            val_acc, n_params = score_mstc_config(cfg, data, "cyber")
            rec.setdefault("mstc", {})[key] = {"config": cfg, "val_acc": val_acc,
                                               "n_params": n_params}
            save_record(rec)
        results.append((val_acc, cfg, n_params))
        log(f"  {key} -> val_acc={val_acc:.4f} params={n_params:,}")
        if deadline is not None and time.time() > deadline:
            log("  max-minutes reached in MSTC search; stopping.")
            return None
    results.sort(key=lambda r: r[0], reverse=True)
    best = results[0][1]
    log(f"  best MSTC arch: {best} (val_acc={results[0][0]:.4f})")

    # contrastive-weight search on the both modality (holds lr/dropout/d fixed)
    log("=== MSTC-IDS contrastive-weight search (both modality) ===")
    cw_results = []
    for cw in [0.05, 0.1, 0.2]:
        cfg = {**best, "contrastive_weight": cw}
        key = json.dumps(cfg)
        if trial_done(rec, "mstc_cw", key):
            val_acc = rec["mstc_cw"][key]["val_acc"]
        else:
            val_acc, _ = score_mstc_config(cfg, data, "both")
            rec.setdefault("mstc_cw", {})[key] = {"config": cfg, "val_acc": val_acc}
            save_record(rec)
        cw_results.append((val_acc, cw))
        log(f"  cw={cw} -> val_acc={val_acc:.4f}")
        if deadline is not None and time.time() > deadline:
            log("  max-minutes reached in contrastive search; stopping.")
            return None
    cw_results.sort(key=lambda r: r[0], reverse=True)
    best["contrastive_weight"] = cw_results[0][1]
    log(f"  best MSTC config: {best}")
    return best


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated: svm,fnn,lstm,cnn,mstc")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--tag", default="", help="worker tag to isolate record files")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes > 0 else None

    global SEARCH_RECORD, BEST_CONFIGS
    SEARCH_RECORD, BEST_CONFIGS = _tagged_paths(args.tag)

    def want(*names):
        return only is None or any(n in only for n in names)

    rec = load_record()
    best = {}
    if os.path.exists(BEST_CONFIGS):
        with open(BEST_CONFIGS) as f:
            best = json.load(f)

    data, _, _, _ = build_pipeline(DATA_DIR, cache_dir=CACHE_DIR)

    if want("svm"):
        features = {}
        for m in ["cyber", "physical", "fused"]:
            features[m] = {
                "train": build_modality_features(data["Xc_train"], data["Xp_train"],
                                                 data["Xt_train"], data["Xf_train"], m),
                "val": build_modality_features(data["Xc_val"], data["Xp_val"],
                                               data["Xt_val"], data["Xf_val"], m),
            }
        cfg = search_svm(data, features, rec, deadline)
        if cfg:
            best["svm"] = cfg
            with open(BEST_CONFIGS, "w") as f:
                json.dump(best, f, indent=2)
        if deadline is not None and time.time() > deadline:
            log("stopping (deadline).")
            return

    for fam, famkey in [("FNN", "fnn"), ("LSTM", "lstm"), ("1D-CNN", "cnn")]:
        if not want(famkey):
            continue
        cfg = search_keras_baseline(fam, rec, deadline)
        if cfg:
            best[famkey] = cfg
            with open(BEST_CONFIGS, "w") as f:
                json.dump(best, f, indent=2)
        if deadline is not None and time.time() > deadline:
            log("stopping (deadline).")
            return

    if want("mstc"):
        cfg = search_mstc(data, rec, deadline)
        if cfg:
            best["mstc"] = cfg
            with open(BEST_CONFIGS, "w") as f:
                json.dump(best, f, indent=2)

    log("Search complete.")
    log("best_configs: " + json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
