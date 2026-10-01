"""Decision-level (late) fusion as an alternative to early (cross-attention) fusion.

Both fusion strategies are kept for the cyber-physical model:

  * early fusion  - a single network trained on concatenated cyber+physical
                    streams with gated cross-attention (modality "both").
  * late fusion   - the independently trained cyber-only and physical-only
                    branches are combined at the decision level,
                        p = w * p_physical + (1 - w) * p_cyber,
                    with w selected on the validation split (accuracy), the
                    same criterion used for checkpoint selection everywhere.

This script *adds* late-fusion results; it does not remove the early-fusion
ones. It evaluates the saved branch models on the validation split, selects w,
and writes:

  results/late_fusion/late_fusion_summary.json   full provenance + metrics
                                                 (proposed, baselines, ablation)
  results/probabilities/component_probas.npz     adds {variant}__late
  results/probabilities/baseline_probas.npz      adds {family}-latefusion
"""

import os
import json
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import joblib
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_pipeline, CLASS_NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
SEGMENT_DIR = os.path.join(ROOT, "segment_data")
CACHE_DIR = os.path.join(ROOT, "data")
CKPT_COMP = os.path.join(ROOT, "models", "component_modality_ablation")
CKPT_BASE = os.path.join(ROOT, "models", "baselines")
RESULTS = os.path.join(ROOT, "results")
PROBAS = os.path.join(RESULTS, "probabilities")
OUT = os.path.join(RESULTS, "late_fusion")

COMP_JSON = os.path.join(RESULTS, "component_modality_ablation_full.json")
COMP_NPZ = os.path.join(PROBAS, "component_probas.npz")
BASE_NPZ = os.path.join(PROBAS, "baseline_probas.npz")

NUM_CLASSES = 5
FAMILIES = ["SVM", "FNN", "LSTM", "1D-CNN"]
# branch-encoder components; cross-attention is early-fusion-only and the
# contrastive alignment loss was only active for the early-fusion model.
LATE_VARIANTS = ["full", "no_time2vec", "no_pyramid", "no_causal", "no_freshness"]
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def build_modality_features(Xc, Xp, Xt, Xf, modality):
    """Byte-identical feature construction to train_all.py."""
    if modality == "cyber":
        seq = np.concatenate([Xc, Xt, Xf], axis=-1)
    elif modality == "physical":
        seq = Xp
    else:
        raise ValueError(modality)
    return seq, seq.reshape(seq.shape[0], -1)


def mstc_val_proba(variant, modality, data):
    ckpt = os.path.join(CKPT_COMP, f"mstc_ids_{variant}__{modality}_best.keras")
    model = keras.models.load_model(ckpt, compile=False)
    x = (data["Xc_val"], data["Xp_val"], data["Xt_val"], data["Xf_val"])
    return model(x, training=False).numpy()


def baseline_val_proba(family, modality, data, features):
    if family == "SVM":
        svm = joblib.load(os.path.join(CKPT_BASE, f"baseline_SVM-{modality}.pkl"))
        raw = svm.predict_proba(features[modality]["val"][1])
        proba = np.zeros((raw.shape[0], NUM_CLASSES))
        for i, c in enumerate(svm.classes_):
            proba[:, c] = raw[:, i]
        return proba
    ckpt = os.path.join(CKPT_BASE, f"baseline_{family}-{modality}_best.keras")
    model = keras.models.load_model(ckpt, compile=False)
    x = features[modality]["val"][1 if family == "FNN" else 0]
    return model.predict(x, verbose=0)


def metrics(y, proba):
    yp = proba.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, yp)),
        "precision": float(precision_score(y, yp, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, yp, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
    }
    pcf = f1_score(y, yp, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    out["per_class_f1"] = {CLASS_NAMES[i]: float(pcf[i]) for i in range(NUM_CLASSES)}
    out["confusion_matrix"] = confusion_matrix(y, yp, labels=list(range(NUM_CLASSES))).tolist()
    return out


def late_fuse(p_c_val, p_p_val, p_c_test, p_p_test, y_val, y_test):
    sweep = []
    for w in W_GRID:
        v = w * p_p_val + (1 - w) * p_c_val
        sweep.append((float(w), float(accuracy_score(y_val, v.argmax(1)))))
    best_w, best_va = max(sweep, key=lambda r: (r[1], -abs(r[0] - 0.5)))
    test_proba = best_w * p_p_test + (1 - best_w) * p_c_test
    return best_w, best_va, test_proba, float(metrics(y_test, test_proba)["f1"])


def main():
    os.makedirs(OUT, exist_ok=True)
    data, _, _, _ = build_pipeline(SEGMENT_DIR, cache_dir=CACHE_DIR)
    y_val, y_test = data["y_val"], data["y_test"]

    features = {
        m: {"val": build_modality_features(data["Xc_val"], data["Xp_val"],
                                           data["Xt_val"], data["Xf_val"], m)}
        for m in ("cyber", "physical")
    }

    comp_metrics = json.load(open(COMP_JSON))
    comp_test = dict(np.load(COMP_NPZ))
    base_test = dict(np.load(BASE_NPZ))

    # idempotent: drop any late-fusion keys from an earlier run before re-adding
    for store in (comp_metrics, comp_test):
        for k in [k for k in store if k.startswith("late") and k.endswith("__both")]:
            del store[k]

    print("evaluating MSTC-IDS branch models on validation split ...")
    val_c = {v: mstc_val_proba(v, "cyber", data) for v in LATE_VARIANTS}
    val_p = {v: mstc_val_proba(v, "physical", data) for v in LATE_VARIANTS}

    print("evaluating baseline models on validation split ...")
    base_val_c = {f: baseline_val_proba(f, "cyber", data, features) for f in FAMILIES}
    base_val_p = {f: baseline_val_proba(f, "physical", data, features) for f in FAMILIES}

    summary = {"method": "validation-selected decision-level late fusion "
                         "(p = w * physical + (1 - w) * cyber)",
               "strategies": ["early (cross-attention)", "late (decision-level)"],
               "proposed": {}, "baselines": {}, "ablation": {}}

    # ---- proposed late fusion of the two full branches ----
    for v in LATE_VARIANTS:
        w, va, proba, tf1 = late_fuse(val_c[v], val_p[v],
                                      comp_test[f"{v}__cyber"], comp_test[f"{v}__physical"],
                                      y_val, y_test)
        key = f"{v}__late"
        comp_test[key] = proba
        m = metrics(y_test, proba)
        comp_metrics[key] = {
            "name": key, "component": v, "modality": "late", "fusion": "late",
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision"],
            "recall_weighted": m["recall"],
            "f1_weighted": m["f1"],
            "macro_auc": m["macro_auc"],
            "per_class_f1": m["per_class_f1"],
            "confusion_matrix": m["confusion_matrix"],
            "trainable_params": (comp_metrics[f"{v}__cyber"]["trainable_params"]
                                 + comp_metrics[f"{v}__physical"]["trainable_params"]),
            "epochs_trained": None,
            "checkpoint": f"late fusion of mstc_ids_{v}__cyber_best.keras and "
                          f"mstc_ids_{v}__physical_best.keras",
        }
        block = {"w_physical": w, "val_accuracy": va, "test": m}
        if v == "full":
            summary["proposed"] = block
        summary["ablation"][v] = block
        print(f"late fusion [{v}]: w={w:.2f} val_acc={va:.4f} "
              f"test_acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    # ---- baselines, late-fused per family ----
    for f in FAMILIES:
        w, va, proba, tf1 = late_fuse(base_val_c[f], base_val_p[f],
                                      base_test[f"{f}-cyber"], base_test[f"{f}-physical"],
                                      y_val, y_test)
        base_test[f"{f}-latefusion"] = proba
        summary["baselines"][f] = {"w_physical": w, "val_accuracy": va,
                                   "test": metrics(y_test, proba)}
        print(f"baseline late fusion [{f}]: w={w:.2f} val_acc={va:.4f} "
              f"test_acc={summary['baselines'][f]['test']['accuracy']:.4f}")

    with open(COMP_JSON, "w") as fh:
        json.dump(comp_metrics, fh, indent=2)
    np.savez(COMP_NPZ, **comp_test)
    np.savez(BASE_NPZ, **base_test)

    with open(os.path.join(OUT, "late_fusion_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(OUT, "late_fusion_weight_sweep.csv"), "w") as fh:
        fh.write("configuration,w_physical,val_accuracy\n")
        for name, block in [("proposed", summary["proposed"])] + \
                [(f"baseline_{f}", summary["baselines"][f]) for f in FAMILIES] + \
                [(f"ablation_{v}", summary["ablation"][v]) for v in LATE_VARIANTS]:
            fh.write(f"{name},{block['w_physical']:.2f},{block['val_accuracy']:.4f}\n")

    print("\nlate fusion complete ->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
