"""
Statistical comparison for the MSTC-IDS paper:

  - McNemar significance tests: proposed model vs each baseline, per modality
    group (cyber-only, physical-only, cyber-physical/fused).
  - Fairness: per-class F1 spread (max - min) for the proposed model vs baselines.
  - Latency: end-to-end per-window inference time of the proposed fused model.

Output:
  results/statistical_comparison.csv
  results/statistical_comparison_summary.txt
"""

import os
import json
import time
import numpy as np
from sklearn.metrics import f1_score

import pipeline
from mstc_ids_model import MSTCIDS

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
GROUPS = {
    "cyber": ("cyber", "cyber"),
    "physical": ("physical", "physical"),
    "fused_early": ("both", "fused"),
    "fused_late": ("late", "latefusion"),
}
# proposed key per modality group: full__{internal}; baselines use {fam}-{m}
BASELINE_FAMS = ["SVM", "FNN", "LSTM", "1D-CNN"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def mcnemar(y_true, p1, p2):
    a = np.sum((p1 == y_true) & (p2 == y_true))
    b = np.sum((p1 == y_true) & (p2 != y_true))
    c = np.sum((p1 != y_true) & (p2 == y_true))
    d = np.sum((p1 != y_true) & (p2 != y_true))
    n = b + c
    if n == 0:
        return float("nan"), 1.0
    stat = (abs(b - c) - 1) ** 2 / max(n, 1)
    from scipy.stats import chi2
    p_val = 1.0 - chi2.cdf(stat, 1)
    return stat, float(p_val)


def main():
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    component_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "component_probas.npz")))
    baseline_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "baseline_probas.npz")))
    with open(os.path.join(RESULTS_DIR, "component_modality_ablation_full.json")) as f:
        component_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "baseline_comparison_merged.json")) as f:
        baseline_metrics = json.load(f)

    rows = []
    summary = []
    for group, (internal, m) in GROUPS.items():
        prop_key = f"full__{internal}"
        y_prop = np.argmax(component_probas[prop_key], axis=1)
        for fam in BASELINE_FAMS:
            bname = f"{fam}-{m}"
            y_b = np.argmax(baseline_probas[bname], axis=1)
            stat, p = mcnemar(y_test, y_prop, y_b)
            acc_prop = component_metrics[prop_key]["accuracy"]
            acc_base = baseline_metrics[bname]["accuracy"]
            rows.append({
                "group": group, "proposed": "MSTC-IDS", "baseline": fam,
                "proposed_accuracy": round(acc_prop, 4), "baseline_accuracy": round(acc_base, 4),
                "proposed_wins": bool(acc_prop > acc_base),
                "mcnemar_stat": round(stat, 3), "p_value": f"{p:.4g}",
                "significant_05": bool(p < 0.05),
            })
        # fairness: per-class F1 spread
        def spread(metrics_dict, key):
            f1s = list(metrics_dict[key]["per_class_f1"].values())
            return max(f1s) - min(f1s)

        spread_prop = spread(component_metrics, prop_key)
        for fam in BASELINE_FAMS:
            bname = f"{fam}-{m}"
            rows.append({
                "group": group, "proposed": "MSTC-IDS", "baseline": fam,
                "fairness_delta_proposed": round(spread_prop, 4),
                "fairness_delta_baseline": round(spread(baseline_metrics, bname), 4),
            })
        summary.append(f"[{group}] proposed fairness delta (per-class F1 spread): {spread_prop:.4f}")

    # latency for both fusion strategies
    Xc = np.load(os.path.join(DATA_DIR, "Xc_test.npy"))[:64]
    Xp = np.load(os.path.join(DATA_DIR, "Xp_test.npy"))[:64]
    Xt = np.load(os.path.join(DATA_DIR, "Xt_test.npy"))[:64]
    Xf = np.load(os.path.join(DATA_DIR, "Xf_test.npy"))[:64]
    import tf_keras as keras
    ckpt_dir = os.path.join(ROOT, "models", "component_modality_ablation")

    def load_ckpt(name):
        return keras.models.load_model(os.path.join(ckpt_dir, name),
                                       compile=False, custom_objects={"MSTCIDS": MSTCIDS})

    warm = (np.zeros((1, 20, Xc.shape[-1])), np.zeros((1, 20, Xp.shape[-1])),
            np.zeros((1, 20, 1)), np.zeros((1, 20, 1)))
    batch = (Xc, Xp, Xt, Xf)

    def latency_ms(models):
        for m in models:
            m(warm, training=False)  # warmup
        t0 = time.perf_counter()
        for _ in range(100):
            for m in models:
                m(batch, training=False)
        return (time.perf_counter() - t0) / 100 / Xc.shape[0] * 1000

    early_models = [load_ckpt("mstc_ids_full__both_best.keras")]
    late_models = [load_ckpt("mstc_ids_full__cyber_best.keras"),
                   load_ckpt("mstc_ids_full__physical_best.keras")]
    early_ms = latency_ms(early_models)
    late_ms = latency_ms(late_models)
    summary.append(f"proposed early-fusion inference (single cross-attention model): "
                   f"{early_ms:.3f} ms / window (batch {Xc.shape[0]})")
    summary.append(f"proposed late-fusion inference (cyber+physical branches): "
                   f"{late_ms:.3f} ms / window (batch {Xc.shape[0]})")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "statistical_comparison.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "statistical_comparison_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print("statistical_comparison.csv written")


if __name__ == "__main__":
    main()
