"""
Parameter-based performance analysis.

Consolidates every model (21 MSTC-IDS component x modality + 12 baselines) into a
single table pairing model size (# trainable parameters) with test-set
performance, per modality, and adds efficiency metrics:

  * Params                 # trainable parameters (SVM: number of training samples)
  * Accuracy / F1 / MacroAUC
  * F1 per 1K params       efficiency: weighted F1 per 1,000 parameters
  * Acc per 1K params      efficiency: accuracy per 1,000 parameters
  * Params log10           for log-scale plotting

Outputs:
  results/parameter_analysis/performance_vs_parameters.csv
  results/parameter_analysis/performance_vs_parameters_group_{modality}.csv
  results/parameter_analysis/efficiency_ranking.csv
  figures/parameter_analysis/f1_vs_parameters.png
  figures/parameter_analysis/accuracy_vs_parameters.png
  figures/parameter_analysis/efficiency_bar.png

Run after compile_results.py.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures", "parameter_analysis")
OUT_DIR = os.path.join(RESULTS_DIR, "parameter_analysis")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]

# display modality per family: baselines use fused/latefusion, MSTC uses both/late
MODALITY_DISPLAY = {"cyber": "Cyber-only", "physical": "Physical-only",
                    "both": "Cyber-Physical (early fusion)",
                    "fused": "Cyber-Physical (early fusion)",
                    "late": "Cyber-Physical (late fusion)",
                    "latefusion": "Cyber-Physical (late fusion)"}
GROUP_FILES = [("Cyber-only", "cyberonly"),
               ("Physical-only", "physicalonly"),
               ("Cyber-Physical (early fusion)", "cyberphysical"),
               ("Cyber-Physical (late fusion)", "cyberphysicallate")]

FAMILY_COLORS = {
    "SVM": "#1f77b4", "FNN": "#ff7f0e", "LSTM": "#2ca02c",
    "1D-CNN": "#d62728", "MSTC-IDS": "#9467bd",
    "MSTC-IDS (ablated)": "#8c8c8c",
}


def family_of(model_name):
    for f in ["MSTC-IDS", "1D-CNN", "LSTM", "FNN", "SVM"]:
        if model_name.startswith(f):
            return f
    # component ablation names like "full__cyber"
    return "ablation"


def modality_of(model_name):
    for m in ["latefusion", "late", "fused", "both", "physical", "cyber"]:
        if model_name.endswith(m):
            return m
    return "cyber"


def main():
    component_metrics = json.load(open(os.path.join(RESULTS_DIR, "component_modality_ablation_full.json")))
    baseline_metrics = json.load(open(os.path.join(RESULTS_DIR, "baseline_comparison_merged.json")))

    rows = []
    # baselines (family models only; MSTC-IDS rows come from the component table)
    for name, r in baseline_metrics.items():
        if name.startswith("MSTC-IDS"):
            continue
        fam = family_of(name)
        rows.append({
            "Model": name,
            "Family": fam,
            "Modality": MODALITY_DISPLAY[modality_of(name)],
            "Params": r["trainable_params"],
            "Epochs": r["epochs_trained"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision_weighted"],
            "Recall": r["recall_weighted"],
            "F1": r["f1_weighted"],
            "MacroAUC": r["macro_auc"],
            **{f"F1_{c}": r["per_class_f1"][c] for c in CLASS_NAMES},
        })
    # MSTC-IDS component x modality rows; the full__* rows double as the proposed model
    for key, r in component_metrics.items():
        fam = "MSTC-IDS" if key.startswith("full__") else "MSTC-IDS (ablated)"
        rows.append({
            "Model": key,
            "Family": fam,
            "Modality": MODALITY_DISPLAY[modality_of(key)],
            "Params": r["trainable_params"],
            "Epochs": r["epochs_trained"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision_weighted"],
            "Recall": r["recall_weighted"],
            "F1": r["f1_weighted"],
            "MacroAUC": r["macro_auc"],
            **{f"F1_{c}": r["per_class_f1"][c] for c in CLASS_NAMES},
        })

    df = pd.DataFrame(rows)
    df["F1_per_1K_params"] = df["F1"] / (df["Params"] / 1000.0)
    df["Acc_per_1K_params"] = df["Accuracy"] / (df["Params"] / 1000.0)
    df["Params_log10"] = np.log10(df["Params"])
    df = df.round(4)

    df.to_csv(os.path.join(OUT_DIR, "performance_vs_parameters.csv"), index=False)

    for mod, tag in GROUP_FILES:
        sub = df[df["Modality"] == mod].sort_values("F1", ascending=False)
        sub.to_csv(os.path.join(OUT_DIR, f"performance_vs_parameters_group_{tag}.csv"), index=False)

    # efficiency ranking (F1 per 1K params), best per modality
    ranking = df.sort_values("F1_per_1K_params", ascending=False)
    ranking.to_csv(os.path.join(OUT_DIR, "efficiency_ranking.csv"), index=False)
    print("parameter_analysis CSVs written")

    # ---------------- figures ----------------
    order = [mod for mod, _ in GROUP_FILES]
    for metric, ylab, fname in [("F1", "Weighted F1", "f1_vs_parameters.png"),
                                ("Accuracy", "Accuracy", "accuracy_vs_parameters.png")]:
        fig, axes = plt.subplots(1, 4, figsize=(25, 6))
        for ax, mod in zip(axes, order):
            sub = df[df["Modality"] == mod]
            for fam, color in FAMILY_COLORS.items():
                fam_df = sub[sub["Family"] == fam]
                if fam_df.empty:
                    continue
                ax.scatter(fam_df["Params"], fam_df[metric], s=55, alpha=0.85,
                           color=color, edgecolors="black", linewidth=0.5,
                           label=fam, zorder=3)
            ax.set_xscale("log")
            ax.set_xlabel("Trainable parameters (log scale)")
            ax.set_ylabel(ylab)
            ax.set_title(f"{mod} ({len(sub)} models)")
            ax.grid(True, which="both", alpha=0.3)
            if mod == order[0]:
                ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
    print("scatter figures written")

    # efficiency bar chart: F1 per 1K params, best 2 per family per modality
    top = df.sort_values("F1", ascending=False).groupby(["Modality", "Family"]).head(2)
    top = top.sort_values(["Modality", "F1_per_1K_params"], ascending=[True, False])
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    for ax, mod in zip(axes, order):
        sub = top[top["Modality"] == mod]
        y = sub["F1_per_1K_params"].values
        x = np.arange(len(sub))
        colors = [FAMILY_COLORS[f] for f in sub["Family"]]
        ax.bar(x, y, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{mod} — weighted F1 per 1K params")
        ax.set_ylabel("F1 per 1K params")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "efficiency_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("efficiency figure written")


if __name__ == "__main__":
    main()
