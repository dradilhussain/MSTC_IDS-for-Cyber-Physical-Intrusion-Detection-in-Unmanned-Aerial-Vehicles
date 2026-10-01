"""
Builds the manuscript-ready `comparison/` folder, organized to match the
Results & Analysis narrative, directly from the saved predictions/metrics:

  comparison/
  1_proposed_model/                MSTC-IDS: cyber, physical, early + late fusion
  2_baseline_comparison/
      {cyber_only, physical_only, cyber_physical, cyber_physical_late}/
          summary_<modality>.csv      grouped table (SVM/FNN/LSTM/1D-CNN/MSTC-IDS)
          roc.png                     grouped macro-OvR ROC
          metrics_bar.png             grouped accuracy/precision/recall/F1
          per_class_f1.png            grouped per-class F1
          confusion_matrix.png        1x5 row of confusion matrices
  3_component_ablation/
      {cyber_only, physical_only, cyber_physical, cyber_physical_late}/
          ablation_table_<modality>.csv   filtered component x modality table
          roc.png                        all component variants, one modality
          impact.png                     F1-drop ranking vs. full model
          confusion_matrix.png           1xN row of confusion matrices

Both cyber-physical fusion strategies are reported:

  * early fusion - a single network trained jointly on the concatenated cyber
    and physical streams with gated cross-attention ("cyber_physical" folders);
  * late fusion  - the validation-selected decision-level combination of the
    trained branches ("cyber_physical_late" folders).

Every number and figure here is generated from the same saved predictions used by
the notebooks and figures/ folder, so they always stay in sync.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
COMP_DIR = os.path.join(ROOT, "comparison")

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
NUM_CLASSES = 5
FAMILY_ORDER = ["SVM", "FNN", "LSTM", "1D-CNN", "MSTC-IDS"]
EARLY_VARIANTS = ["full", "no_time2vec", "no_cross_attention", "no_contrastive",
                  "no_pyramid", "no_causal", "no_freshness"]
LATE_VARIANTS = ["full", "no_time2vec", "no_pyramid", "no_causal", "no_freshness"]
VARIANTS_BY_MODALITY = {
    "cyber": EARLY_VARIANTS, "physical": EARLY_VARIANTS,
    "both": EARLY_VARIANTS, "late": LATE_VARIANTS,
}
component_display = {
    "full": "Full model", "no_time2vec": "w/o Time2Vec",
    "no_cross_attention": "w/o Cross-Attention",
    "no_contrastive": "w/o Contrastive Loss",
    "no_pyramid": "w/o Multi-Scale Pyramid", "no_causal": "w/o Causal Masking",
    "no_freshness": "w/o Freshness Embedding",
}

# baseline groups: modality key -> comparison subdirectory / figure label
baseline_dir = {"cyber": "cyber_only", "physical": "physical_only",
                "fused": "cyber_physical", "latefusion": "cyber_physical_late"}
modality_label = {"cyber": "Cyber-only", "physical": "Physical-only",
                  "fused": "Cyber-Physical (early fusion)",
                  "latefusion": "Cyber-Physical (late fusion)"}
# component ablation: internal modality key -> dir / table label (compile_results)
comp_dir = {"cyber": "cyber_only", "physical": "physical_only",
            "both": "cyber_physical", "late": "cyber_physical_late"}
comp_table_label = {"cyber": "Cyber-only", "physical": "Physical-only",
                    "both": "Early fusion", "late": "Late fusion"}
comp_title = {"cyber": "Cyber-only", "physical": "Physical-only",
              "both": "Cyber-Physical (early fusion)",
              "late": "Cyber-Physical (late fusion)"}

sns.set_theme()
model_colors = {"SVM": "#1f77b4", "FNN": "#ff7f0e", "LSTM": "#2ca02c", "1D-CNN": "#9467bd", "MSTC-IDS": "#d62728"}


def ensure_dirs():
    os.makedirs(os.path.join(COMP_DIR, "1_proposed_model"), exist_ok=True)
    for group in ["2_baseline_comparison", "3_component_ablation"]:
        for m in set(baseline_dir.values()) | set(comp_dir.values()):
            os.makedirs(os.path.join(COMP_DIR, group, m), exist_ok=True)


def macro_ovr_roc(y_true, y_pred_proba, n_classes=NUM_CLASSES):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_grid = np.linspace(0, 1, 200)
    tprs = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        tprs.append(np.interp(fpr_grid, fpr_i, tpr_i))
    mean_tpr = np.mean(tprs, axis=0)
    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)


def cm_plot(y_true, y_pred, ax, title):
    cm_ = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)


def main():
    ensure_dirs()
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    baseline_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "baseline_probas.npz")))
    component_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "component_probas.npz")))
    with open(os.path.join(RESULTS_DIR, "baseline_comparison_merged.json")) as f:
        baseline_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "component_modality_ablation_full.json")) as f:
        component_metrics = json.load(f)

    mstc_source = {"cyber": "full__cyber", "physical": "full__physical",
                   "fused": "full__both", "latefusion": "full__late"}
    for m, key in mstc_source.items():
        baseline_probas[f"MSTC-IDS-{m}"] = component_probas[key]

    # ------------------------------------------------------------
    # 1. Proposed model
    # ------------------------------------------------------------
    out1 = os.path.join(COMP_DIR, "1_proposed_model")

    pd.read_csv(os.path.join(RESULTS_DIR, "modality_ablation_summary.csv"), index_col=0).to_csv(
        os.path.join(out1, "modality_ablation_summary.csv"))
    pd.read_csv(os.path.join(RESULTS_DIR, "modality_ablation_per_class_f1.csv"), index_col=0).to_csv(
        os.path.join(out1, "modality_ablation_per_class_f1.csv"))

    variants1 = [
        ("Cyber-only", "full__cyber"),
        ("Physical-only", "full__physical"),
        ("Cyber-Physical (early fusion)", "full__both"),
        ("Cyber-Physical (late fusion)", "full__late"),
    ]

    per_class_df = pd.DataFrame({
        label: component_metrics[key]["per_class_f1"] for label, key in variants1
    }).T[CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(11, 5))
    per_class_df.T.plot(kind="bar", ax=ax)
    ax.set_title("Proposed MSTC-IDS: Per-Class F1 by Modality")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Class")
    ax.legend(title="Variant")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out1, "per_class_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary_rows = []
    for label, key in variants1:
        r = component_metrics[key]
        summary_rows.append({"Variant": label, "Accuracy": r["accuracy"],
                             "Precision": r["precision_weighted"], "Recall": r["recall_weighted"],
                             "F1": r["f1_weighted"]})
    summary_df = pd.DataFrame(summary_rows).set_index("Variant")
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_df[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
    ax.set_title("Proposed MSTC-IDS: Overall Metrics by Modality")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out1, "overall_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))
    for ax, (label, key) in zip(axes, variants1):
        cm_plot(y_test, np.argmax(component_probas[key], axis=1), ax, label)
    plt.tight_layout()
    plt.savefig(os.path.join(out1, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    for (label, key), color in zip(variants1, colors):
        fpr_grid, mean_tpr, auc_val = macro_ovr_roc(y_test, component_probas[key])
        lw = 3 if "fusion" in label else 1.5
        ax.plot(fpr_grid, mean_tpr, label=f"{label} (AUC={auc_val:.3f})",
                color=color, linewidth=lw)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Full Model ROC: Cyber-only vs. Physical-only vs. Cyber-Physical")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out1, "roc_full_model.png"), dpi=150, bbox_inches="tight")
    plt.close()

    hist_dir = os.path.join(RESULTS_DIR, "histories")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in ["cyber", "physical"]:
        hp = os.path.join(hist_dir, f"full__{m}.json")
        if not os.path.exists(hp):
            continue
        with open(hp) as f:
            h = json.load(f)
        axes[0].plot(h["val_loss"], label=modality_label[m])
        axes[1].plot(h["val_accuracy"], label=modality_label[m])
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out1, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("1_proposed_model done")

    # ------------------------------------------------------------
    # 2. Baseline comparison
    # ------------------------------------------------------------
    for m in baseline_dir:
        sub = baseline_dir[m]
        out = os.path.join(COMP_DIR, "2_baseline_comparison", sub)

        pd.read_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_group_{m}.csv"), index_col=0).to_csv(
            os.path.join(out, f"summary_{m}.csv"))

        names = [f"{fam}-{m}" for fam in FAMILY_ORDER]

        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for fam in FAMILY_ORDER:
            fpr_grid, mean_tpr, auc_val = macro_ovr_roc(y_test, baseline_probas[f"{fam}-{m}"])
            is_proposed = fam == "MSTC-IDS"
            ax.plot(fpr_grid, mean_tpr, label=f"{fam} ({auc_val:.3f})",
                    color=model_colors[fam], linewidth=3 if is_proposed else 1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"Baseline Comparison — {modality_label[m]}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "roc.png"), dpi=150, bbox_inches="tight")
        plt.close()

        rows = []
        for name in names:
            r = baseline_metrics[name]
            rows.append({"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision_weighted"],
                         "Recall": r["recall_weighted"], "F1": r["f1_weighted"]})
        bdf = pd.DataFrame(rows).set_index("Model")
        fig, ax = plt.subplots(figsize=(8, 5))
        bdf[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
        ax.set_title(f"Metrics — {modality_label[m]}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "metrics_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

        pcf = pd.DataFrame({name: baseline_metrics[name]["per_class_f1"] for name in names}).T[CLASS_NAMES]
        fig, ax = plt.subplots(figsize=(9, 5))
        pcf.T.plot(kind="bar", ax=ax)
        ax.set_title(f"Per-Class F1 — {modality_label[m]}")
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Class")
        ax.legend(title="Model", fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "per_class_f1.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(4.8 * len(FAMILY_ORDER), 5.2))
        for ax, fam in zip(np.atleast_1d(axes), FAMILY_ORDER):
            cm_plot(y_test, np.argmax(baseline_probas[f"{fam}-{m}"], axis=1), ax,
                    f"{fam}\n{modality_label[m]}")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"2_baseline_comparison/{sub} done")

    # ------------------------------------------------------------
    # 3. Component ablation
    # ------------------------------------------------------------
    comp_table = pd.read_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_summary.csv"), index_col=0)
    for m in comp_dir:
        sub = comp_dir[m]
        out = os.path.join(COMP_DIR, "3_component_ablation", sub)
        variants = VARIANTS_BY_MODALITY[m]
        palette = cm.tab10.colors[:len(variants)]

        keep = comp_table["Modality"] == comp_table_label[m]
        comp_table[keep].to_csv(os.path.join(out, f"ablation_table_{m}.csv"))

        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for i, c in enumerate(variants):
            fpr_grid, mean_tpr, auc_val = macro_ovr_roc(y_test, component_probas[f"{c}__{m}"])
            is_full = c == "full"
            ax.plot(fpr_grid, mean_tpr, label=f"{component_display[c]} ({auc_val:.3f})",
                    color="black" if is_full else palette[i], linewidth=2.5 if is_full else 1)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"Component Ablation — {comp_title[m]}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "roc.png"), dpi=150, bbox_inches="tight")
        plt.close()

        full_f1 = component_metrics[f"full__{m}"]["f1_weighted"]
        rows = []
        for c in variants:
            if c == "full":
                continue
            delta = full_f1 - component_metrics[f"{c}__{m}"]["f1_weighted"]
            rows.append({"Component removed": component_display[c], "F1 drop": round(delta, 4)})
        idf = pd.DataFrame(rows).sort_values("F1 drop", ascending=False).set_index("Component removed")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in idf["F1 drop"]]
        idf["F1 drop"].plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Impact Ranking — {comp_title[m]}")
        ax.set_xlabel("F1 drop vs. full model (same modality)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "impact.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, len(variants), figsize=(4.3 * len(variants), 5.2))
        for ax, c in zip(np.atleast_1d(axes), variants):
            cm_plot(y_test, np.argmax(component_probas[f"{c}__{m}"], axis=1), ax,
                    f"{component_display[c]}\n{comp_title[m]}")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"3_component_ablation/{sub} done")

    print("comparison/ folder complete")


if __name__ == "__main__":
    main()
