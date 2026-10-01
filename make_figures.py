"""
MSTC-IDS figure generator.

Rebuilds every figure from the notebooks (and the manuscript-style versions)
from the saved model predictions, confusion matrices, and training histories
produced by train_all.py. No model retraining happens here.

Output:
  figures/proposed_model/         (MSTC-IDS full model, all 3 modalities)
  figures/modality_ablation/
  figures/baseline_comparison/
  figures/component_modality_ablation/
  figures/manuscript_v6_final/    (larger-font, manuscript-ready figures)
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
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
NUM_CLASSES = 5
MODALITIES = ["cyber", "physical", "both", "late"]
BASELINE_MODALITIES = ["cyber", "physical", "fused", "latefusion"]
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
modality_display = {"cyber": "Cyber-only", "physical": "Physical-only",
                    "both": "Early fusion", "late": "Late fusion"}
baseline_modality_display = {"cyber": "Cyber-only", "physical": "Physical-only",
                             "fused": "Cyber-Physical (early fusion)",
                             "latefusion": "Cyber-Physical (late fusion)"}

sns.set_theme()


def ensure_dirs():
    for d in ["proposed_model", "modality_ablation", "baseline_comparison",
              "component_modality_ablation", "manuscript_v6_final"]:
        os.makedirs(os.path.join(FIG_DIR, d), exist_ok=True)


def load_data():
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    baseline_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "baseline_probas.npz")))
    component_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "component_probas.npz")))
    with open(os.path.join(RESULTS_DIR, "baseline_comparison_merged.json")) as f:
        baseline_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "component_modality_ablation_full.json")) as f:
        component_metrics = json.load(f)
    return y_test, baseline_probas, component_probas, baseline_metrics, component_metrics


def load_history(key):
    p = os.path.join(RESULTS_DIR, "histories", f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def macro_ovr_roc(y_true, y_pred_proba, n_classes=NUM_CLASSES):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_grid = np.linspace(0, 1, 200)
    tprs = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        tprs.append(np.interp(fpr_grid, fpr_i, tpr_i))
    mean_tpr = np.mean(tprs, axis=0)
    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)


def get_pred(proba):
    return np.argmax(proba, axis=1)


# ============================================================
# 1. Proposed model (complete-pipeline style figures)
# ============================================================

def figures_proposed_model(y_test, component_probas):
    out = os.path.join(FIG_DIR, "proposed_model")
    # deployed fused model = validation-selected decision-level late fusion
    proba = component_probas["full__both"]
    y_pred = get_pred(proba)
    acc = accuracy_score(y_test, y_pred)

    # class distribution across splits
    dist = pd.DataFrame({
        "train": np.bincount(np.load(os.path.join(DATA_DIR, "y_train.npy")), minlength=NUM_CLASSES),
        "val": np.bincount(np.load(os.path.join(DATA_DIR, "y_val.npy")), minlength=NUM_CLASSES),
        "test": np.bincount(y_test, minlength=NUM_CLASSES),
    }, index=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 5))
    dist.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution across splits")
    ax.set_ylabel("Number of windows")
    ax.set_xlabel("Class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "class_distribution_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    hist = load_history("full__both")
    if hist:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(hist["loss"], label="train loss")
        axes[0].plot(hist["val_loss"], label="val loss")
        axes[0].set_title("Total loss (early-fusion model)")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(hist["accuracy"], label="train accuracy")
        axes[1].plot(hist["val_accuracy"], label="val accuracy")
        axes[1].set_title("Accuracy (early-fusion model)")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["classification_loss"], label="train classification loss")
        ax.plot(hist["val_classification_loss"], label="val classification loss")
        ax.plot(hist["contrastive_loss"], label="train contrastive loss")
        ax.plot(hist["val_contrastive_loss"], label="val contrastive loss")
        ax.set_title("Loss components")
        ax.set_xlabel("epoch")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "training_curves_2.png"), dpi=150, bbox_inches="tight")
        plt.close()

    cm_ = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"MSTC-IDS Confusion Matrix - early fusion (Test Set, acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    proba_late = component_probas["full__late"]
    acc_late = accuracy_score(y_test, get_pred(proba_late))
    cm_late = confusion_matrix(y_test, get_pred(proba_late))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_late, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"MSTC-IDS Confusion Matrix - late fusion (Test Set, acc={acc_late:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix_late_fusion.png"), dpi=150, bbox_inches="tight")
    plt.close()

    y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(NUM_CLASSES):
        ax.plot(fpr[i], tpr[i], label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves (One-vs-Rest) - MSTC-IDS")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "roc_auc_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # MC-dropout-style per-class uncertainty/confidence (deterministic proxy: softmax std over classes)
    unc = proba.std(axis=1)
    conf = proba.max(axis=1)
    per_class_unc = [unc[y_test == i].mean() for i in range(NUM_CLASSES)]
    per_class_conf = [conf[y_test == i].mean() for i in range(NUM_CLASSES)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(CLASS_NAMES, per_class_unc, color="steelblue")
    axes[0].set_title("Average Predictive Uncertainty per Class")
    axes[0].set_ylabel("Mean predictive std")
    axes[1].bar(CLASS_NAMES, per_class_conf, color="darkorange")
    axes[1].set_title("Average Prediction Confidence per Class")
    axes[1].set_ylabel("Mean max-softmax confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "mc_dropout_uncertainty_estimation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("proposed_model figures done")


# ============================================================
# 2. Modality ablation
# ============================================================

def figures_modality_ablation(y_test, component_probas, component_metrics):
    out = os.path.join(FIG_DIR, "modality_ablation")
    display_names = modality_display

    per_class_df = pd.DataFrame({
        display_names[m]: component_metrics[f"full__{m}"]["per_class_f1"] for m in MODALITIES
    }).T[CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 5))
    per_class_df.T.plot(kind="bar", ax=ax)
    ax.set_title("Per-Class F1 Score by Modality")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Class")
    ax.legend(title="Variant")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_class_f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary_rows = []
    for m in MODALITIES:
        r = component_metrics[f"full__{m}"]
        summary_rows.append({
            "Variant": display_names[m], "Accuracy": r["accuracy"], "Precision": r["precision_weighted"],
            "Recall": r["recall_weighted"], "F1": r["f1_weighted"],
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Variant")
    fig, ax = plt.subplots(figsize=(9, 5))
    summary_df[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
    ax.set_title("Overall Metrics by Modality")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "overall_metrics_comparison_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, m in zip(axes, MODALITIES):
        y_pred = get_pred(component_probas[f"full__{m}"])
        cm_ = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
        sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(display_names[m])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrices_side_by_side.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in ["cyber", "physical"]:
        hist = load_history(f"full__{m}")
        if hist is None:
            continue
        axes[0].plot(hist["val_loss"], label=display_names[m])
        axes[1].plot(hist["val_accuracy"], label=display_names[m])
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "training_curves_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("modality_ablation figures done")


# ============================================================
# 3. Baseline comparison
# ============================================================

def model_name(family, modality):
    return f"{family}-{modality}"


def figures_baseline_comparison(y_test, baseline_probas, baseline_metrics):
    out = os.path.join(FIG_DIR, "baseline_comparison")
    model_colors = {"SVM": "#1f77b4", "FNN": "#ff7f0e", "LSTM": "#2ca02c", "1D-CNN": "#9467bd", "MSTC-IDS": "#d62728"}

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    for ax, modality in zip(axes, BASELINE_MODALITIES):
        for fam in FAMILY_ORDER:
            name = model_name(fam, modality)
            proba = baseline_probas[name]
            fpr_grid, mean_tpr, roc_auc_val = macro_ovr_roc(y_test, proba)
            is_proposed = fam == "MSTC-IDS"
            ax.plot(fpr_grid, mean_tpr, label=f"{fam} ({roc_auc_val:.3f})",
                    color=model_colors[fam], linewidth=3 if is_proposed else 1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(baseline_modality_display[modality])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Combined ROC (macro-average OvR) - Grouped by Modality", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "grouped_comparisons_all_modalities_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary_rows = []
    for m in BASELINE_MODALITIES:
        for fam in FAMILY_ORDER:
            name = model_name(fam, m)
            r = baseline_metrics[name]
            summary_rows.append({"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision_weighted"],
                                 "Recall": r["recall_weighted"], "F1": r["f1_weighted"]})
    summary_df = pd.DataFrame(summary_rows).set_index("Model")

    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
    for ax, modality in zip(axes, BASELINE_MODALITIES):
        names = [model_name(fam, modality) for fam in FAMILY_ORDER]
        summary_df.loc[names][["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax, legend=(modality == BASELINE_MODALITIES[0]))
        ax.set_title(baseline_modality_display[modality])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "grouped_comparisons_all_modalities_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    per_class_df = pd.DataFrame({name: baseline_metrics[name]["per_class_f1"]
                                 for name in [model_name(fam, m) for m in BASELINE_MODALITIES for fam in FAMILY_ORDER]}).T[CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 13))
    sns.heatmap(per_class_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "F1 score"})
    ax.set_title("Per-Class F1 - All 20 Models")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_class_f1_heatmap_all_models.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    for ax, modality in zip(axes, BASELINE_MODALITIES):
        names = [model_name(fam, modality) for fam in FAMILY_ORDER]
        per_class_df.loc[names].T.plot(kind="bar", ax=ax, legend=(modality == BASELINE_MODALITIES[0]))
        ax.set_title(baseline_modality_display[modality])
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Class")
        plt.setp(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_class_f1_all_models_grouped_bar_charts.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(22, 17))
    for row, modality in enumerate(BASELINE_MODALITIES):
        for col, fam in enumerate(FAMILY_ORDER):
            ax = axes[row, col]
            name = model_name(fam, modality)
            y_pred = get_pred(baseline_probas[name])
            cm_ = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
            sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
            ax.set_title(f"{fam}\n{baseline_modality_display[modality]}", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrices_4_5_grid_modality_model_family.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("baseline_comparison figures done")


# ============================================================
# 4. Component x modality ablation
# ============================================================

def figures_component_ablation(y_test, component_probas, component_metrics):
    out = os.path.join(FIG_DIR, "component_modality_ablation")

    pivot_acc = pd.DataFrame({
        m: pd.Series({component_display[c]: component_metrics[f"{c}__{m}"]["accuracy"]
                      for c in VARIANTS_BY_MODALITY[m]})
        for m in MODALITIES
    }).reindex([component_display[c] for c in EARLY_VARIANTS])
    pivot_acc.columns = [modality_display[m] for m in MODALITIES]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot_acc.round(4), annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_title("Accuracy - Component x Modality")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pivoted_view_component_modality.png"), dpi=150, bbox_inches="tight")
    plt.close()

    def display_name(key):
        c, m = key.split("__")
        return f"{component_display[c]} ({modality_display[m]})"

    ALL_KEYS = [f"{c}__{m}" for m in MODALITIES for c in VARIANTS_BY_MODALITY[m]]
    per_class_df = pd.DataFrame({display_name(key): component_metrics[key]["per_class_f1"] for key in ALL_KEYS}).T[CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(per_class_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "F1 score"})
    ax.set_title(f"Per-Class F1 - All {len(ALL_KEYS)} Combinations")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_class_f1_heatmap_all_combinations.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(25, 6), sharex=True)
    for ax, modality in zip(axes, MODALITIES):
        full_f1 = component_metrics[f"full__{modality}"]["f1_weighted"]
        rows = []
        for c in VARIANTS_BY_MODALITY[modality]:
            if c == "full":
                continue
            delta = full_f1 - component_metrics[f"{c}__{modality}"]["f1_weighted"]
            rows.append({"Component removed": component_display[c], "F1 drop": round(delta, 4)})
        df_ = pd.DataFrame(rows).sort_values("F1 drop", ascending=False).set_index("Component removed")
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in df_["F1 drop"]]
        df_["F1 drop"].plot(kind="barh", ax=ax, color=colors)
        ax.set_title(modality_display[modality])
        ax.set_xlabel("F1 drop vs. full model (same modality)")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "impact_ranking_within_each_modality.png"), dpi=150, bbox_inches="tight")
    plt.close()

    palette = cm.tab10.colors[:len(EARLY_VARIANTS)]
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    for ax, modality in zip(axes, MODALITIES):
        for i, c in enumerate(VARIANTS_BY_MODALITY[modality]):
            key = f"{c}__{modality}"
            fpr_grid, mean_tpr, roc_auc_val = macro_ovr_roc(y_test, component_probas[key])
            is_full = c == "full"
            ax.plot(fpr_grid, mean_tpr, label=f"{component_display[c]} ({roc_auc_val:.3f})",
                    color="black" if is_full else palette[i], linewidth=2.5 if is_full else 1)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(modality_display[modality])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "combined_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"cyber": "#1f77b4", "physical": "#ff7f0e", "both": "#d62728", "late": "#9467bd"}
    for modality in MODALITIES:
        key = f"full__{modality}"
        fpr_grid, mean_tpr, roc_auc_val = macro_ovr_roc(y_test, component_probas[key])
        lw = 3 if modality in ("both", "late") else 1.5
        ax.plot(fpr_grid, mean_tpr, label=f"{modality_display[modality]} (AUC={roc_auc_val:.3f})",
                color=colors[modality], linewidth=lw)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Combined ROC - Full Model: Cyber-only vs. Physical-only vs. Cyber-Physical")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "combined_roc_curves_2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(4, 7, figsize=(28, 16))
    for row, modality in enumerate(MODALITIES):
        for ax in axes[row]:
            ax.axis("off")
        for col, c in enumerate(VARIANTS_BY_MODALITY[modality]):
            ax = axes[row, col]
            ax.axis("on")
            key = f"{c}__{modality}"
            y_pred = get_pred(component_probas[key])
            cm_ = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
            sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
            ax.set_title(f"{component_display[c]}\n{modality_display[modality]}", fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrices_4_7_grid_modality_component.png"), dpi=150, bbox_inches="tight")
    plt.close()

    curve_modalities = ["cyber", "physical", "both"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for col, modality in enumerate(curve_modalities):
        for c in VARIANTS_BY_MODALITY[modality]:
            hist = load_history(f"{c}__{modality}")
            if hist is None:
                continue
            axes[0, col].plot(hist["val_loss"], label=component_display[c])
            axes[1, col].plot(hist["val_accuracy"], label=component_display[c])
        axes[0, col].set_title(f"Val Loss - {modality_display[modality]}")
        axes[1, col].set_title(f"Val Accuracy - {modality_display[modality]}")
        axes[0, col].set_xlabel("epoch")
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "training_curves_grouped_by_modality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("component_modality_ablation figures done")


# ============================================================
# 5. Manuscript-style figures
# ============================================================

def figures_manuscript(y_test, baseline_probas, component_probas):
    out = os.path.join(FIG_DIR, "manuscript_v6_final")
    model_colors = {"SVM": "#1f77b4", "FNN": "#ff7f0e", "LSTM": "#2ca02c", "1D-CNN": "#9467bd", "MSTC-IDS": "#d62728"}
    baseline_families = ["SVM", "FNN", "LSTM", "1D-CNN"]
    modality_labels = {"cyber": "Cyber-only", "physical": "Physical-only",
                       "fused": "Cyber-Physical (early fusion)",
                       "latefusion": "Cyber-Physical (late fusion)"}
    # internal (component-probas / component-metrics) modality key + figure tag
    comp_modalities = [("cyber", "cyber"), ("physical", "physical"),
                       ("both", "early_fusion"), ("late", "late_fusion")]

    # 1. Proposed model CM 1x4 (incl. both fusion strategies)
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
    for ax, m in zip(axes, BASELINE_MODALITIES):
        proba = baseline_probas[f"MSTC-IDS-{m}"]
        cm_ = confusion_matrix(y_test, get_pred(proba), labels=list(range(NUM_CLASSES)))
        sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, annot_kws={"size": 13}, square=True)
        ax.set_title(modality_labels[m], fontsize=14)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_proposed_cm_v2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Baseline CM 2x2 per modality / fusion strategy
    for m in BASELINE_MODALITIES:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
        for ax, fam in zip(axes.flatten(), baseline_families):
            proba = baseline_probas[f"{fam}-{m}"]
            cm_ = confusion_matrix(y_test, get_pred(proba), labels=list(range(NUM_CLASSES)))
            sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False, annot_kws={"size": 13}, square=True)
            ax.set_title(fam, fontsize=15)
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("True", fontsize=12)
            ax.tick_params(labelsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"fig_baseline_{m}_cm_2x2_v2.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Component ablation CM per modality (2x3 panel; late fusion has fewer cells)
    for internal_m, tag in comp_modalities:
        variants = [c for c in VARIANTS_BY_MODALITY[internal_m] if c != "full"]
        ncols = 3 if len(variants) > 4 else 2
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 10.5))
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, c in zip(np.ravel(axes), variants):
            ax.axis("on")
            proba = component_probas[f"{c}__{internal_m}"]
            cm_ = confusion_matrix(y_test, get_pred(proba), labels=list(range(NUM_CLASSES)))
            sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False, annot_kws={"size": 12}, square=True)
            ax.set_title(component_display[c], fontsize=13)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"fig_component_{tag}_cm_v2.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Baseline merged ROC
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2))
    for ax, m in zip(axes, BASELINE_MODALITIES):
        for fam in FAMILY_ORDER:
            proba = baseline_probas[f"{fam}-{m}"]
            fpr_grid, mean_tpr, auc_val = macro_ovr_roc(y_test, proba)
            is_proposed = fam == "MSTC-IDS"
            ax.plot(fpr_grid, mean_tpr, label=f"{fam} ({auc_val:.3f})",
                    color=model_colors[fam], linewidth=3.2 if is_proposed else 1.8)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(modality_labels[m], fontsize=15)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_baseline_roc_merged.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Component ablation merged ROC
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2))
    for ax, (internal_m, tag) in zip(axes, comp_modalities):
        variants = [c for c in VARIANTS_BY_MODALITY[internal_m] if c != "full"]
        palette = cm.tab10.colors[:len(variants)]
        for i, c in enumerate(variants):
            proba = component_probas[f"{c}__{internal_m}"]
            fpr_grid, mean_tpr, auc_val = macro_ovr_roc(y_test, proba)
            ax.plot(fpr_grid, mean_tpr, label=f"{component_display[c]} ({auc_val:.3f})",
                    color=palette[i], linewidth=2.0)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(modality_labels.get(internal_m, modality_display[internal_m]), fontsize=15)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_component_roc_merged.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("manuscript_v6_final figures done")


if __name__ == "__main__":
    ensure_dirs()
    y_test, baseline_probas, component_probas, baseline_metrics, component_metrics = load_data()

    # inject MSTC-IDS entries into baseline metrics/probas for figures that include it
    internal_map = {"cyber": "full__cyber", "physical": "full__physical",
                    "fused": "full__both", "latefusion": "full__late"}
    for m in BASELINE_MODALITIES:
        src = internal_map[m]
        baseline_probas[f"MSTC-IDS-{m}"] = component_probas[src]
        baseline_metrics[f"MSTC-IDS-{m}"] = component_metrics[src]

    figures_proposed_model(y_test, component_probas)
    figures_modality_ablation(y_test, component_probas, component_metrics)
    figures_baseline_comparison(y_test, baseline_probas, baseline_metrics)
    figures_component_ablation(y_test, component_probas, component_metrics)
    figures_manuscript(y_test, baseline_probas, component_probas)
    print("ALL FIGURES DONE")
