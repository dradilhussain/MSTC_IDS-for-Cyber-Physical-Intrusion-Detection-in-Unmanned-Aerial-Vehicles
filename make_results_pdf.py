"""
Builds a single self-contained PDF report of all results & analysis.

Layout follows the requested order inside every comparison block:
  1) confusion matrices
  2) tables
  3) AUC / F1 / metrics figures

Sections:
  A. Proposed MSTC-IDS model, three modalities (cyber / physical / cyber-physical)
  B. Baseline comparison, per modality (3 separate sub-comparisons)
  C. Hyperparameters & model size, per modality
  D. Component ablation, per modality

Output: results/results_analysis_report.pdf

Run after compile_results.py, make_figures.py, make_comparison.py and
parameter_analysis.py (all outputs this script reads already exist).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
COMP_DIR = os.path.join(ROOT, "comparison")
FIG_DIR = os.path.join(ROOT, "figures")
OUT_PDF = os.path.join(RESULTS_DIR, "results_analysis_report.pdf")

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
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
# component-ablation modality labels (match compile_results modality_display)
comp_label = {"cyber": "Cyber-only", "physical": "Physical-only",
              "both": "Early fusion", "late": "Late fusion"}
# baseline / proposed display labels
modality_label = {"cyber": "Cyber-only", "physical": "Physical-only",
                  "fused": "Cyber-Physical (early fusion)",
                  "latefusion": "Cyber-Physical (late fusion)",
                  "both": "Early fusion"}

PAGE_W, PAGE_H = 11.69, 8.27  # A4 landscape (inches)
_page_no = [0]


def _next_page(pdf):
    _page_no[0] += 1
    return _page_no[0]


def _header(fig, section, title):
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.965, section, fontsize=10, color="#555555",
             ha="left", va="top", family="sans-serif")
    fig.text(0.98, 0.965, title, fontsize=14, fontweight="bold", color="#1a1a1a",
             ha="right", va="top", family="sans-serif")
    fig.text(0.5, 0.955, "", fontsize=9, color="#666666", ha="center", va="top")
    fig.text(0.02, 0.012, f"Page {_page_no[0]}", fontsize=8, color="#999999",
             ha="left", va="bottom")


def new_page(pdf, section, title):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _header(fig, section, title)
    _next_page(pdf)
    return fig


def embed_png(pdf, path, section, title, caption=None):
    img = plt.imread(path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _header(fig, section, title)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
    ax.imshow(img, aspect="equal")
    ax.axis("off")
    if caption:
        fig.text(0.5, 0.018, caption, fontsize=8.5, color="#555555",
                 ha="center", va="bottom", wrap=True)
    _next_page(pdf)
    pdf.savefig(fig)
    plt.close(fig)


def render_table(pdf, df, section, title, caption=None, fontsize=None,
                 index_label=None):
    df = df.copy()
    if index_label is not None:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": index_label})
    nrows, ncols = df.shape
    col_labels = [str(c) for c in df.columns]
    cell_text = [[_fmt(v) for v in row] for row in df.values]

    # proportional column widths from content length, plus adaptive font so the
    # widest row always fits on the landscape page (prevents text clipping)
    head_lens = [len(h) for h in col_labels]
    body_lens = [max(len(cell_text[i][j]) for i in range(nrows)) for j in range(ncols)]
    maxlens = [max(a, b) + 1 for a, b in zip(head_lens, body_lens)]
    total_units = float(sum(maxlens))
    col_widths = [m / total_units for m in maxlens]
    available_pt = 0.94 * PAGE_W * 72.0
    fit_font = available_pt / (0.62 * total_units) if total_units > 0 else 20.0
    if fontsize is None:
        fontsize = min(9.5, fit_font)
    fontsize = max(4.5, min(fontsize, fit_font))

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _header(fig, section, title)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                   cellLoc="center", colLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.45)
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor("#e8eef7")
        cell.set_text_props(fontweight="bold", fontsize=fontsize + 0.5)
    for i in range(nrows):
        if i % 2 == 1:
            for j in range(ncols):
                tbl[i + 1, j].set_facecolor("#f5f7fa")
    if caption:
        fig.text(0.5, 0.02, caption, fontsize=8.5, color="#555555",
                 ha="center", va="bottom", wrap=True)
    _next_page(pdf)
    pdf.savefig(fig)
    plt.close(fig)


def _fmt(v):
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        return f"{v:.4f}" if abs(v) < 1 else f"{v:g}"
    return str(v)


def title_page(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.82,
             "MSTC-IDS: Multi-Scale Temporal Convolutional\nCyber-Physical "
             "Intrusion Detection for UAV Networks",
             fontsize=24, fontweight="bold", color="#1a1a1a",
             ha="center", va="center", linespacing=1.4)
    fig.text(0.5, 0.66, "Full Results & Analysis Report", fontsize=16,
             color="#444444", ha="center", va="center")
    fig.text(0.5, 0.6, "Test split: chronological per-segment 70/15/15, "
                      "window size 20", fontsize=10, color="#777777",
             ha="center", va="center")
    toc = [
        "A.  Proposed Model - MSTC-IDS across input modalities and fusion strategies",
        "     A1. Confusion matrices    A2. Summary & per-class tables    A3. ROC / F1 / metrics",
        "",
        "B.  Baseline Comparison - one sub-section per modality / fusion strategy",
        "     B1. Cyber-only   B2. Physical-only   B3. Early fusion   B4. Late fusion",
        "     Each: confusion matrices -> summary table -> ROC / metrics",
        "",
        "C.  Hyperparameters & Model Size - per modality",
        "     Trained hyperparameters, parameters vs. performance, efficiency",
        "",
        "D.  Component Ablation - per modality",
        "     Confusion matrices -> ablation tables -> ROC / impact ranking",
    ]
    fig.text(0.5, 0.42, "\n".join(toc), fontsize=11.5, color="#333333",
             ha="center", va="center", family="monospace", linespacing=1.55)
    fig.text(0.5, 0.06,
             "Generated from results/ (records + saved prediction probabilities); "
             "figures reused from comparison/.",
             fontsize=8.5, color="#999999", ha="center", va="center")
    _next_page(pdf)
    pdf.savefig(fig)
    plt.close(fig)


def divider(pdf, section, title, body):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("#f2f5fa")
    fig.text(0.5, 0.72, section, fontsize=26, fontweight="bold", color="#1a1a1a",
             ha="center", va="center")
    fig.text(0.5, 0.58, title, fontsize=15, color="#333333",
             ha="center", va="center")
    fig.text(0.5, 0.42, body, fontsize=10.5, color="#555555",
             ha="center", va="center", linespacing=1.5)
    _next_page(pdf)
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_csv(path, index_col=0):
    return pd.read_csv(path, index_col=index_col)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        title_page(pdf)

        # ----------------------------------------------------
        # SECTION A: Proposed model
        # ----------------------------------------------------
        divider(pdf, "A", "Proposed Model - MSTC-IDS across input modalities",
                "The same MSTC-IDS architecture trained on cyber-only, physical-only\n"
                "streams, plus two cyber-physical fusion strategies: early (gated\n"
                "cross-attention) and late (validation-selected decision-level).\n"
                "Reported on the held-out test split.\n\n"
                "Order: confusion matrices -> tables -> ROC / F1 / metrics.")

        a = os.path.join(COMP_DIR, "1_proposed_model")
        embed_png(pdf, os.path.join(a, "confusion_matrices.png"), "A. Proposed Model",
                  "A1. Confusion Matrices - Cyber / Physical / Early / Late Fusion",
                  "Confusion matrices of the full MSTC-IDS model for each input "
                  "configuration.")
        render_table(pdf, load_csv(os.path.join(RESULTS_DIR, "modality_ablation_summary.csv")),
                     "A. Proposed Model", "A2. Summary Table - Overall Metrics by Modality",
                     "Weighted accuracy / precision / recall / F1 and macro-average ROC-AUC, "
                     "with trainable parameters and epochs trained.",
                     index_label="Modality")
        render_table(pdf, load_csv(os.path.join(RESULTS_DIR, "modality_ablation_per_class_f1.csv")),
                     "A. Proposed Model", "A2. Per-Class F1 Table by Modality",
                     "F1 per attack class (macro-average over OvR ROC for AUC; "
                     "classes: Benign, DoS, Replay, EvilTwin, FDI).",
                     index_label="Modality")
        embed_png(pdf, os.path.join(a, "roc_full_model.png"), "A. Proposed Model",
                  "A3. ROC Curves - Macro-OvR, all input configurations",
                  "Macro-averaged one-vs-rest ROC; legend shows macro AUC per configuration.")
        fig = new_page(pdf, "A. Proposed Model", "A3. Overall Metrics & Per-Class F1")
        ax1 = fig.add_axes([0.04, 0.10, 0.44, 0.80])
        ax1.imshow(plt.imread(os.path.join(a, "overall_metrics.png")), aspect="equal")
        ax1.axis("off")
        ax2 = fig.add_axes([0.52, 0.10, 0.44, 0.80])
        ax2.imshow(plt.imread(os.path.join(a, "per_class_f1.png")), aspect="equal")
        ax2.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        embed_png(pdf, os.path.join(a, "training_curves.png"), "A. Proposed Model",
                  "A3. Training Curves - validation loss & accuracy",
                  "Training behaviour of the full branch models (val_loss is noisy due to "
                  "the contrastive term; val_accuracy is the monitored signal).")

        # ----------------------------------------------------
        # SECTION B: Baseline comparison, per modality
        # ----------------------------------------------------
        divider(pdf, "B", "Baseline Comparison - one sub-section per modality",
                "Separate comparisons, one per input modality / fusion strategy:\n"
                "MSTC-IDS vs. SVM, FNN, LSTM and 1D-CNN, all trained on byte-identical data.\n"
                "The late-fusion rows combine the trained cyber-only and physical-only\n"
                "baselines at the decision level.\n\n"
                "Each sub-section: confusion matrices -> summary table -> ROC / metrics.")

        bcs = load_csv(os.path.join(RESULTS_DIR, "baseline_comparison_per_class_f1.csv"))
        b_mods = [("cyber", "cyber_only", "Cyber-only"),
                  ("physical", "physical_only", "Physical-only"),
                  ("fused", "cyber_physical", "Cyber-Physical (early fusion)"),
                  ("latefusion", "cyber_physical_late", "Cyber-Physical (late fusion)")]
        for idx, (m, subdir, label) in enumerate(b_mods, start=1):
            b = os.path.join(COMP_DIR, "2_baseline_comparison", subdir)
            tag = f"B{idx}"
            embed_png(pdf, os.path.join(b, "confusion_matrix.png"), f"B. Baseline - {label}",
                      f"{tag}. Confusion Matrices - {label}",
                      "One row of confusion matrices for all five models "
                      "(SVM, FNN, LSTM, 1D-CNN, MSTC-IDS) on the same test split.")
            render_table(pdf,
                         load_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_group_{m}.csv")),
                         f"B. Baseline - {label}", f"{tag}. Summary Table - {label}",
                         "Weighted accuracy / precision / recall / F1 and macro ROC-AUC, "
                         "with model size and epochs.", index_label="Model")
            pcf_mod = bcs[bcs.index.str.endswith(f"-{m}")]
            render_table(pdf, pcf_mod, f"B. Baseline - {label}",
                         f"{tag}. Per-Class F1 Table - {label}",
                         "Per-class F1 for every model in this modality group.",
                         index_label="Model")
            embed_png(pdf, os.path.join(b, "roc.png"), f"B. Baseline - {label}",
                      f"{tag}. ROC Curves - {label}",
                      "Macro-averaged one-vs-rest ROC across the five models; "
                      "MSTC-IDS highlighted.")
            fig = new_page(pdf, f"B. Baseline - {label}",
                           f"{tag}. Overall Metrics & Per-Class F1")
            ax1 = fig.add_axes([0.04, 0.10, 0.44, 0.80])
            ax1.imshow(plt.imread(os.path.join(b, "metrics_bar.png")), aspect="equal")
            ax1.axis("off")
            ax2 = fig.add_axes([0.52, 0.10, 0.44, 0.80])
            ax2.imshow(plt.imread(os.path.join(b, "per_class_f1.png")), aspect="equal")
            ax2.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # SECTION C: Hyperparameters & model size, per modality
        # ----------------------------------------------------
        divider(pdf, "C", "Hyperparameters & Model Size - per modality",
                "Optimized training hyperparameters for every model family, plus\n"
                "parameters-vs-performance and efficiency (F1 per 1K parameters)\n"
                "evaluated separately for each modality.\n\n"
                "Configs come from results/optimization/best_configs.json.")

        cfg = json.load(open(os.path.join(RESULTS_DIR, "optimization", "best_configs.json")))

        def fmt_cfg(d):
            parts = []
            for k, v in d.items():
                if k == "lr":
                    s = f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e").replace("e-", "e-")
                    parts.append(f"lr={s}")
                elif isinstance(v, list):
                    parts.append(f"{k}={v}")
                elif isinstance(v, float):
                    parts.append(f"{k}={v:g}")
                else:
                    parts.append(f"{k}={v}")
            return ", ".join(parts)

        overrides = {
            "SVM": {"cyber": "svm_cyber"},
            "FNN": {"cyber": "fnn_cyber"},
            "1D-CNN": {"physical": "cnn_physical", "fused": "cnn_fused"},
            "MSTC-IDS": {"both": "mstc_both"},
        }
        base_keys = {"SVM": "svm", "FNN": "fnn", "LSTM": "lstm",
                     "1D-CNN": "cnn", "MSTC-IDS": "mstc"}
        mod_cols = ["cyber", "physical", "both"]
        hyper_rows = []
        for fam in ["SVM", "FNN", "LSTM", "1D-CNN", "MSTC-IDS"]:
            row = {"Family": fam}
            for mc in mod_cols:
                key = overrides.get(fam, {}).get(mc)
                if key is None:
                    key = base_keys[fam]
                row[modality_label[mc]] = fmt_cfg(cfg[key])
            hyper_rows.append(row)
        render_table(pdf, pd.DataFrame(hyper_rows).set_index("Family"),
                     "C. Hyperparameters", "C1. Optimized Training Hyperparameters",
                     "Per-family configuration after hyperparameter optimization; "
                     "overrides shown for the modality where they differ.")

        pa = os.path.join(RESULTS_DIR, "parameter_analysis")
        pa_cols = ["Model", "Params", "Epochs", "Accuracy", "Precision", "Recall",
                   "F1", "MacroAUC", "F1_per_1K_params"]
        for mc, label, fname in [("cyber", "Cyber-only", "cyberonly"),
                                 ("physical", "Physical-only", "physicalonly"),
                                 ("both", "Cyber-Physical (early fusion)", "cyberphysical"),
                                 ("late", "Cyber-Physical (late fusion)", "cyberphysicallate")]:
            grp = load_csv(os.path.join(pa, f"performance_vs_parameters_group_{fname}.csv"),
                           index_col=None)
            grp = grp[pa_cols]
            render_table(pdf, grp, "C. Hyperparameters",
                         f"C2. Parameters & Performance - {label}",
                         "Model size (trainable parameters), epochs and test-set "
                         "performance for every model in this modality group.")

        fig = new_page(pdf, "C. Hyperparameters", "C3. Parameters vs. Performance (all modalities)")
        ax1 = fig.add_axes([0.03, 0.08, 0.47, 0.84])
        ax1.imshow(plt.imread(os.path.join(FIG_DIR, "parameter_analysis", "f1_vs_parameters.png")),
                   aspect="equal")
        ax1.axis("off")
        ax2 = fig.add_axes([0.52, 0.08, 0.47, 0.84])
        ax2.imshow(plt.imread(os.path.join(FIG_DIR, "parameter_analysis", "accuracy_vs_parameters.png")),
                   aspect="equal")
        ax2.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
        embed_png(pdf, os.path.join(FIG_DIR, "parameter_analysis", "efficiency_bar.png"),
                  "C. Hyperparameters", "C3. Efficiency - Weighted F1 per 1K Parameters",
                  "Top-2 models per family per modality by F1, ranked by "
                  "F1 per 1,000 parameters.")

        # ----------------------------------------------------
        # SECTION D: Component ablation, per modality
        # ----------------------------------------------------
        divider(pdf, "D", "Component Ablation - per modality",
                "Component ablations of the proposed model per modality, each removing\n"
                "one component: Time2Vec, cross-attention (early fusion only), contrastive\n"
                "loss (early fusion only), multi-scale pyramid, causal masking, freshness\n"
                "embedding. Both fusion strategies are ablated.\n\n"
                "Each sub-section: confusion matrices -> ablation table -> ROC / impact.")

        comp_all = load_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_summary.csv"))
        comp_pcf = load_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_per_class_f1.csv"))
        d_mods = [("cyber", "cyber_only", "Cyber-only"),
                  ("physical", "physical_only", "Physical-only"),
                  ("both", "cyber_physical", "Cyber-Physical (early fusion)"),
                  ("late", "cyber_physical_late", "Cyber-Physical (late fusion)")]
        for idx, (m, subdir, label) in enumerate(d_mods, start=1):
            b = os.path.join(COMP_DIR, "3_component_ablation", subdir)
            tag = f"D{idx}"
            variants = VARIANTS_BY_MODALITY[m]
            embed_png(pdf, os.path.join(b, "confusion_matrix.png"), f"D. Ablation - {label}",
                      f"{tag}. Confusion Matrices - {label}",
                      f"Full model plus the {len(variants) - 1} ablations "
                      f"(1x{len(variants)}). Full model first.")
            ab = comp_all[comp_all["Modality"] == comp_label[m]].drop(columns=["Modality"])
            render_table(pdf, ab, f"D. Ablation - {label}",
                         f"{tag}. Ablation Table - {label}",
                         "Accuracy / precision / recall / F1 / macro AUC for the full "
                         f"model and all {len(variants) - 1} ablations.", index_label="Combination")
            pcf_mod = comp_pcf[comp_pcf.index.str.endswith(f"({comp_label[m]})")]
            render_table(pdf, pcf_mod, f"D. Ablation - {label}",
                         f"{tag}. Per-Class F1 Table - {label}",
                         f"Per-class F1 across the {len(variants)} variants.",
                         index_label="Combination")
            embed_png(pdf, os.path.join(b, "roc.png"), f"D. Ablation - {label}",
                      f"{tag}. ROC Curves - {label}",
                      f"Macro-OvR ROC for all {len(variants)} variants; full model in black.")
            embed_png(pdf, os.path.join(b, "impact.png"), f"D. Ablation - {label}",
                      f"{tag}. Impact Ranking - {label}",
                      "Weighted-F1 drop when each component is removed vs. the full model; "
                      "positive = component helps.")

        print("PDF written:", OUT_PDF)


if __name__ == "__main__":
    main()
