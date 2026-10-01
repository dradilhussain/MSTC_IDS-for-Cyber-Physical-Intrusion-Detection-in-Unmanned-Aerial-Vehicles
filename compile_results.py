"""
MSTC-IDS results compiler.

Reads the per-model metrics/confusion matrices recorded by train_all.py and
writes every summary CSV/JSON that the notebooks' "Save Results" cells produce,
plus the proposed-model summary JSON for the complete-pipeline notebook.

Output:
  results/baseline_comparison_summary.csv
  results/baseline_comparison_group_{cyber,physical,fused}.csv
  results/baseline_comparison_per_class_f1.csv
  results/modality_ablation_summary.csv
  results/modality_ablation_per_class_f1.csv
  results/modality_ablation_full.json
  results/component_modality_ablation_summary.csv
  results/component_modality_ablation_pivot_f1.csv
  results/component_modality_ablation_per_class_f1.csv
  results/component_modality_ablation_impact_ranking.csv
  results/proposed_model/summary_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
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

os.makedirs(os.path.join(RESULTS_DIR, "proposed_model"), exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    baseline_metrics = load_json(os.path.join(RESULTS_DIR, "baseline_comparison_full.json"))
    component_metrics = load_json(os.path.join(RESULTS_DIR, "component_modality_ablation_full.json"))

    # ----------------------------------------------------------
    # Baseline comparison
    # ----------------------------------------------------------
    def model_name(fam, m):
        return f"{fam}-{m}"

    MODEL_ORDER = [model_name(fam, m) for fam in FAMILY_ORDER for m in BASELINE_MODALITIES]

    # fill MSTC-IDS entries from the component full models
    # (fused == early cross-attention fusion; latefusion == validation-selected
    #  decision-level combination of the trained branches)
    MSTC_SOURCE = {"cyber": "full__cyber", "physical": "full__physical",
                   "fused": "full__both", "latefusion": "full__late"}
    for m in BASELINE_MODALITIES:
        src = component_metrics[MSTC_SOURCE[m]]
        baseline_metrics[model_name("MSTC-IDS", m)] = {
            **src, "name": model_name("MSTC-IDS", m), "component": None, "modality": m,
        }

    # late-fusion baselines come from make_late_fusion.py (validation-selected
    # decision-level combination of the trained cyber/physical branches)
    late_summary = load_json(os.path.join(RESULTS_DIR, "late_fusion", "late_fusion_summary.json"))
    for fam in FAMILY_ORDER:
        if fam == "MSTC-IDS":
            continue
        t = late_summary["baselines"][fam]["test"]
        c, p = baseline_metrics[f"{fam}-cyber"], baseline_metrics[f"{fam}-physical"]
        params = (c["trainable_params"] + p["trainable_params"]
                  if c["trainable_params"] is not None and p["trainable_params"] is not None
                  else None)
        baseline_metrics[f"{fam}-latefusion"] = {
            "name": f"{fam}-latefusion", "component": None, "modality": "latefusion",
            "accuracy": t["accuracy"], "precision_weighted": t["precision"],
            "recall_weighted": t["recall"], "f1_weighted": t["f1"],
            "macro_auc": t["macro_auc"], "per_class_f1": t["per_class_f1"],
            "confusion_matrix": t["confusion_matrix"],
            "trainable_params": params, "epochs_trained": None,
            "checkpoint": f"late fusion of baseline_{fam}-cyber and baseline_{fam}-physical",
        }
    baseline_metrics.pop("MSTC-IDS-fused-late", None)

    summary_rows = []
    for name in MODEL_ORDER:
        r = baseline_metrics[name]
        summary_rows.append({
            "Model": name,
            "Params/N": f"{r['trainable_params']:,}" if r["trainable_params"] is not None else "-",
            "Epochs": r["epochs_trained"] if r["epochs_trained"] is not None else "-",
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision_weighted"], 4),
            "Recall": round(r["recall_weighted"], 4),
            "F1": round(r["f1_weighted"], 4),
            "Macro AUC": round(r["macro_auc"], 4) if r["macro_auc"] is not None else None,
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    summary_df.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison_summary.csv"))

    for m in BASELINE_MODALITIES:
        group = summary_df.loc[[model_name(fam, m) for fam in FAMILY_ORDER]]
        group.to_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_group_{m}.csv"))

    per_class_df = pd.DataFrame({name: baseline_metrics[name]["per_class_f1"] for name in MODEL_ORDER}).T
    per_class_df = per_class_df[CLASS_NAMES]
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison_per_class_f1.csv"))

    with open(os.path.join(RESULTS_DIR, "baseline_comparison_merged.json"), "w") as f:
        json.dump({name: {**baseline_metrics[name], "confusion_matrix": baseline_metrics[name]["confusion_matrix"]}
                   for name in MODEL_ORDER}, f, indent=2)
    print("baseline_comparison_*.csv written")

    # ----------------------------------------------------------
    # Modality ablation (reuses full__{m})
    # ----------------------------------------------------------
    display_names = {"cyber": "Cyber-only", "physical": "Physical-only",
                     "both": "Cyber-Physical (early fusion)",
                     "late": "Cyber-Physical (late fusion)"}

    modality_summary_rows = []
    for m in MODALITIES:
        r = component_metrics[f"full__{m}"]
        modality_summary_rows.append({
            "Variant": display_names[m],
            "Params": f"{r['trainable_params']:,}",
            "Epochs": r["epochs_trained"] if r["epochs_trained"] is not None else "-",
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision_weighted"], 4),
            "Recall": round(r["recall_weighted"], 4),
            "F1": round(r["f1_weighted"], 4),
            "Macro AUC": round(r["macro_auc"], 4) if r["macro_auc"] is not None else None,
        })
    modality_summary_df = pd.DataFrame(modality_summary_rows).set_index("Variant")
    modality_summary_df.to_csv(os.path.join(RESULTS_DIR, "modality_ablation_summary.csv"))

    modality_per_class = pd.DataFrame({
        display_names[m]: component_metrics[f"full__{m}"]["per_class_f1"] for m in MODALITIES
    }).T[CLASS_NAMES]
    modality_per_class.to_csv(os.path.join(RESULTS_DIR, "modality_ablation_per_class_f1.csv"))

    with open(os.path.join(RESULTS_DIR, "modality_ablation_full.json"), "w") as f:
        json.dump({
            m: {**component_metrics[f"full__{m}"],
                "confusion_matrix": component_metrics[f"full__{m}"]["confusion_matrix"]}
            for m in MODALITIES
        }, f, indent=2)
    print("modality_ablation_*.csv written")

    # ----------------------------------------------------------
    # Component x modality ablation
    # ----------------------------------------------------------
    ALL_KEYS = [f"{c}__{m}" for m in MODALITIES for c in VARIANTS_BY_MODALITY[m]]

    def display_name(key):
        c, m = key.split("__")
        return f"{component_display[c]} ({modality_display[m]})"

    comp_summary_rows = []
    for key in ALL_KEYS:
        r = component_metrics[key]
        comp_summary_rows.append({
            "Combination": display_name(key),
            "Component": component_display[r["component"]],
            "Modality": modality_display[r["modality"]],
            "Params": f"{r['trainable_params']:,}",
            "Epochs": r["epochs_trained"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision_weighted"], 4),
            "Recall": round(r["recall_weighted"], 4),
            "F1": round(r["f1_weighted"], 4),
            "Macro AUC": round(r["macro_auc"], 4) if r["macro_auc"] is not None else None,
        })
    comp_summary_df = pd.DataFrame(comp_summary_rows).set_index("Combination")
    comp_summary_df.to_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_summary.csv"))

    pivot_f1 = pd.DataFrame({
        m: pd.Series({component_display[c]: component_metrics[f"{c}__{m}"]["f1_weighted"]
                      for c in VARIANTS_BY_MODALITY[m]})
        for m in MODALITIES
    }).reindex([component_display[c] for c in EARLY_VARIANTS])
    pivot_f1.columns = [modality_display[m] for m in MODALITIES]
    pivot_f1.round(4).to_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_pivot_f1.csv"))

    comp_per_class = pd.DataFrame({display_name(key): component_metrics[key]["per_class_f1"] for key in ALL_KEYS}).T
    comp_per_class = comp_per_class[CLASS_NAMES]
    comp_per_class.to_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_per_class_f1.csv"))

    impact_dfs = {}
    for m in MODALITIES:
        full_f1 = component_metrics[f"full__{m}"]["f1_weighted"]
        rows = []
        for c in VARIANTS_BY_MODALITY[m]:
            if c == "full":
                continue
            delta = full_f1 - component_metrics[f"{c}__{m}"]["f1_weighted"]
            rows.append({"Component removed": component_display[c], "F1 drop": round(delta, 4)})
        impact_dfs[m] = pd.DataFrame(rows).sort_values("F1 drop", ascending=False).set_index("Component removed")
    impact_df = pd.concat(impact_dfs, names=["Modality", "Component removed"])
    impact_df.to_csv(os.path.join(RESULTS_DIR, "component_modality_ablation_impact_ranking.csv"))
    print("component_modality_ablation_*.csv written")

    # ----------------------------------------------------------
    # Proposed-model summary JSON (complete notebook)
    # ----------------------------------------------------------
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    from sklearn.metrics import accuracy_score, roc_auc_score
    comp_probas = dict(np.load(os.path.join(RESULTS_DIR, "probabilities", "component_probas.npz")))

    def auc(y, proba):
        return (float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
                float(roc_auc_score(y, proba, multi_class="ovr", average="weighted")))

    r = component_metrics["full__both"]
    proba = comp_probas["full__both"]
    macro_auc, weighted_auc = auc(y_test, proba)
    acc = float(accuracy_score(y_test, proba.argmax(1)))

    r_late = component_metrics["full__late"]
    proba_late = comp_probas["full__late"]
    macro_auc_late, weighted_auc_late = auc(y_test, proba_late)
    acc_late = float(accuracy_score(y_test, proba_late.argmax(1)))

    # Joint-training history is available for the early-fusion model only.
    hist_path = os.path.join(RESULTS_DIR, "histories", "full__both.json")
    with open(os.path.join(RESULTS_DIR, "proposed_model", "summary_metrics.json"), "w") as f:
        json.dump({
            "fusion": "early (gated cross-attention) fusion of the cyber and "
                      "physical streams",
            "fusion_strategies": {
                "early_cross_attention": {
                    "test_accuracy": acc,
                    "weighted_precision": r["precision_weighted"],
                    "weighted_recall": r["recall_weighted"],
                    "weighted_f1": r["f1_weighted"],
                    "macro_auc": macro_auc,
                    "weighted_auc": weighted_auc,
                    "trainable_params": r["trainable_params"],
                    "checkpoint": r["checkpoint"],
                },
                "late_decision_level": {
                    "test_accuracy": acc_late,
                    "weighted_precision": r_late["precision_weighted"],
                    "weighted_recall": r_late["recall_weighted"],
                    "weighted_f1": r_late["f1_weighted"],
                    "macro_auc": macro_auc_late,
                    "weighted_auc": weighted_auc_late,
                    "trainable_params": r_late["trainable_params"],
                    "checkpoint": r_late["checkpoint"],
                    "rule": "validation-selected w: "
                            "p = w * p_physical + (1 - w) * p_cyber",
                },
            },
            "test_accuracy": acc,
            "weighted_precision": r["precision_weighted"],
            "weighted_recall": r["recall_weighted"],
            "weighted_f1": r["f1_weighted"],
            "macro_auc": macro_auc,
            "weighted_auc": weighted_auc,
            "epochs_trained": r["epochs_trained"],
            "trainable_params": r["trainable_params"],
            "checkpoint": r["checkpoint"],
            "class_names": CLASS_NAMES,
            "history_file": os.path.relpath(hist_path, ROOT),
            "history_note": "early-fusion joint training history; the late-fusion "
                            "strategy combines the saved branch outputs",
        }, f, indent=2)
    print("proposed_model/summary_metrics.json written")

    # notebook-facing aliases used by MSTC_IDS_Complete.ipynb
    import shutil
    shutil.copy2(os.path.join(RESULTS_DIR, "proposed_model", "summary_metrics.json"),
                 os.path.join(RESULTS_DIR, "summary_metrics.json"))
    shutil.copy2(hist_path, os.path.join(RESULTS_DIR, "training_history.json"))
    print("notebook aliases summary_metrics.json / training_history.json written")

    print("\nAll result tables written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
