"""Regenerate the manuscript performance tables under results/performance_tables/

T1_proposed_all_modalities.csv        : proposed MSTC-IDS per input modality
                                        (cyber, physical, early and late fusion).
T2_baseline_comparison_{m}.csv        : proposed + tuned baselines per modality
                                        (cyber, physical, and both fusion strategies).
T3_ablation_{m}.csv                   : proposed component ablations per modality.

Both cyber-physical fusion strategies are reported:

  * early fusion - a single network trained jointly on the concatenated cyber
    and physical streams with gated cross-attention;
  * late fusion  - the validation-selected decision-level combination of the
    independently trained cyber-only and physical-only branch outputs.

Inputs are the CSVs produced by compile_results.py from the canonical results
jsons, so the tables always reflect the same metrics used everywhere else.
"""

import os
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "performance_tables")
os.makedirs(OUT, exist_ok=True)

MODALITY_GROUP = {
    "cyber": "cyber",
    "physical": "physical",
    "cyber_physical_early": "fused",
    "cyber_physical_late": "latefusion",
}
T3_MODALITY = {"cyber": "Cyber-only", "physical": "Physical-only",
               "cyber_physical_early": "Early fusion",
               "cyber_physical_late": "Late fusion"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", os.path.relpath(path, ROOT))


def colmap(row):
    # numeric-safe subset of the columns shared by the intermediate CSVs
    def num(key, default=None):
        v = row.get(key)
        try:
            return float(v.replace(",", "")) if isinstance(v, str) else float(v)
        except (TypeError, ValueError):
            return default
    return {
        "Accuracy": num("Accuracy"), "Precision": num("Precision"),
        "Recall": num("Recall"), "F1": num("F1"), "MacroAUC": num("Macro AUC", num("MacroAUC")),
    }


# ---- T1 : proposed model across the input modalities / fusion strategies ----
comp = read_csv(os.path.join(RESULTS, "component_modality_ablation_summary.csv"))
full = {r["Modality"]: r for r in comp if r["Component"] == "Full model"}
t1_rows = []
for label, mod in [("Cyber-only", "Cyber-only"), ("Physical-only", "Physical-only"),
                   ("Cyber-Physical (early fusion)", "Early fusion"),
                   ("Cyber-Physical (late fusion)", "Late fusion")]:
    r = full[mod]
    t1_rows.append({
        "Modality": label, **{k: f"{v:.4f}" for k, v in colmap(r).items()},
        "Params": r["Params"], "Epochs": r["Epochs"],
    })
write_csv(os.path.join(OUT, "T1_proposed_all_modalities.csv"), t1_rows,
          ["Modality", "Accuracy", "Precision", "Recall", "F1", "MacroAUC", "Params", "Epochs"])


# ---- T2 : proposed vs tuned baselines per modality / fusion strategy ----
for tbl_mod, group_mod in MODALITY_GROUP.items():
    group = read_csv(os.path.join(RESULTS, f"baseline_comparison_group_{group_mod}.csv"))
    mstc = [r for r in group if r["Model"].startswith("MSTC-IDS-")]
    proposed = mstc[0]
    baselines = [r for r in group if r is not proposed]
    order = {"SVM-": 0, "FNN-": 1, "LSTM-": 2, "1D-CNN-": 3}
    baselines.sort(key=lambda r: order.get(next((k for k in order if r["Model"].startswith(k)), ""), 9))

    def base_name(model):
        for suffix in ("-cyber", "-physical", "-fused", "-latefusion"):
            if model.endswith(suffix):
                return model[: -len(suffix)]
        return model

    rows = [{"Model": "PROPOSED (MSTC-IDS)", **{k: f"{v:.4f}" for k, v in colmap(proposed).items()}}]
    for b in baselines:
        rows.append({"Model": base_name(b["Model"]), **{k: f"{v:.4f}" for k, v in colmap(b).items()}})
    write_csv(os.path.join(OUT, f"T2_baseline_comparison_{tbl_mod}.csv"), rows,
              ["Model", "Accuracy", "Precision", "Recall", "F1", "MacroAUC"])


# ---- T3 : component ablations of the proposed model per modality ----
label_map = {
    "Full model": "PROPOSED (full)",
    "w/o Time2Vec": "w/o Time2Vec",
    "w/o Cross-Attention": "w/o Cross-Attention",
    "w/o Contrastive Loss": "w/o Contrastive Loss",
    "w/o Multi-Scale Pyramid": "w/o Multi-Scale Pyramid",
    "w/o Causal Masking": "w/o Causal Masking",
    "w/o Freshness Embedding": "w/o Freshness Embedding",
}
for tbl_mod, comp_mod in T3_MODALITY.items():
    rows = []
    for r in comp:
        if r["Modality"] != comp_mod or r["Component"] not in label_map:
            continue
        rows.append({"Model": label_map[r["Component"]],
                     **{k: f"{v:.4f}" for k, v in colmap(r).items()}})
    write_csv(os.path.join(OUT, f"T3_ablation_{tbl_mod}.csv"), rows,
              ["Model", "Accuracy", "Precision", "Recall", "F1", "MacroAUC"])

print("all performance tables regenerated")
