"""Regenerate the manuscript-ready (larger-font) figures in figures/manuscript_v6_final/.

This is a thin wrapper around the manuscript figure routines in make_figures.py so
the two scripts can never drift apart. It rebuilds:

  fig_proposed_cm_v2.png                 proposed MSTC-IDS confusion matrices
                                         (cyber, physical, early + late fusion)
  fig_baseline_{m}_cm_2x2_v2.png         baseline confusion matrices per modality
  fig_component_{tag}_cm_v2.png          component-ablation confusion matrices
  fig_baseline_roc_merged.png            baseline macro-OvR ROC per modality
  fig_component_roc_merged.png           component-ablation macro-OvR ROC per modality

Run after compile_results.py + make_late_fusion.py.
"""

import os

import make_figures as mf


def main():
    mf.ensure_dirs()
    y_test, baseline_probas, component_probas, baseline_metrics, component_metrics = mf.load_data()

    internal_map = {"cyber": "full__cyber", "physical": "full__physical",
                    "fused": "full__both", "latefusion": "full__late"}
    for m in mf.BASELINE_MODALITIES:
        src = internal_map[m]
        baseline_probas[f"MSTC-IDS-{m}"] = component_probas[src]
        baseline_metrics[f"MSTC-IDS-{m}"] = component_metrics[src]

    mf.figures_manuscript(y_test, baseline_probas, component_probas)
    print("manuscript figures regenerated ->", os.path.join(mf.FIG_DIR, "manuscript_v6_final"))


if __name__ == "__main__":
    main()
