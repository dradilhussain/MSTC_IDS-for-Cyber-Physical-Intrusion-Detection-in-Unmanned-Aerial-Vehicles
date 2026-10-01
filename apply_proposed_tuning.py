"""Adopt the validation-selected proposed-model configs into the canonical results.

Selection policy
----------------
For the fused (modality="both") model we keep the paper's design contrastive
weight (default 0.1) unless `--cw` says otherwise. This keeps the T3 component
ablation valid (the "no contrastive" variant remains a genuine ablation). We
select the best configuration *by validation macro-F1* within that weight.
For the cyber model we select the best by validation macro-F1 outright.

Two phases
----------
  python apply_proposed_tuning.py --select   # phase A: pick winners, write configs
  python apply_proposed_tuning.py --merge    # phase B: fold the `_tuned` run into canonical

Phase A writes `results/optimization/best_configs.json` (keys mstc_both /
mstc_cyber) so that a subsequent `train_all.py --modalities both,cyber` with
`MSTC_RESULTS_SUFFIX=_tuned` reproduces the tuned configs for *all* ablation
variants. Phase B merges those `__both` / `__cyber` entries into the canonical
ablation/probability files.
"""

import os
import sys
import json
import argparse
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_all as TA

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_JSON = os.environ.get(
    "MSTC_SWEEP_JSON",
    os.path.join(ROOT, "results", "optimization", "proposed_sweep.json"))
BEST_CFG_JSON = os.path.join(ROOT, "results", "optimization", "best_configs.json")
ABL_JSON = os.path.join(ROOT, "results", "component_modality_ablation_full.json")
ABL_TUNED_JSON = os.path.join(ROOT, "results", "component_modality_ablation_full_tuned.json")
PROBAS = os.path.join(ROOT, "results", "probabilities", "component_probas.npz")
PROBAS_TUNED = os.path.join(ROOT, "results", "probabilities", "component_probas_tuned.npz")
SUFFIX = os.environ.get("MSTC_RESULTS_SUFFIX", "_tuned")


def log(msg):
    print(f"[apply] {msg}", flush=True)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def select(cw_policy):
    """Pick the best-by-validation config per modality.

    For the fused model we require a *positive* contrastive weight so that the
    T3 "no contrastive" ablation remains a genuine component removal. The exact
    positive weight (0.02 / 0.05 / 0.10 ...) is chosen by validation.
    """
    sweep = load_json(SWEEP_JSON)
    winners = {}
    global_best = {}
    for tag, rec in sweep.items():
        m = rec["config"]["modality"]
        if m not in global_best or rec["best_val_accuracy"] > global_best[m]["best_val_accuracy"]:
            global_best[m] = rec
        if m == "both" and cw_policy is None and rec["config"]["cw"] <= 0:
            continue
        if m == "both" and cw_policy is not None and abs(rec["config"]["cw"] - cw_policy) > 1e-9:
            continue
        if m not in winners or rec["best_val_accuracy"] > winners[m]["best_val_accuracy"]:
            winners[m] = rec
    return winners, global_best


def phase_select(cw_policy):
    winners, global_best = select(cw_policy)
    log("global best by val (any cw):")
    for m, r in global_best.items():
        log(f"  {m}: {r['tag']} cw={r['config']['cw']} "
            f"valAcc={r['best_val_accuracy']:.4f} testAcc={r['test_accuracy']:.4f}")
    log("selected:")
    for m, r in winners.items():
        log(f"  {m}: {r['tag']} valAcc={r['best_val_accuracy']:.4f} "
            f"testAcc={r['test_accuracy']:.4f} cfg={r['config']}")

    cfg = load_json(BEST_CFG_JSON)
    for m, r in winners.items():
        c = r["config"]
        cfg[f"mstc_{m}"] = {
            "d": c["d"], "heads": c["heads"], "ff": c["ff"],
            "dropout": c["dropout"], "lr": c["lr"],
            "contrastive_weight": c["cw"], "weight_decay": c["wd"],
        }
    save_json(BEST_CFG_JSON, cfg)
    save_json(os.path.join(ROOT, "results", "optimization", "proposed_best_by_val.json"),
              {m: r["config"] | {"tag": r["tag"]} for m, r in winners.items()})
    log(f"wrote {os.path.relpath(BEST_CFG_JSON, ROOT)}")
    log("next: MSTC_CACHE=$PWD/data MSTC_RESULTS_SUFFIX=_tuned "
        "python3 train_all.py --only mstc --modalities both,cyber")


def phase_merge():
    abl = load_json(ABL_JSON)
    abl_t = load_json(ABL_TUNED_JSON)
    merged = 0
    for key, rec in abl_t.items():
        if key.endswith("__both") or key.endswith("__cyber"):
            abl[key] = rec
            merged += 1
    save_json(ABL_JSON, abl)
    log(f"merged {merged} tuned ablation entries into {os.path.relpath(ABL_JSON, ROOT)}")

    d = dict(np.load(PROBAS))
    d_t = dict(np.load(PROBAS_TUNED))
    pmerged = 0
    for key, proba in d_t.items():
        if key.endswith("__both") or key.endswith("__cyber"):
            d[key] = proba
            pmerged += 1
    np.savez(PROBAS, **d)
    log(f"merged {pmerged} tuned probability arrays into {os.path.relpath(PROBAS, ROOT)}")

    for m in ("both", "cyber"):
        src = os.path.join(TA.MODELS_DIR, "component_modality_ablation",
                           f"mstc_ids_full__{m}_best.keras")
        dst = os.path.join(TA.MODELS_DIR, "proposed", f"mstc_ids_{m}_best.keras")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log(f"copied checkpoint -> {os.path.relpath(dst, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cw", type=float,
                    default=(float(os.environ["MSTC_FUSED_CW"])
                             if os.environ.get("MSTC_FUSED_CW") else None),
                    help="pin the fused contrastive weight (default: best positive by validation)")
    args = ap.parse_args()
    if args.select:
        phase_select(args.cw)
    elif args.merge:
        phase_merge()
    else:
        ap.error("pass --select or --merge")


if __name__ == "__main__":
    main()
