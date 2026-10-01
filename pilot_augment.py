"""Pilot experiment: does the engineered replay-signature feature set help the
fused MSTC-IDS flagship on DoS-vs-Replay separation?

Trains the SAME fused "full__both" model twice:
  * base : current cached tensors (46 cyber features) from data/
  * aug  : augmented tensors (+seq_repeat / seq_inc / content_repeat flags,
           49 cyber features) from data_aug/

Both runs use the persisted tuned mstc_both config, the same 70/10 train/val
splits, identical seeds, and are evaluated on the VALIDATION split only
(test split is never touched during the pilot).
"""

import os
import sys
import json
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MSTC_EPOCHS"] = "45"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import train_all as TA
from pipeline import build_pipeline, CLASS_NAMES


def run(which, cache_dir, out_path):
    data, _, _, meta = build_pipeline(TA.DATA_DIR, cache_dir)
    train = {k: data[k] for k in ("Xc_train", "Xp_train", "Xt_train", "Xf_train", "y_train")}
    val = {k: data[k] for k in ("Xc_val", "Xp_val", "Xt_val", "Xf_val", "y_val")}
    pilot = {**train, **{k.replace("_val", "_test"): v for k, v in val.items()},
             **val}  # test slots = val tensors -> evaluate on val only

    ytr = data["y_train"]
    cw = TA.calculate_class_weights(ytr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.time()
    metrics, _ = TA.train_mstc_variant(
        "full", "both", {}, 0.1, pilot, cw,
        out_path, out_path.replace(".json", "_probas.npz"))
    metrics["cache"] = cache_dir
    metrics["n_cyber"] = int(meta["n_cyber_features"])
    metrics["split"] = "val"
    metrics["seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=list)
    print(f"[pilot:{which}] acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f} "
          f"seconds={metrics['seconds']}", flush=True)


if __name__ == "__main__":
    import shutil
    TA.MODELS_DIR = os.path.join(TA.RESULTS_DIR, "pilot_models")
    TA.HIST_DIR = os.path.join(TA.RESULTS_DIR, "pilot", "histories")
    TA.make_dirs()

    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        run("base", "data", os.path.join(TA.RESULTS_DIR, "pilot", "base_val.json"))
    elif which == "aug":
        run("aug", "data_aug", os.path.join(TA.RESULTS_DIR, "pilot", "aug_val.json"))
    else:
        raise SystemExit(f"unknown: {which}")
