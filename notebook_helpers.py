"""
Helpers used by the notebooks to reuse the results produced by the consolidated
full run (train_all.py + compile_results.py).

The notebooks keep their original training code intact, but before training any
model they check whether the corresponding checkpoint + recorded metrics already
exist under results/ (i.e. the full EPOCHS=50 run). If so they load the saved
predictions/confusion matrices/training histories instead of retraining, so
executing a notebook reproduces exactly the same tables and figures in a few
minutes rather than re-running hours of training.
"""

import os
import json
import numpy as np

ROOT = os.getcwd()


def precomputed_result(rel_json, key):
    p = os.path.join(ROOT, rel_json)
    if os.path.exists(p):
        with open(p) as f:
            recs = json.load(f)
        if key in recs:
            return recs[key]
    return None


def precomputed_proba(rel_npz, key):
    p = os.path.join(ROOT, rel_npz)
    if os.path.exists(p):
        with np.load(p) as d:
            if key in d:
                return d[key]
    return None


def precomputed_history(rel_json):
    p = os.path.join(ROOT, rel_json)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def get_or_train(rel_json, rel_npz, hist_dir, key, train_fn, *args, **kwargs):
    """(metrics, cm, hist_dict, proba) — reuse precomputed results when available."""
    metrics = precomputed_result(rel_json, key)
    proba = precomputed_proba(rel_npz, key)
    if metrics is not None and proba is not None:
        cm = np.array(metrics["confusion_matrix"])
        hist = precomputed_history(os.path.join(hist_dir, f"{key}.json"))
        if hist is None:
            hist = {}
        print(f"[notebook] reusing precomputed results for {key}")
        return metrics, cm, hist, proba
    return train_fn(*args, **kwargs)


def get_or_train_baseline(rel_json, rel_npz, key, train_fn, *args, **kwargs):
    """(metrics, cm, proba) — reuse precomputed results when available."""
    metrics = precomputed_result(rel_json, key)
    proba = precomputed_proba(rel_npz, key)
    if metrics is not None and proba is not None:
        cm = np.array(metrics["confusion_matrix"])
        print(f"[notebook] reusing precomputed results for {key}")
        return metrics, cm, proba
    return train_fn(*args, **kwargs)
