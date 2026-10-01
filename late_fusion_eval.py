"""Diagnostic: val-selected late fusion of the trained cyber and physical branches.

Loads the two branch checkpoints, predicts validation + test, and sweeps a
mixing weight w (p = w*p_phys + (1-w)*p_cyber). The weight is chosen on the
validation split and the resulting test metrics are reported. This gives the
ceiling a decision-level fusion could reach without retraining.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import build_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "models", "component_modality_ablation")


def pred(ckpt, data, split):
    m = keras.models.load_model(ckpt, compile=False)
    x = (data["Xc_" + split], data["Xp_" + split],
         data["Xt_" + split], data["Xf_" + split])
    return m(x, training=False).numpy()


def main():
    data, _, _, _ = build_pipeline(os.path.join(ROOT, "segment_data"),
                                   cache_dir=os.path.join(ROOT, "data"))
    p_c = {s: pred(os.path.join(CKPT, f"mstc_ids_full__cyber_best.keras"), data, s)
           for s in ("val", "test")}
    p_p = {s: pred(os.path.join(CKPT, f"mstc_ids_full__physical_best.keras"), data, s)
           for s in ("val", "test")}

    print("branch validation/test accuracy:")
    for s in ("val", "test"):
        print(f"  {s}: cyber={accuracy_score(data['y_'+s], p_c[s].argmax(1)):.4f} "
              f"physical={accuracy_score(data['y_'+s], p_p[s].argmax(1)):.4f}")

    best = None
    rows = []
    for w in np.arange(0.0, 1.0001, 0.05):
        v = w * p_p["val"] + (1 - w) * p_c["val"]
        va = accuracy_score(data["y_val"], v.argmax(1))
        t = w * p_p["test"] + (1 - w) * p_c["test"]
        ta = accuracy_score(data["y_test"], t.argmax(1))
        tf1 = f1_score(data["y_test"], t.argmax(1), average="weighted", zero_division=0)
        rows.append((w, va, ta, tf1))
        if best is None or va > best[1]:
            best = (w, va)
    print("\n  w(phys)  val_acc  test_acc  test_f1")
    for w, va, ta, tf1 in rows:
        print(f"  {w:5.2f}    {va:.4f}   {ta:.4f}    {tf1:.4f}")
    w, va = best
    t = w * p_p["test"] + (1 - w) * p_c["test"]
    ta = accuracy_score(data["y_test"], t.argmax(1))
    tf1 = f1_score(data["y_test"], t.argmax(1), average="weighted", zero_division=0)
    print(f"\nval-selected w(phys)={w:.2f} val_acc={va:.4f}")
    print(f"late-fusion TEST: acc={ta:.4f} f1_weighted={tf1:.4f}")
    tw = max(rows, key=lambda r: r[2])
    print(f"test-best (diagnostic only) w(phys)={tw[0]:.2f} test_acc={tw[2]:.4f} f1={tw[3]:.4f}")


if __name__ == "__main__":
    main()
