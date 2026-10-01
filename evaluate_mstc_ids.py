import tensorflow as tf
import numpy as np


from mstc_ids_model import MSTCIDS
from uncertainty import (
    mc_dropout_predict,
    evaluate_predictions,
    replay_analysis
)



MODEL_PATH="mstc_ids_best.keras"



CLASS_NAMES=[

    "Benign",
    "DoS",
    "Replay",
    "EvilTwin",
    "FDI"

]



# ============================================================
# Load model
# ============================================================


model=tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)



# ============================================================
# Test data
# Loaded from Part 3 preprocessing
# ============================================================


Xc_test=np.load(
    "Xc_test.npy"
)

Xp_test=np.load(
    "Xp_test.npy"
)

Xt_test=np.load(
    "Xt_test.npy"
)

Xf_test=np.load(
    "Xf_test.npy"
)

y_test=np.load(
    "y_test.npy"
)



inputs=(

    Xc_test,

    Xp_test,

    Xt_test,

    Xf_test

)



# ============================================================
# MC-Dropout inference
# ============================================================


mean_pred, uncertainty, all_pred = mc_dropout_predict(

    model,

    inputs,

    n_samples=50

)



confidence=np.max(

    mean_pred,

    axis=1

)



print(
    "Average confidence:",
    np.mean(confidence)
)


print(
    "Average uncertainty:",
    np.mean(uncertainty)
)



# ============================================================
# Metrics
# ============================================================


results=evaluate_predictions(

    y_test,

    mean_pred,

    CLASS_NAMES

)



# ============================================================
# Replay analysis
# ============================================================


replay_analysis(

    y_test,

    mean_pred,

    uncertainty,

    replay_label=2

)