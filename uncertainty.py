"""
MSTC-IDS v2
MC-Dropout uncertainty estimation

Implements:

I6:
Monte-Carlo Dropout inference

Outputs:

prediction
confidence
uncertainty
"""


import numpy as np
import tensorflow as tf



# ============================================================
# MC Dropout Prediction
# ============================================================


def mc_dropout_predict(
        model,
        inputs,
        n_samples=50
):

    """
    Perform stochastic forward passes.

    Dropout remains active:

        training=True


    Parameters
    ----------

    model:
        MSTCIDS model


    inputs:
        (
        cyber,
        physical,
        time,
        freshness
        )


    n_samples:
        number of stochastic runs


    Returns
    -------

    mean_prediction

    uncertainty

    all_predictions

    """



    predictions=[]



    for i in range(n_samples):


        pred=model(

            inputs,

            training=True

        )


        predictions.append(

            pred.numpy()

        )



    predictions=np.array(
        predictions
    )



    # Mean prediction

    mean_prediction=np.mean(
        predictions,
        axis=0
    )


    # Uncertainty

    uncertainty=np.std(
        predictions,
        axis=0
    )



    return (

        mean_prediction,

        uncertainty,

        predictions

    )





# ============================================================
# Confidence score
# ============================================================


def prediction_confidence(
        probability
):


    """
    Maximum class probability
    """

    return np.max(
        probability,
        axis=-1
    )





# ============================================================
# Evaluation Metrics
# ============================================================


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report

)





def evaluate_predictions(
        y_true,
        y_probability,
        class_names
):


    """

    Complete IDS evaluation


    """



    y_pred=np.argmax(
        y_probability,
        axis=1
    )



    acc=accuracy_score(
        y_true,
        y_pred
    )


    precision=precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )


    recall=recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )


    f1=f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )



    try:

        auc=roc_auc_score(

            y_true,

            y_probability,

            multi_class="ovr"

        )

    except:

        auc=0





    print("\n==============================")

    print("MSTC-IDS Test Results")

    print("==============================")



    print(
        f"Accuracy : {acc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"AUC      : {auc:.4f}"
    )



    print("\nClassification Report")

    print(

        classification_report(

            y_true,

            y_pred,

            target_names=class_names,

            zero_division=0

        )

    )



    print("\nConfusion Matrix")

    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )



    return {

        "accuracy":acc,

        "precision":precision,

        "recall":recall,

        "f1":f1,

        "auc":auc

    }





# ============================================================
# Replay-specific evaluation
# ============================================================


def replay_analysis(

        y_true,

        predictions,

        uncertainty,

        replay_label=2

):


    """

    Analyze Replay attack behavior.

    Returns:

    - replay recall
    - replay confidence
    - replay uncertainty


    """


    replay_idx=(

        y_true==replay_label

    )



    replay_prob=predictions[

        replay_idx

    ]



    replay_uncertainty=uncertainty[

        replay_idx

    ]



    replay_pred=np.argmax(

        replay_prob,

        axis=1

    )



    recall=np.mean(

        replay_pred==replay_label

    )


    confidence=np.mean(

        np.max(

            replay_prob,

            axis=1

        )

    )



    uncertainty=np.mean(

        np.mean(

            replay_uncertainty,

            axis=1

        )

    )



    print("\n==============================")

    print("Replay Attack Analysis")

    print("==============================")


    print(

        f"Replay Recall: {recall:.4f}"

    )


    print(

        f"Average Confidence: {confidence:.4f}"

    )


    print(

        f"Average Uncertainty: {uncertainty:.4f}"

    )



    return {

        "replay_recall":recall,

        "confidence":confidence,

        "uncertainty":uncertainty

    }