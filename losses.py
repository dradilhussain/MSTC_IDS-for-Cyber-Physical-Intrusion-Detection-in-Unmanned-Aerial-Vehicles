"""
MSTC-IDS v2 Loss Functions

Loss components:

1. Classification loss
   - Sparse categorical cross entropy

2. Cyber-Physical Contrastive Alignment
   - Symmetric InfoNCE loss

3. Combined MSTC-IDS objective

        L = L_cls + lambda * L_contrastive

"""

import tensorflow as tf

# See note in mstc_ids_model.py: import tf_keras directly rather than via
# tensorflow's `tf.keras` lazy-loading proxy.
import tf_keras as keras



# ============================================================
# Classification Loss
# ============================================================


def classification_loss(
        y_true,
        y_pred,
        sample_weight=None
):

    """
    Standard intrusion classification loss.

    Supports:
        Binary extension
        Multi-class IDS

    """

    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        reduction="none"
    )


    loss = loss_fn(
        y_true,
        y_pred
    )


    if sample_weight is not None:

        sample_weight = tf.cast(sample_weight, loss.dtype)

        loss *= sample_weight


    return tf.reduce_mean(loss)




# ============================================================
# InfoNCE Contrastive Loss
# ============================================================


def contrastive_alignment_loss(
        cyber_embedding,
        physical_embedding,
        temperature=0.1
):

    """
    Cyber-Physical alignment using InfoNCE.

    Positive pairs:

        cyber_i <----> physical_i


    Negative pairs:

        cyber_i <----> physical_j


    i != j


    """



    # Normalize embeddings

    cyber_embedding=tf.math.l2_normalize(
        cyber_embedding,
        axis=-1
    )


    physical_embedding=tf.math.l2_normalize(
        physical_embedding,
        axis=-1
    )



    # Similarity matrix

    logits=tf.matmul(
        cyber_embedding,
        physical_embedding,
        transpose_b=True
    )


    logits/=temperature



    batch_size=tf.shape(logits)[0]


    labels=tf.range(
        batch_size
    )



    # Cyber -> Physical

    loss_c2p=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )



    # Physical -> Cyber

    loss_p2c=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=tf.transpose(logits)
    )



    return (
        tf.reduce_mean(loss_c2p)
        +
        tf.reduce_mean(loss_p2c)
    ) / 2.0





# ============================================================
# Total MSTC-IDS Loss
# ============================================================


def mstc_ids_loss(
        y_true,
        prediction,
        cyber_embedding,
        physical_embedding,
        contrastive_weight=0.1,
        sample_weight=None
):


    """
    Complete MSTC-IDS objective:

        Loss =
        Classification Loss
        +
        Contrastive Alignment Loss


    """



    cls_loss = classification_loss(
        y_true,
        prediction,
        sample_weight
    )



    con_loss = contrastive_alignment_loss(
        cyber_embedding,
        physical_embedding
    )



    total_loss = (
        cls_loss
        +
        contrastive_weight * con_loss
    )



    return (
        total_loss,
        cls_loss,
        con_loss
    )





# ============================================================
# Custom MSTC-IDS Trainer
# ============================================================


class MSTCTrainer(keras.Model):


    """
    Wrapper model for custom training.

    It enables:

        forward pass
        classification loss
        InfoNCE loss
        gradient update


    Usage:

        trainer = MSTCTrainer(model)

        trainer.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3)
        )

    """



    def __init__(
            self,
            base_model,
            contrastive_weight=0.1
    ):

        super().__init__()


        self.base_model=base_model

        self.contrastive_weight=contrastive_weight



        self.loss_tracker=keras.metrics.Mean(
            name="loss"
        )


        self.cls_tracker=keras.metrics.Mean(
            name="classification_loss"
        )


        self.con_tracker=keras.metrics.Mean(
            name="contrastive_loss"
        )


        self.acc_tracker=keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )




    def call(
            self,
            inputs,
            training=False
    ):


        return self.base_model(
            inputs,
            training=training
        )




    def train_step(
            self,
            data
    ):

        # Keras appends sample_weight to the batch tuple whenever
        # `class_weight` is passed to .fit(), turning data into
        # (x, y, sample_weight) instead of (x, y). Handle both.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None



        with tf.GradientTape() as tape:


            prediction,cyber,physical = self.base_model(
                x,
                training=True,
                return_embeddings=True
            )



            total,cls,con=mstc_ids_loss(

                y,

                prediction,

                cyber,

                physical,

                self.contrastive_weight,

                sample_weight=sample_weight

            )



        gradients=tape.gradient(
            total,
            self.base_model.trainable_variables
        )


        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.base_model.trainable_variables
            )
        )



        self.loss_tracker.update_state(
            total
        )


        self.cls_tracker.update_state(
            cls
        )


        self.con_tracker.update_state(
            con
        )


        self.acc_tracker.update_state(
            y,
            prediction
        )



        return {

            "loss":
                self.loss_tracker.result(),

            "classification_loss":
                self.cls_tracker.result(),

            "contrastive_loss":
                self.con_tracker.result(),

            "accuracy":
                self.acc_tracker.result()

        }



    def test_step(
            self,
            data
    ):

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None



        prediction,cyber,physical = self.base_model(
            x,
            training=False,
            return_embeddings=True
        )



        total,cls,con=mstc_ids_loss(

            y,

            prediction,

            cyber,

            physical,

            self.contrastive_weight,

            sample_weight=sample_weight

        )



        self.loss_tracker.update_state(
            total
        )


        self.cls_tracker.update_state(
            cls
        )


        self.con_tracker.update_state(
            con
        )


        self.acc_tracker.update_state(
            y,
            prediction
        )



        return {

            "loss":
                self.loss_tracker.result(),

            "classification_loss":
                self.cls_tracker.result(),

            "contrastive_loss":
                self.con_tracker.result(),

            "accuracy":
                self.acc_tracker.result()

        }



    @property
    def metrics(self):

        return [

            self.loss_tracker,

            self.cls_tracker,

            self.con_tracker,

            self.acc_tracker

        ]