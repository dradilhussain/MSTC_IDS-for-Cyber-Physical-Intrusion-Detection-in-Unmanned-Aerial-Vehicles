"""
MSTC-IDS v2
Multi-Scale Temporal Cross-Attention Intrusion Detection System

Implemented components:

I1  Separate Time2Vec irregular-time encoding
I2  Gated cross-attention + contrastive alignment
I3  Multi-scale temporal attention pyramid
I4  End CLS token pooling
I5  Strict causal self-attention
I6  MC-Dropout uncertainty compatible
"""

import tensorflow as tf

# Import tf_keras directly (bypassing tensorflow's `tf.keras` lazy-loading
# proxy) to avoid a RecursionError some TensorFlow/Anaconda builds hit in
# `tensorflow.python.util.lazy_loader.KerasLazyLoader` even when
# TF_USE_LEGACY_KERAS=1 is set and a matching tf_keras version is installed.
# Requires: pip install tf_keras
import tf_keras as keras
from tf_keras import layers


# ============================================================
# I1: Time2Vec
# ============================================================

class Time2Vec(layers.Layer):

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k


    def build(self, input_shape):

        self.w0 = self.add_weight(
            name="linear_weight",
            shape=(1,1),
            initializer="glorot_uniform"
        )

        self.b0 = self.add_weight(
            name="linear_bias",
            shape=(1,1),
            initializer="zeros"
        )


        self.w = self.add_weight(
            name="periodic_weight",
            shape=(1,self.k),
            initializer="glorot_uniform"
        )

        self.b = self.add_weight(
            name="periodic_bias",
            shape=(1,self.k),
            initializer="zeros"
        )


    def call(self,t):

        linear = self.w0*t+self.b0

        periodic = tf.sin(
            t*self.w+self.b
        )

        return tf.concat(
            [linear,periodic],
            axis=-1
        )



# ============================================================
# Replay freshness embedding
# ============================================================

class FreshnessEmbedding(layers.Layer):

    """
    Encodes packet freshness:

        freshness =
        current_timestamp - original_timestamp

    Useful for replay attack detection.
    """

    def __init__(self,d):

        super().__init__()

        self.encoder = keras.Sequential([
            layers.Dense(d,activation="gelu"),
            layers.Dense(d)
        ])


    def call(self,x):

        return self.encoder(x)



# ============================================================
# I5: Correct causal self attention
# ============================================================


class CausalTransformerBlock(layers.Layer):

    def __init__(
        self,
        d,
        heads,
        ff,
        dropout=0.1,
        causal=True
    ):

        super().__init__()

        self.causal = causal

        self.attention = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=d//heads,
            dropout=dropout
        )


        self.ffn = keras.Sequential([
            layers.Dense(ff,activation="gelu"),
            layers.Dense(d)
        ])


        self.norm1=layers.LayerNormalization()
        self.norm2=layers.LayerNormalization()


        self.drop1=layers.Dropout(dropout)
        self.drop2=layers.Dropout(dropout)



    def causal_mask(self,length):

        i=tf.range(length)[:,None]

        j=tf.range(length)[None,:]

        mask=i>=j

        return mask



    def call(self,x,training=False):

        seq=tf.shape(x)[1]

        mask=self.causal_mask(seq) if self.causal else None


        attn=self.attention(
            x,
            x,
            attention_mask=mask,
            training=training
        )


        x=self.norm1(
            x+self.drop1(
                attn,
                training=training
            )
        )


        ff=self.ffn(x)


        x=self.norm2(
            x+self.drop2(
                ff,
                training=training
            )
        )


        return x



# ============================================================
# I4 CLS token
# ============================================================


class CLSToken(layers.Layer):

    def build(self,input_shape):

        d=input_shape[-1]


        self.cls=self.add_weight(
            name="CLS",
            shape=(1,1,d),
            initializer="zeros",
            trainable=True
        )


    def call(self,x):

        batch=tf.shape(x)[0]


        cls=tf.tile(
            self.cls,
            [batch,1,1]
        )


        # append at END
        return tf.concat(
            [x,cls],
            axis=1
        )



# ============================================================
# I2 Gated Cross Attention
# ============================================================


class GatedCrossAttention(layers.Layer):


    def __init__(
        self,
        d,
        heads,
        dropout=0.1
    ):

        super().__init__()


        self.attn=layers.MultiHeadAttention(
            heads,
            d//heads,
            dropout=dropout
        )


        self.gate=layers.Dense(
            d,
            activation="sigmoid"
        )


        self.norm=layers.LayerNormalization()



    def call(
        self,
        query,
        context,
        training=False
    ):


        cross=self.attn(
            query,
            context,
            training=training
        )


        g=self.gate(
            tf.concat(
                [query,cross],
                axis=-1
            )
        )


        fused=(
            g*cross+
            (1-g)*query
        )


        return self.norm(fused)



# ============================================================
# I3 Multi-scale Temporal Pyramid
# ============================================================


class TemporalPyramid(layers.Layer):


    def __init__(self,scales=(5,10,20),d=64):

        super().__init__()

        self.blocks=[]


        for s in scales:

            self.blocks.append(
                keras.Sequential([
                    layers.Conv1D(
                        d,
                        kernel_size=s,
                        padding="same",
                        activation="gelu"
                    ),

                    layers.GlobalAveragePooling1D()
                ])
            )


    def call(self,x):

        outputs=[]

        for block in self.blocks:

            outputs.append(
                block(x)
            )


        return tf.concat(
            outputs,
            axis=-1
        )



# ============================================================
# Fusion Layer
# ============================================================


class FusionLayer(layers.Layer):


    def __init__(self,d):

        super().__init__()

        self.fc=keras.Sequential([

            layers.Dense(
                d,
                activation="gelu"
            ),

            layers.LayerNormalization()

        ])


    def call(self,x):

        return self.fc(x)



# ============================================================
# Complete MSTC IDS
# ============================================================


@keras.utils.register_keras_serializable(package="mstc_ids")
class MSTCIDS(keras.Model):
    """
    modality: "both" (default, full cyber-physical fusion), "cyber"
    (cyber-only), or "physical" (physical-only).

    Component ablation flags (all default True = full model):
      use_time2vec       — Time2Vec irregular-time encoding vs. a plain
                            linear projection of raw inter-arrival delta.
      use_cross_attention — Gated cross-attention fusion vs. naive
                            concatenation of independently-encoded CLS
                            tokens. Only meaningful when modality="both".
      use_pyramid         — Multi-scale temporal convolution pyramid vs.
                            CLS-token-only representation.
      causal              — Strict causal self-attention vs. full
                            (bidirectional) self-attention.
      use_freshness       — Packet-freshness embedding (replay-detection
                            signal) added to the cyber stream, vs. omitted.

    Note: whether the InfoNCE contrastive alignment loss is applied is a
    *training*-time choice (MSTCTrainer's contrastive_weight), not an
    architectural one, so it has no corresponding flag here — set
    contrastive_weight=0.0 when training to ablate it.

    For every ablation, everything not explicitly disabled is kept
    identical to the full model, so each variant isolates the effect of
    exactly one component.
    """

    def __init__(
        self,
        n_cyber,
        n_phys,
        classes,
        d=64,
        heads=4,
        ff=128,
        dropout=0.2,
        modality="both",
        use_time2vec=True,
        use_cross_attention=True,
        use_pyramid=True,
        causal=True,
        use_freshness=True,
        use_decision_fusion=False,
        modality_dropout=0.0,
        **kwargs
    ):


        super().__init__(**kwargs)

        if modality not in ("both", "cyber", "physical"):
            raise ValueError(
                f"modality must be 'both', 'cyber', or 'physical', got {modality!r}"
            )

        # store constructor args so the model can be serialized/deserialized
        # via get_config()/from_config() (required by Keras 3 for full
        # `.save()`/`load_model()` of subclassed models).
        self.n_cyber = n_cyber
        self.n_phys = n_phys
        self.classes = classes
        self.d = d
        self.heads = heads
        self.ff = ff
        self.dropout = dropout
        self.modality = modality
        self.use_time2vec = use_time2vec
        self.use_cross_attention = use_cross_attention
        self.use_pyramid = use_pyramid
        self.causal = causal
        self.use_freshness = use_freshness
        self.modality_dropout = modality_dropout

        self.use_cyber = modality in ("both", "cyber")
        self.use_phys = modality in ("both", "physical")
        # cross-attention fusion only applies when both branches exist
        self.use_fusion = (modality == "both") and use_cross_attention
        # decision-level (late) fusion: one head per branch + a learned gate
        self.use_decision_fusion = use_decision_fusion and (modality == "both")

        if self.use_cyber:

            self.cyber_proj=layers.Dense(d)

            if use_time2vec:
                self.cyber_time=Time2Vec(8)

            if use_freshness:
                self.freshness=FreshnessEmbedding(d)

            self.cls_c=CLSToken()

            self.encoder_c=CausalTransformerBlock(
                d,heads,ff,dropout,causal=causal
            )

        if self.use_phys:

            self.phys_proj=layers.Dense(d)

            if use_time2vec:
                self.phys_time=Time2Vec(8)

            self.cls_p=CLSToken()

            self.encoder_p=CausalTransformerBlock(
                d,heads,ff,dropout,causal=causal
            )

        if self.use_cyber or self.use_phys:

            if use_time2vec:
                self.time_proj=layers.Dense(d)
            else:
                # plain linear projection of the raw inter-arrival delta,
                # used in place of the sinusoidal Time2Vec encoding
                self.time_proj=layers.Dense(d)

        if self.use_fusion:

            self.cross_c2p=GatedCrossAttention(
                d,heads
            )

            self.cross_p2c=GatedCrossAttention(
                d,heads
            )


        if use_pyramid:

            self.pyramid=TemporalPyramid(
                d=d
            )


        if not self.use_decision_fusion:

            self.fusion=FusionLayer(
                d*8
            )

            self.classifier=keras.Sequential([

                layers.Dense(128,activation="gelu"),

                layers.Dropout(dropout),

                layers.Dense(
                    classes,
                    activation="softmax"
                )

            ])

        else:

            # per-branch classification heads
            self.head_c = keras.Sequential([
                layers.Dense(128, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(classes, activation="softmax"),
            ])
            self.head_p = keras.Sequential([
                layers.Dense(128, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(classes, activation="softmax"),
            ])
            # per-sample gate over the two branch CLS embeddings
            self.fusion_gate = keras.Sequential([
                layers.Dense(d, activation="gelu"),
                layers.Dense(1, activation="sigmoid"),
            ])



    def _time_embed(self, time_layer, time):
        # Either the Time2Vec featurization (periodic + linear) or, when
        # ablated, a direct linear projection of the raw scalar delta.
        if self.use_time2vec:
            return self.time_proj(time_layer(time))
        return self.time_proj(time)


    def call(
        self,
        inputs,
        training=False,
        return_embeddings=False
    ):


        cyber,phys,time,fresh=inputs


        if self.modality == "both":

            tc=self._time_embed(self.cyber_time if self.use_time2vec else None, time)
            tp=self._time_embed(self.phys_time if self.use_time2vec else None, time)

            cyber=self.cyber_proj(cyber)+tc
            if self.use_freshness:
                cyber = cyber + self.freshness(fresh)
            phys=self.phys_proj(phys)+tp

            cyber=self.cls_c(cyber)
            phys=self.cls_p(phys)

            cyber=self.encoder_c(cyber, training=training)
            phys=self.encoder_p(phys, training=training)

            if self.use_fusion:
                cyber_f=self.cross_c2p(cyber, phys, training=training)
                phys_f=self.cross_p2c(phys, cyber, training=training)
            else:
                # naive late fusion: no cross-attention, branches stay independent
                cyber_f=cyber
                phys_f=phys

            # per-sample modality dropout: mask whole branches so the model
            # learns to rely on either modality (never drop both at once)
            if self.modality_dropout > 0.0 and training:
                b = tf.shape(cyber_f)[0]
                keep_c = tf.cast(
                    tf.random.uniform([b, 1, 1]) >= self.modality_dropout,
                    cyber_f.dtype)
                keep_p = tf.cast(
                    tf.random.uniform([b, 1, 1]) >= self.modality_dropout,
                    phys_f.dtype)
                both = (1.0 - keep_c) * (1.0 - keep_p)
                keep_c = keep_c + both
                cyber_f = cyber_f * keep_c
                phys_f = phys_f * keep_p

            cls_c=cyber_f[:,-1,:]
            cls_p=phys_f[:,-1,:]

            emb_a, emb_b = cls_c, cls_p

            if self.use_decision_fusion:
                rep_c = cls_c
                rep_p = cls_p
                if self.use_pyramid:
                    rep_c = tf.concat(
                        [rep_c, self.pyramid(cyber_f[:,:-1,:])], axis=-1)
                    rep_p = tf.concat(
                        [rep_p, self.pyramid(phys_f[:,:-1,:])], axis=-1)
                gate = self.fusion_gate(tf.concat([cls_c, cls_p], axis=-1))
                out_c = self.head_c(rep_c, training=training)
                out_p = self.head_p(rep_p, training=training)
                output = gate * out_p + (1.0 - gate) * out_c
                if return_embeddings:
                    return output, emb_a, emb_b
                return output

            parts = [cls_c, cls_p]
            if self.use_pyramid:
                parts.append(self.pyramid(cyber_f[:,:-1,:]))

            representation=tf.concat(parts, axis=-1)

        elif self.modality == "cyber":

            tc=self._time_embed(self.cyber_time if self.use_time2vec else None, time)

            cyber=self.cyber_proj(cyber)+tc
            if self.use_freshness:
                cyber = cyber + self.freshness(fresh)

            cyber=self.cls_c(cyber)
            cyber=self.encoder_c(cyber, training=training)

            cls_c=cyber[:,-1,:]

            parts = [cls_c]
            if self.use_pyramid:
                parts.append(self.pyramid(cyber[:,:-1,:]))

            representation=tf.concat(parts, axis=-1)

            emb_a, emb_b = cls_c, cls_c

        else:  # "physical"

            tp=self._time_embed(self.phys_time if self.use_time2vec else None, time)

            phys=self.phys_proj(phys)+tp
            phys=self.cls_p(phys)
            phys=self.encoder_p(phys, training=training)

            cls_p=phys[:,-1,:]

            parts = [cls_p]
            if self.use_pyramid:
                parts.append(self.pyramid(phys[:,:-1,:]))

            representation=tf.concat(parts, axis=-1)

            emb_a, emb_b = cls_p, cls_p


        representation=self.fusion(
            representation
        )



        output=self.classifier(
            representation,
            training=training
        )


        if return_embeddings:

            return output, emb_a, emb_b


        return output

    def get_config(self):

        config = super().get_config()

        config.update({
            "n_cyber": self.n_cyber,
            "n_phys": self.n_phys,
            "classes": self.classes,
            "d": self.d,
            "heads": self.heads,
            "ff": self.ff,
            "dropout": self.dropout,
            "modality": self.modality,
            "use_time2vec": self.use_time2vec,
            "use_cross_attention": self.use_cross_attention,
            "use_pyramid": self.use_pyramid,
            "causal": self.causal,
            "use_freshness": self.use_freshness,
            "use_decision_fusion": self.use_decision_fusion,
            "modality_dropout": self.modality_dropout,
        })

        return config


    @classmethod
    def from_config(cls, config):

        return cls(**config)
