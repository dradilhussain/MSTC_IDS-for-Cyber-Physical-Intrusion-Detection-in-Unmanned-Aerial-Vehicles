"""
MSTC-IDS v2 Training Pipeline

Cyber-Physical Intrusion Detection System

Dataset:
CPS Cyber + Physical Attack Dataset

Classes:
0 - Benign
1 - DoS
2 - Replay
3 - EvilTwin
4 - FDI


Implemented:
- Cyber-Physical synchronization
- Irregular time encoding support
- Segment-wise temporal split
- Multi-class IDS
- Contrastive cyber-physical learning

"""


# ==========================================================
# Imports
# ==========================================================

import os
import json
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.utils.class_weight import (
    compute_class_weight
)


from mstc_ids_model import MSTCIDS
from losses import MSTCTrainer



# ==========================================================
# Configuration
# ==========================================================


DATA_DIR = "segment_data"


WINDOW_SIZE = 20


BATCH_SIZE = 128


EPOCHS = 50


LEARNING_RATE = 1e-4


NUM_CLASSES = 5



LABEL_MAP = {

    "benign":0,

    "dos":1,

    "replay":2,

    "eviltwin":3,

    "fdi":4

}



CLASS_NAMES = [

    "Benign",

    "DoS",

    "Replay",

    "EvilTwin",

    "FDI"

]



# ==========================================================
# Timestamp Detection
# ==========================================================


def fix_timestamp_column(df):
    """
    Convert different timestamp names
    into unified 'timestamp'
    """


    timestamp_candidates = [

        "timestamp",

        "timestamp_c",

        "timestamp_p",

        "frame.time_epoch",

        "time"

    ]


    found = None


    for c in timestamp_candidates:


        if c in df.columns:

            found = c

            break



    if found is None:


        raise ValueError(

            "No timestamp column found\n"

            f"Available columns:\n{df.columns.tolist()}"

        )



    if found != "timestamp":


        df = df.rename(

            columns={

                found:"timestamp"

            }

        )



    return df





# ==========================================================
# Load Individual Segment
# ==========================================================


def load_segment(path, attack, domain):

    """
    Load one cyber or physical segment.

    Handles:
    - Cyber timestamps: timestamp_c
    - Physical timestamps: timestamp_p
    - Missing physical timestamps (FDI case)
    - Label leakage removal
    """

    df = pd.read_csv(path)


    # ==================================================
    # Normalize column names
    # ==================================================

    df.columns = [
        str(c).lower().strip()
        for c in df.columns
    ]


    print(
        f"\nLoading: {path}"
    )

    print(
        "Original shape:",
        df.shape
    )


    # ==================================================
    # Timestamp detection
    # ==================================================

    timestamp_candidates = [

        "timestamp",
        "timestamp_c",
        "timestamp_p",
        "frame.time_epoch",
        "time"

    ]


    timestamp_col = None


    for col in timestamp_candidates:

        if col in df.columns:

            timestamp_col = col

            break



    # ==================================================
    # If timestamp exists
    # ==================================================

    if timestamp_col is not None:


        if timestamp_col != "timestamp":

            df.rename(
                columns={
                    timestamp_col:"timestamp"
                },
                inplace=True
            )


    # ==================================================
    # Missing timestamp handling
    # ==================================================

    else:


        if domain == "physical":


            print(
                "No timestamp found."
            )

            print(
                "Generating synthetic physical timestamp..."
            )


            # UAV/physical sensors
            # assume 100 Hz sampling

            sampling_interval = 0.01


            df["timestamp"] = (

                np.arange(
                    len(df)
                )
                *
                sampling_interval

            )


        else:


            raise ValueError(

                f"""
                No timestamp found in cyber data:

                {path}

                Columns:
                {df.columns.tolist()}
                """

            )


    # ==================================================
    # Timestamp conversion
    # ==================================================

    df["timestamp"] = pd.to_numeric(

        df["timestamp"],

        errors="coerce"

    )


    # remove missing timestamps

    df = df.dropna(

        subset=[
            "timestamp"
        ]

    )


    # sort chronologically

    df = df.sort_values(

        "timestamp"

    ).reset_index(

        drop=True

    )


    # ==================================================
    # Remove label leakage
    # ==================================================

    leakage_columns = [

        "class",
        "label"

    ]


    for col in leakage_columns:

        if col in df.columns:

            df.drop(

                columns=[col],

                inplace=True

            )


    # ==================================================
    # Add metadata
    # ==================================================

    df["attack"] = attack

    df["domain"] = domain



    print(
        "Final shape:",
        df.shape
    )


    return df

# ==========================================================
# Remove Constant Features
# ==========================================================


def remove_constant_features(df):
    """
    Remove features with only one unique value.
    Keep timestamp and metadata.
    """


    drop_cols = []


    protected = [

        "timestamp",

        "attack",

        "domain"

    ]



    for c in df.columns:


        if c in protected:

            continue



        if df[c].nunique(dropna=False) <= 1:


            drop_cols.append(c)



    if len(drop_cols)>0:

        print(
            "Removing constants:",
            drop_cols
        )



    return df.drop(

        columns=drop_cols,

        errors="ignore"

    )





# ==========================================================
# Detect Numeric and Categorical Features
# ==========================================================


def identify_features(df):
    """
    Separate numerical and categorical columns
    """


    ignore = [

        "timestamp",

        "attack",

        "domain"

    ]



    feature_cols=[

        c for c in df.columns

        if c not in ignore

    ]



    numeric_cols=[]

    categorical_cols=[]



    for c in feature_cols:


        if pd.api.types.is_numeric_dtype(
            df[c]
        ):

            numeric_cols.append(c)


        else:

            categorical_cols.append(c)



    return (

        numeric_cols,

        categorical_cols

    )





# ==========================================================
# Encode Categorical Features
# ==========================================================


def encode_categorical_features(
        train_df,
        val_df=None,
        test_df=None
):
    """
    Label encode categorical variables.

    Encoder is fitted only on training data.
    """


    encoders={}



    numeric_cols, categorical_cols = identify_features(
        train_df
    )



    for col in categorical_cols:


        encoder = LabelEncoder()



        # training fit

        train_df[col]=encoder.fit_transform(

            train_df[col]
            .astype(str)

        )



        encoders[col]=encoder



        # validation

        if val_df is not None:


            val_df[col]=val_df[col].astype(str)


            val_df[col]=val_df[col].map(

                lambda x:
                encoder.transform([x])[0]
                if x in encoder.classes_
                else -1

            )



        # test

        if test_df is not None:


            test_df[col]=test_df[col].astype(str)


            test_df[col]=test_df[col].map(

                lambda x:
                encoder.transform([x])[0]
                if x in encoder.classes_
                else -1

            )



    return (

        train_df,

        val_df,

        test_df,

        encoders

    )





# ==========================================================
# Synchronize Cyber and Physical Streams
# ==========================================================


def synchronize_streams(
        cyber,
        physical
):


    """
    Synchronize cyber packets with physical measurements
    using nearest timestamp matching.
    """


    cyber = cyber.sort_values(
        "timestamp"
    ).copy()


    physical = physical.sort_values(
        "timestamp"
    ).copy()


    # ==========================================
    # Make timestamps compatible
    # ==========================================

    cyber["timestamp"] = pd.to_numeric(
        cyber["timestamp"],
        errors="coerce"
    )


    physical["timestamp"] = pd.to_numeric(
        physical["timestamp"],
        errors="coerce"
    )


    # remove invalid timestamps

    cyber = cyber.dropna(
        subset=["timestamp"]
    )


    physical = physical.dropna(
        subset=["timestamp"]
    )


    # convert both to float

    cyber["timestamp"] = (
        cyber["timestamp"]
        .astype("float64")
    )


    physical["timestamp"] = (
        physical["timestamp"]
        .astype("float64")
    )


    # ==========================================
    # Synchronization
    # ==========================================


    fused = pd.merge_asof(

        cyber,

        physical,

        on="timestamp",

        direction="nearest",

        suffixes=(

            "_cyber",

            "_physical"

        )

    )


    return fused





# ==========================================================
# Load One Attack Segment Pair
# ==========================================================


def load_attack_segment(
        attack
):


    cyber_path=os.path.join(

        DATA_DIR,

        f"{attack}_cyber.csv"

    )


    physical_path=os.path.join(

        DATA_DIR,

        f"{attack}_physical.csv"

    )



    cyber=load_segment(

        cyber_path,

        attack,

        "cyber"

    )



    physical=load_segment(

        physical_path,

        attack,

        "physical"

    )



    # remove constants

    cyber=remove_constant_features(
        cyber
    )


    physical=remove_constant_features(
        physical
    )



    # synchronize

    fused=synchronize_streams(

        cyber,

        physical

    )



    # assign numerical label

    fused["label"]=LABEL_MAP[attack]



    return fused





# ==========================================================
# Load Complete Dataset
# ==========================================================


def load_all_segments():



    attacks=[

        "benign",

        "dos",

        "replay",

        "eviltwin",

        "fdi"

    ]



    segments=[]



    for attack in attacks:


        print(
            "\n===================="
        )

        print(
            "Processing:",
            attack
        )


        fused=load_attack_segment(
            attack
        )


        print(
            "Shape:",
            fused.shape
        )


        segments.append(
            fused
        )



    return segments

# ==========================================================
# Global Feature Alignment
# ==========================================================


def align_feature_dimensions(
        segments
):
    """
    Creates common cyber and physical
    feature spaces across all attacks.

    Missing features are filled with zeros.
    """


    cyber_features=set()

    physical_features=set()



    # collect all features

    for df in segments:


        for c in df.columns:


            if c.endswith("_cyber"):

                cyber_features.add(c)



            elif c.endswith("_physical"):

                physical_features.add(c)



    cyber_features=sorted(
        list(cyber_features)
    )


    physical_features=sorted(
        list(physical_features)
    )



    print(
        "\nGlobal cyber features:",
        len(cyber_features)
    )


    print(
        "Global physical features:",
        len(physical_features)
    )



    aligned=[]



    for df in segments:


        # add missing columns

        for c in cyber_features:


            if c not in df.columns:

                df[c]=0



        for c in physical_features:


            if c not in df.columns:

                df[c]=0



        # keep only ordered features

        keep = [

            "timestamp"

        ] + cyber_features + physical_features + [

            "attack",

            "domain",

            "label"

        ]



        df=df[keep]



        aligned.append(df)



    return (

        aligned,

        cyber_features,

        physical_features

    )





# ==========================================================
# Convert to Numeric Arrays
# ==========================================================


def prepare_numeric_features(
        df,
        cyber_features,
        physical_features
):


    Xc=df[cyber_features].values.astype(
        np.float32
    )


    Xp=df[physical_features].values.astype(
        np.float32
    )


    timestamp=df["timestamp"].values


    y=df["label"].values



    return (

        Xc,

        Xp,

        timestamp,

        y

    )





# ==========================================================
# Sliding Window Generator
# ==========================================================


def create_windows(
        df,
        cyber_features,
        physical_features
):


    Xc=[]

    Xp=[]

    Xt=[]

    Xf=[]

    y=[]



    cyber, physical, timestamp, labels = prepare_numeric_features(

        df,

        cyber_features,

        physical_features

    )



    n=len(df)



    for i in range(

        n-WINDOW_SIZE

    ):



        # -----------------------
        # Cyber window
        # -----------------------

        Xc.append(

            cyber[
                i:i+WINDOW_SIZE
            ]

        )



        # -----------------------
        # Physical window
        # -----------------------

        Xp.append(

            physical[
                i:i+WINDOW_SIZE
            ]

        )



        # -----------------------
        # Irregular time interval
        # for Time2Vec
        # -----------------------


        delta=np.diff(

            timestamp[
                i:i+WINDOW_SIZE
            ]

        )



        delta=np.insert(

            delta,

            0,

            0

        )



        Xt.append(

            delta.reshape(
                WINDOW_SIZE,
                1
            )

        )



        # -----------------------
        # Packet freshness
        # -----------------------

        Xf.append(

            delta.reshape(
                WINDOW_SIZE,
                1
            )

        )



        # label of last packet

        y.append(

            labels[
                i+WINDOW_SIZE-1
            ]

        )



    return (

        np.asarray(
            Xc,
            dtype=np.float32
        ),


        np.asarray(
            Xp,
            dtype=np.float32
        ),


        np.asarray(
            Xt,
            dtype=np.float32
        ),


        np.asarray(
            Xf,
            dtype=np.float32
        ),


        np.asarray(
            y,
            dtype=np.int64
        )

    )





# ==========================================================
# Segment-wise Temporal Split
# ==========================================================


def temporal_split(
        windows
):


    Xc,Xp,Xt,Xf,y=windows



    n=len(y)



    train_end=int(
        n*0.70
    )


    val_end=int(
        n*0.85
    )



    train=(

        Xc[:train_end],

        Xp[:train_end],

        Xt[:train_end],

        Xf[:train_end],

        y[:train_end]

    )



    val=(

        Xc[train_end:val_end],

        Xp[train_end:val_end],

        Xt[train_end:val_end],

        Xf[train_end:val_end],

        y[train_end:val_end]

    )



    test=(

        Xc[val_end:],

        Xp[val_end:],

        Xt[val_end:],

        Xf[val_end:],

        y[val_end:]

    )



    return (

        train,

        val,

        test

    )





# ==========================================================
# Scaling
# ==========================================================


def scale_windows(
        train,
        val,
        test
):


    Xc_train,Xp_train,Xt_train,Xf_train,y_train=train


    Xc_val,Xp_val,Xt_val,Xf_val,y_val=val


    Xc_test,Xp_test,Xt_test,Xf_test,y_test=test



    scaler_c=StandardScaler()

    scaler_p=StandardScaler()



    # ----------------------
    # Cyber scaling
    # ----------------------

    shape=Xc_train.shape


    Xc_train=scaler_c.fit_transform(

        Xc_train.reshape(
            -1,
            shape[-1]
        )

    ).reshape(shape)



    Xc_val=scaler_c.transform(

        Xc_val.reshape(
            -1,
            Xc_val.shape[-1]
        )

    ).reshape(
        Xc_val.shape
    )



    Xc_test=scaler_c.transform(

        Xc_test.reshape(
            -1,
            Xc_test.shape[-1]
        )

    ).reshape(
        Xc_test.shape
    )




    # ----------------------
    # Physical scaling
    # ----------------------


    shape=Xp_train.shape


    Xp_train=scaler_p.fit_transform(

        Xp_train.reshape(
            -1,
            shape[-1]
        )

    ).reshape(shape)



    Xp_val=scaler_p.transform(

        Xp_val.reshape(
            -1,
            Xp_val.shape[-1]
        )

    ).reshape(
        Xp_val.shape
    )



    Xp_test=scaler_p.transform(

        Xp_test.reshape(
            -1,
            Xp_test.shape[-1]
        )

    ).reshape(
        Xp_test.shape
    )



    joblib.dump(
        scaler_c,
        "cyber_scaler.pkl"
    )


    joblib.dump(
        scaler_p,
        "physical_scaler.pkl"
    )



    return (

        (
        Xc_train,
        Xp_train,
        Xt_train,
        Xf_train,
        y_train
        ),


        (
        Xc_val,
        Xp_val,
        Xt_val,
        Xf_val,
        y_val
        ),


        (
        Xc_test,
        Xp_test,
        Xt_test,
        Xf_test,
        y_test
        )

    )

# ==========================================================
# Combine Windows From All Attacks
# ==========================================================


def combine_splits(
        split_list
):
    """
    Combine train/val/test
    from all attack segments
    """


    combined=[]


    for index in range(5):

        combined.append(

            np.concatenate(

                [
                    x[index]
                    for x in split_list
                ],

                axis=0

            )

        )


    return tuple(combined)





# ==========================================================
# Create TensorFlow Dataset
# ==========================================================


def create_tf_dataset(
        data,
        shuffle=False
):


    Xc,Xp,Xt,Xf,y=data



    dataset=tf.data.Dataset.from_tensor_slices(

        (

            (

                Xc,

                Xp,

                Xt,

                Xf

            ),

            y

        )

    )


    if shuffle:


        dataset=dataset.shuffle(

            buffer_size=len(y)

        )



    dataset=dataset.batch(

        BATCH_SIZE

    )


    dataset=dataset.prefetch(

        tf.data.AUTOTUNE

    )


    return dataset





# ==========================================================
# Calculate Class Weights
# ==========================================================


def calculate_class_weights(
        y
):


    classes=np.unique(y)



    weights=compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y

    )



    class_weights=dict(

        zip(

            classes,

            weights

        )

    )


    return class_weights





# ==========================================================
# Save Dataset
# ==========================================================


def save_processed_data(
        train,
        val,
        test
):


    os.makedirs(

        "processed_data",

        exist_ok=True

    )



    names=[

        "train",

        "val",

        "test"

    ]



    for name,data in zip(

        names,

        [

            train,

            val,

            test

        ]

    ):


        Xc,Xp,Xt,Xf,y=data



        np.save(

            f"processed_data/{name}_cyber.npy",

            Xc

        )


        np.save(

            f"processed_data/{name}_physical.npy",

            Xp

        )


        np.save(

            f"processed_data/{name}_time.npy",

            Xt

        )


        np.save(

            f"processed_data/{name}_freshness.npy",

            Xf

        )


        np.save(

            f"processed_data/{name}_labels.npy",

            y

        )



    print(
        "Processed data saved"
    )





# ==========================================================
# Training Pipeline
# ==========================================================


def train_mstc_ids():



    print("\nLoading dataset...\n")



    # --------------------------------
    # 1. Load segments
    # --------------------------------


    segments=load_all_segments()



    # --------------------------------
    # 2. Align features
    # --------------------------------


    segments, cyber_features, physical_features = (

        align_feature_dimensions(

            segments

        )

    )



    print(

        "\nCyber dimension:",

        len(cyber_features)

    )


    print(

        "Physical dimension:",

        len(physical_features)

    )



    train_parts=[]

    val_parts=[]

    test_parts=[]



    # --------------------------------
    # 3. Window each attack segment
    # --------------------------------


    for segment in segments:


        attack=segment["attack"].iloc[0]


        print(

            "\nCreating windows:",

            attack

        )



        windows=create_windows(

            segment,

            cyber_features,

            physical_features

        )



        train,val,test=temporal_split(

            windows

        )



        train_parts.append(train)

        val_parts.append(val)

        test_parts.append(test)



    # --------------------------------
    # 4. Combine attacks
    # --------------------------------


    train=combine_splits(

        train_parts

    )


    val=combine_splits(

        val_parts

    )


    test=combine_splits(

        test_parts

    )



    print("\nBefore scaling")

    print(

        "Train:",

        train[0].shape

    )

    print(

        "Validation:",

        val[0].shape

    )

    print(

        "Test:",

        test[0].shape

    )



    # --------------------------------
    # 5. Normalize
    # --------------------------------


    train,val,test=scale_windows(

        train,

        val,

        test

    )



    save_processed_data(

        train,

        val,

        test

    )



    Xc,Xp,Xt,Xf,y=train



    # --------------------------------
    # 6. Class weights
    # --------------------------------


    class_weights=calculate_class_weights(

        y

    )



    print(

        "\nClass weights:"

    )


    for k,v in class_weights.items():

        print(

            CLASS_NAMES[k],

            ":",

            round(v,3)

        )



    # --------------------------------
    # 7. Create datasets
    # --------------------------------


    train_ds=create_tf_dataset(

        train,

        shuffle=True

    )


    val_ds=create_tf_dataset(

        val

    )



    test_ds=create_tf_dataset(

        test

    )



    # --------------------------------
    # 8. Build MSTC-IDS
    # --------------------------------


    print(

        "\nBuilding MSTC-IDS..."

    )


    model=MSTCIDS(

        n_cyber=Xc.shape[-1],

        n_phys=Xp.shape[-1],

        classes=NUM_CLASSES

    )



    model.summary()



    # --------------------------------
    # 9. Trainer
    # --------------------------------


    trainer=MSTCTrainer(

        model,

        contrastive_weight=0.1

    )



    trainer.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

        )

    )



    # --------------------------------
    # 10. Callbacks
    # --------------------------------


    callbacks=[


        tf.keras.callbacks.ModelCheckpoint(

            filepath="mstc_ids_best.keras",

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        ),



        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True,

            verbose=1

        ),



        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-7,

            verbose=1

        )

    ]



    # --------------------------------
    # 11. Train
    # --------------------------------


    history=trainer.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        class_weight=class_weights,

        callbacks=callbacks

    )



    print(

        "\nTraining completed"

    )



    return (

        model,

        history,

        test_ds,

        test

    )





# ==========================================================
# Main
# ==========================================================


if __name__=="__main__":


    model,history,test_ds,test=train_mstc_ids()