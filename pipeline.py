"""
MSTC-IDS consolidated data pipeline.

Loads the raw cyber/physical segment CSVs, synchronizes the two streams on
nearest timestamp, aligns feature dimensions across attacks, generates sliding
windows (Xc, Xp, Xt, Xf, y), performs a chronological per-segment
train/val/test split, and fits StandardScalers on the train split only.

The preprocessed tensors and scalers are cached under a CACHE_DIR so every
experiment (proposed model, baselines, ablations) consumes byte-identical data.

This module is the single source of truth for the data preprocessing used by all
training drivers in this project. It preserves the exact behavior (and bug
fixes) documented in the original Phase-1 notebook.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

SEED = 42
WINDOW_SIZE = 20
LABEL_MAP = {"benign": 0, "dos": 1, "replay": 2, "eviltwin": 3, "fdi": 4}
CLASS_NAMES = ["Benign", "DoS", "Replay", "EvilTwin", "FDI"]
ATTACKS = ["benign", "dos", "replay", "eviltwin", "fdi"]


def fix_and_load_raw(path, attack, domain):
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    synthetic_ts = False
    timestamp_candidates = ["timestamp", "timestamp_c", "timestamp_p", "frame.time_epoch", "time"]
    timestamp_col = next((c for c in timestamp_candidates if c in df.columns), None)

    if timestamp_col is not None:
        if timestamp_col != "timestamp":
            df = df.rename(columns={timestamp_col: "timestamp"})
    else:
        if domain == "physical":
            # no timestamp present (e.g. FDI physical) -> placeholder index, later
            # rescaled onto the cyber timestamp range so the two streams actually
            # interleave instead of collapsing every cyber row onto one physical row.
            df["timestamp"] = np.arange(len(df), dtype=np.float64)
            synthetic_ts = True
        else:
            raise ValueError(f"No timestamp column found in cyber file: {path}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # drop label-leakage columns; label is reattached explicitly per attack later
    for col in ("class", "label"):
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, synthetic_ts


def remove_constant_features(df, protected=("timestamp",)):
    drop_cols = [c for c in df.columns if c not in protected and df[c].nunique(dropna=False) <= 1]
    if drop_cols:
        print("  removing constant columns:", drop_cols)
    return df.drop(columns=drop_cols, errors="ignore")


def _alignment_quality(cyber_t, physical_t):
    """Median absolute distance from every cyber time to its nearest physical time."""
    idx = np.clip(np.searchsorted(physical_t, cyber_t), 0, len(physical_t) - 1)
    dist = np.abs(physical_t[idx] - cyber_t)
    if idx.max() < len(physical_t):
        dist = np.minimum(dist, np.abs(physical_t[np.maximum(idx - 1, 0)] - cyber_t))
    return float(np.median(dist))


def synchronize_streams(cyber, physical):
    """Fuse the cyber and physical streams on a per-attack timeline.

    The per-attack segment pairs come from separate captures whose clock
    origins are not guaranteed to share an absolute time base (audit found a
    ~5 h constant offset for the evil-twin pair while both streams span an
    identical ~6.26 h duration). Two alignments are therefore evaluated:

      * absolute : merge on the raw timestamps
      * relative : subtract each stream's own start time, then merge

    The mode with the smaller median nearest-physical distance is used, which
    reduces a ~4105 s median misalignment for evil-twin to ~0 s without
    regressing the segments (benign/dos/replay) that already share a clock.
    """
    cyber = cyber.sort_values("timestamp").copy()
    physical = physical.sort_values("timestamp").copy()
    cyber["timestamp"] = pd.to_numeric(cyber["timestamp"], errors="coerce")
    physical["timestamp"] = pd.to_numeric(physical["timestamp"], errors="coerce")
    cyber = cyber.dropna(subset=["timestamp"]).astype({"timestamp": "float64"})
    physical = physical.dropna(subset=["timestamp"]).astype({"timestamp": "float64"})

    abs_cyber_t = cyber["timestamp"].values
    abs_phys_t = physical["timestamp"].values
    rel_cyber_t = abs_cyber_t - abs_cyber_t.min()
    rel_phys_t = abs_phys_t - abs_phys_t.min()

    use_relative = _alignment_quality(rel_cyber_t, rel_phys_t) < _alignment_quality(abs_cyber_t, abs_phys_t)
    print(f"  [sync] alignment mode: {'relative (re-anchored)' if use_relative else 'absolute'}")

    if use_relative:
        cyber = cyber.assign(timestamp=rel_cyber_t)
        physical = physical.assign(timestamp=rel_phys_t)

    # explicit, collision-proof suffixing (fixes the original merge_asof bug)
    cyber = cyber.rename(columns={c: f"{c}_cyber" for c in cyber.columns if c != "timestamp"})
    physical = physical.rename(columns={c: f"{c}_physical" for c in physical.columns if c != "timestamp"})

    fused = pd.merge_asof(cyber, physical, on="timestamp", direction="nearest")
    return fused


def load_attack_segment(data_dir, attack):
    cyber_path = os.path.join(data_dir, f"{attack}_cyber.csv")
    physical_path = os.path.join(data_dir, f"{attack}_physical.csv")

    cyber, _ = fix_and_load_raw(cyber_path, attack, "cyber")
    physical, phys_synthetic = fix_and_load_raw(physical_path, attack, "physical")

    if phys_synthetic:
        # Physical telemetry arrives without timestamps (e.g. FDI), so a time
        # base must be synthesised. Spread the M physical samples proportionally
        # over the *occupied* cyber timeline (packet-rank quantiles) instead of
        # uniformly over [min,max]: for FDI the cyber span covers two ~1-minute
        # attack bursts 9.9 days apart, and a uniform spread places essentially
        # every physical sample in the empty inter-burst gap, collapsing the
        # physical signal to a near-constant for every fused window.
        cyber_t = cyber["timestamp"].values
        n = len(cyber_t)
        m = len(physical)
        ranks = np.linspace(0, n - 1, m).astype(np.int64)
        physical["timestamp"] = cyber_t[np.minimum(ranks, n - 1)]
        print(f"  [{attack}] synthetic physical timestamps mapped rank-proportionally "
              f"onto {n} occupied cyber timestamps")

    cyber = remove_constant_features(cyber)
    physical = remove_constant_features(physical)

    fused = synchronize_streams(cyber, physical)
    fused["attack"] = attack
    fused["label"] = LABEL_MAP[attack]
    return fused


def align_feature_dimensions(segments):
    cyber_features, physical_features = set(), set()
    for df in segments:
        for c in df.columns:
            if c.endswith("_cyber"):
                cyber_features.add(c)
            elif c.endswith("_physical"):
                physical_features.add(c)

    cyber_features = sorted(cyber_features)
    physical_features = sorted(physical_features)

    aligned = []
    for df in segments:
        df = df.copy()
        for c in cyber_features + physical_features:
            if c not in df.columns:
                df[c] = 0.0
        keep = ["timestamp"] + cyber_features + physical_features + ["attack", "label"]
        aligned.append(df[keep])

    return aligned, cyber_features, physical_features


def create_windows(df, cyber_features, physical_features, window_size=WINDOW_SIZE):
    Xc = df[cyber_features].values.astype(np.float32)
    Xp = df[physical_features].values.astype(np.float32)
    timestamp = df["timestamp"].values.astype(np.float64)
    labels = df["label"].values.astype(np.int64)

    Xc_w, Xp_w, Xt_w, Xf_w, y = [], [], [], [], []
    n = len(df)

    for i in range(n - window_size):
        Xc_w.append(Xc[i:i + window_size])
        Xp_w.append(Xp[i:i + window_size])

        delta = np.diff(timestamp[i:i + window_size])
        delta = np.insert(delta, 0, 0)
        delta = np.maximum(delta, 0.0)

        # Xt: log1p inter-arrival delta -> well-conditioned Time2Vec input
        # Xf: log1p cumulative elapsed time within the window -> genuine
        #     freshness / window-temporal-extent signal (distinct from Xt)
        Xt_w.append(np.log1p(delta).reshape(window_size, 1))
        Xf_w.append(np.log1p(np.cumsum(delta)).reshape(window_size, 1))
        y.append(labels[i + window_size - 1])

    return (
        np.asarray(Xc_w, dtype=np.float32),
        np.asarray(Xp_w, dtype=np.float32),
        np.asarray(Xt_w, dtype=np.float32),
        np.asarray(Xf_w, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
    )


def temporal_split(windows, train_frac=0.70, val_frac=0.15):
    Xc, Xp, Xt, Xf, y = windows
    n = len(y)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = tuple(x[:train_end] for x in (Xc, Xp, Xt, Xf, y))
    val = tuple(x[train_end:val_end] for x in (Xc, Xp, Xt, Xf, y))
    test = tuple(x[val_end:] for x in (Xc, Xp, Xt, Xf, y))
    return train, val, test


def combine_splits(split_list):
    return tuple(
        np.concatenate([s[idx] for s in split_list], axis=0)
        for idx in range(5)
    )


def scale_windows(train, val, test):
    Xc_train, Xp_train, Xt_train, Xf_train, y_train = train
    Xc_val, Xp_val, Xt_val, Xf_val, y_val = val
    Xc_test, Xp_test, Xt_test, Xf_test, y_test = test

    scaler_c = StandardScaler()
    scaler_p = StandardScaler()

    shape = Xc_train.shape
    Xc_train = scaler_c.fit_transform(Xc_train.reshape(-1, shape[-1])).reshape(shape)
    Xc_val = scaler_c.transform(Xc_val.reshape(-1, Xc_val.shape[-1])).reshape(Xc_val.shape)
    Xc_test = scaler_c.transform(Xc_test.reshape(-1, Xc_test.shape[-1])).reshape(Xc_test.shape)

    shape = Xp_train.shape
    Xp_train = scaler_p.fit_transform(Xp_train.reshape(-1, shape[-1])).reshape(shape)
    Xp_val = scaler_p.transform(Xp_val.reshape(-1, Xp_val.shape[-1])).reshape(Xp_val.shape)
    Xp_test = scaler_p.transform(Xp_test.reshape(-1, Xp_test.shape[-1])).reshape(Xp_test.shape)

    train_s = (Xc_train, Xp_train, Xt_train, Xf_train, y_train)
    val_s = (Xc_val, Xp_val, Xt_val, Xf_val, y_val)
    test_s = (Xc_test, Xp_test, Xt_test, Xf_test, y_test)
    return train_s, val_s, test_s, scaler_c, scaler_p


def build_pipeline(data_dir, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)

    # cached artifact paths
    keys = ["Xc_train", "Xp_train", "Xt_train", "Xf_train", "y_train",
            "Xc_val", "Xp_val", "Xt_val", "Xf_val", "y_val",
            "Xc_test", "Xp_test", "Xt_test", "Xf_test", "y_test"]
    paths = {k: os.path.join(cache_dir, f"{k}.npy") for k in keys}
    scaler_c_path = os.path.join(cache_dir, "cyber_scaler.pkl")
    scaler_p_path = os.path.join(cache_dir, "physical_scaler.pkl")
    meta_path = os.path.join(cache_dir, "meta.json")

    if all(os.path.exists(p) for p in paths.values()) and os.path.exists(scaler_c_path) \
            and os.path.exists(scaler_p_path) and os.path.exists(meta_path):
        print(f"[pipeline] Loading cached tensors from {cache_dir}/")
        data = {k: np.load(paths[k]) for k in keys}
        scaler_c = joblib.load(scaler_c_path)
        scaler_p = joblib.load(scaler_p_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return data, scaler_c, scaler_p, meta

    print("[pipeline] Building data pipeline from raw CSVs...")

    segments = [load_attack_segment(data_dir, attack) for attack in ATTACKS]
    print("\nPer-attack fused shapes:")
    for seg in segments:
        print(" ", seg["attack"].iloc[0], seg.shape)

    segments, cyber_features, physical_features = align_feature_dimensions(segments)
    print("\nGlobal cyber feature dimension   :", len(cyber_features))
    print("Global physical feature dimension:", len(physical_features))

    windowed_segments = []
    for seg in segments:
        attack = seg["attack"].iloc[0]
        w = create_windows(seg, cyber_features, physical_features)
        print(f"{attack:>9s}: Xc={w[0].shape}  y={w[4].shape}")
        windowed_segments.append(w)

    train_parts, val_parts, test_parts = [], [], []
    for w in windowed_segments:
        tr, va, te = temporal_split(w)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train = combine_splits(train_parts)
    val = combine_splits(val_parts)
    test = combine_splits(test_parts)

    train, val, test, scaler_c, scaler_p = scale_windows(train, val, test)

    Xc_train, Xp_train, Xt_train, Xf_train, y_train = train
    Xc_val, Xp_val, Xt_val, Xf_val, y_val = val
    Xc_test, Xp_test, Xt_test, Xf_test, y_test = test

    data = {
        "Xc_train": Xc_train, "Xp_train": Xp_train, "Xt_train": Xt_train,
        "Xf_train": Xf_train, "y_train": y_train,
        "Xc_val": Xc_val, "Xp_val": Xp_val, "Xt_val": Xt_val,
        "Xf_val": Xf_val, "y_val": y_val,
        "Xc_test": Xc_test, "Xp_test": Xp_test, "Xt_test": Xt_test,
        "Xf_test": Xf_test, "y_test": y_test,
    }

    for k in keys:
        np.save(paths[k], data[k])
    joblib.dump(scaler_c, scaler_c_path)
    joblib.dump(scaler_p, scaler_p_path)

    meta = {
        "window_size": WINDOW_SIZE,
        "n_cyber_features": int(Xc_train.shape[-1]),
        "n_physical_features": int(Xp_train.shape[-1]),
        "train_windows": int(Xc_train.shape[0]),
        "val_windows": int(Xc_val.shape[0]),
        "test_windows": int(Xc_test.shape[0]),
        "cyber_features": cyber_features,
        "physical_features": physical_features,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=list)

    print("\nCached preprocessed tensors ->", cache_dir)
    return data, scaler_c, scaler_p, meta


def class_counts(y, num_classes=5):
    counts = np.bincount(y, minlength=num_classes)
    return dict(zip(CLASS_NAMES, [int(c) for c in counts]))
