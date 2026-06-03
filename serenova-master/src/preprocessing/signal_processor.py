# src/preprocessing/signal_processor.py

import numpy as np
from scipy.signal import butter, filtfilt


# ───────────── FILTER ─────────────
def bandpass(signal, low, high, fs):
    nyq = fs / 2.0
    low = low / nyq
    high = high / nyq

    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ───────────── NORMALIZATION ─────────────
def normalize(x):
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


# ───────────── SIGNAL CONFIG ─────────────
CONFIG = {
    "ECG": (0.5, 40, 250),
    "PPG": (0.5, 8, 100),
    "RESP": (0.05, 2, 50),
    "FHR": (0.01, 5, 4),
    "PLETH": (0.5, 8, 100),
}


# ───────────── MAIN PROCESSOR ─────────────
def process_signals(raw):
    cleaned = {}

    for name, values in raw.items():
        arr = np.array(values, dtype=np.float32)

        # handle NaN / inf
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = np.nanmean(arr) if np.isfinite(arr).any() else 0.0

        if name in CONFIG:
            try:
                arr = bandpass(arr, *CONFIG[name])
            except Exception:
                pass  # short signals may fail

        cleaned[name] = normalize(arr)

    return cleaned


# ───────────── SCALAR FEATURES (IMPORTANT FOR AGENTS) ─────────────
def extract_scalar_features(raw):
    features = {}

    for name, values in raw.items():
        arr = np.array(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            continue

        features[f"{name}_mean"] = float(arr.mean())
        features[f"{name}_std"] = float(arr.std())
        features[f"{name}_min"] = float(arr.min())
        features[f"{name}_max"] = float(arr.max())

    return features