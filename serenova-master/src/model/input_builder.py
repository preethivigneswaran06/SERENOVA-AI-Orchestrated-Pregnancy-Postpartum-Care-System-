# src/model/input_builder.py

import numpy as np
import torch


def build_input_tensor(processed_signals: dict, min_length: int = 128):
    """
    Build CNN input tensor from processed signals.

    - Uses FHR signal (model trained on 1 channel)
    - Handles missing values safely
    - Does NOT force fixed length (adaptive pooling handles it)
    """

    # 🔹 Get FHR signal
    signal = processed_signals.get("FHR")

    if signal is None or len(signal) == 0:
        signal = np.zeros(min_length)

    signal = np.array(signal, dtype=np.float32)

    # 🔹 Clean NaN / Inf
    mask = ~np.isfinite(signal)
    if mask.any():
        signal[mask] = np.nanmean(signal) if np.isfinite(signal).any() else 0.0

    # 🔹 Ensure minimum length (avoid tiny inputs)
    if len(signal) < min_length:
        signal = np.pad(signal, (0, min_length - len(signal)))

    # 🔹 Shape → (1, 1, L)
    x = np.expand_dims(signal, axis=0)   # (1, L)
    x = np.expand_dims(x, axis=0)        # (1, 1, L)

    return torch.tensor(x, dtype=torch.float32)