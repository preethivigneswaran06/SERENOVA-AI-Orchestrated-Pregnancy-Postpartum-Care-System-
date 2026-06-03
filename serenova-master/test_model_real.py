# test_model_real.py

import json
import numpy as np
from src.model.infer import FetalHypoxiaPredictor

# LOAD JSON
with open("sensor_data.json", "r") as f:
    data = json.load(f)

# EXTRACT ECG
ecg = [row["ecg"] for row in data]

signal = np.array(ecg, dtype=np.float32)

# NORMALIZE
signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

# MODEL
model = FetalHypoxiaPredictor()

# WINDOW
window_size = 128
probs = []

for i in range(0, len(signal) - window_size, window_size):
    chunk = signal[i:i+window_size]
    res = model.predict({"FHR": chunk})
    probs.append(res["hypoxia_probability"])

# FINAL
if probs:
    avg = sum(probs) / len(probs)
    print("\n🔥 FINAL RESULT")
    print("Hypoxia Probability:", round(avg, 4))
else:
    print("❌ Not enough data")
print("Total rows:", len(data))