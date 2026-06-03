import numpy as np
from src.pipeline.orchestrator import Orchestrator


def fake_signals():
    t = np.linspace(0, 10, 512)
    return {
        "ECG": np.sin(t),
        "PPG": np.sin(t),
        "RESP": np.sin(t),
        "FHR": np.ones(512) * 140,
        "PLETH": np.sin(t),
        "mfcc": np.random.randn(512),
    }


orc = Orchestrator()

# ✅ Build payload correctly
payload = {
    "raw_signals": fake_signals(),

    "maternal_context": {
        "HR_value": 90,
        "SpO2_value": 98,
        "blood_pressure": {"systolic": 120, "diastolic": 80},
        "hrv_ms": 60,
        "glucose": 100,
        "sleep_score": 80,
        "weight_kg": 65,
        "RESP_rate": 16
    },
"nutrition_input": {
    "calories": 2200,
    "protein_g": 75,
    "iron_mg": 25
},

"supplement_log": {
    "folic_acid": True,
    "iron": True,
    "calcium": True
},

"hydration_ml": 2200,
    "gestational_week": 28,

    "trend_history": [],

    "nutrition_score": 70,

    "due_date": "2026-01-01",

    "birth_date": "2025-01-01",

    "question": "How am I doing?"
}

# ✅ Correct call
result = orc.run_sync(payload)

import json

print(json.dumps(result, indent=2))