# src/model/infer.py

import torch
import numpy as np
from pathlib import Path

from src.model.model import BloomCNN
from src.model.input_builder import build_input_tensor
from src.preprocessing.signal_processor import process_signals


_MODEL_PATH = Path("models/serenpfm.pt")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FetalHypoxiaPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return

        if not _MODEL_PATH.exists():
            raise FileNotFoundError("Place serenpfm.pt inside /models folder")

        self.model = BloomCNN()
        self.model.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE))
        self.model.to(_DEVICE)
        self.model.eval()

        self._loaded = True
        print(f"[INFO] Model loaded on {_DEVICE}")

    def predict(self, raw_signals: dict):
        self._load()

        # Step 1: preprocess
        cleaned = process_signals(raw_signals)

        # Step 2: tensor build
        tensor = build_input_tensor(cleaned).to(_DEVICE)

        # Step 3: inference
        with torch.no_grad():
            logits = self.model(tensor)

            if logits.shape[-1] == 2:
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                prob = torch.sigmoid(logits).item()

        return {
            "hypoxia_probability": round(prob, 4),
            "hypoxia_flag": prob >= 0.5
        }