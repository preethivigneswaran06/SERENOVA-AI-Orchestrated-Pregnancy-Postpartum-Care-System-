import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class SerenovaDataset(Dataset):

    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))

        # 🔥 FIXED FEATURE ORDER
        self.feature_keys = sorted(self.data[0]["features"].keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        y = sample["labels"]["fetal_hypoxia"]
        feats = sample["features"]

        # 🔥 FEATURE VECTOR
        x = [feats.get(k, 0.0) for k in self.feature_keys]
        x = np.array(x, dtype=np.float32)

        # 🔥 FIX 1: REMOVE NaN / INF
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 🔥 FIX 2: CLIP EXTREME VALUES
        x = np.clip(x, -1e3, 1e3)

        # 🔥 FIX 3: SAFE NORMALIZATION
        mean = np.mean(x)
        std = np.std(x)

        if std < 1e-6:
            x = np.zeros_like(x)
        else:
            x = (x - mean) / std

        # 🔥 CNN INPUT SHAPE → (1, features)
        x = np.expand_dims(x, axis=0)

        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)