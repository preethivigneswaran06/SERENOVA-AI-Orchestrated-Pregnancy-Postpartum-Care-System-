# src/model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BloomCNN(nn.Module):
    def __init__(self):
        super(BloomCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)

        self.pool = nn.MaxPool1d(2)

        # 🔥 KEY FIX: use Adaptive pooling → fixes ANY input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)  # 32 × 10 = 320

        self.fc = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # 🔥 This guarantees fixed size = 320
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)  # → always (B, 320)

        x = self.fc(x)

        return x