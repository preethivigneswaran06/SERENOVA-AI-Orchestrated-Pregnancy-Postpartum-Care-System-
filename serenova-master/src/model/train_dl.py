import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils.dataset import SerenovaDataset
from src.model.model import SerenPFM


def train():

    dataset = SerenovaDataset("data/final_dataset.pkl")

    input_len = len(dataset.feature_keys)

    # 🔥 SPLIT
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    model = SerenPFM(input_len)

    # 🔥 STABLE LOSS
    criterion = nn.BCEWithLogitsLoss()

    # 🔥 LOWER LR
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(5):

        print(f"\n🚀 Epoch {epoch+1}")
        model.train()
        total_loss = 0

        for X, y in tqdm(train_loader):

            # 🔥 SKIP BAD DATA
            if torch.isnan(X).any():
                continue

            y = y.unsqueeze(1)

            pred = model(X)
            loss = criterion(pred, y)

            # 🔥 SKIP NaN LOSS
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        # 🔥 VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:

                if torch.isnan(X).any():
                    continue

                y = y.unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), "data/serenpfm.pt")
    print("✅ Model saved")