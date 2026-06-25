import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


@dataclass
class TrainConfig:
    window: int = 60
    horizon: int = 1
    stride: int = 1
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pos_loss_weight: float = 1.0
    valid_loss_weight: float = 0.2
    grad_clip: float = 1.0
    seed: int = 42
    input_size: int = 1
    output_size: int = 9

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ControlDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: TrainConfig):
        """
        time,target_pressure,current_pressure,filtered_pressure,valve,marker_0_x,marker_0_y,marker_0_z,marker_0_valid,marker_1_x,marker_1_y,marker_1_z,marker_1_valid,marker_2_x,marker_2_y,marker_2_z,marker_2_valid,marker_3_x,marker_3_y,marker_3_z,marker_3_valid,marker_4_x,marker_4_y,marker_4_z,marker_4_valid,marker_5_x,marker_5_y,marker_5_z,marker_5_valid
        """
        self.cfg = cfg

        pressure = df["current_pressure"].to_numpy()

        links = [None for _ in range(5)]
        joints = [None for _ in range(4)]

        for i in range(5):
            x1 = df[f"marker_{i}_x"].to_numpy()
            y1 = df[f"marker_{i}_y"].to_numpy()

            x2 = df[f"marker_{i+1}_x"].to_numpy()
            y2 = df[f"marker_{i+1}_y"].to_numpy()

            links[i] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for i in range(4):
            x1 = df[f"marker_{i}_x"].to_numpy() - df["marker_0_x"].to_numpy()
            y1 = df[f"marker_{i}_y"].to_numpy() - df["marker_0_y"].to_numpy()

            x2 = df[f"marker_{i+1}_x"].to_numpy() - df["marker_0_x"].to_numpy()
            y2 = df[f"marker_{i+1}_y"].to_numpy() - df["marker_0_y"].to_numpy()

            x3 = df[f"marker_{i+2}_x"].to_numpy() - df["marker_0_x"].to_numpy()
            y3 = df[f"marker_{i+2}_y"].to_numpy() - df["marker_0_y"].to_numpy()

            vec1 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            vec2 = np.array([x3 - x2, y3 - y2], dtype=np.float64)

            joints[i] = np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])

        self.data = pressure  # shape (N,)
        self.label = np.stack(links + joints, axis=1, dtype=np.float32)  # shape (N, 9)

    def __len__(self):
        total = len(self.data) - self.cfg.window - self.cfg.horizon + 1
        return max(0, (total + self.cfg.stride - 1) // self.cfg.stride)

    def __getitem__(self, idx):
        start = idx * self.cfg.stride

        x = self.data[start : start + self.cfg.window]  # (window,)
        x = x[:, None]  # (window, 1)

        # 미래 1-step 예측 기준
        y_index = start + self.cfg.window + self.cfg.horizon - 1
        y = self.label[y_index]  # (9,)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class ControlLSTM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


def train_one_epoch(
    model: ControlLSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
):
    total_loss = 0.0

    model.train()
    mse = nn.MSELoss()

    for x_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        x_batch = x_batch.to(device)  # shape: (batch, window, 1)
        y_batch = y_batch.to(device)  # shape: (batch, 9)

        pred = model(x_batch)  # shape: (batch, 9)

        link_loss = mse(pred[:, :5], y_batch[:, :5])
        joint_loss = mse(pred[:, 5:], y_batch[:, 5:])

        loss = cfg.pos_loss_weight * link_loss + cfg.valid_loss_weight * joint_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: ControlLSTM,
    dataloader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[float, float]:

    mse = nn.MSELoss()
    model.eval()

    total_mse = 0.0

    for x_batch, y_batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x_batch: torch.Tensor = x_batch.to(device)
        y_batch: torch.Tensor = y_batch.to(device)

        pred = model(x_batch)
        loss = mse(pred, y_batch)
        total_mse += loss.item() * x_batch.size(0)

    return total_mse / len(dataloader.dataset)


def main(*args, **kwargs):
    cfg = TrainConfig(
        window=20,
        horizon=1,
        stride=1,
        batch_size=256,
        num_workers=4,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        pos_loss_weight=1.0,
        valid_loss_weight=0.2,
        grad_clip=1.0,
        seed=42,
    )
    set_seed(cfg.seed)

    output_dir = "control_lstm_results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)

    dataset = ControlDataset(
        df=pd.read_csv("data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"), cfg=cfg
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    best_val = float("inf")
    train_losses = []
    val_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlLSTM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    for ep in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        train_loss = train_one_epoch(model, dataloader, optimizer, device, cfg)
        valid = evaluate(model, dataloader, device, cfg)

        train_losses.append(train_loss)

        torch.save(
            model.state_dict(), os.path.join(output_dir, f"model_epoch_{ep}.pth")
        )

        if valid < best_val:
            best_val = valid
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_best.pth"))
            print(f"Epoch {ep}: New best validation MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
