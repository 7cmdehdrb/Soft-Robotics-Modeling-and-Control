# train_lstm_inverse.py
# Inverse model: predict control signal from marker positions
# Input: relative positions of marker_3..7 w.r.t marker_2 (5 markers × 3 coords = 15 dims)
# Output: data[1] (control signal)

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    grad_clip: float = 1.0
    seed: int = 42


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_continuous_df(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time order (no shuffle) to preserve temporal continuity.
    """
    if (
        not (0 < train_ratio < 1)
        or not (0 <= val_ratio < 1)
        or (train_ratio + val_ratio >= 1)
    ):
        raise ValueError("Invalid ratios: require 0<train<1, 0<=val<1, train+val<1")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder -> test
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


class InverseSlidingWindowDataset(Dataset):
    """
    Inverse model dataset:
    X: [window, 15] relative marker positions (marker_3..7 w.r.t marker_2)
    Y: [1] control signal (data[1])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: TrainConfig,
        output_col: str,  # data[1]
        center_xyz_cols: Tuple[str, str, str],  # marker_2_x/y/z
        in_marker_ids: List[int],  # [3,4,5,6,7]
        normalize_stats: Optional[dict] = None,
        fit_stats: bool = False,
    ):
        self.cfg = cfg

        # Required columns
        pos_cols = []
        for mid in in_marker_ids:
            pos_cols.extend([f"marker_{mid}_x", f"marker_{mid}_y", f"marker_{mid}_z"])

        required = [output_col, *center_xyz_cols, *pos_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        y = df[output_col].astype(np.float32).to_numpy()  # [T] control signal
        c = df[list(center_xyz_cols)].astype(np.float32).to_numpy()  # [T,3]
        in_pos = df[pos_cols].astype(np.float32).to_numpy()  # [T, 15]

        # relative positions: marker_i - center(marker_2)
        in_pos = in_pos.reshape(len(df), len(in_marker_ids), 3)  # [T,5,3]
        rel_pos = in_pos - c[:, None, :]  # [T,5,3]
        x = rel_pos.reshape(len(df), len(in_marker_ids) * 3)  # [T,15]

        T = len(df)
        self.window = cfg.window
        self.horizon = cfg.horizon
        self.stride = cfg.stride

        self.max_start = T - (self.window + self.horizon)
        if self.max_start < 0:
            raise ValueError(
                f"Too short: T={T}, window={self.window}, horizon={self.horizon}"
            )

        self.starts = np.arange(0, self.max_start + 1, self.stride, dtype=np.int64)

        # normalization
        if normalize_stats is None:
            normalize_stats = {}

        if fit_stats:
            x_mean = x.mean(axis=0).astype(np.float32)  # [15]
            x_std = (x.std(axis=0) + 1e-8).astype(np.float32)  # [15]

            y_mean = float(y.mean())
            y_std = float(y.std() + 1e-8)

            normalize_stats = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

        self.stats = normalize_stats
        self.x = x
        self.y = y
        self.input_dim = len(in_marker_ids) * 3  # 15

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        w = self.window
        h = self.horizon
        t_y = i + w - 1 + h

        x_seq = self.x[i : i + w]  # [w, 15]
        y_t = self.y[t_y]  # scalar

        # normalize x
        x_seq = (x_seq - self.stats["x_mean"]) / self.stats["x_std"]
        x_seq = x_seq.astype(np.float32)  # [w, 15]

        # normalize y
        y_out = (y_t - self.stats["y_mean"]) / self.stats["y_std"]
        y_out = np.array([y_out], dtype=np.float32)  # [1]

        return torch.from_numpy(x_seq), torch.from_numpy(y_out)


class InverseLSTMRegressor(nn.Module):
    """
    Inverse model: predict control signal from marker positions
    Input: [batch, window, 15] marker positions
    Output: [batch, 1] control signal
    """

    def __init__(self, input_dim: int, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: [B,T,15]
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # [B, hidden_size]
        output = self.regressor(h)  # [B, 1]
        return output


@torch.no_grad()
def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    mse = nn.MSELoss()

    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = mse(pred, y)

        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / n


def train_one_epoch(model, loader, optimizer, device, cfg: TrainConfig):
    model.train()
    mse = nn.MSELoss()

    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = mse(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / n


def main():
    ap = argparse.ArgumentParser()

    import datetime

    ap.add_argument(
        "--out_dir",
        default=f"./runs/lstm_inverse_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)

    # split strategy:
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--output_col", default="data[1]")

    args = ap.parse_args()

    cfg = TrainConfig(
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    df1 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/merged_data.csv"
    )
    df2 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/merged_data.csv"
    )

    center_cols = ("marker_2_x", "marker_2_y", "marker_2_z")
    in_marker_ids = [3, 4, 5, 6, 7]  # use markers 3-7 as input

    t1, v1, te1 = split_continuous_df(df1, args.train_ratio, args.val_ratio)
    t2, v2, te2 = split_continuous_df(df2, args.train_ratio, args.val_ratio)

    train_df = pd.concat([t1, t2], axis=0, ignore_index=True)
    val_df = pd.concat([v1, v2], axis=0, ignore_index=True)
    test_df = pd.concat([te1, te2], axis=0, ignore_index=True)

    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Val samples: {len(val_df)}")
    print(f"[INFO] Test samples: {len(test_df)}")

    # Fit stats on train only
    train_ds = InverseSlidingWindowDataset(
        train_df,
        cfg,
        args.output_col,
        center_cols,
        in_marker_ids,
        normalize_stats=None,
        fit_stats=True,
    )
    stats = train_ds.stats
    val_ds = InverseSlidingWindowDataset(
        val_df,
        cfg,
        args.output_col,
        center_cols,
        in_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )
    test_ds = InverseSlidingWindowDataset(
        test_df,
        cfg,
        args.output_col,
        center_cols,
        in_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )

    print(f"[INFO] Train windows: {len(train_ds)}")
    print(f"[INFO] Val windows: {len(val_ds)}")
    print(f"[INFO] Test windows: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    input_dim = len(in_marker_ids) * 3  # 15

    model = InverseLSTMRegressor(
        input_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")
    train_losses = []
    val_losses = []

    print(f"\n[INFO] Starting training for {cfg.epochs} epochs...")
    print(f"[INFO] Output directory: {args.out_dir}")

    for ep in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        tr = train_one_epoch(model, train_loader, optimizer, device, cfg)
        va = evaluate(model, val_loader, device, cfg)
        train_losses.append(tr)
        val_losses.append(va)

        tqdm.write(f"[E{ep:03d}] train={tr:.6f} val={va:.6f}")

        torch.save(
            {"model": model.state_dict(), "stats": stats, "cfg": cfg.__dict__},
            os.path.join(args.out_dir, "last.pt"),
        )

        if va < best_val:
            best_val = va
            torch.save(
                {"model": model.state_dict(), "stats": stats, "cfg": cfg.__dict__},
                os.path.join(args.out_dir, "best.pt"),
            )
            tqdm.write(f"[INFO] saved best (val={best_val:.6f})")

    test_loss = evaluate(model, test_loader, device, cfg)
    print(f"\n[TEST] loss={test_loss:.6f}")

    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
    )
    print(f"[INFO] saved stats -> {os.path.join(args.out_dir, 'norm_stats.npz')}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, cfg.epochs + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Inverse Model: Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] saved training plot -> {plot_path}")
    plt.close()

    # Save final results summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Inverse LSTM Model Training Summary ===\n\n")
        f.write(f"Model Type: Inverse (Markers → Control Signal)\n")
        f.write(f"Input: Relative positions of markers 3-7 (15 dims)\n")
        f.write(f"Output: Control signal (data[1])\n\n")
        f.write("Configuration:\n")
        f.write(f"  Window: {cfg.window}\n")
        f.write(f"  Horizon: {cfg.horizon}\n")
        f.write(f"  Batch size: {cfg.batch_size}\n")
        f.write(f"  Epochs: {cfg.epochs}\n")
        f.write(f"  Learning rate: {cfg.lr}\n")
        f.write(f"  Hidden size: {cfg.hidden_size}\n")
        f.write(f"  Num layers: {cfg.num_layers}\n\n")
        f.write("Results:\n")
        f.write(f"  Best validation loss: {best_val:.6f}\n")
        f.write(f"  Test loss: {test_loss:.6f}\n")
        f.write(f"  Final train loss: {train_losses[-1]:.6f}\n")
    print(f"[INFO] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
