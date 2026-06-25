# train_lstm_from_two_logs.py
# Build train/val/test from TWO continuous 10-min logs, then train LSTM.
# Center point: marker_2 (relative positions marker_3..7 w.r.t marker_2)
# Enhanced with test evaluation, visualization, and hyperparameter tuning

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json
import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] optuna not available. Install with: pip install optuna")


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

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Args:
    out_dir: str
    window: int
    horizon: int
    stride: int

    train_ratio: float
    val_ratio: float

    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    num_layers: int
    dropout: float

    input_col: str

    tune: bool
    n_trials: int


def parse_args() -> Args:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default=f"./runs/lstm_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)

    # split strategy
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--input_col", default="current_pressure")

    # Hyperparameter tuning
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n_trials", type=int, default=50)

    ns = ap.parse_args()

    return Args(**vars(ns))


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


class SlidingWindowDataset(Dataset):
    """
    X: [window, 1+20] from data[1] (pressure + relative markers)
    Y: relative markers (dx,dy,dz,valid)
       center marker: marker_2, predict marker_3..7 -> 5 markers
       dims = 5*(3+1)=20
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: TrainConfig,
        input_col: str,
        center_xyz_cols: Tuple[str, str, str],  # marker_2_x/y/z
        out_marker_ids: List[int],  # [3,4,5,6,7]
        normalize_stats: Optional[dict] = None,
        fit_stats: bool = False,
    ):
        self.cfg = cfg

        # Required columns
        pos_cols = []
        valid_cols = []
        for mid in out_marker_ids:
            pos_cols.extend([f"marker_{mid}_x", f"marker_{mid}_y", f"marker_{mid}_z"])
            valid_cols.append(f"marker_{mid}_valid")

        required = [input_col, *center_xyz_cols, *pos_cols, *valid_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        x = df[input_col].astype(np.float32).to_numpy()  # [T]
        c = df[list(center_xyz_cols)].astype(np.float32).to_numpy()  # [T,3]
        out_pos = df[pos_cols].astype(np.float32).to_numpy()  # [T, 15]
        out_valid = df[valid_cols].astype(np.float32).to_numpy()  # [T, 5]

        # relative positions: marker_i - center(marker_2)
        out_pos = out_pos.reshape(len(df), len(out_marker_ids), 3)  # [T,5,3]
        rel_pos = out_pos - c[:, None, :]  # [T,5,3]
        rel_pos = rel_pos.reshape(len(df), len(out_marker_ids) * 3)  # [T,15]

        y = np.concatenate([rel_pos, out_valid], axis=1).astype(np.float32)  # [T,20]

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
            x_mean = float(x.mean())
            x_std = float(x.std() + 1e-8)

            y_pos = y[:, : len(out_marker_ids) * 3]  # 15
            y_mean = y_pos.mean(axis=0).astype(np.float32)
            y_std_pos = (y_pos.std(axis=0) + 1e-8).astype(np.float32)

            normalize_stats = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std_pos": y_std_pos,
            }

        self.stats = normalize_stats
        self.x = x
        self.y = y
        self.pos_dim = len(out_marker_ids) * 3  # 15
        self.total_dim = self.pos_dim + len(out_marker_ids)  # 20

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        w = self.window
        h = self.horizon
        t_y = i + w - 1 + h

        x_seq = self.x[i : i + w]  # [w]
        y_seq = self.y[i : i + w]  # [w, 20]
        y_t = self.y[t_y]  # [20]

        # normalize x
        x_seq = (x_seq - self.stats["x_mean"]) / self.stats["x_std"]
        x_seq = x_seq[:, None].astype(np.float32)  # [w,1]

        # normalize y_seq (position part only)
        y_seq_pos = y_seq[:, : self.pos_dim]
        y_seq_valid = y_seq[:, self.pos_dim :]
        y_seq_pos = (y_seq_pos - self.stats["y_mean"]) / self.stats["y_std_pos"]
        y_seq_norm = np.concatenate([y_seq_pos, y_seq_valid], axis=1).astype(np.float32)

        in_seq = np.concatenate([x_seq, y_seq_norm], axis=1)  # [w, 21]

        # normalize position part only for target
        y_pos = y_t[: self.pos_dim]
        y_valid = y_t[self.pos_dim :]
        y_pos = (y_pos - self.stats["y_mean"]) / self.stats["y_std_pos"]

        y_out = np.concatenate([y_pos, y_valid], axis=0).astype(np.float32)
        return torch.from_numpy(in_seq), torch.from_numpy(y_out)


class LSTMRegressor(nn.Module):
    def __init__(
        self, pos_dim: int, valid_dim: int, hidden_size=128, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.valid_dim = valid_dim
        input_size = (
            1 + pos_dim + valid_dim
        )  # 1 (pressure) + 15 (position) + 5 (valid) = 21

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pos_dim),
        )
        self.valid_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, valid_dim),
        )

    def forward(self, x):
        # x: [B,T,1+pos_dim+valid_dim]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        pos = self.pos_head(h)
        valid_logits = self.valid_head(h)
        return pos, valid_logits


@torch.no_grad()
def evaluate(
    model: LSTMRegressor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pos_dim: int,
    cfg: TrainConfig,
    detailed: bool = False,
):
    model.eval()
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    tot_loss, tot_mse, tot_mae, n = 0.0, 0.0, 0.0, 0

    if detailed:
        all_preds_pos = []
        all_targets_pos = []
        all_preds_valid = []
        all_targets_valid = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pos = y[:, :pos_dim]
        y_valid = y[:, pos_dim:]

        pred_pos, pred_valid_logits = model(x)
        loss_pos: nn.MSELoss = mse(pred_pos, y_pos)
        loss_mae: nn.L1Loss = mae(pred_pos, y_pos)
        loss_valid: nn.BCEWithLogitsLoss = bce(pred_valid_logits, y_valid)
        loss = cfg.pos_loss_weight * loss_pos + cfg.valid_loss_weight * loss_valid

        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_mse += loss_pos.item() * bs
        tot_mae += loss_mae.item() * bs
        n += bs

        pred_pos: torch.Tensor
        y_pos: torch.Tensor
        y_valid: torch.Tensor

        if detailed:
            all_preds_pos.append(pred_pos.cpu().numpy())
            all_targets_pos.append(y_pos.cpu().numpy())
            all_preds_valid.append(torch.sigmoid(pred_valid_logits).cpu().numpy())
            all_targets_valid.append(y_valid.cpu().numpy())

    if detailed:
        return {
            "loss": tot_loss / n,
            "mse": tot_mse / n,
            "mae": tot_mae / n,
            "rmse": np.sqrt(tot_mse / n),
            "preds_pos": np.concatenate(all_preds_pos, axis=0),
            "targets_pos": np.concatenate(all_targets_pos, axis=0),
            "preds_valid": np.concatenate(all_preds_valid, axis=0),
            "targets_valid": np.concatenate(all_targets_valid, axis=0),
        }
    return tot_loss / n


def train_one_epoch(
    model: LSTMRegressor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_dim: int,
    cfg: TrainConfig,
):
    model.train()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    tot, n = 0.0, 0
    for x, y in loader:
        x: torch.Tensor
        y: torch.Tensor

        x, y = x.to(device), y.to(device)
        y_pos = y[:, :pos_dim]
        y_valid = y[:, pos_dim:]

        pred_pos, pred_valid_logits = model(x)
        loss_pos = mse(pred_pos, y_pos)
        loss_valid = bce(pred_valid_logits, y_valid)
        loss = cfg.pos_loss_weight * loss_pos + cfg.valid_loss_weight * loss_valid

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / n


def visualize_test_results(results: Dict, out_dir: str, num_markers: int = 5):
    """
    Visualize test set predictions vs ground truth.
    """
    os.makedirs(out_dir, exist_ok=True)

    preds = results["preds_pos"]
    targets = results["targets_pos"]

    # Per-marker errors
    fig, axes = plt.subplots(num_markers, 3, figsize=(15, 3 * num_markers))
    fig.suptitle("Test Set: Predicted vs Actual (Relative Positions)", fontsize=16)

    for i in range(num_markers):
        for j, coord in enumerate(["X", "Y", "Z"]):
            ax = axes[i, j] if num_markers > 1 else axes[j]
            idx = i * 3 + j

            # Sample for visualization (max 500 points)
            sample_size = min(500, len(preds))
            indices = np.linspace(0, len(preds) - 1, sample_size, dtype=int)

            ax.plot(
                indices,
                targets[indices, idx],
                "b-",
                alpha=0.6,
                label="Ground Truth",
                linewidth=1,
            )
            ax.plot(
                indices,
                preds[indices, idx],
                "r--",
                alpha=0.6,
                label="Predicted",
                linewidth=1,
            )
            ax.set_title(f"Marker {i+3} - {coord}", fontsize=10)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Position (normalized)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "test_predictions.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Error distribution
    errors = np.abs(preds - targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-marker MAE
    marker_errors = []
    for i in range(num_markers):
        marker_err = errors[:, i * 3 : (i + 1) * 3].mean()
        marker_errors.append(marker_err)

    axes[0].bar(range(num_markers), marker_errors, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Marker ID", fontsize=12)
    axes[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes[0].set_title("Per-Marker MAE", fontsize=14)
    axes[0].set_xticks(range(num_markers))
    axes[0].set_xticklabels([f"M{i+3}" for i in range(num_markers)])
    axes[0].grid(True, alpha=0.3, axis="y")

    # Error histogram
    axes[1].hist(errors.flatten(), bins=50, color="coral", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Absolute Error", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Error Distribution", fontsize=14)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_errors.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter plot: predicted vs actual
    fig, ax = plt.subplots(figsize=(8, 8))
    sample_size = min(1000, len(preds))
    indices = np.random.choice(len(preds), sample_size, replace=False)

    ax.scatter(
        targets[indices].flatten(),
        preds[indices].flatten(),
        alpha=0.3,
        s=10,
        color="blue",
    )

    # Perfect prediction line
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Ground Truth", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Prediction Scatter Plot", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Test visualizations saved to {out_dir}")


def save_test_metrics(results: Dict, out_dir: str, num_markers: int = 5):
    """
    Save detailed test metrics to JSON.
    """
    preds = results["preds_pos"]
    targets = results["targets_pos"]
    errors = np.abs(preds - targets)

    metrics = {
        "overall": {
            "mae": float(results["mae"]),
            "mse": float(results["mse"]),
            "rmse": float(results["rmse"]),
            "loss": float(results["loss"]),
        },
        "per_marker": {},
    }

    for i in range(num_markers):
        marker_preds = preds[:, i * 3 : (i + 1) * 3]
        marker_targets = targets[:, i * 3 : (i + 1) * 3]
        marker_errors = errors[:, i * 3 : (i + 1) * 3]

        metrics["per_marker"][f"marker_{i+3}"] = {
            "mae": float(marker_errors.mean()),
            "mse": float(((marker_preds - marker_targets) ** 2).mean()),
            "rmse": float(np.sqrt(((marker_preds - marker_targets) ** 2).mean())),
            "mae_x": float(marker_errors[:, 0].mean()),
            "mae_y": float(marker_errors[:, 1].mean()),
            "mae_z": float(marker_errors[:, 2].mean()),
        }

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Test metrics saved to {os.path.join(out_dir, 'test_metrics.json')}")
    return metrics


def hyperparameter_tuning(
    train_df,
    val_df,
    test_df,
    input_col: str,
    center_cols: Tuple,
    out_marker_ids: List[int],
    device,
    n_trials: int = 50,
):
    """
    Perform hyperparameter tuning using Optuna.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. Install with: pip install optuna"
        )

    def objective(trial):
        # Suggest hyperparameters
        cfg = TrainConfig(
            window=1,  # Fixed to 1
            horizon=trial.suggest_int("horizon", 1, 10),
            stride=trial.suggest_int("stride", 1, 5),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            epochs=trial.suggest_int("epochs", 20, 50),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
            num_layers=trial.suggest_int("num_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            grad_clip=trial.suggest_float("grad_clip", 0.5, 2.0),
        )

        try:
            # Create datasets
            train_ds = SlidingWindowDataset(
                train_df,
                cfg,
                input_col,
                center_cols,
                out_marker_ids,
                normalize_stats=None,
                fit_stats=True,
            )
            stats = train_ds.stats

            val_ds = SlidingWindowDataset(
                val_df,
                cfg,
                input_col,
                center_cols,
                out_marker_ids,
                normalize_stats=stats,
                fit_stats=False,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
            )

            pos_dim = len(out_marker_ids) * 3
            valid_dim = len(out_marker_ids)

            model = LSTMRegressor(
                pos_dim,
                valid_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

            best_val = float("inf")
            patience = 5
            patience_counter = 0

            for ep in range(1, cfg.epochs + 1):
                train_one_epoch(model, train_loader, optimizer, device, pos_dim, cfg)
                val_loss = evaluate(
                    model, val_loader, device, pos_dim, cfg, detailed=False
                )

                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    break

                # Report intermediate value for pruning
                trial.report(val_loss, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val

        except Exception as e:
            print(f"[ERROR] Trial failed: {e}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n[TUNING RESULTS]")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    return study.best_params, study


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default=f"./runs/lstm_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--input_col", default="current_pressure")

    # Hyperparameter tuning
    ap.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    ap.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials")

    # args: argparse.Namespace = ap.parse_args()
    args = parse_args()

    cfg = TrainConfig(
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # Load data
    df1 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"
    )
    df2 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/new_merged_data.csv"
    )

    center_cols = ("marker_0_x", "marker_0_y", "marker_0_z")
    out_marker_ids = [1, 2, 3, 4, 5]  # predict others relative to marker_2

    # Split data
    t1, v1, te1 = split_continuous_df(df1, args.train_ratio, args.val_ratio)
    t2, v2, te2 = split_continuous_df(df2, args.train_ratio, args.val_ratio)

    train_df = pd.concat([t1, t2], axis=0, ignore_index=True)
    val_df = pd.concat([v1, v2], axis=0, ignore_index=True)
    test_df = pd.concat([te1, te2], axis=0, ignore_index=True)

    print(
        f"[INFO] Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Hyperparameter tuning
    if args.tune:
        print("\n[INFO] Starting hyperparameter tuning...")
        best_params, study = hyperparameter_tuning(
            train_df,
            val_df,
            test_df,
            args.input_col,
            center_cols,
            out_marker_ids,
            device,
            n_trials=args.n_trials,
        )

        # Update config with best params
        for key, value in best_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        print(f"\n[INFO] Updated config with best params")

        # Save tuning results
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "tuning_results.json"), "w") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "best_value": study.best_value,
                    "n_trials": args.n_trials,
                },
                f,
                indent=2,
            )

        # Plot optimization history
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(os.path.join(args.out_dir, "tuning_history.png"))

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(args.out_dir, "param_importances.png"))
        except Exception as e:
            print(f"[WARNING] Could not save tuning plots: {e}")

    # Fit stats on train only
    train_ds = SlidingWindowDataset(
        train_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=None,
        fit_stats=True,
    )
    stats = train_ds.stats
    val_ds = SlidingWindowDataset(
        val_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )
    test_ds = SlidingWindowDataset(
        test_df,
        cfg,
        args.input_col,
        center_cols,
        out_marker_ids,
        normalize_stats=stats,
        fit_stats=False,
    )

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

    pos_dim = len(out_marker_ids) * 3
    valid_dim = len(out_marker_ids)

    model = LSTMRegressor(
        pos_dim,
        valid_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    best_val = float("inf")
    train_losses = []
    val_losses = []

    print(f"\n[INFO] Starting training for {cfg.epochs} epochs...")
    for ep in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        tr = train_one_epoch(model, train_loader, optimizer, device, pos_dim, cfg)
        va = evaluate(model, val_loader, device, pos_dim, cfg, detailed=False)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"[E{ep:03d}] train={tr:.6f} val={va:.6f}")

        torch.save(
            {"model": model.state_dict(), "stats": stats, "cfg": cfg.to_dict()},
            os.path.join(args.out_dir, "last.pt"),
        )

        if va < best_val:
            best_val = va
            torch.save(
                {"model": model.state_dict(), "stats": stats, "cfg": cfg.to_dict()},
                os.path.join(args.out_dir, "best.pt"),
            )
            print(f"[INFO] saved best (val={best_val:.6f})")

    # Load best model for testing
    print("\n[INFO] Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.out_dir, "best.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model"])

    # Detailed test evaluation
    print("\n[INFO] Evaluating on test set...")
    test_results = evaluate(model, test_loader, device, pos_dim, cfg, detailed=True)

    print("\n=== TEST SET RESULTS ===")
    print(f"Loss: {test_results['loss']:.6f}")
    print(f"MAE: {test_results['mae']:.6f}")
    print(f"MSE: {test_results['mse']:.6f}")
    print(f"RMSE: {test_results['rmse']:.6f}")

    # Save detailed test metrics
    metrics = save_test_metrics(
        test_results, args.out_dir, num_markers=len(out_marker_ids)
    )

    # Visualize test results
    visualize_test_results(test_results, args.out_dir, num_markers=len(out_marker_ids))

    # Visualize test results
    visualize_test_results(test_results, args.out_dir, num_markers=len(out_marker_ids))

    # Print per-marker metrics
    print("\n=== PER-MARKER METRICS ===")
    for marker_name, marker_metrics in metrics["per_marker"].items():
        print(f"\n{marker_name}:")
        print(f"  MAE: {marker_metrics['mae']:.6f}")
        print(f"  RMSE: {marker_metrics['rmse']:.6f}")
        print(
            f"  MAE (X/Y/Z): {marker_metrics['mae_x']:.6f} / {marker_metrics['mae_y']:.6f} / {marker_metrics['mae_z']:.6f}"
        )

    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std_pos=stats["y_std_pos"],
    )
    print(f"\n[INFO] saved stats -> {os.path.join(args.out_dir, 'norm_stats.npz')}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, cfg.epochs + 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="s", linewidth=2)
    plt.axhline(
        y=test_results["loss"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Test Loss ({test_results['loss']:.4f})",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training, Validation, and Test Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] saved training plot -> {plot_path}")
    plt.close()

    print(f"\n[INFO] All results saved to {args.out_dir}")
    print("\nGenerated files:")
    print("  - best.pt, last.pt (model checkpoints)")
    print("  - config.json (training configuration)")
    print("  - norm_stats.npz (normalization statistics)")
    print("  - test_metrics.json (detailed test metrics)")
    print("  - training_history.png (training curves)")
    print("  - test_predictions.png (prediction vs ground truth)")
    print("  - test_errors.png (error analysis)")
    print("  - test_scatter.png (scatter plot)")
    if args.tune:
        print("  - tuning_results.json (hyperparameter tuning results)")
        print("  - tuning_history.png (tuning optimization history)")
        print("  - param_importances.png (parameter importance)")


if __name__ == "__main__":
    main()
