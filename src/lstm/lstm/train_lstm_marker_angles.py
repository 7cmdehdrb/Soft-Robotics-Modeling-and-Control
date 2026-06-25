import argparse
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


ANGLE_TRIPLES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5),
)
EXPECTED_MARKER_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
REQUIRED_BASE_COLUMNS: Tuple[str, ...] = (
    "time",
    "target_pressure",
    "current_pressure",
    "filtered_pressure",
    "valve",
)


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

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Args:
    csv_paths: List[str]
    out_dir: str
    input_cols: List[str]
    window: int
    horizon: int
    stride: int
    train_ratio: float
    val_ratio: float
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    dropout: float
    grad_clip: float
    seed: int
    tune: bool
    n_trials: int


class SlidingWindowAngleDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: TrainConfig,
        input_cols: Sequence[str],
        normalize_stats: Optional[Dict[str, np.ndarray]] = None,
        fit_stats: bool = False,
    ):
        validate_dataframe_schema(df)

        self.cfg = cfg
        self.input_cols = list(input_cols)
        self.window = cfg.window
        self.horizon = cfg.horizon
        self.stride = cfg.stride

        x = df[self.input_cols].astype(np.float32).to_numpy()  # [T, C]
        angles_deg = compute_angle_targets_deg(df).astype(np.float32)  # [T, 4]

        total_steps = len(df)
        self.max_start = total_steps - (self.window + self.horizon)
        if self.max_start < 0:
            raise ValueError(
                f"Too short: T={total_steps}, window={self.window}, horizon={self.horizon}"
            )

        self.starts = np.arange(0, self.max_start + 1, self.stride, dtype=np.int64)

        if normalize_stats is None:
            normalize_stats = {}

        if fit_stats:
            x_mean = x.mean(axis=0).astype(np.float32)
            x_std = (x.std(axis=0) + 1e-8).astype(np.float32)
            y_mean = angles_deg.mean(axis=0).astype(np.float32)
            y_std = (angles_deg.std(axis=0) + 1e-8).astype(np.float32)
            normalize_stats = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

        required_stats = ("x_mean", "x_std", "y_mean", "y_std")
        missing_stats = [k for k in required_stats if k not in normalize_stats]
        if missing_stats:
            raise ValueError(f"Missing normalization stats: {missing_stats}")

        self.stats = normalize_stats
        self.x = x
        self.y_deg = angles_deg
        self.input_dim = x.shape[1]
        self.target_dim = angles_deg.shape[1]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        w = self.window
        h = self.horizon
        target_index = start + w - 1 + h

        x_seq = self.x[start : start + w]
        y_deg = self.y_deg[target_index]

        x_seq = (x_seq - self.stats["x_mean"]) / self.stats["x_std"]
        y_norm = (y_deg - self.stats["y_mean"]) / self.stats["y_std"]

        return torch.from_numpy(x_seq.astype(np.float32)), torch.from_numpy(
            y_norm.astype(np.float32)
        )


class LSTMAngleRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train an LSTM to predict 4 acute XY-plane marker angles from CSV logs."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more CSV files with the expected marker columns.",
    )
    parser.add_argument(
        "--out_dir",
        default=f"./runs/lstm_marker_angles_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--input_cols",
        nargs="+",
        default=["current_pressure"],
        help="Input feature columns for the LSTM sequence.",
    )
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=30)
    ns = parser.parse_args()
    return Args(**vars(ns))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_dataframe_schema(df: pd.DataFrame) -> None:
    required = list(REQUIRED_BASE_COLUMNS)
    for marker_id in EXPECTED_MARKER_IDS:
        required.extend(
            [
                f"marker_{marker_id}_x",
                f"marker_{marker_id}_y",
                f"marker_{marker_id}_z",
                f"marker_{marker_id}_valid",
            ]
        )

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def split_continuous_df(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must satisfy 0 < train_ratio < 1")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


def load_and_split_csvs(
    csv_paths: Sequence[str],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for path in csv_paths:
        df = pd.read_csv(path)
        validate_dataframe_schema(df)
        train_df, val_df, test_df = split_continuous_df(df, train_ratio, val_ratio)
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def compute_angle_targets_deg(df: pd.DataFrame) -> np.ndarray:
    marker_xy = []
    for marker_id in EXPECTED_MARKER_IDS:
        cols = [f"marker_{marker_id}_x", f"marker_{marker_id}_y"]
        marker_xy.append(df[cols].astype(np.float32).to_numpy())

    marker_xy = np.stack(marker_xy, axis=1)  # [T, 6, 2]
    angles = []
    eps = 1e-8

    for left_idx, center_idx, right_idx in ANGLE_TRIPLES:
        v1 = marker_xy[:, left_idx] - marker_xy[:, center_idx]
        v2 = marker_xy[:, right_idx] - marker_xy[:, center_idx]

        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        denom = np.clip(norm1 * norm2, eps, None)
        cosine = np.sum(v1 * v2, axis=1) / denom
        cosine = np.clip(cosine, -1.0, 1.0)

        # Acute angle only: theta in [0, 90]
        acute_cosine = np.abs(cosine)
        angle_rad = np.arccos(np.clip(acute_cosine, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        zero_length_mask = (norm1 < eps) | (norm2 < eps)
        angle_deg[zero_length_mask] = 0.0
        angles.append(angle_deg)

    return np.stack(angles, axis=1).astype(np.float32)  # [T, 4]


@torch.no_grad()
def denormalize_predictions(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pred_deg = pred_norm * stats["y_std"] + stats["y_mean"]
    target_deg = target_norm * stats["y_std"] + stats["y_mean"]
    pred_deg = np.clip(pred_deg, 0.0, 90.0)
    target_deg = np.clip(target_deg, 0.0, 90.0)
    return pred_deg, target_deg


@torch.no_grad()
def evaluate(
    model: LSTMAngleRegressor,
    loader: DataLoader,
    device: torch.device,
    stats: Dict[str, np.ndarray],
    detailed: bool = False,
):
    model.eval()
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    total_loss = 0.0
    total_mae_deg = 0.0
    total_mse_deg = 0.0
    total_samples = 0

    if detailed:
        pred_norm_list = []
        target_norm_list = []
        pred_deg_list = []
        target_deg_list = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = mse(pred, y)

        pred_norm = pred.detach().cpu().numpy()
        target_norm = y.detach().cpu().numpy()
        pred_deg, target_deg = denormalize_predictions(pred_norm, target_norm, stats)

        batch_mae_deg = np.abs(pred_deg - target_deg).mean()
        batch_mse_deg = ((pred_deg - target_deg) ** 2).mean()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_mae_deg += batch_mae_deg * batch_size
        total_mse_deg += batch_mse_deg * batch_size
        total_samples += batch_size

        if detailed:
            pred_norm_list.append(pred_norm)
            target_norm_list.append(target_norm)
            pred_deg_list.append(pred_deg)
            target_deg_list.append(target_deg)

    result = {
        "loss": total_loss / total_samples,
        "mae_deg": total_mae_deg / total_samples,
        "mse_deg": total_mse_deg / total_samples,
        "rmse_deg": math.sqrt(total_mse_deg / total_samples),
    }

    if detailed:
        result.update(
            {
                "preds_norm": np.concatenate(pred_norm_list, axis=0),
                "targets_norm": np.concatenate(target_norm_list, axis=0),
                "preds_deg": np.concatenate(pred_deg_list, axis=0),
                "targets_deg": np.concatenate(target_deg_list, axis=0),
            }
        )

    return result


def train_one_epoch(
    model: LSTMAngleRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.train()
    mse = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = mse(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_metrics_json(results: Dict[str, np.ndarray], out_dir: str) -> Dict:
    preds_deg = results["preds_deg"]
    targets_deg = results["targets_deg"]
    abs_err = np.abs(preds_deg - targets_deg)

    metrics = {
        "overall": {
            "loss": float(results["loss"]),
            "mae_deg": float(results["mae_deg"]),
            "mse_deg": float(results["mse_deg"]),
            "rmse_deg": float(results["rmse_deg"]),
        },
        "per_angle": {},
    }

    for angle_idx, triple in enumerate(ANGLE_TRIPLES):
        angle_key = f"angle_{angle_idx}"
        metrics["per_angle"][angle_key] = {
            "marker_triplet": list(triple),
            "mae_deg": float(abs_err[:, angle_idx].mean()),
            "mse_deg": float(
                ((preds_deg[:, angle_idx] - targets_deg[:, angle_idx]) ** 2).mean()
            ),
            "rmse_deg": float(
                np.sqrt(
                    ((preds_deg[:, angle_idx] - targets_deg[:, angle_idx]) ** 2).mean()
                )
            ),
        }

    path = os.path.join(out_dir, "test_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def plot_training_history(
    train_losses: List[float], val_losses: List[float], out_dir: str
) -> None:
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "training_history.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_test_results(results: Dict[str, np.ndarray], out_dir: str) -> None:
    preds_deg = results["preds_deg"]
    targets_deg = results["targets_deg"]
    abs_err = np.abs(preds_deg - targets_deg)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Test Predictions vs Ground Truth (degrees)", fontsize=16)

    sample_count = min(500, len(preds_deg))
    indices = np.linspace(0, len(preds_deg) - 1, sample_count, dtype=int)

    for angle_idx, ax in enumerate(axes):
        triple = ANGLE_TRIPLES[angle_idx]
        ax.plot(
            indices,
            targets_deg[indices, angle_idx],
            label="ground_truth",
            linewidth=1.5,
        )
        ax.plot(
            indices, preds_deg[indices, angle_idx], label="prediction", linewidth=1.2
        )
        ax.set_ylabel(f"A{angle_idx} (deg)")
        ax.set_title(f"Angle {angle_idx}: markers {triple}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "test_predictions.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(4), abs_err.mean(axis=0))
    plt.xticks(np.arange(4), [f"A{i}" for i in range(4)])
    plt.ylabel("MAE (deg)")
    plt.title("Per-Angle MAE")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "test_angle_mae.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    sample_count = min(1000, len(preds_deg))
    indices = np.random.choice(len(preds_deg), size=sample_count, replace=False)
    plt.scatter(
        targets_deg[indices].reshape(-1),
        preds_deg[indices].reshape(-1),
        alpha=0.3,
        s=10,
    )
    min_val = min(targets_deg.min(), preds_deg.min())
    max_val = max(targets_deg.max(), preds_deg.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("Ground Truth (deg)")
    plt.ylabel("Prediction (deg)")
    plt.title("Prediction Scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: TrainConfig,
    input_cols: Sequence[str],
) -> Tuple[
    SlidingWindowAngleDataset,
    SlidingWindowAngleDataset,
    SlidingWindowAngleDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    train_ds = SlidingWindowAngleDataset(
        train_df,
        cfg,
        input_cols=input_cols,
        normalize_stats=None,
        fit_stats=True,
    )
    stats = train_ds.stats
    val_ds = SlidingWindowAngleDataset(
        val_df,
        cfg,
        input_cols=input_cols,
        normalize_stats=stats,
        fit_stats=False,
    )
    test_ds = SlidingWindowAngleDataset(
        test_df,
        cfg,
        input_cols=input_cols,
        normalize_stats=stats,
        fit_stats=False,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def hyperparameter_tuning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_cols: Sequence[str],
    base_cfg: TrainConfig,
    device: torch.device,
    n_trials: int,
) -> Dict[str, float]:
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run `pip install optuna` first.")

    def objective(trial: "optuna.Trial") -> float:
        cfg = TrainConfig(
            window=trial.suggest_int("window", 10, 120, step=10),
            horizon=trial.suggest_int("horizon", 1, 10),
            stride=trial.suggest_int("stride", 1, 5),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            num_workers=base_cfg.num_workers,
            epochs=trial.suggest_int("epochs", 15, 40),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
            num_layers=trial.suggest_int("num_layers", 1, 4),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            grad_clip=trial.suggest_float("grad_clip", 0.5, 2.0),
            seed=base_cfg.seed,
        )

        try:
            train_ds, _, _, train_loader, val_loader, _ = build_dataloaders(
                train_df, val_df, val_df, cfg, input_cols
            )
            model = LSTMAngleRegressor(
                input_dim=train_ds.input_dim,
                output_dim=train_ds.target_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

            best_val = float("inf")
            patience = 5
            patience_count = 0

            for epoch in range(1, cfg.epochs + 1):
                train_one_epoch(model, train_loader, optimizer, device, cfg)
                metrics = evaluate(
                    model, val_loader, device, train_ds.stats, detailed=False
                )
                val_loss = metrics["loss"]

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_loss < best_val:
                    best_val = val_loss
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= patience:
                    break

            return best_val
        except Exception as exc:
            print(f"[WARN] Trial failed: {exc}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def save_run_artifacts(
    out_dir: str,
    cfg: TrainConfig,
    args: Args,
    stats: Dict[str, np.ndarray],
    train_losses: List[float],
    val_losses: List[float],
    test_results: Dict[str, np.ndarray],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_config": cfg.to_dict(),
                "input_cols": args.input_cols,
                "csv_paths": args.csv_paths,
                "angle_triplets": [list(t) for t in ANGLE_TRIPLES],
                "angle_unit": "degree",
                "acute_angle_only": True,
                "xy_plane_only": True,
            },
            f,
            indent=2,
        )

    np.savez(
        os.path.join(out_dir, "norm_stats.npz"),
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
    )

    plot_training_history(train_losses, val_losses, out_dir)
    plot_test_results(test_results, out_dir)
    save_metrics_json(test_results, out_dir)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        window=args.window,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_df, val_df, test_df = load_and_split_csvs(
        args.csv_paths,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        f"[INFO] Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}"
    )
    print(f"[INFO] Input columns: {args.input_cols}")
    print(
        f"[INFO] Target: 4 acute angles in XY plane from marker triplets {ANGLE_TRIPLES}"
    )

    if args.tune:
        print("[INFO] Starting hyperparameter tuning...")
        best_params = hyperparameter_tuning(
            train_df=train_df,
            val_df=val_df,
            input_cols=args.input_cols,
            base_cfg=cfg,
            device=device,
            n_trials=args.n_trials,
        )
        for key, value in best_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        print(f"[INFO] Best params applied: {best_params}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = (
        build_dataloaders(
            train_df,
            val_df,
            test_df,
            cfg,
            args.input_cols,
        )
    )

    model = LSTMAngleRegressor(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.target_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(
            model, val_loader, device, train_ds.stats, detailed=False
        )
        val_loss = val_metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"[E{epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae_deg={val_metrics['mae_deg']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "stats": train_ds.stats,
            "cfg": cfg.to_dict(),
            "input_cols": args.input_cols,
            "angle_triplets": [list(t) for t in ANGLE_TRIPLES],
            "target_unit": "degree",
        }
        torch.save(checkpoint, os.path.join(args.out_dir, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.out_dir, "best.pt"))
            print(f"[INFO] Saved best checkpoint (val_loss={best_val_loss:.6f})")

    best_checkpoint = torch.load(
        os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])

    test_results = evaluate(model, test_loader, device, train_ds.stats, detailed=True)
    print("\n=== TEST RESULTS ===")
    print(f"Loss (normalized MSE): {test_results['loss']:.6f}")
    print(f"MAE (deg): {test_results['mae_deg']:.6f}")
    print(f"RMSE (deg): {test_results['rmse_deg']:.6f}")

    metrics = save_metrics_json(test_results, args.out_dir)
    plot_training_history(train_losses, val_losses, args.out_dir)
    plot_test_results(test_results, args.out_dir)

    np.savez(
        os.path.join(args.out_dir, "norm_stats.npz"),
        x_mean=train_ds.stats["x_mean"],
        x_std=train_ds.stats["x_std"],
        y_mean=train_ds.stats["y_mean"],
        y_std=train_ds.stats["y_std"],
    )
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_config": cfg.to_dict(),
                "input_cols": args.input_cols,
                "csv_paths": args.csv_paths,
                "angle_triplets": [list(t) for t in ANGLE_TRIPLES],
                "angle_unit": "degree",
                "acute_angle_only": True,
                "xy_plane_only": True,
            },
            f,
            indent=2,
        )

    print("\n=== PER-ANGLE METRICS ===")
    for angle_name, angle_metrics in metrics["per_angle"].items():
        print(
            f"{angle_name} | triplet={tuple(angle_metrics['marker_triplet'])} | "
            f"MAE={angle_metrics['mae_deg']:.4f} deg | RMSE={angle_metrics['rmse_deg']:.4f} deg"
        )

    print(f"\n[INFO] All results saved to {args.out_dir}")
    print("Generated files:")
    print("  - best.pt, last.pt")
    print("  - config.json")
    print("  - norm_stats.npz")
    print("  - test_metrics.json")
    print("  - training_history.png")
    print("  - test_predictions.png")
    print("  - test_angle_mae.png")
    print("  - test_scatter.png")


if __name__ == "__main__":
    main()


"""
python src/lstm/lstm/train_lstm_marker_angles.py \
  data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv data/rosbag2_2026_03_03-16_24_27/new_merged_data.csv \
  --input_cols current_pressure \
  --window 60 \
  --horizon 1 \
  --stride 1 \
  --epochs 1000 \
  --batch_size 512

"""
