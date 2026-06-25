# train_lstm_autoregressive.py
# State-Autoregressive LSTM: [State(t), Pressure(t+1)] -> State(t+1)
# Center point: marker_0 (relative positions marker_1..5 w.r.t marker_0)

import argparse
import os
import json
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] optuna not available. Install with: pip install optuna")


@dataclass
class TrainConfig:
    window: int = 2  # 2: last state (t-1) & current state (t) 를 봄
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
    if (
        not (0 < train_ratio < 1)
        or not (0 <= val_ratio < 1)
        or (train_ratio + val_ratio >= 1)
    ):
        raise ValueError("Invalid ratios.")
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


class AutoregressiveDataset(Dataset):
    """
    Input (X): Sequence of [State_k, Pressure_{k+1}] for k in [t, t+window-1]
    Target (Y): State_{t+window}
    State: marker_1..5 relative to marker_0 (15 dims)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: TrainConfig,
        input_col: str,
        center_xyz_cols: Tuple[str, str, str],
        out_marker_ids: List[int],  # [1, 2, 3, 4, 5]
        normalize_stats: Optional[dict] = None,
        fit_stats: bool = False,
    ):
        self.cfg = cfg
        self.window = cfg.window
        self.stride = cfg.stride

        # Extract features
        pos_cols = []
        for mid in out_marker_ids:
            pos_cols.extend([f"marker_{mid}_x", f"marker_{mid}_y", f"marker_{mid}_z"])

        pressure = df[input_col].astype(np.float32).to_numpy()  # [T]
        center = df[list(center_xyz_cols)].astype(np.float32).to_numpy()  # [T, 3]
        out_pos = df[pos_cols].astype(np.float32).to_numpy()  # [T, 15]

        # Calculate relative positions
        out_pos = out_pos.reshape(len(df), len(out_marker_ids), 3)
        rel_pos = out_pos - center[:, None, :]  # [T, 5, 3]
        state = rel_pos.reshape(len(df), len(out_marker_ids) * 3)  # [T, 15]

        self.pressure = pressure
        self.state = state

        # We need sequence up to t+window. Max start index:
        self.max_start = len(df) - self.window - 1
        self.starts = np.arange(0, self.max_start + 1, self.stride, dtype=np.int64)

        if fit_stats or normalize_stats is None:
            normalize_stats = {
                "p_mean": float(pressure.mean()),
                "p_std": float(pressure.std() + 1e-8),
                "s_mean": state.mean(axis=0).astype(np.float32),
                "s_std": (state.std(axis=0) + 1e-8).astype(np.float32),
            }
        self.stats = normalize_stats

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        w = self.window

        # State sequence: S_t ~ S_{t+w-1}
        s_seq = self.state[i : i + w]
        # Pressure sequence: P_{t+1} ~ P_{t+w}
        p_seq = self.pressure[i + 1 : i + w + 1]
        # Target state: S_{t+w}
        target_s = self.state[i + w]

        # Normalize
        s_seq = (s_seq - self.stats["s_mean"]) / self.stats["s_std"]
        p_seq = (p_seq - self.stats["p_mean"]) / self.stats["p_std"]
        target_s = (target_s - self.stats["s_mean"]) / self.stats["s_std"]

        # Concat State and Pressure -> [w, 16]
        p_seq = p_seq[:, None]  # [w, 1]
        x_seq = np.concatenate([s_seq, p_seq], axis=1).astype(np.float32)

        return torch.from_numpy(x_seq), torch.from_numpy(target_s.astype(np.float32))


class LSTMStateRegressor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        pressure_dim: int = 1,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        input_size = state_dim + pressure_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, x):
        # x: [B, Window, 16]
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # Use last hidden state
        pred_state = self.head(h)
        return pred_state


@torch.no_grad()
def evaluate(model, loader, device, detailed: bool = False):
    model.eval()
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    tot_loss, tot_mae, n = 0.0, 0.0, 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = mse_fn(pred, y)
        mae = mae_fn(pred, y)

        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_mae += mae.item() * bs
        n += bs

        if detailed:
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    res = {
        "mse": tot_loss / n,
        "mae": tot_mae / n,
        "rmse": np.sqrt(tot_loss / n),
    }
    if detailed:
        res["preds"] = np.concatenate(all_preds, axis=0)
        res["targets"] = np.concatenate(all_targets, axis=0)
    return res


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    mse_fn = nn.MSELoss()
    tot_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = mse_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x.size(0)
        tot_loss += loss.item() * bs
        n += bs
    return tot_loss / n


def visualize_test_results(results: Dict, out_dir: str, num_markers: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    preds = results["preds"]
    targets = results["targets"]

    fig, axes = plt.subplots(num_markers, 3, figsize=(15, 3 * num_markers))
    fig.suptitle("Test Set: Predicted vs Actual (Relative States)", fontsize=16)

    for i in range(num_markers):
        for j, coord in enumerate(["X", "Y", "Z"]):
            ax = axes[i, j] if num_markers > 1 else axes[j]
            idx = i * 3 + j
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
            ax.set_title(f"Marker {i+1} - {coord}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "test_predictions.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def hyperparameter_tuning(
    train_df, val_df, input_col, center_cols, out_marker_ids, device, n_trials=50
):
    if not OPTUNA_AVAILABLE:
        return {}, None

    def objective(trial):
        cfg = TrainConfig(
            window=trial.suggest_int("window", 2, 10),  # Window 최적화 범위
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            epochs=trial.suggest_int("epochs", 20, 40),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
        )

        train_ds = AutoregressiveDataset(
            train_df, cfg, input_col, center_cols, out_marker_ids, fit_stats=True
        )
        val_ds = AutoregressiveDataset(
            val_df,
            cfg,
            input_col,
            center_cols,
            out_marker_ids,
            normalize_stats=train_ds.stats,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model = LSTMStateRegressor(
            state_dim=len(out_marker_ids) * 3,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

        best_val = float("inf")
        for ep in range(1, cfg.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, cfg)
            val_res = evaluate(model, val_loader, device)
            best_val = min(best_val, val_res["mse"])
            trial.report(val_res["mse"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=f"./runs/lstm_ar_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    ap.add_argument(
        "--window", type=int, default=2, help="Sequence length (e.g. 2 for t-1, t)"
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tune", action="store_true")
    args, _ = ap.parse_known_args()

    cfg = TrainConfig(window=args.window, epochs=args.epochs)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading
    df1 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"
    )
    df2 = pd.read_csv(
        "/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/new_merged_data.csv"
    )

    center_cols = ("marker_0_x", "marker_0_y", "marker_0_z")
    out_marker_ids = [1, 2, 3, 4, 5]

    t1, v1, te1 = split_continuous_df(df1)
    t2, v2, te2 = split_continuous_df(df2)
    train_df = pd.concat([t1, t2], ignore_index=True)
    val_df = pd.concat([v1, v2], ignore_index=True)
    test_df = pd.concat([te1, te2], ignore_index=True)

    if args.tune:
        best_params, _ = hyperparameter_tuning(
            train_df, val_df, "current_pressure", center_cols, out_marker_ids, device
        )
        for k, v in best_params.items():
            setattr(cfg, k, v)

    # Datasets
    train_ds = AutoregressiveDataset(
        train_df, cfg, "current_pressure", center_cols, out_marker_ids, fit_stats=True
    )
    val_ds = AutoregressiveDataset(
        val_df,
        cfg,
        "current_pressure",
        center_cols,
        out_marker_ids,
        normalize_stats=train_ds.stats,
    )
    test_ds = AutoregressiveDataset(
        test_df,
        cfg,
        "current_pressure",
        center_cols,
        out_marker_ids,
        normalize_stats=train_ds.stats,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = LSTMStateRegressor(
        state_dim=len(out_marker_ids) * 3,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for ep in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_res = evaluate(model, val_loader, device)

        if val_res["mse"] < best_val:
            best_val = val_res["mse"]
            torch.save(
                {"model": model.state_dict(), "stats": train_ds.stats},
                os.path.join(args.out_dir, "best.pt"),
            )

    # Evaluation
    checkpoint = torch.load(os.path.join(args.out_dir, "best.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_res = evaluate(model, test_loader, device, detailed=True)

    visualize_test_results(test_res, args.out_dir, num_markers=len(out_marker_ids))
    print(
        f"\n[TEST] MSE: {test_res['mse']:.6f} | MAE: {test_res['mae']:.6f} | RMSE: {test_res['rmse']:.6f}"
    )
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
