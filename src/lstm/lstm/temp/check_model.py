import numpy as np
import torch
import torch.nn as nn


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


class LSTMAnglePredictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.cfg = ckpt["cfg"]
        self.stats = ckpt["stats"]
        self.input_cols = list(ckpt.get("input_cols", ["current_pressure"]))
        self.angle_triplets = ckpt.get(
            "angle_triplets",
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        )

        self.window = int(self.cfg["window"])
        self.in_dim = len(self.input_cols)
        self.out_dim = len(self.angle_triplets)

        self.x_mean, self.x_std = self._resolve_input_stats(self.stats, self.in_dim)
        self.y_mean, self.y_std = self._resolve_output_stats(self.stats, self.out_dim)

        self.model = LSTMAngleRegressor(
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            hidden_size=int(self.cfg["hidden_size"]),
            num_layers=int(self.cfg["num_layers"]),
            dropout=float(self.cfg["dropout"]),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _pick_first_existing_key(self, d: dict, candidates):
        for k in candidates:
            if k in d:
                return d[k]
        return None

    def _resolve_input_stats(self, stats: dict, in_dim: int):
        x_mean = self._pick_first_existing_key(
            stats,
            ["in_mean", "x_mean", "input_mean"],
        )
        x_std = self._pick_first_existing_key(
            stats,
            ["in_std", "x_std", "input_std"],
        )

        if x_mean is None or x_std is None:
            raise KeyError(
                f"input normalization stats not found. available keys: {list(stats.keys())}"
            )

        x_mean = np.asarray(x_mean, dtype=np.float32).reshape(-1)
        x_std = np.asarray(x_std, dtype=np.float32).reshape(-1)

        if x_mean.size == 1 and in_dim > 1:
            x_mean = np.repeat(x_mean, in_dim)
        if x_std.size == 1 and in_dim > 1:
            x_std = np.repeat(x_std, in_dim)

        if x_mean.size != in_dim or x_std.size != in_dim:
            raise ValueError(
                f"input stats size mismatch: expected {in_dim}, got mean={x_mean.shape}, std={x_std.shape}"
            )

        return x_mean.reshape(1, in_dim), x_std.reshape(1, in_dim)

    def _resolve_output_stats(self, stats: dict, out_dim: int):
        y_mean = self._pick_first_existing_key(
            stats,
            ["out_mean", "y_mean", "target_mean"],
        )
        y_std = self._pick_first_existing_key(
            stats,
            ["out_std", "y_std", "y_std_pos", "target_std"],
        )

        if y_mean is None or y_std is None:
            raise KeyError(
                f"output normalization stats not found. available keys: {list(stats.keys())}"
            )

        y_mean = np.asarray(y_mean, dtype=np.float32).reshape(-1)
        y_std = np.asarray(y_std, dtype=np.float32).reshape(-1)

        if y_mean.size == 1 and out_dim > 1:
            y_mean = np.repeat(y_mean, out_dim)
        if y_std.size == 1 and out_dim > 1:
            y_std = np.repeat(y_std, out_dim)

        if y_mean.size != out_dim or y_std.size != out_dim:
            raise ValueError(
                f"output stats size mismatch: expected {out_dim}, got mean={y_mean.shape}, std={y_std.shape}"
            )

        return y_mean, y_std

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / (self.x_std + 1e-8)

    def _denormalize_output(self, y: np.ndarray) -> np.ndarray:
        return y * (self.y_std + 1e-8) + self.y_mean

    def _to_numpy_window(self, x_window) -> np.ndarray:
        x = np.asarray(x_window, dtype=np.float32)

        if x.ndim == 1:
            if self.in_dim != 1:
                raise ValueError(
                    f"1D input is allowed only when input_dim=1, but input_dim={self.in_dim}"
                )
            x = x.reshape(-1, 1)

        if x.ndim != 2:
            raise ValueError(
                f"x_window must be 2D with shape ({self.window}, {self.in_dim}), got {x.shape}"
            )

        if x.shape != (self.window, self.in_dim):
            raise ValueError(
                f"x_window must have shape ({self.window}, {self.in_dim}), got {x.shape}"
            )

        if not np.isfinite(x).all():
            raise ValueError("x_window contains NaN or Inf")

        return x

    def predict(self, x_window) -> np.ndarray:
        x = self._to_numpy_window(x_window)
        x = self._normalize_input(x)

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y = self.model(x_tensor)

        y = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
        y = self._denormalize_output(y)
        y = np.clip(y, 0.0, 90.0)

        return y


predictor = LSTMAnglePredictor(
    "/home/min/project_SORO/runs/lstm_marker_angles_20260326-183235/best.pt",
    device="cuda",
)

# input_cols == ["current_pressure"] 인 경우
x_window = np.random.randn(predictor.window).astype(np.float32)

print(type(x_window), x_window.shape)

y = predictor.predict(x_window)
print(y)
print(type(y), y.shape)
