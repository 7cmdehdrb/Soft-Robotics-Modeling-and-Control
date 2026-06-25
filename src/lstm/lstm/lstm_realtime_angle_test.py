#!/usr/bin/env python3
"""
ROS2 Node for real-time LSTM prediction of 4 acute XY-plane marker angles.

This node is adapted from the user's original real-time test node, but updated
for the new training target:
- No marker position output
- No valid/confidence output
- Predict 4 angles (degrees)
- Publish predicted angles as Float32MultiArray and String

Default expected CSV-like input column order:
    time, target_pressure, current_pressure, filtered_pressure, valve

Default topic assumption for /arduino_data (Float32MultiArray):
    data[0] -> time
    data[1] -> target_pressure
    data[2] -> current_pressure
    data[3] -> filtered_pressure
    data[4] -> valve

If your runtime topic layout differs, override INPUT_INDEX_MAP_JSON.
"""

# Standard library
import json
import os
import time
from collections import deque
from typing import Dict, List

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn

# ROS2 core
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

# ROS2 Messages
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String


DEFAULT_INPUT_INDEX_MAP: Dict[str, int] = {
    "time": 0,
    "target_pressure": 1,
    "current_pressure": 2,
    "filtered_pressure": 3,
    "valve": 4,
}

ANGLE_TRIPLES = ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5))


class LSTMAngleRegressor(nn.Module):
    """LSTM model for 4-angle prediction (must match training)."""

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


class LSTMAnglePredictorNode(Node):
    """ROS2 node for real-time LSTM angle prediction."""

    def __init__(self):
        super().__init__("lstm_angle_predictor")

        # ----- runtime configuration -----
        self.model_path = os.environ.get("MODEL_PATH", "runs/lstm_marker_angles/best.pt")
        self.input_index_map = self._load_input_index_map()
        self.publish_rate_hz = float(os.environ.get("PUBLISH_RATE_HZ", "30.0"))
        self.publish_text = os.environ.get("PUBLISH_TEXT", "1") != "0"
        self.publish_pretty_log_every = int(os.environ.get("LOG_EVERY_N", "100"))

        # ----- load checkpoint -----
        self.get_logger().info(f"Loading model from: {self.model_path}")
        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load checkpoint: {exc}")
            raise

        try:
            self.stats = checkpoint["stats"]
            self.cfg_dict = checkpoint["cfg"]
            self.input_cols: List[str] = checkpoint["input_cols"]
            self.angle_triplets = checkpoint.get(
                "angle_triplets", [list(t) for t in ANGLE_TRIPLES]
            )
        except KeyError as exc:
            self.get_logger().error(f"Checkpoint format is invalid. Missing key: {exc}")
            raise

        self.window_size = int(self.cfg_dict["window"])
        self.input_dim = len(self.input_cols)
        self.output_dim = 4

        self.model = LSTMAngleRegressor(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_size=int(self.cfg_dict["hidden_size"]),
            num_layers=int(self.cfg_dict["num_layers"]),
            dropout=float(self.cfg_dict["dropout"]),
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self._validate_stats()
        self._validate_input_mapping()

        self.get_logger().info(f"Model loaded successfully on {self.device}")
        self.get_logger().info(f"Window size: {self.window_size}")
        self.get_logger().info(f"Input columns: {self.input_cols}")
        self.get_logger().info(f"Input index map: {self.input_index_map}")
        self.get_logger().info(f"Angle triplets: {self.angle_triplets}")

        # Sliding window buffer: each item is [input_dim]
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.buffer_filled = False

        # ROS interfaces
        self.arduino_sub = self.create_subscription(
            Float32MultiArray,
            "/arduino_data",
            self.arduino_callback,
            qos_profile=qos_profile_system_default,
        )
        self.angle_pub = self.create_publisher(Float32MultiArray, "/predicted_angles_deg", 10)
        self.angle_text_pub = self.create_publisher(String, "/predicted_angles_text", 10)

        # Statistics
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.get_logger().info("LSTM Angle Predictor Node initialized")
        self.get_logger().info("Waiting for /arduino_data ...")

    def _load_input_index_map(self) -> Dict[str, int]:
        raw = os.environ.get("INPUT_INDEX_MAP_JSON", "")
        if not raw.strip():
            return dict(DEFAULT_INPUT_INDEX_MAP)

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("INPUT_INDEX_MAP_JSON must decode to a dict")
            return {str(k): int(v) for k, v in parsed.items()}
        except Exception as exc:
            raise RuntimeError(f"Invalid INPUT_INDEX_MAP_JSON: {exc}") from exc

    def _validate_stats(self) -> None:
        required_keys = ["x_mean", "x_std", "y_mean", "y_std"]
        for key in required_keys:
            if key not in self.stats:
                raise KeyError(f"stats['{key}'] is missing in checkpoint")

        x_mean = np.asarray(self.stats["x_mean"], dtype=np.float32)
        x_std = np.asarray(self.stats["x_std"], dtype=np.float32)
        y_mean = np.asarray(self.stats["y_mean"], dtype=np.float32)
        y_std = np.asarray(self.stats["y_std"], dtype=np.float32)

        if x_mean.shape != (self.input_dim,):
            raise ValueError(f"x_mean shape mismatch: {x_mean.shape} != {(self.input_dim,)}")
        if x_std.shape != (self.input_dim,):
            raise ValueError(f"x_std shape mismatch: {x_std.shape} != {(self.input_dim,)}")
        if y_mean.shape != (self.output_dim,):
            raise ValueError(f"y_mean shape mismatch: {y_mean.shape} != {(self.output_dim,)}")
        if y_std.shape != (self.output_dim,):
            raise ValueError(f"y_std shape mismatch: {y_std.shape} != {(self.output_dim,)}")

        self.stats["x_mean"] = x_mean
        self.stats["x_std"] = x_std
        self.stats["y_mean"] = y_mean
        self.stats["y_std"] = y_std

    def _validate_input_mapping(self) -> None:
        missing = [col for col in self.input_cols if col not in self.input_index_map]
        if missing:
            raise KeyError(
                "Input mapping is missing required keys for checkpoint input_cols: "
                f"{missing}. INPUT_INDEX_MAP_JSON={self.input_index_map}"
            )

    def arduino_callback(self, msg: Float32MultiArray) -> None:
        try:
            feature_vec = self._extract_feature_vector(msg)
            self.window_buffer.append(feature_vec)

            if len(self.window_buffer) == self.window_size and not self.buffer_filled:
                self.buffer_filled = True
                self.get_logger().info(
                    f"Window buffer filled ({self.window_size} samples, input_dim={self.input_dim})"
                )

            if self.buffer_filled:
                self.predict_and_publish()

        except Exception as exc:
            self.get_logger().error(f"Error in arduino_callback: {exc}")

    def _extract_feature_vector(self, msg: Float32MultiArray) -> np.ndarray:
        data = msg.data
        features: List[float] = []

        for col in self.input_cols:
            idx = self.input_index_map[col]
            if idx < 0 or idx >= len(data):
                raise IndexError(
                    f"Input '{col}' expects msg.data[{idx}], but len(data)={len(data)}"
                )
            features.append(float(data[idx]))

        return np.asarray(features, dtype=np.float32)

    def predict_and_publish(self) -> None:
        try:
            start_time = time.time()

            # [T, C]
            input_data = np.stack(list(self.window_buffer), axis=0).astype(np.float32)

            # normalize
            x_norm = (input_data - self.stats["x_mean"]) / self.stats["x_std"]
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float().to(self.device)  # [1,T,C]

            with torch.no_grad():
                pred_norm = self.model(x_tensor)  # [1,4]

            pred_deg = pred_norm.cpu().numpy().squeeze(0)
            pred_deg = pred_deg * self.stats["y_std"] + self.stats["y_mean"]
            pred_deg = np.clip(pred_deg, 0.0, 90.0).astype(np.float32)

            self._publish_angles(pred_deg)

            inference_time = time.time() - start_time
            self.prediction_count += 1
            self.total_inference_time += inference_time

            if self.prediction_count % self.publish_pretty_log_every == 0:
                avg_ms = 1000.0 * self.total_inference_time / self.prediction_count
                summary = ", ".join(
                    [
                        f"A{i}{tuple(self.angle_triplets[i])}={pred_deg[i]:.2f}deg"
                        for i in range(self.output_dim)
                    ]
                )
                self.get_logger().info(
                    f"Predictions={self.prediction_count}, avg_inference={avg_ms:.2f}ms | {summary}"
                )

        except Exception as exc:
            self.get_logger().error(f"Error in predict_and_publish: {exc}")

    def _publish_angles(self, pred_deg: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="angles", size=4, stride=4),
        ]
        msg.data = [float(v) for v in pred_deg]
        self.angle_pub.publish(msg)

        if self.publish_text:
            text = String()
            text.data = " | ".join(
                [
                    f"angle_{i} markers{tuple(self.angle_triplets[i])}: {pred_deg[i]:.3f} deg"
                    for i in range(self.output_dim)
                ]
            )
            self.angle_text_pub.publish(text)


def main(args=None):
    import threading

    rclpy.init(args=args)
    node = LSTMAnglePredictorNode()
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(node.publish_rate_hz)
        while rclpy.ok():
            r.sleep()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LSTM Angle Predictor Node...")
    except Exception as exc:
        print(f"Error | Exception: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        th.join()


if __name__ == "__main__":
    main()
