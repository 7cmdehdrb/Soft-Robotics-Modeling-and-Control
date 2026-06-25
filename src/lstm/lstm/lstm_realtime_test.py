#!/usr/bin/env python3
"""
ROS2 Node for Real-time LSTM prediction of marker positions.

Subscribes to: /arduino_data (std_msgs/Float32MultiArray)
Publishes to: /predicted_markers (visualization_msgs/MarkerArray)
"""

# Standard library
import os
from collections import deque
from typing import Optional

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn

# ROS2 core
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration

# ROS2 TF2
from tf2_ros import *


class LSTMRegressor(nn.Module):
    """LSTM model for marker position prediction (must match training)"""

    def __init__(
        self, pos_dim: int, valid_dim: int, hidden_size=128, num_layers=2, dropout=0.1
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.valid_dim = valid_dim

        self.lstm = nn.LSTM(
            input_size=1,
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
        # x: [B,T,1]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        pos = self.pos_head(h)
        valid_logits = self.valid_head(h)
        return pos, valid_logits


class LSTMPredictorNode(Node):
    """ROS2 node for real-time LSTM marker prediction"""

    def __init__(self):
        super().__init__("lstm_predictor")

        # Configuration variables
        model_path = "runs/lstm_20260304-164414/best.pt"
        self.window_size = 60
        self.init_point = np.array(
            [0.300912082195282, 0.3230445384979248, -0.446285218000412],
            dtype=np.float32,
        )
        self.marker_ids = [3, 4, 5, 6, 7]
        self.publish_rate = 30.0  # Hz
        self.confidence_threshold = 0.3

        # Load model and stats
        self.get_logger().info(f"Loading model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.stats = checkpoint["stats"]
            cfg_dict = checkpoint["cfg"]

            # Reconstruct model
            pos_dim = len(self.marker_ids) * 3
            valid_dim = len(self.marker_ids)

            self.model = LSTMRegressor(
                pos_dim,
                valid_dim,
                hidden_size=cfg_dict["hidden_size"],
                num_layers=cfg_dict["num_layers"],
                dropout=cfg_dict["dropout"],
            )
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()

            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)

            self.get_logger().info(f"Model loaded successfully on {self.device}")
            self.get_logger().info(f"Window size: {self.window_size}")
            self.get_logger().info(f"Marker IDs: {self.marker_ids}")
            self.get_logger().info(f"Init point: {self.init_point}")

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        # Sliding window buffer
        self.window_buffer = deque(maxlen=self.window_size)
        self.buffer_filled = False

        # Create subscriber
        self.arduino_sub = self.create_subscription(
            Float32MultiArray,
            "/arduino_data",
            self.arduino_callback,
            qos_profile=qos_profile_system_default,
        )

        # Create publisher
        self.marker_pub = self.create_publisher(MarkerArray, "/predicted_markers", 10)

        # Statistics
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.get_logger().info("LSTM Predictor Node initialized")
        self.get_logger().info("Waiting for arduino_data...")

    def arduino_callback(self, msg: Float32MultiArray):
        """Callback for arduino_data topic"""
        try:
            # Extract data[1] as input
            if len(msg.data) < 2:
                self.get_logger().warn("Arduino data has insufficient elements")
                return

            input_value = float(msg.data[1])

            # Add to sliding window
            self.window_buffer.append(input_value)

            # Check if buffer is filled
            if len(self.window_buffer) == self.window_size and not self.buffer_filled:
                self.buffer_filled = True
                self.get_logger().info(
                    f"Window buffer filled ({self.window_size} samples)"
                )

            # Perform prediction if buffer is full
            if self.buffer_filled:
                self.predict_and_publish()

        except Exception as e:
            self.get_logger().error(f"Error in arduino_callback: {e}")

    def predict_and_publish(self):
        """Run LSTM prediction and publish results"""
        try:
            import time

            start_time = time.time()

            # Convert buffer to numpy array
            input_data = np.array(list(self.window_buffer), dtype=np.float32)

            # Normalize
            x_norm = (input_data - self.stats["x_mean"]) / self.stats["x_std"]
            x_tensor = (
                torch.from_numpy(x_norm[:, None]).unsqueeze(0).float().to(self.device)
            )

            # Predict
            with torch.no_grad():
                pred_pos, pred_valid_logits = self.model(x_tensor)
                pred_pos: torch.Tensor

            # Denormalize position prediction
            pred_pos = pred_pos.cpu().numpy().squeeze(0)  # [15]
            pred_pos = pred_pos * self.stats["y_std_pos"] + self.stats["y_mean"]
            pred_pos = pred_pos.reshape(len(self.marker_ids), 3)  # [5, 3]

            # Get valid predictions (confidence)
            pred_valid = (
                torch.sigmoid(pred_valid_logits).cpu().numpy().squeeze(0)
            )  # [5]

            # Convert relative predictions to absolute positions (w.r.t init_point)
            pred_abs = pred_pos + self.init_point[None, :]  # [5, 3]

            # Create and publish MarkerArray
            marker_array = MarkerArray()
            current_time = self.get_clock().now().to_msg()

            marker_array.markers.append(
                Marker(
                    header=Header(
                        stamp=self.get_clock().now().to_msg(),
                        frame_id="opti_world",
                    ),
                    ns="predicted_markers",
                    id=999,  # Temporary marker for debugging
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    pose=Pose(
                        position=Point(
                            x=float(self.init_point[0]),
                            y=float(self.init_point[1]),
                            z=float(self.init_point[2]),
                        )
                    ),
                    scale=Vector3(x=0.01, y=0.01, z=0.01),
                    color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.4),
                    lifetime=BuiltinDuration(sec=0, nanosec=100000000),
                )
            )

            for i, marker_id in enumerate(self.marker_ids):
                if pred_valid[i] > self.confidence_threshold:

                    marker = Marker(
                        header=Header(
                            stamp=self.get_clock().now().to_msg(),
                            frame_id="opti_world",
                        ),
                        ns="predicted_markers",
                        id=marker_id,
                        type=Marker.SPHERE,
                        action=Marker.ADD,
                        pose=Pose(
                            position=Point(
                                x=float(pred_abs[i][0]),
                                y=float(pred_abs[i][1]),
                                z=float(pred_abs[i][2]),
                            )
                        ),
                        scale=Vector3(x=0.01, y=0.01, z=0.01),
                        color=ColorRGBA(
                            r=0.0, g=0.0, b=1.0, a=float(pred_valid[i]) * 0.4
                        ),
                        lifetime=BuiltinDuration(sec=0, nanosec=100000000),
                    )

                    marker_array.markers.append(marker)

            # Publish
            self.marker_pub.publish(marker_array)

            # Update statistics
            inference_time = time.time() - start_time
            self.prediction_count += 1
            self.total_inference_time += inference_time

            # Log statistics periodically
            if self.prediction_count % 100 == 0:
                avg_time = self.total_inference_time / self.prediction_count
                self.get_logger().info(
                    f"Predictions: {self.prediction_count}, "
                    f"Avg inference time: {avg_time*1000:.2f}ms, "
                    f"Markers published: {len(marker_array.markers)}"
                )

        except Exception as e:
            self.get_logger().error(f"Error in predict_and_publish: {e}")

    def set_init_point(self, x: float, y: float, z: float):
        """Update the init_point (origin) for predictions"""
        self.init_point = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f"Init point updated to: [{x}, {y}, {z}]")


def main(args=None):
    import threading

    rclpy.init(args=args)

    node = LSTMPredictorNode()
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(30.0)
        while rclpy.ok():
            r.sleep()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LSTM Predictor Node...")
    except Exception as e:
        print(f"Error | Exception: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        th.join()


if __name__ == "__main__":
    main()
