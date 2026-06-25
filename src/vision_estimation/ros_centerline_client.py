from __future__ import annotations

import argparse
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from app import (
    HSV_PRESETS,
    HSVRange,
    estimate_centerline_from_image,
    make_debug_grid,
    parse_hsv_triplet,
)


class GenesisImageCenterlineClient(Node):
    def __init__(self, args: argparse.Namespace, hsv_range: HSVRange):
        super().__init__("genesis_image_centerline_client")
        self.args = args
        self.hsv_range = hsv_range
        self._lock = threading.Lock()
        self._latest_image_bgr: np.ndarray | None = None
        self._latest_stamp = 0.0
        self._last_processed_stamp = 0.0
        self._last_status_time = 0.0

        self.create_subscription(Image, args.topic, self._image_callback, 5)
        self.create_timer(1.0 / max(args.process_hz, 0.1), self._process_latest)

        self.fig = None
        self.ax = None
        self.im = None
        if args.show:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.axis("off")
            self.fig.canvas.manager.set_window_title("Genesis Centerline Debug Grid")

        self.get_logger().info(f"Subscribing image: {args.topic}")
        self.get_logger().info(
            f"HSV lower={hsv_range.lower}, upper={hsv_range.upper}"
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            image_bgr = ros_image_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        with self._lock:
            self._latest_image_bgr = image_bgr
            self._latest_stamp = time.time()

    def _process_latest(self) -> None:
        with self._lock:
            if self._latest_image_bgr is None:
                return
            if self._latest_stamp == self._last_processed_stamp:
                return
            image_bgr = self._latest_image_bgr.copy()
            self._last_processed_stamp = self._latest_stamp

        try:
            result = estimate_centerline_from_image(
                image_bgr=image_bgr,
                hsv_range=self.hsv_range,
                backbone_points=self.args.points,
                epochs=self.args.epochs,
                alpha0=self.args.alpha,
                radius0=self.args.radius,
                canny_low=self.args.canny_low,
                canny_high=self.args.canny_high,
            )
            grid_bgr = make_debug_grid(result)
            status = (
                f"mask_px={int(np.count_nonzero(result.mask))} "
                f"contour={len(result.contour_points)} "
                f"centerline={np.round(result.som_points, 1).tolist()}"
            )
        except RuntimeError as exc:
            grid_bgr = make_failure_grid(image_bgr, str(exc))
            status = f"centerline failed: {exc}"

        now = time.time()
        if now - self._last_status_time >= self.args.log_period:
            self._last_status_time = now
            self.get_logger().info(status)

        if self.args.show:
            self._show_grid(grid_bgr)

    def _show_grid(self, grid_bgr: np.ndarray) -> None:
        grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
        if self.im is None:
            self.im = self.ax.imshow(grid_rgb)
            self.fig.tight_layout()
        else:
            self.im.set_data(grid_rgb)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("Received empty image.")

    encoding = msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        channels = 3
        min_step = int(msg.width * channels)
        step = int(msg.step) if msg.step else min_step
        expected = int(msg.height * step)
        if step < min_step or data.size < expected:
            raise ValueError(
                f"Image data too small for {msg.encoding}: got {data.size}, expected {expected}"
            )
        image = data[:expected].reshape((msg.height, step))[:, :min_step]
        image = image.reshape((msg.height, msg.width, channels))
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    if encoding in ("rgba8", "bgra8"):
        channels = 4
        min_step = int(msg.width * channels)
        step = int(msg.step) if msg.step else min_step
        expected = int(msg.height * step)
        if step < min_step or data.size < expected:
            raise ValueError(
                f"Image data too small for {msg.encoding}: got {data.size}, expected {expected}"
            )
        image = data[:expected].reshape((msg.height, step))[:, :min_step]
        image = image.reshape((msg.height, msg.width, channels))
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if encoding in ("mono8", "8uc1"):
        min_step = int(msg.width)
        step = int(msg.step) if msg.step else min_step
        expected = int(msg.height * step)
        if step < min_step or data.size < expected:
            raise ValueError(
                f"Image data too small for {msg.encoding}: got {data.size}, expected {expected}"
            )
        image = data[:expected].reshape((msg.height, step))[:, :min_step]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def make_failure_grid(image_bgr: np.ndarray, message: str) -> np.ndarray:
    display = image_bgr.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1] - 1, 56), (20, 20, 20), -1)
    cv2.putText(
        display,
        message[:120],
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to a Genesis ROS2 image and extract a real-time sky-blue soft robot centerline.",
    )
    parser.add_argument("--topic", default="/genesis/side_camera/image_preview")
    parser.add_argument(
        "--target",
        choices=sorted(HSV_PRESETS),
        default="skyblue",
    )
    parser.add_argument(
        "--hsv-lower", type=parse_hsv_triplet, help="Manual lower HSV as H,S,V."
    )
    parser.add_argument(
        "--hsv-upper", type=parse_hsv_triplet, help="Manual upper HSV as H,S,V."
    )
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--process-hz", type=float, default=5.0)
    parser.add_argument("--log-period", type=float, default=1.0)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("--points must be at least 2.")
    if bool(args.hsv_lower) != bool(args.hsv_upper):
        raise ValueError("--hsv-lower and --hsv-upper must be provided together.")

    hsv_range = (
        HSVRange(args.hsv_lower, args.hsv_upper)
        if args.hsv_lower and args.hsv_upper
        else HSV_PRESETS[args.target]
    )

    rclpy.init()
    node = GenesisImageCenterlineClient(args, hsv_range)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
