from __future__ import annotations

import argparse
import math
import struct
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from app import (
    HSVRange,
    estimate_centerline_from_image,
    make_debug_grid,
    parse_hsv_triplet,
)


WHITE_ROBOT_HSV = HSVRange((0, 0, 150), (179, 80, 255))


class WhiteImageCenterlineClient(Node):
    def __init__(self, args: argparse.Namespace, hsv_range: HSVRange):
        super().__init__("white_image_centerline_client")
        self.args = args
        self.hsv_range = hsv_range
        self._lock = threading.Lock()
        self._latest_image_bgr: np.ndarray | None = None
        self._latest_image_msg_stamp = None
        self._latest_stamp = 0.0
        self._latest_cloud: PointCloud2 | None = None
        self._latest_cloud_stamp = 0.0
        self._last_processed_stamp = 0.0
        self._last_status_time = 0.0

        self.create_subscription(Image, args.topic, self._image_callback, 5)
        self.create_subscription(
            PointCloud2, args.pointcloud_topic, self._pointcloud_callback, 5
        )
        self.marker_pub = self.create_publisher(MarkerArray, args.marker_topic, 10)
        self.pose_array_pub = self.create_publisher(
            PoseArray, args.pose_array_topic, 10
        )
        self.create_timer(1.0 / max(args.process_hz, 0.1), self._process_latest)

        self.fig = None
        self.ax = None
        self.im = None
        if args.show:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.axis("off")
            self.fig.canvas.manager.set_window_title("White Centerline Debug Grid")

        self.get_logger().info(f"Subscribing image: {args.topic}")
        self.get_logger().info(f"Subscribing pointcloud: {args.pointcloud_topic}")
        self.get_logger().info(f"Publishing 3D markers: {args.marker_topic}")
        self.get_logger().info(f"Publishing 3D poses: {args.pose_array_topic}")
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
            self._latest_image_msg_stamp = msg.header.stamp
            self._latest_stamp = time.time()

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        with self._lock:
            self._latest_cloud = msg
            self._latest_cloud_stamp = time.time()

    def _process_latest(self) -> None:
        with self._lock:
            if self._latest_image_bgr is None:
                return
            if self._latest_stamp == self._last_processed_stamp:
                return
            image_bgr = self._latest_image_bgr.copy()
            image_msg_stamp = self._latest_image_msg_stamp
            cloud = self._latest_cloud
            cloud_age = time.time() - self._latest_cloud_stamp
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
            if cloud is not None and cloud_age <= self.args.max_cloud_age:
                try:
                    points_3d = centerline_pixels_to_3d(
                        cloud=cloud,
                        pixels_xy=result.som_points,
                        image_shape=image_bgr.shape[:2],
                        search_radius=self.args.point_search_radius,
                        allow_zero_points=self.args.allow_zero_points,
                    )
                    valid_points = points_3d[np.isfinite(points_3d).all(axis=1)]
                    if len(valid_points) > 0:
                        self._publish_centerline_3d(
                            points_3d=points_3d,
                            stamp=image_msg_stamp or cloud.header.stamp,
                            frame_id=cloud.header.frame_id,
                        )
                    status += (
                        f" centerline_3d_valid={len(valid_points)}/{len(points_3d)}"
                    )
                except RuntimeError as exc:
                    status += f" centerline_3d_failed={exc}"
            elif cloud is None:
                status += " centerline_3d=waiting_for_pointcloud"
            else:
                status += f" centerline_3d=stale_cloud({cloud_age:.2f}s)"
        except RuntimeError as exc:
            grid_bgr = make_failure_grid(image_bgr, str(exc))
            status = f"centerline failed: {exc}"

        now = time.time()
        if now - self._last_status_time >= self.args.log_period:
            self._last_status_time = now
            self.get_logger().info(status)

        if self.args.show:
            self._show_grid(grid_bgr)

    def _publish_centerline_3d(
        self, points_3d: np.ndarray, stamp, frame_id: str
    ) -> None:
        header = self._make_header(stamp, frame_id)

        pose_array = PoseArray()
        pose_array.header = header
        for point in points_3d:
            if not np.isfinite(point).all():
                continue
            pose = Pose()
            pose.position = Point(
                x=float(point[0]), y=float(point[1]), z=float(point[2])
            )
            pose.orientation = Quaternion(w=1.0)
            pose_array.poses.append(pose)
        self.pose_array_pub.publish(pose_array)

        marker_array = MarkerArray()
        marker_array.markers.append(
            make_delete_marker(header=header, namespace="white_centerline_3d")
        )

        line_points: list[Point] = []
        for idx, point in enumerate(points_3d):
            if not np.isfinite(point).all():
                continue
            ros_point = Point(
                x=float(point[0]), y=float(point[1]), z=float(point[2])
            )
            line_points.append(ros_point)
            marker_array.markers.append(
                make_sphere_marker(
                    header=header,
                    namespace="white_centerline_3d",
                    marker_id=idx,
                    point=ros_point,
                    diameter=self.args.marker_diameter,
                    color=ColorRGBA(r=1.0, g=0.2, b=0.05, a=0.95),
                    lifetime_sec=self.args.marker_lifetime,
                )
            )

        if len(line_points) >= 2:
            marker_array.markers.append(
                make_line_marker(
                    header=header,
                    namespace="white_centerline_3d",
                    marker_id=10_000,
                    points=line_points,
                    width=self.args.line_width,
                    color=ColorRGBA(r=0.05, g=0.9, b=1.0, a=0.95),
                    lifetime_sec=self.args.marker_lifetime,
                )
            )

        self.marker_pub.publish(marker_array)

    def _make_header(self, stamp, frame_id: str) -> Header:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

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


def centerline_pixels_to_3d(
    cloud: PointCloud2,
    pixels_xy: np.ndarray,
    image_shape: tuple[int, int],
    search_radius: int,
    allow_zero_points: bool,
) -> np.ndarray:
    image_height, image_width = image_shape
    if cloud.height <= 0 or cloud.width <= 0:
        raise RuntimeError("Received empty PointCloud2.")
    if cloud.height == 1 and image_height > 1:
        raise RuntimeError(
            "PointCloud2 is not organized; enable an ordered/aligned point cloud."
        )
    if cloud.row_step <= 0 or cloud.point_step <= 0:
        raise RuntimeError("PointCloud2 has invalid row_step or point_step.")

    unpacker = PointCloudXYZUnpacker(cloud)
    points = np.full((len(pixels_xy), 3), np.nan, dtype=np.float32)

    for idx, pixel in enumerate(pixels_xy):
        u = scale_pixel_coordinate(pixel[0], image_width, cloud.width)
        v = scale_pixel_coordinate(pixel[1], image_height, cloud.height)
        point = sample_cloud_xyz(
            cloud=cloud,
            unpacker=unpacker,
            u=u,
            v=v,
            search_radius=search_radius,
            allow_zero_points=allow_zero_points,
        )
        if point is not None:
            points[idx] = point

    return points


class PointCloudXYZUnpacker:
    def __init__(self, cloud: PointCloud2):
        fields = {field.name: field for field in cloud.fields}
        missing = [name for name in ("x", "y", "z") if name not in fields]
        if missing:
            raise RuntimeError(f"PointCloud2 is missing fields: {missing}")

        self.offsets = []
        for name in ("x", "y", "z"):
            field = fields[name]
            if field.datatype != PointField.FLOAT32:
                raise RuntimeError(f"PointCloud2 field {name} must be FLOAT32.")
            self.offsets.append(field.offset)

        endian = ">" if cloud.is_bigendian else "<"
        self.structs = [struct.Struct(f"{endian}f") for _ in self.offsets]

    def xyz_at(self, cloud: PointCloud2, u: int, v: int) -> np.ndarray:
        base = int(v * cloud.row_step + u * cloud.point_step)
        data = cloud.data
        return np.array(
            [
                self.structs[i].unpack_from(data, base + self.offsets[i])[0]
                for i in range(3)
            ],
            dtype=np.float32,
        )


def scale_pixel_coordinate(value: float, image_size: int, cloud_size: int) -> int:
    if image_size <= 1:
        return 0
    scaled = float(value) * float(cloud_size - 1) / float(image_size - 1)
    return int(np.clip(round(scaled), 0, cloud_size - 1))


def sample_cloud_xyz(
    cloud: PointCloud2,
    unpacker: PointCloudXYZUnpacker,
    u: int,
    v: int,
    search_radius: int,
    allow_zero_points: bool,
) -> np.ndarray | None:
    point = unpacker.xyz_at(cloud, u, v)
    if is_valid_xyz(point, allow_zero_points):
        return point

    radius = max(0, int(search_radius))
    best_point = None
    best_distance = math.inf
    for dv in range(-radius, radius + 1):
        vv = v + dv
        if vv < 0 or vv >= cloud.height:
            continue
        for du in range(-radius, radius + 1):
            uu = u + du
            if uu < 0 or uu >= cloud.width:
                continue
            distance = du * du + dv * dv
            if distance >= best_distance:
                continue
            candidate = unpacker.xyz_at(cloud, uu, vv)
            if is_valid_xyz(candidate, allow_zero_points):
                best_point = candidate
                best_distance = distance

    return best_point


def is_valid_xyz(point: np.ndarray, allow_zero_points: bool) -> bool:
    if not np.isfinite(point).all():
        return False
    if not allow_zero_points and np.linalg.norm(point) < 1e-9:
        return False
    return True


def make_delete_marker(header, namespace: str) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = 0
    marker.action = Marker.DELETEALL
    return marker


def make_sphere_marker(
    header,
    namespace: str,
    marker_id: int,
    point: Point,
    diameter: float,
    color: ColorRGBA,
    lifetime_sec: float,
) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = point
    marker.pose.orientation.w = 1.0
    marker.scale = Vector3(x=diameter, y=diameter, z=diameter)
    marker.color = color
    marker.lifetime = seconds_to_duration(lifetime_sec)
    return marker


def make_line_marker(
    header,
    namespace: str,
    marker_id: int,
    points: list[Point],
    width: float,
    color: ColorRGBA,
    lifetime_sec: float,
) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = width
    marker.color = color
    marker.points = points
    marker.lifetime = seconds_to_duration(lifetime_sec)
    return marker


def seconds_to_duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    nanosec = int(round((seconds - whole) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        whole += 1
        nanosec -= 1_000_000_000
    return Duration(sec=whole, nanosec=nanosec)


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
        description="Subscribe to a ROS2 image and extract a real-time white soft robot centerline on a dark background.",
    )
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--pointcloud-topic", default="/camera/camera/depth/color/points"
    )
    parser.add_argument(
        "--marker-topic", default="/white_centerline_3d/markers"
    )
    parser.add_argument(
        "--pose-array-topic", default="/white_centerline_3d/poses"
    )
    parser.add_argument(
        "--hsv-lower",
        type=parse_hsv_triplet,
        help="Manual lower HSV as H,S,V. Default is tuned for a white robot.",
    )
    parser.add_argument(
        "--hsv-upper",
        type=parse_hsv_triplet,
        help="Manual upper HSV as H,S,V. Default is tuned for a white robot.",
    )
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--process-hz", type=float, default=5.0)
    parser.add_argument("--log-period", type=float, default=1.0)
    parser.add_argument("--max-cloud-age", type=float, default=0.5)
    parser.add_argument("--point-search-radius", type=int, default=3)
    parser.add_argument("--allow-zero-points", action="store_true")
    parser.add_argument("--marker-diameter", type=float, default=0.012)
    parser.add_argument("--line-width", type=float, default=0.006)
    parser.add_argument("--marker-lifetime", type=float, default=0.5)
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
        else WHITE_ROBOT_HSV
    )

    rclpy.init()
    node = WhiteImageCenterlineClient(args, hsv_range)
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
