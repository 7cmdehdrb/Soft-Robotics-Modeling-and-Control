import argparse
import time
from pathlib import Path
from threading import Lock

import numpy as np
import roslibpy
import torch
from PIL import Image

import genesis as gs

DEFAULT_JOINT_NAMES = ("joint2", "joint3", "joint4", "joint5")
URDF_PATH = Path(__file__).resolve().parent / "resource" / "four_joint_arm.urdf"
DEFAULT_CAMERA_POS = (0.065, -0.19, 0.58)
DEFAULT_CAMERA_LOOKAT = (0.065, 0.0, 0.58)
DEFAULT_CAMERA_UP = (0.0, 0.0, 1.0)


class JointStateSubscriber:
    def __init__(self, host, port, topic_name, joint_names):
        self.host = host
        self.port = port
        self._client = roslibpy.Ros(host=host, port=port)
        self._topic = roslibpy.Topic(
            self._client,
            topic_name,
            "sensor_msgs/JointState",
        )
        self._joint_names = tuple(joint_names)
        self._positions = np.zeros(len(self._joint_names), dtype=np.float32)
        self._lock = Lock()
        self._last_msg_time = 0.0

    def start(self):
        self._client.run()
        if not self._client.is_connected:
            raise RuntimeError("rosbridge connection failed")

        self._topic.subscribe(self._callback)
        print(f"Connected to rosbridge: ws://{self.host}:{self.port}/")
        print(f"Subscribing JointState: {self._topic.name}")

    def stop(self):
        self._topic.unsubscribe()
        self._client.terminate()

    @property
    def client(self):
        return self._client

    def _callback(self, message):
        positions = message.get("position", [])
        if not positions:
            return

        names = message.get("name", [])
        next_positions = None

        if names:
            by_name = {
                name: float(pos)
                for name, pos in zip(names, positions)
                if name in self._joint_names
            }
            if by_name:
                with self._lock:
                    next_positions = self._positions.copy()
                    for idx, joint_name in enumerate(self._joint_names):
                        if joint_name in by_name:
                            next_positions[idx] = by_name[joint_name]
        elif len(positions) >= len(self._joint_names):
            next_positions = np.asarray(
                positions[: len(self._joint_names)],
                dtype=np.float32,
            )

        if next_positions is None:
            return

        with self._lock:
            self._positions[:] = next_positions
            self._last_msg_time = time.time()

    @property
    def positions(self):
        with self._lock:
            return self._positions.copy()

    @property
    def last_msg_age(self):
        with self._lock:
            if self._last_msg_time <= 0.0:
                return float("inf")
            return time.time() - self._last_msg_time


class RawImagePublisher:
    def __init__(self, client, topic_name, frame_id, max_width=None):
        self._topic = roslibpy.Topic(client, topic_name, "sensor_msgs/Image")
        self._frame_id = frame_id
        self._max_width = max_width
        self._seq = 0
        self._topic.advertise()
        print(f"Publishing raw camera image: {topic_name}")

    def stop(self):
        self._topic.unadvertise()

    def publish_rgb(self, rgb):
        rgb = self._to_rgb8(rgb)
        rgb = self._resize_if_needed(rgb, self._max_width)
        height, width, channels = rgb.shape
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1_000_000_000)

        self._topic.publish(
            roslibpy.Message(
                {
                    "header": {
                        "stamp": {"sec": sec, "nanosec": nanosec},
                        "frame_id": self._frame_id,
                    },
                    "height": height,
                    "width": width,
                    "encoding": "rgb8",
                    "is_bigendian": 0,
                    "step": width * channels,
                    "data": rgb.reshape(-1).tolist(),
                }
            )
        )
        self._seq += 1

    @staticmethod
    def _to_rgb8(rgb):
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)

        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                max_value = float(np.max(rgb)) if rgb.size else 0.0
                scale = 255.0 if max_value <= 1.0 else 1.0
                rgb = np.clip(rgb * scale, 0.0, 255.0)
            rgb = rgb.astype(np.uint8)

        return np.ascontiguousarray(rgb)

    @staticmethod
    def _resize_if_needed(rgb, max_width):
        if max_width is None or rgb.shape[1] <= max_width:
            return rgb

        height, width = rgb.shape[:2]
        scale = float(max_width) / float(width)
        size = (int(max_width), max(1, int(round(height * scale))))
        image = Image.fromarray(rgb, mode="RGB")
        return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


class RenderDebugger:
    def __init__(self, enabled, save_path, log_period):
        self.enabled = enabled
        self.save_path = Path(save_path)
        self.log_period = max(float(log_period), 0.1)
        self._last_log_time = 0.0
        self._saved_once = False

    def inspect(self, rgb, pos, lookat):
        if not self.enabled:
            return

        rgb8 = RawImagePublisher._to_rgb8(rgb)
        now = time.time()
        if now - self._last_log_time >= self.log_period:
            self._last_log_time = now
            print(
                "Render RGB "
                f"shape={rgb8.shape} "
                f"min={int(rgb8.min())} "
                f"max={int(rgb8.max())} "
                f"mean={float(rgb8.mean()):.1f} "
                f"pos={np.round(pos, 4).tolist()} "
                f"lookat={np.round(lookat, 4).tolist()}"
            )

        if not self._saved_once:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb8, mode="RGB").save(self.save_path)
            print(f"Saved first rendered RGB frame to {self.save_path}")
            self._saved_once = True


class CameraPoseSubscriber:
    def __init__(self, client, topic_name, default_pos, default_lookat, default_up):
        self._topic = roslibpy.Topic(
            client,
            topic_name,
            "std_msgs/Float32MultiArray",
        )
        self._lock = Lock()
        self._pos = np.asarray(default_pos, dtype=np.float32)
        self._lookat = np.asarray(default_lookat, dtype=np.float32)
        self._up = np.asarray(default_up, dtype=np.float32)
        self._fixed = False
        self._version = 0
        self._last_log_time = 0.0
        print(f"Subscribing camera pose: {topic_name}")

    def start(self):
        self._topic.subscribe(self._callback)

    def stop(self):
        self._topic.unsubscribe()

    def _callback(self, message):
        data = message.get("data", [])
        if len(data) < 9:
            return

        pos = np.asarray(data[0:3], dtype=np.float32)
        lookat = np.asarray(data[3:6], dtype=np.float32)
        up = np.asarray(data[6:9], dtype=np.float32)
        fixed = bool(data[9]) if len(data) >= 10 else False

        if not np.isfinite(pos).all():
            return
        if not np.isfinite(lookat).all():
            return
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            return
        if np.linalg.norm(pos - lookat) < 1e-6:
            return

        with self._lock:
            self._pos[:] = pos
            self._lookat[:] = lookat
            self._up[:] = up / np.linalg.norm(up)
            self._fixed = fixed
            self._version += 1

        now = time.time()
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            print(
                "Camera pose received "
                f"pos={np.round(pos, 4).tolist()} "
                f"lookat={np.round(lookat, 4).tolist()} "
                f"fixed={fixed}"
            )

    @property
    def pose(self):
        with self._lock:
            return (
                self._version,
                self._pos.copy(),
                self._lookat.copy(),
                self._up.copy(),
                self._fixed,
            )


def create_scene(show_viewer, camera_gui, camera_res):
    gs.init(
        backend=gs.gpu,
        seed=0,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=30,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.38, 0.58),
            camera_lookat=(0.06, 0.0, 0.56),
            camera_fov=35,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -9.8),
            enable_collision=True,
            enable_self_collision=False,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.2),
            upper_bound=(0.5, 0.5, 1.2),
            gravity=(0.0, 0.0, 0.0),
            enable_CPIC=True,
        ),
        vis_options=gs.options.VisOptions(
            background_color=(0.04, 0.08, 0.12),
            ambient_light=(0.28, 0.28, 0.28),
            shadow=True,
            lights=[
                {
                    "type": "directional",
                    "dir": (0.0, 1.0, -0.25),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 4.5,
                },
            ],
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.72, 0.72, 0.72, 1.0)),
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(URDF_PATH),
            pos=(0.0, 0.0, 0.5),
            euler=(0.0, -90.0, 0.0),
            scale=1.0,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            material_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=1000.0,
                model="neohooken",
            ),
            thickness=0.01,
            damping=100.0,
            func_instantiate_rigid_from_soft=None,
            func_instantiate_soft_from_rigid=None,
            func_instantiate_rigid_soft_association=None,
        ),
        surface=gs.surfaces.Default(
            color=(0.02, 0.72, 0.95, 1.0),
            vis_mode="visual",
        ),
    )

    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.065, 0.12, 0.58),
            size=(1.6, 0.015, 1.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )

    side_camera = scene.add_camera(
        res=tuple(camera_res),
        pos=DEFAULT_CAMERA_POS,
        lookat=DEFAULT_CAMERA_LOOKAT,
        up=DEFAULT_CAMERA_UP,
        fov=28,
        GUI=camera_gui,
        env_idx=0,
        near=0.02,
        far=2.0,
    )

    scene.build(n_envs=1, env_spacing=(0.0, 0.0))
    scene.reset()
    return scene, robot, side_camera


def run(args):
    joint_sub = JointStateSubscriber(
        host=args.ros_host,
        port=args.ros_port,
        topic_name=args.joint_topic,
        joint_names=DEFAULT_JOINT_NAMES,
    )
    joint_sub.start()

    image_publishers = []
    if not args.no_preview_image:
        image_publishers.append(
            RawImagePublisher(
                client=joint_sub.client,
                topic_name=args.image_preview_topic,
                frame_id=args.camera_frame,
                max_width=args.preview_width,
            )
        )
    if args.publish_raw_image:
        image_publishers.append(
            RawImagePublisher(
                client=joint_sub.client,
                topic_name=args.image_topic,
                frame_id=args.camera_frame,
                max_width=None,
            )
        )

    camera_pose_sub = CameraPoseSubscriber(
        client=joint_sub.client,
        topic_name=args.camera_pose_topic,
        default_pos=DEFAULT_CAMERA_POS,
        default_lookat=DEFAULT_CAMERA_LOOKAT,
        default_up=DEFAULT_CAMERA_UP,
    )
    camera_pose_sub.start()

    scene, robot, side_camera = create_scene(
        show_viewer=not args.headless,
        camera_gui=args.camera_gui,
        camera_res=(args.image_width, args.image_height),
    )

    if robot.n_dofs != len(DEFAULT_JOINT_NAMES):
        print(
            f"Warning: URDF DOF count is {robot.n_dofs}, "
            f"but {len(DEFAULT_JOINT_NAMES)} JointState names are configured."
        )

    publish_period = 1.0 / max(args.image_hz, 0.1)
    next_image_time = time.time()
    last_camera_pose_version = -1
    latest_camera_pos = np.asarray(DEFAULT_CAMERA_POS, dtype=np.float32)
    latest_camera_lookat = np.asarray(DEFAULT_CAMERA_LOOKAT, dtype=np.float32)
    render_debugger = RenderDebugger(
        enabled=args.debug_render,
        save_path=args.debug_render_path,
        log_period=args.debug_render_log_period,
    )

    try:
        while True:
            dofs = joint_sub.positions[: robot.n_dofs]
            dofs_ctrl = torch.tensor(
                dofs,
                dtype=gs.tc_float,
                device=gs.device,
            ).unsqueeze(0)

            robot.control_dofs_position(dofs_ctrl)
            scene.step()

            camera_pose_version, pos, lookat, up, _ = camera_pose_sub.pose
            if camera_pose_version != last_camera_pose_version:
                side_camera.set_pose(
                    pos=pos,
                    lookat=lookat,
                    up=up,
                )
                latest_camera_pos = pos
                latest_camera_lookat = lookat
                last_camera_pose_version = camera_pose_version

            now = time.time()
            if now >= next_image_time:
                rgb, _, _, _ = side_camera.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                    normal=False,
                    force_render=True,
                )
                render_debugger.inspect(rgb, latest_camera_pos, latest_camera_lookat)
                for image_pub in image_publishers:
                    image_pub.publish_rgb(rgb)
                next_image_time = now + publish_period

    except KeyboardInterrupt:
        pass
    finally:
        for image_pub in image_publishers:
            image_pub.stop()
        camera_pose_sub.stop()
        joint_sub.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genesis hybrid soft robot scene driven by ROS2 JointState over rosbridge.",
    )
    parser.add_argument("--ros-host", default="localhost")
    parser.add_argument("--ros-port", type=int, default=9090)
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--camera-pose-topic", default="/genesis/side_camera/pose")
    parser.add_argument("--image-topic", default="/genesis/side_camera/image_raw")
    parser.add_argument("--image-preview-topic", default="/genesis/side_camera/image_preview")
    parser.add_argument("--camera-frame", default="genesis_side_camera")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--image-hz", type=float, default=5.0)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--no-preview-image", action="store_true")
    parser.add_argument("--publish-raw-image", action="store_true")
    parser.add_argument("--debug-render", action="store_true")
    parser.add_argument(
        "--debug-render-path",
        default="/tmp/genesis_side_camera_debug.png",
    )
    parser.add_argument("--debug-render-log-period", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera-gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
