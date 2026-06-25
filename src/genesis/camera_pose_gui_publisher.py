import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


DEFAULT_CAMERA_POSE = {
    "pos": [0.065, -0.19, 0.58],
    "lookat": [0.065, 0.0, 0.58],
    "up": [0.0, 0.0, 1.0],
    "fixed": False,
}


class CameraPosePublisherNode(Node):
    def __init__(self, topic_name):
        super().__init__("genesis_camera_pose_gui_publisher")
        self.publisher = self.create_publisher(Float32MultiArray, topic_name, 10)

    def publish_pose(self, pos, lookat, up, fixed):
        message = Float32MultiArray()
        message.data = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(lookat[0]),
            float(lookat[1]),
            float(lookat[2]),
            float(up[0]),
            float(up[1]),
            float(up[2]),
            1.0 if fixed else 0.0,
        ]
        self.publisher.publish(message)


class CameraPoseGui:
    def __init__(self, args):
        self.args = args
        rclpy.init()
        self.node = CameraPosePublisherNode(args.topic)

        self.root = tk.Tk()
        self.root.title("Genesis Camera Pose Publisher")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fixed_var = tk.BooleanVar(value=DEFAULT_CAMERA_POSE["fixed"])
        self.lock_text = tk.StringVar(value="Pose is UNLOCKED")
        self.vars = {
            group: [
                tk.DoubleVar(value=DEFAULT_CAMERA_POSE[group][idx])
                for idx in range(3)
            ]
            for group in ("pos", "lookat", "up")
        }
        self._widgets_to_lock = []
        self._is_running = True

        self._build_ui()
        self._schedule_publish()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)

        lock = ttk.Checkbutton(
            self.root,
            textvariable=self.lock_text,
            variable=self.fixed_var,
            command=self._on_lock_toggle,
        )
        lock.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))

        row = 1
        for group, label in (
            ("pos", "Position"),
            ("lookat", "Look At"),
            ("up", "Up Vector"),
        ):
            group_frame = ttk.LabelFrame(self.root, text=label)
            group_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=6)
            group_frame.columnconfigure(1, weight=1)

            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                var = self.vars[group][axis_idx]
                ttk.Label(group_frame, text=axis_name, width=2).grid(
                    row=axis_idx,
                    column=0,
                    sticky="w",
                    padx=(8, 4),
                    pady=4,
                )
                scale = ttk.Scale(
                    group_frame,
                    from_=self.args.min_value,
                    to=self.args.max_value,
                    variable=var,
                    command=lambda _value: self._publish(),
                )
                scale.grid(row=axis_idx, column=1, sticky="ew", padx=6, pady=4)
                spinbox = ttk.Spinbox(
                    group_frame,
                    from_=self.args.min_value,
                    to=self.args.max_value,
                    increment=self.args.step,
                    textvariable=var,
                    width=9,
                    command=self._publish,
                )
                spinbox.grid(row=axis_idx, column=2, sticky="e", padx=(4, 8), pady=4)
                spinbox.bind("<Return>", lambda _event: self._publish())
                spinbox.bind("<FocusOut>", lambda _event: self._publish())
                self._widgets_to_lock.extend([scale, spinbox])

            row += 1

        buttons = ttk.Frame(self.root)
        buttons.grid(row=row, column=0, sticky="ew", padx=12, pady=12)
        ttk.Button(buttons, text="Side View", command=self._set_side_view).grid(
            row=0,
            column=0,
        )
        ttk.Button(buttons, text="Save", command=self._save_pose).grid(
            row=0,
            column=1,
            padx=8,
        )
        ttk.Button(buttons, text="Load", command=self._load_pose).grid(
            row=0,
            column=2,
        )
        ttk.Button(buttons, text="Quit", command=self._on_close).grid(
            row=0,
            column=3,
            padx=8,
        )

        self.status = ttk.Label(
            self.root,
            text=f"Publishing camera pose on {self.args.topic}",
        )
        self.status.grid(row=row + 1, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _on_lock_toggle(self):
        locked = self.fixed_var.get()
        self.lock_text.set("Pose is LOCKED" if locked else "Pose is UNLOCKED")
        state = "disabled" if locked else "normal"
        for widget in self._widgets_to_lock:
            widget.configure(state=state)
        self._publish()

    def _set_side_view(self):
        self._set_pose(DEFAULT_CAMERA_POSE)

    def _set_pose(self, pose):
        for group in ("pos", "lookat", "up"):
            values = pose[group]
            for idx, value in enumerate(values):
                self.vars[group][idx].set(float(value))
        self.fixed_var.set(bool(pose.get("fixed", self.fixed_var.get())))
        self._on_lock_toggle()
        self._publish()

    def _current_pose(self):
        return {
            group: [var.get() for var in self.vars[group]]
            for group in ("pos", "lookat", "up")
        } | {"fixed": self.fixed_var.get()}

    def _save_pose(self):
        path = filedialog.asksaveasfilename(
            title="Save camera pose",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(self._current_pose(), indent=2),
            encoding="utf-8",
        )

    def _load_pose(self):
        path = filedialog.askopenfilename(
            title="Load camera pose",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pose = json.loads(Path(path).read_text(encoding="utf-8"))
            self._validate_pose(pose)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self._set_pose(pose)

    @staticmethod
    def _validate_pose(pose):
        for group in ("pos", "lookat", "up"):
            if group not in pose or len(pose[group]) != 3:
                raise ValueError(f"'{group}' must contain three numbers")
            for value in pose[group]:
                float(value)

    def _schedule_publish(self):
        if not self._is_running:
            return
        self._publish()
        period_ms = max(1, int(1000.0 / max(self.args.hz, 0.1)))
        self.root.after(period_ms, self._schedule_publish)

    def _publish(self):
        pose = self._current_pose()
        self.node.publish_pose(
            pose["pos"],
            pose["lookat"],
            pose["up"],
            pose["fixed"],
        )
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def _on_close(self):
        self._is_running = False
        try:
            self.node.destroy_node()
            rclpy.shutdown()
        finally:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROS2 GUI publisher for Genesis camera position/lookat/up.",
    )
    parser.add_argument("--topic", default="/genesis/side_camera/pose")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--min-value", type=float, default=-1.0)
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    CameraPoseGui(parse_args()).run()
