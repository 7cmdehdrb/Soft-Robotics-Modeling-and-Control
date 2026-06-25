import argparse
import math
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ("joint2", "joint3", "joint4", "joint5")


class JointStatePublisherNode(Node):
    def __init__(self, topic_name):
        super().__init__("genesis_joint_state_gui_publisher")
        self.publisher = self.create_publisher(JointState, topic_name, 10)

    def publish_positions(self, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = positions
        message.velocity = []
        message.effort = []
        self.publisher.publish(message)


class JointStateGui:
    def __init__(self, args):
        self.args = args

        rclpy.init()
        self.node = JointStatePublisherNode(args.topic)

        self.root = tk.Tk()
        self.root.title("Genesis JointState Publisher")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.link_enabled = tk.BooleanVar(value=False)
        self.link_text = tk.StringVar(value="Link is OFF")
        self.degree_vars = [
            tk.DoubleVar(value=float(args.initial_deg)) for _ in JOINT_NAMES
        ]
        self.value_labels = []
        self._updating_linked_sliders = False
        self._is_running = True

        self._build_ui()
        self._schedule_publish()

    def _build_ui(self):
        root = self.root
        root.columnconfigure(0, weight=1)

        link = ttk.Checkbutton(
            root,
            textvariable=self.link_text,
            variable=self.link_enabled,
            command=self._on_link_toggle,
        )
        link.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))

        for row, (joint_name, var) in enumerate(zip(JOINT_NAMES, self.degree_vars), 1):
            frame = ttk.Frame(root)
            frame.grid(row=row, column=0, sticky="ew", padx=12, pady=5)
            frame.columnconfigure(1, weight=1)

            ttk.Label(frame, text=joint_name, width=8).grid(row=0, column=0, sticky="w")
            slider = ttk.Scale(
                frame,
                from_=self.args.min_deg,
                to=self.args.max_deg,
                variable=var,
                command=lambda value, idx=row - 1: self._on_slider(idx, value),
            )
            slider.grid(row=0, column=1, sticky="ew", padx=8)

            label = ttk.Label(frame, width=9)
            label.grid(row=0, column=2, sticky="e")
            self.value_labels.append(label)

        buttons = ttk.Frame(root)
        buttons.grid(row=len(JOINT_NAMES) + 1, column=0, sticky="ew", padx=12, pady=12)
        ttk.Button(buttons, text="Zero", command=self._zero).grid(row=0, column=0)
        ttk.Button(buttons, text="Quit", command=self._on_close).grid(
            row=0,
            column=1,
            padx=8,
        )

        self.status = ttk.Label(root, text="Disconnected")
        self.status.grid(
            row=len(JOINT_NAMES) + 2,
            column=0,
            sticky="ew",
            padx=12,
            pady=(0, 12),
        )
        self.status.configure(text=f"Publishing ROS2 JointState on {self.args.topic}")
        self._refresh_labels()

    def _on_slider(self, changed_idx, value):
        if self._updating_linked_sliders:
            return

        if self.link_enabled.get():
            self._updating_linked_sliders = True
            linked_value = float(value)
            for idx, var in enumerate(self.degree_vars):
                if idx != changed_idx:
                    var.set(linked_value)
            self._updating_linked_sliders = False

        self._refresh_labels()
        self._publish()

    def _on_link_toggle(self):
        if self.link_enabled.get():
            self.link_text.set("Link is ON")
            value = self.degree_vars[0].get()
            for var in self.degree_vars[1:]:
                var.set(value)
        else:
            self.link_text.set("Link is OFF")
        self._refresh_labels()
        self._publish()

    def _zero(self):
        for var in self.degree_vars:
            var.set(0.0)
        self._refresh_labels()
        self._publish()

    def _refresh_labels(self):
        for var, label in zip(self.degree_vars, self.value_labels):
            label.configure(text=f"{var.get():7.2f} deg")

    def _schedule_publish(self):
        if not self._is_running:
            return
        self._publish()
        period_ms = max(1, int(1000.0 / max(self.args.hz, 0.1)))
        self.root.after(period_ms, self._schedule_publish)

    def _publish(self):
        positions = [math.radians(var.get()) for var in self.degree_vars]
        self.node.publish_positions(positions)
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
        description="ROS2 JointState GUI publisher.",
    )
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--min-deg", type=float, default=-90.0)
    parser.add_argument("--max-deg", type=float, default=90.0)
    parser.add_argument("--initial-deg", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    JointStateGui(parse_args()).run()
