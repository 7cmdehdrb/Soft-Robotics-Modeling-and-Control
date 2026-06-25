import os
import sys
import time
import roslibpy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import genesis as gs


# CLASS


class RosSubscriber:
    def __init__(self, host: str, port: int, topic_name: str, topic_type: str):
        self._host = host
        self._port = port

        self._topic_name = topic_name
        self._topic_type = topic_type

        self._client = roslibpy.Ros(host=self._host, port=self._port)
        self._topic = roslibpy.Topic(self._client, self._topic_name, self._topic_type)

        self._current_pressure: float = 0.0
        self._pressure_history = deque(maxlen=60)  # 최근 60개 데이터 저장

    def callback(self, message):
        self._current_pressure = message["data"][1]  # current_pressure
        self._pressure_history.append(self._current_pressure)

    def start(self):
        self._client.run()

        if not self._client.is_connected:
            raise RuntimeError("rosbridge connection failed")

        print(f"Connected to rosbridge: ws://{self._host}:{self._port}/")

        self._topic.subscribe(self.callback)

    def __delete__(self, instance):
        self._topic.unsubscribe()
        self._client.terminate()

    @property
    def current_pressure(self) -> float:
        return self._current_pressure

    @property
    def pressure_history(self) -> list:
        if len(self._pressure_history) != 60:
            return np.zeros(60, dtype=np.float32)  # 초기에는 0으로 채운 numpy 배열 반환
        return np.array(self._pressure_history, dtype=np.float32)


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


def main():

    ######################## Instances ########################

    ros_sub = RosSubscriber(
        host="localhost",  # ROS2 서버 IP로 변경
        port=9090,
        topic_name="/arduino_data",
        topic_type="std_msgs/Float32MultiArray",
    )
    ros_sub.start()

    lstm_predictor = LSTMAnglePredictor(
        "/home/min/project_SORO/runs/lstm_marker_angles_20260326-183235/best.pt",
        device="cuda",
    )

    ########################## init ##########################
    gs.init(
        backend=gs.gpu,
        seed=0,
        precision="32",
        logging_level="info",
    )

    ######################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=30,  # 1. substeps 증가 (수치적 드리프트 방지)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 2.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.8),
            enable_collision=True,
            enable_self_collision=False,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -1.0),
            upper_bound=(1.0, 1.0, 1.0),
            gravity=(0, 0, 0),
            enable_CPIC=True,
        ),
        vis_options=gs.options.VisOptions(
            # ... (기존과 동일)
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    ########################## entities ##########################
    scene.add_entity(morph=gs.morphs.Plane())

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="/home/min/project_SORO/src/genesis/resource/four_joint_arm.urdf",
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
            thickness=0.01,  # 2. scale(0.2)에 맞춰 thickness 축소
            damping=100.0,
            func_instantiate_rigid_from_soft=None,
            func_instantiate_soft_from_rigid=None,
            func_instantiate_rigid_soft_association=None,
        ),
        surface=gs.surfaces.Default(vis_mode="visual"),
    )

    # env마다 "있음 / 없음"을 바꿀 rigid wall

    # default_wall_pos = (0.08, 0.0, 0.47)  # 기본 위치: robot 앞쪽
    default_wall_pos = (0.03, 0.0, 0.58)  # 기본 위치: robot 앞쪽

    wall = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=default_wall_pos,
            radius=0.02,
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )

    ########################## build ##########################

    B = 2

    scene.build(
        n_envs=B,
        env_spacing=(0.3, 0.3),  # 보기 좋게만 띄움. 물리 좌표는 안 바뀜
    )

    ########################## reset ##########################

    scene.reset()

    ########################## wall presence mask ##########################

    # 앞 절반 env: wall 있음
    # 뒤 절반 env: wall 없음 -> 멀리 치움
    wall_pos = torch.zeros((B, 3), dtype=gs.tc_float, device=gs.device)

    # 기본 위치: wall이 "있는" 환경
    wall_pos[:, 0] = default_wall_pos[0]
    wall_pos[:, 1] = default_wall_pos[1]
    wall_pos[:, 2] = default_wall_pos[2]

    # 절반은 제거 대신 먼 위치로 이동
    # 너무 말도 안 되게 크게 보내기보다, 작업 공간 밖으로만 치웁니다.

    no_wall_mask = torch.arange(B, device=gs.device) >= (B // 2)
    wall_pos[no_wall_mask, 0] = 2.0
    wall_pos[no_wall_mask, 1] = 2.0
    wall_pos[no_wall_mask, 2] = 2.0

    # # 모든 env에 대해 한 번에 설정
    wall.set_pos(wall_pos)

    ########################## run ##########################
    try:
        while True:

            pressure_history: np.ndarray = ros_sub.pressure_history  # 최근 60개 데이터
            predicted_angles: np.ndarray = np.deg2rad(
                lstm_predictor.predict(pressure_history)
            )  # (4,) 예측된 4개 관절 각도, RADIAN

            print(
                f"Pressure History: {pressure_history[:5]} ... {pressure_history[-5:]}"
            )  # 최근 5개와 처음 5개만 출력
            print(f"Predicted Angles (RADIAN): {predicted_angles}")  # 예측된 각도 출력

            noise_scale = 0.01
            noise = noise_scale * torch.randn(
                (B, robot.n_dofs),
                dtype=gs.tc_float,
                device=gs.device,
            )

            # (4,) -> (1, 4) -> (B, 4)
            predicted_angles_tensor = (
                torch.tensor(
                    predicted_angles,
                    dtype=gs.tc_float,
                    device=gs.device,
                )
                .unsqueeze(0)
                .repeat(B, 1)
            )

            # # 기본값으로 채우기
            # dofs_ctrl = torch.full(
            #     (B, robot.n_dofs),
            #     fill_value=0.0,
            #     dtype=gs.tc_float,
            #     device=gs.device,
            # )

            # # 예측값을 특정 DOF에 할당 (예: 1~4번 DOF)
            # dofs_ctrl[:, 1:5] = predicted_angles_tensor

            dofs_ctrl = predicted_angles_tensor + noise
            # dofs_ctrl[:, 0] = 0.0

            # 3. Velocity Control -> Position Control로 변경
            robot.control_dofs_position(dofs_ctrl)

            scene.step()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Error during simulation:", e)


if __name__ == "__main__":
    main()
