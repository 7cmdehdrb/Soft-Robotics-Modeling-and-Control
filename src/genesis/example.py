import numpy as np
import torch
import genesis as gs

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
        euler=(0.0, 0.0, 0.0),
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

default_wall_pos = (0.08, 0.0, 0.47)  # 기본 위치: robot 앞쪽

wall = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=default_wall_pos,
        radius=0.02,
        fixed=True,
    ),
    material=gs.materials.Rigid(),
)

########################## build ##########################

B = 4

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
        for i in range(1000):
            sin_val = np.sin(2 * np.pi * i * 0.001)
            angle_deg = 30.0 * sin_val + 30.0
            angle_rad = np.deg2rad(angle_deg)
            base = float(angle_rad)

            noise_scale = 0.01
            noise = noise_scale * torch.randn(
                (B, robot.n_dofs),
                dtype=gs.tc_float,
                device=gs.device,
            )

            dofs_ctrl = (
                torch.full(
                    (B, robot.n_dofs),
                    fill_value=base,
                    dtype=gs.tc_float,
                    device=gs.device,
                )
                + noise
            )

            # 모든 env에서 0번 DOF(베이스) 완벽히 고정
            dofs_ctrl[:, 0] = 0.0

            dofs_pos = robot.get_dofs_position()

            # 3. Velocity Control -> Position Control로 변경
            robot.control_dofs_position(dofs_ctrl)

            scene.step()

except KeyboardInterrupt:
    pass
except Exception as e:
    print("Error during simulation:", e)
