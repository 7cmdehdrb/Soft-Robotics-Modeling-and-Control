import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_map(x, a, b):
    """
    y = a*x + b
    """
    return a * x + b


def get_joint_angles(q1, joint_linear_params):
    """
    q1: 실제 제어하는 첫 번째 조인트 값
    joint_linear_params:
        [
            (a2, b2),   # q2 = a2*q1 + b2
            (a3, b3),   # q3 = a3*q1 + b3
            (a4, b4),   # q4 = a4*q1 + b4
        ]

    반환:
        [q1, q2, q3, q4]
    """
    qs = [q1]
    for a, b in joint_linear_params:
        qs.append(linear_map(q1, a, b))
    return np.array(qs, dtype=float)


def get_link_lengths(q1, link_linear_params, clip_min=0.0):
    """
    q1: 실제 제어하는 첫 번째 조인트 값
    link_linear_params:
        [
            (c1, d1),   # L1 = c1*q1 + d1
            (c2, d2),   # L2 = c2*q1 + d2
            (c3, d3),   # L3 = c3*q1 + d3
            (c4, d4),   # L4 = c4*q1 + d4
            (c5, d5),   # L5 = c5*q1 + d5
        ]

    반환:
        [L1, L2, L3, L4, L5]
    """
    lengths = np.array(
        [linear_map(q1, a, b) for a, b in link_linear_params], dtype=float
    )

    if clip_min is not None:
        lengths = np.maximum(lengths, clip_min)

    return lengths


def forward_kinematics_2d(q1, joint_linear_params, link_linear_params):
    """
    2D serial chain forward kinematics

    구조:
        링크 5개, 조인트 4개
        L1 -- q1 -- L2 -- q2 -- L3 -- q3 -- L4 -- q4 -- L5

    해석:
        - 첫 링크 L1은 base에서 q1 이전까지의 첫 번째 링크
        - 이후 각 조인트 회전을 누적하며 다음 링크 방향이 정해짐
        - 마지막 L5는 q4 이후 말단까지의 마지막 링크

    반환:
        points: shape (6, 2)
            [base, p1, p2, p3, p4, end_effector]
    """
    q = get_joint_angles(q1, joint_linear_params)  # [q1, q2, q3, q4]
    L = get_link_lengths(q1, link_linear_params)  # [L1, L2, L3, L4, L5]

    points = np.zeros((6, 2), dtype=float)
    x, y = 0.0, 0.0
    angle_sum = 0.0

    # Link 1: q1 방향
    angle_sum += q[0]
    x += L[0] * np.cos(angle_sum)
    y += L[0] * np.sin(angle_sum)
    points[1] = [x, y]

    # Link 2: q1 + q2 방향
    angle_sum += q[1]
    x += L[1] * np.cos(angle_sum)
    y += L[1] * np.sin(angle_sum)
    points[2] = [x, y]

    # Link 3: q1 + q2 + q3 방향
    angle_sum += q[2]
    x += L[2] * np.cos(angle_sum)
    y += L[2] * np.sin(angle_sum)
    points[3] = [x, y]

    # Link 4: q1 + q2 + q3 + q4 방향
    angle_sum += q[3]
    x += L[3] * np.cos(angle_sum)
    y += L[3] * np.sin(angle_sum)
    points[4] = [x, y]

    # Link 5: 마지막도 동일한 최종 방향으로 연장
    x += L[4] * np.cos(angle_sum)
    y += L[4] * np.sin(angle_sum)
    points[5] = [x, y]

    return points


def sample_workspace(
    q1_min, q1_max, joint_linear_params, link_linear_params, n_samples=2000
):
    q1_values = np.linspace(q1_min, q1_max, n_samples)

    all_points = []
    ee_points = []

    for q1 in q1_values:
        pts = forward_kinematics_2d(q1, joint_linear_params, link_linear_params)
        all_points.append(pts)
        ee_points.append(pts[-1])

    return q1_values, all_points, np.array(ee_points)


def plot_workspace(
    q1_min,
    q1_max,
    joint_linear_params,
    link_linear_params,
    n_samples=2000,
    n_pose_samples=12,
    show_joint_trace=False,
):
    q1_values, all_points, ee_points = sample_workspace(
        q1_min=q1_min,
        q1_max=q1_max,
        joint_linear_params=joint_linear_params,
        link_linear_params=link_linear_params,
        n_samples=n_samples,
    )

    df: pd.DataFrame = pd.read_csv(
        "data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # End-effector trajectory
    ax.plot(
        ee_points[:, 0], ee_points[:, 1], linewidth=2.0, label="End-effector trajectory"
    )

    # 몇 개 자세 샘플 표시
    sample_indices = np.linspace(0, n_samples - 1, n_pose_samples, dtype=int)
    for idx in sample_indices:
        pts = all_points[idx]
        ax.plot(pts[:, 0], pts[:, 1], marker="o", alpha=0.4)

    # 각 관절 위치 궤적도 보고 싶으면 표시
    if show_joint_trace:
        all_points_np = np.array(all_points)  # (N, 6, 2)
        for point_idx in range(1, 5):  # joint-related intermediate points
            ax.plot(
                all_points_np[:, point_idx, 0],
                all_points_np[:, point_idx, 1],
                linestyle="--",
                alpha=0.7,
                label=f"Point {point_idx} trace",
            )

    ax.scatter([0.0], [0.0], s=80, marker="s", label="Base")

    origin_x = df["marker_0_x"].to_numpy()
    origin_y = df["marker_0_y"].to_numpy()

    for i in range(6):
        x = df[f"marker_{i}_x"].to_numpy() - origin_x
        y = df[f"marker_{i}_y"].to_numpy() - origin_y

        positions = np.column_stack((x, y))

        ax.scatter(
            positions[:, 1], positions[:, 0], s=1, alpha=0.5, label=f"Marker {i}"
        )

    ax.set_title("2D Manipulator Workspace / End-effector Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # -----------------------------
    # 실제 제어 입력 q1 범위 [rad]
    # -----------------------------
    q1_min = 0.16840263616496753
    q1_max = 0.7155122543449552

    # -----------------------------------------
    # 나머지 3개 조인트가 q1의 1차 함수로 연동
    # q2 = a2*q1 + b2
    # q3 = a3*q1 + b3
    # q4 = a4*q1 + b4
    # -----------------------------------------
    joint_linear_params = [
        (0.903213, -0.013040),  # q2
        (0.937013, -0.115220),  # q3
        (0.962278, -0.073702),  # q4
    ]

    # -----------------------------------------
    # 5개 링크 길이가 q1의 1차 함수로 연동
    # L1 = c1*q1 + d1
    # ...
    # L5 = c5*q1 + d5
    # -----------------------------------------
    link_linear_params = [
        (0.015133, 0.021742),  # L1
        (0.015215, 0.022741),  # L2
        (0.014660, 0.021937),  # L3
        (0.014744, 0.021818),  # L4
        (0.015799, 0.022715),  # L5
    ]

    plot_workspace(
        q1_min=q1_min,
        q1_max=q1_max,
        joint_linear_params=joint_linear_params,
        link_linear_params=link_linear_params,
        n_samples=3000,
        n_pose_samples=15,
        show_joint_trace=False,
    )
