import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from scipy.optimize import minimize


def fit_circle(positions):
    """Fit a circle to 2D arc positions using least squares optimization."""

    def circle_error(params):
        cx, cy, r = params
        distances = np.linalg.norm(positions - np.array([cx, cy]), axis=1)
        return np.sum((distances - r) ** 2)

    # Initial guess: center at mean, radius from mean distance
    center_init = np.mean(positions, axis=0)
    radius_init = np.mean(np.linalg.norm(positions - center_init, axis=1))

    result = minimize(circle_error, [center_init[0], center_init[1], radius_init])
    cx, cy, r = result.x

    angles = np.arctan2(positions[:, 1] - cy, positions[:, 0] - cx)
    min_theta = np.min(angles)
    max_theta = max(
        np.max(angles), (np.pi / 2.0) + np.abs(min_theta) / 2.0
    )  # Ensure at least 90 degrees of arc

    return np.array([cx, cy]), r, (min_theta, max_theta)


def plot_circle(center, radius, thetas, origins, **kwargs):
    """Plot a circle given its center and radius."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    theta = np.linspace(thetas[0], thetas[1], 300)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    ax.scatter(
        origins[:, 0],
        origins[:, 1],
        color="black",
        alpha=1.0,
        label="Original Positions",
        s=0.3,
    )
    ax.plot(
        x,
        y,
        **kwargs,
    )
    ax.scatter(0.0, 0.0, color="black", label="Base (0,0)", marker="x", s=50)
    ax.scatter(
        center[0],
        center[1],
        color=kwargs.get("color", "black"),
        label="circle center",
        marker="x",
        s=50,
    )

    ax.set_aspect("equal")
    ax.set_title("Fitted Circle to EEF Trajectory", fontsize=14)
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    plt.show()


def plot_prpr_workspace_new_origin(
    P1_vec, R1_params, P2_vec, R2_params, num_samples=400
):
    """
    2D PRPR 매니퓰레이터의 Workspace 시각화
    - R2 계열(P2, R2)의 로컬 기준 좌표계는 Base(0,0)에서 R1 끝점을 잇는 벡터를 기준으로 생성
    """
    L1, (th1_min, th1_max) = R1_params
    L2, (th2_min, th2_max) = R2_params

    th1_vals = np.linspace(th1_min, th1_max, num_samples)
    th2_vals = np.linspace(th2_min, th2_max, num_samples)
    T1, T2 = np.meshgrid(th1_vals, th2_vals)
    T1 = T1.flatten()
    T2 = T2.flatten()

    # --- Forward Kinematics ---
    x1, y1 = P1_vec

    # R1 끝점 계산
    x2 = x1 + L1 * np.cos(T1)
    y2 = y1 + L1 * np.sin(T1)

    # --- [수정된 부분] 로컬 좌표계 기준 변경 ---
    # Base(0,0)에서 R1 끝점(x2, y2)을 잇는 벡터의 각도를 구함
    base_to_r1_angle = np.arctan2(y2, x2)
    # 해당 벡터를 로컬 Y축으로 삼기 위해 -90도(pi/2) 회전 (기존 조건 유지)
    local_angle = base_to_r1_angle - np.pi / 2

    # P2 (Fixed)
    dx2, dy2 = P2_vec
    x3 = x2 + (dx2 * np.cos(local_angle) - dy2 * np.sin(local_angle))
    y3 = y2 + (dx2 * np.sin(local_angle) + dy2 * np.cos(local_angle))

    # R2 (Active)
    x4 = x3 + L2 * np.cos(local_angle + T2)
    y4 = y3 + L2 * np.sin(local_angle + T2)

    # --- 시각화 ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(x4, y4, s=1, c="royalblue", alpha=0.2, label="Workspace")

    p0 = (0, 0)
    p1 = P1_vec

    # ==========================================
    # 1. First Pose (R1 최소 각도 기준)
    # ==========================================
    t1_first = th1_min
    p2_first = (p1[0] + L1 * np.cos(t1_first), p1[1] + L1 * np.sin(t1_first))

    # First Pose의 새로운 로컬 기준 각도 계산
    angle_first = np.arctan2(p2_first[1], p2_first[0])
    local_first = angle_first - np.pi / 2

    p3_first = (
        p2_first[0] + dx2 * np.cos(local_first) - dy2 * np.sin(local_first),
        p2_first[1] + dx2 * np.sin(local_first) + dy2 * np.cos(local_first),
    )
    p4_first = (
        p3_first[0] + L2 * np.cos(local_first + th2_min),
        p3_first[1] + L2 * np.sin(local_first + th2_min),
    )

    px_first = [p0[0], p1[0], p2_first[0], p3_first[0], p4_first[0]]
    py_first = [p0[1], p1[1], p2_first[1], p3_first[1], p4_first[1]]

    ax.plot(
        [px_first[0], px_first[1]],
        [py_first[0], py_first[1]],
        "k-",
        linewidth=3,
        label="P1 Link (Fixed)",
    )
    ax.plot(
        [px_first[1], px_first[2]],
        [py_first[1], py_first[2]],
        "r-o",
        linewidth=2.5,
        markersize=8,
        label="R1 Link (First Pose)",
    )

    # 시각적 참고용 가상 벡터 (Base -> R1 끝점)
    ax.plot(
        [0, p2_first[0]],
        [0, p2_first[1]],
        "m:",
        linewidth=1.5,
        label="Base to R1 Tip (Ref Vector)",
    )

    ax.plot([px_first[2], px_first[3]], [py_first[2], py_first[3]], "k-", linewidth=3)
    ax.plot(
        [px_first[3], px_first[4]],
        [py_first[3], py_first[4]],
        "g-o",
        linewidth=2.5,
        markersize=8,
        label="R2 Link (First Pose)",
    )

    r2_wedge_first = Wedge(
        center=p3_first,
        r=L2,
        theta1=np.degrees(local_first + th2_min),
        theta2=np.degrees(local_first + th2_max),
        facecolor="green",
        alpha=0.3,
        label="R2 Area (First Pose)",
    )
    ax.add_patch(r2_wedge_first)

    # ==========================================
    # 2. Last Pose (R1 최대 각도 기준)
    # ==========================================
    t1_last = th1_max
    p2_last = (p1[0] + L1 * np.cos(t1_last), p1[1] + L1 * np.sin(t1_last))

    # Last Pose의 새로운 로컬 기준 각도 계산
    angle_last = np.arctan2(p2_last[1], p2_last[0])
    local_last = angle_last - np.pi / 2

    p3_last = (
        p2_last[0] + dx2 * np.cos(local_last) - dy2 * np.sin(local_last),
        p2_last[1] + dx2 * np.sin(local_last) + dy2 * np.cos(local_last),
    )
    p4_last = (
        p3_last[0] + L2 * np.cos(local_last + th2_min),
        p3_last[1] + L2 * np.sin(local_last + th2_min),
    )

    px_last = [p0[0], p1[0], p2_last[0], p3_last[0], p4_last[0]]
    py_last = [p0[1], p1[1], p2_last[1], p3_last[1], p4_last[1]]

    ax.plot(
        [px_last[1], px_last[2]],
        [py_last[1], py_last[2]],
        "r--o",
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="R1 Link (Last Pose)",
    )

    # 시각적 참고용 가상 벡터 (Base -> R1 끝점, Last)
    ax.plot([0, p2_last[0]], [0, p2_last[1]], "m:", linewidth=1.5, alpha=0.5)

    ax.plot(
        [px_last[2], px_last[3]],
        [py_last[2], py_last[3]],
        "k--",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        [px_last[3], px_last[4]],
        [py_last[3], py_last[4]],
        "darkorange",
        marker="o",
        linestyle="--",
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="R2 Link (Last Pose)",
    )

    r2_wedge_last = Wedge(
        center=p3_last,
        r=L2,
        theta1=np.degrees(local_last + th2_min),
        theta2=np.degrees(local_last + th2_max),
        facecolor="orange",
        alpha=0.3,
        label="R2 Area (Last Pose)",
    )
    ax.add_patch(r2_wedge_last)

    # ==========================================
    # 3. 공통 요소
    # ==========================================
    r1_wedge = Wedge(
        center=p1,
        r=L1,
        theta1=np.degrees(th1_min),
        theta2=np.degrees(th1_max),
        facecolor="red",
        alpha=0.1,
    )
    ax.add_patch(r1_wedge)

    ax.plot(0, 0, "ks", markersize=10, label="Base (0,0)")

    ax.set_title("2D PRPR Workspace (Ref: Base to R1 Tip Vector)", fontsize=16)
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.legend(loc="upper right")
    # plt.tight_layout()
    plt.show()


df: pd.DataFrame = pd.read_csv("data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv")

origin_x = np.mean(df["marker_0_x"].to_numpy())
origin_y = np.mean(df["marker_0_y"].to_numpy())
print(f"origin (XY): ({origin_x}, {origin_y})")

x = df["marker_5_x"].to_numpy() - origin_x
y = df["marker_5_y"].to_numpy() - origin_y
positions = np.column_stack((x, y))

#
second_circle_center, second_circle_radius, second_circle_thetas = fit_circle(positions)

first_circle_center = (1.0 / 3.0) * second_circle_center.copy()
first_circle_radius = (1.0 / 3.0) * second_circle_radius
first_circle_thetas = second_circle_thetas

print(
    f"First circle center: {first_circle_center}, radius: {first_circle_radius}, thetas: {first_circle_thetas}"
)
print(
    f"Second circle center: {second_circle_center}, radius: {second_circle_radius}, thetas: {second_circle_thetas}"
)

# first_circle_thetas = (0.0, 0.01)
# second_circle_thetas = (0.0, 0.01)
# first_circle_thetas = (first_circle_thetas[1] - 0.3, first_circle_thetas[1])
# second_circle_thetas = (second_circle_thetas[1] - 0.3, second_circle_thetas[1])

plot_circle(
    center=second_circle_center,
    radius=second_circle_radius,
    thetas=(0.0, 2 * np.pi),  # second_circle_thetas,
    origins=positions,
    color="green",
    label="Approximated Circle",
)

# plot_prpr_workspace_new_origin(
#     P1_vec=first_circle_center,
#     R1_params=(first_circle_radius, first_circle_thetas),
#     P2_vec=second_circle_center,
#     R2_params=(second_circle_radius - first_circle_radius, second_circle_thetas),
# )
