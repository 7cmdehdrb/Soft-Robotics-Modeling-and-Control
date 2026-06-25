from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def fit_first_order(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]

    if xv.size < 2:
        return np.nan, np.nan, np.nan

    a, b = np.polyfit(xv, yv, 1)
    y_pred = a * xv + b
    ss_res = np.sum((yv - y_pred) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


df: pd.DataFrame = pd.read_csv("data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv")


fig = plt.figure(figsize=(50, 10))
ax = fig.add_subplot(111)

times = df["time"].to_numpy()

mask = (times >= 0) & (times <= 20)
times = times[mask]

link0 = np.empty_like(times)
link1 = np.empty_like(times)
link2 = np.empty_like(times)
link3 = np.empty_like(times)
link4 = np.empty_like(times)

links = [link0, link1, link2, link3, link4]

joint0 = np.empty_like(times)
joint1 = np.empty_like(times)
joint2 = np.empty_like(times)
joint3 = np.empty_like(times)

joints = [joint0, joint1, joint2, joint3]

for idx in range(5):
    x1 = df[f"marker_{idx}_x"].to_numpy()[mask]
    y1 = df[f"marker_{idx}_y"].to_numpy()[mask]

    x2 = df[f"marker_{idx+1}_x"].to_numpy()[mask]
    y2 = df[f"marker_{idx+1}_y"].to_numpy()[mask]

    link_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    links[idx] = link_length.copy()

    # ax.plot(times, link_length, label=f"Marker {idx} to {idx+1}")


for idx in range(4):
    x1 = df[f"marker_{idx}_x"].to_numpy()[mask]
    y1 = df[f"marker_{idx}_y"].to_numpy()[mask]

    x2 = df[f"marker_{idx+1}_x"].to_numpy()[mask]
    y2 = df[f"marker_{idx+1}_y"].to_numpy()[mask]

    x3 = df[f"marker_{idx+2}_x"].to_numpy()[mask]
    y3 = df[f"marker_{idx+2}_y"].to_numpy()[mask]

    vec1 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    vec2 = np.array([x3 - x2, y3 - y2], dtype=np.float64)

    joint: np.ndarray = np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])

    joints[idx] = joint.copy()

    # ax.plot(times, joint, label=f"Marker {idx} to {idx+1} to {idx+2}")

plt.scatter(joints[0], links[0], label="Joint 0 vs Link 0", alpha=0.5, s=0.5)
plt.scatter(joints[0], links[1], label="Joint 1 vs Link 1", alpha=0.5, s=0.5)
plt.scatter(joints[0], links[2], label="Joint 2 vs Link 2", alpha=0.5, s=0.5)
plt.scatter(joints[0], links[3], label="Joint 3 vs Link 3", alpha=0.5, s=0.5)
plt.scatter(joints[0], links[4], label="Joint 4 vs Link 4", alpha=0.5, s=0.5)

for n in range(5):
    a, b, r2 = fit_first_order(joints[0], links[n])
    print(f"N={n}: y = {a:.6f}x + {b:.6f}, R^2={r2:.4f}")

    x_line = np.linspace(np.nanmin(joints[0]), np.nanmax(joints[0]), 200)
    y_line = a * x_line + b
    ax.plot(x_line, y_line, linewidth=2, label=f"Fit {n}: y={a:.3f}x+{b:.3f}")

print(min(joints[0]), max(joints[0]))

# plt.scatter(joints[0], joints[1], label="Joint 0 vs Joint 1", alpha=0.5, s=0.5)
# plt.scatter(joints[0], joints[2], label="Joint 0 vs Joint 2", alpha=0.5, s=0.5)
# plt.scatter(joints[0], joints[3], label="Joint 0 vs Joint 3", alpha=0.5, s=0.5)

# for n in range(1, 4):
#     a, b, r2 = fit_first_order(joints[0], joints[n])
#     print(f"N={n}: y = {a:.6f}x + {b:.6f}, R^2={r2:.4f}")

#     x_line = np.linspace(np.nanmin(joints[0]), np.nanmax(joints[0]), 200)
#     y_line = a * x_line + b
#     ax.plot(x_line, y_line, linewidth=2, label=f"Fit Joint {n}: y={a:.3f}x+{b:.3f}")

# ax.set_xlabel("Time")
# ax.set_ylabel("Angle (degrees)")
# ax.set_title("Angle between Consecutive Markers Over Time")

# ax.axis("equal")
ax.grid(True)

ax.legend()
plt.show()
