import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = "data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv"

df: pd.DataFrame = pd.read_csv(data)

pressure = df["current_pressure"].to_numpy()

x = df["marker_5_x"].to_numpy()
y = df["marker_5_y"].to_numpy()
z = df["marker_5_z"].to_numpy()

positions = np.column_stack((x, y, z))

fig = plt.figure()

plt.scatter(x=pressure, y=x, c="r", alpha=0.3, s=0.4, label="X")
plt.scatter(x=pressure, y=y, c="g", alpha=0.3, s=0.4, label="Y")
plt.scatter(x=pressure, y=z, c="b", alpha=0.3, s=0.4, label="Z")

# plt.axis("equal")
plt.xlabel("Pressure")
plt.ylabel("Position")
plt.legend()

plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="b", marker="o")
ax.plot(xs=positions[:, 0], ys=positions[:, 1], zs=positions[:, 2], c="b")

x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])

max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

mid_x = (x_min + x_max) * 0.5
mid_y = (y_min + y_max) * 0.5
mid_z = (z_min + z_max) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.show()
