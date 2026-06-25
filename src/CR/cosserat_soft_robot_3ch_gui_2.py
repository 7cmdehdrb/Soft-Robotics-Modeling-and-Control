"""
Cosserat rod based quasi-static simulator for the updated 3-chamber
soft continuum robot.

This model keeps the boundary-value/shooting structure used in
cosserat_soft_robot_gui.py, but updates the robot to:
- length: 150 mm
- 3 pneumatic chambers spaced by 120 deg
- chamber path radius: 9.947 mm
- cross-section properties from GEOMETRIC_PARAM.md
- Dragon Skin 30 Young's modulus range from PHYSICS_PARAM.md

Run:
    python cosserat_soft_robot_3ch_gui_2.py

Dependencies:
    numpy, scipy, matplotlib

Notes:
    The chamber is not circular and its effective pneumatic area is not
    specified in the available documents. This simulator exposes Aeff as a
    calibration slider instead of hard-coding a false-precision value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.optimize import root


# True: mouse rotation is constrained to azimuth-only viewing around the Z axis.
# False: matplotlib's default free 3D rotation is available.
LOCK_VIEW_TO_Z_ROTATION = True
LOCKED_VIEW_ELEVATION_DEG = 25.0


# -----------------------------
# Basic SO(3) utilities
# -----------------------------

def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array(
        [[0.0, -wz, wy],
         [wz, 0.0, -wx],
         [-wy, wx, 0.0]],
        dtype=float,
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map for a rotation vector w."""
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < 1e-12:
        return np.eye(3) + W
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1.0 - np.cos(theta)) / theta**2) * (W @ W)
    )


# -----------------------------
# Updated robot parameters
# -----------------------------

@dataclass
class RobotParams:
    # Geometry
    L: float = 0.150                    # total length [m]
    area: float = 328.879e-6            # cross-section area [m^2]
    Ixx: float = 19507.636e-12          # second moment of area [m^4]
    Iyy: float = 19507.636e-12          # second moment of area [m^4]
    chamber_path_radius: float = 9.947e-3  # chamber centroid distance [m]

    # Visualization-only geometry. The exact outer radius/chamber width was not
    # specified with the supplied section properties, so these do not affect the
    # Cosserat solve.
    visual_outer_radius: float = 15.0e-3
    visual_chamber_line_width: float = 5.0

    # Material. Dragon Skin 30 range from PHYSICS_PARAM.md: 0.7-1.2 MPa.
    young_modulus: float = 0.95e6       # [Pa]
    poisson_ratio: float = 0.49         # near-incompressible silicone assumption

    # Pneumatic actuation calibration.
    effective_area: float = 30.0e-6     # per-chamber Aeff [m^2], user-calibrated
    pressure_scale: float = 1.0

    # External payload at tip. New robot payload is unspecified, so default to none.
    tip_load_mass: float = 0.0          # [kg]
    gravity: float = 9.81

    # Integration
    N: int = 90

    # Reference strain
    v_star: np.ndarray | None = None
    u_star: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.v_star is None:
            self.v_star = np.array([0.0, 0.0, 1.0])
        if self.u_star is None:
            self.u_star = np.array([0.0, 0.0, 0.0])

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def Izz(self) -> float:
        return self.Ixx + self.Iyy


def chamber_paths(params: RobotParams, phase_rad: float = 0.0) -> np.ndarray:
    """Return 3 chamber locations in the local cross-section frame."""
    angles = phase_rad + np.deg2rad([0.0, 120.0, 240.0])
    return np.column_stack([
        params.chamber_path_radius * np.cos(angles),
        params.chamber_path_radius * np.sin(angles),
        np.zeros(3),
    ])


def tube_mesh(
    points: np.ndarray,
    rotations: np.ndarray,
    radius: float,
    theta_count: int = 36,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a circular tube surface following the solved centerline."""
    theta = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=True)
    local_circle = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta),
    ])

    rings = np.empty((len(points), theta_count, 3), dtype=float)
    for i, (p_i, R_i) in enumerate(zip(points, rotations)):
        rings[i] = p_i + local_circle @ R_i.T

    return rings[:, :, 0], rings[:, :, 1], rings[:, :, 2]


def offset_curve(points: np.ndarray, rotations: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return a material curve at a fixed local cross-section offset."""
    return np.asarray([p_i + R_i @ offset for p_i, R_i in zip(points, rotations)])


# -----------------------------
# Stiffness and actuation
# -----------------------------

def stiffness_matrices(params: RobotParams) -> Tuple[np.ndarray, np.ndarray]:
    E = params.young_modulus
    G = params.shear_modulus
    A = params.area
    Kse = np.diag([G * A, G * A, E * A])
    Kbt = np.diag([E * params.Ixx, E * params.Iyy, G * params.Izz])
    return Kse, Kbt


def pneumatic_load_local(
    params: RobotParams,
    pressures_kpa: np.ndarray,
    phase_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent pneumatic force and moment in the local tip frame."""
    pressures_pa = np.asarray(pressures_kpa, dtype=float) * 1000.0 * params.pressure_scale
    paths = chamber_paths(params, phase_rad)

    e3 = np.array([0.0, 0.0, 1.0])
    nP = np.zeros(3)
    mP = np.zeros(3)
    for Pi, path_i in zip(pressures_pa, paths):
        Fi = Pi * params.effective_area * e3
        nP += Fi
        mP += np.cross(path_i, Fi)
    return nP, mP


# -----------------------------
# Cosserat integration and shooting
# -----------------------------

def integrate_rod(
    params: RobotParams,
    n0: np.ndarray,
    m0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate p, R, n, m from base to tip for a guessed n(0), m(0)."""
    ds = params.L / params.N

    p = np.zeros(3)
    R = np.eye(3)
    n = n0.astype(float).copy()
    m = m0.astype(float).copy()

    points = [p.copy()]
    rotations = [R.copy()]
    Kse, Kbt = stiffness_matrices(params)

    for _ in range(params.N):
        v = np.linalg.solve(Kse, R.T @ n) + params.v_star
        u = np.linalg.solve(Kbt, R.T @ m) + params.u_star

        p_dot = R @ v
        n_dot = np.zeros(3)
        m_dot = -np.cross(p_dot, n)

        p = p + ds * p_dot
        R = R @ exp_so3(u * ds)
        n = n + ds * n_dot
        m = m + ds * m_dot

        points.append(p.copy())
        rotations.append(R.copy())

    return np.asarray(points), np.asarray(rotations), n, m, R


def solve_shape(
    params: RobotParams,
    pressures_kpa: np.ndarray,
    phase_rad: float,
    initial_guess: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Solve the static shape using a shooting method."""
    if initial_guess is None:
        initial_guess = np.zeros(6)

    nP_local, mP_local = pneumatic_load_local(params, pressures_kpa, phase_rad)
    Fext_global = np.array([0.0, 0.0, -params.tip_load_mass * params.gravity])

    def residual(g: np.ndarray) -> np.ndarray:
        n0 = g[:3]
        m0 = g[3:]
        _, _, nL, mL, RL = integrate_rod(params, n0, m0)
        target_nL = RL @ nP_local + Fext_global
        target_mL = RL @ mP_local
        return np.concatenate([nL - target_nL, mL - target_mL])

    sol = root(residual, initial_guess, method="hybr", options={"xtol": 1e-8, "maxfev": 100})
    g = sol.x if np.all(np.isfinite(sol.x)) else initial_guess
    points, rotations, _, _, _ = integrate_rod(params, g[:3], g[3:])
    res_norm = float(np.linalg.norm(residual(g)))
    return points, rotations, g, bool(sol.success), res_norm


# -----------------------------
# Interactive GUI
# -----------------------------

def run_gui() -> None:
    params = RobotParams()
    last_guess = np.zeros(6)
    pressures0 = np.zeros(3)
    phase0 = 0.0

    points, rotations, last_guess, success, res_norm = solve_shape(params, pressures0, phase0, last_guess)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.08, right=0.70, bottom=0.08, top=0.93)

    tube_x, tube_y, tube_z = tube_mesh(points, rotations, params.visual_outer_radius)
    tube_surface = ax.plot_surface(
        tube_x,
        tube_y,
        tube_z,
        color="#77d992",
        alpha=0.42,
        linewidth=0,
        shade=True,
        label="Robot body",
    )
    line, = ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="#1f6f3d",
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
        label="Centerline",
    )
    tip_scatter = ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=55, label="Tip")
    chamber_scatter = ax.scatter([], [], [], s=45, marker="o", label="Chamber layout")
    chamber_lines = [
        ax.plot([], [], [], color="#1f7a3a", linewidth=params.visual_chamber_line_width, alpha=0.72)[0]
        for _ in range(3)
    ]
    air_channel_lines = [
        ax.plot([], [], [], color="#c8ff2b", linewidth=1.4, alpha=0.90)[0]
        for _ in range(3)
    ]

    ax.set_title("3-chamber soft continuum robot Cosserat model")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper left")
    status_text = ax.text2D(0.02, 0.94, "", transform=ax.transAxes)

    def set_axes_equalish() -> None:
        lim = 0.12
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0.0, 0.18)
        elev = LOCKED_VIEW_ELEVATION_DEG if LOCK_VIEW_TO_Z_ROTATION else ax.elev
        ax.view_init(elev=elev, azim=ax.azim)

    def constrain_view_to_z_rotation(_event=None) -> None:
        if not LOCK_VIEW_TO_Z_ROTATION:
            return
        ax.view_init(elev=LOCKED_VIEW_ELEVATION_DEG, azim=ax.azim)
        fig.canvas.draw_idle()

    set_axes_equalish()

    sliders = []
    y0 = 0.89
    dy = 0.055

    ax_e = fig.add_axes([0.75, y0, 0.21, 0.025])
    s_e = Slider(ax_e, "E [MPa]", 0.70, 1.20, valinit=params.young_modulus / 1e6, valstep=0.01)
    sliders.append(s_e)

    ax_aeff = fig.add_axes([0.75, y0 - dy, 0.21, 0.025])
    s_aeff = Slider(ax_aeff, "Aeff [mm^2]", 5.0, 80.0, valinit=params.effective_area * 1e6, valstep=1.0)
    sliders.append(s_aeff)

    ax_mass = fig.add_axes([0.75, y0 - 2 * dy, 0.21, 0.025])
    s_mass = Slider(ax_mass, "Tip mass [g]", 0.0, 200.0, valinit=0.0, valstep=1.0)
    sliders.append(s_mass)

    ax_phase = fig.add_axes([0.75, y0 - 3 * dy, 0.21, 0.025])
    s_phase = Slider(ax_phase, "Phase [deg]", -180.0, 180.0, valinit=0.0, valstep=5.0)
    sliders.append(s_phase)

    pressure_sliders = []
    for i in range(3):
        ax_i = fig.add_axes([0.75, y0 - (4 + i) * dy, 0.21, 0.025])
        sl = Slider(ax_i, f"P{i + 1} [kPa]", 0.0, 250.0, valinit=0.0, valstep=5.0)
        sliders.append(sl)
        pressure_sliders.append(sl)

    ax_reset = fig.add_axes([0.75, 0.10, 0.095, 0.04])
    btn_reset = Button(ax_reset, "Reset")
    ax_demo = fig.add_axes([0.865, 0.10, 0.095, 0.04])
    btn_demo = Button(ax_demo, "Demo")

    def get_pressures() -> np.ndarray:
        return np.array([sl.val for sl in pressure_sliders], dtype=float)

    def update_chamber_layout(phase_rad: float) -> None:
        nonlocal chamber_scatter
        chamber_scatter.remove()
        paths = chamber_paths(params, phase_rad)
        # Draw the cross-section layout slightly below the base plane.
        chamber_scatter = ax.scatter(
            paths[:, 0],
            paths[:, 1],
            np.full(3, -0.002),
            s=45,
            marker="o",
            label="Chamber layout",
        )

    def update_body_visuals(pts: np.ndarray, rots: np.ndarray, phase_rad: float) -> None:
        nonlocal tube_surface
        tube_surface.remove()
        tube_x, tube_y, tube_z = tube_mesh(pts, rots, params.visual_outer_radius)
        tube_surface = ax.plot_surface(
            tube_x,
            tube_y,
            tube_z,
            color="#77d992",
            alpha=0.42,
            linewidth=0,
            shade=True,
        )

        paths = chamber_paths(params, phase_rad)
        for chamber_line, air_line, path_i in zip(chamber_lines, air_channel_lines, paths):
            chamber_curve = offset_curve(pts, rots, path_i)
            chamber_line.set_data(chamber_curve[:, 0], chamber_curve[:, 1])
            chamber_line.set_3d_properties(chamber_curve[:, 2])

            air_curve = offset_curve(pts, rots, 0.82 * path_i)
            air_line.set_data(air_curve[:, 0], air_curve[:, 1])
            air_line.set_3d_properties(air_curve[:, 2])

    def update(_=None) -> None:
        nonlocal last_guess, tip_scatter
        params.young_modulus = s_e.val * 1e6
        params.effective_area = s_aeff.val * 1e-6
        params.tip_load_mass = s_mass.val / 1000.0
        phase_rad = np.deg2rad(s_phase.val)
        pressures = get_pressures()

        pts, rots, last_guess, ok, rn = solve_shape(params, pressures, phase_rad, last_guess)

        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])

        tip_scatter.remove()
        tip_scatter = ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=55, label="Tip")
        update_body_visuals(pts, rots, phase_rad)
        update_chamber_layout(phase_rad)

        nP, mP = pneumatic_load_local(params, pressures, phase_rad)
        status_text.set_text(
            f"tip = [{pts[-1,0]:+.4f}, {pts[-1,1]:+.4f}, {pts[-1,2]:+.4f}] m\n"
            f"|nP| = {np.linalg.norm(nP):.3f} N, |mP| = {np.linalg.norm(mP):.4f} N m\n"
            f"shooting success = {ok}, residual = {rn:.2e}"
        )
        set_axes_equalish()
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)

    def reset(_event) -> None:
        nonlocal last_guess
        last_guess = np.zeros(6)
        for sl in sliders:
            sl.reset()
        update()

    def demo(_event) -> None:
        nonlocal last_guess
        last_guess = np.zeros(6)
        s_e.set_val(0.95)
        s_aeff.set_val(30.0)
        s_mass.set_val(0.0)
        s_phase.set_val(0.0)
        pressure_sliders[0].set_val(150.0)
        pressure_sliders[1].set_val(0.0)
        pressure_sliders[2].set_val(0.0)
        update()

    btn_reset.on_clicked(reset)
    btn_demo.on_clicked(demo)
    fig.canvas.mpl_connect("motion_notify_event", constrain_view_to_z_rotation)
    fig.canvas.mpl_connect("button_release_event", constrain_view_to_z_rotation)

    update()
    plt.show()


if __name__ == "__main__":
    run_gui()
