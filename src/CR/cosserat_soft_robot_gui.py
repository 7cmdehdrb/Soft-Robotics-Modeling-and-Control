"""
Cosserat rod based quasi-static model for the soft continuum robot in
Wang & Rojas (2024), with an interactive matplotlib GUI.

What this script does
- 9 independent positive-pressure chambers are controlled by sliders.
- Growing spine length is controlled by a slider.
- The spine section uses a combined Young's modulus based on the paper.
- The rod shape is solved by shooting method + spatial integration.

Run:
    python cosserat_soft_robot_gui.py

Dependencies:
    numpy, scipy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import root


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
# Model parameters from paper
# -----------------------------

@dataclass
class RobotParams:
    # Geometry from Table I
    ro: float = 0.05          # outer radius [m]
    ri: float = 0.029         # inner radius [m]
    rc: float = 0.005         # chamber radius [m]
    r_path: float = 0.04      # chamber path radius [m]
    L: float = 0.40           # robot length [m]

    # Material from Table I
    E_cont: float = 0.507147e6    # continuum Young's modulus [Pa]
    density: float = 1300.0       # kg/m^3, not used by default

    # External tip load from Table I
    tip_load_mass: float = 0.053  # kg
    gravity: float = 9.81

    # Integration
    N: int = 80

    # Reference strain
    v_star: np.ndarray = None
    u_star: np.ndarray = None

    # Optional scale for visualization stability. 1.0 means no artificial scaling.
    # If the GUI becomes too sensitive, set pressure_scale to 0.2 ~ 0.5.
    pressure_scale: float = 1.0

    def __post_init__(self):
        if self.v_star is None:
            self.v_star = np.array([0.0, 0.0, 1.0])
        if self.u_star is None:
            self.u_star = np.array([0.0, 0.0, 0.0])

    @property
    def A_cont(self) -> float:
        # Hollow circular continuum body area
        return np.pi * (self.ro**2 - self.ri**2)

    @property
    def A_spine(self) -> float:
        # Approximation: growing spine occupies the hollow channel
        return np.pi * self.ri**2

    @property
    def A_chamber_norm(self) -> float:
        return np.pi * self.rc**2

    @property
    def I_cont_xx(self) -> float:
        # As written in the paper for circular section: pi r^4 / 4.
        # For hollow circular section: pi(ro^4 - ri^4)/4.
        return np.pi * (self.ro**4 - self.ri**4) / 4.0

    @property
    def I_spine_xx(self) -> float:
        return np.pi * self.ri**4 / 4.0


# Young's modulus of jammed growing spine from Table II [Pa]
SPINE_E_TABLE: Dict[float, float] = {
    0.00: 0.0,
    0.05: 0.318e6,
    0.10: 1.323e6,
    0.15: 2.032e6,
    0.20: 3.069e6,
    0.25: 3.763e6,
    0.30: 4.389e6,
}

# A_effect calibration factors described in Section IV-C.
# 0 cm: 1.5 A_norm. For 5 to 30 cm: 1.5, 1.7, 1.9, 2.0, 2.15, 2.4 A_norm.
AEFF_FACTOR_TABLE: Dict[float, float] = {
    0.00: 1.50,
    0.05: 1.50,
    0.10: 1.70,
    0.15: 1.90,
    0.20: 2.00,
    0.25: 2.15,
    0.30: 2.40,
}


def interp_table(x: float, table: Dict[float, float]) -> float:
    keys = np.array(sorted(table.keys()), dtype=float)
    vals = np.array([table[k] for k in keys], dtype=float)
    return float(np.interp(x, keys, vals))


def chamber_paths(params: RobotParams) -> np.ndarray:
    """Return 9 chamber locations in the local cross-section frame."""
    angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    return np.column_stack([
        params.r_path * np.cos(angles),
        params.r_path * np.sin(angles),
        np.zeros(9),
    ])


# -----------------------------
# Stiffness and actuation
# -----------------------------

def section_properties(params: RobotParams, s: float, spine_length: float) -> Tuple[float, float, float, float, float, float]:
    """Return E, G, A, Ixx, Iyy, Izz at arclength s.

    Paper-inspired approximation:
    - If s is inside the inserted growing spine portion, use volume-weighted E_eq.
    - Otherwise use continuum body E.
    - A and I are approximated using continuum-only section or combined section.
    """
    if spine_length > 1e-12 and s <= spine_length:
        Es = interp_table(spine_length, SPINE_E_TABLE)
        Ec = params.E_cont
        Ac = params.A_cont
        As = params.A_spine
        A_total = Ac + As
        Eeq = (Ac / A_total) * Ec + (As / A_total) * Es
        E = Eeq
        A = A_total
        Ixx = params.I_cont_xx + params.I_spine_xx
    else:
        E = params.E_cont
        A = params.A_cont
        Ixx = params.I_cont_xx

    G = E / 3.0
    Iyy = Ixx
    Izz = Ixx + Iyy
    return E, G, A, Ixx, Iyy, Izz


def stiffness_matrices(params: RobotParams, s: float, spine_length: float) -> Tuple[np.ndarray, np.ndarray]:
    E, G, A, Ixx, Iyy, Izz = section_properties(params, s, spine_length)
    Kse = np.diag([G * A, G * A, E * A])
    Kbt = np.diag([E * Ixx, E * Iyy, E * Izz])
    return Kse, Kbt


def effective_area(params: RobotParams, spine_length: float) -> float:
    factor = interp_table(spine_length, AEFF_FACTOR_TABLE)
    return factor * params.A_chamber_norm


def pneumatic_load_local(params: RobotParams, pressures_kpa: np.ndarray, spine_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Equivalent pneumatic force and moment in the local tip frame.

    pressures_kpa: 9 pressure values [kPa].
    Returns nP_local [N], mP_local [N m].
    """
    pressures_pa = np.asarray(pressures_kpa, dtype=float) * 1000.0 * params.pressure_scale
    Aeff = effective_area(params, spine_length)
    paths = chamber_paths(params)

    e3 = np.array([0.0, 0.0, 1.0])
    nP = np.zeros(3)
    mP = np.zeros(3)
    for Pi, path_i in zip(pressures_pa, paths):
        Fi = Pi * Aeff * e3
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
    spine_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate p, R, n, m from base to tip for a guessed n(0), m(0)."""
    N = params.N
    ds = params.L / N

    p = np.zeros(3)
    R = np.eye(3)
    n = n0.astype(float).copy()
    m = m0.astype(float).copy()

    points = [p.copy()]
    rotations = [R.copy()]

    for i in range(N):
        s = i * ds
        Kse, Kbt = stiffness_matrices(params, s, spine_length)

        # Constitutive equations. n and m are global; transform to local by R.T.
        v = np.linalg.solve(Kse, R.T @ n) + params.v_star
        u = np.linalg.solve(Kbt, R.T @ m) + params.u_star

        p_dot = R @ v

        # No distributed body force or distributed moment by default.
        # The 53 g load is included as a tip boundary load in shooting.
        f = np.zeros(3)
        l = np.zeros(3)
        n_dot = -f
        m_dot = -np.cross(p_dot, n) - l

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
    spine_length: float,
    initial_guess: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Solve BVP using shooting method.

    Returns points, rotations, solution_guess, success, residual_norm.
    """
    if initial_guess is None:
        initial_guess = np.zeros(6)

    nP_local, mP_local = pneumatic_load_local(params, pressures_kpa, spine_length)
    Fext_global = np.array([0.0, 0.0, -params.tip_load_mass * params.gravity])

    def residual(g: np.ndarray) -> np.ndarray:
        n0 = g[:3]
        m0 = g[3:]
        _, _, nL, mL, RL = integrate_rod(params, n0, m0, spine_length)

        # Convert local pneumatic tip load to global using final tip orientation.
        target_nL = RL @ nP_local + Fext_global
        target_mL = RL @ mP_local
        return np.concatenate([nL - target_nL, mL - target_mL])

    sol = root(residual, initial_guess, method="hybr", options={"xtol": 1e-7, "maxfev": 80})
    g = sol.x if np.all(np.isfinite(sol.x)) else initial_guess
    points, rotations, _, _, _ = integrate_rod(params, g[:3], g[3:], spine_length)
    res_norm = float(np.linalg.norm(residual(g)))
    return points, rotations, g, bool(sol.success), res_norm


# -----------------------------
# Interactive GUI
# -----------------------------

def run_gui() -> None:
    params = RobotParams()
    pressures0 = np.zeros(9)
    spine0 = 0.0
    last_guess = np.zeros(6)

    points, _, last_guess, success, res_norm = solve_shape(params, pressures0, spine0, last_guess)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.08, right=0.72, bottom=0.08, top=0.93)

    line, = ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=3, label="Cosserat centerline")
    tip_scatter = ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=50, label="Tip")
    spine_line, = ax.plot([], [], [], linewidth=7, alpha=0.3, label="Inserted spine section")

    ax.set_title("Soft continuum robot Cosserat model")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper left")

    def set_axes_equalish():
        lim = 0.45
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0.0, 0.55)
        ax.view_init(elev=25, azim=-60)

    set_axes_equalish()

    status_text = ax.text2D(0.02, 0.94, "", transform=ax.transAxes)

    slider_axes = []
    sliders = []
    y0 = 0.88
    dy = 0.055

    # Spine length slider
    ax_spine = fig.add_axes([0.76, y0, 0.20, 0.025])
    s_spine = Slider(ax_spine, "Spine [cm]", 0.0, 30.0, valinit=0.0, valstep=5.0)
    sliders.append(s_spine)
    slider_axes.append(ax_spine)

    # Chamber pressure sliders
    for i in range(9):
        ax_i = fig.add_axes([0.76, y0 - (i + 1) * dy, 0.20, 0.025])
        sl = Slider(ax_i, f"P{i+1} [kPa]", 0.0, 250.0, valinit=0.0, valstep=5.0)
        sliders.append(sl)
        slider_axes.append(ax_i)

    # Buttons
    ax_reset = fig.add_axes([0.76, 0.07, 0.09, 0.04])
    btn_reset = Button(ax_reset, "Reset")
    ax_demo = fig.add_axes([0.87, 0.07, 0.09, 0.04])
    btn_demo = Button(ax_demo, "Demo")

    def get_pressures() -> np.ndarray:
        return np.array([sliders[i + 1].val for i in range(9)], dtype=float)

    def update(_=None):
        nonlocal last_guess, tip_scatter
        pressures = get_pressures()
        spine_length = s_spine.val / 100.0
        pts, _, last_guess, ok, rn = solve_shape(params, pressures, spine_length, last_guess)

        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])

        # Update tip marker
        tip_scatter.remove()
        tip_scatter = ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=50, label="Tip")

        # Highlight inserted spine portion along the computed centerline
        if spine_length > 1e-9:
            n_spine = max(2, int(round(spine_length / params.L * params.N)))
            spine_pts = pts[: n_spine + 1]
            spine_line.set_data(spine_pts[:, 0], spine_pts[:, 1])
            spine_line.set_3d_properties(spine_pts[:, 2])
        else:
            spine_line.set_data([], [])
            spine_line.set_3d_properties([])

        status_text.set_text(
            f"tip = [{pts[-1,0]:+.3f}, {pts[-1,1]:+.3f}, {pts[-1,2]:+.3f}] m\n"
            f"shooting success = {ok}, residual = {rn:.2e}"
        )
        set_axes_equalish()
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)

    def reset(_event):
        nonlocal last_guess
        last_guess = np.zeros(6)
        s_spine.reset()
        for sl in sliders[1:]:
            sl.reset()
        update()

    def demo(_event):
        nonlocal last_guess
        last_guess = np.zeros(6)
        s_spine.set_val(20.0)
        # Pressurize 3 adjacent chambers as in the paper's bending experiment idea.
        vals = [150.0, 150.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for sl, val in zip(sliders[1:], vals):
            sl.set_val(val)
        update()

    btn_reset.on_clicked(reset)
    btn_demo.on_clicked(demo)

    update()
    plt.show()


if __name__ == "__main__":
    run_gui()
