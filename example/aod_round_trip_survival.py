"""AOD round-trip survival vs. drag time.

Translation of `tmp/aod_slm_movement_v2/aod_slm_movement_v2/single_round_trip.ipynb`.
A single AOD trap holds the atom at the origin, drags it `drag_distance` away,
holds, drags it back, and the simulation ends with the AOD at rest at the
origin. We sweep `t_drag` and report survival in the v2 sense:

    n_init = #atoms with KE + V_AOD(r_init, t=0) < 0
    n_sol  = #atoms with KE + V_AOD(r_final, t=0) < 0     (AOD is back at origin)
    survival = n_sol / n_init

Run from the repository root:

    python3 example/aod_round_trip_survival.py

Saves a survival-vs-drag-time plot next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import bound_to_trap  # noqa: E402
from src.constants import (  # noqa: E402
    ATOMIC_MASS_UNIT_KG,
    BOLTZMANN_CONSTANT_J_PER_K,
)
from src.ramp import RampSequence, arb_fifth_poly  # noqa: E402
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.trap import AstigmaticAODTrap  # noqa: E402

HERE = os.path.dirname(__file__)

# v2 used Yb-171 trap parameters; keep that mass.
ATOM_MASS_KG = 171.0 * ATOMIC_MASS_UNIT_KG
PLANCK_CONST_J_S = 6.626070e-34


def hz_to_uK(depth_hz: float) -> float:
    return depth_hz * PLANCK_CONST_J_S / BOLTZMANN_CONSTANT_J_PER_K / 1.0e-6


# --- Trap parameters from single_round_trip.ipynb cell 2 --------------------
T_UNIT_S = 11.0e-6
X_UNIT_M = 4.8e-6
T_HOLD_S = T_UNIT_S
DRAG_DISTANCE_M = 10.0 * X_UNIT_M  # 48 um

AOD_WAIST_M = 500.0e-9
AOD_DEPTH_UK = hz_to_uK(2.8e6)  # ~134.4 uK; trap freq ~ 51 kHz per v2 comment.
AOD_WAVELENGTH_M = 487.0e-9
BETA = 1.5625
DXDT2Z = 0.025 / 4.0 / 650.0

# --- Numerics ---------------------------------------------------------------
INITIAL_TEMPERATURE_UK = 10.0
ENSEMBLE_SIZE = 500
TIMESTEP_S = 1.0e-6
RANDOM_SEED = 42
LOSS_RADIUS_M = 200.0e-6

# v2 sweeps t_drag from 50 us to 600 us in 10 steps.
T_DRAG_GRID_S = np.linspace(50.0e-6, 600.0e-6, 10)


def build_aod_ramp(t_drag_s: float) -> RampSequence:
    times_s = np.array(
        [0.0, t_drag_s, t_drag_s + T_HOLD_S, 2.0 * t_drag_s + T_HOLD_S],
        dtype=float,
    )
    centers_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [DRAG_DISTANCE_M, 0.0, 0.0],
            [DRAG_DISTANCE_M, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    depths_uK = np.array(
        [AOD_DEPTH_UK, AOD_DEPTH_UK, AOD_DEPTH_UK, AOD_DEPTH_UK], dtype=float
    )
    return RampSequence(
        times_s=times_s,
        centers_m=centers_m,
        depths_uK=depths_uK,
        position_profile=arb_fifth_poly(BETA),
    )


def run_one(t_drag_s: float, dxdt2z: float) -> dict:
    ramp = build_aod_ramp(t_drag_s)
    aod = AstigmaticAODTrap(
        waist_radial_m=AOD_WAIST_M,
        wavelength_m=AOD_WAVELENGTH_M,
        ramp=ramp,
        dxdt2z=dxdt2z,
        name="AOD",
    )
    duration_s = 2.0 * t_drag_s + T_HOLD_S
    # Disable the lab-frame energy loss check: peak drag velocity exceeds the
    # cylindrical escape velocity, so co-moving atoms register as unbound by
    # the lab-frame `KE + V >= 0` rule. Survival is recovered post-hoc against
    # the AOD at rest (which is the case at both t=0 and t=duration).
    config = SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=duration_s,
        ensemble_size=ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
        mass_kg=ATOM_MASS_KG,
        loss_radius_m=LOSS_RADIUS_M,
        track_energy_loss=False,
    )
    result = run_simulation([aod], config)

    bound_initial = bound_to_trap(
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
        aod,
        mass_kg=config.mass_kg,
        time_s=0.0,
    )
    bound_final = bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        aod,
        mass_kg=config.mass_kg,
        time_s=0.0,
    )
    n_init = int(np.sum(bound_initial))
    n_sol = int(np.sum(bound_final & ~result.lost))
    return {
        "t_drag_s": t_drag_s,
        "n_init": n_init,
        "n_sol": n_sol,
        "survival": n_sol / max(n_init, 1),
    }


def sweep(dxdt2z: float = 0.0) -> list[dict]:
    rows: list[dict] = []
    label = "lensing" if dxdt2z != 0.0 else "no-lensing"
    print(f"  sweep t_drag with {label} (dxdt2z = {dxdt2z:.3e}):")
    for t_drag in T_DRAG_GRID_S:
        row = run_one(t_drag, dxdt2z)
        rows.append(row)
        print(
            f"    t_drag = {t_drag*1e6:6.1f} us  n_init = {row['n_init']:3d}  "
            f"n_sol = {row['n_sol']:3d}  survival = {row['survival']:.3f}"
        )
    return rows


def plot_survival(no_lens_rows, with_lens_rows):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
    for rows, label, marker, color in (
        (no_lens_rows, "no lensing (dxdt2z=0)", "o", "tab:blue"),
        (with_lens_rows, "with lensing", "s", "tab:red"),
    ):
        ts = np.array([row["t_drag_s"] for row in rows])
        survival = np.array([row["survival"] for row in rows])
        n_init = np.array([row["n_init"] for row in rows])
        err = np.sqrt(np.maximum(survival * (1.0 - survival), 0.0) / np.maximum(n_init, 1))
        ax.errorbar(
            ts * 1.0e6,
            survival,
            yerr=err,
            fmt=marker,
            color=color,
            markerfacecolor="none",
            label=label,
        )
    ax.set_xlabel("Drag time t_drag  [us]")
    ax.set_ylabel("Survival ratio")
    ax.set_title(
        f"AOD round-trip survival\n"
        f"AOD waist {AOD_WAIST_M*1e9:.0f} nm, depth {AOD_DEPTH_UK:.1f} uK, "
        f"drag {DRAG_DISTANCE_M*1e6:.0f} um, T = {INITIAL_TEMPERATURE_UK:.1f} uK"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    return figure, ax


def main() -> None:
    print("AOD round-trip survival sweep")
    print(
        f"  AOD: waist={AOD_WAIST_M*1e9:.0f} nm, depth={AOD_DEPTH_UK:.2f} uK, "
        f"lambda={AOD_WAVELENGTH_M*1e9:.0f} nm"
    )
    print(
        f"  drag distance = {DRAG_DISTANCE_M*1e6:.1f} um, "
        f"hold time = {T_HOLD_S*1e6:.1f} us, beta = {BETA}"
    )
    print(
        f"  ensemble = {ENSEMBLE_SIZE}, T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"dt = {TIMESTEP_S*1e6:.1f} us"
    )
    no_lens_rows = sweep(dxdt2z=0.0)
    with_lens_rows = sweep(dxdt2z=DXDT2Z)

    figure, _ = plot_survival(no_lens_rows, with_lens_rows)
    out_path = os.path.join(HERE, "aod_round_trip_survival.png")
    figure.savefig(out_path, dpi=180)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
