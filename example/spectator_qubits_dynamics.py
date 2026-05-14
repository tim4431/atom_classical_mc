"""Render trajectory GIFs at three depth scales for one spectator-qubit cell.

Picks the `(U_AOD/U_SLM = 0.71, d/w_r = 0.60)` cell where the heatmap in
`spectator_qubits.py` shows non-monotonic drag-out vs overall depth scale
(0.61 -> 0.07 -> 0.34 as scale goes 0.5 -> 1.0 -> 2.0). For each scale
this script runs a small ensemble with stored trajectories and writes a
GIF to `example/render/`.

Run from the repository root:

    python3 example/spectator_qubits_dynamics.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.units import um  # noqa: E402
from src.visualization import render_animation  # noqa: E402

from example.spectator_qubits import (  # noqa: E402
    AOD_WAIST_RADIAL_UM,
    DURATION_S,
    INITIAL_TEMPERATURE_UK,
    LOSS_RADIUS_UM,
    SLM_DEPTH_UK,
    TIMESTEP_S,
    build_aod_base,
    build_flyby_ramp,
    build_static_slm,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

CELL_RATIO = 50.0 / 70.0  # U_AOD / U_SLM
CELL_D_OVER_WR = 0.60     # d / w_r

SCALE_FACTORS = (0.5, 1.0, 2.0)
ENSEMBLE_SIZE_GIF = 80
TRAJ_STRIDE = 5
N_FRAMES = 70
FPS = 20
DPI = 95
SEED = 42


def render_one(scale: float) -> str:
    slm_depth_uK = scale * SLM_DEPTH_UK
    aod_depth_uK = CELL_RATIO * slm_depth_uK
    distance_um = CELL_D_OVER_WR * AOD_WAIST_RADIAL_UM

    slm = build_static_slm(slm_depth_uK=slm_depth_uK)
    aod_base = build_aod_base()
    ramp = build_flyby_ramp(aod_depth_uK, distance_um)

    config = SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=DURATION_S,
        ensemble_size=ENSEMBLE_SIZE_GIF,
        random_seed=SEED,
        loss_radius_m=float(um(LOSS_RADIUS_UM)),
        initial_center_m=um([0.0, 0.0, 0.0]),
        store_trajectories=True,
        trajectory_stride=TRAJ_STRIDE,
    )
    result = run_simulation(slm, aod_base, ramp, config)

    n_lost = int(result.lost.sum())
    print(
        f"scale={scale:>4.2f}  SLM={slm_depth_uK:>5.1f} uK  "
        f"AOD={aod_depth_uK:>5.1f} uK  d={distance_um:.2f} um  "
        f"loss={result.loss_fraction:.3f}  ({n_lost}/{ENSEMBLE_SIZE_GIF})"
    )

    gif_path = os.path.join(
        RENDER_DIR, f"spectator_qubits_dynamics_scale{scale:.2f}.gif"
    )
    render_animation(
        result, slm, aod_base, ramp, gif_path,
        view="xy", n_frames=N_FRAMES, fps=FPS, dpi=DPI, grid_n=160,
        show_trails=True, trail_samples=8,
    )
    print(f"  saved: {gif_path}")
    return gif_path


def main() -> None:
    print(
        f"Rendering dynamics GIFs for cell "
        f"U_AOD/U_SLM={CELL_RATIO:.3f}, d/w_r={CELL_D_OVER_WR:.2f} "
        f"across scales {SCALE_FACTORS}\n"
    )
    for scale in SCALE_FACTORS:
        render_one(scale)


if __name__ == "__main__":
    main()
