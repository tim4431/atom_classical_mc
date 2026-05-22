"""Render trajectory GIFs for selected spectator-qubit fly-by cells.

Three cells, all with the static SLM at `SLM_DEPTH_UK = 500 uK`:

  1. The "representative" point used by `spectator_qubits.py` —
     AOD = REPRESENTATIVE_DEPTH_UK, d = REPRESENTATIVE_DISTANCE_UM.
  2. AOD = 500 uK (U_AOD/U_SLM = 1.0) at d/w_r = 0.9 at nominal speed —
     deep in the resonant regime where drag-out depends wildly on speed.
  3. Same cell as (2) but at 4x nominal speed, well past the resonance,
     where drag-out saturates near 1.0 (clean rip-out).

Run from the repository root:

    python3 example/spectator_qubits_dynamics.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.units import um  # noqa: E402
from src.visualization import render_animation  # noqa: E402

from example.spectator.spectator_qubits import (  # noqa: E402
    AOD_WAIST_RADIAL_UM,
    DURATION_S,
    INITIAL_TEMPERATURE_UK,
    LOSS_RADIUS_UM,
    REPRESENTATIVE_DEPTH_UK,
    REPRESENTATIVE_DISTANCE_UM,
    SLM_DEPTH_UK,
    TIMESTEP_S,
    build_aod_base,
    build_flyby_ramp,
    build_static_slm,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render_aod")
os.makedirs(RENDER_DIR, exist_ok=True)

CELLS = (
    {
        "tag": "representative",
        "aod_depth_uK": REPRESENTATIVE_DEPTH_UK,
        "distance_um": REPRESENTATIVE_DISTANCE_UM,
        "speed_factor": 1.0,
    },
    {
        "tag": "ratio1_d0p9wr",
        "aod_depth_uK": 500.0,
        "distance_um": 0.9 * AOD_WAIST_RADIAL_UM,
        "speed_factor": 1.0,
    },
    {
        "tag": "ratio1_d0p9wr_fast",
        "aod_depth_uK": 500.0,
        "distance_um": 0.9 * AOD_WAIST_RADIAL_UM,
        "speed_factor": 4.0,
    },
)

ENSEMBLE_SIZE_GIF = 80
TRAJ_STRIDE = 5
N_FRAMES = 70
FPS = 20
DPI = 95
SEED = 42


def render_one(
    tag: str, aod_depth_uK: float, distance_um: float, speed_factor: float = 1.0,
) -> str:
    duration_s = DURATION_S / float(speed_factor)
    slm = build_static_slm()
    aod_base = build_aod_base()
    ramp = build_flyby_ramp(aod_depth_uK, distance_um, duration_s=duration_s)

    config = SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=duration_s,
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
        f"[{tag}]  SLM={SLM_DEPTH_UK:.0f} uK  AOD={aod_depth_uK:.0f} uK  "
        f"d={distance_um:.2f} um  speed={speed_factor:.1f}x nominal  "
        f"(U_AOD/U_SLM={aod_depth_uK / SLM_DEPTH_UK:.2f}, "
        f"d/w_r={distance_um / AOD_WAIST_RADIAL_UM:.2f})  "
        f"loss={result.loss_fraction:.3f}  ({n_lost}/{ENSEMBLE_SIZE_GIF})"
    )

    gif_path = os.path.join(RENDER_DIR, f"spectator_qubits_dynamics_{tag}.gif")
    render_animation(
        result, slm, aod_base, ramp, gif_path,
        view="xy", n_frames=N_FRAMES, fps=FPS, dpi=DPI, grid_n=160,
        show_trails=True, trail_samples=8,
    )
    print(f"  saved: {gif_path}")
    return gif_path


def main() -> None:
    for cell in CELLS:
        render_one(**cell)


if __name__ == "__main__":
    main()
