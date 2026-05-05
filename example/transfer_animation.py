"""Render a GIF movie of the SLM-to-AOD transfer.

Run from the repository root:

    python3 example/transfer_animation.py

Produces:
  - transfer_frame_0p20ms.png  (single still at one timestep)
  - transfer.gif               (full ramp animation, split xy + 3D view)
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Re-use the example's geometry so the animation matches the static plots.
from example.slm_to_aod_transfer import build_transfer_problem  # noqa: E402

from src.simulation import run_simulation  # noqa: E402
from src.visualization import draw_frame, render_animation  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)


def main() -> None:
    slm_trap, aod_trap_base, ramp, config = build_transfer_problem()
    result = run_simulation(slm_trap, aod_trap_base, ramp, config)
    print(
        f"survival = {result.survival_probability:.3f}  "
        f"mean energy gain = {result.mean_energy_gain_uK:.2f} uK"
    )

    still_t = 0.5 * (ramp.start_time_s + ramp.end_time_s)
    figure, _ = draw_frame(
        still_t, result, slm_trap, aod_trap_base, ramp,
        view="split", grid_n=220,
    )
    still_path = os.path.join(HERE, f"transfer_frame_{still_t * 1e3:.2f}ms.png")
    figure.savefig(still_path, dpi=120, facecolor="white")
    plt.close(figure)
    print(f"saved still: {still_path}")

    gif_path = os.path.join(HERE, "transfer.gif")
    render_animation(
        result, slm_trap, aod_trap_base, ramp, gif_path,
        view="split", n_frames=80, fps=18, dpi=85, grid_n=160,
        show_progress=True,
    )
    print(f"saved gif: {gif_path}")


if __name__ == "__main__":
    main()
