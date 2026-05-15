"""Speed scan: drag-out / loss / AOD-capture vs AOD fly-by speed.

Fixes one cell — `(AOD = 1000 uK, d/w_r = 0.9)` — and sweeps the fly-by
speed across a wide range. Reveals the resonance pattern from interaction
time matching integer multiples of the SLM trap period.

Run from the repository root:

    python3 example/spectator_qubits_speed_scan.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.constants import RB87_MASS_KG  # noqa: E402
from src.units import microkelvin_to_joule  # noqa: E402

from example.spectator_qubits import (  # noqa: E402
    AOD_WAIST_RADIAL_UM,
    DURATION_S,
    FLYBY_HALF_LENGTH_UM,
    INITIAL_TEMPERATURE_UK,
    SLM_DEPTH_UK,
    SLM_WAIST_RADIAL_UM,
    aod_capture_probability,
    drag_out_probability,
    run_flyby,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

CELL_AOD_DEPTH_UK = 1000.0
CELL_DISTANCE_UM = 0.9 * AOD_WAIST_RADIAL_UM
SPEED_FACTORS = np.geomspace(0.25, 4.0, 500)  # 500 log-spaced points, 0.25x .. 4x
N_ATOMS = 1000


def slm_trap_period_s() -> float:
    U = float(microkelvin_to_joule(SLM_DEPTH_UK))
    w = SLM_WAIST_RADIAL_UM * 1.0e-6
    omega = (4.0 * U / (RB87_MASS_KG * w * w)) ** 0.5
    return 2.0 * np.pi / omega


def main() -> None:
    import matplotlib.pyplot as plt

    nominal_speed_um_per_ms = 2.0 * FLYBY_HALF_LENGTH_UM / (DURATION_S * 1.0e3)
    T_trap_us = slm_trap_period_s() * 1.0e6
    speeds_um_per_ms = SPEED_FACTORS * nominal_speed_um_per_ms

    print(
        f"Speed scan at AOD={CELL_AOD_DEPTH_UK:.0f} uK, "
        f"d={CELL_DISTANCE_UM:.2f} um (d/w_r={CELL_DISTANCE_UM / AOD_WAIST_RADIAL_UM:.2f})"
    )
    print(
        f"  SLM = {SLM_DEPTH_UK:.0f} uK, T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"SLM trap period ~ {T_trap_us:.2f} us"
    )
    print(
        f"  speeds: {speeds_um_per_ms[0]:.2f} .. {speeds_um_per_ms[-1]:.2f} um/ms "
        f"({len(speeds_um_per_ms)} values, log-spaced)"
    )

    drag = np.zeros_like(speeds_um_per_ms)
    loss = np.zeros_like(speeds_um_per_ms)
    aod_cap = np.zeros_like(speeds_um_per_ms)

    for k, sf in enumerate(SPEED_FACTORS):
        duration_s = DURATION_S / float(sf)
        result, ramp, slm, aod_base = run_flyby(
            CELL_AOD_DEPTH_UK, CELL_DISTANCE_UM,
            duration_s=duration_s, ensemble_size=N_ATOMS,
        )
        drag[k] = drag_out_probability(result, slm, RB87_MASS_KG)
        loss[k] = result.loss_fraction
        aod_cap[k] = aod_capture_probability(
            result, aod_base, ramp, duration_s, RB87_MASS_KG
        )
        t_int_us = AOD_WAIST_RADIAL_UM / speeds_um_per_ms[k] * 1.0e3
        print(
            f"  v={speeds_um_per_ms[k]:6.2f} um/ms  t_int={t_int_us:5.2f} us  "
            f"t_int/T_trap={t_int_us / T_trap_us:5.2f}  "
            f"P(drag)={drag[k]:.3f}  P(lost)={loss[k]:.3f}  P(AOD)={aod_cap[k]:.3f}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(speeds_um_per_ms, drag, label="P(drag-out)", color="C0", linewidth=1.0)
    ax.plot(speeds_um_per_ms, loss, label="P(lost to vacuum)", color="C3", linewidth=1.0)
    ax.plot(speeds_um_per_ms, aod_cap, label="P(captured by AOD)", color="C2", linewidth=1.0)
    ax.axvline(
        nominal_speed_um_per_ms, color="0.5", linestyle="--", linewidth=1,
        label=f"nominal speed = {nominal_speed_um_per_ms:.1f} um/ms",
    )
    ax.set_xscale("log")
    ax.set_xlabel("AOD fly-by speed  [um/ms]")
    ax.set_ylabel("Probability  (N = {} atoms per speed)".format(N_ATOMS))
    ax.set_title(
        f"Speed scan at AOD={CELL_AOD_DEPTH_UK:.0f} uK, "
        f"d/w_r={CELL_DISTANCE_UM / AOD_WAIST_RADIAL_UM:.2f}\n"
        f"(SLM={SLM_DEPTH_UK:.0f} uK, T={INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"SLM trap period ~ {T_trap_us:.1f} us)"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize="small")

    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda v: 1.0e3 / np.where(v > 0, v, np.nan) / T_trap_us,
            lambda r: 1.0e3 / np.where(r > 0, r, np.nan) / T_trap_us,
        ),
    )
    secax.set_xlabel(r"$t_\mathrm{int}\,/\,T_\mathrm{trap}$"
                     r"   ($t_\mathrm{int} = w_r / v$)")

    out_path = os.path.join(RENDER_DIR, "spectator_qubits_speed_scan.png")
    fig.savefig(out_path, dpi=180)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
