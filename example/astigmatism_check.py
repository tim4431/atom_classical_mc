"""Sanity check for `AstigmaticAODTrap`.

Translation of `tmp/aod_slm_movement_v2/aod_slm_movement_v2/astigmatism_analysis.ipynb`.
The original notebook does two things:

1. Plot `U(0, 0, z)` for several axial focal offsets `z01` to show how the
   astigmatic shift deforms the well along z.
2. Verify analytically computed forces against central-difference gradients.

Run from the repository root:

    python3 example/astigmatism_check.py

Saves `astigmatism_check.png` into the `render/` subdir next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.ramp import RampSequence  # noqa: E402
from src.trap import AstigmaticAODTrap  # noqa: E402
from src.units import joule_to_microkelvin  # noqa: E402

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

WAIST_RADIAL_M = 500.0e-9
WAVELENGTH_M = 486.0e-9
DEPTH_UK = 100.0


def astigmatic_with_focal_offsets(
    z01_m: float, z02_m: float
) -> AstigmaticAODTrap:
    """Build an AstigmaticAODTrap whose focal offsets at t = 0.5 are (z01, z02)
    AND whose lab-frame center coincides with the origin at t = 0.5.

    A linear ramp from `(-z01, -z02, 0)` at t=0 to `(+z01, +z02, 0)` at t=1
    has midpoint center = origin and constant velocity `(2 z01, 2 z02, 0)`.
    With `dxdt2z = 0.5`, the velocity-coupled focal offsets at t = 0.5 are
    `(z01, z02)`, exactly matching v2's `gaussian_trap_potential(...)` call
    where the trap is at the origin.
    """

    ramp = RampSequence(
        times_s=[0.0, 1.0],
        centers_m=[[-z01_m, -z02_m, 0.0], [z01_m, z02_m, 0.0]],
        depths_uK=[DEPTH_UK, DEPTH_UK],
    )
    return AstigmaticAODTrap(
        waist_radial_m=WAIST_RADIAL_M,
        wavelength_m=WAVELENGTH_M,
        ramp=ramp,
        dxdt2z=0.5,
    )


def axial_potential_curves():
    """Reproduce notebook cell 2: U(0, 0, z) for several z01."""

    zR = np.pi * WAIST_RADIAL_M**2 / WAVELENGTH_M
    z01_list = np.arange(0, 5, 1) * zR  # 0, zR, 2zR, 3zR, 4zR
    z_list = np.linspace(-10.0 * zR, 10.0 * zR, 401)
    curves = []
    for z01 in z01_list:
        trap = astigmatic_with_focal_offsets(z01, 0.0)
        positions = np.zeros((z_list.size, 3))
        positions[:, 2] = z_list
        # Evaluate at t = 0.5 (mid-segment with constant velocity).
        u_joule = trap.potential(positions, time_s=0.5)
        curves.append((z01, joule_to_microkelvin(u_joule)))
    return zR, z_list, curves


def force_finite_difference_check(num_samples: int = 100, seed: int = 0) -> dict:
    """Reproduce notebook cell 4: compare analytic F to central-difference."""

    zR = np.pi * WAIST_RADIAL_M**2 / WAVELENGTH_M
    rng = np.random.default_rng(seed)
    dx = 1.0e-3 * WAIST_RADIAL_M
    max_relative_error = 0.0
    max_absolute_error = 0.0
    for _ in range(num_samples):
        x = float(rng.uniform(-5.0, 5.0)) * WAIST_RADIAL_M
        y = float(rng.uniform(-5.0, 5.0)) * WAIST_RADIAL_M
        z = float(rng.uniform(-5.0, 5.0)) * zR
        z01 = float(rng.uniform(-5.0, 5.0)) * zR
        z02 = float(rng.uniform(-5.0, 5.0)) * zR
        trap = astigmatic_with_focal_offsets(z01, z02)

        position = np.array([x, y, z])
        f_analytic = trap.force(position, time_s=0.5)
        f_numerical = np.zeros(3, dtype=float)
        basis = np.eye(3)
        for axis in range(3):
            plus = trap.potential(position + dx * basis[axis], time_s=0.5)
            minus = trap.potential(position - dx * basis[axis], time_s=0.5)
            f_numerical[axis] = -(plus - minus) / (2.0 * dx)
        diff = np.abs(f_analytic - f_numerical)
        scale = np.maximum(np.abs(f_analytic), np.abs(f_numerical)) + 1.0e-300
        max_relative_error = max(max_relative_error, float(np.max(diff / scale)))
        max_absolute_error = max(max_absolute_error, float(np.max(diff)))
    return {
        "samples": num_samples,
        "max_relative_error": max_relative_error,
        "max_absolute_error_n": max_absolute_error,
    }


def plot_summary(zR, z_list, curves, fd_summary):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
    for z01, u_uK in curves:
        ax.plot(
            z_list / zR,
            u_uK,
            label=f"z01 = {z01 / zR:.0f} z_R",
        )
    ax.set_xlabel("z / z_R")
    ax.set_ylabel("U(0, 0, z)  [uK]")
    ax.set_title(
        "Astigmatic Gaussian axial potential\n"
        f"max |F_analytic - F_fdiff| / |F| = {fd_summary['max_relative_error']:.2e}"
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, linestyle=":", alpha=0.5)
    return figure, ax


def main() -> None:
    zR, z_list, curves = axial_potential_curves()
    fd_summary = force_finite_difference_check()

    print("Astigmatism sanity check")
    print(f"  waist_radial_m = {WAIST_RADIAL_M * 1.0e9:.0f} nm")
    print(f"  wavelength_m   = {WAVELENGTH_M * 1.0e9:.0f} nm")
    print(f"  rayleigh_length = {zR * 1.0e6:.3f} um")
    print("Force finite-difference comparison:")
    print(f"  samples              = {fd_summary['samples']}")
    print(f"  max relative error   = {fd_summary['max_relative_error']:.3e}")
    print(f"  max absolute error N = {fd_summary['max_absolute_error_n']:.3e}")

    figure, _ = plot_summary(zR, z_list, curves, fd_summary)
    out_path = os.path.join(RENDER_DIR, "astigmatism_check.png")
    figure.savefig(out_path, dpi=180)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
