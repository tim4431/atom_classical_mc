"""Compare GriddedTrap dynamics against the analytical GaussianTrap.

The two traps describe the same static Gaussian beam, with identical waists
and depth. We sample a single initial ensemble from the analytical trap's
harmonic approximation, then propagate both traps with a hand-rolled
velocity-Verlet loop using those same initial conditions. Identical IC +
identical integrator => any observed divergence is purely the force/potential
error introduced by tabulating the field.

Run from the repository root:

    python3 example/gridded_vs_analytical.py

By default both interpolation modes ("tricubic" and "trilinear") are run on
the same grid, so the plots show the relative quality of the two. Pass
``--save-plot`` to write a PNG instead of opening a Matplotlib window.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.constants import RB87_MASS_KG  # noqa: E402
from src.sampling import (  # noqa: E402
    sample_thermal_positions_harmonic,
    sample_thermal_velocities,
)
from src.trap import (  # noqa: E402
    GaussianTrap,
    GriddedTrap,
    total_force,
    total_potential,
)
from src.units import joule_to_microkelvin, um  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "render")


def run_velocity_verlet(
    trap,
    positions: np.ndarray,
    velocities: np.ndarray,
    timestep_s: float,
    n_steps: int,
    mass_kg: float,
    snapshots: int,
) -> dict:
    """Propagate a static trap with vanilla velocity-Verlet. Time-independent
    forces -- no `time_s` plumbing needed.
    """

    p = positions.copy()
    v = velocities.copy()
    stride = max(1, n_steps // snapshots)

    times = [0.0]
    pos_history = [p.copy()]
    vel_history = [v.copy()]
    ke_history = [_kinetic_energy_j(v, mass_kg)]
    pe_history = [total_potential(trap, p)]

    accel = total_force(trap, p) / mass_kg
    for step in range(1, n_steps + 1):
        p = p + v * timestep_s + 0.5 * accel * timestep_s * timestep_s
        accel_next = total_force(trap, p) / mass_kg
        v = v + 0.5 * (accel + accel_next) * timestep_s
        accel = accel_next
        if step % stride == 0 or step == n_steps:
            times.append(step * timestep_s)
            pos_history.append(p.copy())
            vel_history.append(v.copy())
            ke_history.append(_kinetic_energy_j(v, mass_kg))
            pe_history.append(total_potential(trap, p))

    return {
        "times": np.asarray(times),
        "positions": np.stack(pos_history),  # (T, N, 3)
        "velocities": np.stack(vel_history),
        "kinetic_j": np.stack(ke_history),  # (T, N)
        "potential_j": np.stack(pe_history),  # (T, N)
    }


def _kinetic_energy_j(velocities: np.ndarray, mass_kg: float) -> np.ndarray:
    return 0.5 * mass_kg * np.sum(velocities * velocities, axis=-1)


def temperature_uK_from_kinetic(kinetic_j: np.ndarray) -> np.ndarray:
    """Map per-atom KE to a sample kinetic temperature in uK, per time slice."""
    return joule_to_microkelvin(np.mean(kinetic_j, axis=-1)) * 2.0 / 3.0


def summarize(label: str, traj: dict, ref: dict | None = None) -> None:
    ke_uK = temperature_uK_from_kinetic(traj["kinetic_j"])
    total_per_atom = traj["kinetic_j"] + traj["potential_j"]
    e_total_mean_j = np.mean(total_per_atom, axis=-1)
    e_total_drift_uK = joule_to_microkelvin(e_total_mean_j - e_total_mean_j[0])

    print(f"-- {label} --")
    print(f"  initial / final KE temperature [uK]: "
          f"{ke_uK[0]:.4f} / {ke_uK[-1]:.4f}")
    print(f"  total-energy drift from t=0 [uK / atom]: "
          f"min={float(np.min(e_total_drift_uK)):+.4f}, "
          f"max={float(np.max(e_total_drift_uK)):+.4f}, "
          f"final={float(e_total_drift_uK[-1]):+.4f}")
    if ref is not None:
        dr = np.linalg.norm(traj["positions"] - ref["positions"], axis=-1)
        dv = np.linalg.norm(traj["velocities"] - ref["velocities"], axis=-1)
        print(f"  per-atom |Δr| vs reference [um]: "
              f"mean={1e6 * float(np.mean(dr[-1])):.4f}, "
              f"max={1e6 * float(np.max(dr[-1])):.4f}")
        print(f"  per-atom |Δv| vs reference [mm/s]: "
              f"mean={1e3 * float(np.mean(dv[-1])):.4f}, "
              f"max={1e3 * float(np.max(dv[-1])):.4f}")


def plot_comparison(results: dict, args, output_path: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Pick the first atom for trajectory snapshot.
    atom_index = 0
    for label, traj in results.items():
        t_us = traj["times"] * 1e6
        ax = axes[0, 0]
        ax.plot(t_us, 1e6 * traj["positions"][:, atom_index, 0], label=f"{label}")
        ax = axes[0, 1]
        ax.plot(t_us, 1e6 * traj["positions"][:, atom_index, 2], label=f"{label}")
        ax = axes[1, 0]
        ax.plot(t_us, temperature_uK_from_kinetic(traj["kinetic_j"]), label=label)
        ax = axes[1, 1]
        total = np.mean(traj["kinetic_j"] + traj["potential_j"], axis=-1)
        ax.plot(t_us, joule_to_microkelvin(total - total[0]), label=label)

    axes[0, 0].set_title(f"Atom {atom_index}: x(t)")
    axes[0, 0].set_xlabel("time [µs]")
    axes[0, 0].set_ylabel("x [µm]")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title(f"Atom {atom_index}: z(t)")
    axes[0, 1].set_xlabel("time [µs]")
    axes[0, 1].set_ylabel("z [µm]")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title("Ensemble kinetic temperature")
    axes[1, 0].set_xlabel("time [µs]")
    axes[1, 0].set_ylabel("KE temperature [µK]")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title("Mean total energy drift")
    axes[1, 1].set_xlabel("time [µs]")
    axes[1, 1].set_ylabel("⟨E_total⟩ - ⟨E_total⟩(0)  [µK]")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Gaussian trap: analytical vs gridded "
        f"(waist {args.waist_r_um:.2f}/{args.waist_z_um:.1f} µm, "
        f"depth {args.depth_uK:.0f} µK, "
        f"grid {args.n_radial}×{args.n_radial}×{args.n_axial})"
    )

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=140)
        print(f"Saved plot to {output_path}")


def build_traps(args):
    analytical = GaussianTrap(
        center_m=[0.0, 0.0, 0.0],
        waist_radial_m=float(um(args.waist_r_um)),
        waist_axial_m=float(um(args.waist_z_um)),
        depth_uK=args.depth_uK,
    )
    half = np.array([
        args.box_radial_waists * analytical.waist_radial_m,
        args.box_radial_waists * analytical.waist_radial_m,
        args.box_axial_waists * analytical.waist_axial_m,
    ])
    bounds = np.stack([-half, half], axis=-1)
    shape = (args.n_radial, args.n_radial, args.n_axial)
    tricubic = GriddedTrap.from_trap(
        analytical, bounds_m=bounds, shape=shape, interpolation="tricubic"
    )
    trilinear = GriddedTrap.from_trap(
        analytical, bounds_m=bounds, shape=shape, interpolation="trilinear"
    )
    return analytical, tricubic, trilinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waist-r-um", type=float, default=1.0)
    parser.add_argument("--waist-z-um", type=float, default=4.0)
    parser.add_argument("--depth-uK", type=float, default=500.0)
    parser.add_argument("--temperature-uK", type=float, default=10.0)
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--duration-us", type=float, default=200.0)
    parser.add_argument("--timestep-ns", type=float, default=10.0)
    parser.add_argument("--snapshots", type=int, default=400)
    parser.add_argument("--n-radial", type=int, default=41)
    parser.add_argument("--n-axial", type=int, default=81)
    parser.add_argument("--box-radial-waists", type=float, default=3.0)
    parser.add_argument("--box-axial-waists", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--save-plot",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Save the comparison plot. Pass a path, or omit a value for the "
            "default location next to this script."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still prints the numerical summary).",
    )
    args = parser.parse_args()

    analytical, tricubic, trilinear = build_traps(args)

    # Shared initial conditions: sampled from the analytical trap.
    rng = np.random.default_rng(args.seed)
    positions0 = sample_thermal_positions_harmonic(
        [analytical], args.temperature_uK, args.atoms, rng,
    )
    velocities0 = sample_thermal_velocities(
        args.temperature_uK, args.atoms, rng, mass_kg=RB87_MASS_KG,
    )

    dt = args.timestep_ns * 1e-9
    duration_s = args.duration_us * 1e-6
    n_steps = int(round(duration_s / dt))
    snap = max(2, args.snapshots)

    print(
        f"Trap: waist_r={args.waist_r_um:.3f} um, waist_z={args.waist_z_um:.2f} um, "
        f"depth={args.depth_uK:.0f} uK. T_init={args.temperature_uK:.2f} uK. "
        f"Atoms={args.atoms}. Duration={args.duration_us:.1f} us, dt={args.timestep_ns:.1f} ns "
        f"({n_steps} steps). Grid shape ({args.n_radial},{args.n_radial},{args.n_axial})."
    )

    runs = {}
    for label, trap in (
        ("analytical", analytical),
        ("gridded (tricubic)", tricubic),
        ("gridded (trilinear)", trilinear),
    ):
        runs[label] = run_velocity_verlet(
            trap, positions0, velocities0, dt, n_steps, RB87_MASS_KG, snap
        )

    print()
    summarize("analytical", runs["analytical"], ref=None)
    summarize("gridded (tricubic)", runs["gridded (tricubic)"], ref=runs["analytical"])
    summarize("gridded (trilinear)", runs["gridded (trilinear)"], ref=runs["analytical"])

    if args.no_plot:
        return

    if args.save_plot == "auto":
        os.makedirs(RENDER_DIR, exist_ok=True)
        out_path = os.path.join(RENDER_DIR, "gridded_vs_analytical.png")
    else:
        out_path = args.save_plot

    plot_comparison(runs, args, out_path)


if __name__ == "__main__":
    main()
