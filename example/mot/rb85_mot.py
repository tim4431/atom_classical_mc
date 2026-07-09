"""Rb85 magneto-optical trap: capture, cooling, and compression.

A MOT is just one configuration of the general light-matter machinery:
a `LightMatterSystem` made of a quadrupole `MagneticFieldConfig` plus
three retro-reflected sigma+/sigma- `LaserBeam` pairs red-detuned from
the Rb85 D2 cycling transition F=3 -> F'=4, attached to the standard
`run_simulation` driver via the `scattering` argument (no conservative
traps in this example). The internal-state backend evolves populations;
the momentum update applies per-photon recoil.

Run from the repository root:

    python3 example/mot/rb85_mot.py
    python3 example/mot/rb85_mot.py --backend rate-equation
    python3 example/mot/rb85_mot.py --plot          # or --save-plot
    python3 example/mot/rb85_mot.py --gif           # animated cloud GIF

Prints capture fraction, temperature evolution, cloud size, and photon
scattering statistics.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.analysis import kinetic_temperature_time_series_uK  # noqa: E402
from src.fields import QuadrupoleMagneticField  # noqa: E402
from src.internal_state import (  # noqa: E402
    AdiabaticSteadyState,
    RateEquationPopulations,
)
from src.laser import six_beam_mot  # noqa: E402
from src.light_matter import LightMatterSystem  # noqa: E402
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.species import RB85_D2  # noqa: E402
from src.units import gauss_per_cm, ms  # noqa: E402

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")

BACKENDS = {
    "steady-state": AdiabaticSteadyState,
    "rate-equation": RateEquationPopulations,
}


def build_system() -> LightMatterSystem:
    species = RB85_D2
    gamma_hz = species.linewidth_rad_s / (2.0 * np.pi)
    beams = six_beam_mot(
        detuning_hz=-1.5 * gamma_hz,  # -1.5 Gamma, typical capture detuning
        saturation=2.0,  # s0 = I / I_sat per beam
        waist_m=5.0e-3,  # 5 mm beams
    )
    quadrupole = QuadrupoleMagneticField(
        gradient_T_per_m=float(gauss_per_cm(10.0))  # 10 G/cm radial
    )
    return LightMatterSystem(species=species, beams=beams, magnetic_fields=[quadrupole])


def build_config(seed: int) -> SimulationConfig:
    return SimulationConfig(
        initial_temperature_uK=3000.0,  # 3 mK cloud, e.g. post-slowing
        initial_cloud_sigma_m=1.0e-3,  # 1 mm rms
        initial_mean_velocity_m_per_s=(1.0, 0.0, 0.0),  # slow drift
        timestep_s=5.0e-8,
        duration_s=float(ms(5.0)),
        ensemble_size=400,
        mass_kg=RB85_D2.mass_kg,
        loss_radius_m=8.0e-3,  # atoms leaving the beam volume are lost
        random_seed=seed,
        store_trajectories=True,
        trajectory_stride=250,  # sample every 50 us
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="steady-state",
        help="internal-state backend",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="show summary figure")
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="save summary figure into the render/ subdir",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="render an animated GIF of the cooling cloud into the render/ subdir",
    )
    args = parser.parse_args()

    system = build_system()
    config = build_config(args.seed)
    species = system.species

    print(f"Species: {species.name}")
    print(f"  Doppler limit     : {species.doppler_temperature_uK:8.1f} uK")
    print(f"  recoil velocity   : {species.recoil_velocity_m_per_s * 1e3:8.2f} mm/s")
    print(f"Internal-state backend: {args.backend}")
    print(
        f"Ensemble: {config.ensemble_size} atoms at "
        f"{config.initial_temperature_uK:.0f} uK, "
        f"duration {config.duration_s * 1e3:.1f} ms\n"
    )

    result = run_simulation(
        [], config, scattering=system, internal_model=BACKENDS[args.backend]()
    )
    temperatures_uK = kinetic_temperature_time_series_uK(
        result, mass_kg=species.mass_kg
    )

    survivors = ~result.lost
    final_rms_mm = float(
        np.sqrt(np.mean(np.sum(result.final_positions_m[survivors] ** 2, axis=-1)))
        * 1e3
    )
    initial_rms_mm = float(
        np.sqrt(np.mean(np.sum(result.initial_positions_m**2, axis=-1))) * 1e3
    )

    print(f"Capture fraction      : {result.survival_probability:8.3f}")
    print(f"Initial temperature   : {temperatures_uK[0]:8.1f} uK")
    print(f"Final temperature     : {temperatures_uK[-1]:8.1f} uK")
    print(f"Cloud rms radius      : {initial_rms_mm:6.2f} mm -> {final_rms_mm:6.2f} mm")
    print(
        "Mean scattered photons: " f"{float(np.mean(result.scattered_photons)):10.0f}"
    )
    print(
        "Mean excited fraction : "
        f"{float(np.mean(result.final_excited_fraction[survivors])):8.3f}"
    )

    print("\n  t [ms]   T [uK]   rms radius [mm]")
    times = result.trajectory_times_s
    stride = max(1, len(times) // 10)
    for i in range(0, len(times), stride):
        pos = result.trajectory_positions_m[i]
        alive = ~result.trajectory_lost[i]
        rms = float(np.sqrt(np.mean(np.sum(pos[alive] ** 2, axis=-1))) * 1e3)
        print(f"  {times[i] * 1e3:6.2f}  {temperatures_uK[i]:8.1f}  {rms:10.3f}")

    if args.plot or args.save_plot:
        _plot(result, temperatures_uK, args.save_plot)

    if args.gif:
        _render_gif(result, system)


def _render_gif(result, system) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.visualization import render_cloud_animation

    os.makedirs(RENDER_DIR, exist_ok=True)
    out = os.path.join(RENDER_DIR, "rb85_mot_cloud.gif")
    render_cloud_animation(
        result,
        out,
        mass_kg=system.species.mass_kg,
        plane="xz",
        beams=system.beams,
        doppler_limit_uK=system.species.doppler_temperature_uK,
        n_frames=80,
        fps=16,
        dpi=85,
    )
    print(f"\nSaved GIF to {out}")


def _plot(result, temperatures_uK, save: bool) -> None:
    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_ms = result.trajectory_times_s * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(times_ms, temperatures_uK)
    axes[0].set_xlabel("time [ms]")
    axes[0].set_ylabel("kinetic temperature [uK]")
    axes[0].set_title("Cooling")
    axes[0].set_yscale("log")

    rms_mm = [
        float(np.sqrt(np.mean(np.sum(p[~l] ** 2, axis=-1))) * 1e3)
        for p, l in zip(result.trajectory_positions_m, result.trajectory_lost)
    ]
    axes[1].plot(times_ms, rms_mm)
    axes[1].set_xlabel("time [ms]")
    axes[1].set_ylabel("cloud rms radius [mm]")
    axes[1].set_title("Compression")

    positions_mm = result.trajectory_positions_m * 1e3
    n_show = min(30, positions_mm.shape[1])
    for atom in range(n_show):
        axes[2].plot(
            positions_mm[:, atom, 0], positions_mm[:, atom, 2], lw=0.5, alpha=0.6
        )
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("z [mm]")
    axes[2].set_title(f"Trajectories ({n_show} atoms)")
    axes[2].set_aspect("equal")

    fig.tight_layout()
    if save:
        os.makedirs(RENDER_DIR, exist_ok=True)
        out = os.path.join(RENDER_DIR, "rb85_mot_summary.png")
        fig.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
