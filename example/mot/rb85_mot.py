"""Rb85 magneto-optical trap: capture, cooling, and compression.

A thermal Rb85 cloud (a few mK, launched with a small drift velocity)
is released into a six-beam MOT: quadrupole magnetic field plus three
retro-reflected sigma+/sigma- beam pairs, red-detuned from the D2
cycling transition F=3 -> F'=4. The coupled simulation evolves the
internal state (steady-state populations by default) and the motion
(trap-free, radiation pressure + per-photon recoil) together.

Run from the repository root:

    python3 example/mot/rb85_mot.py
    python3 example/mot/rb85_mot.py --backend rate-equation
    python3 example/mot/rb85_mot.py --plot          # or --save-plot

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

from src.fields import QuadrupoleMagneticField  # noqa: E402
from src.internal_state import (  # noqa: E402
    AdiabaticSteadyState,
    RateEquationPopulations,
)
from src.laser import six_beam_mot  # noqa: E402
from src.mot import (  # noqa: E402
    MOTSimulationConfig,
    MOTSystem,
    run_mot_simulation,
)
from src.species import RB85_D2  # noqa: E402
from src.units import gauss_per_cm, ms  # noqa: E402

HERE = os.path.dirname(__file__)

BACKENDS = {
    "steady-state": AdiabaticSteadyState,
    "rate-equation": RateEquationPopulations,
}


def build_system() -> MOTSystem:
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
    return MOTSystem(species=species, beams=beams, magnetic_fields=[quadrupole])


def build_config(seed: int) -> MOTSimulationConfig:
    return MOTSimulationConfig(
        initial_temperature_uK=3000.0,  # 3 mK cloud, e.g. post-slowing
        initial_cloud_sigma_m=1.0e-3,  # 1 mm rms
        initial_mean_velocity_m_per_s=(1.0, 0.0, 0.0),  # slow drift
        timestep_s=2.0e-7,
        duration_s=float(ms(20.0)),
        ensemble_size=400,
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
        help="save summary figure next to this script",
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

    result = run_mot_simulation(
        system, config, internal_model=BACKENDS[args.backend]()
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
    print(f"Initial temperature   : {result.initial_temperature_uK_all:8.1f} uK")
    print(f"Final temperature     : {result.final_temperature_uK_survivors:8.1f} uK")
    print(f"Cloud rms radius      : {initial_rms_mm:6.2f} mm -> {final_rms_mm:6.2f} mm")
    print(f"Mean scattered photons: {result.mean_scattered_photons:10.0f}")
    print(
        "Mean excited fraction : "
        f"{float(np.mean(result.final_excited_fraction[survivors])):8.3f}"
    )

    if result.trajectory_times_s is not None:
        print("\n  t [ms]   T [uK]   rms radius [mm]")
        times = result.trajectory_times_s
        stride = max(1, len(times) // 10)
        for i in range(0, len(times), stride):
            pos = result.trajectory_positions_m[i]
            alive = ~result.trajectory_lost[i]
            rms = float(np.sqrt(np.mean(np.sum(pos[alive] ** 2, axis=-1))) * 1e3)
            print(
                f"  {times[i] * 1e3:6.2f}  {result.trajectory_temperature_uK[i]:8.1f}"
                f"  {rms:10.3f}"
            )

    if args.plot or args.save_plot:
        _plot(result, args.save_plot)


def _plot(result, save: bool) -> None:
    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_ms = result.trajectory_times_s * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(times_ms, result.trajectory_temperature_uK)
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
        out = os.path.join(HERE, "rb85_mot_summary.png")
        fig.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
