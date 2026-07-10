"""Magnetic quadrupole trap and dipole-beam tweezer: the new physics modules.

Part 1 — `ZeemanPotential`: an Rb87 cloud in the weak-field-seeking
|F=2, m_F=2> sublevel is held by a bare quadrupole field (U = mu |B|,
linear cone potential), while the anti-trapped m_F=-2 sublevel is
expelled (Stern-Gerlach). No light involved: this is a pure magnetic
trap, impossible to express in the pre-multiphysics code.

Part 2 — `DipoleBeamPotential`: an 850 nm tweezer defined by laboratory
parameters (power, waist, wavelength) instead of a hand-specified
`depth_uK`. Its depth and trap frequencies follow from Gaussian-beam
optics; we cross-check the dynamics against an equivalent `GaussianTrap`
whose depth matches and whose axial waist is chosen to reproduce the
Rayleigh-range curvature (w_z = sqrt(2) z_R).

Run from the repository root:

    python3 example/magnetic_dipole/quadrupole_and_dipole.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from atommc import (  # noqa: E402
    AtomSystem,
    DipoleBeamPotential,
    GaussianTrap,
    QuadrupoleMagneticField,
    RB87_D2,
    SimulationConfig,
    ZeemanPotential,
    gauss_per_cm,
    ms,
    simulate,
    um,
)


def magnetic_trap_demo() -> None:
    print("=== Part 1: quadrupole magnetic trap (ZeemanPotential) ===")
    quadrupole = QuadrupoleMagneticField(gradient_T_per_m=float(gauss_per_cm(100.0)))
    config = SimulationConfig(
        initial_temperature_uK=100.0,
        initial_cloud_sigma_m=150.0e-6,
        timestep_s=2.0e-6,
        duration_s=float(ms(20.0)),
        ensemble_size=300,
        random_seed=1,
        loss_radius_m=5.0e-3,
        reject_initially_lost=False,
        energy_loss="off",  # the |B|=0 node makes E>=0 transiently possible
    )

    for m_f, label in ((2.0, "trapped |F=2, m_F=+2>"), (-2.0, "anti-trapped m_F=-2")):
        zeeman = ZeemanPotential.for_sublevel(quadrupole, g_f=0.5, m_f=m_f)
        system = AtomSystem(species=RB87_D2, modules=[zeeman])
        result = simulate(system, config)
        rms_mm = float(
            np.sqrt(np.mean(np.sum(result.final_positions_m[~result.lost] ** 2, axis=-1)))
            * 1e3
        ) if np.any(~result.lost) else float("nan")
        print(f"  {label}:")
        print(f"    moment           : {zeeman.moment_j_per_t / 9.274e-24:+.2f} mu_B")
        print(f"    survival (20 ms) : {result.survival_probability:6.3f}")
        print(f"    cloud rms radius : {rms_mm:6.3f} mm")
    print()


def dipole_tweezer_demo() -> None:
    print("=== Part 2: 850 nm dipole tweezer (DipoleBeamPotential) ===")
    beam = DipoleBeamPotential(
        species=RB87_D2,
        power_w=4.0e-3,
        waist_m=float(um(1.0)),
        wavelength_m=850.0e-9,
        name="850nm tweezer",
    )
    mass = RB87_D2.mass_kg
    omega_r = np.sqrt(4.0 * beam.depth_j / (mass * beam.waist_m**2))
    omega_z = np.sqrt(2.0 * beam.depth_j / (mass * beam.rayleigh_length_m**2))
    print(f"  power / waist      : {beam.power_w * 1e3:.1f} mW / {beam.waist_m * 1e6:.1f} um")
    print(f"  detuning           : {beam.detuning_rad_s / RB87_D2.linewidth_rad_s:.3e} Gamma")
    print(f"  Rayleigh length    : {beam.rayleigh_length_m * 1e6:6.2f} um")
    print(f"  trap depth         : {beam.depth_uK:8.1f} uK")
    print(f"  radial / axial freq: {omega_r / 2e3 / np.pi:6.1f} / {omega_z / 2e3 / np.pi:6.1f} kHz")
    print(f"  photon scattering  : {beam.peak_scattering_rate_per_s:8.1f} /s at the focus")

    # Equivalent hand-tuned Gaussian: same depth and radial waist; axial
    # 1/e^2 scale sqrt(2) z_R reproduces the Rayleigh-range curvature.
    gaussian = GaussianTrap(
        waist_radial_m=beam.waist_m,
        waist_axial_m=float(np.sqrt(2.0) * beam.rayleigh_length_m),
        depth_uK=beam.depth_uK,
        name="matched Gaussian",
    )

    config = SimulationConfig(
        initial_temperature_uK=0.1 * beam.depth_uK,
        timestep_s=1.0e-7,
        duration_s=float(ms(2.0)),
        ensemble_size=400,
        random_seed=2,
    )
    print(f"\n  holding a {config.initial_temperature_uK:.0f} uK cloud for 2 ms:")
    for module in (beam, gaussian):
        result = simulate(AtomSystem(species=RB87_D2, modules=[module]), config)
        print(
            f"    {module.name:16s}: survival {result.survival_probability:6.3f}, "
            f"temperature {result.initial_temperature_uK_all:6.1f} -> "
            f"{result.final_temperature_uK_all:6.1f} uK"
        )


if __name__ == "__main__":
    magnetic_trap_demo()
    dipole_tweezer_demo()
