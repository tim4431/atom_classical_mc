"""Recoil heating of a tweezer-trapped Rb87 atom under near-resonant light.

Demonstrates mixing the two physics layers in one `run_simulation` call:
a conservative `GaussianTrap` (optical tweezer) plus a
`LightMatterSystem` carrying a single weak probe/imaging beam. Photon
scattering heats the atom at the recoil rate; survival is recovered
post-hoc with `bound_to_trap` since the energy loss criterion is
disabled under non-conservative light forces.

Run from the repository root:

    python3 example/tweezer_probe_heating.py

Sweeps probe saturation and prints heating rate and retention.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import bound_to_trap  # noqa: E402
from src.laser import LaserBeam  # noqa: E402
from src.light_matter import LightMatterSystem  # noqa: E402
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.species import RB87_D2  # noqa: E402
from src.trap import GaussianTrap  # noqa: E402
from src.units import ms, um  # noqa: E402


def main() -> None:
    trap = GaussianTrap(
        waist_radial_m=float(um(1.0)),
        waist_axial_m=float(um(5.0)),
        depth_uK=1000.0,
        name="tweezer",
    )
    config = SimulationConfig(
        initial_temperature_uK=20.0,
        timestep_s=1.0e-7,
        duration_s=float(ms(2.0)),
        ensemble_size=300,
        random_seed=1,
    )

    # Expected recoil heating: dT/dt = 2/3 * (2 T_rec) * R_sc with
    # R_sc = (Gamma/2) s / (1 + s) on resonance (absorption + emission
    # each contribute one recoil temperature T_rec per photon on average).
    print(f"Tweezer: {trap.depth_uK:.0f} uK deep")
    print(f"Doppler limit: {RB87_D2.doppler_temperature_uK:.1f} uK\n")
    print("  s0       photons   dT [uK]   retention")

    for saturation in (0.0, 0.001, 0.01, 0.1):
        if saturation == 0.0:
            result = run_simulation(trap, config)
        else:
            probe = LaserBeam(
                direction=(0.0, 0.0, 1.0),
                detuning_hz=0.0,
                saturation=saturation,
                name="probe",
            )
            system = LightMatterSystem(species=RB87_D2, beams=[probe])
            result = run_simulation(trap, config, scattering=system)

        retained = bound_to_trap(
            result.final_positions_m,
            result.final_velocities_m_per_s,
            trap,
            time_s=config.duration_s,
        )
        photons = (
            float(np.mean(result.scattered_photons))
            if result.scattered_photons is not None
            else 0.0
        )
        gain = (
            result.final_temperature_uK_all - result.initial_temperature_uK_all
        )
        print(
            f"  {saturation:6.3f}  {photons:9.0f}  {gain:8.1f}"
            f"  {float(np.mean(retained)):9.3f}"
        )


if __name__ == "__main__":
    main()
