"""Example: pull an Rb87 atom ensemble from an SLM trap with a moving AOD tweezer.

Run from the repository root:

    python3 example/slm_to_aod_transfer.py

For plots, install the optional visualization dependency:

    python3 -m pip install -e ".[viz]"
    python3 example/slm_to_aod_transfer.py --plot
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from atom_classical_mc import (  # noqa: E402
    RampSequence,
    SimulationConfig,
    TrapConfig,
    capture_probability,
    classify_final_trap_occupation,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
    ms,
    run_simulation,
    survival_probability_time_series,
    um,
)
from atom_classical_mc.visualization import plot_transfer_summary  # noqa: E402


def build_transfer_problem() -> (
    tuple[TrapConfig, TrapConfig, RampSequence, SimulationConfig]
):
    """Configure a representative SLM-to-AOD pull-out ramp.

    The AOD depth is the model's direct intensity proxy. The ramp first turns on
    the AOD at the SLM site, then translates it by 6 um, then holds briefly so
    final AOD capture can be classified in the lab frame.
    """

    slm_trap = TrapConfig(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(1.2)),
        waist_axial_m=float(um(6.0)),
        depth_uK=70.0,
        name="static SLM",
    )

    aod_trap_base = TrapConfig(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(1.0)),
        waist_axial_m=float(um(5.0)),
        depth_uK=0.0,
        name="moving AOD",
    )

    ramp = RampSequence(
        times_s=ms([0.0, 0.08, 0.48, 0.68]),
        centers_m=um(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ]
        ),
        depths_uK=np.array([0.0, 85.0, 85.0, 85.0]),
    )

    config = SimulationConfig(
        initial_temperature_uK=8.0,
        timestep_s=float(ms(0.0002)),
        duration_s=float(ms(0.68)),
        ensemble_size=1500,
        random_seed=42,
        loss_radius_m=float(um(30.0)),
        store_trajectories=True,
        trajectory_stride=25,
    )
    return slm_trap, aod_trap_base, ramp, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="show Matplotlib summary plots"
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        help="save the 6-panel Matplotlib summary plot to this path",
    )
    args = parser.parse_args()

    slm_trap, aod_trap_base, ramp, config = build_transfer_problem()
    result = run_simulation(slm_trap, aod_trap_base, ramp, config)

    final_aod_center_m, final_aod_depth_uK = ramp.at(config.duration_s)
    final_aod_trap = aod_trap_base.with_center_depth(
        final_aod_center_m, final_aod_depth_uK
    )
    aod_capture = capture_probability(result, final_aod_trap, mass_kg=config.mass_kg)
    aod_capture_given_survival = capture_probability(
        result,
        final_aod_trap,
        mass_kg=config.mass_kg,
        conditional_on_survival=True,
    )
    occupation = classify_final_trap_occupation(
        result,
        slm_trap,
        final_aod_trap,
        mass_kg=config.mass_kg,
    )
    kinetic_trace_uK = mean_kinetic_energy_time_series_uK(
        result, mass_kg=config.mass_kg
    )
    survival_trace = survival_probability_time_series(result)
    loss_trace = loss_probability_time_series(result)

    print("SLM-to-AOD transfer simulation")
    print(f"ensemble size: {config.ensemble_size}")
    print(f"initial temperature: {config.initial_temperature_uK:.3f} uK")
    print(f"AOD final depth: {final_aod_depth_uK:.3f} uK")
    print(f"AOD final position: {final_aod_center_m * 1.0e6} um")
    print(f"survival probability: {result.survival_probability:.4f}")
    print(f"loss fraction: {result.loss_fraction:.4f}")
    print(f"AOD transition/capture probability: {aod_capture:.4f}")
    print(f"AOD capture among survivors: {aod_capture_given_survival:.4f}")
    print(f"final SLM-only occupation: {occupation.slm_probability:.4f}")
    print(f"final AOD-only occupation: {occupation.aod_probability:.4f}")
    print(f"final ambiguous SLM-and-AOD occupation: {occupation.ambiguous_probability:.4f}")
    print(f"final unbound survivor fraction: {occupation.unbound_survivor_probability:.4f}")
    print(f"mean energy gain of survivors: {result.mean_energy_gain_uK:.4f} uK")
    print(f"median energy gain of survivors: {result.median_energy_gain_uK:.4f} uK")
    print(f"final kinetic temperature: {result.final_temperature_uK:.4f} uK")
    print(f"temperature gain: {result.temperature_gain_uK:.4f} uK")
    print(f"initial mean kinetic energy: {kinetic_trace_uK[0]:.4f} uK")
    print(f"final mean kinetic energy: {kinetic_trace_uK[-1]:.4f} uK")
    print(f"stored final survival: {survival_trace[-1]:.4f}")
    print(f"stored final loss: {loss_trace[-1]:.4f}")

    if args.plot or args.save_plot:
        figure, _ = plot_transfer_summary(result, ramp, slm_trap)
        if args.save_plot:
            output_path = args.save_plot
        else:
            output_path = os.path.join(os.path.dirname(__file__), "slm_to_aod_transfer.png")
        figure.savefig(output_path, dpi=180)
        print(f"saved plot: {output_path}")

        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
