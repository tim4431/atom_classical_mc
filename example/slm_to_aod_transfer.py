"""Example: pull an Rb87 atom ensemble from an SLM trap with a moving AOD tweezer.

Run from the repository root:

    python3 example/slm_to_aod_transfer.py

For plots, install the optional visualization dependency:

    python3 -m pip install -e ".[viz]"
    python3 example/slm_to_aod_transfer.py --plot
    python3 example/slm_to_aod_transfer.py --save-3d-plot example/slm_to_aod_transfer_3d.png
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
    approximate_harmonic_potential,
    capture_probability,
    classify_final_trap_occupation,
    decompose_motion_into_harmonic_modes,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
    ms,
    run_simulation,
    summarize_mode_occupations,
    survival_probability_time_series,
    um,
)
from atom_classical_mc.visualization import (  # noqa: E402
    plot_transfer_energy_summary,
    plot_transfer_trajectories_3d,
    plot_transfer_trajectory_summary,
)


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


def _print_harmonic_summary(label, approximation) -> None:
    print(label)
    for mode_label, frequency_hz in zip(
        approximation.mode_labels,
        approximation.frequencies_hz,
    ):
        print(f"  {mode_label}: {frequency_hz / 1.0e3:.3f} kHz")


def _print_occupation_summary(label, summary) -> None:
    print(label)
    for mode_label, stats in summary.items():
        print(
            f"  {mode_label}: mean={stats['mean']:.3f}, "
            f"median={stats['median']:.3f}, std={stats['std']:.3f}"
        )


def _suffixed_plot_path(path: str, suffix: str) -> str:
    root, extension = os.path.splitext(path)
    if not extension:
        extension = ".png"
    return f"{root}_{suffix}{extension}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="show Matplotlib summary plots"
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        help="base path for 2D plots; writes *_traj and *_energy files",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="show the 3D atom trajectory and AOD path plot",
    )
    parser.add_argument(
        "--save-3d-plot",
        default=None,
        help="save the 3D trajectory plot to this path",
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

    initial_slm_harmonic = approximate_harmonic_potential(
        slm_trap,
        slm_trap.center_m,
        mass_kg=config.mass_kg,
    )
    final_aod_harmonic = approximate_harmonic_potential(
        final_aod_trap,
        final_aod_trap.center_m,
        mass_kg=config.mass_kg,
    )
    initial_slm_modes = decompose_motion_into_harmonic_modes(
        initial_slm_harmonic,
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
    )
    final_aod_modes = decompose_motion_into_harmonic_modes(
        final_aod_harmonic,
        result.final_positions_m,
        result.final_velocities_m_per_s,
    )
    initial_slm_mode_summary = summarize_mode_occupations(initial_slm_modes)
    final_aod_mode_summary = summarize_mode_occupations(
        final_aod_modes,
        mask=occupation.aod_mask,
    )

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
    print(f"initial rejected thermal draws: {result.initial_rejected_count}")
    print(f"initial rejection fraction: {result.initial_rejection_fraction:.4f}")
    _print_harmonic_summary("initial SLM harmonic modes", initial_slm_harmonic)
    _print_occupation_summary("initial SLM mean n", initial_slm_mode_summary)
    _print_harmonic_summary("final AOD harmonic modes", final_aod_harmonic)
    _print_occupation_summary(
        "final AOD mean n for AOD-only atoms",
        final_aod_mode_summary,
    )

    if args.plot or args.save_plot:
        output_base = args.save_plot or os.path.join(
            os.path.dirname(__file__),
            "slm_to_aod_transfer.png",
        )
        trajectory_figure, _ = plot_transfer_trajectory_summary(result, ramp, slm_trap)
        energy_figure, _ = plot_transfer_energy_summary(
            result,
            initial_mode_summary=initial_slm_mode_summary,
            final_mode_summary=final_aod_mode_summary,
        )
        trajectory_path = _suffixed_plot_path(output_base, "traj")
        energy_path = _suffixed_plot_path(output_base, "energy")
        trajectory_figure.savefig(trajectory_path, dpi=180)
        energy_figure.savefig(energy_path, dpi=180)
        print(f"saved trajectory plot: {trajectory_path}")
        print(f"saved energy plot: {energy_path}")

    if args.plot_3d or args.save_3d_plot:
        figure_3d, _ = plot_transfer_trajectories_3d(result, ramp, slm_trap)
        if args.save_3d_plot:
            output_path_3d = args.save_3d_plot
        else:
            output_path_3d = os.path.join(
                os.path.dirname(__file__),
                "slm_to_aod_transfer_3d.png",
            )
        figure_3d.savefig(output_path_3d, dpi=180)
        print(f"saved 3D plot: {output_path_3d}")

    if args.plot or args.plot_3d:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
