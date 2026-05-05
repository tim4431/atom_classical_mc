"""Pull an Rb87 ensemble from an SLM trap with a moving AOD tweezer.

Run from the repository root:

    python3 example/slm_to_aod_transfer.py

Saves trajectory, energy, and 3D plots next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import (  # noqa: E402
    capture_probability,
    classify_final_trap_occupation,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
    survival_probability_time_series,
)
from src.harmonic import (  # noqa: E402
    approximate_harmonic_potential,
    decompose_motion_into_harmonic_modes,
    summarize_mode_occupations,
)
from src.ramp import RampSequence  # noqa: E402
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.trap import TrapConfig  # noqa: E402
from src.units import ms, um  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_transfer_energy_summary,
    plot_transfer_trajectories_3d,
    plot_transfer_trajectory_summary,
)

HERE = os.path.dirname(__file__)


def build_transfer_problem() -> (
    tuple[TrapConfig, TrapConfig, RampSequence, SimulationConfig]
):
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
    ramp = build_position_ramp_sequence(
        samples=81,
        start_um=np.array([0.0, 0.0, 0.0]),
        stop_um=np.array([6.0, 0.0, 0.0]),
        depth_uK=200.0,
        load_end_ms=0.08,
        move_end_ms=0.48,
        hold_end_ms=0.68,
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


def build_position_ramp_sequence(
    samples: int,
    start_um: np.ndarray,
    stop_um: np.ndarray,
    depth_uK: float,
    load_end_ms: float,
    move_end_ms: float,
    hold_end_ms: float,
) -> RampSequence:
    u = np.linspace(0.0, 1.0, samples)
    move_centers_um = start_um + u[:, np.newaxis] * (stop_um - start_um)
    times_ms = np.concatenate(
        [[0.0], np.linspace(load_end_ms, move_end_ms, samples), [hold_end_ms]]
    )
    centers_um = np.vstack([start_um, move_centers_um, stop_um])
    depths_uK = np.concatenate([[0.0], np.full(samples, depth_uK), [depth_uK]])
    return RampSequence(
        times_s=ms(times_ms), centers_m=um(centers_um), depths_uK=depths_uK
    )


def analyze_transfer_result(slm_trap, aod_trap_base, ramp, config, result):
    final_aod_center_m, final_aod_depth_uK = ramp.at(config.duration_s)
    final_aod_trap = aod_trap_base.with_center_depth(
        final_aod_center_m, final_aod_depth_uK
    )
    occupation = classify_final_trap_occupation(
        result, slm_trap, final_aod_trap, mass_kg=config.mass_kg
    )
    initial_slm_harmonic = approximate_harmonic_potential(
        slm_trap, slm_trap.center_m, mass_kg=config.mass_kg
    )
    final_aod_harmonic = approximate_harmonic_potential(
        final_aod_trap, final_aod_trap.center_m, mass_kg=config.mass_kg
    )
    initial_slm_modes = decompose_motion_into_harmonic_modes(
        initial_slm_harmonic,
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
    )
    final_aod_modes = decompose_motion_into_harmonic_modes(
        final_aod_harmonic, result.final_positions_m, result.final_velocities_m_per_s
    )
    return {
        "final_aod_center_m": final_aod_center_m,
        "final_aod_depth_uK": final_aod_depth_uK,
        "final_aod_trap": final_aod_trap,
        "aod_capture": capture_probability(
            result, final_aod_trap, mass_kg=config.mass_kg
        ),
        "aod_capture_given_survival": capture_probability(
            result, final_aod_trap, mass_kg=config.mass_kg, conditional_on_survival=True
        ),
        "occupation": occupation,
        "kinetic_trace_uK": mean_kinetic_energy_time_series_uK(
            result, mass_kg=config.mass_kg
        ),
        "survival_trace": survival_probability_time_series(result),
        "loss_trace": loss_probability_time_series(result),
        "initial_slm_harmonic": initial_slm_harmonic,
        "final_aod_harmonic": final_aod_harmonic,
        "initial_slm_modes": initial_slm_modes,
        "final_aod_modes": final_aod_modes,
        "initial_slm_mode_summary": summarize_mode_occupations(initial_slm_modes),
        "final_aod_mode_summary": summarize_mode_occupations(
            final_aod_modes, mask=occupation.aod_mask
        ),
    }


def print_transfer_report(config, result, analysis) -> None:
    occupation = analysis["occupation"]
    kinetic_trace_uK = analysis["kinetic_trace_uK"]
    print("SLM-to-AOD transfer simulation")
    print(f"ensemble size: {config.ensemble_size}")
    print(f"initial temperature: {config.initial_temperature_uK:.3f} uK")
    print(f"AOD final depth: {analysis['final_aod_depth_uK']:.3f} uK")
    print(f"AOD final position: {analysis['final_aod_center_m'] * 1.0e6} um")
    print(f"survival probability: {result.survival_probability:.4f}")
    print(f"loss fraction: {result.loss_fraction:.4f}")
    print(f"AOD transition/capture probability: {analysis['aod_capture']:.4f}")
    print(f"AOD capture among survivors: {analysis['aod_capture_given_survival']:.4f}")
    print(f"final SLM-only occupation: {occupation.slm_probability:.4f}")
    print(f"final AOD-only occupation: {occupation.aod_probability:.4f}")
    print(
        f"final ambiguous SLM-and-AOD occupation: {occupation.ambiguous_probability:.4f}"
    )
    print(
        f"final unbound survivor fraction: {occupation.unbound_survivor_probability:.4f}"
    )
    print(f"mean energy gain of survivors: {result.mean_energy_gain_uK:.4f} uK")
    print(f"final kinetic temperature: {result.final_temperature_uK:.4f} uK")
    print(f"temperature gain: {result.temperature_gain_uK:.4f} uK")
    print(f"initial mean kinetic energy: {kinetic_trace_uK[0]:.4f} uK")
    print(f"final mean kinetic energy: {kinetic_trace_uK[-1]:.4f} uK")

    print("initial SLM harmonic modes")
    for label, hz in zip(
        analysis["initial_slm_harmonic"].mode_labels,
        analysis["initial_slm_harmonic"].frequencies_hz,
    ):
        print(f"  {label}: {hz / 1.0e3:.3f} kHz")
    print("final AOD harmonic modes")
    for label, hz in zip(
        analysis["final_aod_harmonic"].mode_labels,
        analysis["final_aod_harmonic"].frequencies_hz,
    ):
        print(f"  {label}: {hz / 1.0e3:.3f} kHz")


def main() -> None:
    slm_trap, aod_trap_base, ramp, config = build_transfer_problem()
    result = run_simulation(slm_trap, aod_trap_base, ramp, config)
    analysis = analyze_transfer_result(slm_trap, aod_trap_base, ramp, config, result)
    print_transfer_report(config, result, analysis)

    trajectory_figure, _ = plot_transfer_trajectory_summary(result, ramp, slm_trap)
    energy_figure, _ = plot_transfer_energy_summary(
        result,
        initial_modes=analysis["initial_slm_modes"],
        final_modes=analysis["final_aod_modes"],
        final_mask=analysis["occupation"].aod_mask,
    )
    figure_3d, _ = plot_transfer_trajectories_3d(result, ramp, slm_trap)

    trajectory_path = os.path.join(HERE, "slm_to_aod_transfer_traj.png")
    energy_path = os.path.join(HERE, "slm_to_aod_transfer_energy.png")
    path_3d = os.path.join(HERE, "slm_to_aod_transfer_3d.png")
    trajectory_figure.savefig(trajectory_path, dpi=180)
    energy_figure.savefig(energy_path, dpi=180)
    figure_3d.savefig(path_3d, dpi=180)
    print(f"saved: {trajectory_path}")
    print(f"saved: {energy_path}")
    print(f"saved: {path_3d}")


if __name__ == "__main__":
    main()
