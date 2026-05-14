"""Pull an Rb87 ensemble from one SLM trap into a moving SLM trap.

Run from the repository root:

    python3 example/slm_to_slm_transfer.py

Saves trajectory, energy, 3D, and mode-occupation comparison plots next to this
script. The two traps share identical waists; the moving trap runs deeper (to
capture and transport), so mode frequencies are scaled but the mode axes match
and per-mode occupation histograms can be compared before/after transfer.
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
from src.trap import GaussianTrap  # noqa: E402
from src.units import ms, um  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_transfer_energy_summary,
    plot_transfer_trajectories_3d,
    plot_transfer_trajectory_summary,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

WAIST_RADIAL_UM = 1.2
WAIST_AXIAL_UM = 6.0
STATIC_DEPTH_UK = 70.0
MOVING_DEPTH_UK = 200.0


def build_transfer_problem() -> (
    tuple[GaussianTrap, GaussianTrap, RampSequence, SimulationConfig]
):
    static_slm_trap = GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(WAIST_RADIAL_UM)),
        waist_axial_m=float(um(WAIST_AXIAL_UM)),
        depth_uK=STATIC_DEPTH_UK,
        name="static SLM",
    )
    moving_slm_trap_base = GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(WAIST_RADIAL_UM)),
        waist_axial_m=float(um(WAIST_AXIAL_UM)),
        depth_uK=0.0,
        name="moving SLM",
    )
    ramp = build_position_ramp_sequence(
        samples=81,
        start_um=np.array([0.0, 0.0, 0.0]),
        stop_um=np.array([6.0, 0.0, 0.0]),
        depth_uK=MOVING_DEPTH_UK,
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
    return static_slm_trap, moving_slm_trap_base, ramp, config


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


def analyze_transfer_result(static_trap, moving_trap_base, ramp, config, result):
    final_center_m, final_depth_uK = ramp.at(config.duration_s)
    final_trap = moving_trap_base.with_center_depth(final_center_m, final_depth_uK)
    occupation = classify_final_trap_occupation(
        result, static_trap, final_trap, mass_kg=config.mass_kg
    )
    initial_harmonic = approximate_harmonic_potential(
        static_trap, static_trap.center_m, mass_kg=config.mass_kg
    )
    final_harmonic = approximate_harmonic_potential(
        final_trap, final_trap.center_m, mass_kg=config.mass_kg
    )
    initial_modes = decompose_motion_into_harmonic_modes(
        initial_harmonic,
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
    )
    final_modes = decompose_motion_into_harmonic_modes(
        final_harmonic, result.final_positions_m, result.final_velocities_m_per_s
    )
    return {
        "final_center_m": final_center_m,
        "final_depth_uK": final_depth_uK,
        "final_trap": final_trap,
        "moving_capture": capture_probability(
            result, final_trap, mass_kg=config.mass_kg
        ),
        "moving_capture_given_survival": capture_probability(
            result, final_trap, mass_kg=config.mass_kg, conditional_on_survival=True
        ),
        "occupation": occupation,
        "kinetic_trace_uK": mean_kinetic_energy_time_series_uK(
            result, mass_kg=config.mass_kg
        ),
        "survival_trace": survival_probability_time_series(result),
        "loss_trace": loss_probability_time_series(result),
        "initial_harmonic": initial_harmonic,
        "final_harmonic": final_harmonic,
        "initial_modes": initial_modes,
        "final_modes": final_modes,
        "initial_mode_summary": summarize_mode_occupations(initial_modes),
        "final_mode_summary": summarize_mode_occupations(
            final_modes, mask=occupation.aod_mask
        ),
    }


def print_transfer_report(config, result, analysis) -> None:
    occupation = analysis["occupation"]
    kinetic_trace_uK = analysis["kinetic_trace_uK"]
    print("SLM-to-SLM transfer simulation")
    print(f"ensemble size: {config.ensemble_size}")
    print(f"initial temperature: {config.initial_temperature_uK:.3f} uK")
    print(f"moving SLM final depth: {analysis['final_depth_uK']:.3f} uK")
    print(f"moving SLM final position: {analysis['final_center_m'] * 1.0e6} um")
    print(f"survival probability: {result.survival_probability:.4f}")
    print(f"loss fraction: {result.loss_fraction:.4f}")
    print(f"moving SLM capture probability: {analysis['moving_capture']:.4f}")
    print(
        f"moving SLM capture among survivors: {analysis['moving_capture_given_survival']:.4f}"
    )
    print(f"final static-SLM-only occupation: {occupation.slm_probability:.4f}")
    print(f"final moving-SLM-only occupation: {occupation.aod_probability:.4f}")
    print(
        f"final ambiguous static-and-moving occupation: {occupation.ambiguous_probability:.4f}"
    )
    print(
        f"final unbound survivor fraction: {occupation.unbound_survivor_probability:.4f}"
    )
    print(f"mean energy gain of survivors: {result.mean_energy_gain_uK:.4f} uK")
    print(f"final kinetic temperature: {result.final_temperature_uK:.4f} uK")
    print(f"temperature gain: {result.temperature_gain_uK:.4f} uK")
    print(f"initial mean kinetic energy: {kinetic_trace_uK[0]:.4f} uK")
    print(f"final mean kinetic energy: {kinetic_trace_uK[-1]:.4f} uK")

    print("initial / final harmonic mode frequencies (kHz)")
    for label, hz_initial, hz_final in zip(
        analysis["initial_harmonic"].mode_labels,
        analysis["initial_harmonic"].frequencies_hz,
        analysis["final_harmonic"].frequencies_hz,
    ):
        print(f"  {label}: {hz_initial / 1.0e3:.3f}  ->  {hz_final / 1.0e3:.3f}")

    print("mean occupation per mode (initial -> final, captured atoms)")
    for label in analysis["initial_mode_summary"]:
        n_init = analysis["initial_mode_summary"][label]["mean"]
        n_final = analysis["final_mode_summary"][label]["mean"]
        print(f"  {label}: {n_init:.3f}  ->  {n_final:.3f}")


def plot_mode_occupation_comparison(analysis, bins: int = 40):
    import matplotlib.pyplot as plt

    captured_mask = np.asarray(analysis["occupation"].aod_mask, dtype=bool)
    initial_n = np.asarray(analysis["initial_modes"].mean_occupations, dtype=float)
    final_n = np.asarray(analysis["final_modes"].mean_occupations, dtype=float)
    initial_n = initial_n[captured_mask]
    final_n = final_n[captured_mask]
    labels = list(analysis["initial_modes"].mode_labels)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for mode_index, (axis, label) in enumerate(zip(axes, labels)):
        before = initial_n[:, mode_index]
        after = final_n[:, mode_index]
        finite = np.concatenate([before, after])
        finite = finite[np.isfinite(finite)]
        upper = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        edges = np.linspace(0.0, max(upper, 1.0), bins + 1)
        axis.hist(
            np.clip(before, edges[0], edges[-1]),
            bins=edges,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=f"before  <n>={before.mean():.2f}",
        )
        axis.hist(
            np.clip(after, edges[0], edges[-1]),
            bins=edges,
            density=True,
            alpha=0.55,
            color="tab:green",
            label=f"after   <n>={after.mean():.2f}",
        )
        axis.set_title(label)
        axis.set_xlabel("occupation number  n")
        axis.set_ylabel("probability density")
        axis.legend(loc="best", fontsize="small")
        axis.grid(axis="y", linestyle=":", alpha=0.5)
    figure.suptitle("Per-mode occupation distribution: before vs after transfer")
    return figure, axes


def main() -> None:
    static_trap, moving_trap_base, ramp, config = build_transfer_problem()
    result = run_simulation(static_trap, moving_trap_base, ramp, config)
    analysis = analyze_transfer_result(
        static_trap, moving_trap_base, ramp, config, result
    )
    print_transfer_report(config, result, analysis)

    trajectory_figure, _ = plot_transfer_trajectory_summary(result, ramp, static_trap)
    energy_figure, _ = plot_transfer_energy_summary(
        result,
        initial_modes=analysis["initial_modes"],
        final_modes=analysis["final_modes"],
        final_mask=analysis["occupation"].aod_mask,
    )
    figure_3d, _ = plot_transfer_trajectories_3d(result, ramp, static_trap)
    occupation_figure, _ = plot_mode_occupation_comparison(analysis)

    trajectory_path = os.path.join(RENDER_DIR, "slm_to_slm_transfer_traj.png")
    energy_path = os.path.join(RENDER_DIR, "slm_to_slm_transfer_energy.png")
    path_3d = os.path.join(RENDER_DIR, "slm_to_slm_transfer_3d.png")
    occupation_path = os.path.join(RENDER_DIR, "slm_to_slm_transfer_occupation.png")
    trajectory_figure.savefig(trajectory_path, dpi=180)
    energy_figure.savefig(energy_path, dpi=180)
    figure_3d.savefig(path_3d, dpi=180)
    occupation_figure.savefig(occupation_path, dpi=180)
    print(f"saved: {trajectory_path}")
    print(f"saved: {energy_path}")
    print(f"saved: {path_3d}")
    print(f"saved: {occupation_path}")


if __name__ == "__main__":
    main()
