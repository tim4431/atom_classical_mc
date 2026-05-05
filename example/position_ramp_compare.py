"""Compare AOD position ramp profiles for SLM-to-AOD atom transfer."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import capture_probability  # noqa: E402
from src.ramp import RampSequence  # noqa: E402
from src.simulation import SimulationConfig, SimulationResult, run_simulation  # noqa: E402
from src.trap import TrapConfig  # noqa: E402
from src.units import ms, um  # noqa: E402

HERE = os.path.dirname(__file__)

POSITION_RAMP_PROFILES = (
    "linear",
    "cubic_smoothstep",
    "quintic_minimum_jerk",
    "sinusoidal",
)
POSITION_SAMPLES = 81
AOD_DEPTH_UK = 120.0
ENSEMBLE_SIZE = 1500
RANDOM_SEED = 42
SAVE_PLOT_PATH = os.path.join(HERE, "position_ramp_compare.png")
SHOW_PLOT = False


@dataclass(frozen=True)
class TransferProblem:
    """Traps, ramp, and numerical settings for one comparison case."""

    static_trap: TrapConfig
    moving_trap_base: TrapConfig
    ramp: RampSequence
    config: SimulationConfig


@dataclass(frozen=True)
class PositionRampCase:
    """Simulation output and derived metrics for one position ramp profile."""

    profile_name: str
    problem: TransferProblem
    result: SimulationResult
    aod_capture_probability: float


def position_profile_fraction(profile_name: str, fraction: np.ndarray) -> np.ndarray:
    """Map a normalized time coordinate to normalized position."""

    u = np.clip(np.asarray(fraction, dtype=float), 0.0, 1.0)
    if profile_name == "linear":
        shaped = u
    elif profile_name == "cubic_smoothstep":
        shaped = u * u * (3.0 - 2.0 * u)
    elif profile_name == "quintic_minimum_jerk":
        shaped = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
    elif profile_name == "sinusoidal":
        shaped = 0.5 - 0.5 * np.cos(np.pi * u)
    else:
        raise ValueError(
            f"Unknown position ramp profile {profile_name!r}. "
            f"Choose from: {', '.join(POSITION_RAMP_PROFILES)}."
        )

    if shaped.ndim > 0:
        shaped = shaped.copy()
        shaped[u <= 0.0] = 0.0
        shaped[u >= 1.0] = 1.0
    return shaped


def build_position_ramp_sequence(
    profile_name: str,
    samples: int,
    start_um: np.ndarray,
    stop_um: np.ndarray,
    depth_uK: float,
    load_end_ms: float,
    move_end_ms: float,
    hold_end_ms: float,
) -> RampSequence:
    """Build a load-move-hold AOD ramp using a named position profile."""

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    u = np.linspace(0.0, 1.0, samples)
    shaped_u = position_profile_fraction(profile_name, u)
    move_centers_um = start_um + shaped_u[:, np.newaxis] * (stop_um - start_um)
    times_ms = np.concatenate(
        [[0.0], np.linspace(load_end_ms, move_end_ms, samples), [hold_end_ms]]
    )
    centers_um = np.vstack([start_um, move_centers_um, stop_um])
    depths_uK = np.concatenate([[0.0], np.full(samples, depth_uK), [depth_uK]])
    return RampSequence(
        times_s=ms(times_ms),
        centers_m=um(centers_um),
        depths_uK=depths_uK,
    )


def build_transfer_problem(
    position_profile: str,
    *,
    position_samples: int = POSITION_SAMPLES,
    aod_depth_uK: float = AOD_DEPTH_UK,
    ensemble_size: int = ENSEMBLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> TransferProblem:
    """Create the shared SLM-to-AOD transfer problem for one ramp profile."""

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
        profile_name=position_profile,
        samples=position_samples,
        start_um=np.array([0.0, 0.0, 0.0]),
        stop_um=np.array([6.0, 0.0, 0.0]),
        depth_uK=aod_depth_uK,
        load_end_ms=0.08,
        move_end_ms=0.48,
        hold_end_ms=0.68,
    )
    config = SimulationConfig(
        initial_temperature_uK=8.0,
        timestep_s=float(ms(0.0002)),
        duration_s=float(ms(0.68)),
        ensemble_size=ensemble_size,
        random_seed=random_seed,
        loss_radius_m=float(um(30.0)),
        store_trajectories=False,
    )
    return TransferProblem(
        static_trap=slm_trap,
        moving_trap_base=aod_trap_base,
        ramp=ramp,
        config=config,
    )


def analyze_position_ramp_case(
    profile_name: str,
    problem: TransferProblem,
    result: SimulationResult,
) -> PositionRampCase:
    """Compute final AOD transfer metrics."""

    final_aod_center_m, final_aod_depth_uK = problem.ramp.at(problem.config.duration_s)
    final_aod_trap = problem.moving_trap_base.with_center_depth(
        final_aod_center_m,
        final_aod_depth_uK,
    )
    return PositionRampCase(
        profile_name=profile_name,
        problem=problem,
        result=result,
        aod_capture_probability=capture_probability(
            result,
            final_aod_trap,
            mass_kg=problem.config.mass_kg,
        ),
    )


def compare_position_ramps(
    profile_names: Sequence[str] = POSITION_RAMP_PROFILES,
    *,
    position_samples: int = POSITION_SAMPLES,
    aod_depth_uK: float = AOD_DEPTH_UK,
    ensemble_size: int = ENSEMBLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> list[PositionRampCase]:
    """Run the same transfer simulation for several position profiles."""

    cases: list[PositionRampCase] = []
    for profile_name in profile_names:
        problem = build_transfer_problem(
            profile_name,
            position_samples=position_samples,
            aod_depth_uK=aod_depth_uK,
            ensemble_size=ensemble_size,
            random_seed=random_seed,
        )
        result = run_simulation(
            problem.static_trap,
            problem.moving_trap_base,
            problem.ramp,
            problem.config,
        )
        cases.append(analyze_position_ramp_case(profile_name, problem, result))
    return cases


def print_position_ramp_comparison(cases: Sequence[PositionRampCase]) -> None:
    """Print a compact metric table for the compared position ramps."""

    print("Position ramp comparison")
    print(
        f"{'profile':<24} {'1-survival':>12} {'1-AOD capture':>15} "
        f"{'temp gain uK':>13}"
    )
    for case in cases:
        result = case.result
        print(
            f"{case.profile_name:<24} "
            f"{1.0 - result.survival_probability:12.4f} "
            f"{1.0 - case.aod_capture_probability:15.4f} "
            f"{result.temperature_gain_uK:13.4f}"
        )


def plot_position_ramp_comparison(cases: Sequence[PositionRampCase]):
    """Create a three-panel position-ramp comparison figure."""

    import matplotlib.pyplot as plt

    if not cases:
        raise ValueError("At least one case is required for plotting.")

    names = [case.profile_name for case in cases]
    x = np.arange(len(cases))

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)

    ax = axes[0]
    for case in cases:
        ramp = case.problem.ramp
        ax.plot(
            ramp.times_s * 1.0e3,
            ramp.centers_m[:, 0] * 1.0e6,
            linewidth=1.8,
            label=case.profile_name,
        )
    ax.set_title("Position Ramp Profiles")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("AOD x position (um)")
    ax.legend(loc="upper left")

    width = 0.36
    ax = axes[1]
    survival_error = [1.0 - case.result.survival_probability for case in cases]
    capture_error = [1.0 - case.aod_capture_probability for case in cases]
    survival_bars = ax.bar(
        x - width / 2.0, survival_error, width, label="1 - survival"
    )
    capture_bars = ax.bar(
        x + width / 2.0, capture_error, width, label="1 - AOD capture"
    )
    ax.set_title("Transfer Error Probability")
    _format_small_probability_axis(ax, cases, survival_error + capture_error)
    _annotate_probability_bars(ax, survival_bars)
    _annotate_probability_bars(ax, capture_bars)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(loc="upper right")

    ax = axes[2]
    temperature_gain = [case.result.temperature_gain_uK for case in cases]
    ax.bar(x, temperature_gain, color="tab:orange", alpha=0.75)
    ax.set_title("Temperature Gain")
    ax.set_ylabel("uK")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")

    return figure, axes


def _format_small_probability_axis(
    ax,
    cases: Sequence[PositionRampCase],
    values: Sequence[float],
) -> None:
    ensemble_size = min(case.problem.config.ensemble_size for case in cases)
    one_atom_fraction = 1.0 / ensemble_size
    max_value = max(values, default=0.0)
    upper = max(1.35 * max_value, 1.25 * one_atom_fraction)
    ax.set_ylim(0.0, upper)
    ax.axhline(
        one_atom_fraction,
        color="0.55",
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )
    ax.text(
        0.99,
        one_atom_fraction / upper,
        "1 atom",
        color="0.35",
        fontsize=8,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )


def _annotate_probability_bars(ax, bars) -> None:
    _, upper = ax.get_ylim()
    y_offset = 0.018 * upper
    zero_y = 0.035 * upper
    for bar in bars:
        value = float(bar.get_height())
        label = f"{100.0 * value:.1f}%"
        y = value + y_offset if value > 0.0 else zero_y
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def main() -> None:
    """Run the position-ramp comparison example."""

    cases = compare_position_ramps(
        POSITION_RAMP_PROFILES,
        position_samples=POSITION_SAMPLES,
        aod_depth_uK=AOD_DEPTH_UK,
        ensemble_size=ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
    )
    print_position_ramp_comparison(cases)

    if SAVE_PLOT_PATH or SHOW_PLOT:
        figure, _ = plot_position_ramp_comparison(cases)
        if SAVE_PLOT_PATH:
            save_path = os.path.abspath(SAVE_PLOT_PATH)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            figure.savefig(save_path, dpi=180)
            print(f"saved: {save_path}")
        if SHOW_PLOT:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
