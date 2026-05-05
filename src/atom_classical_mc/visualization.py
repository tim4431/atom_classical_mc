"""Optional Matplotlib visualization helpers."""

from __future__ import annotations

import numpy as np

from .analysis import (
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
)
from .ramp import RampSequence
from .simulation import SimulationResult
from .trap import TrapConfig


def plot_transfer_summary(
    result: SimulationResult,
    ramp: RampSequence,
    static_trap: TrapConfig,
    max_trajectories: int = 64,
):
    """Create the trajectory-focused 2D transfer summary figure.

    This is kept as a compatibility wrapper around
    `plot_transfer_trajectory_summary`.
    """

    return plot_transfer_trajectory_summary(
        result,
        ramp,
        static_trap,
        max_trajectories=max_trajectories,
    )


def plot_transfer_trajectory_summary(
    result: SimulationResult,
    ramp: RampSequence,
    static_trap: TrapConfig,
    max_trajectories: int = 64,
):
    """Create a 2x2 spatial/ramp summary figure for a stored-trajectory run."""

    if result.trajectory_times_s is None or result.trajectory_positions_m is None:
        raise ValueError("Simulation was run without stored trajectories.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install with `pip install .[viz]`."
        ) from exc

    times_ms = result.trajectory_times_s * 1.0e3
    positions_um = result.trajectory_positions_m * 1.0e6
    aod_centers_um = np.asarray([ramp.at(t)[0] for t in result.trajectory_times_s]) * 1.0e6
    aod_depths_uK = np.asarray([ramp.at(t)[1] for t in result.trajectory_times_s])
    survivor_masks = (
        ~result.trajectory_lost
        if result.trajectory_lost is not None
        else np.ones(positions_um.shape[:2], dtype=bool)
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    trajectory_axis = axes[0, 0]
    mean_position_axis = axes[0, 1]
    position_ramp_axis = axes[1, 0]
    depth_ramp_axis = axes[1, 1]

    particle_count = positions_um.shape[1]
    shown = min(max_trajectories, particle_count)
    indices = np.linspace(0, particle_count - 1, shown, dtype=int)
    for atom_index in indices:
        trajectory_axis.plot(
            positions_um[:, atom_index, 0],
            positions_um[:, atom_index, 1],
            color="tab:blue",
            alpha=0.25,
            linewidth=0.8,
        )
    trajectory_axis.plot(
        aod_centers_um[:, 0],
        aod_centers_um[:, 1],
        color="tab:red",
        linewidth=2.0,
        label="AOD center",
    )
    trajectory_axis.scatter(
        [static_trap.center_m[0] * 1.0e6],
        [static_trap.center_m[1] * 1.0e6],
        color="black",
        marker="x",
        label="SLM center",
    )
    trajectory_axis.set_xlabel("x (um)")
    trajectory_axis.set_ylabel("y (um)")
    trajectory_axis.set_title("Ensemble Trajectories")
    trajectory_axis.legend(loc="best")
    trajectory_axis.axis("equal")

    mean_positions_um = np.full((positions_um.shape[0], 3), np.nan, dtype=float)
    for time_index, survivor_mask in enumerate(survivor_masks):
        if np.any(survivor_mask):
            mean_positions_um[time_index] = np.mean(
                positions_um[time_index, survivor_mask], axis=0
            )
    for axis_index, label in enumerate(("x", "y", "z")):
        mean_position_axis.plot(
            times_ms,
            mean_positions_um[:, axis_index],
            label=f"mean atom {label}",
        )
        mean_position_axis.plot(
            times_ms,
            aod_centers_um[:, axis_index],
            linestyle="--",
            linewidth=1.0,
            label=f"AOD {label}",
        )
    mean_position_axis.set_xlabel("time (ms)")
    mean_position_axis.set_ylabel("position (um)")
    mean_position_axis.set_title("Mean Atom Position")
    mean_position_axis.legend(loc="best", ncol=2, fontsize="small")

    for axis_index, label in enumerate(("x", "y", "z")):
        position_ramp_axis.plot(times_ms, aod_centers_um[:, axis_index], label=label)
    position_ramp_axis.set_xlabel("time (ms)")
    position_ramp_axis.set_ylabel("AOD position (um)")
    position_ramp_axis.set_title("AOD Position Ramp")
    position_ramp_axis.legend(loc="best")

    depth_ramp_axis.plot(times_ms, aod_depths_uK, color="tab:purple")
    depth_ramp_axis.set_xlabel("time (ms)")
    depth_ramp_axis.set_ylabel("AOD depth (uK)")
    depth_ramp_axis.set_title("AOD Depth Ramp")

    return figure, axes


def plot_transfer_energy_summary(
    result: SimulationResult,
    initial_mode_summary: dict[str, dict[str, float]] | None = None,
    final_mode_summary: dict[str, dict[str, float]] | None = None,
):
    """Create a 2x2 heating/loss/motional-occupation summary figure."""

    if result.trajectory_times_s is None:
        raise ValueError("Simulation was run without stored trajectories.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install with `pip install .[viz]`."
        ) from exc

    times_ms = result.trajectory_times_s * 1.0e3
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    kinetic_axis = axes[0, 0]
    loss_axis = axes[0, 1]
    initial_mode_axis = axes[1, 0]
    final_mode_axis = axes[1, 1]

    kinetic_axis.plot(
        times_ms,
        mean_kinetic_energy_time_series_uK(result),
        color="tab:orange",
    )
    kinetic_axis.set_xlabel("time (ms)")
    kinetic_axis.set_ylabel("mean kinetic energy (uK)")
    kinetic_axis.set_title("Heating During Ramp")

    loss_probability = loss_probability_time_series(result)
    loss_axis.plot(times_ms, loss_probability, color="tab:red")
    loss_axis.set_xlabel("time (ms)")
    loss_axis.set_ylabel("loss probability")
    loss_axis.set_ylim(0.0, _probability_axis_upper_limit(loss_probability))
    loss_axis.set_title("Loss = 1 - Survival")

    _plot_mode_occupation_summary(
        initial_mode_axis,
        initial_mode_summary,
        "Initial SLM Motional Occupation",
    )
    _plot_mode_occupation_summary(
        final_mode_axis,
        final_mode_summary,
        "Final AOD Motional Occupation",
    )

    return figure, axes


def plot_transfer_trajectories_3d(
    result: SimulationResult,
    ramp: RampSequence,
    static_trap: TrapConfig | None = None,
    max_trajectories: int = 96,
):
    """Plot atom trajectories and the moving AOD center path in 3D."""

    if result.trajectory_times_s is None or result.trajectory_positions_m is None:
        raise ValueError("Simulation was run without stored trajectories.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install with `pip install .[viz]`."
        ) from exc

    positions_um = result.trajectory_positions_m * 1.0e6
    aod_centers_um = np.asarray([ramp.at(t)[0] for t in result.trajectory_times_s]) * 1.0e6

    figure = plt.figure(figsize=(9, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    particle_count = positions_um.shape[1]
    shown = min(max_trajectories, particle_count)
    indices = np.linspace(0, particle_count - 1, shown, dtype=int)
    for atom_index in indices:
        axis.plot(
            positions_um[:, atom_index, 0],
            positions_um[:, atom_index, 1],
            positions_um[:, atom_index, 2],
            color="tab:blue",
            alpha=0.22,
            linewidth=0.8,
        )

    axis.plot(
        aod_centers_um[:, 0],
        aod_centers_um[:, 1],
        aod_centers_um[:, 2],
        color="tab:red",
        linewidth=2.5,
        label="AOD center",
    )
    axis.scatter(
        aod_centers_um[-1:, 0],
        aod_centers_um[-1:, 1],
        aod_centers_um[-1:, 2],
        color="tab:red",
        s=36,
        label="AOD final",
    )
    if static_trap is not None:
        slm_center_um = np.asarray(static_trap.center_m, dtype=float) * 1.0e6
        axis.scatter(
            [slm_center_um[0]],
            [slm_center_um[1]],
            [slm_center_um[2]],
            color="black",
            marker="x",
            s=60,
            label="SLM center",
        )
    else:
        slm_center_um = None

    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_zlabel("z (um)")
    axis.set_title("3D Atom Trajectories and AOD Path")
    axis.legend(loc="best")
    _set_equal_3d_limits(axis, positions_um[:, indices], aod_centers_um, slm_center_um)

    return figure, axis


def _probability_axis_upper_limit(probability: np.ndarray) -> float:
    finite_probability = np.asarray(probability, dtype=float)
    finite_probability = finite_probability[np.isfinite(finite_probability)]
    if finite_probability.size == 0:
        return 0.01

    peak = float(np.max(finite_probability))
    if peak <= 0.0:
        return 0.01
    return min(1.0, max(0.01, 1.25 * peak))


def _plot_mode_occupation_summary(axis, summary, title: str) -> None:
    axis.set_title(title)
    axis.set_ylabel("mean occupation n")
    if not summary:
        axis.text(
            0.5,
            0.5,
            "not computed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        return

    labels = list(summary.keys())
    means = np.asarray([summary[label]["mean"] for label in labels], dtype=float)
    medians = np.asarray([summary[label]["median"] for label in labels], dtype=float)
    x = np.arange(len(labels))
    axis.bar(x, means, color="tab:blue", alpha=0.75, label="mean")
    axis.scatter(x, medians, color="black", marker="_", s=160, label="median")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    finite_means = means[np.isfinite(means)]
    if finite_means.size > 0:
        axis.set_ylim(0.0, max(1.0, 1.2 * float(np.max(finite_means))))
    axis.legend(loc="best")


def _set_equal_3d_limits(axis, trajectory_points_um, aod_centers_um, slm_center_um) -> None:
    point_sets = [
        np.reshape(trajectory_points_um, (-1, 3)),
        np.reshape(aod_centers_um, (-1, 3)),
    ]
    if slm_center_um is not None:
        point_sets.append(np.reshape(slm_center_um, (1, 3)))

    points = np.vstack(point_sets)
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if finite_points.size == 0:
        return

    minimum = np.min(finite_points, axis=0)
    maximum = np.max(finite_points, axis=0)
    center = 0.5 * (minimum + maximum)
    half_range = 0.5 * float(np.max(maximum - minimum))
    half_range = max(half_range, 0.5)
    padding = 1.1 * half_range

    axis.set_xlim(center[0] - padding, center[0] + padding)
    axis.set_ylim(center[1] - padding, center[1] + padding)
    axis.set_zlim(center[2] - padding, center[2] + padding)
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect((1.0, 1.0, 1.0))
