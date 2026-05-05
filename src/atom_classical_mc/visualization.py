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
    """Create a 3x2 Matplotlib summary figure for a stored-trajectory run."""

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

    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    trajectory_axis = axes[0, 0]
    mean_position_axis = axes[0, 1]
    kinetic_axis = axes[1, 0]
    loss_axis = axes[1, 1]
    position_ramp_axis = axes[2, 0]
    depth_ramp_axis = axes[2, 1]

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

    kinetic_axis.plot(
        times_ms,
        mean_kinetic_energy_time_series_uK(result),
        color="tab:orange",
    )
    kinetic_axis.set_xlabel("time (ms)")
    kinetic_axis.set_ylabel("mean kinetic energy (uK)")
    kinetic_axis.set_title("Heating During Ramp")

    loss_axis.plot(
        times_ms,
        loss_probability_time_series(result),
        color="tab:red",
    )
    loss_axis.set_xlabel("time (ms)")
    loss_axis.set_ylabel("loss probability")
    loss_axis.set_ylim(-0.02, 1.02)
    loss_axis.set_title("Loss = 1 - Survival")

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
