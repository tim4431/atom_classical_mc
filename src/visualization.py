"""Optional Matplotlib visualization helpers."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from .analysis import (
    kinetic_temperature_time_series_uK,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
)
from .constants import RB87_MASS_KG
from .harmonic import MotionalDecomposition
from .laser import LaserBeam
from .ramp import RampSequence
from .simulation import SimulationResult
from .trap import TrapConfig, total_potential

FramePlane = Literal["xy", "xz", "yz"]
FrameView = Literal["xy", "xz", "yz", "3d", "split"]


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
    initial_modes: MotionalDecomposition | None = None,
    final_modes: MotionalDecomposition | None = None,
    initial_mask: np.ndarray | None = None,
    final_mask: np.ndarray | None = None,
    occupation_bins: int = 48,
):
    """Create a 2x2 heating/loss/motional-occupation distribution figure."""

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

    _plot_mode_occupation_distribution(
        initial_mode_axis,
        initial_modes,
        "Initial SLM Motional Occupation",
        mask=initial_mask,
        bins=occupation_bins,
    )
    _plot_mode_occupation_distribution(
        final_mode_axis,
        final_modes,
        "Final AOD Motional Occupation",
        mask=final_mask,
        bins=occupation_bins,
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


def _plot_mode_occupation_distribution(
    axis,
    decomposition: MotionalDecomposition | None,
    title: str,
    mask: np.ndarray | None = None,
    bins: int = 48,
) -> None:
    axis.set_title(title)
    axis.set_xlabel("occupation estimate n")
    axis.set_ylabel("probability")
    if decomposition is None:
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

    occupations = np.asarray(decomposition.mean_occupations, dtype=float)
    if mask is not None:
        occupations = occupations[np.asarray(mask, dtype=bool)]
    occupations = occupations[np.all(np.isfinite(occupations), axis=1)]
    if occupations.size == 0:
        axis.text(
            0.5,
            0.5,
            "no atoms",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        return

    x_max = _occupation_axis_upper_limit(occupations)
    edges = np.linspace(0.0, x_max, bins + 1)
    clipped = np.clip(occupations, edges[0], edges[-1])
    for mode_index, label in enumerate(decomposition.mode_labels):
        counts, _ = np.histogram(clipped[:, mode_index], bins=edges)
        probability = counts / np.sum(counts)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.step(
            centers,
            probability,
            where="mid",
            linewidth=1.8,
            label=label,
        )
        axis.fill_between(
            centers,
            probability,
            step="mid",
            alpha=0.12,
        )

    axis.set_xlim(0.0, x_max)
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="best")


def _occupation_axis_upper_limit(occupations: np.ndarray) -> float:
    finite = occupations[np.isfinite(occupations)]
    if finite.size == 0:
        return 1.0
    upper = float(np.percentile(finite, 99.5))
    maximum = float(np.max(finite))
    if maximum <= 20.0:
        return max(1.0, np.ceil(maximum))
    return max(1.0, np.ceil(min(maximum, upper)))


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


# --- per-frame rendering --------------------------------------------------
#
# The trap is U(r) = -U0 exp(-2(x^2+y^2)/wr^2 - 2 z^2/wz^2). Two views:
#   * 2D: heatmap of the *summed* potential on a chosen plane (uK).
#         Iso-contour at U = -U0/e^2 marks the waist boundary cleanly.
#   * 3D: one wireframe ellipsoid per trap at the U = -U_i/e^2 surface,
#         which for a Gaussian is exactly the (wr, wr, wz) ellipsoid.
#
# `draw_frame(t, result, ...)` composes both views for one timestep.


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_PLANE_AXES: dict[FramePlane, tuple[str, str, str]] = {
    "xy": ("x", "y", "z"),
    "xz": ("x", "z", "y"),
    "yz": ("y", "z", "x"),
}


def _moving_trap_at(
    moving_trap_base: TrapConfig, ramp: RampSequence, time_s: float
) -> TrapConfig:
    """Snapshot the moving trap at one ramp time (mirror of simulation._moving_trap_at)."""
    center, depth = ramp.at(time_s)
    return moving_trap_base.with_center_depth(center, depth)


def _trap_list(trap_or_iter) -> list[TrapConfig]:
    if isinstance(trap_or_iter, TrapConfig):
        return [trap_or_iter]
    return list(trap_or_iter)


def _interpolate_trajectory_state(
    result: SimulationResult, time_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Pick the stored trajectory snapshot closest to `time_s`.

    Returns (positions_m, velocities_m_per_s, lost) at the nearest stored
    sample. We snap to the nearest index instead of interpolating to keep
    the `lost` flag boolean and avoid mid-step physics inconsistency.
    """
    if result.trajectory_times_s is None or result.trajectory_positions_m is None:
        raise ValueError(
            "Simulation was run without stored trajectories; "
            "set SimulationConfig.store_trajectories=True."
        )
    times = np.asarray(result.trajectory_times_s, dtype=float)
    index = int(np.argmin(np.abs(times - float(time_s))))
    positions = np.asarray(result.trajectory_positions_m[index], dtype=float)
    if result.trajectory_velocities_m_per_s is not None:
        velocities = np.asarray(result.trajectory_velocities_m_per_s[index], dtype=float)
    else:
        velocities = np.zeros_like(positions)
    if result.trajectory_lost is not None:
        lost = np.asarray(result.trajectory_lost[index], dtype=bool)
    else:
        lost = np.zeros(positions.shape[0], dtype=bool)
    return positions, velocities, lost


def _auto_extent_um(
    traps: Iterable[TrapConfig],
    result: SimulationResult | None,
    pad_waists: float = 3.0,
) -> tuple[float, float, float, float, float, float]:
    """Pick a (xmin, xmax, ymin, ymax, zmin, zmax) bounding box in um.

    Includes every trap center plus `pad_waists` of its largest scale, and
    the full extent of stored trajectory positions when available.
    """
    centers = []
    halfwidths = []
    for trap in traps:
        centers.append(np.asarray(trap.center_m, dtype=float))
        scales = np.asarray(trap.scales_m, dtype=float)
        halfwidths.append(pad_waists * scales)
    centers_arr = np.asarray(centers) * 1.0e6
    halfwidths_arr = np.asarray(halfwidths) * 1.0e6
    lo = (centers_arr - halfwidths_arr).min(axis=0)
    hi = (centers_arr + halfwidths_arr).max(axis=0)

    if result is not None and result.trajectory_positions_m is not None:
        positions_um = np.asarray(result.trajectory_positions_m, dtype=float) * 1.0e6
        finite = positions_um[np.all(np.isfinite(positions_um), axis=-1)]
        if finite.size:
            lo = np.minimum(lo, finite.min(axis=(0, 1)) if finite.ndim == 3 else finite.min(axis=0))
            hi = np.maximum(hi, finite.max(axis=(0, 1)) if finite.ndim == 3 else finite.max(axis=0))

    span = hi - lo
    pad = 0.05 * np.where(span > 0, span, 1.0)
    lo = lo - pad
    hi = hi + pad
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])


def draw_traps_2d(
    ax,
    traps: Iterable[TrapConfig],
    *,
    plane: FramePlane = "xy",
    extent_um: tuple[float, float, float, float] | None = None,
    slice_coord_um: float = 0.0,
    grid_n: int = 220,
    cmap: str = "magma",
    vmin_uK: float | None = None,
    show_contours: bool = True,
    show_centers: bool = True,
) -> None:
    """Heatmap the summed Gaussian potential on a planar slice in microkelvin.

    `plane` selects which two axes span the figure; `slice_coord_um` is the
    fixed value of the third axis. The drawn quantity is U(r) in microkelvin
    (negative inside trap wells); contours mark each trap's 1/e^2 boundary.
    Pass `vmin_uK` (a negative value) to fix the color range across frames;
    by default it is the negative of the deepest currently-present trap.
    """
    from .units import joule_to_microkelvin
    import matplotlib.pyplot as plt  # noqa: F401  (ensure backend init)

    trap_list = _trap_list(traps)
    if not trap_list:
        return

    horiz_name, vert_name, depth_name = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    d_axis = _AXIS_INDEX[depth_name]

    if extent_um is None:
        full = _auto_extent_um(trap_list, None)
        h_lo, h_hi = full[2 * h_axis], full[2 * h_axis + 1]
        v_lo, v_hi = full[2 * v_axis], full[2 * v_axis + 1]
    else:
        h_lo, h_hi, v_lo, v_hi = extent_um

    hh = np.linspace(h_lo, h_hi, grid_n)
    vv = np.linspace(v_lo, v_hi, grid_n)
    H, V = np.meshgrid(hh, vv, indexing="xy")

    positions = np.zeros(H.shape + (3,), dtype=float)
    positions[..., h_axis] = H * 1.0e-6
    positions[..., v_axis] = V * 1.0e-6
    positions[..., d_axis] = slice_coord_um * 1.0e-6

    potential_j = total_potential(trap_list, positions)
    potential_uK = joule_to_microkelvin(potential_j)

    if vmin_uK is None:
        vmax_depth = max(
            (trap.depth_uK for trap in trap_list if trap.depth_uK > 0), default=1.0
        )
        vmin = -float(vmax_depth)
    else:
        vmin = float(vmin_uK)
    ax.imshow(
        potential_uK,
        extent=(h_lo, h_hi, v_lo, v_hi),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=0.0,
        aspect="equal",
        interpolation="bilinear",
    )

    if show_contours:
        for trap in trap_list:
            if trap.depth_uK <= 0.0:
                continue
            iso = -trap.depth_uK / math.e ** 2
            ax.contour(
                hh, vv, potential_uK, levels=[iso],
                colors="white", linewidths=0.6, alpha=0.55,
            )

    if show_centers:
        for trap in trap_list:
            c = np.asarray(trap.center_m, dtype=float) * 1.0e6
            if abs(c[d_axis] - slice_coord_um) > 0.5 * max(
                abs(h_hi - h_lo), abs(v_hi - v_lo)
            ):
                continue
            ax.plot(
                c[h_axis], c[v_axis], marker="+", color="#ffd700",
                markersize=8, markeredgewidth=1.4,
            )

    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(v_lo, v_hi)
    ax.set_xlabel(f"{horiz_name} (um)")
    ax.set_ylabel(f"{vert_name} (um)")


def draw_atoms_2d(
    ax,
    positions_m: NDArray[np.float64],
    lost: NDArray[np.bool_] | None,
    *,
    plane: FramePlane = "xy",
    bound_color: str = "#1ed1ff",
    lost_color: str = "#888888",
    size: float = 14.0,
) -> None:
    """Scatter atom positions on a 2D plane, colored by lost mask."""
    horiz_name, vert_name, _ = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    positions_um = np.asarray(positions_m, dtype=float) * 1.0e6
    if lost is None:
        lost = np.zeros(positions_um.shape[0], dtype=bool)

    bound = ~lost
    if np.any(bound):
        ax.scatter(
            positions_um[bound, h_axis], positions_um[bound, v_axis],
            s=size, c=bound_color, edgecolors="white", linewidths=0.4,
            alpha=0.9, zorder=5,
        )
    if np.any(lost):
        ax.scatter(
            positions_um[lost, h_axis], positions_um[lost, v_axis],
            s=size * 0.6, c=lost_color, alpha=0.4,
            linewidths=0, zorder=4,
        )


def _depth_alpha_factor(
    depth_uK: float, reference_uK: float | None, max_depth_uK: float
) -> float:
    """Linear opacity factor in [0, 1] proportional to trap depth.

    With `reference_uK=None` we normalize against the deepest trap in the
    current scene, so the strongest trap is always fully opaque. Pass an
    explicit reference (e.g. the global max across an animation) to keep
    opacity comparable across frames.
    """
    ref = float(reference_uK) if reference_uK is not None else float(max_depth_uK)
    if ref <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(depth_uK) / ref))


def draw_tweezer_beams_3d(
    ax,
    traps: Iterable[TrapConfig],
    *,
    surface_n: int = 28,
    axial_extent_zR: float = 3.0,
    color_static: str = "#ffd24a",
    color_moving: str = "#ff5050",
    moving_indices: Iterable[int] | None = None,
    alpha: float = 0.22,
    edge_alpha: float = 0.35,
    focal_ring_alpha: float = 0.85,
    depth_reference_uK: float | None = None,
    show_focal_ring: bool = True,
) -> None:
    """Draw each TrapConfig as a Gaussian-beam hyperboloid (the conventional view).

    The conventional Gaussian beam is `w(z) = w0 * sqrt(1 + (z/zR)^2)`,
    pinching to `w0 = waist_radial_m` at the focal plane and flaring on
    the scale of `zR = axial_scale_m`. We render the surface of revolution
    around the z-axis as a translucent hourglass.

    Each trap's opacity scales linearly with `trap.depth_uK /
    depth_reference_uK` so a ramping AOD visibly fades in/out across
    frames. `depth_reference_uK=None` falls back to the deepest currently-
    present trap, fine for a still; pass the global max depth of the
    ramp for animations.

    Note: the simulator's potential is the separable Gaussian
    `exp(-2(x^2+y^2)/wr^2 - 2 z^2/wz^2)`, whose 1/e^2 iso-surface is the
    (wr, wr, wz) ellipsoid drawn by `draw_trap_ellipsoids_3d`. The
    hyperboloid drawn here is the *beam profile* people recognize as a
    focused Gaussian beam, not a strict iso-potential surface — the
    visual mismatch is intentional.
    """
    trap_list = _trap_list(traps)
    moving_set = set(int(i) for i in moving_indices) if moving_indices else set()
    max_depth = max((t.depth_uK for t in trap_list), default=0.0)

    for index, trap in enumerate(trap_list):
        if trap.depth_uK <= 0.0:
            continue
        depth_factor = _depth_alpha_factor(
            trap.depth_uK, depth_reference_uK, max_depth
        )
        if depth_factor <= 0.0:
            continue

        center = np.asarray(trap.center_m, dtype=float) * 1.0e6
        w0 = float(trap.waist_radial_m) * 1.0e6
        zR = float(trap.axial_scale_m) * 1.0e6

        z_local = np.linspace(-axial_extent_zR * zR, axial_extent_zR * zR, surface_n)
        theta = np.linspace(0.0, 2.0 * np.pi, surface_n)
        Z, Theta = np.meshgrid(z_local, theta, indexing="ij")
        R = w0 * np.sqrt(1.0 + (Z / zR) ** 2)
        X = R * np.cos(Theta) + center[0]
        Y = R * np.sin(Theta) + center[1]
        Z_world = Z + center[2]

        color = color_moving if index in moving_set else color_static
        ax.plot_surface(
            X, Y, Z_world,
            color=color, alpha=alpha * depth_factor,
            linewidth=0.0, antialiased=True,
            rcount=14, ccount=14, shade=False,
        )
        ax.plot(
            center[0] + R[:, 0], np.full_like(z_local, center[1]),
            Z_world[:, 0], color=color, linewidth=1.0,
            alpha=edge_alpha * depth_factor,
        )
        ax.plot(
            center[0] - R[:, 0], np.full_like(z_local, center[1]),
            Z_world[:, 0], color=color, linewidth=1.0,
            alpha=edge_alpha * depth_factor,
        )
        if show_focal_ring:
            theta_ring = np.linspace(0.0, 2.0 * np.pi, 64)
            ax.plot(
                center[0] + w0 * np.cos(theta_ring),
                center[1] + w0 * np.sin(theta_ring),
                np.full_like(theta_ring, center[2]),
                color=color, linewidth=1.4,
                alpha=focal_ring_alpha * depth_factor,
            )


def draw_tweezer_beam_side_2d(
    ax,
    traps: Iterable[TrapConfig],
    *,
    plane: FramePlane = "xz",
    extent_um: tuple[float, float, float, float] | None = None,
    color_static: str = "#ffd24a",
    color_moving: str = "#ff5050",
    moving_indices: Iterable[int] | None = None,
    fill_alpha: float = 0.18,
    edge_alpha: float = 0.85,
    depth_reference_uK: float | None = None,
) -> None:
    """Side-view hyperbolic envelope `w(z) = w0 sqrt(1 + (z/zR)^2)` per trap.

    Draws the two bounding curves and a translucent fill — the standard
    way Gaussian beams are sketched in textbooks and tweezer papers.
    Only meaningful for `plane in ('xz', 'yz')`. Opacity scales with
    `trap.depth_uK / depth_reference_uK` (defaults to the deepest trap
    in the scene if None).
    """
    if plane == "xy":
        return
    trap_list = _trap_list(traps)
    moving_set = set(int(i) for i in moving_indices) if moving_indices else set()
    max_depth = max((t.depth_uK for t in trap_list), default=0.0)

    horiz_name, vert_name, _ = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    if vert_name != "z":
        return  # propagation axis is z; only xz/yz make sense

    if extent_um is None:
        full = _auto_extent_um(trap_list, None)
        v_lo, v_hi = full[2 * v_axis], full[2 * v_axis + 1]
    else:
        v_lo, v_hi = extent_um[2], extent_um[3]
    z_local = np.linspace(v_lo, v_hi, 121)

    for index, trap in enumerate(trap_list):
        if trap.depth_uK <= 0.0:
            continue
        depth_factor = _depth_alpha_factor(
            trap.depth_uK, depth_reference_uK, max_depth
        )
        if depth_factor <= 0.0:
            continue
        center = np.asarray(trap.center_m, dtype=float) * 1.0e6
        w0 = float(trap.waist_radial_m) * 1.0e6
        zR = float(trap.axial_scale_m) * 1.0e6

        dz = z_local - center[2]
        r = w0 * np.sqrt(1.0 + (dz / zR) ** 2)
        h_pos = center[h_axis] + r
        h_neg = center[h_axis] - r

        color = color_moving if index in moving_set else color_static
        ax.fill_betweenx(
            z_local, h_neg, h_pos, color=color,
            alpha=fill_alpha * depth_factor, linewidth=0.0,
        )
        ax.plot(h_pos, z_local, color=color, linewidth=1.1,
                alpha=edge_alpha * depth_factor)
        ax.plot(h_neg, z_local, color=color, linewidth=1.1,
                alpha=edge_alpha * depth_factor)


def draw_trap_ellipsoids_3d(
    ax,
    traps: Iterable[TrapConfig],
    *,
    surface_n: int = 24,
    color_static: str = "#ffd700",
    color_moving: str = "#ff5050",
    moving_indices: Iterable[int] | None = None,
    alpha: float = 0.18,
) -> None:
    """Draw one wireframe ellipsoid per trap at its 1/e^2 (waist) boundary.

    For U = -U0 exp(-2 r^2/w^2), U = -U0/e^2 ⇔ r = w along each axis, so
    the iso-surface is exactly the ellipsoid with semi-axes (wr, wr, wz).
    Skips traps with zero depth. This is the strict iso-potential surface;
    use `draw_tweezer_beams_3d` for the conventional Gaussian-beam hourglass.
    """
    trap_list = _trap_list(traps)
    moving_set = set(int(i) for i in moving_indices) if moving_indices else set()

    u = np.linspace(0.0, 2.0 * np.pi, surface_n)
    v = np.linspace(0.0, np.pi, surface_n // 2 + 1)
    cu, cv = np.cos(u), np.sin(u)
    sv, cv_ = np.sin(v), np.cos(v)

    for index, trap in enumerate(trap_list):
        if trap.depth_uK <= 0.0:
            continue
        center = np.asarray(trap.center_m, dtype=float) * 1.0e6
        wr = float(trap.waist_radial_m) * 1.0e6
        wz = float(trap.axial_scale_m) * 1.0e6

        x = center[0] + wr * np.outer(cu, sv)
        y = center[1] + wr * np.outer(np.sin(u), sv)
        z = center[2] + wz * np.outer(np.ones_like(u), cv_)

        color = color_moving if index in moving_set else color_static
        ax.plot_wireframe(
            x, y, z, color=color, linewidth=0.6, alpha=alpha, rcount=12, ccount=12,
        )


def draw_atoms_3d(
    ax,
    positions_m: NDArray[np.float64],
    lost: NDArray[np.bool_] | None,
    *,
    bound_color: str = "#1ed1ff",
    lost_color: str = "#888888",
    size: float = 16.0,
) -> None:
    """Scatter atom positions in 3D, colored by lost mask."""
    positions_um = np.asarray(positions_m, dtype=float) * 1.0e6
    if lost is None:
        lost = np.zeros(positions_um.shape[0], dtype=bool)
    bound = ~lost
    if np.any(bound):
        ax.scatter(
            positions_um[bound, 0], positions_um[bound, 1], positions_um[bound, 2],
            s=size, c=bound_color, edgecolors="white", linewidths=0.3,
            alpha=0.9, depthshade=False,
        )
    if np.any(lost):
        ax.scatter(
            positions_um[lost, 0], positions_um[lost, 1], positions_um[lost, 2],
            s=size * 0.6, c=lost_color, alpha=0.35, depthshade=False,
        )


def draw_atom_trails_2d(
    ax,
    result: SimulationResult,
    end_index: int,
    *,
    plane: FramePlane = "xy",
    trail_samples: int = 6,
    color: str = "#1ed1ff",
    max_atoms: int = 80,
) -> None:
    """Draw fading position trails from the previous few stored snapshots."""
    if result.trajectory_positions_m is None or trail_samples <= 1:
        return
    horiz_name, vert_name, _ = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]

    start = max(0, end_index - trail_samples + 1)
    if end_index <= start:
        return
    positions_um = np.asarray(
        result.trajectory_positions_m[start : end_index + 1], dtype=float
    ) * 1.0e6
    n_atoms = positions_um.shape[1]
    indices = np.linspace(0, n_atoms - 1, min(max_atoms, n_atoms), dtype=int)

    for atom in indices:
        traj = positions_um[:, atom, :]
        ax.plot(
            traj[:, h_axis], traj[:, v_axis], color=color,
            alpha=0.25, linewidth=0.7, zorder=3,
        )


def draw_frame(
    t: float,
    result: SimulationResult,
    static_traps: TrapConfig | Iterable[TrapConfig],
    moving_trap_base: TrapConfig,
    ramp: RampSequence,
    *,
    view: FrameView = "split",
    grid_n: int = 220,
    extent_um: tuple[float, float, float, float, float, float] | None = None,
    cmap: str = "magma",
    vmin_uK: float | None = None,
    depth_reference_uK: float | None = None,
    show_trails: bool = True,
    trail_samples: int = 6,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Render one snapshot at simulation time `t`.

    Builds a fresh figure containing (depending on `view`):
      * "xy" / "xz" / "yz": one 2D potential heatmap with atoms overlaid.
        For "xz"/"yz" the conventional Gaussian-beam side envelope is drawn
        on top of the heatmap.
      * "3d": 3D scatter plus the conventional Gaussian-beam hyperboloid
        for every active trap (`draw_tweezer_beams_3d`).
      * "split" (default): xy heatmap + 3D beam view side-by-side.

    Color and opacity normalization:
      * `vmin_uK` (negative, in uK) pins the heatmap minimum so the same
        color always means the same potential.
      * `depth_reference_uK` pins the beam-envelope opacity so each trap's
        alpha reads as `trap.depth_uK / depth_reference_uK` — a ramping
        AOD will visibly fade in/out across frames.
    Both default to auto-fitting against the currently-present traps,
    which is fine for one-off stills but not for animations;
    `render_animation` supplies globally-fixed values automatically.
    `t` does not have to land on a stored snapshot — the nearest one is used.
    Returns (figure, axes_list).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    static_list = _trap_list(static_traps)
    moving_now = _moving_trap_at(moving_trap_base, ramp, float(t))
    all_traps = static_list + [moving_now]
    moving_index = len(static_list)

    positions_m, _velocities, lost = _interpolate_trajectory_state(result, t)

    times = np.asarray(result.trajectory_times_s, dtype=float)
    end_index = int(np.argmin(np.abs(times - float(t))))

    if extent_um is None:
        extent_um = _auto_extent_um(all_traps, result)

    title = title or f"t = {t * 1.0e3:.3f} ms   AOD depth = {moving_now.depth_uK:.1f} uK"

    if view in ("xy", "xz", "yz"):
        plane: FramePlane = view  # type: ignore[assignment]
        figsize = figsize or (6.0, 5.4)
        figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _, _, depth_name = _PLANE_AXES[plane]
        slice_coord = float(np.asarray(moving_now.center_m)[_AXIS_INDEX[depth_name]] * 1.0e6)
        h_axis = _AXIS_INDEX[_PLANE_AXES[plane][0]]
        v_axis = _AXIS_INDEX[_PLANE_AXES[plane][1]]
        plane_extent = (
            extent_um[2 * h_axis], extent_um[2 * h_axis + 1],
            extent_um[2 * v_axis], extent_um[2 * v_axis + 1],
        )
        draw_traps_2d(
            ax, all_traps, plane=plane, extent_um=plane_extent,
            slice_coord_um=slice_coord, grid_n=grid_n, cmap=cmap,
            vmin_uK=vmin_uK,
        )
        if plane in ("xz", "yz"):
            draw_tweezer_beam_side_2d(
                ax, all_traps, plane=plane,
                extent_um=plane_extent, moving_indices=[moving_index],
                depth_reference_uK=depth_reference_uK,
            )
        if show_trails:
            draw_atom_trails_2d(
                ax, result, end_index, plane=plane, trail_samples=trail_samples,
            )
        draw_atoms_2d(ax, positions_m, lost, plane=plane)
        ax.set_title(title)
        return figure, [ax]

    if view == "3d":
        figsize = figsize or (7.5, 6.5)
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
        draw_tweezer_beams_3d(
            ax, all_traps, moving_indices=[moving_index],
            depth_reference_uK=depth_reference_uK,
        )
        draw_atoms_3d(ax, positions_m, lost)
        ax.set_xlim(extent_um[0], extent_um[1])
        ax.set_ylim(extent_um[2], extent_um[3])
        ax.set_zlim(extent_um[4], extent_um[5])
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("z (um)")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_title(title)
        return figure, [ax]

    if view == "split":
        figsize = figsize or (12.0, 5.4)
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        ax2d = figure.add_subplot(1, 2, 1)
        ax3d = figure.add_subplot(1, 2, 2, projection="3d")
        slice_z = float(np.asarray(moving_now.center_m)[2] * 1.0e6)
        plane_extent = (extent_um[0], extent_um[1], extent_um[2], extent_um[3])
        draw_traps_2d(
            ax2d, all_traps, plane="xy", extent_um=plane_extent,
            slice_coord_um=slice_z, grid_n=grid_n, cmap=cmap,
            vmin_uK=vmin_uK,
        )
        if show_trails:
            draw_atom_trails_2d(
                ax2d, result, end_index, plane="xy", trail_samples=trail_samples,
            )
        draw_atoms_2d(ax2d, positions_m, lost, plane="xy")
        ax2d.set_title("xy focal plane")

        draw_tweezer_beams_3d(
            ax3d, all_traps, moving_indices=[moving_index],
            depth_reference_uK=depth_reference_uK,
        )
        draw_atoms_3d(ax3d, positions_m, lost)
        ax3d.set_xlim(extent_um[0], extent_um[1])
        ax3d.set_ylim(extent_um[2], extent_um[3])
        ax3d.set_zlim(extent_um[4], extent_um[5])
        ax3d.set_xlabel("x (um)")
        ax3d.set_ylabel("y (um)")
        ax3d.set_zlabel("z (um)")
        if hasattr(ax3d, "set_box_aspect"):
            ax3d.set_box_aspect((1.0, 1.0, 1.0))
        ax3d.set_title("Gaussian beam waists")

        figure.suptitle(title)
        return figure, [ax2d, ax3d]

    raise ValueError(f"Unknown view {view!r}; expected one of xy/xz/yz/3d/split.")


def render_animation(
    result: SimulationResult,
    static_traps: TrapConfig | Iterable[TrapConfig],
    moving_trap_base: TrapConfig,
    ramp: RampSequence,
    output_path: str | Path,
    *,
    view: FrameView = "split",
    n_frames: int = 80,
    fps: int = 20,
    dpi: int = 90,
    grid_n: int = 180,
    show_trails: bool = True,
    trail_samples: int = 6,
    cmap: str = "magma",
    vmin_uK: float | None = None,
    depth_reference_uK: float | None = None,
    extent_um: tuple[float, float, float, float, float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    show_progress: bool = True,
) -> Path:
    """Render a GIF by calling `draw_frame` per timestep, then stitching frames.

    The simulation must have been run with `store_trajectories=True`. The
    animation samples `n_frames` evenly-spaced times across the stored
    duration and snaps each to the nearest stored snapshot.

    Both color and beam opacity are fixed across the whole animation:
      * `vmin_uK` pins the heatmap minimum so the same color always means
        the same potential.
      * `depth_reference_uK` pins the beam-envelope alpha so each trap's
        opacity reads as `depth / depth_reference_uK`. A ramping AOD
        visibly fades in/out across frames.
    Both default to `max(static_depths ∪ ramp.depths_uK)`, the deepest
    well present anywhere in the run.

    Output extension `.gif` (default), `.webp`, or `.png` (animated PNG)
    determines the encoder.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Animation requires matplotlib and Pillow; install with `pip install .[viz]`."
        ) from exc

    if result.trajectory_times_s is None:
        raise ValueError(
            "Simulation was run without stored trajectories; "
            "set SimulationConfig.store_trajectories=True."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    static_list = _trap_list(static_traps)
    ramp_max_depth = (
        float(np.max(np.asarray(ramp.depths_uK, dtype=float)))
        if len(ramp.depths_uK) else 0.0
    )
    static_max_depth = max(
        (t.depth_uK for t in static_list if t.depth_uK > 0), default=0.0
    )
    global_max_depth = max(static_max_depth, ramp_max_depth, 1.0)
    if vmin_uK is None:
        vmin_uK = -global_max_depth
    if depth_reference_uK is None:
        depth_reference_uK = global_max_depth

    if extent_um is None:
        # Pre-compute over the worst-case extent so axes don't wiggle
        # frame-to-frame: cover every ramp waypoint as a phantom trap.
        phantom_traps = [
            moving_trap_base.with_center_depth(c, 1.0) for c in ramp.centers_m
        ]
        extent_um = _auto_extent_um(static_list + phantom_traps, result)

    duration = float(result.trajectory_times_s[-1])
    times = np.linspace(0.0, duration, max(2, int(n_frames)))

    progress = _progress_iter if show_progress else lambda x, **_: x

    images: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="atom_mc_frames_") as tmpdir:
        for k, t in enumerate(progress(times, total=len(times), desc="render frames")):
            figure, _ = draw_frame(
                float(t), result, static_traps, moving_trap_base, ramp,
                view=view, grid_n=grid_n, extent_um=extent_um, cmap=cmap,
                vmin_uK=vmin_uK, depth_reference_uK=depth_reference_uK,
                show_trails=show_trails, trail_samples=trail_samples,
                figsize=figsize,
            )
            frame_path = os.path.join(tmpdir, f"frame_{k:05d}.png")
            figure.savefig(frame_path, dpi=dpi, facecolor="white")
            plt.close(figure)
            images.append(Image.open(frame_path).convert("RGB").copy())

        _save_animation(images, out, fps=fps)

    return out


# --- trap-free cloud rendering (light-force / MOT runs) --------------------
#
# `draw_frame` / `render_animation` above are built around tweezers: they
# draw trap potentials and beam envelopes for `TrapConfig`s. Scattering-only
# runs (a MOT, a molasses) have no traps — the interesting picture is the
# cloud itself: atom positions colored by speed, next to the cooling curve.


def _cloud_extent_mm(
    result: SimulationResult, plane: FramePlane
) -> tuple[float, float, float, float]:
    """Bounding box in mm covering every stored atom position on `plane`.

    Symmetric about the origin (the MOT / quadrupole center), so the cloud
    sits centered and beam arrows aim at the field zero.
    """
    horiz_name, vert_name, _ = _PLANE_AXES[plane]
    positions_mm = np.asarray(result.trajectory_positions_m, dtype=float) * 1.0e3
    finite = positions_mm[np.all(np.isfinite(positions_mm), axis=-1)]
    if finite.size == 0:
        return -1.0, 1.0, -1.0, 1.0
    half = 1.08 * np.maximum(np.abs(finite.min(axis=0)), np.abs(finite.max(axis=0)))
    half = np.where(half > 0, half, 1.0)
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    return (
        -float(half[h_axis]), float(half[h_axis]),
        -float(half[v_axis]), float(half[v_axis]),
    )


def _draw_beam_arrows_2d(
    ax,
    beams: Iterable[LaserBeam],
    plane: FramePlane,
    extent_mm: tuple[float, float, float, float],
    *,
    color: str = "#ff5050",
) -> None:
    """Draw an inward arrow at the frame edge for each in-plane beam."""
    horiz_name, vert_name, depth_name = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    d_axis = _AXIS_INDEX[depth_name]
    h_mid = 0.5 * (extent_mm[0] + extent_mm[1])
    v_mid = 0.5 * (extent_mm[2] + extent_mm[3])
    half_h = 0.5 * (extent_mm[1] - extent_mm[0])
    half_v = 0.5 * (extent_mm[3] - extent_mm[2])

    for beam in beams:
        direction = np.asarray(beam.direction, dtype=float)
        if abs(direction[d_axis]) > 0.2:
            continue  # points out of the drawn plane
        dh, dv = direction[h_axis], direction[v_axis]
        # The beam enters from the -direction side and propagates inward.
        tail = (h_mid - 0.88 * dh * half_h, v_mid - 0.88 * dv * half_v)
        head = (h_mid - 0.60 * dh * half_h, v_mid - 0.60 * dv * half_v)
        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, alpha=0.85),
            zorder=6,
        )


def _draw_beam_footprint_2d(
    ax,
    beams: Iterable[LaserBeam],
    plane: FramePlane,
    extent_mm: tuple[float, float, float, float],
    *,
    slice_coord_mm: float = 0.0,
    grid_n: int = 260,
    color: str = "#ff5050",
    max_alpha: float = 0.4,
) -> None:
    """Shade the summed beam saturation on `plane` behind the cloud.

    Faithful for any beam geometry (Gaussian, top-hat, pencil, grating
    prism): it evaluates each beam's `saturation_at` on the slice and
    paints a translucent field whose opacity grows with how strongly the
    point is illuminated, so overlapping beams (the trapping volume) read
    brightest. Unlike `_draw_beam_arrows_2d` it makes no assumption that
    the beams pass through the origin, so it draws the real footprints —
    the through-hole pencil, the grating prisms, the top-hat edge — rather
    than a single edge arrow per in-plane beam.
    """
    from matplotlib.colors import to_rgb

    horiz_name, vert_name, depth_name = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]
    d_axis = _AXIS_INDEX[depth_name]

    hh = np.linspace(extent_mm[0], extent_mm[1], grid_n)
    vv = np.linspace(extent_mm[2], extent_mm[3], grid_n)
    H, V = np.meshgrid(hh, vv)
    points = np.zeros(H.shape + (3,), dtype=float)
    points[..., h_axis] = H * 1.0e-3
    points[..., v_axis] = V * 1.0e-3
    points[..., d_axis] = slice_coord_mm * 1.0e-3

    total = np.zeros(H.shape, dtype=float)
    for beam in beams:
        total += np.asarray(beam.saturation_at(points), dtype=float)
    peak = float(np.max(total))
    if peak <= 0.0:
        return

    rgb = to_rgb(color)
    # sqrt keeps a lone top-hat/pencil footprint visible while regions
    # where several beams overlap saturate toward max_alpha.
    alpha = max_alpha * np.sqrt(np.clip(total / peak, 0.0, 1.0))
    rgba = np.empty(H.shape + (4,), dtype=float)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        extent=(extent_mm[0], extent_mm[1], extent_mm[2], extent_mm[3]),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )


def draw_cloud_frame(
    t: float,
    result: SimulationResult,
    *,
    mass_kg: float = RB87_MASS_KG,
    plane: FramePlane = "xz",
    beams: Iterable[LaserBeam] | None = None,
    beam_overlay: bool = False,
    extent_mm: tuple[float, float, float, float] | None = None,
    speed_vmax_m_per_s: float | None = None,
    temperatures_uK: NDArray[np.float64] | None = None,
    doppler_limit_uK: float | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Render one snapshot of a trap-free (scattering) run at time `t`.

    Left panel: atom positions on `plane` in mm, colored by speed on a fixed
    sequential scale (lost atoms in gray). If `beams` is given, their
    geometry is drawn behind the cloud: as a translucent saturation
    footprint when `beam_overlay=True` (faithful for any geometry —
    grating prisms, through-hole pencils), or otherwise as one inward edge
    arrow per in-plane beam (a clean sketch for beams through the origin,
    e.g. a six-beam MOT). Right panel: the kinetic-temperature
    time series with a cursor at `t`, plus an optional Doppler-limit line.

    `extent_mm`, `speed_vmax_m_per_s`, and `temperatures_uK` default to
    values computed from the stored trajectories; pass explicit values (as
    `render_cloud_animation` does) to keep axes, colors, and the cooling
    curve fixed across frames. `t` snaps to the nearest stored snapshot.
    Returns (figure, [cloud_axis, temperature_axis]).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install with `pip install .[viz]`."
        ) from exc

    positions_m, velocities, lost = _interpolate_trajectory_state(result, t)
    times = np.asarray(result.trajectory_times_s, dtype=float)
    end_index = int(np.argmin(np.abs(times - float(t))))

    if temperatures_uK is None:
        temperatures_uK = kinetic_temperature_time_series_uK(result, mass_kg=mass_kg)
    if extent_mm is None:
        extent_mm = _cloud_extent_mm(result, plane)
    if speed_vmax_m_per_s is None:
        initial_speeds = np.linalg.norm(
            np.asarray(result.trajectory_velocities_m_per_s[0], dtype=float), axis=-1
        )
        speed_vmax_m_per_s = max(float(np.percentile(initial_speeds, 95.0)), 1.0e-9)

    horiz_name, vert_name, _ = _PLANE_AXES[plane]
    h_axis = _AXIS_INDEX[horiz_name]
    v_axis = _AXIS_INDEX[vert_name]

    figsize = figsize or (11.5, 4.8)
    figure, (cloud_axis, temperature_axis) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
        gridspec_kw={"width_ratios": (1.2, 1.0)},
    )

    positions_mm = positions_m * 1.0e3
    speeds = np.linalg.norm(velocities, axis=-1)
    norm = colors.Normalize(vmin=0.0, vmax=float(speed_vmax_m_per_s))
    bound = ~lost
    if np.any(bound):
        cloud_axis.scatter(
            positions_mm[bound, h_axis], positions_mm[bound, v_axis],
            c=speeds[bound], cmap=cmap, norm=norm,
            s=14, edgecolors="white", linewidths=0.3, alpha=0.9, zorder=5,
        )
    if np.any(lost):
        cloud_axis.scatter(
            positions_mm[lost, h_axis], positions_mm[lost, v_axis],
            s=9, c="#888888", alpha=0.35, linewidths=0, zorder=4,
        )
    if beams is not None:
        if beam_overlay:
            _draw_beam_footprint_2d(cloud_axis, beams, plane, extent_mm)
        else:
            _draw_beam_arrows_2d(cloud_axis, beams, plane, extent_mm)
    figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=cloud_axis, label="atom speed (m/s)", fraction=0.046, pad=0.03,
    )
    cloud_axis.set_xlim(extent_mm[0], extent_mm[1])
    cloud_axis.set_ylim(extent_mm[2], extent_mm[3])
    cloud_axis.set_aspect("equal")
    cloud_axis.set_xlabel(f"{horiz_name} (mm)")
    cloud_axis.set_ylabel(f"{vert_name} (mm)")
    cloud_axis.set_title("Atom cloud")

    times_ms = times * 1.0e3
    temperature_axis.plot(times_ms, temperatures_uK, color="tab:blue")
    if doppler_limit_uK is not None:
        temperature_axis.axhline(
            doppler_limit_uK, color="#888888", linestyle="--", linewidth=1.0,
        )
        temperature_axis.text(
            times_ms[-1], doppler_limit_uK, " Doppler limit",
            color="#888888", fontsize="small", ha="right", va="bottom",
        )
    temperature_axis.axvline(
        times_ms[end_index], color="#ff5050", linewidth=1.0, alpha=0.8,
    )
    temperature_axis.plot(
        times_ms[end_index], temperatures_uK[end_index],
        marker="o", color="#ff5050", markersize=5,
    )
    temperature_axis.set_yscale("log")
    temperature_axis.set_xlabel("time (ms)")
    temperature_axis.set_ylabel("kinetic temperature (uK)")
    temperature_axis.set_title("Cooling")

    trapped_fraction = float(np.mean(bound))
    title = title or (
        f"t = {t * 1.0e3:6.2f} ms    "
        f"T = {temperatures_uK[end_index]:7.1f} uK    "
        f"trapped {trapped_fraction:.0%}"
    )
    figure.suptitle(title)
    return figure, [cloud_axis, temperature_axis]


def render_cloud_animation(
    result: SimulationResult,
    output_path: str | Path,
    *,
    mass_kg: float = RB87_MASS_KG,
    plane: FramePlane = "xz",
    beams: Iterable[LaserBeam] | None = None,
    beam_overlay: bool = False,
    n_frames: int = 80,
    fps: int = 20,
    dpi: int = 90,
    extent_mm: tuple[float, float, float, float] | None = None,
    speed_vmax_m_per_s: float | None = None,
    doppler_limit_uK: float | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    show_progress: bool = True,
) -> Path:
    """Render a GIF of a scattering run by stitching `draw_cloud_frame`s.

    The simulation must have been run with `store_trajectories=True` (and
    stored velocities, which the default settings keep). Axis extent, the
    speed color scale, and the cooling curve are computed once so they stay
    fixed across all frames. Output extension `.gif` (default), `.webp`, or
    `.png`/`.apng` selects the encoder, as in `render_animation`.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Animation requires matplotlib and Pillow; install with `pip install .[viz]`."
        ) from exc

    if result.trajectory_times_s is None or result.trajectory_velocities_m_per_s is None:
        raise ValueError(
            "Simulation was run without stored trajectories; "
            "set SimulationConfig.store_trajectories=True."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    temperatures_uK = kinetic_temperature_time_series_uK(result, mass_kg=mass_kg)
    if extent_mm is None:
        extent_mm = _cloud_extent_mm(result, plane)
    if speed_vmax_m_per_s is None:
        initial_speeds = np.linalg.norm(
            np.asarray(result.trajectory_velocities_m_per_s[0], dtype=float), axis=-1
        )
        speed_vmax_m_per_s = max(float(np.percentile(initial_speeds, 95.0)), 1.0e-9)

    duration = float(result.trajectory_times_s[-1])
    frame_times = np.linspace(0.0, duration, max(2, int(n_frames)))

    progress = _progress_iter if show_progress else lambda x, **_: x

    images: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="atom_mc_frames_") as tmpdir:
        for k, t in enumerate(
            progress(frame_times, total=len(frame_times), desc="render frames")
        ):
            figure, _ = draw_cloud_frame(
                float(t), result, mass_kg=mass_kg, plane=plane, beams=beams,
                beam_overlay=beam_overlay,
                extent_mm=extent_mm, speed_vmax_m_per_s=speed_vmax_m_per_s,
                temperatures_uK=temperatures_uK,
                doppler_limit_uK=doppler_limit_uK, cmap=cmap, figsize=figsize,
            )
            frame_path = os.path.join(tmpdir, f"frame_{k:05d}.png")
            figure.savefig(frame_path, dpi=dpi, facecolor="white")
            plt.close(figure)
            images.append(Image.open(frame_path).convert("RGB").copy())

        _save_animation(images, out, fps=fps)

    return out


def _save_animation(images: list, output_path: Path, *, fps: int) -> None:
    """Stitch frames into a GIF/WEBP/APNG by extension."""
    from PIL import Image

    if not images:
        raise ValueError("no frames to save")
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))

    ext = output_path.suffix.lower()
    if ext == ".gif":
        # Quantize against a shared 128-color palette taken from the middle
        # frame so the GIF encoder can reuse colors across frames.
        ref = images[len(images) // 2].quantize(
            colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE,
        )
        quantized = [
            img.quantize(palette=ref, dither=Image.Dither.NONE) for img in images
        ]
        quantized[0].save(
            output_path, save_all=True, append_images=quantized[1:],
            duration=duration_ms, loop=0, optimize=True,
        )
    elif ext == ".webp":
        rgba = [img.convert("RGBA") for img in images]
        rgba[0].save(
            output_path, format="WebP", save_all=True, append_images=rgba[1:],
            duration=duration_ms, loop=0, lossless=True, method=6,
        )
    elif ext in (".apng", ".png"):
        rgba = [img.convert("RGBA") for img in images]
        rgba[0].save(
            output_path, format="PNG", save_all=True, append_images=rgba[1:],
            duration=duration_ms, loop=0,
        )
    else:
        raise ValueError(
            f"Cannot infer animation format from extension {ext!r}; "
            "use .gif, .webp, or .apng."
        )


def _progress_iter(iterable, *, total, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="frame")
    except ImportError:
        return iterable
