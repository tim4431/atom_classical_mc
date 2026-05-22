"""Spectator-qubit fly-by drag-out and heating study with a gridded RIPA trap.

Mirrors example/spectator_qubits.py but replaces the analytical Gaussian
AOD with a `GriddedTrap` loaded from the .npz produced by
`generate_ripa_gridded_trap.py`. Same static SLM, same fly-by geometry,
same (depth x distance) sweep, and the same speed-comb averaging used by
`example/spectator_qubits.py` to smear out sharp fly-by resonances.

Run from the repository root, after `generate_ripa_gridded_trap.py` has
produced `ripa_gridded_trap.npz`:

    python3 example/spectator/spectator_qubits_ripa.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.analysis import bound_to_trap  # noqa: E402
from src.constants import RB87_MASS_KG  # noqa: E402
from src.ramp import RampSequence  # noqa: E402
from src.simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from src.trap import GaussianTrap, GriddedTrap, total_potential  # noqa: E402
from src.units import joule_to_microkelvin, microkelvin_to_joule, ms, um  # noqa: E402

RENDER_DIR = os.path.join(HERE, "render_ripa")
os.makedirs(RENDER_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Static SLM (identical to example/spectator_qubits.py).
# ---------------------------------------------------------------------------
SLM_WAIST_RADIAL_UM = 1.2
SLM_WAIST_AXIAL_UM = 6.0
SLM_DEPTH_UK = 500.0

# Effective waist of the RIPA focal spot for plot/print labelling only;
# `GriddedTrap` is shape-defined by the tabulated potential, not by a waist.
RIPA_WAIST_UM = 1.18

# The npz is produced by `example/spectator/ripa/generate_ripa_gridded_trap.py`
# and stays next to that generator — we just consume it from here.
RIPA_NPZ_PATH = os.path.join(HERE, "ripa", "ripa_gridded_trap.npz")

INITIAL_TEMPERATURE_UK = 8.0
ENSEMBLE_SIZE = 100
TIMESTEP_S = float(ms(0.0002))
DURATION_S = float(ms(0.5))
LOSS_RADIUS_UM = 40.0
RANDOM_SEED = 42
OUTPUT_DPI = 600
SWEEP_WORKERS = max(1, int(os.environ.get("RIPA_SWEEP_WORKERS", "1")))

FLYBY_HALF_LENGTH_UM = 8.0
FLYBY_RAMP_SAMPLES = 81

# Match example/spectator_qubits.py: path length stays fixed while duration is
# scaled by 1 / speed_factor, then statistics are averaged across this comb.
SPEED_FACTORS = np.geomspace(0.25, 4.0, 100)

# Use the same normalized distance samples as example/spectator_qubits.py.
# Since the RIPA/RIPA nominal waist differs from AOD, the physical offsets
# are scaled so the plotted d / w points match the AOD figure.
DISTANCES_D_OVER_W = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
DISTANCES_UM = tuple(float(d * RIPA_WAIST_UM) for d in DISTANCES_D_OVER_W)
DEPTHS_UK = (50.0, 100.0, 250.0, 500.0, 1000.0)

# Representative case for the fly-by animation. Chosen so the GIF shows a
# mix of "stays in SLM" and "dragged out by RIPA" outcomes.
ANIM_DEPTH_UK = 500.0
ANIM_DISTANCE_UM = 1.5
ANIM_ENSEMBLE_SIZE = 60
ANIM_TRAJECTORY_STRIDE = 5
ANIM_FRAMES = 80
ANIM_FPS = 20
ANIM_GRID_N = 200
GENERATE_ANIMATION = False


# ---------------------------------------------------------------------------
# RIPA grid loading + per-depth rescaling.
# ---------------------------------------------------------------------------
def load_ripa_grid(npz_path: str = RIPA_NPZ_PATH) -> dict:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"RIPA grid not found at {npz_path}. "
            f"Run `python3 example/spectator/ripa/generate_ripa_gridded_trap.py` first."
        )
    with np.load(npz_path) as data:
        x_axis = np.asarray(data["x_axis"], dtype=float)
        y_axis = np.asarray(data["y_axis"], dtype=float)
        z_axis = np.asarray(data["z_axis"], dtype=float)
        intensity = np.asarray(data["intensity"], dtype=float)
        peak_intensity = float(data["peak_intensity"])
        stored_depth_uK = float(data["depth_uK"])
        params_name = str(data["params_name"])
    intensity_normalized = intensity / peak_intensity
    origin = np.array([x_axis[0], y_axis[0], z_axis[0]], dtype=float)
    spacing = np.array(
        [x_axis[1] - x_axis[0], y_axis[1] - y_axis[0], z_axis[1] - z_axis[0]],
        dtype=float,
    )
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "intensity_normalized": intensity_normalized,
        "origin_local_m": origin,
        "spacing_m": spacing,
        "stored_depth_uK": stored_depth_uK,
        "params_name": params_name,
        "shape": intensity_normalized.shape,
    }


def build_ripa_trap(
    grid_data: dict,
    depth_uK: float,
    ramp: RampSequence,
    *,
    potential_j: np.ndarray | None = None,
) -> GriddedTrap:
    """Build a `GriddedTrap` with peak depth `depth_uK` and the given ramp.

    Re-uses the normalized intensity array; multiplies once by the target
    depth in joules to make the per-depth potential grid.
    """

    if potential_j is None:
        depth_j = float(microkelvin_to_joule(depth_uK))
        potential_j = -depth_j * grid_data["intensity_normalized"]
    return GriddedTrap(
        grid_potential_j=potential_j,
        origin_local_m=grid_data["origin_local_m"],
        spacing_m=grid_data["spacing_m"],
        center_m=np.zeros(3, dtype=float),
        ramp=ramp,
        interpolation="tricubic",
        name=f"ripa_{grid_data['params_name']}",
    )


# ---------------------------------------------------------------------------
# SLM, ramp, config, runner.
# ---------------------------------------------------------------------------
def build_static_slm() -> GaussianTrap:
    return GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(SLM_WAIST_RADIAL_UM)),
        waist_axial_m=float(um(SLM_WAIST_AXIAL_UM)),
        depth_uK=SLM_DEPTH_UK,
        name="spectator SLM",
    )


def build_flyby_ramp(
    transverse_distance_um: float,
    half_length_um: float = FLYBY_HALF_LENGTH_UM,
    duration_s: float = DURATION_S,
    samples: int = FLYBY_RAMP_SAMPLES,
) -> RampSequence:
    """Constant-intensity moving trap translating linearly along x.

    Mirrors the AOD fly-by ramp in example/spectator_qubits.py:
    centre goes from (-half_length, d, 0) to (+half_length, d, 0).
    The `depths_uK` column is unused by `GriddedTrap` (which bakes the
    depth into the stored grid); it is set to a constant for API hygiene.
    """

    times_s = np.linspace(0.0, duration_s, samples)
    u = np.linspace(0.0, 1.0, samples)
    centers_um = np.zeros((samples, 3))
    centers_um[:, 0] = -half_length_um + 2.0 * half_length_um * u
    centers_um[:, 1] = transverse_distance_um
    depths_uK = np.ones(samples)  # ignored by GriddedTrap
    return RampSequence(
        times_s=times_s, centers_m=um(centers_um), depths_uK=depths_uK
    )


def build_config(duration_s: float = DURATION_S) -> SimulationConfig:
    return SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=duration_s,
        ensemble_size=ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
        loss_radius_m=float(um(LOSS_RADIUS_UM)),
        initial_center_m=um([0.0, 0.0, 0.0]),
    )


def run_flyby(
    grid_data: dict, depth_uK: float, transverse_distance_um: float,
    *,
    cached_slm: GaussianTrap | None = None,
    duration_s: float = DURATION_S,
    potential_j: np.ndarray | None = None,
) -> tuple[SimulationResult, RampSequence, GaussianTrap, GriddedTrap]:
    slm = cached_slm if cached_slm is not None else build_static_slm()
    ramp = build_flyby_ramp(transverse_distance_um, duration_s=duration_s)
    ripa = build_ripa_trap(grid_data, depth_uK, ramp, potential_j=potential_j)
    config = build_config(duration_s=duration_s)
    result = run_simulation([slm, ripa], config)
    return result, ramp, slm, ripa


# ---------------------------------------------------------------------------
# Capture / drag-out probabilities.
# ---------------------------------------------------------------------------
def drag_out_probability(result: SimulationResult, slm: GaussianTrap) -> float:
    survivors = ~result.lost
    still_in_slm = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        slm,
        mass_kg=RB87_MASS_KG,
    )
    return 1.0 - float(np.mean(still_in_slm))


def ripa_capture_probability(
    result: SimulationResult, ripa: GriddedTrap, duration_s: float
) -> float:
    """Bound to the (moving) RIPA trap at the end of the fly-by."""

    survivors = ~result.lost
    captured = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        ripa,
        mass_kg=RB87_MASS_KG,
        time_s=duration_s,
    )
    return float(np.mean(captured))


# ---------------------------------------------------------------------------
# Sweep + plots.
# ---------------------------------------------------------------------------
_WORKER_GRID_DATA: dict | None = None
_WORKER_SPEED_FACTORS: np.ndarray | None = None
_WORKER_SLM: GaussianTrap | None = None


def _format_cell_summary(
    depth: float,
    distance: float,
    drag_values: np.ndarray,
    loss_values: np.ndarray,
    capture_values: np.ndarray,
    heating_values: np.ndarray,
) -> str:
    return (
        f"RIPA={depth:>7.1f} uK  d={distance:.2f} um  "
        f"U_RIPA/U_SLM={depth / SLM_DEPTH_UK:>6.2f}  "
        f"d/w={distance / RIPA_WAIST_UM:>4.2f}  "
        f"<P(drag)>={drag_values.mean():.3f}  "
        f"<P(lost)>={loss_values.mean():.3f}  "
        f"<P(RIPA)>={capture_values.mean():.3f}  "
        f"<heating>={np.nanmean(heating_values):.2f} uK"
    )


def _init_sweep_worker() -> None:
    global _WORKER_SLM
    _WORKER_SLM = build_static_slm()


def _run_sweep_cell(
    task: tuple[int, int, float, float],
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER_GRID_DATA is None or _WORKER_SPEED_FACTORS is None:
        raise RuntimeError("Sweep worker was not initialized with grid data.")
    slm = _WORKER_SLM if _WORKER_SLM is not None else build_static_slm()
    i, j, depth, distance = task
    depth_j = float(microkelvin_to_joule(depth))
    potential_j = -depth_j * _WORKER_GRID_DATA["intensity_normalized"]
    n_speeds = len(_WORKER_SPEED_FACTORS)
    drag_values = np.zeros(n_speeds)
    loss_values = np.zeros(n_speeds)
    capture_values = np.zeros(n_speeds)
    heating_values = np.zeros(n_speeds)

    for k, sf in enumerate(_WORKER_SPEED_FACTORS):
        duration_s = DURATION_S / sf
        result, ramp, _, ripa = run_flyby(
            _WORKER_GRID_DATA,
            depth,
            distance,
            cached_slm=slm,
            duration_s=duration_s,
            potential_j=potential_j,
        )
        drag_values[k] = drag_out_probability(result, slm)
        loss_values[k] = result.loss_fraction
        capture_values[k] = ripa_capture_probability(result, ripa, duration_s)
        heating_values[k] = result.temperature_gain_uK_at(survivors_only=False)

    return i, j, drag_values, loss_values, capture_values, heating_values


def sweep_grid(
    grid_data: dict,
    depths_uK: tuple[float, ...] = DEPTHS_UK,
    distances_um: tuple[float, ...] = DISTANCES_UM,
    speed_factors: np.ndarray = SPEED_FACTORS,
    workers: int = SWEEP_WORKERS,
) -> dict:
    """Sweep `(RIPA depth, distance)` and average over fly-by speeds."""

    n_speeds = len(speed_factors)
    shape = (len(depths_uK), len(distances_um), n_speeds)
    drag_per_speed = np.zeros(shape)
    loss_per_speed = np.zeros(shape)
    capture_per_speed = np.zeros(shape)
    heating_per_speed = np.zeros(shape)
    workers = max(1, int(workers))

    if workers > 1:
        import multiprocessing as mp

        global _WORKER_GRID_DATA, _WORKER_SPEED_FACTORS
        _WORKER_GRID_DATA = grid_data
        _WORKER_SPEED_FACTORS = np.asarray(speed_factors, dtype=float)
        tasks = [
            (i, j, float(depth), float(distance))
            for i, depth in enumerate(depths_uK)
            for j, distance in enumerate(distances_um)
        ]
        print(f"Running sweep with {workers} worker processes")
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, initializer=_init_sweep_worker) as pool:
            for i, j, drag, loss, capture, heating in pool.imap_unordered(
                _run_sweep_cell, tasks
            ):
                drag_per_speed[i, j] = drag
                loss_per_speed[i, j] = loss
                capture_per_speed[i, j] = capture
                heating_per_speed[i, j] = heating
                print(
                    _format_cell_summary(
                        float(depths_uK[i]), float(distances_um[j]),
                        drag, loss, capture, heating,
                    )
                )
        _WORKER_GRID_DATA = None
        _WORKER_SPEED_FACTORS = None
        return {
            "depths_uK": np.asarray(depths_uK, dtype=float),
            "distances_um": np.asarray(distances_um, dtype=float),
            "speed_factors": np.asarray(speed_factors, dtype=float),
            "drag_out_per_speed": drag_per_speed,
            "loss_per_speed": loss_per_speed,
            "ripa_capture_per_speed": capture_per_speed,
            "heating_uK_per_speed": heating_per_speed,
        }

    slm = build_static_slm()
    for i, depth in enumerate(depths_uK):
        depth_j = float(microkelvin_to_joule(depth))
        potential_j = -depth_j * grid_data["intensity_normalized"]
        for j, d in enumerate(distances_um):
            for k, sf in enumerate(speed_factors):
                duration_s = DURATION_S / sf
                result, ramp, _, ripa = run_flyby(
                    grid_data, depth, d,
                    cached_slm=slm,
                    duration_s=duration_s,
                    potential_j=potential_j,
                )
                drag_per_speed[i, j, k] = drag_out_probability(result, slm)
                loss_per_speed[i, j, k] = result.loss_fraction
                capture_per_speed[i, j, k] = ripa_capture_probability(
                    result, ripa, duration_s
                )
                heating_per_speed[i, j, k] = result.temperature_gain_uK_at(
                    survivors_only=False
                )

            print(
                _format_cell_summary(
                    float(depth), float(d),
                    drag_per_speed[i, j],
                    loss_per_speed[i, j],
                    capture_per_speed[i, j],
                    heating_per_speed[i, j],
                )
            )

    return {
        "depths_uK": np.asarray(depths_uK, dtype=float),
        "distances_um": np.asarray(distances_um, dtype=float),
        "speed_factors": np.asarray(speed_factors, dtype=float),
        "drag_out_per_speed": drag_per_speed,
        "loss_per_speed": loss_per_speed,
        "ripa_capture_per_speed": capture_per_speed,
        "heating_uK_per_speed": heating_per_speed,
    }


def cell_mean_std(per_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean and sample std along the speed axis. NaN-safe."""

    mean = np.nanmean(per_speed, axis=-1)
    if per_speed.shape[-1] > 1:
        std = np.nanstd(per_speed, axis=-1, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def _plot_value_lines(
    sweep_results: dict, *,
    per_speed_key: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
):
    import matplotlib.pyplot as plt

    distances = sweep_results["distances_um"]
    depths = sweep_results["depths_uK"]
    distances_in_waists = distances / RIPA_WAIST_UM
    mean, sigma = cell_mean_std(sweep_results[per_speed_key])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, depth in enumerate(depths):
        color = cmap(i / max(len(depths) - 1, 1))
        ax.errorbar(
            distances_in_waists,
            mean[i],
            yerr=sigma[i],
            marker="o",
            color=color,
            capsize=3.0,
            elinewidth=1.0,
            linewidth=1.7,
            label=f"{depth:.0f} uK",
        )
    ax.set_xlabel(rf"transverse distance  $d / w$  ($w = {RIPA_WAIST_UM:.2f}$ µm)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="RIPA trap depth", fontsize="small")
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def plot_drag_lines(sweep_results: dict):
    return _plot_value_lines(
        sweep_results,
        per_speed_key="drag_out_per_speed",
        ylabel="P(spectator dragged out of SLM)",
        ylim=(-0.02, 1.02),
    )


def plot_heating_lines(sweep_results: dict):
    return _plot_value_lines(
        sweep_results,
        per_speed_key="heating_uK_per_speed",
        ylabel="Average heating (uK)",
    )


def plot_capture_lines(sweep_results: dict):
    return _plot_value_lines(
        sweep_results,
        per_speed_key="ripa_capture_per_speed",
        ylabel="P(captured by RIPA at end of fly-by)",
        ylim=(-0.02, 1.02),
    )


# ---------------------------------------------------------------------------
# Fly-by animation (GIF).
# ---------------------------------------------------------------------------
def run_flyby_with_trajectories(
    grid_data: dict, depth_uK: float, transverse_distance_um: float,
) -> tuple[SimulationResult, RampSequence, GaussianTrap, GriddedTrap]:
    """One fly-by with stored trajectories, sized for the animation."""

    slm = build_static_slm()
    ramp = build_flyby_ramp(transverse_distance_um)
    ripa = build_ripa_trap(grid_data, depth_uK, ramp)
    config = SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=DURATION_S,
        ensemble_size=ANIM_ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
        loss_radius_m=float(um(LOSS_RADIUS_UM)),
        initial_center_m=um([0.0, 0.0, 0.0]),
        store_trajectories=True,
        trajectory_stride=ANIM_TRAJECTORY_STRIDE,
    )
    result = run_simulation([slm, ripa], config)
    return result, ramp, slm, ripa


def render_flyby_gif(
    result: SimulationResult,
    slm: GaussianTrap,
    ripa: GriddedTrap,
    ramp: RampSequence,
    output_path: str,
    *,
    n_frames: int = ANIM_FRAMES,
    fps: int = ANIM_FPS,
    grid_n: int = ANIM_GRID_N,
) -> None:
    """Render a single-panel (xy at z=0) GIF of the fly-by.

    Works for any `TrapConfig` because it queries `total_potential` directly
    at each frame's time (no Gaussian-envelope side-view), unlike
    `src.visualization.render_animation` which is Gaussian-specific.
    """

    import matplotlib.pyplot as plt
    from PIL import Image

    times = np.asarray(result.trajectory_times_s, dtype=float)
    positions_all = np.asarray(result.trajectory_positions_m, dtype=float)
    lost_all = np.asarray(result.trajectory_lost, dtype=bool)
    n_stored = len(times)
    if n_stored < 2:
        raise ValueError(
            "Need at least 2 stored snapshots; raise --animate-ensemble or "
            "lower trajectory_stride."
        )
    frame_indices = np.linspace(0, n_stored - 1, n_frames).astype(int)

    # xy view: covers the full ramp travel in x; a few um in y around the SLM
    # and the RIPA's standoff distance.
    x_min_um, x_max_um = -FLYBY_HALF_LENGTH_UM - 1.0, FLYBY_HALF_LENGTH_UM + 1.0
    y_half_um = max(2.0 * ANIM_DISTANCE_UM, 2.5)
    y_min_um, y_max_um = -y_half_um, y_half_um

    xs = np.linspace(x_min_um * 1e-6, x_max_um * 1e-6, grid_n)
    ys = np.linspace(y_min_um * 1e-6, y_max_um * 1e-6, max(grid_n // 2, 60))

    X_xy, Y_xy = np.meshgrid(xs, ys, indexing="xy")  # (Ny, Nx)
    pts_xy = np.stack([X_xy, Y_xy, np.zeros_like(X_xy)], axis=-1)

    # Fix the colormap range so it doesn't dance frame-to-frame. The deepest
    # well present anywhere is max(SLM, RIPA) depth.
    ripa_depth_uK = float(
        -joule_to_microkelvin(float(np.min(ripa.grid_potential_j)))
    )
    vmin_uK = -max(SLM_DEPTH_UK, ripa_depth_uK)

    images: list = []
    print(f"Rendering {n_frames} frames for GIF ({output_path})...")
    for k, idx in enumerate(frame_indices):
        t = float(times[idx])
        positions = positions_all[idx]
        lost = lost_all[idx]
        survived = ~lost

        u_xy_uK = joule_to_microkelvin(total_potential([slm, ripa], pts_xy, time_s=t))
        ripa_center_um = ripa.center_at(t) * 1e6

        fig, ax_xy = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax_xy.imshow(
            u_xy_uK,
            origin="lower",
            extent=[x_min_um, x_max_um, y_min_um, y_max_um],
            vmin=vmin_uK, vmax=0.0,
            cmap="magma", aspect="equal",
        )
        if np.any(survived):
            ax_xy.scatter(
                positions[survived, 0] * 1e6, positions[survived, 1] * 1e6,
                s=14, c="white", edgecolors="black", linewidths=0.4,
                label="alive",
            )
        if np.any(lost):
            ax_xy.scatter(
                positions[lost, 0] * 1e6, positions[lost, 1] * 1e6,
                s=14, c="red", edgecolors="black", linewidths=0.4,
                label="lost",
            )
        ax_xy.scatter(
            [ripa_center_um[0]], [ripa_center_um[1]],
            marker="x", s=60, c="cyan", linewidths=2.0,
            label="RIPA center",
        )
        ax_xy.set_xlabel(r"$x$ (µm)")
        ax_xy.set_ylabel(r"$y$ (µm)")
        ax_xy.legend(loc="upper right", fontsize="small", framealpha=0.6)

        n_surv = int(np.sum(survived))
        n_lost = int(np.sum(lost))
        ax_xy.set_title(
            f"t = {t * 1e3:7.4f} ms   "
            f"RIPA at (x, y) = ({ripa_center_um[0]:+.2f}, "
            f"{ripa_center_um[1]:+.2f}) µm   "
            f"alive {n_surv}/{len(lost)},  lost {n_lost}"
        )

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1,
        ).convert("RGB")
        images.append(img.copy())
        plt.close(fig)
        if (k + 1) % 10 == 0 or k == n_frames - 1:
            print(f"  frame {k + 1}/{n_frames}")

    duration_ms = max(1, int(round(1000.0 / fps)))
    ref = images[len(images) // 2].quantize(colors=128)
    quantized = [img.quantize(palette=ref) for img in images]
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"saved GIF: {output_path}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def print_setup_banner(grid_data: dict) -> None:
    speed_um_per_ms = 2.0 * FLYBY_HALF_LENGTH_UM / (DURATION_S * 1.0e3)
    shape = grid_data["shape"]
    spacing_um = grid_data["spacing_m"] * 1e6
    print("Spectator-qubit fly-by with RIPA gridded trap")
    print(f"  SLM:  depth={SLM_DEPTH_UK:.1f} uK, "
          f"waists r={SLM_WAIST_RADIAL_UM} um / z={SLM_WAIST_AXIAL_UM} um")
    print(f"  RIPA: tabulated potential from {RIPA_NPZ_PATH}")
    print(f"        grid shape {shape}, "
          f"spacing ({spacing_um[0]:.3f}, {spacing_um[1]:.3f}, {spacing_um[2]:.3f}) um, "
          f"interp=tricubic")
    print(f"        nominal focal waist {RIPA_WAIST_UM} um")
    print(f"  fly-by: x = -{FLYBY_HALF_LENGTH_UM:.1f} -> +{FLYBY_HALF_LENGTH_UM:.1f} um, "
          f"y = d, z = 0; speed = {speed_um_per_ms:.1f} um/ms")
    print(f"  initial T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
          f"ensemble = {ENSEMBLE_SIZE}, duration = {DURATION_S * 1.0e3:.2f} ms")
    print(f"  sweep: depths {list(DEPTHS_UK)} uK x d/w samples "
          f"{list(DISTANCES_D_OVER_W)} "
          f"(distances {[round(d, 3) for d in DISTANCES_UM]} um) "
          f"x {len(SPEED_FACTORS)} speeds in [{SPEED_FACTORS[0]:.2f}, "
          f"{SPEED_FACTORS[-1]:.2f}] x nominal")
    if SWEEP_WORKERS > 1:
        print(f"  sweep workers = {SWEEP_WORKERS}")
    print()


_CACHE_FORMAT_VERSION = 2
_REQUIRED_CACHE_KEYS = frozenset((
    "depths_uK", "distances_um", "speed_factors",
    "drag_out_per_speed", "loss_per_speed",
    "ripa_capture_per_speed", "heating_uK_per_speed",
))


def _load_cache_or_none(cache_path: str) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path) as data:
        keys = set(data.files)
        if not _REQUIRED_CACHE_KEYS.issubset(keys):
            print(f"Cache at {cache_path} is stale (missing keys); will re-run")
            return None
        version = int(data["_format_version"]) if "_format_version" in keys else 0
        if version != _CACHE_FORMAT_VERSION:
            print(
                f"Cache at {cache_path} is format v{version}, "
                f"expected v{_CACHE_FORMAT_VERSION}; will re-run"
            )
            return None
        expected = {
            "depths_uK": np.asarray(DEPTHS_UK, dtype=float),
            "distances_um": np.asarray(DISTANCES_UM, dtype=float),
            "speed_factors": np.asarray(SPEED_FACTORS, dtype=float),
        }
        for key, expected_values in expected.items():
            values = np.asarray(data[key], dtype=float)
            if values.shape != expected_values.shape or not np.allclose(
                values, expected_values
            ):
                print(f"Cache at {cache_path} is stale ({key} changed); will re-run")
                return None
        return {k: data[k] for k in keys if k != "_format_version"}


def main() -> None:
    grid_data = load_ripa_grid()
    print_setup_banner(grid_data)

    cache_path = os.path.join(RENDER_DIR, "spectator_qubits_ripa_sweep.npz")
    sweep_results = _load_cache_or_none(cache_path)
    if sweep_results is None:
        sweep_results = sweep_grid(grid_data)
        np.savez(
            cache_path, **sweep_results,
            _format_version=np.int32(_CACHE_FORMAT_VERSION),
        )
        print(f"saved cache: {cache_path}")
    else:
        print(f"Loaded cached sweep from {cache_path}  (delete to force re-run)")

    figures = {
        "spectator_qubits_ripa_drag_lines": plot_drag_lines(sweep_results),
        "spectator_qubits_ripa_heating_lines": plot_heating_lines(sweep_results),
        "spectator_qubits_ripa_capture_lines": plot_capture_lines(sweep_results),
    }
    for stem, (fig, _) in figures.items():
        for ext in ("png", "pdf"):
            out_path = os.path.join(RENDER_DIR, f"{stem}.{ext}")
            fig.savefig(out_path, dpi=OUTPUT_DPI)
            print(f"saved: {out_path}")

    # Representative fly-by animation.
    if not GENERATE_ANIMATION:
        return

    print()
    print(
        f"Rendering animation for RIPA={ANIM_DEPTH_UK:.0f} uK, "
        f"d={ANIM_DISTANCE_UM} um ({ANIM_ENSEMBLE_SIZE} atoms)..."
    )
    anim_result, anim_ramp, anim_slm, anim_ripa = run_flyby_with_trajectories(
        grid_data, ANIM_DEPTH_UK, ANIM_DISTANCE_UM
    )
    gif_path = os.path.join(
        RENDER_DIR,
        f"spectator_qubits_ripa_flyby_d{ANIM_DEPTH_UK:.0f}uK_"
        f"y{ANIM_DISTANCE_UM:.2f}um.gif",
    )
    render_flyby_gif(anim_result, anim_slm, anim_ripa, anim_ramp, gif_path)


if __name__ == "__main__":
    main()
