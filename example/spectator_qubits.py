"""Spectator-qubit fly-by drag-out and heating study.

A static SLM trap at the origin holds a spectator Rb87 atom. A
constant-intensity moving AOD tweezer flies past at a fixed transverse
distance d in y, traveling along x. We sweep over (AOD depth, transverse
distance), average over a small set of fly-by speeds (to smear out the
`interaction time = integer * trap period` resonance), and report:

  * P(spectator dragged out of SLM)
  * P(captured by AOD), P(lost to vacuum)
  * Heating, defined as the kinetic-temperature gain of the entire
    ensemble (all atoms, including those captured by the AOD), in uK

Sweep results are cached to `render/spectator_qubits_sweep.npz`; delete
that file to force a re-run.

Run from the repository root:

    python3 example/spectator_qubits.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import bound_to_trap  # noqa: E402
from src.constants import RB87_MASS_KG  # noqa: E402
from src.ramp import RampSequence  # noqa: E402
from src.simulation import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from src.trap import GaussianTrap  # noqa: E402
from src.units import ms, um  # noqa: E402

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

SLM_WAIST_RADIAL_UM = 1.2
SLM_WAIST_AXIAL_UM = 6.0
SLM_DEPTH_UK = 500.0

AOD_WAIST_RADIAL_UM = 1.0
AOD_WAIST_AXIAL_UM = 5.0

INITIAL_TEMPERATURE_UK = 8.0
ENSEMBLE_SIZE = 100
TIMESTEP_S = float(ms(0.0002))
DURATION_S = float(ms(0.5))
LOSS_RADIUS_UM = 40.0
RANDOM_SEED = 42

FLYBY_HALF_LENGTH_UM = 8.0
FLYBY_RAMP_SAMPLES = 81

# Fine speed sweep: each factor is realised by scaling the duration as
# `DURATION_S / speed_factor` (path length stays fixed). Drag-out / loss /
# heating are averaged across this comb to smear out the
# `interaction time = integer * trap period` resonance, which is sharp
# (see `spectator_qubits_speed_scan.py`). 500 log-spaced points over a
# factor-16 range — heavy averaging compensates for ENSEMBLE_SIZE = 100.
SPEED_FACTORS = np.geomspace(0.25, 4.0, 100)

DISTANCES_UM = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
# AOD/SLM depth ratios 0.1x .. 10x of SLM_DEPTH_UK.
DEPTHS_UK = (50.0, 100.0, 250.0, 500.0, 1000.0)

REPRESENTATIVE_DEPTH_UK = 1000.0
REPRESENTATIVE_DISTANCE_UM = 1.0


def build_static_slm() -> GaussianTrap:
    return GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(SLM_WAIST_RADIAL_UM)),
        waist_axial_m=float(um(SLM_WAIST_AXIAL_UM)),
        depth_uK=SLM_DEPTH_UK,
        name="spectator SLM",
    )


def build_aod_base() -> GaussianTrap:
    return GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(AOD_WAIST_RADIAL_UM)),
        waist_axial_m=float(um(AOD_WAIST_AXIAL_UM)),
        depth_uK=0.0,
        name="fly-by AOD",
    )


def build_flyby_ramp(
    depth_uK: float,
    transverse_distance_um: float,
    half_length_um: float = FLYBY_HALF_LENGTH_UM,
    duration_s: float = DURATION_S,
    samples: int = FLYBY_RAMP_SAMPLES,
) -> RampSequence:
    """Constant-intensity AOD translating linearly past the SLM along x.

    AOD center moves from (-half_length_um, d, 0) to (+half_length_um, d, 0)
    over `duration_s` with depth fixed at `depth_uK` for the entire window.
    """

    times_s = np.linspace(0.0, duration_s, samples)
    u = np.linspace(0.0, 1.0, samples)
    centers_um = np.zeros((samples, 3))
    centers_um[:, 0] = -half_length_um + 2.0 * half_length_um * u
    centers_um[:, 1] = transverse_distance_um
    depths_uK = np.full(samples, depth_uK)
    return RampSequence(
        times_s=times_s, centers_m=um(centers_um), depths_uK=depths_uK
    )


def build_config(
    store_trajectories: bool = False,
    ensemble_size: int = ENSEMBLE_SIZE,
    duration_s: float = DURATION_S,
) -> SimulationConfig:
    return SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=duration_s,
        ensemble_size=ensemble_size,
        random_seed=RANDOM_SEED,
        loss_radius_m=float(um(LOSS_RADIUS_UM)),
        initial_center_m=um([0.0, 0.0, 0.0]),
        store_trajectories=store_trajectories,
        trajectory_stride=25,
    )


def run_flyby(
    depth_uK: float,
    transverse_distance_um: float,
    *,
    store_trajectories: bool = False,
    ensemble_size: int = ENSEMBLE_SIZE,
    duration_s: float = DURATION_S,
) -> tuple[SimulationResult, RampSequence, GaussianTrap, GaussianTrap]:
    slm = build_static_slm()
    aod_base = build_aod_base()
    ramp = build_flyby_ramp(depth_uK, transverse_distance_um, duration_s=duration_s)
    config = build_config(
        store_trajectories=store_trajectories,
        ensemble_size=ensemble_size,
        duration_s=duration_s,
    )
    result = run_simulation(slm, aod_base, ramp, config)
    return result, ramp, slm, aod_base


def drag_out_probability(
    result: SimulationResult, slm: GaussianTrap, mass_kg: float
) -> float:
    survivors = ~result.lost
    still_in_slm = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        slm,
        mass_kg=mass_kg,
    )
    return 1.0 - float(np.mean(still_in_slm))


def aod_capture_probability(
    result: SimulationResult,
    aod_base: GaussianTrap,
    ramp: RampSequence,
    duration_s: float,
    mass_kg: float,
) -> float:
    final_center, final_depth = ramp.at(duration_s)
    aod_final = aod_base.with_center_depth(final_center, final_depth)
    survivors = ~result.lost
    captured = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        aod_final,
        mass_kg=mass_kg,
    )
    return float(np.mean(captured))


def sweep_grid(
    depths_uK: tuple[float, ...] = DEPTHS_UK,
    distances_um: tuple[float, ...] = DISTANCES_UM,
    speed_factors: np.ndarray = SPEED_FACTORS,
) -> dict:
    """Sweep `(AOD depth, transverse distance)` and average over fly-by speeds.

    For each cell we run one simulation per `speed_factor` (each realised by
    duration `DURATION_S / speed_factor`, path length fixed). The reported
    `drag_out`, `loss`, and `aod_capture` are means across speeds, which
    smears out the resonance from the fly-by interaction time being a
    rational multiple of the trap period.
    """

    n_speeds = len(speed_factors)
    shape = (len(depths_uK), len(distances_um), n_speeds)
    drag_per_speed = np.zeros(shape)
    loss_per_speed = np.zeros(shape)
    aod_capture_per_speed = np.zeros(shape)
    heating_uK_per_speed = np.zeros(shape)

    for i, depth in enumerate(depths_uK):
        for j, d in enumerate(distances_um):
            for k, sf in enumerate(speed_factors):
                duration_s = DURATION_S / sf
                result, ramp, slm, aod_base = run_flyby(
                    depth, d, duration_s=duration_s
                )
                drag_per_speed[i, j, k] = drag_out_probability(
                    result, slm, RB87_MASS_KG
                )
                loss_per_speed[i, j, k] = result.loss_fraction
                aod_capture_per_speed[i, j, k] = aod_capture_probability(
                    result, aod_base, ramp, duration_s, RB87_MASS_KG
                )
                heating_uK_per_speed[i, j, k] = result.temperature_gain_uK_at(
                    survivors_only=False
                )

            print(
                f"AOD={depth:>7.1f} uK  d={d:.2f} um  "
                f"U_AOD/U_SLM={depth / SLM_DEPTH_UK:>6.2f}  "
                f"d/w_r={d / AOD_WAIST_RADIAL_UM:>4.2f}  "
                f"<P(drag)>={drag_per_speed[i, j].mean():.3f}  "
                f"<P(lost)>={loss_per_speed[i, j].mean():.3f}  "
                f"<P(AOD)>={aod_capture_per_speed[i, j].mean():.3f}  "
                f"<heating>={np.nanmean(heating_uK_per_speed[i, j]):.2f} uK"
            )

    return {
        "depths_uK": np.asarray(depths_uK, dtype=float),
        "distances_um": np.asarray(distances_um, dtype=float),
        "speed_factors": np.asarray(speed_factors, dtype=float),
        "drag_out_per_speed": drag_per_speed,
        "loss_per_speed": loss_per_speed,
        "aod_capture_per_speed": aod_capture_per_speed,
        "heating_uK_per_speed": heating_uK_per_speed,
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
    title: str,
    ylim: tuple[float, float] | None = None,
):
    """Line plot vs distance with shaded sigma band per AOD depth."""

    import matplotlib.pyplot as plt

    distances = sweep_results["distances_um"]
    depths = sweep_results["depths_uK"]
    distances_in_waists = distances / AOD_WAIST_RADIAL_UM
    mean, sigma = cell_mean_std(sweep_results[per_speed_key])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, depth in enumerate(depths):
        color = cmap(i / max(len(depths) - 1, 1))
        ax.plot(
            distances_in_waists, mean[i],
            marker="o", color=color, label=f"{depth:.0f} uK",
        )
        ax.fill_between(
            distances_in_waists,
            mean[i] - sigma[i], mean[i] + sigma[i],
            color=color, alpha=0.18, linewidth=0,
        )
    ax.set_xlabel(rf"transverse distance  $d / w_r$  ($w_r = {AOD_WAIST_RADIAL_UM:.2f}$ um)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="AOD depth", fontsize="small")
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def plot_drag_lines(sweep_results: dict):
    return _plot_value_lines(
        sweep_results,
        per_speed_key="drag_out_per_speed",
        ylabel="P(spectator dragged out of SLM)",
        title="Drag-out probability vs distance (band = ± sample std over speeds)",
        ylim=(-0.02, 1.02),
    )


def plot_heating_lines(sweep_results: dict):
    return _plot_value_lines(
        sweep_results,
        per_speed_key="heating_uK_per_speed",
        ylabel="heating of full ensemble  [uK]",
        title="Heating vs distance (all atoms; band = ± sample std over speeds)",
    )


def print_setup_banner() -> None:
    speed_um_per_ms = 2.0 * FLYBY_HALF_LENGTH_UM / (DURATION_S * 1.0e3)
    print("Spectator-qubit fly-by simulation")
    print(f"  SLM:  depth={SLM_DEPTH_UK:.1f} uK, "
          f"waists r={SLM_WAIST_RADIAL_UM} um / z={SLM_WAIST_AXIAL_UM} um")
    print(f"  AOD:  waists r={AOD_WAIST_RADIAL_UM} um / z={AOD_WAIST_AXIAL_UM} um, "
          f"constant intensity (no ramp)")
    print(f"  AOD path: x = -{FLYBY_HALF_LENGTH_UM:.1f} -> +{FLYBY_HALF_LENGTH_UM:.1f} um, "
          f"y = d, z = 0; nominal speed = {speed_um_per_ms:.1f} um/ms")
    print(f"  initial T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
          f"ensemble = {ENSEMBLE_SIZE}, duration = {DURATION_S * 1.0e3:.2f} ms")
    print(f"  sweep: depths {list(DEPTHS_UK)} uK "
          f"x distances {list(DISTANCES_UM)} um "
          f"x {len(SPEED_FACTORS)} speeds in [{SPEED_FACTORS[0]:.2f}, "
          f"{SPEED_FACTORS[-1]:.2f}] x nominal")
    print()


# Bump when the on-disk semantics of cached arrays change (e.g. when the
# definition of `heating_uK_per_speed` changes), so old caches are auto
# invalidated. Required-key check is for older caches without the version.
_CACHE_FORMAT_VERSION = 3
_REQUIRED_CACHE_KEYS = frozenset((
    "depths_uK", "distances_um", "speed_factors",
    "drag_out_per_speed", "loss_per_speed",
    "aod_capture_per_speed", "heating_uK_per_speed",
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
        return {k: data[k] for k in keys if k != "_format_version"}


def main() -> None:
    print_setup_banner()

    cache_path = os.path.join(RENDER_DIR, "spectator_qubits_sweep.npz")
    sweep_results = _load_cache_or_none(cache_path)
    if sweep_results is None:
        sweep_results = sweep_grid()
        np.savez(
            cache_path, **sweep_results,
            _format_version=np.int32(_CACHE_FORMAT_VERSION),
        )
        print(f"saved cache: {cache_path}")
    else:
        print(f"Loaded cached sweep from {cache_path}  (delete to force re-run)")

    figures = {
        "spectator_qubits_drag_lines.png": plot_drag_lines(sweep_results),
        "spectator_qubits_heating_lines.png": plot_heating_lines(sweep_results),
    }
    for filename, (fig, _) in figures.items():
        out_path = os.path.join(RENDER_DIR, filename)
        fig.savefig(out_path, dpi=180)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
