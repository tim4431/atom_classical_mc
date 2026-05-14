"""Spectator-qubit fly-by drag-out study.

A static SLM trap at the origin holds a spectator Rb87 atom. A
constant-intensity moving AOD tweezer flies past at a fixed transverse
distance d in y, traveling along x. We sweep over (AOD depth, transverse
distance) and report the probability that the spectator is dragged out of
the SLM by the fly-by, the AOD-capture fraction, and the loss fraction.

Run from the repository root:

    python3 example/spectator_qubits.py

Saves a (depth, distance) heatmap, line plots, and one representative
trajectory plot next to this script.
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
ENSEMBLE_SIZE = 2000
TIMESTEP_S = float(ms(0.0002))
DURATION_S = float(ms(0.5))
LOSS_RADIUS_UM = 40.0
RANDOM_SEED = 42

FLYBY_HALF_LENGTH_UM = 8.0
FLYBY_RAMP_SAMPLES = 81

DISTANCES_UM = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
# AOD/SLM depth ratios 0.1x .. 10x of SLM_DEPTH_UK.
DEPTHS_UK = (50.0, 100.0, 250.0, 500.0, 1000.0)

REPRESENTATIVE_DEPTH_UK = 1000.0
REPRESENTATIVE_DISTANCE_UM = 1.0


def build_static_slm(slm_depth_uK: float = SLM_DEPTH_UK) -> GaussianTrap:
    return GaussianTrap(
        center_m=um([0.0, 0.0, 0.0]),
        waist_radial_m=float(um(SLM_WAIST_RADIAL_UM)),
        waist_axial_m=float(um(SLM_WAIST_AXIAL_UM)),
        depth_uK=slm_depth_uK,
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
) -> SimulationConfig:
    return SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=DURATION_S,
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
) -> tuple[SimulationResult, RampSequence, GaussianTrap, GaussianTrap]:
    slm = build_static_slm()
    aod_base = build_aod_base()
    ramp = build_flyby_ramp(depth_uK, transverse_distance_um)
    config = build_config(
        store_trajectories=store_trajectories, ensemble_size=ensemble_size
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
) -> dict:
    drag = np.zeros((len(depths_uK), len(distances_um)), dtype=float)
    loss = np.zeros_like(drag)
    aod_capture = np.zeros_like(drag)
    final_T = np.zeros_like(drag)

    for i, depth in enumerate(depths_uK):
        for j, d in enumerate(distances_um):
            result, ramp, slm, aod_base = run_flyby(depth, d)
            drag[i, j] = drag_out_probability(result, slm, RB87_MASS_KG)
            loss[i, j] = result.loss_fraction
            aod_capture[i, j] = aod_capture_probability(
                result, aod_base, ramp, DURATION_S, RB87_MASS_KG
            )
            final_T[i, j] = result.final_temperature_uK
            print(
                f"AOD={depth:>7.1f} uK  d={d:.2f} um  "
                f"U_AOD/U_SLM={depth / SLM_DEPTH_UK:>6.2f}  "
                f"d/w_r={d / AOD_WAIST_RADIAL_UM:>4.2f}  "
                f"P(drag-out)={drag[i, j]:.3f}  "
                f"P(lost)={loss[i, j]:.3f}  "
                f"P(AOD)={aod_capture[i, j]:.3f}"
            )

    return {
        "depths_uK": np.asarray(depths_uK, dtype=float),
        "distances_um": np.asarray(distances_um, dtype=float),
        "drag_out": drag,
        "loss": loss,
        "aod_capture": aod_capture,
        "final_temperature_uK": final_T,
    }


def plot_sweep_heatmaps(sweep_results: dict):
    import matplotlib.pyplot as plt

    distances = sweep_results["distances_um"]
    depths = sweep_results["depths_uK"]
    grid = sweep_results["drag_out"]
    distances_in_waists = distances / AOD_WAIST_RADIAL_UM

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(len(distances_in_waists)))
    ax.set_xticklabels([f"{d:.2f}" for d in distances_in_waists])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([f"{int(d)}" for d in depths])
    ax.set_xlabel(rf"transverse distance  $d / w_r$  ($w_r = {AOD_WAIST_RADIAL_UM:.2f}$ um)")
    ax.set_ylabel("AOD depth  [uK]")
    ax.set_title("P(spectator dragged out of SLM)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            color = "white" if value < 0.55 else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8, color=color,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Fly-by spectator drag-out: SLM={SLM_DEPTH_UK:.0f} uK, "
        f"T={INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"AOD speed = "
        f"{2*FLYBY_HALF_LENGTH_UM/(DURATION_S*1.0e3):.1f} um/ms"
    )
    return fig, ax


def plot_sweep_lines(sweep_results: dict):
    import matplotlib.pyplot as plt

    distances = sweep_results["distances_um"]
    depths = sweep_results["depths_uK"]
    distances_in_waists = distances / AOD_WAIST_RADIAL_UM

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
    for i, depth in enumerate(depths):
        ax.plot(
            distances_in_waists,
            sweep_results["drag_out"][i],
            marker="o",
            label=f"{depth:.0f} uK",
        )

    ax.set_xlabel(rf"transverse distance  $d / w_r$  ($w_r = {AOD_WAIST_RADIAL_UM:.2f}$ um)")
    ax.set_ylabel("P(spectator dragged out of SLM)")
    ax.set_title("Drag-out probability vs distance")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="AOD depth", fontsize="small")
    ax.set_ylim(-0.02, 1.02)
    return fig, ax


def representative_trajectory_figure():
    from src.visualization import plot_transfer_trajectory_summary

    result, ramp, slm, _ = run_flyby(
        REPRESENTATIVE_DEPTH_UK,
        REPRESENTATIVE_DISTANCE_UM,
        store_trajectories=True,
        ensemble_size=300,
    )
    fig, _ = plot_transfer_trajectory_summary(result, ramp, slm)
    return fig


def print_setup_banner() -> None:
    speed_um_per_ms = 2.0 * FLYBY_HALF_LENGTH_UM / (DURATION_S * 1.0e3)
    print("Spectator-qubit fly-by simulation")
    print(
        f"  SLM:  depth={SLM_DEPTH_UK:.1f} uK, "
        f"waists r={SLM_WAIST_RADIAL_UM} um / z={SLM_WAIST_AXIAL_UM} um"
    )
    print(
        f"  AOD:  waists r={AOD_WAIST_RADIAL_UM} um / z={AOD_WAIST_AXIAL_UM} um, "
        f"constant intensity (no ramp)"
    )
    print(
        f"  AOD path: x = -{FLYBY_HALF_LENGTH_UM:.1f} -> +{FLYBY_HALF_LENGTH_UM:.1f} um, "
        f"y = d, z = 0; speed = {speed_um_per_ms:.1f} um/ms"
    )
    print(
        f"  initial T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"ensemble = {ENSEMBLE_SIZE}, duration = {DURATION_S * 1.0e3:.2f} ms"
    )
    print(
        f"  sweep: depths {list(DEPTHS_UK)} uK x distances {list(DISTANCES_UM)} um"
    )
    print()


def main() -> None:
    print_setup_banner()

    sweep_results = sweep_grid()

    heatmap_fig, _ = plot_sweep_heatmaps(sweep_results)
    lines_fig, _ = plot_sweep_lines(sweep_results)
    rep_fig = representative_trajectory_figure()

    heatmap_path = os.path.join(RENDER_DIR, "spectator_qubits_heatmap.png")
    lines_path = os.path.join(RENDER_DIR, "spectator_qubits_lines.png")
    rep_path = os.path.join(RENDER_DIR, "spectator_qubits_trajectory.png")

    heatmap_fig.savefig(heatmap_path, dpi=180)
    lines_fig.savefig(lines_path, dpi=180)
    rep_fig.savefig(rep_path, dpi=180)
    print(f"saved: {heatmap_path}")
    print(f"saved: {lines_path}")
    print(f"saved: {rep_path}")


if __name__ == "__main__":
    main()
