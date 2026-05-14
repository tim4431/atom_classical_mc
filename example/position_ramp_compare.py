"""AOD ramp-profile parametric heating study.

Spirit of v2's `analytical_calc/parametric_osci_w_drags.nb`: how does the
choice of position ramp shape affect heating and survival in a moving-AOD
drag?

The `arb_fifth_poly(beta)` family is parameterized so that **beta is the
normalized peak velocity** at `u = 0.5` (peak velocity in units of
`drag_distance / drag_time`). The endpoint velocity and the location of
peak velocity at `u = 0.5` are built in for every beta. Endpoint
*acceleration* is `30 - 16 beta` at `u = 0` (and the negative at `u = 1`),
so:

  - `beta = 15/8 = 1.875` reduces to the classic quintic min-jerk
    (zero endpoint acceleration, smoothest possible quintic).
  - `beta < 1.875` gives lower peak velocity but a non-zero "kick" at start
    and end (the v2 author chose `beta = 1.5625`).
  - `beta > 1.875` gives smoother endpoints but a higher peak velocity, and
    therefore a larger peak velocity-coupled focal shift `z01 = dxdt2z * vx`
    when AOD lensing is on.

We run an AOD round-trip (drag out, hold, drag back) for each profile and
report survival and temperature gain. `LINEAR` and `CUBIC_SMOOTHSTEP`
appear in the table as references (peak vel = 1.0 and 1.5 respectively).

Run from the repository root:

    python3 example/position_ramp_compare.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import bound_to_trap  # noqa: E402
from src.constants import (  # noqa: E402
    ATOMIC_MASS_UNIT_KG,
    BOLTZMANN_CONSTANT_J_PER_K,
)
from src.ramp import (  # noqa: E402
    CONST_JERK,
    LINEAR,
    PolynomialConnector,
    RampSequence,
    arb_fifth_poly,
)
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.trap import AstigmaticAODTrap  # noqa: E402

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)
PLANCK_CONST_J_S = 6.626070e-34


def hz_to_uK(depth_hz: float) -> float:
    return depth_hz * PLANCK_CONST_J_S / BOLTZMANN_CONSTANT_J_PER_K / 1.0e-6


# --- AOD setup (matched to v2 single_round_trip parameters) -----------------
ATOM_MASS_KG = 171.0 * ATOMIC_MASS_UNIT_KG
AOD_WAIST_M = 500.0e-9
AOD_DEPTH_UK = hz_to_uK(2.8e6)  # ~134.4 uK; trap freq ~ 51 kHz
AOD_WAVELENGTH_M = 487.0e-9
DRAG_DISTANCE_M = 48.0e-6
# 400 us puts us safely above the survival cliff seen in
# `aod_round_trip_survival.py` (~300 us with lensing on); at this drag time
# all smooth profiles survive, so the comparison highlights heating
# differences rather than catastrophic loss.
T_DRAG_S = 300.0e-6
T_HOLD_S = 11.0e-6
DXDT2Z = 0.025 / 4.0 / 650.0  # AOD f0 / V_sound; lensing on by default.

# --- Numerics ---------------------------------------------------------------
INITIAL_TEMPERATURE_UK = 10.0
ENSEMBLE_SIZE = 500
TIMESTEP_S = 1.0e-6
RANDOM_SEED = 42
LOSS_RADIUS_M = 200.0e-6


@dataclass(frozen=True)
class ProfileCase:
    label: str
    connector: PolynomialConnector
    peak_normalized_velocity: float


def peak_normalized_velocity(connector: PolynomialConnector) -> float:
    """Numeric peak of `dy/du` on `[0, 1]` for any polynomial connector."""

    u = np.linspace(0.0, 1.0, 4001)
    return float(np.max(np.abs(connector.derivative(u))))


def profile_cases() -> list[ProfileCase]:
    cases: list[ProfileCase] = []
    cases.append(ProfileCase("linear", LINEAR, peak_normalized_velocity(LINEAR)))
    cases.append(
        ProfileCase(
            "cubic_smoothstep",
            CONST_JERK,
            peak_normalized_velocity(CONST_JERK),
        )
    )
    for beta in [1.00, 1.25, 1.5625, 1.75, 1.875, 2.00]:
        connector = arb_fifth_poly(beta)
        annotation = "  (= quintic min-jerk)" if beta == 1.875 else ""
        if beta == 1.5625:
            annotation = "  (v2 default)"
        cases.append(
            ProfileCase(
                f"arb_fifth_poly(beta={beta:.4f}){annotation}",
                connector,
                peak_normalized_velocity(connector),
            )
        )
    return cases


def build_round_trip_ramp(connector: PolynomialConnector) -> RampSequence:
    times_s = np.array(
        [0.0, T_DRAG_S, T_DRAG_S + T_HOLD_S, 2.0 * T_DRAG_S + T_HOLD_S],
        dtype=float,
    )
    centers_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [DRAG_DISTANCE_M, 0.0, 0.0],
            [DRAG_DISTANCE_M, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    depths_uK = np.full(4, AOD_DEPTH_UK, dtype=float)
    return RampSequence(
        times_s=times_s,
        centers_m=centers_m,
        depths_uK=depths_uK,
        position_profile=connector,
    )


@dataclass(frozen=True)
class CaseResult:
    case: ProfileCase
    survival: float
    n_init: int
    n_final: int
    temperature_gain_uK_bound: float
    temperature_gain_uK_all: float
    peak_lab_velocity_m_per_s: float
    peak_focal_shift_m: float


def _kinetic_temperature_uK(velocities: np.ndarray, mass_kg: float) -> float:
    if velocities.size == 0:
        return float("nan")
    mean_v_sq = float(np.mean(np.sum(velocities * velocities, axis=-1)))
    return mass_kg * mean_v_sq / (3.0 * BOLTZMANN_CONSTANT_J_PER_K) / 1.0e-6


def run_case(case: ProfileCase) -> CaseResult:
    ramp = build_round_trip_ramp(case.connector)
    aod = AstigmaticAODTrap(
        waist_radial_m=AOD_WAIST_M,
        wavelength_m=AOD_WAVELENGTH_M,
        ramp=ramp,
        dxdt2z=DXDT2Z,
        name="AOD",
    )
    duration_s = 2.0 * T_DRAG_S + T_HOLD_S
    config = SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=duration_s,
        ensemble_size=ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
        mass_kg=ATOM_MASS_KG,
        loss_radius_m=LOSS_RADIUS_M,
        track_energy_loss=False,
    )
    result = run_simulation([aod], config)

    bound_initial = bound_to_trap(
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
        aod,
        mass_kg=config.mass_kg,
        time_s=0.0,
    )
    bound_final = bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        aod,
        mass_kg=config.mass_kg,
        time_s=0.0,
    )
    bound_final_mask = bound_final & ~result.lost
    n_init = int(np.sum(bound_initial))
    n_final = int(np.sum(bound_final_mask))
    survival = n_final / max(n_init, 1)

    peak_lab_v = case.peak_normalized_velocity * DRAG_DISTANCE_M / T_DRAG_S
    peak_focal_shift = DXDT2Z * peak_lab_v

    # Kinetic temperature at t=0 (where AOD is at rest, so lab and trap frames
    # coincide) and at t=duration_s (AOD again at rest, having returned home).
    # Restrict to atoms that survived the round trip bound to the AOD.
    if n_final > 0:
        t_init_bound = _kinetic_temperature_uK(
            result.initial_velocities_m_per_s[bound_final_mask], config.mass_kg
        )
        t_final_bound = _kinetic_temperature_uK(
            result.final_velocities_m_per_s[bound_final_mask], config.mass_kg
        )
        gain_bound = t_final_bound - t_init_bound
    else:
        gain_bound = float("nan")

    t_init_all = _kinetic_temperature_uK(
        result.initial_velocities_m_per_s, config.mass_kg
    )
    t_final_all = _kinetic_temperature_uK(
        result.final_velocities_m_per_s, config.mass_kg
    )

    return CaseResult(
        case=case,
        survival=survival,
        n_init=n_init,
        n_final=n_final,
        temperature_gain_uK_bound=gain_bound,
        temperature_gain_uK_all=t_final_all - t_init_all,
        peak_lab_velocity_m_per_s=peak_lab_v,
        peak_focal_shift_m=peak_focal_shift,
    )


def print_table(rows: list[CaseResult]) -> None:
    header = (
        f"{'profile':<42} "
        f"{'peak v_n':>9} "
        f"{'v_peak m/s':>11} "
        f"{'z01_peak/zR':>13} "
        f"{'surv':>6} "
        f"{'dT(bound) uK':>13} "
        f"{'dT(all) uK':>11}"
    )
    print(header)
    print("-" * len(header))
    zR = np.pi * AOD_WAIST_M**2 / AOD_WAVELENGTH_M
    for row in rows:
        dT_bound = (
            f"{row.temperature_gain_uK_bound:+13.3f}"
            if np.isfinite(row.temperature_gain_uK_bound)
            else f"{'n/a':>13}"
        )
        print(
            f"{row.case.label:<42} "
            f"{row.case.peak_normalized_velocity:9.4f} "
            f"{row.peak_lab_velocity_m_per_s:11.4f} "
            f"{row.peak_focal_shift_m / zR:13.3f} "
            f"{row.survival:6.3f} "
            f"{dT_bound} "
            f"{row.temperature_gain_uK_all:+11.3f}"
        )


def plot_comparison(rows: list[CaseResult]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # Panel (0,0): position profile y(u) for each case.
    ax = axes[0, 0]
    u = np.linspace(0.0, 1.0, 401)
    for row in rows:
        ax.plot(u, row.case.connector.value(u), label=row.case.label, linewidth=1.4)
    ax.set_xlabel("normalized time  u = t / T_drag")
    ax.set_ylabel("normalized position  y(u)")
    ax.set_title("Drag position profiles")
    ax.legend(loc="lower right", fontsize="x-small")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel (0,1): normalized velocity profile dy/du for each case.
    ax = axes[0, 1]
    for row in rows:
        ax.plot(
            u, row.case.connector.derivative(u), label=row.case.label, linewidth=1.4
        )
    ax.set_xlabel("normalized time  u = t / T_drag")
    ax.set_ylabel("normalized velocity  dy/du")
    ax.set_title("Drag velocity profiles (peak = beta for arb_fifth_poly)")
    ax.legend(loc="lower center", fontsize="x-small", ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel (1,0): survival vs profile (bar).
    ax = axes[1, 0]
    labels = [_short_label(row.case.label) for row in rows]
    x = np.arange(len(rows))
    survival = [row.survival for row in rows]
    bars = ax.bar(x, survival, color="tab:blue", alpha=0.8)
    for bar, value in zip(bars, survival):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("survival ratio")
    ax.set_title(
        f"Round-trip survival (T_drag = {T_DRAG_S * 1e6:.0f} us, "
        f"drag = {DRAG_DISTANCE_M * 1e6:.0f} um, lensing on)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel (1,1): temperature gain vs profile (bar pair).
    ax = axes[1, 1]
    width = 0.4
    dT_bound = [
        (
            row.temperature_gain_uK_bound
            if np.isfinite(row.temperature_gain_uK_bound)
            else 0.0
        )
        for row in rows
    ]
    dT_all = [row.temperature_gain_uK_all for row in rows]
    ax.bar(
        x - width / 2.0,
        dT_bound,
        width,
        label="round-trip bound atoms only",
        color="tab:orange",
    )
    ax.bar(x + width / 2.0, dT_all, width, label="full ensemble", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel("temperature gain (uK)")
    ax.set_title("Heating per profile")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    return figure, axes


def _short_label(label: str) -> str:
    if label.startswith("arb_fifth_poly("):
        # arb_fifth_poly(beta=1.5625) -> beta=1.56
        beta_value = float(label.split("=")[1].split(")")[0])
        return f"beta={beta_value:.3f}"
    return label


def render_comparison_gif(
    profile_indices: list[int],
    output_path: str,
    ensemble_size: int = 250,
    trajectory_stride: int = 4,
    fps: int = 24,
    dpi: int = 110,
) -> str:
    """Render a column-layout GIF comparing dynamics for several ramp profiles.

    Layout: N stacked rows on the left, each showing one profile's atoms in
    the (x, z) plane during the AOD round-trip; one tall panel on the right
    showing each profile's AOD center vs time with a red vertical scrubber
    indicating the current frame's time. Atoms are colored by post-trip
    bound state (green = bound to the AOD at the end, red = lost). z is the
    direction along which AOD lensing pushes the atoms, so the velocity-
    coupled focal-shift kick is visible as out-of-plane drift during the
    fast section of the drag.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle

    cases = profile_cases()
    selected = [cases[i] for i in profile_indices]

    duration_s = 2.0 * T_DRAG_S + T_HOLD_S
    runs: list[dict] = []
    for case in selected:
        ramp = build_round_trip_ramp(case.connector)
        aod = AstigmaticAODTrap(
            waist_radial_m=AOD_WAIST_M,
            wavelength_m=AOD_WAVELENGTH_M,
            ramp=ramp,
            dxdt2z=DXDT2Z,
            name="AOD",
        )
        config = SimulationConfig(
            initial_temperature_uK=INITIAL_TEMPERATURE_UK,
            timestep_s=TIMESTEP_S,
            duration_s=duration_s,
            ensemble_size=ensemble_size,
            random_seed=RANDOM_SEED,
            mass_kg=ATOM_MASS_KG,
            loss_radius_m=LOSS_RADIUS_M,
            track_energy_loss=False,
            store_trajectories=True,
            trajectory_stride=trajectory_stride,
        )
        result = run_simulation([aod], config)
        bound_final = bound_to_trap(
            result.final_positions_m,
            result.final_velocities_m_per_s,
            aod,
            mass_kg=config.mass_kg,
            time_s=0.0,
        )
        bound_initial = bound_to_trap(
            result.initial_positions_m,
            result.initial_velocities_m_per_s,
            aod,
            mass_kg=config.mass_kg,
            time_s=0.0,
        )
        survival = float(np.sum(bound_final & ~result.lost)) / max(
            float(np.sum(bound_initial)), 1.0
        )
        runs.append(
            dict(
                case=case,
                ramp=ramp,
                result=result,
                bound_final=bound_final,
                survival=survival,
            )
        )

    times_s = runs[0]["result"].trajectory_times_s
    n_frames = len(times_s)
    n_panels = len(selected)

    fig_height = max(2.0 + 1.6 * n_panels, 7.5)
    fig = plt.figure(figsize=(14.0, fig_height), dpi=dpi)
    gs = GridSpec(
        n_panels,
        2,
        figure=fig,
        width_ratios=[2.6, 1.0],
        wspace=0.18,
        hspace=0.55,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )
    left_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]
    right_ax = fig.add_subplot(gs[:, 1])

    x_lo, x_hi = -3.0, 53.0
    z_extent = 3.5

    profile_colors = plt.cm.viridis(np.linspace(0.12, 0.85, n_panels))

    artists = []
    for index, (ax, run) in enumerate(zip(left_axes, runs)):
        bound = np.asarray(run["bound_final"], dtype=bool)
        atom_colors = np.where(bound, "tab:green", "tab:red")
        initial_xz_um = run["result"].trajectory_positions_m[0][:, [0, 2]] * 1.0e6
        scatter = ax.scatter(
            initial_xz_um[:, 0],
            initial_xz_um[:, 1],
            s=10,
            c=atom_colors,
            alpha=0.75,
            edgecolors="none",
        )
        aod_marker = ax.plot(
            [], [], marker="x", color="black", markersize=11, mew=2, linestyle="None"
        )[0]
        circle = Circle(
            (0.0, 0.0),
            radius=AOD_WAIST_M * 1.0e6,
            fill=False,
            color="black",
            linestyle="--",
            alpha=0.7,
            linewidth=1.1,
        )
        ax.add_patch(circle)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-z_extent, z_extent)
        ax.set_xlabel("x [um]", fontsize=9)
        ax.set_ylabel("z [um]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        # Colored vertical strip on the y-axis to key this row to its right-
        # panel curve, then the title.
        ax.spines["left"].set_color(profile_colors[index])
        ax.spines["left"].set_linewidth(2.2)
        ax.set_title(
            f"{_short_label(run['case'].label)}   "
            f"final survival = {run['survival']:.2f}",
            fontsize=10,
            color=profile_colors[index],
        )
        artists.append(dict(scatter=scatter, marker=aod_marker, circle=circle, ax=ax))

    # --- Right panel: AOD center x vs time + scrubber -----------------------
    t_dense_s = np.linspace(0.0, duration_s, 601)
    for index, run in enumerate(runs):
        x_um = (
            np.array([run["ramp"].center_at(t)[0] for t in t_dense_s], dtype=float)
            * 1.0e6
        )
        right_ax.plot(
            t_dense_s * 1.0e6,
            x_um,
            color=profile_colors[index],
            linewidth=1.8,
            label=_short_label(run["case"].label),
        )
    right_ax.set_xlim(0.0, duration_s * 1.0e6)
    right_ax.set_ylim(-3.0, max(53.0, DRAG_DISTANCE_M * 1.0e6 + 5.0))
    right_ax.set_xlabel("time [us]", fontsize=10)
    right_ax.set_ylabel("AOD center x [um]", fontsize=10)
    right_ax.set_title("AOD trajectory", fontsize=11)
    right_ax.tick_params(labelsize=8)
    right_ax.grid(True, linestyle=":", alpha=0.4)
    right_ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    scrubber = right_ax.axvline(0.0, color="red", linewidth=1.6, alpha=0.85, zorder=5)

    time_text = fig.suptitle(
        f"AOD round-trip   T_drag = {T_DRAG_S * 1e6:.0f} us   "
        f"drag = {DRAG_DISTANCE_M * 1e6:.0f} um   t = 0.0 us",
        fontsize=12,
        fontweight="bold",
    )
    base_title = (
        f"AOD round-trip   T_drag = {T_DRAG_S * 1e6:.0f} us   "
        f"drag = {DRAG_DISTANCE_M * 1e6:.0f} um"
    )

    def update(frame: int):
        t_us = float(times_s[frame] * 1.0e6)
        for run, art in zip(runs, artists):
            positions_m = run["result"].trajectory_positions_m[frame]
            xz_um = positions_m[:, [0, 2]] * 1.0e6
            art["scatter"].set_offsets(xz_um)
            aod_center_um = run["ramp"].center_at(times_s[frame]) * 1.0e6
            art["marker"].set_data([aod_center_um[0]], [0.0])
            art["circle"].center = (aod_center_um[0], 0.0)
        scrubber.set_xdata([t_us, t_us])
        time_text.set_text(f"{base_title}   t = {t_us:6.1f} us")
        return ()

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    print("AOD ramp-profile parametric heating study")
    print(
        f"  AOD: waist={AOD_WAIST_M*1e9:.0f} nm, depth={AOD_DEPTH_UK:.2f} uK, "
        f"lambda={AOD_WAVELENGTH_M*1e9:.0f} nm"
    )
    print(
        f"  drag distance = {DRAG_DISTANCE_M*1e6:.1f} um, "
        f"T_drag = {T_DRAG_S*1e6:.0f} us, hold = {T_HOLD_S*1e6:.0f} us"
    )
    print(
        f"  ensemble = {ENSEMBLE_SIZE}, T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"dt = {TIMESTEP_S*1e6:.1f} us, lensing on (dxdt2z = {DXDT2Z:.3e})"
    )
    print()

    cases = profile_cases()
    rows = [run_case(case) for case in cases]
    print_table(rows)

    figure, _ = plot_comparison(rows)
    out_path = os.path.join(RENDER_DIR, "position_ramp_compare.png")
    figure.savefig(out_path, dpi=180)
    print(f"\nsaved: {out_path}")

    if "--gif" in sys.argv:
        # Pick four representative profiles for the column-layout animation.
        # Indices match `profile_cases()`: linear, cubic_smoothstep, beta=1.5625
        # (v2 default), beta=1.875 (= quintic min-jerk).
        demo_dir = os.path.abspath(os.path.join(HERE, "..", "demo"))
        os.makedirs(demo_dir, exist_ok=True)
        gif_path = os.path.join(demo_dir, "position_ramp_compare.gif")
        rendered = render_comparison_gif(
            profile_indices=[0, 1, 4, 6],
            output_path=gif_path,
        )
        print(f"saved: {rendered}")


if __name__ == "__main__":
    main()
