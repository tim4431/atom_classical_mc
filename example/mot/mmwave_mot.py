"""Grating MOT (gMOT) loading Rb85 directly from a hot effusive beam.

Geometry (all built from the general light-matter parts, no gMOT code
in the library):

- A hot oven emits a collimated Rb85 atomic beam (600 K, 2 mm diameter,
  no Zeeman slower) traveling along +x through the trap center.
- A single uniform (top-hat) MOT beam, 50 mm diameter, 120 mW,
  circularly polarized, propagates along -z — the quadrupole coil axis
  (the strong-gradient direction).
- A tri-sector diffraction grating chip sits below the trap center.
  Each 120 deg sector diffracts its share of the incident beam upward
  and inward at deflection angle alpha = 46 deg from the chip normal.
  Handedness flips on diffraction, and each diffracted beam is confined
  to the sheared triangular prism defined by back-projecting onto its
  sector (`LaserBeam.profile`).

Local polarization projection: at every timestep, each beam's helicity
is decomposed into sigma+ / pi / sigma- intensity fractions along the
atom's local quantization axis B(r)/|B(r)| (`polarization_fractions`).
This projection is what confines a gMOT axially: the diffracted beams
arrive at cos(theta) = -cos(alpha) to the local field axis, so their
Zeeman-enhanced sigma fraction differs from the incident beam's, and
the imbalance yields a (weaker than 6-beam) restoring force. The script
prints this decomposition, the restoring-force profile, and the
velocity-dependent deceleration before running the Monte Carlo.

Simplifications: the chip is not a physical barrier — atoms that dive
below it simply see zero light (they are lost in practice and never
counted as captured); the zeroth diffraction order and grating losses
are absorbed into the per-sector efficiency.

Because a 600 K beam is far faster than any MOT capture velocity, only
the slow Boltzmann tail can be captured. The simulation samples the
effusive flux distribution `f(v) ~ v^3 exp(-v^2 / 2 sigma^2)` truncated
at `--vmax` and reports both the in-window capture probability and the
absolute capture fraction of the full beam flux (window weight computed
analytically).

Run from the repository root:

    python3 example/mot/mmwave_mot.py
    python3 example/mot/mmwave_mot.py --atoms 500 --vmax 35
    python3 example/mot/mmwave_mot.py --save-plot
    python3 example/mot/mmwave_mot.py --gif   # animated capture GIF
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.constants import BOLTZMANN_CONSTANT_J_PER_K  # noqa: E402
from src.fields import QuadrupoleMagneticField  # noqa: E402
from src.internal_state import AdiabaticSteadyState  # noqa: E402
from src.laser import LaserBeam  # noqa: E402
from src.light_matter import (  # noqa: E402
    LightMatterSystem,
    polarization_fractions,
)
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.species import RB85_D2  # noqa: E402
from src.units import gauss_per_cm, ms  # noqa: E402

HERE = os.path.dirname(__file__)

SPECIES = RB85_D2
GAMMA_HZ = SPECIES.linewidth_rad_s / (2.0 * np.pi)

# --- MOT light -----------------------------------------------------------
MOT_BEAM_RADIUS_M = 25.0e-3  # 50 mm diameter, uniform top-hat
MOT_BEAM_POWER_W = 120.0e-3
DETUNING_GAMMA = -1.5  # laser detuning in units of Gamma
INCIDENT_HELICITY = -1.0  # circular; sign must match the coil polarity
GRADIENT_G_PER_CM = 5.0  # radial; axial (beam axis) is twice this

# --- grating chip --------------------------------------------------------
DEFLECTION_ALPHA_RAD = np.deg2rad(46.0)
SECTOR_AZIMUTHS_RAD = np.deg2rad([0.0, 120.0, 240.0])
DIFFRACTION_EFFICIENCY = 1.0 / 3.0  # power fraction into each +1 order
CHIP_Z_M = -10.0e-3  # chip surface below the quadrupole zero

# --- atomic beam ---------------------------------------------------------
OVEN_TEMPERATURE_K = 600.0
ATOM_BEAM_RADIUS_M = 1.0e-3  # 2 mm diameter, aimed through the trap center
ATOM_BEAM_START_X_M = -30.0e-3
DIVERGENCE_HALF_ANGLE_RAD = 10.0e-3  # residual collimation spread


def incident_saturation() -> float:
    intensity = MOT_BEAM_POWER_W / (np.pi * MOT_BEAM_RADIUS_M**2)
    return float(intensity / SPECIES.saturation_intensity_w_per_m2)


def build_beams(detuning_hz: float) -> list[LaserBeam]:
    """Incident top-hat beam plus three grating-sector diffracted beams."""

    s_inc = incident_saturation()

    def incident_profile(positions):
        pos = np.asarray(positions, dtype=float)
        rho = np.hypot(pos[..., 0], pos[..., 1])
        return ((rho <= MOT_BEAM_RADIUS_M) & (pos[..., 2] >= CHIP_Z_M)).astype(float)

    beams = [
        LaserBeam(
            direction=(0.0, 0.0, -1.0),
            detuning_hz=detuning_hz,
            saturation=s_inc,
            helicity=INCIDENT_HELICITY,
            profile=incident_profile,
            name="incident",
        )
    ]

    # Each diffracted beam: intensity eta / cos(alpha) times the incident
    # (footprint compression), handedness flipped, confined to the prism
    # back-projecting onto its 120 deg sector.
    s_diff = DIFFRACTION_EFFICIENCY * s_inc / np.cos(DEFLECTION_ALPHA_RAD)
    for phi in SECTOR_AZIMUTHS_RAD:
        direction = np.array(
            [
                -np.sin(DEFLECTION_ALPHA_RAD) * np.cos(phi),
                -np.sin(DEFLECTION_ALPHA_RAD) * np.sin(phi),
                np.cos(DEFLECTION_ALPHA_RAD),
            ]
        )
        beams.append(
            LaserBeam(
                direction=direction,
                detuning_hz=detuning_hz,
                saturation=s_diff,
                helicity=-INCIDENT_HELICITY,
                profile=_sector_profile(direction, phi),
                name=f"diffracted_{np.rad2deg(phi):.0f}",
            )
        )
    return beams


def _sector_profile(direction: np.ndarray, phi_center_rad: float):
    """Indicator of the sheared prism above one 120 deg grating sector."""

    kx, ky, kz = (float(c) for c in direction)

    def profile(positions):
        pos = np.asarray(positions, dtype=float)
        t = (pos[..., 2] - CHIP_Z_M) / kz  # distance since leaving the chip
        foot_x = pos[..., 0] - t * kx
        foot_y = pos[..., 1] - t * ky
        rho = np.hypot(foot_x, foot_y)
        dphi = np.arctan2(foot_y, foot_x) - phi_center_rad
        dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
        inside = (t >= 0.0) & (rho <= MOT_BEAM_RADIUS_M) & (np.abs(dphi) <= np.pi / 3.0)
        return inside.astype(float)

    return profile


def build_system(detuning_gamma: float) -> LightMatterSystem:
    detuning_hz = detuning_gamma * GAMMA_HZ
    quadrupole = QuadrupoleMagneticField(
        gradient_T_per_m=float(gauss_per_cm(GRADIENT_G_PER_CM)),
        axis=(0.0, 0.0, 1.0),
    )
    return LightMatterSystem(
        species=SPECIES,
        beams=build_beams(detuning_hz),
        magnetic_fields=[quadrupole],
    )


def sample_oven_beam(
    n_atoms: int, v_max: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample the slow tail of the effusive beam.

    Returns positions, velocities, and the fraction of the total beam
    flux carried by the simulated window `v < v_max` (analytic:
    `1 - exp(-u)(1 + u)` with `u = v_max^2 / 2 sigma^2` for the
    `v^3 exp(-v^2/2 sigma^2)` flux distribution).
    """

    sigma = np.sqrt(BOLTZMANN_CONSTANT_J_PER_K * OVEN_TEMPERATURE_K / SPECIES.mass_kg)
    u = v_max**2 / (2.0 * sigma**2)
    window_fraction = float(1.0 - np.exp(-u) * (1.0 + u))

    # Rejection sampling of f(v) ~ v^3 exp(-v^2/2s^2) on (0, v_max]:
    # envelope v^3 (inverse-CDF v_max * u^(1/4)), accept with the
    # Gaussian factor (nearly 1 for v_max << sigma).
    speeds = np.empty(0)
    while speeds.size < n_atoms:
        candidates = v_max * rng.random(2 * n_atoms) ** 0.25
        accept = rng.random(candidates.size) < np.exp(
            -(candidates**2) / (2.0 * sigma**2)
        )
        speeds = np.concatenate([speeds, candidates[accept]])
    speeds = speeds[:n_atoms]

    # Transverse launch position: uniform disc aimed through the origin.
    disc_phi = rng.uniform(0.0, 2.0 * np.pi, n_atoms)
    disc_r = ATOM_BEAM_RADIUS_M * np.sqrt(rng.random(n_atoms))
    positions = np.column_stack(
        [
            np.full(n_atoms, ATOM_BEAM_START_X_M),
            disc_r * np.cos(disc_phi),
            disc_r * np.sin(disc_phi),
        ]
    )

    # Residual divergence: velocity tilted by a small random angle.
    theta = DIVERGENCE_HALF_ANGLE_RAD * np.sqrt(rng.random(n_atoms))
    tilt_phi = rng.uniform(0.0, 2.0 * np.pi, n_atoms)
    velocities = np.column_stack(
        [
            speeds * np.cos(theta),
            speeds * np.sin(theta) * np.cos(tilt_phi),
            speeds * np.sin(theta) * np.sin(tilt_phi),
        ]
    )
    return positions, velocities, window_fraction


def print_diagnostics(system: LightMatterSystem) -> None:
    """Polarization projection, restoring force, and slowing force."""

    print("Local polarization projection on the atom quantization axis")
    print("(atom at (0, 0, +2 mm), where B points along -z):")
    probe = np.array([[0.0, 0.0, 2.0e-3]])
    b_vec = system.magnetic_field_at(probe)[0]
    b_hat = b_vec / np.linalg.norm(b_vec)
    print("  beam              cos(theta)   f_sigma+   f_pi   f_sigma-")
    for beam in system.beams:
        cos_theta = float(np.dot(beam.direction, b_hat))
        f_plus, f_pi, f_minus = polarization_fractions(
            beam.helicity, np.array([cos_theta])
        )
        print(
            f"  {beam.name:<16s}  {cos_theta:+9.3f}   {float(f_plus[0]):7.3f}"
            f"  {float(f_pi[0]):5.3f}   {float(f_minus[0]):7.3f}"
        )

    print("\nMean restoring force (should point back to the origin):")
    for label, r in (
        ("+2 mm along x", [2e-3, 0.0, 0.0]),
        ("-2 mm along x", [-2e-3, 0.0, 0.0]),
        ("+2 mm along z", [0.0, 0.0, 2e-3]),
        ("-2 mm along z", [0.0, 0.0, -2e-3]),
    ):
        f = system.mean_radiation_force(np.array([r]), np.zeros((1, 3)))[0]
        print(f"  {label}: F = ({f[0]:+.2e}, {f[1]:+.2e}, {f[2]:+.2e}) N")

    print("\nDeceleration vs longitudinal velocity (atom at origin):")
    print("  v_x [m/s]   a_x [km/s^2]")
    for vx in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        f = system.mean_radiation_force(np.zeros((1, 3)), np.array([[vx, 0.0, 0.0]]))[0]
        print(f"  {vx:8.1f}   {f[0] / SPECIES.mass_kg / 1e3:+12.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=250)
    parser.add_argument(
        "--vmax",
        type=float,
        default=30.0,
        help="upper edge of the simulated slow-velocity window [m/s]",
    )
    parser.add_argument("--duration-ms", type=float, default=30.0)
    parser.add_argument("--detuning-gamma", type=float, default=DETUNING_GAMMA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument(
        "--gif",
        action="store_true",
        help="render an animated GIF of the capture next to this script",
    )
    args = parser.parse_args()

    system = build_system(args.detuning_gamma)
    rng = np.random.default_rng(args.seed)
    positions, velocities, window_fraction = sample_oven_beam(
        args.atoms, args.vmax, rng
    )

    s_inc = incident_saturation()
    print("Grating MOT loading from a hot effusive beam")
    print(f"  species             : {SPECIES.name}")
    print(f"  oven temperature    : {OVEN_TEMPERATURE_K:.0f} K")
    print(
        f"  incident saturation : s0 = {s_inc:.2f} "
        f"({MOT_BEAM_POWER_W * 1e3:.0f} mW over {2 * MOT_BEAM_RADIUS_M * 1e3:.0f} mm)"
    )
    print(f"  detuning            : {args.detuning_gamma:+.1f} Gamma")
    print(
        f"  gradient            : {GRADIENT_G_PER_CM:.0f} G/cm radial "
        f"({2 * GRADIENT_G_PER_CM:.0f} G/cm along the beam axis)"
    )
    print(
        f"  deflection angle    : {np.rad2deg(DEFLECTION_ALPHA_RAD):.0f} deg, "
        f"efficiency {DIFFRACTION_EFFICIENCY:.2f}/sector"
    )
    print(
        f"  simulated window    : v < {args.vmax:.0f} m/s = "
        f"{window_fraction:.2e} of the beam flux\n"
    )

    print_diagnostics(system)

    config = SimulationConfig(
        initial_temperature_uK=0.0,  # unused: explicit ensemble below
        timestep_s=2.0e-7,
        duration_s=float(ms(args.duration_ms)),
        ensemble_size=args.atoms,
        mass_kg=SPECIES.mass_kg,
        initial_positions_m=positions,
        initial_velocities_m_per_s_array=velocities,
        reject_initially_lost=False,
        loss_radius_m=40.0e-3,
        random_seed=args.seed,
        store_trajectories=True,
        trajectory_stride=500,  # sample every 100 us
    )
    result = run_simulation(
        [], config, scattering=system, internal_model=AdiabaticSteadyState()
    )

    final_r = np.linalg.norm(result.final_positions_m, axis=-1)
    final_v = np.linalg.norm(result.final_velocities_m_per_s, axis=-1)
    captured = (final_r < 3.0e-3) & (final_v < 1.0)
    n_captured = int(np.sum(captured))
    capture_probability = n_captured / args.atoms

    print("Results")
    print(
        f"  captured atoms          : {n_captured} / {args.atoms} "
        f"(p = {capture_probability:.3f} within the window)"
    )
    print(
        f"  absolute capture fraction of beam flux: "
        f"{capture_probability * window_fraction:.2e}"
    )
    if n_captured > 0:
        initial_speeds = np.linalg.norm(result.initial_velocities_m_per_s, axis=-1)
        v_cap = initial_speeds[captured]
        centered = result.final_velocities_m_per_s[captured] - np.mean(
            result.final_velocities_m_per_s[captured], axis=0
        )
        temp_uK = (
            SPECIES.mass_kg
            * float(np.mean(np.sum(centered**2, axis=-1)))
            / 3.0
            / BOLTZMANN_CONSTANT_J_PER_K
            * 1e6
        )
        cloud_rms_mm = float(
            np.sqrt(np.mean(np.sum(result.final_positions_m[captured] ** 2, axis=-1)))
            * 1e3
        )
        print(
            f"  captured initial speeds : {np.min(v_cap):.1f} - "
            f"{np.max(v_cap):.1f} m/s (mean {np.mean(v_cap):.1f})"
        )
        print(
            f"  captured cloud          : T = {temp_uK:.0f} uK, "
            f"rms radius {cloud_rms_mm:.2f} mm"
        )
        print(
            f"  mean photons (captured) : "
            f"{float(np.mean(result.scattered_photons[captured])):.0f}"
        )

    if args.plot or args.save_plot:
        _plot(result, captured, args.save_plot)

    if args.gif:
        _render_gif(result, system)


def _render_gif(result, system: LightMatterSystem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.visualization import render_cloud_animation

    out = os.path.join(HERE, "mmwave_mot_capture.gif")
    render_cloud_animation(
        result,
        out,
        mass_kg=system.species.mass_kg,
        plane="xz",
        beams=system.beams,
        doppler_limit_uK=system.species.doppler_temperature_uK,
        # Asymmetric window: keep the grating chip (z = -10 mm) and the
        # incoming beam path in view rather than auto-centering.
        extent_mm=(-32.0, 32.0, -13.0, 18.0),
        n_frames=80,
        fps=16,
        dpi=85,
    )
    print(f"\nSaved GIF to {out}")


def _plot(result, captured, save: bool) -> None:
    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # x-z trajectories with the chip and grating sketch. Atoms that hit
    # the chip are not propagated as a physical collision (they only see
    # zero light below it), so clip their trajectories at the surface.
    traj_mm = result.trajectory_positions_m.copy() * 1e3
    below = np.cumsum(traj_mm[:, :, 2] < CHIP_Z_M * 1e3, axis=0) > 0
    traj_mm[below] = np.nan
    n_atoms = traj_mm.shape[1]
    for atom in range(n_atoms):
        color = "C1" if captured[atom] else "C0"
        alpha = 0.9 if captured[atom] else 0.15
        axes[0].plot(
            traj_mm[:, atom, 0], traj_mm[:, atom, 2], lw=0.6, color=color, alpha=alpha
        )
    axes[0].axhline(CHIP_Z_M * 1e3, color="k", lw=2)
    axes[0].plot(0.0, 0.0, "r+", ms=10)
    axes[0].set_xlim(-32, 32)
    axes[0].set_ylim(-14, 20)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[0].set_title("Trajectories (orange = captured)")

    # Capture probability vs initial longitudinal speed.
    speeds = np.linalg.norm(result.initial_velocities_m_per_s, axis=-1)
    bins = np.linspace(0.0, np.max(speeds), 16)
    total, _ = np.histogram(speeds, bins=bins)
    caught, _ = np.histogram(speeds[captured], bins=bins)
    centers = 0.5 * (bins[1:] + bins[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, caught / total, np.nan)
    axes[1].bar(centers, fraction, width=0.9 * (bins[1] - bins[0]), color="C1")
    axes[1].set_xlabel("initial speed [m/s]")
    axes[1].set_ylabel("capture probability")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Capture vs initial speed")

    # Speed of each captured atom vs time.
    speeds_t = np.linalg.norm(result.trajectory_velocities_m_per_s, axis=-1)
    times_ms_axis = result.trajectory_times_s * 1e3
    for atom in np.flatnonzero(captured):
        axes[2].plot(times_ms_axis, speeds_t[:, atom], lw=0.8, alpha=0.7)
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("speed [m/s]")
    axes[2].set_title("Slowing of captured atoms")

    fig.tight_layout()
    if save:
        out = os.path.join(HERE, "mmwave_mot_summary.png")
        fig.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
