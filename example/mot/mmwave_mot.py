"""Grating MOT (gMOT) loading Rb85 directly from a hot effusive beam.

Geometry (all built from the general light-matter parts, no gMOT code
in the library):

- A single uniform (top-hat) MOT beam, 50 mm diameter, 120 mW,
  circularly polarized, propagates along -z — the quadrupole coil axis
  (the strong-gradient direction).
- A tri-sector diffraction grating chip sits below the trap center.
  Each 120 deg sector diffracts its share of the incident beam upward
  and inward at deflection angle alpha = 46 deg from the chip normal.
  Handedness flips on diffraction, and each diffracted beam is confined
  to the sheared triangular prism defined by back-projecting onto its
  sector (`LaserBeam.profile`).
- A hot oven below the chip emits a collimated Rb85 atomic beam (600 K,
  2 mm diameter, no Zeeman slower) traveling along +z — head-on into
  the incident MOT beam — through a small hole in the grating center.
  The incident beam leaks through that hole as a downward pencil beam,
  so incoming atoms are Doppler-slowed below the chip before they ever
  reach the trap volume (through-hole loading).

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

Forces shaping the loading dynamics:

- Static balance: the z-momentum flux ratio of diffracted to incident
  light is exactly `3 * eta` — the 1/cos(alpha) intensity compression
  of each diffracted beam cancels the cos(alpha) projection of its
  push. Only `eta = 1/3` balances; larger eta ejects slow atoms upward,
  smaller lets the incident beam press the cloud toward the chip.
- Axial slowing: a rising atom is head-on with the incident beam
  (Doppler-enhanced, resonant near `v = |delta|/k` shifted up by the
  Zeeman term, which grows with depth — a built-in Zeeman-slower ramp)
  while the three diffracted beams co-propagate with it and are
  Doppler-suppressed. On axis the transverse pushes of the three
  diffracted beams cancel by symmetry.
- Stall depth sets the chip position: at s0 ~ 4 the pencil beam is a
  wall, not a gentle slower — every atom below the punch-through speed
  (~45 m/s here) is driven to rest at the depth where the Zeeman shift
  detunes the light at v = 0, `|z| ~ hbar(|delta| + Gamma/2) /
  (mu_eff * 2b')` (~8.5 mm). If the chip is shallower than that, atoms
  stall below the surface and are expelled back down the beamline
  (capture ~ 0); if deeper, they stall inside the trap volume and are
  captured with near-unity probability over the whole window.

Because a 600 K beam is far faster than any MOT capture velocity, only
the slow Boltzmann tail can be captured. The simulation samples the
effusive flux distribution `f(v) ~ v^3 exp(-v^2 / 2 sigma^2)` truncated
at `--vmax` and reports both the in-window capture probability and the
absolute capture fraction of the full beam flux (window weight computed
analytically).

Run from the repository root:

    python3 example/mot/mmwave_mot.py                 # sim + summary + GIF
    python3 example/mot/mmwave_mot.py --atoms 500 --vmax 35
    python3 example/mot/mmwave_mot.py --no-gif        # skip the animation
    python3 example/mot/mmwave_mot.py --save-beams    # beam geometry only
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from atommc import (  # noqa: E402
    AdiabaticSteadyState,
    AtomSystem,
    BOLTZMANN_CONSTANT_J_PER_K,
    EffusiveBeam,
    LaserBeam,
    LightMatterSystem,
    LightScattering,
    QuadrupoleMagneticField,
    RB85_D2,
    SimulationConfig,
    gauss_per_cm,
    ms,
    polarization_fractions,
    simulate,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")

SPECIES = RB85_D2
GAMMA_HZ = SPECIES.linewidth_rad_s / (2.0 * np.pi)

# --- MOT light -----------------------------------------------------------
MOT_BEAM_RADIUS_M = 25.0e-3  # 50 mm diameter, uniform top-hat
MOT_BEAM_POWER_W = 20.0e-3
DETUNING_GAMMA = -1.5  # laser detuning in units of Gamma
INCIDENT_HELICITY = -1.0  # circular; sign must match the coil polarity
GRADIENT_G_PER_CM = 5.0  # radial; axial (beam axis) is twice this

# --- grating chip --------------------------------------------------------
DEFLECTION_ALPHA_RAD = np.deg2rad(46.0)
# DEFLECTION_ALPHA_RAD = np.deg2rad(34.0)
SECTOR_AZIMUTHS_RAD = np.deg2rad([0.0, 120.0, 240.0])
# Power fraction into each sector's used +1 order. Radiation-pressure
# balance requires 3 * eta = 1; other values give no stable trap.
DIFFRACTION_EFFICIENCY = 1.0 / 3.0
# Chip depth below the quadrupole zero. Must exceed the pencil-beam
# stall depth |z| ~ hbar(|delta| + Gamma/2) / (mu_eff * 2b') (~8.5 mm
# here): incoming atoms decelerate to rest at that depth, and only if
# the stall point lies ABOVE the chip - inside the trap volume - are
# they captured instead of falling back down the beamline.
CHIP_Z_M = -5.0e-3
GRATING_HOLE_RADIUS_M = 1.5e-3  # central hole passing the atomic beam

# --- atomic beam ---------------------------------------------------------
OVEN_TEMPERATURE_K = 600.0
ATOM_BEAM_RADIUS_M = 1.0e-3  # 2 mm diameter, aimed up the +z axis
ATOM_BEAM_START_Z_M = -30.0e-3  # launch plane below the chip
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
        above_chip = (rho <= MOT_BEAM_RADIUS_M) & (pos[..., 2] >= CHIP_Z_M)
        # The chip blocks the beam except for the central hole, which
        # transmits a downward pencil beam into the atomic-beam path.
        through_hole = (rho <= GRATING_HOLE_RADIUS_M) & (pos[..., 2] < CHIP_Z_M)
        return (above_chip | through_hole).astype(float)

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
        inside = (
            (t >= 0.0)
            & (rho <= MOT_BEAM_RADIUS_M)
            & (rho >= GRATING_HOLE_RADIUS_M)  # no grating in the hole
            & (np.abs(dphi) <= np.pi / 3.0)
        )
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


def build_oven_beam(v_min: float, v_max: float) -> EffusiveBeam:
    """The effusive Rb85 beam as a reusable `EnsembleSource`.

    A hot oven leaks Rb85 through a small hole up the +z axis, aimed
    through the grating hole at the trap center. `EffusiveBeam` samples
    the flux distribution `f(v) ~ v^3 exp(-v^2/2 sigma^2)` over the slow,
    capturable window `[v_min, v_max]` (its `weight` is the flux fraction
    of that window), with the bare thin-aperture cosine angular law over
    the residual collimation cone.
    """

    return EffusiveBeam(
        temperature_K=OVEN_TEMPERATURE_K,
        mass_kg=SPECIES.mass_kg,
        aperture_radius_m=ATOM_BEAM_RADIUS_M,
        v_min_m_per_s=v_min,
        v_max_m_per_s=v_max,
        direction=(0.0, 0.0, 1.0),
        launch_center_m=(0.0, 0.0, ATOM_BEAM_START_Z_M),
        divergence_half_angle_rad=DIVERGENCE_HALF_ANGLE_RAD,
        angular="cosine",
    )


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

    print(
        "\nAxial deceleration vs climb velocity "
        "(in the pencil beam below the chip, and at the trap center):"
    )
    below = np.array([[0.0, 0.0, -15.0e-3]])
    center = np.zeros((1, 3))
    print("  v_z [m/s]   a_z(-15 mm) [km/s^2]   a_z(0) [km/s^2]")
    for vz in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        v = np.array([[0.0, 0.0, vz]])
        a_below = system.mean_radiation_force(below, v)[0, 2] / SPECIES.mass_kg / 1e3
        a_center = system.mean_radiation_force(center, v)[0, 2] / SPECIES.mass_kg / 1e3
        print(f"  {vz:8.1f}   {a_below:+18.2f}   {a_center:+14.2f}")
    print()


def _beams_figure(system: LightMatterSystem, save: bool) -> None:
    """Map the four beam volumes: counts and total saturation.

    Left: how many beams illuminate each point of the vertical x-z
    plane. Middle: the same count in the horizontal x-y plane at the
    trap height, showing the three sector prisms. Right: total
    saturation in x-z. The atomic beam climbs the +z axis from below,
    through the pencil beam transmitted by the grating hole; full
    radiation-pressure balance exists only where all four beams overlap.
    """

    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sample_plane(h_vals_m, v_vals_m, plane):
        hh, vv = np.meshgrid(h_vals_m, v_vals_m)
        if plane == "xz":
            points = np.stack([hh, np.zeros_like(hh), vv], axis=-1)
        else:  # xy at trap height z = 0
            points = np.stack([hh, vv, np.zeros_like(hh)], axis=-1)
        count = np.zeros(hh.shape)
        total = np.zeros(hh.shape)
        for beam in system.beams:
            s = beam.saturation_at(points)
            count += (s > 1.0e-12).astype(float)
            total += s
        return count, total

    x_mm = np.linspace(-32.0, 32.0, 481)
    z_mm = np.linspace(ATOM_BEAM_START_Z_M * 1e3 - 1.0, 21.0, 361)
    y_mm = np.linspace(-30.0, 30.0, 451)
    count_xz, total_xz = sample_plane(x_mm * 1e-3, z_mm * 1e-3, "xz")
    count_xy, _ = sample_plane(x_mm * 1e-3, y_mm * 1e-3, "xy")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    count_cmap = plt.get_cmap("viridis", 5)

    im0 = axes[0].pcolormesh(
        x_mm, z_mm, count_xz, cmap=count_cmap, vmin=-0.5, vmax=4.5, shading="auto"
    )
    axes[0].axhline(CHIP_Z_M * 1e3, color="k", lw=2.5)
    axes[0].annotate(
        "grating chip", (20.0, CHIP_Z_M * 1e3 - 2.0), color="k", fontsize=9
    )
    axes[0].plot(0.0, 0.0, "r+", ms=12, mew=2)
    axes[0].annotate(
        "",
        xy=(0.0, -14.0),
        xytext=(0.0, -28.0),
        arrowprops=dict(arrowstyle="-|>", color="w", lw=1.8),
    )
    axes[0].annotate("atomic beam\n(through hole)", (2.5, -26.0), color="w", fontsize=9)
    axes[0].annotate(
        "",
        xy=(0.0, 13.0),
        xytext=(0.0, 19.5),
        arrowprops=dict(arrowstyle="-|>", color="#ff5050", lw=2.0),
    )
    axes[0].annotate("incident", (1.0, 16.0), color="#ff5050", fontsize=9)
    alpha_deg = np.rad2deg(DEFLECTION_ALPHA_RAD)
    axes[0].annotate(
        "",
        xy=(-17.0, 6.0),
        xytext=(
            -17.0 + 7.0 * np.sin(DEFLECTION_ALPHA_RAD),
            6.0 - 7.0 * np.cos(DEFLECTION_ALPHA_RAD),
        ),
        arrowprops=dict(arrowstyle="-|>", color="#ff5050", lw=2.0),
    )
    axes[0].annotate(
        f"diffracted\n(alpha = {alpha_deg:.0f} deg)",
        (-31.0, 9.0),
        color="#ff5050",
        fontsize=9,
    )
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm]")
    axes[0].set_title("Beams per point, x-z plane (y = 0)")
    fig.colorbar(im0, ax=axes[0], ticks=[0, 1, 2, 3, 4], label="beam count")

    im1 = axes[1].pcolormesh(
        x_mm, y_mm, count_xy, cmap=count_cmap, vmin=-0.5, vmax=4.5, shading="auto"
    )
    axes[1].plot(0.0, 0.0, "r+", ms=12, mew=2)
    axes[1].annotate("atomic beam\n(out of page)", (2.0, 2.0), color="w", fontsize=9)
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_title("Beams per point, x-y plane (z = 0)")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], ticks=[0, 1, 2, 3, 4], label="beam count")

    im2 = axes[2].pcolormesh(x_mm, z_mm, total_xz, cmap="magma", shading="auto")
    axes[2].axhline(CHIP_Z_M * 1e3, color="w", lw=2.5)
    axes[2].plot(0.0, 0.0, "c+", ms=12, mew=2)
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("z [mm]")
    axes[2].set_title("Total saturation, x-z plane")
    fig.colorbar(im2, ax=axes[2], label="s = I / I_sat (summed)")

    fig.tight_layout()
    if save:
        os.makedirs(RENDER_DIR, exist_ok=True)
        out = os.path.join(RENDER_DIR, "mmwave_mot_beams.png")
        fig.savefig(out, dpi=150)
        print(f"Saved beam-geometry figure to {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=250)
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="lower edge of the simulated velocity window [m/s]",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=55.0,
        help="upper edge of the simulated velocity window [m/s]",
    )
    parser.add_argument("--duration-ms", type=float, default=30.0)
    parser.add_argument("--detuning-gamma", type=float, default=DETUNING_GAMMA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also show the summary figure interactively",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the summary figure (default on; --no-summary to skip)",
    )
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the capture GIF (default on; --no-gif to skip)",
    )
    parser.add_argument(
        "--beams",
        action="store_true",
        help="show the beam-geometry figure and exit (no simulation)",
    )
    parser.add_argument(
        "--save-beams",
        action="store_true",
        help="save the beam-geometry figure into the render/ subdir and exit",
    )
    args = parser.parse_args()

    system = build_system(args.detuning_gamma)
    if args.beams or args.save_beams:
        _beams_figure(system, save=args.save_beams)
        return
    oven_beam = build_oven_beam(args.vmin, args.vmax)
    window_fraction = oven_beam.window_fraction()

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
    balance = 3.0 * DIFFRACTION_EFFICIENCY
    note = "" if abs(balance - 1.0) < 0.05 else "  <-- UNBALANCED, no stable trap"
    print(f"  z-momentum balance  : 3*eta = {balance:.2f} (1.00 = balanced){note}")
    print(
        f"  simulated window    : {args.vmin:.0f} < v < {args.vmax:.0f} m/s = "
        f"{window_fraction:.2e} of the beam flux\n"
    )

    print_diagnostics(system)

    config = SimulationConfig(
        initial_temperature_uK=0.0,  # unused: the effusive-beam source below
        timestep_s=2.0e-7,
        duration_s=float(ms(args.duration_ms)),
        ensemble_size=args.atoms,
        initial_source=oven_beam,
        reject_initially_lost=False,
        loss_radius_m=40.0e-3,
        random_seed=args.seed,
        store_trajectories=True,
        trajectory_stride=500,  # sample every 100 us
    )
    atom_system = AtomSystem(
        species=SPECIES,
        modules=[LightScattering(system, model=AdiabaticSteadyState())],
    )
    result = simulate(atom_system, config)

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

    if args.summary or args.plot:
        _plot(result, captured, save=args.summary, show=args.plot)

    if args.gif:
        _render_gif(result, system)


def _render_gif(result, system: LightMatterSystem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from atommc.postprocess.visualization import render_cloud_animation

    os.makedirs(RENDER_DIR, exist_ok=True)
    out = os.path.join(RENDER_DIR, "mmwave_mot_capture.gif")
    render_cloud_animation(
        result,
        out,
        mass_kg=system.species.mass_kg,
        plane="xz",
        beams=system.beams,
        # Footprint shading, not edge arrows: the diffracted beams
        # emanate from the grating prisms and the incident beam narrows
        # to a through-hole pencil below the chip, so origin-centered
        # arrows would misplace them. The overlay draws the real slices.
        beam_overlay=True,
        doppler_limit_uK=system.species.doppler_temperature_uK,
        # Asymmetric window: keep the grating chip and the atomic-beam
        # climb from below in view rather than auto-centering.
        extent_mm=(-22.0, 22.0, ATOM_BEAM_START_Z_M * 1e3 - 1.0, 20.0),
        n_frames=80,
        fps=16,
        dpi=85,
    )
    print(f"\nSaved GIF to {out}")


def _plot(result, captured, save: bool, show: bool = False) -> None:
    import matplotlib

    if save and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # x-z trajectories with the chip sketch. The chip is not a physical
    # barrier in the model, so clip an atom's trace once it is below the
    # surface while outside the hole (i.e. once it has hit the chip).
    traj_mm = result.trajectory_positions_m.copy() * 1e3
    rho_mm = np.hypot(traj_mm[:, :, 0], traj_mm[:, :, 1])
    hit_chip = (traj_mm[:, :, 2] < CHIP_Z_M * 1e3) & (
        rho_mm > GRATING_HOLE_RADIUS_M * 1e3
    )
    traj_mm[np.cumsum(hit_chip, axis=0) > 0] = np.nan
    n_atoms = traj_mm.shape[1]
    for atom in range(n_atoms):
        color = "C1" if captured[atom] else "C0"
        alpha = 0.9 if captured[atom] else 0.15
        axes[0].plot(
            traj_mm[:, atom, 0], traj_mm[:, atom, 2], lw=0.6, color=color, alpha=alpha
        )
    hole_mm = GRATING_HOLE_RADIUS_M * 1e3
    axes[0].plot([-22, -hole_mm], [CHIP_Z_M * 1e3] * 2, "k-", lw=2)
    axes[0].plot([hole_mm, 22], [CHIP_Z_M * 1e3] * 2, "k-", lw=2)
    axes[0].plot(0.0, 0.0, "r+", ms=10)
    axes[0].set_xlim(-22, 22)
    axes[0].set_ylim(ATOM_BEAM_START_Z_M * 1e3 - 1.0, 20)
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
        os.makedirs(RENDER_DIR, exist_ok=True)
        out = os.path.join(RENDER_DIR, "mmwave_mot_summary.png")
        fig.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
