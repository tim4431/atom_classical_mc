"""Rb85 six-beam MOT with m_F-resolved hyperfine internal states.

Three demonstrations of the `HyperfineScattering` module against the
effective two-level `LightScattering` treatment:

1.  **Optical pumping transient** — an atom held at rest inside the MOT
    light sees its 12 ground-sublevel populations redistribute on a
    ~10 us timescale: the local sigma+/sigma- imbalance polarizes the
    atom toward the stretched states of F=3, while off-resonant
    excitation of F'=3 slowly leaks population into the dark F=2 level.
2.  **Dark states vs repumping** — the same atom followed for 1 ms with
    the repumper on and off. Without a repumper the leak (F'=3 is only
    ~20 Gamma below F'=4) empties the cooling manifold in ~0.1 ms and
    photon scattering stops; a weak repumper (F=2 -> F'=3, ~2.9 GHz
    blue of the cooling light) closes the cycle.
3.  **Full MOT Monte Carlo** — three `simulate()` runs of a 6-beam MOT
    on a 300 uK cloud: the two-level backend, the hyperfine backend
    with a repumper, and the hyperfine backend without one. With the
    repumper the m_F-resolved MOT cools like the two-level model
    predicts; without it the cloud goes dark and stops cooling — physics
    the two-level model cannot express.

Run from the repository root:

    python3 example/mot/hyperfine_mot.py             # all demos + figure
    python3 example/mot/hyperfine_mot.py --atoms 50 --duration-ms 10
    python3 example/mot/hyperfine_mot.py --no-mot    # skip the slow MC part
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from atommc import (  # noqa: E402
    AtomSystem,
    BOLTZMANN_CONSTANT_J_PER_K,
    HyperfineScattering,
    LaserBeam,
    LightMatterSystem,
    LightScattering,
    QuadrupoleMagneticField,
    RB85_D2_HFS,
    SimulationConfig,
    gauss_per_cm,
    ms,
    simulate,
    six_beam_mot,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")

HF = RB85_D2_HFS
SPECIES = HF.base
GAMMA_HZ = SPECIES.linewidth_rad_s / (2.0 * np.pi)

DETUNING_GAMMA = -1.5
SATURATION_PER_BEAM = 1.5
REPUMP_SATURATION = 0.3
GRADIENT_G_PER_CM = 10.0


def build_light(with_repump: bool) -> LightMatterSystem:
    beams = six_beam_mot(
        detuning_hz=DETUNING_GAMMA * GAMMA_HZ, saturation=SATURATION_PER_BEAM
    )
    if with_repump:
        # Repump light on F=2 -> F'=3 (~2.915 GHz blue of the cooling
        # line), split into a counter-propagating pair so its net push
        # vanishes. Atoms spend almost no time in F=2, so its recoil
        # contribution is tiny either way.
        offset = HF.transition_offset_hz(2, 3)
        for direction in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):
            beams.append(
                LaserBeam(
                    direction=direction,
                    detuning_hz=offset,
                    saturation=REPUMP_SATURATION / 2.0,
                    helicity=+1.0,
                    name=f"repump_{'up' if direction[2] > 0 else 'down'}",
                )
            )
    quadrupole = QuadrupoleMagneticField(
        gradient_T_per_m=float(gauss_per_cm(GRADIENT_G_PER_CM)),
        axis=(0.0, 0.0, 1.0),
    )
    return LightMatterSystem(
        species=SPECIES, beams=beams, magnetic_fields=[quadrupole]
    )


def evolve_static_atom(
    light: LightMatterSystem,
    position_m: np.ndarray,
    n_steps: int,
    dt_s: float,
    seed: int = 0,
):
    """Step one atom at rest; return times, populations, photon count."""

    process = HyperfineScattering(
        light=light, hyperfine=HF, initial_populations="cooling-uniform"
    )
    rng = np.random.default_rng(seed)
    state = process.initialize(1, rng)
    positions = position_m.reshape(1, 3)
    velocities = np.zeros((1, 3))

    times = np.empty(n_steps + 1)
    populations = np.empty((n_steps + 1, HF.n_levels))
    photons = np.zeros(n_steps + 1)
    times[0], populations[0] = 0.0, state[0]
    for step in range(1, n_steps + 1):
        result = process.step(
            state, positions, velocities, times[step - 1], dt_s, rng
        )
        state = result.state
        times[step] = times[step - 1] + dt_s
        populations[step] = state[0]
        photons[step] = photons[step - 1] + float(
            result.diagnostics["scattered_photons"][0]
        )
    return times, populations, photons


def run_mot(process_kind: str, args) -> dict:
    """One MOT Monte Carlo run; returns summary + temperature series."""

    with_repump = process_kind != "hyperfine, no repump"
    light = build_light(with_repump=with_repump)
    if process_kind == "two-level":
        module = LightScattering(light)
    else:
        module = HyperfineScattering(light=light, hyperfine=HF)
    system = AtomSystem(species=SPECIES, modules=[module])
    config = SimulationConfig(
        initial_temperature_uK=300.0,
        timestep_s=2.5e-7,
        duration_s=float(ms(args.duration_ms)),
        ensemble_size=args.atoms,
        initial_cloud_sigma_m=(1.0e-3, 1.0e-3, 1.0e-3),
        loss_radius_m=2.0e-2,
        random_seed=args.seed,
        store_trajectories=True,
        trajectory_stride=400,  # sample every 100 us
    )
    result = simulate(system, config)

    velocities = result.trajectory_velocities_m_per_s
    temp_uK = (
        SPECIES.mass_kg
        * np.mean(np.sum(velocities**2, axis=-1), axis=1)
        / 3.0
        / BOLTZMANN_CONSTANT_J_PER_K
        * 1e6
    )
    return {
        "label": process_kind,
        "times_ms": result.trajectory_times_s * 1e3,
        "temperature_uK": temp_uK,
        "photons": float(np.mean(result.scattered_photons)),
        "survival": float(result.survival_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=30)
    parser.add_argument("--duration-ms", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the Monte Carlo comparison (default on; --no-mot to skip)",
    )
    parser.add_argument("--plot", action="store_true", help="show interactively")
    args = parser.parse_args()

    print("Rb85 hyperfine MOT demo")
    print(f"  cooling  : 6 beams, {DETUNING_GAMMA:+.1f} Gamma, "
          f"s = {SATURATION_PER_BEAM} per beam")
    print(f"  repump   : F=2 -> F'=3, offset {HF.transition_offset_hz(2, 3)/1e9:+.3f} "
          f"GHz, s = {REPUMP_SATURATION}")
    print(f"  gradient : {GRADIENT_G_PER_CM:.0f} G/cm radial")
    print(f"  levels   : {HF.n_ground} ground + {HF.n_excited} excited sublevels, "
          f"{HF.n_transitions} dipole-allowed transitions\n")

    # Demo 1: pumping transient, atom at rest 2 mm off-axis (B = 2 G).
    probe = np.array([2.0e-3, 0.0, 0.0])
    t1, pop1, _ = evolve_static_atom(
        build_light(with_repump=True), probe, n_steps=5000, dt_s=2.0e-8
    )

    # Demo 2: dark-state accumulation vs repumping, 1 ms.
    t2, pop2_dark, phot_dark = evolve_static_atom(
        build_light(with_repump=False), probe, n_steps=10000, dt_s=1.0e-7
    )
    _, pop2_pump, phot_pump = evolve_static_atom(
        build_light(with_repump=True), probe, n_steps=10000, dt_s=1.0e-7
    )
    f2_idx = np.flatnonzero(HF.ground_f == 2)
    dark_f2 = pop2_dark[:, f2_idx].sum(axis=1)
    print("Static-atom demos (atom held at x = 2 mm)")
    print(f"  F=2 population after 1 ms, no repump : {dark_f2[-1]:.3f}")
    print(f"  F=2 population after 1 ms, repumped  : "
          f"{pop2_pump[:, f2_idx].sum(axis=1)[-1]:.3f}")
    print(f"  photons in 1 ms   no repump / repump : "
          f"{phot_dark[-1]:.0f} / {phot_pump[-1]:.0f}\n")

    runs = []
    if args.mot:
        for kind in ("two-level", "hyperfine + repump", "hyperfine, no repump"):
            print(f"MOT Monte Carlo: {kind} ...")
            runs.append(run_mot(kind, args))
        print("\n  configuration           T_final    survival   photons/atom")
        for run in runs:
            print(
                f"  {run['label']:<22s}  {run['temperature_uK'][-1]:7.0f} uK"
                f"   {run['survival']:8.2f}   {run['photons']:11.0f}"
            )

    _plot(t1, pop1, t2, dark_f2, pop2_pump[:, f2_idx].sum(axis=1),
          phot_dark, phot_pump, runs, show=args.plot)


def _plot(t1, pop1, t2, dark_f2, pump_f2, phot_dark, phot_pump, runs, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if runs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))

    # Panel 1: every F=3 sublevel + F=2 and excited totals, first 100 us.
    ax = axes[0]
    t_us = t1 * 1e6
    f3_idx = np.flatnonzero(HF.ground_f == 3)
    colors = plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, f3_idx.size))
    for color, idx in zip(colors, f3_idx):
        m = int(HF.ground_m[idx])
        ax.plot(t_us, pop1[:, idx], color=color, lw=1.4, label=f"$m_F={m:+d}$")
    f2_idx = np.flatnonzero(HF.ground_f == 2)
    ax.plot(t_us, pop1[:, f2_idx].sum(axis=1), "k--", lw=1.6, label="F=2 (dark)")
    ax.plot(t_us, pop1[:, HF.n_ground:].sum(axis=1), "k:", lw=1.6, label="excited")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("population")
    ax.set_title("Optical pumping of the F=3 sublevels")
    ax.legend(fontsize=7, ncol=2)

    # Panel 2: dark-state accumulation vs repumping.
    ax = axes[1]
    t_ms = t2 * 1e3
    ax.plot(t_ms, dark_f2, "C3", lw=1.8, label="F=2, no repump")
    ax.plot(t_ms, pump_f2, "C0", lw=1.8, label="F=2, repumped")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("dark-level population")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="center right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t_ms, phot_dark / 1e3, "C3", ls=":", lw=1.2)
    ax2.plot(t_ms, phot_pump / 1e3, "C0", ls=":", lw=1.2)
    ax2.set_ylabel("scattered photons [x1000] (dotted)")
    ax.set_title("Dark state fills, scattering stops")

    # Panel 3: MOT cooling comparison.
    if runs:
        ax = axes[2]
        for run, style in zip(runs, ("C0-", "C1-", "C3--")):
            ax.plot(run["times_ms"], run["temperature_uK"], style, lw=1.8,
                    label=run["label"])
        ax.axhline(SPECIES.doppler_temperature_uK, color="gray", lw=0.8, ls=":")
        ax.annotate("Doppler limit", (0.4, SPECIES.doppler_temperature_uK * 1.08),
                    color="gray", fontsize=8)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("kinetic temperature [uK]")
        ax.set_title("MOT cooling: backends compared")
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(RENDER_DIR, exist_ok=True)
    out = os.path.join(RENDER_DIR, "hyperfine_mot.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
