"""SLM-to-AOD handoff with vs. without AOD lensing.

Translation of `tmp/aod_slm_movement_v2/aod_slm_movement_v2/main_lensing.ipynb`.
The SLM is on, the AOD ramps on at the origin, drags the atom out by 50 um,
holds, drags it back, and ramps off; the SLM goes off during the AOD hold so
the atom briefly sits in the AOD only. The simulation is run twice: once with
the v2 lensing coefficient `dxdt2z = f0 / V_sound` and once with `dxdt2z = 0`
to isolate the velocity-coupled focal-shift heating.

Unlike `slm_to_aod_transfer.py` (a gentle one-way Rb87 handoff, where the
peak drag velocity is so small that the lensing shift would be ~1e-5 Rayleigh
lengths), this drag is fast enough that the focal shift is comparable to the
Rayleigh length and dominates the heating.

Run from the repository root:

    python3 example/aod/slm_to_aod_with_lensing.py

Saves a comparison plot into the render/ subdir next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from atommc import (  # noqa: E402
    AstigmaticAODTrap,
    AtomSpecies,
    AtomSystem,
    RampSequence,
    SimulationConfig,
    arb_fifth_poly,
    simulate,
)
from atommc.constants import (  # noqa: E402
    ATOMIC_MASS_UNIT_KG,
    BOLTZMANN_CONSTANT_J_PER_K,
    PLANCK_CONSTANT_J_S,
)
from atommc.postprocess.analysis import (  # noqa: E402
    bound_to_trap,
    single_trap_energy_uK,
)

HERE = os.path.dirname(__file__)
RENDER_DIR = os.path.join(HERE, "render")
os.makedirs(RENDER_DIR, exist_ok=True)

# v2 used Yb-171 trap parameters; keep that mass so the dynamics match. The
# run is dark (purely conservative), so only `mass_kg` is read from the
# species; the cycling-transition fields (Yb 1S0 -> 1P1) are required by
# `AtomSpecies` but never used here.
YB171 = AtomSpecies(
    name="Yb171 (mass only)",
    mass_kg=171.0 * ATOMIC_MASS_UNIT_KG,
    wavelength_m=398.9e-9,
    linewidth_rad_s=2.0 * np.pi * 29.1e6,
    saturation_intensity_w_per_m2=600.0,
    g_ground=0.0,
    g_excited=1.0,
    f_ground=0.5,
    f_excited=1.5,
)


def hz_to_uK(depth_hz: float) -> float:
    """Convert a trap depth in Hz (`U/h`) to the equivalent microkelvin."""

    return depth_hz * PLANCK_CONSTANT_J_S / BOLTZMANN_CONSTANT_J_PER_K / 1.0e-6


# --- Trap parameters, lifted from main_lensing.ipynb cell 2 -----------------
T_RAMP_S = 0.61e-3
T_DRAG_S = 0.25e-3
T_HOLD_S = 0.15e-3
DRAG_DISTANCE_M = 50.0e-6

SLM_WAIST_M = 440.0e-9
SLM_DEPTH_UK = hz_to_uK(1.4e6)
SLM_WAVELENGTH_M = 486.0e-9

AOD_WAIST_M = 500.0e-9
AOD_DEPTH_UK = hz_to_uK(6.0e6)
AOD_WAVELENGTH_M = 532.0e-9

# Beta in arb_fifth_poly(beta) was chosen as 1.5625 by the v2 author. Note
# that beta = 15/8 = 1.875 reduces to true min-jerk; 1.5625 is a slightly
# more aggressive quintic, used in the v2 work as the baseline.
BETA = 1.5625
DXDT2Z = 0.025 / 4.0 / 650.0  # m / (m/s); AOD f0 / V_sound, see v2 comment.

# --- Numerics ---------------------------------------------------------------
INITIAL_TEMPERATURE_UK = 10.0
ENSEMBLE_SIZE = 500
TIMESTEP_S = 1.0e-6
DURATION_S = 2.0e-3
RANDOM_SEED = 42
LOSS_RADIUS_M = 200.0e-6

# v2 toggles the SLM with dt = 0 (truly instantaneous step). We track loss
# per-step in the *current* total potential, so an instantaneous SLM step
# would inject ~SLM_DEPTH_UK of "energy" into the bookkeeping in one step
# and falsely flag everyone as lost. 50 us is short compared to the ramp
# durations (>250 us) but slow enough that the depth transition is
# adiabatic at the SLM trap frequency (~40 kHz, period ~25 us).
SLM_TRANSITION_S = 50.0e-6


def build_slm_ramp() -> RampSequence:
    t_first_off = T_RAMP_S + T_DRAG_S
    t_back_on = t_first_off + T_HOLD_S
    t_end = t_back_on + T_RAMP_S + T_DRAG_S
    times_s = [
        0.0,
        t_first_off,
        t_first_off + SLM_TRANSITION_S,
        t_back_on,
        t_back_on + SLM_TRANSITION_S,
        t_end,
    ]
    centers_m = [[0.0, 0.0, 0.0]] * 6
    depths_uK = [SLM_DEPTH_UK, SLM_DEPTH_UK, 0.0, 0.0, SLM_DEPTH_UK, SLM_DEPTH_UK]
    return RampSequence(
        times_s=np.asarray(times_s, dtype=float),
        centers_m=np.asarray(centers_m, dtype=float),
        depths_uK=np.asarray(depths_uK, dtype=float),
        position_profile=arb_fifth_poly(BETA),
    )


def build_aod_ramp() -> RampSequence:
    t_after_load = T_RAMP_S
    t_after_drag_out = t_after_load + T_DRAG_S
    t_after_hold = t_after_drag_out + T_HOLD_S
    t_after_drag_back = t_after_hold + T_DRAG_S
    t_end = t_after_drag_back + T_RAMP_S
    times_s = [0.0, t_after_load, t_after_drag_out, t_after_hold, t_after_drag_back, t_end]
    centers_m = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [DRAG_DISTANCE_M, 0.0, 0.0],
        [DRAG_DISTANCE_M, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    depths_uK = [0.0, AOD_DEPTH_UK, AOD_DEPTH_UK, AOD_DEPTH_UK, AOD_DEPTH_UK, 0.0]
    return RampSequence(
        times_s=np.asarray(times_s, dtype=float),
        centers_m=np.asarray(centers_m, dtype=float),
        depths_uK=np.asarray(depths_uK, dtype=float),
        position_profile=arb_fifth_poly(BETA),
    )


def build_traps(slm_ramp: RampSequence, aod_ramp: RampSequence, dxdt2z: float):
    slm = AstigmaticAODTrap(
        waist_radial_m=SLM_WAIST_M,
        wavelength_m=SLM_WAVELENGTH_M,
        ramp=slm_ramp,
        dxdt2z=0.0,  # SLM doesn't move so the velocity-coupled shift is zero anyway.
        name="SLM",
    )
    aod = AstigmaticAODTrap(
        waist_radial_m=AOD_WAIST_M,
        wavelength_m=AOD_WAVELENGTH_M,
        ramp=aod_ramp,
        dxdt2z=dxdt2z,
        name="AOD",
    )
    return [slm, aod]


def build_config() -> SimulationConfig:
    # During the AOD drag, peak trap velocity ~ 0.31 m/s exceeds the
    # cylindrical escape velocity sqrt(2*U0/m) ~ 0.17 m/s, so the lab-frame
    # `KE + V >= 0` loss check would flag every co-moving atom as lost.
    # Disable it; survival is recovered post-hoc against the SLM at t = 0
    # (where everything is at rest).
    return SimulationConfig(
        initial_temperature_uK=INITIAL_TEMPERATURE_UK,
        timestep_s=TIMESTEP_S,
        duration_s=DURATION_S,
        ensemble_size=ENSEMBLE_SIZE,
        random_seed=RANDOM_SEED,
        loss_radius_m=LOSS_RADIUS_M,
        energy_loss="off",
    )


def run_one(label: str, dxdt2z: float):
    slm_ramp = build_slm_ramp()
    aod_ramp = build_aod_ramp()
    traps = build_traps(slm_ramp, aod_ramp, dxdt2z)
    slm_trap = traps[0]
    config = build_config()
    system = AtomSystem(species=YB171, modules=traps)
    result = simulate(system, config)

    # v2-style post-hoc survival: an atom is bound iff
    # KE_final + V_SLM(r_final, t=0) < 0, evaluated where the SLM is on at
    # full depth and at rest (which is also the case at t = duration here).
    init_e_uK = single_trap_energy_uK(
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
        slm_trap,
        mass_kg=YB171.mass_kg,
        time_s=0.0,
    )
    final_e_uK = single_trap_energy_uK(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        slm_trap,
        mass_kg=YB171.mass_kg,
        time_s=0.0,
    )
    bound_initial = bound_to_trap(
        result.initial_positions_m,
        result.initial_velocities_m_per_s,
        slm_trap,
        mass_kg=YB171.mass_kg,
        time_s=0.0,
    )
    bound_final = bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        slm_trap,
        mass_kg=YB171.mass_kg,
        time_s=0.0,
    )
    n_init = int(np.sum(bound_initial))
    n_final = int(np.sum(bound_final & ~result.lost))
    posthoc_survival = n_final / max(n_init, 1)
    print(
        f"  {label:<14}  posthoc survival = {n_final}/{n_init} = "
        f"{posthoc_survival:.3f}  "
        f"<dE_SLM>(survivors) = "
        f"{float(np.mean(final_e_uK[bound_final & ~result.lost]) - np.mean(init_e_uK[bound_initial])):+7.3f} uK"
    )
    return {
        "result": result,
        "init_e_uK": init_e_uK,
        "final_e_uK": final_e_uK,
        "bound_initial": bound_initial,
        "bound_final": bound_final,
        "posthoc_survival": posthoc_survival,
    }


def plot_comparison(no_lens, with_lens):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    bin_edges = np.linspace(-90.0, 50.0, 35)
    axes[0].hist(
        with_lens["init_e_uK"][with_lens["bound_initial"]],
        bins=bin_edges, alpha=0.55, label="initial (lensing on)",
    )
    axes[0].hist(
        with_lens["final_e_uK"][with_lens["bound_final"]],
        bins=bin_edges, alpha=0.55, label="final (lensing on)",
    )
    axes[0].set_xlabel("E in SLM at t = 0 [uK]")
    axes[0].set_ylabel("counts (bound atoms)")
    axes[0].set_title("Lensing on: SLM-frame energy")
    axes[0].legend(loc="upper left")

    axes[1].hist(
        no_lens["final_e_uK"][no_lens["bound_final"]],
        bins=bin_edges, alpha=0.55, label=f"lensing off  surv={no_lens['posthoc_survival']:.2f}",
    )
    axes[1].hist(
        with_lens["final_e_uK"][with_lens["bound_final"]],
        bins=bin_edges, alpha=0.55, label=f"lensing on   surv={with_lens['posthoc_survival']:.2f}",
    )
    axes[1].set_xlabel("Final E in SLM at t = 0 [uK]")
    axes[1].set_ylabel("counts (bound survivors)")
    axes[1].set_title("Survivor final energy: with vs without lensing")
    axes[1].legend(loc="upper left")

    return figure, axes


def main() -> None:
    print("SLM-to-AOD handoff with vs. without lensing")
    print(
        f"  SLM: waist={SLM_WAIST_M*1e9:.0f} nm, depth={SLM_DEPTH_UK:.2f} uK, "
        f"lambda={SLM_WAVELENGTH_M*1e9:.0f} nm"
    )
    print(
        f"  AOD: waist={AOD_WAIST_M*1e9:.0f} nm, depth={AOD_DEPTH_UK:.2f} uK, "
        f"lambda={AOD_WAVELENGTH_M*1e9:.0f} nm"
    )
    print(
        f"  ramp: t_load={T_RAMP_S*1e3:.2f} ms, t_drag={T_DRAG_S*1e3:.2f} ms, "
        f"t_hold={T_HOLD_S*1e3:.2f} ms, drag_distance={DRAG_DISTANCE_M*1e6:.0f} um"
    )
    print(f"  beta = {BETA}, dxdt2z = {DXDT2Z:.6e} (m / (m/s))")
    print(
        f"  ensemble = {ENSEMBLE_SIZE}, T = {INITIAL_TEMPERATURE_UK:.1f} uK, "
        f"dt = {TIMESTEP_S*1e6:.1f} us, duration = {DURATION_S*1e3:.2f} ms"
    )

    no_lens = run_one("no lensing", 0.0)
    with_lens = run_one("with lensing", DXDT2Z)

    figure, _ = plot_comparison(no_lens, with_lens)
    out_path = os.path.join(RENDER_DIR, "slm_to_aod_with_lensing.png")
    figure.savefig(out_path, dpi=180)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
