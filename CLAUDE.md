# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Classical Monte Carlo simulator for Rb87 atom heating and loss when a moving AOD optical tweezer pulls an atom out of a static SLM trap, plus a general near-resonant light-force extension (laser beams + magnetic fields + internal-state dynamics + photon recoil for Rb85/Rb87; a MOT is one configuration of it, not a special case). Pure NumPy; Matplotlib is optional. See `doc/plan.md` for the tweezer physics model and `doc/light_matter_plan.md` for the light-force model; `src/README.md` is a one-line index of every function in the package.

## Commands

Run from the repository root.

- Tests: `python3 -m unittest tests.test_core` and `python3 -m unittest tests.test_light_matter` (or `python3 -m unittest discover tests`; the light-matter module takes ~1.5 min — it runs real cooling simulations)
- Single test: `python3 -m unittest tests.test_core.CorePhysicsTests.test_static_trap_has_low_numerical_heating`
- SLM-to-AOD example: `python3 example/slm_to_aod_transfer.py` (add `--plot` / `--save-plot` for 2D figures, `--plot-3d` / `--save-3d-plot` for 3D, `--compare-position-ramps --save-comparison-plot ...` to sweep ramp shapes)
- Other examples: `example/slm_to_slm_transfer.py`, `example/spectator/spectator_qubits.py` (AOD fly-by), `example/spectator/spectator_qubits_ripa.py` (gridded RIPA fly-by), `example/spectator/ripa/generate_ripa_gridded_trap.py` (RIPA trap generator), `example/position_ramp_compare.py`, `example/gridded_vs_analytical.py`. See [example/spectator/README.md](example/spectator/README.md) for the spectator-qubit suite.
- Light-force examples: `python3 example/mot/rb85_mot.py` (Rb85 MOT capture + cooling; `--backend rate-equation` to switch internal-state backends, `--plot` / `--save-plot` for figures, `--gif` for an animated cloud GIF), `python3 example/mot/mmwave_mot.py` (tri-sector grating MOT loading Rb85 from a 600 K effusive beam; sector-prism beam profiles via `LaserBeam.profile`, explicit flux-distribution initial ensemble), `python3 example/tweezer_probe_heating.py` (recoil heating of a tweezer-trapped atom, mixed conservative + light physics)
- Install deps: `pip install -e .` (core: `numpy>=1.23`); for plotting `pip install -e ".[viz]"` (adds `matplotlib>=3.7`).

There is no lint/format config. Python ≥3.9.

## Import layout (non-standard)

`pyproject.toml` declares `packages = ["src"]`, so the package literally is named `src`. Imports inside the library are relative (`from .trap import ...`); call sites use `from src.trap import ...`. Tests and examples bootstrap this by inserting the repo root onto `sys.path` before importing — preserve that pattern when adding new entry points. Don't rename `src/` to the project name without also fixing every import.

## Architecture

A single-pass pipeline driven by `run_simulation` in [src/simulation.py](src/simulation.py):

1. **Geometry** — `TrapConfig` ([src/trap.py](src/trap.py)) is the abstract interface for any time-dependent potential `U(r, t)`. Concrete implementations: `GaussianTrap` (cylindrical, time-independent, the v1 trap) with analytic `potential` / `force` / `hessian`; `MovingGaussianTrap` (cylindrical, time-dependent via a `RampSequence`); `AstigmaticAODTrap` (full astigmatic Gaussian with velocity-coupled focal lensing — `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`, modeling AOD lensing per `tmp/aod_slm_movement_v2/`). Multiple traps sum via `total_potential` / `total_force` / `total_hessian` at a chosen time.
2. **Time dependence** — `RampSequence` ([src/ramp.py](src/ramp.py)) is a waypoint table over `(times_s, centers_m, depths_uK)` plus a `position_profile` and `depth_profile` from the `PolynomialConnector` family (`LINEAR`, `CUBIC_SMOOTHSTEP`, `QUINTIC_MIN_JERK`, `arb_fifth_poly(beta)`, `const_jerk()`). `ramp.at(t)` clamps outside the table; `ramp.velocity_at(t)` exposes `dcenter/dt`, used by `AstigmaticAODTrap` to compute the focal shift.
3. **Initial ensemble** — [src/sampling.py](src/sampling.py): velocities from Maxwell-Boltzmann; positions from a Gaussian whose covariance is `k_B T · K⁻¹` where `K` is the combined-trap Hessian at the chosen center (deepest trap at `t=0` by default, or `SimulationConfig.initial_center_m`). With `reject_initially_lost=True` (default) the simulator resamples atoms that are unbound at `t=0` so reported loss reflects ramp dynamics, not bad loading. The rejection budget is bounded by `max_initial_resampling_rounds`; exceeding it raises `RuntimeError`.
4. **Propagation** — velocity-Verlet, vectorized over the ensemble. Lost atoms are masked out and not advanced further. An atom is marked lost when its instantaneous mechanical energy in the *current* total potential is ≥ 0, or when it leaves the optional spherical `loss_radius_m`. Once lost, always lost. Each trap is queried with `time_s` set to the appropriate half-step time, so moving and astigmatic traps update consistently. The lab-frame energy criterion fails when a trap moves faster than its escape velocity `sqrt(2*U0/m)` — a perfectly riding atom carries the trap's lab-frame KE and registers as unbound. For those regimes, set `SimulationConfig.track_energy_loss=False` to disable the energy check (relying on `loss_radius_m` only) and recover survival post-hoc with `bound_to_trap`/`capture_probability` evaluated at a time where traps are at rest.
5. **Reporting** — `SimulationResult` carries survival/loss, energy-gain stats, kinetic temperature, full final state, and (if `store_trajectories=True`) sampled `(positions, velocities, lost)` snapshots at `trajectory_stride` intervals. Temperatures come in two flavors: `*_survivors` (over atoms not flagged lost) and `*_all` (whole ensemble); use `result.temperature_gain_uK_at(survivors_only=True|False)` to pick. The legacy properties `final_temperature_uK` and `temperature_gain_uK` keep the survivors-only meaning.
6. **Post-processing** — [src/analysis.py](src/analysis.py) computes single-trap binding (`bound_to_trap`, `capture_probability`) at any time `t` (defaults to `result.duration_s` for moving traps), the SLM-vs-AOD-vs-ambiguous breakdown (`classify_final_trap_occupation`), and time series from stored trajectories. [src/harmonic.py](src/harmonic.py) builds a quadratic Taylor expansion (analytic from `TrapConfig`s at a chosen `time_s`, or finite-difference from any callable), diagonalizes `K/m` to get normal modes, and decomposes phase-space coordinates into per-mode classical energies and coherent-state occupations `n̄ = E/(ℏω)` (a semi-classical proxy — a classical trajectory is not a quantum eigenstate).

## Light-force extension (scattering)

Not a separate pipeline: `run_simulation` accepts an optional `scattering=LightMatterSystem(...)` (plus `internal_model=...`) and then couples two "physics backends" by operator splitting each timestep (see `doc/light_matter_plan.md` for the full model). A MOT, an optical molasses, a probe/imaging beam, or resonant light on top of tweezers are all just different beam/field sets.

1. **Field geometry** — `LightMatterSystem` ([src/light_matter.py](src/light_matter.py)) bundles an `AtomSpecies` ([src/species.py](src/species.py), presets `RB85_D2` / `RB87_D2`), `LaserBeam`s ([src/laser.py](src/laser.py), `six_beam_mot` builder for the standard MOT geometry), and `MagneticFieldConfig`s ([src/fields.py](src/fields.py), uniform + quadrupole). Its `stimulated_rates` reduces all geometry (Gaussian beam profiles, Doppler shift, Zeeman shift with sigma+/pi/sigma- polarization decomposition along the local B axis) to a per-atom, per-beam one-way stimulated rate matrix `W` — the only thing the internal-state backend sees.
2. **Internal-state backend** — `InternalStateModel` ([src/internal_state.py](src/internal_state.py)) evolves the effective two-level populations and reports per-step photon `ScatteringEvents`. Three implementations: `AdiabaticSteadyState` (default, populations at local steady state, Poisson photon counts), `RateEquationPopulations` (exact exponential ODE update, handles pulsed beams, unconditionally stable), `StochasticJumpState` (discrete kinetic-MC trajectory, requires `(Gamma + W_tot) dt < 0.1`). The state array is opaque to the driver — richer (multilevel/MCWF) backends can be added without touching the loop.
3. **Momentum update** — the `run_simulation` loop applies recoil velocity impulses at the start of each velocity-Verlet step (`+hbar k` per absorption along the beam, `-hbar k` per stimulated emission, isotropic random `hbar k` per spontaneous photon, sampled photon-by-photon) on top of the conservative `TrapConfig` forces.

Interaction with the conservative pipeline: attaching `scattering` auto-disables the energy-based loss criterion (radiation pressure is non-conservative; only `loss_radius_m` applies — recover trap survival post-hoc with `analysis.bound_to_trap`); `config.mass_kg` must equal `scattering.species.mass_kg` (validated, since the default is Rb87); `traps` may be empty, in which case `config.initial_cloud_sigma_m` must define the initial Gaussian cloud (`initial_mean_velocity_m_per_s` adds a launch drift). `SimulationResult` gains `scattered_photons` and `final_excited_fraction` (both `None` on dark runs).

Key limitations (documented in `doc/light_matter_plan.md`): beams are mutually incoherent, so no sub-Doppler cooling; effective two-level cycling transition only (perfect repumper assumed); no dipole force from the near-resonant light.

## Unit conventions

The simulator is internally SI (m, s, kg, J), but inputs and human-readable outputs use:

- **Trap depth**: microkelvin (`depth_uK`). Convert with `units.microkelvin_to_joule` / `joule_to_microkelvin` (or via `TrapConfig.depth_joule`).
- **Lengths**: meters in API, but the `units.um(...)` and `units.ms(...)` helpers exist so example/test code can write `um(1.2)` for a waist or `ms(0.5)` for a duration. For MOT inputs there are also `units.mhz(...)`, `units.gauss(...)`, and `units.gauss_per_cm(...)`.
- **Magnetic fields**: tesla (`_T`, `_T_per_m`). **Frequencies**: linear hertz for laser detunings (`detuning_hz`, negative = red), angular rad/s for linewidths and internal rates (`_rad_s`, `_per_s` suffixes).
- **Energies in results**: microkelvin (`*_uK` suffix). Time series come back in joules only inside `harmonic.py` hot paths.

Any new field that carries a physical quantity should follow the `_m` / `_s` / `_kg` / `_uK` / `_j_per_m2` suffix scheme already used everywhere — search `_per_s2` etc. before inventing a new name.

## Trap axial scale

`GaussianTrap` takes `waist_axial_m` directly — there is no `rayleigh_length_m` field. The Gaussian axial 1/e² scale is decoupled from the diffraction limit; if you need the proper relation `zR = π w₀² / λ`, use `AstigmaticAODTrap`, which computes `rayleigh_length_m` internally from `waist_radial_m` and `wavelength_m`.

## Astigmatic AOD lensing

`AstigmaticAODTrap` reproduces the v2 model where AOD lateral motion shifts the axial focus per axis: `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`. `dxdt2z = f0 / V_sound` is the AOD acoustic constant. Setting `dxdt2z = 0` collapses to the diffraction Gaussian-beam profile (still time-dependent through the ramp). The trap velocity comes from `ramp.velocity_at(t)`, so it depends on the ramp's `position_profile` — `LINEAR` ramps give piecewise-constant velocity (and hence step changes in focal shift at every waypoint), so prefer a smooth profile (e.g. `QUINTIC_MIN_JERK`) when running with `dxdt2z != 0`.

## Out of scope

For purely conservative runs (per `doc/plan.md`): no photon recoil, tunneling, background-gas collisions, technical noise, cooling/damping, optical-power-to-depth conversion, or CLI. Photon recoil and cooling **are** modeled when a `scattering` system is attached, but there the exclusions are: sub-Doppler mechanisms, multilevel/dark-state structure, dipole force of the near-resonant light, atom-atom effects (see `doc/light_matter_plan.md`). If a request implies one of these, surface the gap rather than inventing a model.
