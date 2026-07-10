# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Classical Monte Carlo multiphysics simulator for cold atoms (Rb85/Rb87): an ensemble of classical point atoms evolved under composable physics modules — optical tweezers (static/moving/astigmatic/gridded), magnetic (Zeeman) potentials, optical dipole beams, and near-resonant photon scattering with internal-state dynamics and photon recoil. A MOT, an optical molasses, a magnetic trap, or an SLM→AOD tweezer transfer are all just different module sets on one `AtomSystem`. Pure NumPy; Matplotlib is optional. See `doc/plan.md` for the tweezer physics model, `doc/light_matter_plan.md` for the light-force model, and `doc/multiphysics_plan.md` for the module architecture; `atommc/README.md` is a one-line index of every function in the package.

## Commands

Run from the repository root.

- Tests: `python3 -m unittest discover tests` (or per-module: `tests.test_core`, `tests.test_light_matter` (~1 min — real cooling simulations), `tests.test_ensemble`, `tests.test_physics_modules`, `tests.test_hyperfine` (includes an ARC cross-validation class, auto-skipped unless `arc` is installed))
- Single test: `python3 -m unittest tests.test_core.CorePhysicsTests.test_static_trap_has_low_numerical_heating`
- Examples: run every script from the repo root (e.g. `python3 example/mot/rb85_mot.py --save-plot`). See [example/README.md](example/README.md) for the full catalog of scripts and what each covers.
- Install deps: `pip install -e .` (core: `numpy>=1.23`); for plotting `pip install -e ".[viz]"` (adds `matplotlib>=3.7`).

There is no lint/format config. Python ≥3.9.

## Import layout

The package is `atommc` (declared in `pyproject.toml`). Imports inside the library are relative (`from ..ramp import ...`); call sites use `from atommc import ...` — the top-level `atommc/__init__.py` re-exports the whole modeling surface, while post-processing stays behind `from atommc.postprocess import analysis` (keeps matplotlib optional). Tests and examples bootstrap by inserting the repo root onto `sys.path` before importing — preserve that pattern when adding new entry points.

## Architecture (multiphysics registry)

A simulation is an `AtomSystem` (species + physics modules, [atommc/system.py](atommc/system.py)) handed to `simulate(system, config)` in [atommc/driver.py](atommc/driver.py). Modules implement one of two protocols from [atommc/physics/base.py](atommc/physics/base.py); the driver never special-cases any physics.

1. **`ConservativeForce`** — abstract `U(r, t)` [J] with `center_at` / `potential` required and finite-difference `force` / `hessian` defaults. Implementations in `atommc/physics/`:
   - Traps ([traps.py](atommc/physics/traps.py)): `GaussianTrap` (analytic force+hessian; `waist_axial_m` is a free model scale, **not** the Rayleigh length), `MovingGaussianTrap` (template + `RampSequence` from [ramp.py](atommc/ramp.py)), `AstigmaticAODTrap` (velocity-coupled focal lensing `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`; `dxdt2z = f0 / V_sound`; use a smooth ramp profile like `QUINTIC_MIN_JERK` when `dxdt2z != 0` since `LINEAR` gives step changes in focal shift), `GriddedTrap` (tabulated, tricubic/trilinear).
   - `ZeemanPotential` ([zeeman.py](atommc/physics/zeeman.py)): `U = m_F g_F μ_B |B_total|` for a fixed sublevel — magnetic traps, Stern-Gerlach. Analytic force from `MagneticFieldConfig.jacobian`; force clamped to zero at the field node (|B| cusp). No Majorana flips; the bare quadrupole potential is linear at the node, so Hessian-based sampling must not be centered there (use `ThermalCloud`).
   - `DipoleBeamPotential` ([dipole.py](atommc/physics/dipole.py)): far-detuned Gaussian beam from power/waist/wavelength, two-level RWA `U = (ħδ/2)ln(1+s/D)` with proper Rayleigh divergence; requires `|δ| ≥ 100Γ` (conservative-only; check `peak_scattering_rate_per_s`). Deliberately not a `LaserBeam`.
2. **`StochasticProcess`** — per-step non-conservative physics with opaque per-atom state (leading axis N; the driver only slices/writes rows) returning `StochasticStepResult` (state, optional velocity kick, named diagnostics declared via `DiagnosticSpec`). Implementations:
   - `LightScattering` ([scattering.py](atommc/physics/scattering.py)) = `LightMatterSystem` (geometry → per-atom/per-beam stimulated-rate matrix `W` with Doppler + Zeeman σ±/π decomposition, [light_matter.py](atommc/physics/light_matter.py)) + an `InternalStateModel` backend ([internal_state.py](atommc/physics/internal_state.py): `AdiabaticSteadyState` default, `RateEquationPopulations` unconditionally stable, `StochasticJumpState` needs `(Γ+W)dt < 0.1`) + recoil kicks. Emits `scattered_photons` (sum) and `excited_fraction` (last) diagnostics.
   - `HyperfineScattering` ([hyperfine.py](atommc/physics/hyperfine.py)) — m_F-resolved alternative on the same `LightMatterSystem` geometry. `HyperfineSpecies` (presets `RB85_D2_HFS`/`RB87_D2_HFS`) derives every `|F, m_F>` sublevel plus per-transition strengths/branching from exact Wigner algebra ([wigner.py](atommc/physics/wigner.py)) and Steck A/B constants (cross-validated against the optional ARC package in `tests/test_hyperfine.py`). Populations advance by one unconditionally stable implicit-Euler rate-equation step per timestep (trace/positivity exact, exact steady states; optical-pumping transients resolved while `(pump rate)·dt ≲ 0.3`). Models optical pumping, hyperfine dark states, and explicit repumpers — beam `detuning_hz` stays referenced to the cycling line, `HyperfineSpecies.transition_offset_hz(f_g, f_e)` re-references a beam (e.g. a repump). Adds per-level `ground_f<F>_population` diagnostics. Still no coherences/sub-Doppler; Zeeman shifts linear only.
3. **Shared geometry** (plain data, `atommc/geometry/`): `LaserBeam` (+ arbitrary `profile` callable, `six_beam_mot` builder) and `MagneticFieldConfig` (`UniformMagneticField`, `QuadrupoleMagneticField`; `vector` + analytic-or-FD `jacobian`). The same field instance can feed both a `LightScattering`'s Zeeman shifts and a `ZeemanPotential`'s force. All geometry signatures carry `time_s` (current impls static — pulsed beams/ramped coils slot in later).
4. **Driver loop** ([driver.py](atommc/driver.py)): sample initial ensemble → optional rejection resampling (budget `max_initial_resampling_rounds`, exceeding raises `RuntimeError`) → per step: every `StochasticProcess` steps and kicks first, then velocity-Verlet under `system.total_force` (operator split; ordering preserved from the legacy driver) → loss flags → trajectory recording. Lost atoms are masked out and never advanced. `AtomSystem.__post_init__` validates species consistency (any module exposing `.species` must match).
5. **Initial ensemble** ([ensemble.py](atommc/ensemble.py)): precedence `initial_source` (an `EnsembleSource`: `ThermalCloud`, `HarmonicTrapCloud`, `EffusiveBeam` with flux-weighted `weight`) > explicit arrays (require `reject_initially_lost=False`) > free Gaussian cloud (`initial_cloud_sigma_m`, required when the system has no forces) > harmonic sampling from the force Hessian at the deepest trap. The low-level samplers (`sample_thermal_velocities`, `sample_thermal_positions_harmonic`) live here too.
6. **Loss criteria**: boundary (`loss_radius_m`) always; energy (`E ≥ 0` in the current total conservative potential) controlled by `SimulationConfig.energy_loss` — `"auto"` (on iff purely conservative with ≥1 force), `"on"` (raises with stochastic processes attached), `"off"` (e.g. traps moving faster than their escape velocity `sqrt(2 U0/m)`, per `doc/plan.md`; recover survival post-hoc with `analysis.bound_to_trap` / `capture_probability` at a time where traps are at rest). Nothing is silently disabled. Once lost, always lost.
7. **Results**: `SimulationResult` core fields (survival/loss, energy gains, `*_survivors` vs `*_all` temperatures via `temperature_gain_uK_at(survivors_only=...)`, final state, optional stride-sampled trajectories) plus `diagnostics[module_name][key]` per-atom channels; `scattered_photons` / `final_excited_fraction` are properties over the conventional scattering channels (`None` on dark runs). Mass lives on `system.species` — `SimulationConfig` has no `mass_kg`.
8. **Post-processing** (`atommc/postprocess/`): [analysis.py](atommc/postprocess/analysis.py) (binding tests, capture probability, SLM/AOD occupation classification, trajectory time series), [harmonic.py](atommc/postprocess/harmonic.py) (quadratic expansion, normal modes, coherent-state occupations `n̄ = E/(ħω)` — a semi-classical proxy), [visualization.py](atommc/postprocess/visualization.py) (matplotlib-gated plots/animations).

## Unit conventions

The simulator is internally SI (m, s, kg, J), but inputs and human-readable outputs use:

- **Trap depth**: microkelvin (`depth_uK`). Convert with `units.microkelvin_to_joule` / `joule_to_microkelvin`.
- **Lengths**: meters in API; `units.um(...)` / `units.ms(...)` helpers for example/test code, plus `units.mhz(...)`, `units.gauss(...)`, `units.gauss_per_cm(...)`.
- **Magnetic fields**: tesla (`_T`, `_T_per_m`); Jacobians in T/m; magnetic moments in `_j_per_t`. **Frequencies**: linear hertz for laser detunings (`detuning_hz`, negative = red), angular rad/s for linewidths and internal rates (`_rad_s`, `_per_s`).
- **Energies in results**: microkelvin (`*_uK` suffix).

Any new field that carries a physical quantity should follow the `_m` / `_s` / `_kg` / `_uK` / `_j_per_m2` suffix scheme already used everywhere — search for an existing suffix before inventing a new one.

## Out of scope

Documented gaps (see `doc/plan.md`, `doc/light_matter_plan.md`, `doc/multiphysics_plan.md`): sub-Doppler mechanisms (beams are mutually incoherent in both scattering modules — `HyperfineScattering` resolves sublevel *populations* but carries no coherences), dark *superposition* states and nonlinear Zeeman (Paschen-Back) regimes (population dark states and repump dynamics ARE modeled by `HyperfineScattering`; `LightScattering` stays the effective two-level fast path with a perfect repumper assumed), dipole force of *near-resonant* light (far-detuned dipole potentials are `DipoleBeamPotential`), atom-atom effects, Majorana spin flips at field zeros, counter-rotating/multi-line polarizability in the dipole potential, tunneling, background-gas collisions, technical noise, CLI. If a request implies one of these, surface the gap rather than inventing a model. New physics should be added as a `ConservativeForce` or `StochasticProcess` module — not by editing the driver loop.
