# Package Function Index

Compact one-line descriptions of functions, classes, and methods in the
`atommc` package. Layout: shared data at the top level, field geometry in
`geometry/`, physics modules in `physics/`, post-processing in
`postprocess/`.

## `system.py`

- `AtomSystem`: A species plus the registered physics modules; partitions them into `.forces` (conservative) and `.processes` (stochastic) and validates species consistency.
- `AtomSystem.total_potential` / `total_force` / `total_hessian`: Summed conservative evaluations (zeros with no forces).
- `AtomSystem.mechanical_energy_j`: Kinetic plus summed conservative potential energy.
- `AtomSystem.is_conservative`: True when no stochastic process is attached.
- `PhysicsModule`: Type alias `ConservativeForce | StochasticProcess`.

## `driver.py`

- `SimulationConfig.__post_init__`: Validate integrator/ensemble parameters and normalize vector fields. Initial ensembles: force-Hessian sampling (default), free Gaussian cloud (`initial_cloud_sigma_m` + `initial_mean_velocity_m_per_s`), a named `initial_source` (`ensemble.EnsembleSource`; reports its `weight` as `initial_ensemble_weight`), or explicit arrays (`initial_positions_m` / `initial_velocities_m_per_s_array`, require `reject_initially_lost=False`). `energy_loss` tri-state (`"auto"`/`"on"`/`"off"`) replaces the old silent disable.
- `SimulationResult`: Aggregate + per-particle outputs; `diagnostics[module][key]` carries per-atom channels declared by stochastic processes.
- `SimulationResult.scattered_photons` / `final_excited_fraction`: Properties reading the conventional `LightScattering` channels (`None` on dark runs).
- `SimulationResult.temperature_gain_uK_at`: Temperature gain for survivors or the full ensemble.
- `simulate`: The multiphysics driver — velocity-Verlet under the summed conservative forces, with every stochastic process operator-split into per-step velocity kicks; boundary loss always, energy loss per `energy_loss`.

## `species.py`

- `AtomSpecies`: Mass plus effective two-level cycling-transition data (wavelength, linewidth, `I_sat`, g-factors).
- `AtomSpecies.wavenumber_rad_per_m` / `recoil_velocity_m_per_s` / `mu_eff_j_per_t` / `doppler_temperature_uK`: Derived transition scales.
- `RB85_D2`, `RB87_D2`: Preset D2 cycling transitions (Steck data).

## `ensemble.py`

- `EnsembleSample`: Sampled initial `(positions, velocities)` plus the physical `weight` the ensemble represents.
- `EnsembleSource`: Abstract source of an initial ensemble; `sample(n, rng)` returns an `EnsembleSample`.
- `ThermalCloud`: Localized cloud — Gaussian blob or uniform box/sphere fill, isotropic Maxwell-Boltzmann velocities plus optional drift (background vapor or pre-cooled cloud).
- `HarmonicTrapCloud`: Equilibrium thermal cloud of conservative forces (covariance `k_B T K^-1`); the default trapped-run ensemble, as a source.
- `EffusiveBeam`: Oven with a small hole — flux speed law `f(v) ~ v^3 exp(-v^2/2 sigma^2)` over a truncated window (sample `weight` = window flux fraction), cosine (default) or uniform-solid-angle angular spread over the collimation cone.
- `EffusiveBeam.flux_cdf` / `window_fraction`: Analytic effusive flux CDF and the flux fraction carried by `[v_min, v_max]`.
- `sample_thermal_velocities`: Sample Maxwell-Boltzmann velocities for a 3D atom ensemble.
- `sample_thermal_positions_harmonic`: Sample thermal positions from a local harmonic approximation of conservative forces at a chosen time.

## `ramp.py`

- `PolynomialConnector`: Smooth interpolation kernel between two waypoints, exposing value and derivative.
- `LINEAR`, `CONST_JERK`, `QUINTIC_MIN_JERK`: Built-in `PolynomialConnector` profiles.
- `arb_fifth_poly`: Quintic family parameterized by `beta`; matches min-jerk at `beta = 15/8`.
- `const_jerk`: Cubic-jerk connector from `aod_slm_movement_v2`.
- `RampSequence.__post_init__`: Validate and normalize ramp time, center, and depth arrays.
- `RampSequence.start_time_s` / `end_time_s`: Endpoint times of the ramp.
- `RampSequence.at`: Interpolate AOD center and depth at a requested time.
- `RampSequence.center_at` / `depth_at`: Per-axis position / depth lookup.
- `RampSequence.velocity_at`: First derivative of the center position with respect to time.
- `RampSequence.depth_rate_at`: First derivative of depth with respect to time.

## `units.py`

- `microkelvin_to_joule` / `joule_to_microkelvin`: Energy conversions via `k_B`.
- `um` / `ms` / `mhz` / `gauss` / `gauss_per_cm`: Input shorthand converters to SI.

## `geometry/laser.py`

- `LaserBeam`: Collimated traveling-wave beam: direction, detuning, saturation `s0`, helicity, optional Gaussian waist, optional arbitrary intensity `profile` callable (apertures, grating-sector prisms, shadows).
- `LaserBeam.saturation_at`: Local saturation parameter with the Gaussian waist and/or custom profile applied.
- `six_beam_mot`: Build the standard three-axis retro-reflected MOT beam set with correct helicities.

## `geometry/fields.py`

- `MagneticFieldConfig`: Abstract base for any time-dependent magnetic field `B(r, t)` in tesla; default finite-difference `jacobian`.
- `UniformMagneticField`: Spatially uniform bias field (analytic zero Jacobian).
- `QuadrupoleMagneticField`: Linear anti-Helmholtz quadrupole `B = b'(x, y, -2z)` about an arbitrary axis (analytic constant Jacobian `b'(I - 3nn^T)`).
- `total_magnetic_field` / `total_magnetic_field_jacobian`: Linear sums of field / Jacobian evaluations at a chosen time.

## `physics/base.py`

- `ConservativeForce`: Abstract conservative potential `U(r, t)` [J]; required `potential` + `center_at`, default finite-difference `force` / `hessian`.
- `StochasticProcess`: Abstract per-step non-conservative physics; `initialize(n, rng)` returns opaque per-atom state (leading axis `n`), `step(state, x, v, t, dt, rng)` returns a `StochasticStepResult`.
- `StochasticStepResult`: Updated state + optional `(n, 3)` velocity kick + named per-atom diagnostics.
- `DiagnosticSpec`: Declares one diagnostic channel (`key`, `reduce="sum"|"last"`, dtype, fill) a process emits.
- `total_potential` / `total_force` / `total_hessian`: Linear sum of conservative-force evaluations at a chosen time.

## `physics/traps.py`

- `GaussianTrap`: Cylindrically symmetric, time-independent 3D Gaussian (analytic force + Hessian).
- `GaussianTrap.with_center_depth`: Copy the trap with new center and depth.
- `MovingGaussianTrap`: Cylindrical Gaussian with center and depth driven by a `RampSequence`.
- `MovingGaussianTrap.snapshot`: Return the equivalent `GaussianTrap` at a chosen time.
- `AstigmaticAODTrap`: Astigmatic Gaussian with velocity-coupled focal lensing (`z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`).
- `AstigmaticAODTrap.rayleigh_length_m`: Computed `pi * w0^2 / lambda`.
- `GriddedTrap`: Potential tabulated on a uniform 3D grid. Force is the analytical derivative of the interpolant (so it is conservative). Switchable `tricubic` (Catmull-Rom, C^1, default) or `trilinear` (C^0) modes; optional rigid translation via a ramp.
- `GriddedTrap.from_callable`: Sample a user-provided `potential_fn(r_local)` on a uniform grid and cache it.
- `GriddedTrap.from_trap`: Cache an existing `ConservativeForce` as a gridded copy at a chosen time.

## `physics/zeeman.py`

- `ZeemanPotential`: Magnetic potential `U = m_F g_F mu_B |B_total|` for a fixed sublevel — quadrupole magnetic traps, Stern-Gerlach expulsion. Analytic force `-mu J^T B_hat` from the field Jacobians; force clamped to zero at (near-)zero field. Majorana flips not modeled.
- `ZeemanPotential.for_sublevel`: Build from `(g_F, m_F)` quantum numbers.

## `physics/dipole.py`

- `DipoleBeamPotential`: Far-detuned focused Gaussian beam from lab parameters (power, waist, wavelength). Two-level RWA light shift `U = (hbar delta/2) ln(1 + s/D)` with proper Rayleigh-range divergence; analytic force. Requires `|delta| >= 100 Gamma` (conservative-only).
- `DipoleBeamPotential.depth_j` / `depth_uK` / `rayleigh_length_m` / `detuning_rad_s` / `peak_saturation` / `peak_scattering_rate_per_s`: Derived beam/trap scales.

## `physics/light_matter.py`

- `LightMatterSystem`: Species + beams + magnetic fields; reduces geometry to per-atom, per-beam stimulated rates.
- `LightMatterSystem.stimulated_rates`: One-way stimulated rate matrix `W(r, v, t)` with Doppler, Zeeman, and polarization decomposition.
- `LightMatterSystem.mean_radiation_force`: Deterministic steady-state radiation-pressure force (for analysis/tests).
- `polarization_fractions`: sigma+/pi/sigma- intensity fractions of a beam relative to the local B axis.

## `physics/internal_state.py`

- `ScatteringEvents`: Per-step photon counts (absorbed / stimulated per beam, spontaneous per atom).
- `InternalStateModel`: Abstract internal-state backend consuming the stimulated rate matrix `W`.
- `AdiabaticSteadyState`: Populations locked to the local steady state; Poisson photon numbers.
- `RateEquationPopulations`: Two-level excited population integrated exactly per step (handles transients).
- `StochasticJumpState`: Discrete ground/excited kinetic-MC trajectory (quantum-jump analog, needs `dt << 1/Gamma`).
- `steady_state_excited_fraction`: Shared two-level steady-state formula `p = W/(Gamma + 2W)`.
- `sample_recoil_velocity_kicks`: Convert photon events into per-atom recoil velocity kicks (exact per-photon spontaneous directions).

## `physics/scattering.py`

- `LightScattering`: The `StochasticProcess` wrapping a `LightMatterSystem` + `InternalStateModel`; per step computes rates, evolves populations, samples photon events, and returns recoil kicks. Emits `scattered_photons` (sum) and `excited_fraction` (last) diagnostics.

## `postprocess/analysis.py`

- `kinetic_energy_uK`: Convert per-atom velocities into kinetic energies in microkelvin units.
- `bound_to_trap`: Test whether atoms are classically bound to one trap at a chosen time.
- `single_trap_energy_uK`: Compute lab-frame mechanical energy in a single-trap potential.
- `capture_probability`: Compute final survival-and-bound probability for a target trap.
- `classify_final_trap_occupation`: Classify final atoms as SLM-only, AOD-only, ambiguous, unbound, or lost.
- `survival_probability_time_series`: Compute survival probability at each stored trajectory time.
- `loss_probability_time_series`: Compute loss probability as `1 - survival`.
- `mean_kinetic_energy_time_series_uK`: Compute mean kinetic energy versus stored trajectory time.
- `kinetic_temperature_time_series_uK`: Kinetic temperature versus stored time, optionally drift-subtracted.
- `snapshot_moving_trap`: Return a static `GaussianTrap` snapshot of a `MovingGaussianTrap` at one time.

## `postprocess/harmonic.py`

- `HarmonicApproximation.frequencies_hz`: Return harmonic normal-mode frequencies in hertz.
- `HarmonicApproximation.gradient_norm_n`: Return local force-imbalance magnitude at the expansion point.
- `HarmonicApproximation.quadratic_potential`: Evaluate the local Taylor-expanded harmonic potential.
- `MotionalDecomposition.mode_labels`: Return labels for the harmonic normal modes.
- `MotionalDecomposition.angular_frequencies_rad_s`: Return angular normal-mode frequencies.
- `MotionalDecomposition.frequencies_hz`: Return normal-mode frequencies in hertz.
- `approximate_harmonic_potential`: Build an analytic harmonic approximation from conservative forces at a chosen time.
- `approximate_harmonic_potential_from_callable`: Build a finite-difference harmonic approximation from a generic potential function.
- `decompose_motion_into_harmonic_modes`: Project atom phase-space coordinates into harmonic normal modes.
- `coherent_fock_probabilities`: Compute coherent-state Fock probabilities from mean occupations.
- `summarize_mode_occupations`: Summarize per-mode occupation distributions with mean, median, and standard deviation.

## `postprocess/visualization.py`

- `plot_transfer_summary`: Compatibility wrapper for the trajectory summary plot.
- `plot_transfer_trajectory_summary`: Plot 2D trajectories, mean position, and AOD ramps.
- `plot_transfer_energy_summary`: Plot kinetic energy, loss, and motional occupation distributions.
- `plot_transfer_trajectories_3d`: Plot atom trajectories and AOD path in 3D.
- `draw_traps_2d`, `draw_atoms_2d`, `draw_tweezer_beams_3d`, `draw_tweezer_beam_side_2d`,
  `draw_trap_ellipsoids_3d`, `draw_atoms_3d`, `draw_atom_trails_2d`, `draw_frame`,
  `render_animation`: lower-level frame and animation helpers.
- `draw_cloud_frame`: Snapshot of a trap-free scattering run — speed-colored cloud plus cooling curve.
- `render_cloud_animation`: Stitch `draw_cloud_frame`s into a GIF/WEBP/APNG for MOT-style runs.
- `_render_frames`: Render every frame of an animation, fanning out across worker processes (serial fallback).
- `_encode_frames` / `_save_animation`: Assign per-frame durations (start/end holds) and stitch PNGs into a GIF/WEBP/APNG; GIF uses a whole-run 256-color palette with optional Floyd-Steinberg dithering and `gifsicle -O3`.
- `_build_gif_palette` / `_try_gifsicle_optimize`: Stratified whole-run GIF palette; in-place `gifsicle` shrink when available.

## `constants.py`

- SI constants (`k_B`, `hbar`, `h`, amu, `c`, `mu_B`) plus Rb85/Rb87 isotope masses.
