# Source Function Index

Compact one-line descriptions of functions, classes, and methods in the local
`src` package.

## `analysis.py`

- `kinetic_energy_uK`: Convert per-atom velocities into kinetic energies in microkelvin units.
- `bound_to_trap`: Test whether atoms are classically bound to one trap at a chosen time.
- `single_trap_energy_uK`: Compute lab-frame mechanical energy in a single-trap potential.
- `capture_probability`: Compute final survival-and-bound probability for a target trap.
- `classify_final_trap_occupation`: Classify final atoms as SLM-only, AOD-only, ambiguous, unbound, or lost.
- `survival_probability_time_series`: Compute survival probability at each stored trajectory time.
- `loss_probability_time_series`: Compute loss probability as `1 - survival`.
- `mean_kinetic_energy_time_series_uK`: Compute mean kinetic energy versus stored trajectory time.
- `snapshot_moving_trap`: Return a static `GaussianTrap` snapshot of a `MovingGaussianTrap` at one time.

## `harmonic.py`

- `HarmonicApproximation.frequencies_hz`: Return harmonic normal-mode frequencies in hertz.
- `HarmonicApproximation.gradient_norm_n`: Return local force-imbalance magnitude at the expansion point.
- `HarmonicApproximation.quadratic_potential`: Evaluate the local Taylor-expanded harmonic potential.
- `MotionalDecomposition.mode_labels`: Return labels for the harmonic normal modes.
- `MotionalDecomposition.angular_frequencies_rad_s`: Return angular normal-mode frequencies.
- `MotionalDecomposition.frequencies_hz`: Return normal-mode frequencies in hertz.
- `approximate_harmonic_potential`: Build an analytic harmonic approximation from trap objects at a chosen time.
- `approximate_harmonic_potential_from_callable`: Build a finite-difference harmonic approximation from a generic potential function.
- `decompose_motion_into_harmonic_modes`: Project atom phase-space coordinates into harmonic normal modes.
- `coherent_fock_probabilities`: Compute coherent-state Fock probabilities from mean occupations.
- `summarize_mode_occupations`: Summarize per-mode occupation distributions with mean, median, and standard deviation.

## `ramp.py`

- `PolynomialConnector`: Smooth interpolation kernel between two waypoints, exposing value and derivative.
- `LINEAR`, `CUBIC_SMOOTHSTEP`, `QUINTIC_MIN_JERK`: Built-in `PolynomialConnector` profiles.
- `arb_fifth_poly`: Quintic family parameterized by `beta`; matches min-jerk at `beta = 15/8`.
- `const_jerk`: Cubic-jerk connector from `aod_slm_movement_v2`.
- `RampSequence.__post_init__`: Validate and normalize ramp time, center, and depth arrays.
- `RampSequence.start_time_s` / `end_time_s`: Endpoint times of the ramp.
- `RampSequence.at`: Interpolate AOD center and depth at a requested time.
- `RampSequence.center_at` / `depth_at`: Per-axis position / depth lookup.
- `RampSequence.velocity_at`: First derivative of the center position with respect to time.
- `RampSequence.depth_rate_at`: First derivative of depth with respect to time.
- `named_profile`: Look up a built-in connector by string name.
- `build_waypoint_ramp`: Convenience constructor with explicit profile keyword arguments.

## `sampling.py`

- `sample_thermal_velocities`: Sample Maxwell-Boltzmann velocities for a 3D atom ensemble.
- `sample_thermal_positions_harmonic`: Sample thermal positions from a local harmonic trap approximation at a chosen time.

## `simulation.py`

- `SimulationConfig.__post_init__`: Validate simulation parameters and normalize vector fields.
- `SimulationResult.final_temperature_uK`: Property returning survivors-only final temperature (alias).
- `SimulationResult.temperature_gain_uK`: Property returning survivors-only temperature gain (alias).
- `SimulationResult.temperature_gain_uK_at`: Method returning temperature gain for survivors or the full ensemble.
- `run_simulation`: Run velocity-Verlet propagation. Accepts either `(traps, config)` or the legacy `(static_trap, moving_trap_base, ramp, config)` form.

## `trap.py`

- `TrapConfig`: Abstract base for any time-dependent trap potential `U(r, t)`.
- `TrapConfig.center_at`: Required override returning the natural anchor point at time `t`.
- `TrapConfig.potential`: Required override evaluating `U(r, t)` in joules.
- `TrapConfig.force` / `hessian`: Default central-difference implementations; subclasses override.
- `GaussianTrap`: Cylindrically symmetric, time-independent 3D Gaussian; was the original `TrapConfig`.
- `GaussianTrap.with_center_depth`: Copy the trap with new center and depth.
- `MovingGaussianTrap`: Cylindrical Gaussian with center and depth driven by a `RampSequence`.
- `MovingGaussianTrap.snapshot`: Return the equivalent `GaussianTrap` at a chosen time.
- `AstigmaticAODTrap`: Astigmatic Gaussian with velocity-coupled focal lensing (`z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`).
- `AstigmaticAODTrap.rayleigh_length_m`: Computed `pi * w0^2 / lambda`.
- `total_potential` / `total_force` / `total_hessian`: Linear sum of trap evaluations at a chosen time.

## `units.py`

- `microkelvin_to_joule`: Convert microkelvin energy units to joules.
- `joule_to_microkelvin`: Convert joules to the equivalent temperature-like scale in microkelvin.
- `um`: Convert micrometers to meters.
- `ms`: Convert milliseconds to seconds.

## `visualization.py`

- `plot_transfer_summary`: Compatibility wrapper for the trajectory summary plot.
- `plot_transfer_trajectory_summary`: Plot 2D trajectories, mean position, and AOD ramps.
- `plot_transfer_energy_summary`: Plot kinetic energy, loss, and motional occupation distributions.
- `plot_transfer_trajectories_3d`: Plot atom trajectories and AOD path in 3D.
- `draw_traps_2d`, `draw_atoms_2d`, `draw_tweezer_beams_3d`, `draw_tweezer_beam_side_2d`,
  `draw_trap_ellipsoids_3d`, `draw_atoms_3d`, `draw_atom_trails_2d`, `draw_frame`,
  `render_animation`: lower-level frame and animation helpers.
