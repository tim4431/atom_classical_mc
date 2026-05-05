# Source Function Index

Compact one-line descriptions of functions and methods in the local `src` package.

## `analysis.py`

- `kinetic_energy_uK`: Convert per-atom velocities into kinetic energies in microkelvin units.
- `bound_to_trap`: Test whether atoms are classically bound to one trap.
- `single_trap_energy_uK`: Compute lab-frame mechanical energy in a single-trap potential.
- `capture_probability`: Compute final survival-and-bound probability for a target trap.
- `classify_final_trap_occupation`: Classify final atoms as SLM-only, AOD-only, ambiguous, unbound, or lost.
- `survival_probability_time_series`: Compute survival probability at each stored trajectory time.
- `loss_probability_time_series`: Compute loss probability as `1 - survival`.
- `mean_kinetic_energy_time_series_uK`: Compute mean kinetic energy versus stored trajectory time.

## `harmonic.py`

- `HarmonicApproximation.frequencies_hz`: Return harmonic normal-mode frequencies in hertz.
- `HarmonicApproximation.gradient_norm_n`: Return local force-imbalance magnitude at the expansion point.
- `HarmonicApproximation.quadratic_potential`: Evaluate the local Taylor-expanded harmonic potential.
- `MotionalDecomposition.mode_labels`: Return labels for the harmonic normal modes.
- `MotionalDecomposition.angular_frequencies_rad_s`: Return angular normal-mode frequencies.
- `MotionalDecomposition.frequencies_hz`: Return normal-mode frequencies in hertz.
- `approximate_harmonic_potential`: Build an analytic harmonic approximation from trap objects.
- `approximate_harmonic_potential_from_callable`: Build a finite-difference harmonic approximation from a generic potential function.
- `_build_harmonic_approximation`: Construct a harmonic approximation from offset, gradient, and Hessian.
- `_evaluate_potential_scalar`: Evaluate a potential callback at one position as a scalar.
- `decompose_motion_into_harmonic_modes`: Project atom phase-space coordinates into harmonic normal modes.
- `coherent_fock_probabilities`: Compute coherent-state Fock probabilities from mean occupations.
- `summarize_mode_occupations`: Summarize per-mode occupation distributions with mean, median, and standard deviation.
- `_default_mode_labels`: Label normal modes by their dominant lab-frame axis.

## `ramp.py`

- `RampSequence.__post_init__`: Validate and normalize ramp time, center, and depth arrays.
- `RampSequence.start_time_s`: Return the first ramp time.
- `RampSequence.end_time_s`: Return the last ramp time.
- `RampSequence.at`: Interpolate AOD center and depth at a requested time.

## `sampling.py`

- `sample_thermal_velocities`: Sample Maxwell-Boltzmann velocities for a 3D atom ensemble.
- `sample_thermal_positions_harmonic`: Sample thermal positions from a local harmonic trap approximation.
- `_default_center`: Choose the deepest trap center for initial position sampling.
- `_trap_list`: Normalize one trap or an iterable of traps into a list.

## `simulation.py`

- `SimulationConfig.__post_init__`: Validate simulation parameters and normalize vector fields.
- `run_simulation`: Run Monte Carlo velocity-Verlet propagation through the ramp.
- `_append_trajectory_sample`: Store one trajectory snapshot of positions, velocities, and loss masks.
- `_resample_initially_lost`: Resample atoms that are unbound at the initial time.
- `_initial_lost_flags`: Mark atoms initially unbound or outside the loss boundary.
- `_moving_trap_at`: Create the moving trap configuration at one ramp time.
- `_mechanical_energy`: Compute kinetic plus trap potential energy.
- `_kinetic_temperature_uK`: Estimate kinetic temperature from velocities.
- `_outside_boundary`: Mark atoms outside the configured spherical loss boundary.
- `_trap_list`: Normalize one trap or an iterable of traps into a list.

## `trap.py`

- `TrapConfig.__post_init__`: Validate trap geometry and store the center as an array.
- `TrapConfig.axial_scale_m`: Return the axial Gaussian scale used by the model.
- `TrapConfig.depth_joule`: Convert trap depth from microkelvin to joules.
- `TrapConfig.scales_m`: Return radial and axial Gaussian scales as a 3-vector.
- `TrapConfig.with_center_depth`: Copy a trap with a new center and depth.
- `TrapConfig.potential`: Evaluate the Gaussian trap potential.
- `TrapConfig.force`: Evaluate the analytic Gaussian trap force.
- `TrapConfig.hessian`: Evaluate the analytic potential Hessian at one point.
- `total_potential`: Sum potentials from one or more traps.
- `total_force`: Sum forces from one or more traps.
- `total_hessian`: Sum potential Hessians from one or more traps.
- `_trap_list`: Normalize one trap or an iterable of traps into a list.
- `_as_positions`: Validate and convert position input arrays.

## `units.py`

- `microkelvin_to_joule`: Convert microkelvin energy units to joules.
- `joule_to_microkelvin`: Convert joules to microkelvin energy units.
- `um`: Convert micrometers to meters.
- `ms`: Convert milliseconds to seconds.

## `visualization.py`

- `plot_transfer_summary`: Compatibility wrapper for the trajectory summary plot.
- `plot_transfer_trajectory_summary`: Plot 2D trajectories, mean position, and AOD ramps.
- `plot_transfer_energy_summary`: Plot kinetic energy, loss, and motional occupation distributions.
- `plot_transfer_trajectories_3d`: Plot atom trajectories and AOD path in 3D.
- `_probability_axis_upper_limit`: Choose a readable y-limit for small probability traces.
- `_plot_mode_occupation_distribution`: Plot normalized per-mode occupation histograms.
- `_occupation_axis_upper_limit`: Choose a readable x-limit for occupation histograms.
- `_set_equal_3d_limits`: Apply equal-scale 3D plot limits around trajectory data.
