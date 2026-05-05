# Classical Monte Carlo Atom Transfer Simulator

## Summary

This project implements a pure-Python library for estimating atom heating and
loss when a moving AOD tweezer pulls an atom from an SLM trap in a neutral atom
computer. The first version models Rb87 atoms with classical Hamiltonian
dynamics in time-dependent 3D Gaussian trap potentials. Trap strengths are
specified directly as depths in microkelvin.

The main workflow is:

1. Define static and moving tweezer geometries.
2. Define the AOD ramp sequence for position and trap depth.
3. Sample a thermal ensemble from the initial trap.
4. Propagate each atom with velocity-Verlet integration.
5. Report survival, loss, energy gain, and final effective temperature.
6. Optionally store trajectory samples for plotting ensemble motion, survival,
   and heating during the ramp.

## Physics Model

- Each tweezer is a red-detuned attractive 3D Gaussian potential:
  `U(r) = -U0 exp[-2 x^2 / wr^2 - 2 y^2 / wr^2 - 2 z^2 / wz^2]`.
- `U0` is supplied as a trap depth in microkelvin and converted to joules with
  Boltzmann's constant.
- The radial waist and axial scale are specified in SI units; the axial scale
  may be provided as a Rayleigh-length-like Gaussian scale for this v1 model.
- The total potential is the sum of the static SLM trap and the moving AOD trap.
- Forces are computed analytically from the potential gradient.
- The AOD ramp linearly interpolates center position and trap depth between
  user-provided time points.

## Initial Ensemble

- Rb87 mass and SI constants are defined in the library.
- Initial velocities are sampled from a Maxwell-Boltzmann distribution at the
  requested temperature.
- Initial positions are sampled from a local harmonic approximation to the
  initial combined trap. The Hessian of the Gaussian potential at the selected
  initial center determines the covariance.
- By default, initially unbound samples are rejected and resampled. This models
  an ensemble conditioned on successful SLM loading, so the reported loss is
  loss induced during the ramp rather than bad initial loading.
- By default, the initial center is the deepest trap center at `t = 0`; users
  may override it through `SimulationConfig.initial_center_m`.

## Public API

- `TrapConfig`: dataclass for Gaussian trap geometry and trap depth.
- `RampSequence`: dataclass for AOD ramp times, centers, and depths.
- `SimulationConfig`: dataclass for temperature, timestep, duration, ensemble
  size, random seed, atom mass, loss boundary, and trajectory storage options.
- `run_simulation(static_trap, moving_trap_base, ramp, config)`: runs the
  Monte Carlo propagation and returns a `SimulationResult`.
- `SimulationResult`: dataclass containing survival probability, loss fraction,
  final temperature, temperature gain, energy gain statistics, final states,
  lost flags, and optional sampled position/velocity/loss trajectories.
- `analysis.py`: helper functions for kinetic energy traces, survival traces,
  and final capture probability in a specified trap.
- `harmonic.py`: helper functions for approximating trap potentials as harmonic
  normal modes and decomposing atom phase-space coordinates into motional
  occupation estimates.
- `visualization.py`: optional Matplotlib helper for a compact transfer summary
  figure.

## Loss and Heating Metrics

- An atom is marked lost if its instantaneous total mechanical energy is
  non-bound relative to the current total potential, or if it leaves the
  optional spherical simulation boundary.
- Heating is measured from the bound final ensemble using:
  - mean and median total energy gain in microkelvin,
  - final kinetic effective temperature,
  - temperature gain relative to the requested initial temperature.
- Transfer probability is estimated as the fraction of the original ensemble
  that survives and is bound to the final AOD trap considered by itself. This is
  a practical classical capture proxy for the SLM-to-AOD transition.

## Harmonic Motional-State Analysis

- Approximate a potential near a chosen point with
  `U(r) ~= U(c) + grad U(c).(r-c) + 1/2 (r-c)^T K (r-c)`.
- Diagonalize `K / m` to obtain normal-mode axes and angular frequencies.
- For Gaussian tweezers, the modes are labeled as axial or radial by their
  dominant lab-frame axis.
- Decompose atom motion in the harmonic basis using relative trap-frame
  velocity, then report per-mode classical oscillator energy and
  coherent-state/semi-classical occupation `nbar = E / (hbar omega)`.
- A classical trajectory is not itself a quantum eigenstate; the reported
  nearest quantum number and optional Fock probabilities are an approximate
  harmonic/coherent-state diagnostic.

## Visualization

- Store trajectory positions to show the ensemble following, lagging, or
  escaping the moving AOD tweezer.
- Store trajectory velocities and loss masks to plot mean kinetic energy and
  survival probability versus time.
- Write separate 2D diagnostic figures with suffixes:
  - `_traj`: ensemble trajectories, mean atom position, AOD position ramp, AOD
    depth ramp.
  - `_energy`: mean kinetic energy, loss probability, initial SLM motional
    occupation distributions, final AOD motional occupation distributions.
- Compare multiple AOD position ramp profiles under identical trap, depth, and
  random-seed conditions. The example includes linear, cubic smoothstep,
  quintic minimum-jerk, and sinusoidal profiles.
- Keep Matplotlib optional: the simulator can run headless, while
  `example/slm_to_aod_transfer.py --plot` produces the diagnostic figures when
  the optional visualization dependency is installed.

## Test Plan

- Verify Gaussian potential symmetry and that forces point back toward the trap
  center.
- Verify ramp interpolation at endpoints, midpoints, and clamped times.
- Verify Maxwell-Boltzmann velocity sampling recovers the requested
  temperature within Monte Carlo tolerance.
- Verify a static-trap simulation has low artificial heating for a small
  timestep.
- Verify an aggressive pull-out ramp has lower survival than a slower ramp.
- Verify stored trajectories produce survival and kinetic-energy time series.

## Current Scope

- Included: classical dynamics, time-dependent Gaussian traps, thermal sampling,
  energy/boundary loss detection, aggregate heating/loss metrics, optional
  trajectory-based visualization helpers, and an SLM-to-AOD example script.
- Excluded from v1: photon recoil, tunneling, background gas collisions,
  technical noise, cooling/damping, optical-power-to-depth conversion, CLI.
