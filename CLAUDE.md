# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Classical Monte Carlo simulator for Rb87 atom heating and loss when a moving AOD optical tweezer pulls an atom out of a static SLM trap. Pure NumPy; Matplotlib is optional. See `doc/plan.md` for the physics model and intended public API; `src/README.md` is a one-line index of every function in the package.

## Commands

Run from the repository root.

- Tests: `python3 -m unittest tests.test_core` (or `python3 -m unittest discover tests`)
- Single test: `python3 -m unittest tests.test_core.CorePhysicsTests.test_static_trap_has_low_numerical_heating`
- SLM-to-AOD example: `python3 example/slm_to_aod_transfer.py` (add `--plot` / `--save-plot` for 2D figures, `--plot-3d` / `--save-3d-plot` for 3D, `--compare-position-ramps --save-comparison-plot ...` to sweep ramp shapes)
- Other examples: `example/slm_to_slm_transfer.py`, `example/spectator_qubits.py`, `example/position_ramp_compare.py`
- Install deps: `pip install -e .` (core: `numpy>=1.23`); for plotting `pip install -e ".[viz]"` (adds `matplotlib>=3.7`).

There is no lint/format config. Python ≥3.9.

## Import layout (non-standard)

`pyproject.toml` declares `packages = ["src"]`, so the package literally is named `src`. Imports inside the library are relative (`from .trap import ...`); call sites use `from src.trap import ...`. Tests and examples bootstrap this by inserting the repo root onto `sys.path` before importing — preserve that pattern when adding new entry points. Don't rename `src/` to the project name without also fixing every import.

## Architecture

A single-pass pipeline driven by `run_simulation` in [src/simulation.py](src/simulation.py):

1. **Geometry** — `TrapConfig` ([src/trap.py](src/trap.py)) is an immutable dataclass for one red-detuned 3D Gaussian tweezer `U(r) = -U0 exp(-2(x²+y²)/w_r² - 2 z²/w_z²)`, with analytic `potential` / `force` / `hessian`. Multiple traps are summed via `total_potential`/`total_force`/`total_hessian`.
2. **Time dependence** — `RampSequence` ([src/ramp.py](src/ramp.py)) is a piecewise-linear interpolator over `(times_s, centers_m, depths_uK)`. `ramp.at(t)` clamps outside the table. The moving trap at time `t` is rebuilt every step via `moving_trap_base.with_center_depth(...)`.
3. **Initial ensemble** — [src/sampling.py](src/sampling.py): velocities from Maxwell-Boltzmann; positions from a Gaussian whose covariance is `k_B T · K⁻¹` where `K` is the combined-trap Hessian at the chosen center (deepest trap by default, or `SimulationConfig.initial_center_m`). With `reject_initially_lost=True` (default) the simulator resamples atoms that are unbound at `t=0` so reported loss reflects ramp dynamics, not bad loading. The rejection budget is bounded by `max_initial_resampling_rounds`; exceeding it raises `RuntimeError`.
4. **Propagation** — velocity-Verlet, vectorized over the ensemble. Lost atoms are masked out and not advanced further. An atom is marked lost when its instantaneous mechanical energy in the *current* total potential is ≥ 0, or when it leaves the optional spherical `loss_radius_m`. Once lost, always lost.
5. **Reporting** — `SimulationResult` carries survival/loss, energy-gain stats, kinetic temperature, full final state, and (if `store_trajectories=True`) sampled `(positions, velocities, lost)` snapshots at `trajectory_stride` intervals. `trap_velocity_m_per_s=0` is hardcoded in some analysis paths; pass the AOD slope explicitly when needed.
6. **Post-processing** — [src/analysis.py](src/analysis.py) computes single-trap binding (`bound_to_trap`, `capture_probability`), the SLM-vs-AOD-vs-ambiguous breakdown (`classify_final_trap_occupation`), and time series from stored trajectories. [src/harmonic.py](src/harmonic.py) builds a quadratic Taylor expansion (analytic from `TrapConfig`s, or finite-difference from any callable), diagonalizes `K/m` to get normal modes, and decomposes phase-space coordinates into per-mode classical energies and coherent-state occupations `n̄ = E/(ℏω)` (a semi-classical proxy — a classical trajectory is not a quantum eigenstate).

## Unit conventions

The simulator is internally SI (m, s, kg, J), but inputs and human-readable outputs use:

- **Trap depth**: microkelvin (`depth_uK`). Convert with `units.microkelvin_to_joule` / `joule_to_microkelvin` (or via `TrapConfig.depth_joule`).
- **Lengths**: meters in API, but the `units.um(...)` and `units.ms(...)` helpers exist so example/test code can write `um(1.2)` for a waist or `ms(0.5)` for a duration.
- **Energies in results**: microkelvin (`*_uK` suffix). Time series come back in joules only inside `harmonic.py` hot paths.

Any new field that carries a physical quantity should follow the `_m` / `_s` / `_kg` / `_uK` / `_j_per_m2` suffix scheme already used everywhere — search `_per_s2` etc. before inventing a new name.

## Trap axial scale

`TrapConfig` requires *exactly one* of `waist_axial_m` or `rayleigh_length_m`. The v1 model treats whichever is supplied as the Gaussian axial scale (it is **not** the diffraction Rayleigh length, despite the name); `axial_scale_m` returns it. Don't add the missing diffraction conversion silently — it would change every existing example's frequencies.

## Out of scope (v1)

Per `doc/plan.md`: no photon recoil, tunneling, background-gas collisions, technical noise, cooling/damping, optical-power-to-depth conversion, or CLI. If a request implies one of these, surface the gap rather than inventing a model.
