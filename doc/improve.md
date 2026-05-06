# Improvement plan: comparison with `aod_slm_movement_v2`

The reference v2 simulator extracted at
[`tmp/aod_slm_movement_v2/`](../tmp/aod_slm_movement_v2/STRUCTURE.md) implements
some real physics this project did not, while lacking the loss accounting and
ergonomic API we already have. This document records the comparison and the
agreed refactor plan.

## What v2 has that we lacked

1. **Astigmatic Gaussian potential with velocity-coupled focal shift.** v2's
   `dipole_traps_astigmatic.py` carries two separate axial focal points
   `z01`, `z02`, fed by the AOD's instantaneous velocity through
   `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`. `dxdt2z = f0 / V_sound` is the
   acoustic-wave constant of the AOD. This is the dominant heating channel for
   accelerating AOD tweezers and is missing from the original cylindrically
   symmetric `TrapConfig`.

2. **Smooth piecewise-polynomial trajectories with derivatives.** v2's
   `trajectories.py` provides `linear`, `const_jerk`, `min_jerk`, and
   `arb_fifth_poly(beta)` connectors via a decorator that returns both `y(x)`
   and `dy/dx(x)`. The β knob in `arb_fifth_poly` parameterizes a quintic
   family that smoothly interpolates between minimum-jerk and aggressive
   profiles. Our original `RampSequence.at` was piecewise-linear — velocity is
   piecewise-constant with discontinuities at every waypoint.

3. **Multi-trap composition.** v2's `MergedTrapMovements` linearly sums the
   forces and potentials of any list of `SingleTrapMovement` objects. Our
   `run_simulation` hardcoded one static trap and one moving trap.

## How v2 decides whether an atom is lost

v2 has **no on-the-fly loss accounting**. The integration loop in
[`mc_solver.solve_trap_movement`](../tmp/aod_slm_movement_v2/aod_slm_movement_v2/servers/mc_solver.py)
propagates every atom unconditionally with `scipy.integrate.odeint`. After
the propagation completes, the survival check happens outside the integrator
in the calling notebook. For example,
[`single_round_trip.ipynb` cell 5](../tmp/aod_slm_movement_v2/aod_slm_movement_v2/single_round_trip.ipynb):

```python
init_energy = mc_ensemble.calc_energy_spec(trap_system.V_txyz, t=0,
                                           f_energy=1e-6/planck_const)
sol_energy  = sol_ensemble.calc_energy_spec(trap_system.V_txyz, t=0,
                                            f_energy=1e-6/planck_const)
n_init.append(np.sum(init_energy < 0))
n_sol.append(np.sum(sol_energy  < 0))
survival_ratio = n_sol / n_init
```

The criterion is `KE + V(t=0, r_final) < 0`, evaluated in the **time-zero**
total potential. So an atom is "bound" iff its mechanical energy in the
*reference* potential (defined at `t=0`) is negative. Two consequences:

- Initially-unbound atoms are excluded by counting `n_init` the same way at
  `t=0`. Survival ratio is `n_sol / n_init`, conditioning on initial loading.
- Lost atoms keep being integrated, which wastes work, can introduce
  numerical instability, and means there is no "instantaneous" loss time —
  only a single bound/unbound test at the end.

This is acceptable for symmetric round-trips (where `V(t=0) ≈ V(t_end)` after
the AOD returns) but ill-defined for one-way handoffs where the SLM ramps off
and the t=0 reference potential has nothing in common with the t=end one. Our
existing per-step loss check (energy ≥ 0 in the *current* total potential, or
outside the boundary radius) is strictly more general and we keep it.

## Decisions for this refactor

| # | Plan item                                                    | Decision |
|---|--------------------------------------------------------------|----------|
| 1 | `TrapConfig` becomes an ABC supporting arbitrary `U(r, t)`; current cylindrical static trap becomes a `GaussianTrap` subclass; add `MovingGaussianTrap` (cylindrical, time-dependent) and `AstigmaticAODTrap` (full v2 lensing physics). | Do |
| 2 | Promote v2's polynomial connectors into `src/ramp.py`; `RampSequence` exposes `velocity_at(t)` and configurable position/depth profiles. | Do |
| 3 | `run_simulation` accepts an iterable of `TrapConfig`s — multi-trap support is built in. Old `(static, moving, ramp, config)` signature provided as a thin compat wrapper. | Do |
| 4 | Drop `rayleigh_length_m` from the cylindrical Gaussian — keep only `waist_axial_m`. | Do |
| 5 | Loss criterion stays as-is (per-step, current total potential). | Keep |
| 6 | Temperature gain selectable: `survivors_only=True` (default, matches old behavior) or `survivors_only=False` (whole ensemble). Both stored in `SimulationResult`. | Do |
| 7 | No depth-vs-temperature regime warning. | Skip |

## Architectural notes

- `TrapConfig.center_at(time_s)` is the abstract anchor point: it gives the
  natural sampling/expansion center at any time. Static traps return the
  fixed `center_m`; moving traps query the ramp.
- All trap evaluation methods take `time_s` so the simulation loop only has
  to know "which time am I at." There is no longer a static/moving split in
  the API.
- `MovingGaussianTrap.snapshot(time_s) -> GaussianTrap` lets analysis code
  build a static trap at a chosen time when it needs to (e.g. for occupation
  classification at the final time).
- `AstigmaticAODTrap` consumes the same `RampSequence` as `MovingGaussianTrap`
  but additionally uses `ramp.velocity_at(t)` to compute the per-axis focal
  offsets `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`.
- `RampSequence` smooth profiles default to `LINEAR` to keep older waypoint-
  table calls behaving identically; passing `position_profile=QUINTIC_MIN_JERK`
  switches to a smooth ramp without altering the API.
- `SimulationResult` adds `*_survivors` and `*_all` temperature fields. The
  legacy `final_temperature_uK` / `temperature_gain_uK` properties keep
  pointing at the survivors-only values.
