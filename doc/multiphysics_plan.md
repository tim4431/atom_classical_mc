# Multiphysics module architecture

This document describes the physics-module architecture introduced by the
2026-07 restructure (package `src` → `atommc`), which replaced the monolithic
`run_simulation` driver with an `AtomSystem` module registry and a `simulate`
driver. The tweezer physics model is unchanged from `doc/plan.md`, and the
scattering model is unchanged from `doc/light_matter_plan.md`; this document
covers how they (and the new magnetic/dipole potentials) compose.

## Object model

A simulation is, in the spirit of a COMSOL model:

- **`AtomSystem`** (`atommc/system.py`) — the model: one `AtomSpecies` plus a
  list of physics modules. The system partitions modules into conservative
  forces and stochastic processes, validates that any species-bearing module
  matches the system species, and exposes summed
  `total_potential/force/hessian` and `mechanical_energy_j`.
- **Physics modules** (`atommc/physics/`) — the physics interfaces. Two
  protocols, defined in `physics/base.py`:
  - **`ConservativeForce`** — a potential `U(r, t)` in joules with
    `F = -grad U`. Required: `potential`, `center_at`; default
    finite-difference `force` and `hessian` (override with analytics where
    available). Implementations: the trap family (`GaussianTrap`,
    `MovingGaussianTrap`, `AstigmaticAODTrap`, `GriddedTrap`),
    `ZeemanPotential`, `DipoleBeamPotential`.
  - **`StochasticProcess`** — per-step, possibly random, non-conservative
    physics. `initialize(n, rng)` returns an opaque per-atom state array
    (leading axis `n`; the driver only slices/writes rows, never interprets
    contents, so a multilevel backend can use an `(n, k)` state without
    driver changes). `step(state, x, v, t, dt, rng)` returns a
    `StochasticStepResult`: the updated state, an optional `(n, 3)` velocity
    kick, and named per-atom diagnostics. `diagnostics_spec()` declares the
    channels (`DiagnosticSpec`: key, `reduce="sum"|"last"`, dtype, fill).
    Implementation: `LightScattering`.
- **`simulate(system, config)`** (`atommc/driver.py`) — the study/solver.
  Per timestep it (1) steps every stochastic process on the active atoms and
  applies its velocity kicks, then (2) advances the ensemble by
  velocity-Verlet under the summed conservative forces (the same operator
  split, kicks-first ordering as the previous driver, so scattering
  statistics are preserved), then (3) applies loss criteria and records
  trajectories. Lost atoms are masked out of everything. Per-atom
  diagnostics land in `SimulationResult.diagnostics[module_name][key]`;
  `result.scattered_photons` / `result.final_excited_fraction` are
  convenience properties over the conventional `LightScattering` channels.

Shared geometry stays plain data: `LaserBeam` and `MagneticFieldConfig`
(`atommc/geometry/`) are consumed by whichever modules need them — the same
`QuadrupoleMagneticField` instance can feed a `LightScattering`'s Zeeman
shifts and a `ZeemanPotential`'s force. All geometry evaluation signatures
carry `time_s` (current implementations are static; pulsed beams and ramped
coils slot in without interface changes).

## Loss criteria

Boundary loss (`loss_radius_m`) always applies. The energy criterion
(mechanical energy >= 0) is controlled by `SimulationConfig.energy_loss`:

- `"auto"` (default): active iff the system is purely conservative and has at
  least one force — the modern equivalent of the old behavior, but explicit.
- `"on"`: demanded; raises `ValueError` if any stochastic process is attached
  (non-conservative kicks invalidate the criterion) or there are no forces.
- `"off"`: disabled; use for fast-moving traps (see `doc/plan.md`) or
  potentials with regions of `U >= 0` along valid trajectories.

Nothing is silently disabled anymore (previously, attaching scattering
switched the energy criterion off without notice).

## New physics modules

### `ZeemanPotential` (magnetic force)

`U(r, t) = mu |B_total(r, t)|` with `mu = m_F g_F mu_B` the signed moment of
the occupied sublevel (`for_sublevel(fields, g_f, m_f)` builds it). The
module owns its full field list because magnitudes do not add
(`|B1 + B2| != |B1| + |B2|`). Force is analytic:
`F = -mu J^T B_hat` with `J = dB_i/dr_j` summed over
`MagneticFieldConfig.jacobian` (finite-difference default; analytic for
uniform and quadrupole fields — the quadrupole Jacobian is the constant
`b'(I - 3 n n^T)`). At a field zero `|B|` is non-differentiable; the force is
clamped to exactly zero below `zero_field_threshold_t`.

Model limits: weak-field (linear) Zeeman regime; adiabatic spin-following in
one fixed sublevel — Majorana spin flips near field zeros are not modeled. A
bare quadrupole's trapped potential is linear (conical) at the node, so
Hessian-based tools (`HarmonicTrapCloud`,
`approximate_harmonic_potential`) must not be centered there; sample such
clouds with `ThermalCloud`.

### `DipoleBeamPotential` (optical potential from beam parameters)

A far-detuned focused Gaussian beam defined by laboratory parameters
(`power_w`, `waist_m`, `wavelength_m`, focus, direction) rather than a
hand-specified `depth_uK`. It deliberately does **not** reuse `LaserBeam`
(a near-resonant collimated description with no axial divergence); it
implements proper Gaussian-beam optics with `z_R = pi w0^2 / lambda`.

Physics: the two-level rotating-wave light shift
`U = (hbar delta / 2) ln(1 + s / D)`, `D = 1 + (2 delta / Gamma)^2`, with
`s = I / I_sat` and `delta` the angular detuning from the species resonance
(reduces to `hbar Gamma^2 s / (8 delta)` far-detuned). The force is analytic
via the chain rule, including the axial (Rayleigh) term. Construction
requires `|delta| >= 100 Gamma` — nearer-resonant light scatters, which this
conservative module does not model (`peak_scattering_rate_per_s` reports the
residual rate). Known approximation: effective two-level, no counter-rotating
term or D1/D2 multi-line polarizability (~10% depth error for Rb at 1064 nm).

## Extending

A new conservative potential is a `ConservativeForce` subclass (gravity would
be ~10 lines). A new non-conservative process (background-gas collisions,
richer internal-state models) is a `StochasticProcess` with its own state and
diagnostics — the driver, result plumbing, and all other modules are
untouched. Out of scope remains as documented in `doc/plan.md` and
`doc/light_matter_plan.md`: sub-Doppler mechanisms, multilevel/dark-state
structure, the dipole force of *near-resonant* light, atom-atom effects, and
Majorana losses.
