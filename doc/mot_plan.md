# MOT / light-force extension: physics model and design

This document specifies the magneto-optical trap (MOT) extension added on
top of the conservative-trap simulator described in `plan.md`. It covers
Rb85/Rb87 (any effective two-level cycling transition), quadrupole + bias
magnetic fields, an arbitrary set of cooling beams, internal-state
dynamics, and photon-recoil momentum diffusion.

## Multiphysics architecture

The problem factors into two coupled solvers advanced by operator
splitting each timestep, in the spirit of COMSOL physics modules:

```
            geometry reduction                     backend 1
  r, v, t ──> MOTSystem.stimulated_rates ──> W (N x n_beams) ──> InternalStateModel.step
              (B field, Doppler, Zeeman,                          │
               polarization decomposition)                        ▼
                                                        ScatteringEvents (photons)
            backend 2                                             │
  r, v <── semi-implicit Euler <── recoil kicks + trap force <────┘
```

- **`MOTSystem`** (`src/mot.py`) owns the *field geometry*: an
  `AtomSpecies`, `LaserBeam`s, and `MagneticFieldConfig`s. Its single
  job is to reduce everything to the per-atom, per-beam one-way
  stimulated rate matrix `W` (s^-1).
- **`InternalStateModel`** (`src/internal_state.py`) owns the *internal
  state*: it consumes `W`, evolves populations, and reports the photon
  `ScatteringEvents` of the step. The state array is opaque to the
  driver, so richer backends (multilevel rate equations, density
  matrix / MCWF) can be added without touching the loop.
- **The momentum backend** (`run_mot_simulation` loop) applies recoil
  velocity kicks derived from the events plus any conservative
  `TrapConfig` force, then drifts positions (semi-implicit Euler; the
  velocity-Verlet integrator of `run_simulation` is not meaningful for
  stochastic, velocity-dependent forces).

## Rate model

For each beam `b` the circular polarization content (helicity `h` in
[-1, 1], mean photon spin along `k`) is decomposed **incoherently** into
sigma+/pi/sigma- fractions relative to the local field direction
`b_hat = B / |B|` with `c = k_hat . b_hat`:

```
f_(+/-1) = (1 + h)/2 * (1 +/- c)^2/4  +  (1 - h)/2 * (1 -/+ c)^2/4
f_0      = (1 - c^2)/2                          (sum_q f_q = 1)
```

Each component `q` sees the detuning

```
delta_q = 2 pi detuning_hz - k . v - q * mu_eff |B| / hbar
```

(laser detuning, Doppler shift, Zeeman shift of the stretched cycling
transition, `mu_eff = (g_e (F_g + 1) - g_g F_g) mu_B` = exactly 1 mu_B
for both Rb85 F=3->4 and Rb87 F=2->3). The one-way stimulated rate is

```
W_b = (Gamma / 2) * sum_q s_b(r) f_q / (1 + (2 delta_q / Gamma)^2)
```

with `s_b(r)` the local saturation parameter (Gaussian beam profile
supported). Saturation competition between beams enters through the
two-level population `p = W_tot / (Gamma + 2 W_tot)`: absorption from
beam `b` proceeds at `W_b (1 - p)`, stimulated emission into `b` at
`W_b p`, spontaneous emission at `Gamma p`. The net beam force is then
the standard saturated `hbar k W_b (1 - 2p)`.

## Internal-state backends

| backend | state | events | dt requirement | use case |
|---|---|---|---|---|
| `AdiabaticSteadyState` | none (populations follow local steady state) | Poisson | rates ~constant over dt | default; CW MOT/molasses |
| `RateEquationPopulations` | excited population `p` per atom, exact exponential update | Poisson at time-averaged `p` | none (unconditionally stable) | pulsed/switched beams |
| `StochasticJumpState` | discrete ground/excited, kinetic-MC jumps | exact 0/1 per step | `(Gamma + W_tot) dt < 0.1` | per-trajectory photon statistics |

The Poisson backends draw absorption / stimulated / spontaneous photon
numbers independently; means and the leading momentum diffusion are
correct, absorption-emission pairing correlations are neglected. The
jump backend is the rate-equation analog of a quantum-jump (MCWF)
trajectory.

## Recoil

Every absorbed photon kicks `+hbar k` along the beam, every stimulated
emission `-hbar k` along the beam (so a stimulated cycle transfers zero
net momentum), and every spontaneous photon `hbar k` in an isotropically
random direction, sampled photon-by-photon (no Gaussian approximation).
Dipole emission patterns are a possible refinement.

## Loss criterion

Radiation pressure is non-conservative, so the energy-based loss flag of
`run_simulation` is meaningless here. `run_mot_simulation` only supports
geometric loss (`loss_radius_m`).

## Explicitly out of scope

- Coherences between beams: standing waves, polarization gradients, and
  therefore **all sub-Doppler cooling** (the simulator bottoms out near
  the Doppler temperature, appropriately rescaled by saturation and
  detuning).
- Multilevel structure beyond the effective two-level cycling
  transition: no hyperfine dark states, no repumper depletion dynamics
  (a repumper is assumed perfect).
- Dipole force of the cooling light (radiation pressure only; conservative
  dipole traps can still be layered via `TrapConfig`).
- Atom-atom effects: reabsorption/radiation trapping, light-assisted
  collisions, density limits.
- Magnetic (Stern-Gerlach) force on the ground state; fine for MOT-scale
  gradients over ms timescales.

Validation targets (covered in `tests/test_mot.py`): two-level resonant
rates, Doppler and Zeeman resonance shifts, molasses damping sign, MOT
restoring force sign, steady-state populations across all three
backends, photon-budget balance, cooling of a hot cloud to the
Doppler-scale equilibrium, and MOT cloud compression.
