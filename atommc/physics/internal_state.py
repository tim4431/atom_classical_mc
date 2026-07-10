"""Internal-state backends for light-matter (scattering) simulations.

The `LightScattering` physics module splits each timestep into an
internal-state update and a momentum update, in the spirit of
multiphysics solvers. This module owns the internal-state side.

The interface contract: `LightMatterSystem.stimulated_rates` reduces
all laser/magnetic-field geometry to a per-atom, per-beam one-way
stimulated rate matrix `W` (s^-1) for an effective two-level atom:

- stimulated absorption from beam `b` occurs at rate `W_b * rho_gg`,
- stimulated emission into beam `b` at rate `W_b * rho_ee`,
- spontaneous emission at rate `Gamma * rho_ee`.

An `InternalStateModel` turns `W` into (a) an evolved internal state and
(b) the photon `ScatteringEvents` of the step, from which the momentum
backend applies recoil kicks: `+hbar k_b` per absorption, `-hbar k_b`
per stimulated emission into beam `b`, and `hbar k` in a random
direction per spontaneous photon.

Three backends are provided:

- `AdiabaticSteadyState` — populations follow the local steady state
  `p = W_tot / (Gamma + 2 W_tot)` instantly; photon numbers are Poisson
  samples. Valid when internal dynamics (`~1/Gamma`) are much faster
  than motional/field timescales, which is almost always true in a MOT.
  Allows the largest timesteps.
- `RateEquationPopulations` — per-atom excited population integrated
  exactly (exponential relaxation toward the local steady state each
  step). Captures transients, e.g. pulsed or switched beams.
- `StochasticJumpState` — per-atom discrete ground/excited state with
  at most one jump sampled per step (kinetic Monte Carlo, the
  rate-equation analog of a quantum-jump trajectory). Requires
  `dt << 1/Gamma`; photon statistics are exact per trajectory.

All backends treat beams as mutually incoherent (no standing-wave or
polarization-gradient interference), so sub-Doppler mechanisms are out
of scope. A density-matrix / multilevel MCWF backend can be added by
implementing the same three-method interface with a richer opaque
`state` array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScatteringEvents:
    """Photon counts of one timestep for an ensemble of `N` atoms.

    - `absorbed_per_beam`: shape `(N, n_beams)`, photons absorbed from
      each beam (recoil `+hbar k_b` each).
    - `stimulated_per_beam`: shape `(N, n_beams)`, photons emitted into
      each beam mode by stimulated emission (recoil `-hbar k_b` each).
    - `spontaneous`: shape `(N,)`, spontaneously emitted photons
      (recoil `hbar k` in a random direction each).
    """

    absorbed_per_beam: NDArray[np.int64]
    stimulated_per_beam: NDArray[np.int64]
    spontaneous: NDArray[np.int64]

    @property
    def total_scattered(self) -> NDArray[np.int64]:
        """Absorbed photon count per atom (one per full cycle)."""

        return np.sum(self.absorbed_per_beam, axis=-1)


def steady_state_excited_fraction(
    total_stimulated_rate_per_s: NDArray[np.float64],
    linewidth_rad_s: float,
) -> NDArray[np.float64]:
    """Two-level steady-state population `p = W_tot / (Gamma + 2 W_tot)`."""

    w_tot = np.asarray(total_stimulated_rate_per_s, dtype=float)
    return w_tot / (linewidth_rad_s + 2.0 * w_tot + _TINY)


class InternalStateModel(ABC):
    """Abstract internal-state backend.

    `state` is an opaque per-atom array owned by the backend; the driver
    only threads it through `step` and masks lost atoms by passing only
    active rows.
    """

    name: str = "internal_state"

    @abstractmethod
    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return the initial internal state for `n_atoms` atoms."""

    @abstractmethod
    def step(
        self,
        state: NDArray[np.float64],
        stimulated_rates_per_s: NDArray[np.float64],
        linewidth_rad_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], ScatteringEvents]:
        """Advance the internal state by `dt_s` and report photon events.

        `stimulated_rates_per_s` has shape `(N, n_beams)` and is assumed
        constant over the step.
        """

    @abstractmethod
    def excited_fraction(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-atom excited-state population (diagnostic), shape `(N,)`."""


class AdiabaticSteadyState(InternalStateModel):
    """Populations locked to the local steady state; Poisson photon counts.

    Steady state of the two-level rate equations with total pump
    `W_tot = sum_b W_b`: `p = W_tot / (Gamma + 2 W_tot)`. Per-step photon
    numbers are independent Poisson draws with means `W_b (1 - p) dt`
    (absorption), `W_b p dt` (stimulated emission), `Gamma p dt`
    (spontaneous). Means and the leading momentum diffusion are correct;
    absorption-emission pairing correlations are neglected.
    """

    name = "adiabatic_steady_state"

    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.zeros(n_atoms, dtype=float)

    def step(
        self,
        state: NDArray[np.float64],
        stimulated_rates_per_s: NDArray[np.float64],
        linewidth_rad_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], ScatteringEvents]:
        w = np.asarray(stimulated_rates_per_s, dtype=float)
        w_tot = np.sum(w, axis=-1)
        p = steady_state_excited_fraction(w_tot, linewidth_rad_s)
        events = _poisson_events(w, p, linewidth_rad_s, dt_s, rng)
        return p, events

    def excited_fraction(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(state, dtype=float)


class RateEquationPopulations(InternalStateModel):
    """Two-level populations integrated as an ODE per atom.

    `dp/dt = W_tot - (Gamma + 2 W_tot) p` is solved exactly over the
    step for piecewise-constant rates:
    `p(t + dt) = p_ss + (p - p_ss) exp(-lambda dt)` with
    `lambda = Gamma + 2 W_tot`. Photon numbers use the time-averaged
    population over the step, so pulsed/switched beams are handled
    without a stability limit on `dt`.
    """

    name = "rate_equation"

    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.zeros(n_atoms, dtype=float)

    def step(
        self,
        state: NDArray[np.float64],
        stimulated_rates_per_s: NDArray[np.float64],
        linewidth_rad_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], ScatteringEvents]:
        w = np.asarray(stimulated_rates_per_s, dtype=float)
        p = np.asarray(state, dtype=float)
        w_tot = np.sum(w, axis=-1)
        decay = linewidth_rad_s + 2.0 * w_tot
        p_ss = steady_state_excited_fraction(w_tot, linewidth_rad_s)
        x = decay * dt_s
        exp_x = np.exp(-x)
        p_next = p_ss + (p - p_ss) * exp_x
        # Time average of p over the step (exact for constant rates).
        relax = np.where(x > 1.0e-12, (1.0 - exp_x) / np.maximum(x, _TINY), 1.0)
        p_avg = p_ss + (p - p_ss) * relax
        events = _poisson_events(w, p_avg, linewidth_rad_s, dt_s, rng)
        return p_next, events

    def excited_fraction(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(state, dtype=float)


class StochasticJumpState(InternalStateModel):
    """Discrete ground/excited trajectory with kinetic-MC jumps.

    Each atom is either ground (0) or excited (1). Per step, at most one
    jump is sampled: a ground atom absorbs from beam `b` with
    probability `~W_b dt`; an excited atom decays with probability
    `~(Gamma + W_tot) dt`, spontaneously with branching
    `Gamma / (Gamma + W_tot)`, else by stimulated emission into a beam.
    This is the rate-equation analog of a quantum-jump (Monte Carlo
    wavefunction) trajectory and gives exact per-trajectory photon
    statistics, at the cost of requiring `dt << 1/Gamma` (~26 ns for Rb).
    A warning-level check enforces `(Gamma + W_tot) dt < 0.1`.
    """

    name = "stochastic_jump"

    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.zeros(n_atoms, dtype=float)

    def step(
        self,
        state: NDArray[np.float64],
        stimulated_rates_per_s: NDArray[np.float64],
        linewidth_rad_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], ScatteringEvents]:
        w = np.asarray(stimulated_rates_per_s, dtype=float)
        excited = np.asarray(state, dtype=float) > 0.5
        n_atoms, n_beams = w.shape
        w_tot = np.sum(w, axis=-1)

        max_rate = float(np.max((linewidth_rad_s + w_tot) * dt_s, initial=0.0))
        if max_rate > 0.1:
            raise ValueError(
                "StochasticJumpState requires (Gamma + W_tot) * dt < 0.1; got "
                f"{max_rate:.3f}. Reduce timestep_s or use RateEquationPopulations."
            )

        absorbed = np.zeros((n_atoms, n_beams), dtype=np.int64)
        stimulated = np.zeros((n_atoms, n_beams), dtype=np.int64)
        spontaneous = np.zeros(n_atoms, dtype=np.int64)
        new_state = excited.copy()

        u_jump = rng.random(n_atoms)
        u_channel = rng.random(n_atoms)

        # Ground atoms: absorption with total probability 1 - exp(-W_tot dt).
        ground = ~excited
        p_abs = 1.0 - np.exp(-w_tot * dt_s)
        jumps_g = ground & (u_jump < p_abs)
        if np.any(jumps_g):
            beam_idx = _choose_channel(w[jumps_g], u_channel[jumps_g])
            absorbed[np.flatnonzero(jumps_g), beam_idx] = 1
            new_state[jumps_g] = True

        # Excited atoms: decay (spontaneous or stimulated).
        total_decay = linewidth_rad_s + w_tot
        p_dec = 1.0 - np.exp(-total_decay * dt_s)
        jumps_e = excited & (u_jump < p_dec)
        if np.any(jumps_e):
            # Channel 0 = spontaneous, channels 1..n_beams = stimulated.
            channels = np.concatenate(
                [
                    np.full((int(np.sum(jumps_e)), 1), linewidth_rad_s),
                    w[jumps_e],
                ],
                axis=1,
            )
            channel_idx = _choose_channel(channels, u_channel[jumps_e])
            rows = np.flatnonzero(jumps_e)
            spont_rows = rows[channel_idx == 0]
            spontaneous[spont_rows] = 1
            stim_mask = channel_idx > 0
            stimulated[rows[stim_mask], channel_idx[stim_mask] - 1] = 1
            new_state[jumps_e] = False

        events = ScatteringEvents(
            absorbed_per_beam=absorbed,
            stimulated_per_beam=stimulated,
            spontaneous=spontaneous,
        )
        return new_state.astype(float), events

    def excited_fraction(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(state, dtype=float)


def sample_recoil_velocity_kicks(
    events: ScatteringEvents,
    beam_directions: NDArray[np.float64],
    recoil_velocity_m_per_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Convert photon events into per-atom velocity kicks, shape `(N, 3)`.

    Absorption kicks `+v_rec` along each beam, stimulated emission
    `-v_rec` along the beam, and one isotropically random `v_rec` kick
    per spontaneous photon (sampled exactly, photon by photon).
    """

    directed_counts = events.absorbed_per_beam - events.stimulated_per_beam
    kicks = recoil_velocity_m_per_s * (directed_counts.astype(float) @ beam_directions)

    spontaneous = np.asarray(events.spontaneous)
    total_spont = int(np.sum(spontaneous))
    if total_spont > 0:
        directions = _random_unit_vectors(total_spont, rng)
        atom_idx = np.repeat(np.arange(spontaneous.shape[0]), spontaneous)
        np.add.at(kicks, atom_idx, recoil_velocity_m_per_s * directions)
    return kicks


_TINY = 1.0e-300


def _poisson_events(
    w: NDArray[np.float64],
    excited_population: NDArray[np.float64],
    linewidth_rad_s: float,
    dt_s: float,
    rng: np.random.Generator,
) -> ScatteringEvents:
    """Independent Poisson photon numbers given the excited population."""

    p = np.asarray(excited_population, dtype=float)[:, np.newaxis]
    absorbed = rng.poisson(w * (1.0 - p) * dt_s)
    stimulated = rng.poisson(w * p * dt_s)
    spontaneous = rng.poisson(linewidth_rad_s * p[:, 0] * dt_s)
    return ScatteringEvents(
        absorbed_per_beam=absorbed.astype(np.int64),
        stimulated_per_beam=stimulated.astype(np.int64),
        spontaneous=spontaneous.astype(np.int64),
    )


def _choose_channel(
    rates: NDArray[np.float64], uniforms: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Pick one channel per row with probability proportional to `rates`."""

    totals = np.sum(rates, axis=-1, keepdims=True)
    cumulative = np.cumsum(rates, axis=-1) / np.maximum(totals, _TINY)
    return np.sum(uniforms[:, np.newaxis] > cumulative, axis=-1).astype(np.int64)


def _random_unit_vectors(
    count: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample `count` isotropic unit vectors."""

    cos_theta = rng.uniform(-1.0, 1.0, size=count)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=count)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1
    )
