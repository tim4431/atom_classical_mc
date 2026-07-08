"""Magneto-optical trap (MOT) system assembly and coupled simulation.

This module glues the physics "backends" together, COMSOL-style:

- **Field geometry** (`MOTSystem`): an `AtomSpecies`, a set of
  `LaserBeam`s, and a set of `MagneticFieldConfig`s. Its job is to
  reduce all geometry to the per-atom, per-beam stimulated rate matrix
  `W` consumed by the internal-state backend (see
  `src.internal_state` for the interface contract).
- **Internal-state backend** (`InternalStateModel`): evolves atomic
  populations and emits photon `ScatteringEvents`.
- **Momentum backend** (`run_mot_simulation` loop): semi-implicit Euler
  propagation under conservative trap forces plus per-photon recoil
  kicks from the scattering events.

Zeeman/polarization model (standard effective two-level MOT theory):
for each beam, the circular-polarization content (`LaserBeam.helicity`)
is decomposed into sigma+, pi, sigma- components relative to the local
magnetic field direction. A component `q` in `(+1, 0, -1)` sees the
detuning

    delta_q = 2 pi detuning_hz - k . v - q * mu_eff * |B| / hbar,

i.e. laser detuning, Doppler shift, and the Zeeman shift of the
stretched cycling transition (`AtomSpecies.mu_eff_j_per_t`). The one-way
stimulated rate of beam `b` is

    W_b = (Gamma / 2) * sum_q s_b(r) f_q / (1 + (2 delta_q / Gamma)^2),

and saturation competition between beams is handled by the internal
state backend through `p = W_tot / (Gamma + 2 W_tot)`. Beams are
mutually incoherent: no standing-wave or polarization-gradient
(sub-Doppler) physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import HBAR_J_S
from .fields import MagneticFieldConfig, total_magnetic_field
from .internal_state import (
    AdiabaticSteadyState,
    InternalStateModel,
    sample_recoil_velocity_kicks,
)
from .laser import LaserBeam
from .sampling import sample_thermal_velocities
from .species import AtomSpecies
from .trap import TrapConfig, total_force
from .units import joule_to_microkelvin


@dataclass(frozen=True)
class MOTSystem:
    """Species + laser beams + magnetic fields, with the rate engine."""

    species: AtomSpecies
    beams: Sequence[LaserBeam]
    magnetic_fields: Sequence[MagneticFieldConfig] = ()

    def __post_init__(self) -> None:
        beams = tuple(self.beams)
        if not beams:
            raise ValueError("MOTSystem requires at least one laser beam.")
        fields = (
            (self.magnetic_fields,)
            if isinstance(self.magnetic_fields, MagneticFieldConfig)
            else tuple(self.magnetic_fields)
        )
        object.__setattr__(self, "beams", beams)
        object.__setattr__(self, "magnetic_fields", fields)

    @property
    def beam_directions(self) -> NDArray[np.float64]:
        """Unit propagation vectors, shape `(n_beams, 3)`."""

        return np.stack([np.asarray(b.direction, dtype=float) for b in self.beams])

    def magnetic_field_at(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        if not self.magnetic_fields:
            return np.zeros_like(positions)
        return total_magnetic_field(self.magnetic_fields, positions, time_s=time_s)

    def stimulated_rates(
        self,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """One-way stimulated rate matrix `W`, shape `(N, n_beams)`, in s^-1."""

        positions = _as_positions(positions_m)
        velocities = np.asarray(velocities_m_per_s, dtype=float)
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match the shape of positions.")

        gamma = self.species.linewidth_rad_s
        k = self.species.wavenumber_rad_per_m
        mu_over_hbar = self.species.mu_eff_j_per_t / HBAR_J_S

        b_vec = self.magnetic_field_at(positions, time_s=time_s)
        b_mag = np.linalg.norm(b_vec, axis=-1)
        # Where B ~ 0 the quantization axis is arbitrary and the Zeeman
        # shift vanishes for every component, so any unit vector works.
        safe = b_mag > 1.0e-15
        b_hat = np.where(
            safe[..., np.newaxis],
            b_vec / np.maximum(b_mag, 1.0e-300)[..., np.newaxis],
            np.array([0.0, 0.0, 1.0]),
        )
        zeeman_rad_s = mu_over_hbar * b_mag

        rates = np.empty(positions.shape[:-1] + (len(self.beams),), dtype=float)
        for index, beam in enumerate(self.beams):
            s_local = beam.saturation_at(positions, time_s=time_s)
            cos_theta = np.sum(beam.direction * b_hat, axis=-1)
            fractions = _polarization_fractions(beam.helicity, cos_theta)
            doppler_rad_s = k * np.sum(velocities * beam.direction, axis=-1)
            base_detuning = 2.0 * np.pi * beam.detuning_hz - doppler_rad_s

            w_beam = np.zeros(positions.shape[:-1], dtype=float)
            for q, fraction in zip((+1.0, 0.0, -1.0), fractions):
                delta_q = base_detuning - q * zeeman_rad_s
                lorentz = 1.0 / (1.0 + (2.0 * delta_q / gamma) ** 2)
                w_beam = w_beam + s_local * fraction * lorentz
            rates[..., index] = 0.5 * gamma * w_beam
        return rates

    def mean_radiation_force(
        self,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Steady-state mean radiation-pressure force, shape `(N, 3)`, newtons.

        `F = hbar k sum_b k_hat_b W_b (1 - 2 p)` with the adiabatic
        steady-state population. Deterministic; useful for computing
        damping/spring coefficients and for tests. Recoil diffusion is
        not included here — it emerges from the stochastic kicks in the
        full simulation.
        """

        w = self.stimulated_rates(positions_m, velocities_m_per_s, time_s=time_s)
        w_tot = np.sum(w, axis=-1)
        p = w_tot / (self.species.linewidth_rad_s + 2.0 * w_tot + 1.0e-300)
        net_rates = w * (1.0 - 2.0 * p)[..., np.newaxis]
        hbar_k = HBAR_J_S * self.species.wavenumber_rad_per_m
        return hbar_k * (net_rates @ self.beam_directions)


def _polarization_fractions(
    helicity: float, cos_theta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Intensity fractions `(f_+1, f_0, f_-1)` relative to the B axis.

    Incoherent mixture of the two circular components with weights
    `(1 +/- h)/2`; a pure circular beam at angle `theta` to the field
    axis has `f_{+/-} = (1 +/- cos)^2 / 4`, `f_pi = sin^2 / 2`.
    """

    c = np.asarray(cos_theta, dtype=float)
    w_plus = 0.5 * (1.0 + helicity)
    w_minus = 0.5 * (1.0 - helicity)
    plus_sq = 0.25 * (1.0 + c) ** 2
    minus_sq = 0.25 * (1.0 - c) ** 2
    f_sigma_plus = w_plus * plus_sq + w_minus * minus_sq
    f_sigma_minus = w_plus * minus_sq + w_minus * plus_sq
    f_pi = 0.5 * (1.0 - c * c)
    return f_sigma_plus, f_pi, f_sigma_minus


@dataclass(frozen=True)
class MOTSimulationConfig:
    """Configuration for a coupled internal-state + motion simulation.

    The initial ensemble is a Gaussian cloud (`initial_cloud_sigma_m`,
    scalar or per-axis) centered at `initial_center_m`, with
    Maxwell-Boltzmann velocities at `initial_temperature_uK` plus an
    optional mean drift `initial_mean_velocity_m_per_s` (for capture
    studies of launched atoms).

    There is no energy-based loss criterion: radiation pressure is not
    conservative. Atoms are lost only by leaving `loss_radius_m` (if
    set) around `boundary_center_m`.

    Timestep guidance: `AdiabaticSteadyState` and
    `RateEquationPopulations` need `dt` small compared to motional
    timescales and to the time over which detunings change (typically
    0.1-1 us is safe); `StochasticJumpState` additionally requires
    `dt << 1/Gamma` (a few ns for Rb).
    """

    initial_temperature_uK: float
    timestep_s: float
    duration_s: float
    ensemble_size: int
    initial_cloud_sigma_m: ArrayLike = (0.0, 0.0, 0.0)
    initial_center_m: ArrayLike = (0.0, 0.0, 0.0)
    initial_mean_velocity_m_per_s: ArrayLike = (0.0, 0.0, 0.0)
    random_seed: int | None = None
    loss_radius_m: float | None = None
    boundary_center_m: ArrayLike = (0.0, 0.0, 0.0)
    store_trajectories: bool = False
    trajectory_stride: int = 1

    def __post_init__(self) -> None:
        if self.initial_temperature_uK < 0.0:
            raise ValueError("initial_temperature_uK must be non-negative.")
        if self.timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive.")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        if self.loss_radius_m is not None and self.loss_radius_m <= 0.0:
            raise ValueError("loss_radius_m must be positive when provided.")
        if self.trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive.")

        sigma = np.asarray(self.initial_cloud_sigma_m, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(3, float(sigma))
        if sigma.shape != (3,) or np.any(sigma < 0.0):
            raise ValueError(
                "initial_cloud_sigma_m must be a non-negative scalar or 3-vector."
            )
        object.__setattr__(self, "initial_cloud_sigma_m", sigma)

        for attr in ("initial_center_m", "initial_mean_velocity_m_per_s", "boundary_center_m"):
            vec = np.asarray(getattr(self, attr), dtype=float)
            if vec.shape != (3,):
                raise ValueError(f"{attr} must be a 3-vector.")
            object.__setattr__(self, attr, vec)


@dataclass(frozen=True)
class MOTSimulationResult:
    """Outputs of `run_mot_simulation`.

    Temperatures are kinetic (`m <v^2> / 3 k_B`) with the ensemble mean
    velocity subtracted, over surviving atoms and over all atoms.
    `scattered_photons` counts absorbed photons per atom over the run.
    """

    survival_probability: float
    loss_fraction: float
    initial_temperature_uK_all: float
    final_temperature_uK_survivors: float
    final_temperature_uK_all: float
    initial_positions_m: NDArray[np.float64]
    initial_velocities_m_per_s: NDArray[np.float64]
    final_positions_m: NDArray[np.float64]
    final_velocities_m_per_s: NDArray[np.float64]
    lost: NDArray[np.bool_]
    scattered_photons: NDArray[np.int64]
    final_excited_fraction: NDArray[np.float64]
    duration_s: float
    trajectory_times_s: NDArray[np.float64] | None = None
    trajectory_positions_m: NDArray[np.float64] | None = None
    trajectory_velocities_m_per_s: NDArray[np.float64] | None = None
    trajectory_lost: NDArray[np.bool_] | None = None
    trajectory_temperature_uK: NDArray[np.float64] | None = None

    @property
    def mean_scattered_photons(self) -> float:
        return float(np.mean(self.scattered_photons))


def run_mot_simulation(
    system: MOTSystem,
    config: MOTSimulationConfig,
    traps: TrapConfig | Iterable[TrapConfig] = (),
    internal_model: InternalStateModel | None = None,
) -> MOTSimulationResult:
    """Run the coupled internal-state + momentum Monte Carlo simulation.

    Per timestep (operator splitting):

    1. Evaluate `B(r, t)` and the stimulated rate matrix `W(r, v, t)`.
    2. Internal-state backend: advance populations, sample photon events.
    3. Momentum backend: recoil velocity kicks from the events, plus the
       conservative trap force, via semi-implicit Euler
       (`v += a dt + kicks; r += v dt`).

    `traps` may add conservative dipole-trap forces on top of the MOT
    light forces (their light is *not* included in the scattering
    model). `internal_model` defaults to `AdiabaticSteadyState()`.
    """

    model = AdiabaticSteadyState() if internal_model is None else internal_model
    trap_list = _trap_list(traps)
    rng = np.random.default_rng(config.random_seed)
    mass = system.species.mass_kg
    gamma = system.species.linewidth_rad_s
    recoil_v = system.species.recoil_velocity_m_per_s
    beam_dirs = system.beam_directions

    positions = config.initial_center_m + rng.normal(
        size=(config.ensemble_size, 3)
    ) * config.initial_cloud_sigma_m
    velocities = (
        sample_thermal_velocities(
            config.initial_temperature_uK, config.ensemble_size, rng, mass_kg=mass
        )
        + config.initial_mean_velocity_m_per_s
    )
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()

    state = np.asarray(model.initialize(config.ensemble_size, rng), dtype=float)
    lost = _outside_boundary(positions, config)
    scattered = np.zeros(config.ensemble_size, dtype=np.int64)
    excited = np.zeros(config.ensemble_size, dtype=float)

    trajectory_times: list[float] = []
    trajectory_positions: list[NDArray[np.float64]] = []
    trajectory_velocities: list[NDArray[np.float64]] = []
    trajectory_lost: list[NDArray[np.bool_]] = []
    trajectory_temperature: list[float] = []

    def _record(time_s: float) -> None:
        trajectory_times.append(time_s)
        trajectory_positions.append(positions.copy())
        trajectory_velocities.append(velocities.copy())
        trajectory_lost.append(lost.copy())
        trajectory_temperature.append(
            _kinetic_temperature_uK(velocities[~lost], mass)
        )

    if config.store_trajectories:
        _record(0.0)

    t = 0.0
    step_index = 0
    while t < config.duration_s:
        dt = min(config.timestep_s, config.duration_s - t)
        active = np.flatnonzero(~lost)

        if active.size > 0:
            r_active = positions[active]
            v_active = velocities[active]

            # Internal-state backend.
            rates = system.stimulated_rates(r_active, v_active, time_s=t)
            state_active, events = model.step(state[active], rates, gamma, dt, rng)
            state[active] = state_active
            scattered[active] += events.total_scattered
            excited[active] = model.excited_fraction(state_active)

            # Momentum backend: recoil kicks + conservative trap force.
            kicks = sample_recoil_velocity_kicks(events, beam_dirs, recoil_v, rng)
            if trap_list:
                acceleration = total_force(trap_list, r_active, time_s=t) / mass
            else:
                acceleration = 0.0
            v_active = v_active + acceleration * dt + kicks
            r_active = r_active + v_active * dt

            positions[active] = r_active
            velocities[active] = v_active
            lost[active] = _outside_boundary(r_active, config)

        t += dt
        step_index += 1
        if config.store_trajectories and step_index % config.trajectory_stride == 0:
            _record(t)

    if config.store_trajectories and (
        not trajectory_times or not np.isclose(trajectory_times[-1], config.duration_s)
    ):
        _record(config.duration_s)

    survivors = ~lost
    survival_probability = float(np.mean(survivors))

    return MOTSimulationResult(
        survival_probability=survival_probability,
        loss_fraction=1.0 - survival_probability,
        initial_temperature_uK_all=_kinetic_temperature_uK(initial_velocities, mass),
        final_temperature_uK_survivors=_kinetic_temperature_uK(
            velocities[survivors], mass
        ),
        final_temperature_uK_all=_kinetic_temperature_uK(velocities, mass),
        initial_positions_m=initial_positions,
        initial_velocities_m_per_s=initial_velocities,
        final_positions_m=positions,
        final_velocities_m_per_s=velocities,
        lost=lost,
        scattered_photons=scattered,
        final_excited_fraction=excited,
        duration_s=float(config.duration_s),
        trajectory_times_s=(
            np.asarray(trajectory_times, dtype=float)
            if config.store_trajectories
            else None
        ),
        trajectory_positions_m=(
            np.asarray(trajectory_positions, dtype=float)
            if config.store_trajectories
            else None
        ),
        trajectory_velocities_m_per_s=(
            np.asarray(trajectory_velocities, dtype=float)
            if config.store_trajectories
            else None
        ),
        trajectory_lost=(
            np.asarray(trajectory_lost, dtype=bool)
            if config.store_trajectories
            else None
        ),
        trajectory_temperature_uK=(
            np.asarray(trajectory_temperature, dtype=float)
            if config.store_trajectories
            else None
        ),
    )


def _kinetic_temperature_uK(
    velocities_m_per_s: NDArray[np.float64], mass_kg: float
) -> float:
    """Kinetic temperature with the ensemble mean (drift) removed."""

    if velocities_m_per_s.size == 0:
        return float("nan")
    centered = velocities_m_per_s - np.mean(velocities_m_per_s, axis=0)
    mean_speed_sq = float(np.mean(np.sum(centered * centered, axis=-1)))
    energy_per_axis_j = mass_kg * mean_speed_sq / 3.0
    return float(joule_to_microkelvin(energy_per_axis_j))


def _outside_boundary(
    positions_m: NDArray[np.float64], config: MOTSimulationConfig
) -> NDArray[np.bool_]:
    if config.loss_radius_m is None:
        return np.zeros(positions_m.shape[:-1], dtype=bool)
    offsets = positions_m - config.boundary_center_m
    return np.linalg.norm(offsets, axis=-1) > config.loss_radius_m


def _trap_list(traps: TrapConfig | Iterable[TrapConfig]) -> list[TrapConfig]:
    if isinstance(traps, TrapConfig):
        return [traps]
    return list(traps)


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
