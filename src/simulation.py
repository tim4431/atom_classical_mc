"""Monte Carlo trajectory propagation and heating/loss analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import RB87_MASS_KG
from .internal_state import (
    AdiabaticSteadyState,
    InternalStateModel,
    sample_recoil_velocity_kicks,
)
from .light_matter import LightMatterSystem
from .ramp import RampSequence
from .sampling import sample_thermal_positions_harmonic, sample_thermal_velocities
from .trap import (
    GaussianTrap,
    MovingGaussianTrap,
    TrapConfig,
    total_force,
    total_potential,
)
from .units import joule_to_microkelvin


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a Monte Carlo trajectory simulation.

    Initial positions come from the local harmonic approximation of the
    traps by default. Set `initial_cloud_sigma_m` (scalar or per-axis)
    to instead sample a free Gaussian cloud around `initial_center_m` —
    required when there are no traps (pure light-force runs) and useful
    for capture studies. `initial_mean_velocity_m_per_s` adds a drift to
    the Maxwell-Boltzmann velocities (e.g. launched atoms).
    """

    initial_temperature_uK: float
    timestep_s: float
    duration_s: float
    ensemble_size: int
    random_seed: int | None = None
    mass_kg: float = RB87_MASS_KG
    loss_radius_m: float | None = None
    boundary_center_m: ArrayLike = (0.0, 0.0, 0.0)
    initial_center_m: ArrayLike | None = None
    initial_cloud_sigma_m: ArrayLike | None = None
    initial_mean_velocity_m_per_s: ArrayLike = (0.0, 0.0, 0.0)
    reject_initially_lost: bool = True
    max_initial_resampling_rounds: int = 100
    store_trajectories: bool = False
    trajectory_stride: int = 1
    track_energy_loss: bool = True

    def __post_init__(self) -> None:
        if self.initial_temperature_uK < 0.0:
            raise ValueError("initial_temperature_uK must be non-negative.")
        if self.timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive.")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if self.loss_radius_m is not None and self.loss_radius_m <= 0.0:
            raise ValueError("loss_radius_m must be positive when provided.")
        if self.trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive.")
        if self.max_initial_resampling_rounds <= 0:
            raise ValueError("max_initial_resampling_rounds must be positive.")

        boundary_center = np.asarray(self.boundary_center_m, dtype=float)
        if boundary_center.shape != (3,):
            raise ValueError("boundary_center_m must be a 3-vector in meters.")
        object.__setattr__(self, "boundary_center_m", boundary_center)

        if self.initial_center_m is not None:
            initial_center = np.asarray(self.initial_center_m, dtype=float)
            if initial_center.shape != (3,):
                raise ValueError("initial_center_m must be a 3-vector in meters.")
            object.__setattr__(self, "initial_center_m", initial_center)

        if self.initial_cloud_sigma_m is not None:
            sigma = np.asarray(self.initial_cloud_sigma_m, dtype=float)
            if sigma.ndim == 0:
                sigma = np.full(3, float(sigma))
            if sigma.shape != (3,) or np.any(sigma < 0.0):
                raise ValueError(
                    "initial_cloud_sigma_m must be a non-negative scalar or 3-vector."
                )
            object.__setattr__(self, "initial_cloud_sigma_m", sigma)

        mean_velocity = np.asarray(self.initial_mean_velocity_m_per_s, dtype=float)
        if mean_velocity.shape != (3,):
            raise ValueError("initial_mean_velocity_m_per_s must be a 3-vector.")
        object.__setattr__(self, "initial_mean_velocity_m_per_s", mean_velocity)


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate and per-particle simulation outputs.

    Temperature fields come in two flavors:

    - `*_survivors`: average over atoms not flagged lost at the relevant time.
    - `*_all`: average over the entire ensemble (including lost atoms).

    The legacy properties `final_temperature_uK` and `temperature_gain_uK`
    return the survivors-only values to match older callers.

    When a `LightMatterSystem` is attached, `scattered_photons` counts
    absorbed photons per atom over the run and `final_excited_fraction`
    is the last excited-state population reported by the internal-state
    backend; both are `None` for purely conservative runs. Energy fields
    then measure the *conservative* mechanical energy (kinetic + trap
    potential), so energy gains include recoil heating by the light.
    """

    survival_probability: float
    loss_fraction: float
    mean_energy_gain_uK: float
    median_energy_gain_uK: float
    initial_temperature_uK_survivors: float
    initial_temperature_uK_all: float
    final_temperature_uK_survivors: float
    final_temperature_uK_all: float
    initial_energies_uK: NDArray[np.float64]
    final_energies_uK: NDArray[np.float64]
    energy_gains_uK: NDArray[np.float64]
    initial_positions_m: NDArray[np.float64]
    initial_velocities_m_per_s: NDArray[np.float64]
    final_positions_m: NDArray[np.float64]
    final_velocities_m_per_s: NDArray[np.float64]
    lost: NDArray[np.bool_]
    duration_s: float = 0.0
    initial_rejected_count: int = 0
    initial_rejection_fraction: float = 0.0
    trajectory_times_s: NDArray[np.float64] | None = None
    trajectory_positions_m: NDArray[np.float64] | None = None
    trajectory_velocities_m_per_s: NDArray[np.float64] | None = None
    trajectory_lost: NDArray[np.bool_] | None = None
    scattered_photons: NDArray[np.int64] | None = None
    final_excited_fraction: NDArray[np.float64] | None = None

    @property
    def final_temperature_uK(self) -> float:
        return self.final_temperature_uK_survivors

    @property
    def temperature_gain_uK(self) -> float:
        return self.temperature_gain_uK_at(survivors_only=True)

    def temperature_gain_uK_at(self, survivors_only: bool = True) -> float:
        if survivors_only:
            return (
                self.final_temperature_uK_survivors
                - self.initial_temperature_uK_survivors
            )
        return self.final_temperature_uK_all - self.initial_temperature_uK_all


def run_simulation(
    *args,
    **kwargs,
) -> SimulationResult:
    """Run a classical Monte Carlo simulation.

    Two calling conventions are supported:

    - New: `run_simulation(traps, config)` where `traps` is one `TrapConfig`
      or an iterable of them (may be empty when `scattering` is given).
      Time-dependent traps (e.g. `MovingGaussianTrap`, `AstigmaticAODTrap`)
      carry their own ramp internally.
    - Legacy: `run_simulation(static_trap, moving_trap_base, ramp, config)`.
      Builds a `MovingGaussianTrap` from `moving_trap_base + ramp` and calls
      the new form.

    Optional light-force physics (keyword-only):

    - `scattering`: a `LightMatterSystem` (laser beams + magnetic fields +
      species). Each step the system is reduced to per-atom, per-beam
      stimulated rates, the internal-state backend evolves populations and
      samples photon events, and the momentum update applies the recoil
      kicks on top of the conservative trap forces. `config.mass_kg` must
      match `scattering.species.mass_kg`.
    - `internal_model`: an `InternalStateModel` backend; defaults to
      `AdiabaticSteadyState()`.

    Because radiation pressure is non-conservative, attaching `scattering`
    disables the energy-based loss criterion (equivalent to
    `track_energy_loss=False`); atoms are then lost only by leaving
    `loss_radius_m`. Recover trap-survival statistics post-hoc with
    `analysis.bound_to_trap` / `capture_probability`.
    """

    scattering = kwargs.pop("scattering", None)
    internal_model = kwargs.pop("internal_model", None)
    if scattering is not None and not isinstance(scattering, LightMatterSystem):
        raise TypeError("scattering must be a LightMatterSystem.")
    if internal_model is not None and not isinstance(
        internal_model, InternalStateModel
    ):
        raise TypeError("internal_model must be an InternalStateModel.")
    if internal_model is not None and scattering is None:
        raise TypeError("internal_model requires a scattering system.")

    traps, config = _normalize_simulation_args(args, kwargs)
    if not traps and scattering is None:
        raise ValueError("At least one trap or a scattering system is required.")
    if scattering is not None and config.mass_kg != scattering.species.mass_kg:
        raise ValueError(
            "SimulationConfig.mass_kg does not match scattering.species.mass_kg "
            f"({config.mass_kg} vs {scattering.species.mass_kg}); set "
            "mass_kg=scattering.species.mass_kg (the default is Rb87)."
        )
    return _run_simulation_core(traps, config, scattering, internal_model)


def _normalize_simulation_args(
    args: tuple, kwargs: dict
) -> tuple[list[TrapConfig], SimulationConfig]:
    if "config" in kwargs:
        config = kwargs.pop("config")
    elif args and isinstance(args[-1], SimulationConfig):
        config = args[-1]
        args = args[:-1]
    else:
        raise TypeError("run_simulation requires a SimulationConfig argument.")

    if "traps" in kwargs:
        traps_arg = kwargs.pop("traps")
    elif len(args) == 1:
        traps_arg = args[0]
    elif len(args) == 3:
        static_trap, moving_trap_base, ramp = args
        if not isinstance(moving_trap_base, GaussianTrap):
            raise TypeError(
                "Legacy run_simulation form requires moving_trap_base to be a "
                "GaussianTrap."
            )
        if not isinstance(ramp, RampSequence):
            raise TypeError(
                "Legacy run_simulation form requires a RampSequence for the ramp."
            )
        moving = MovingGaussianTrap(template=moving_trap_base, ramp=ramp)
        traps_arg = _trap_list(static_trap) + [moving]
    else:
        raise TypeError(
            "run_simulation expects either (traps, config) or "
            "(static_trap, moving_trap_base, ramp, config)."
        )

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    return _trap_list(traps_arg), config


def _run_simulation_core(
    traps: list[TrapConfig],
    config: SimulationConfig,
    scattering: LightMatterSystem | None = None,
    internal_model: InternalStateModel | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(config.random_seed)

    # The energy criterion assumes purely conservative dynamics; any
    # attached light force invalidates it (see run_simulation docstring).
    track_energy = config.track_energy_loss and scattering is None and bool(traps)

    positions, velocities = _sample_initial_ensemble(
        traps, config, config.ensemble_size, rng
    )
    initial_rejected_count = 0
    if config.reject_initially_lost:
        positions, velocities, initial_rejected_count = _resample_initially_lost(
            traps, positions, velocities, config, rng, track_energy
        )
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()

    initial_energies_j = _mechanical_energy(
        traps, positions, velocities, config.mass_kg, time_s=0.0
    )
    final_energies_j = initial_energies_j.copy()
    lost = _initial_lost_flags(traps, positions, velocities, config, track_energy)

    if scattering is not None:
        model = AdiabaticSteadyState() if internal_model is None else internal_model
        internal_state = np.asarray(
            model.initialize(config.ensemble_size, rng), dtype=float
        )
        scattered_photons = np.zeros(config.ensemble_size, dtype=np.int64)
        excited_fraction = np.zeros(config.ensemble_size, dtype=float)
        linewidth = scattering.species.linewidth_rad_s
        recoil_v = scattering.species.recoil_velocity_m_per_s
        beam_dirs = scattering.beam_directions
    else:
        model = None

    trajectory_times: list[float] = []
    trajectory_positions: list[NDArray[np.float64]] = []
    trajectory_velocities: list[NDArray[np.float64]] = []
    trajectory_lost: list[NDArray[np.bool_]] = []
    if config.store_trajectories:
        _append_trajectory_sample(
            trajectory_times,
            trajectory_positions,
            trajectory_velocities,
            trajectory_lost,
            0.0,
            positions,
            velocities,
            lost,
        )

    t = 0.0
    step_index = 0
    while t < config.duration_s:
        dt = min(config.timestep_s, config.duration_s - t)
        t_next = t + dt
        active = np.flatnonzero(~lost)

        if active.size > 0:
            active_velocities = velocities[active]

            if scattering is not None:
                # Internal-state backend: rates -> populations -> photon
                # events -> recoil impulse applied at the start of the step.
                rates = scattering.stimulated_rates(
                    positions[active], active_velocities, time_s=t
                )
                state_active, events = model.step(
                    internal_state[active], rates, linewidth, dt, rng
                )
                internal_state[active] = state_active
                scattered_photons[active] += events.total_scattered
                excited_fraction[active] = model.excited_fraction(state_active)
                active_velocities = active_velocities + sample_recoil_velocity_kicks(
                    events, beam_dirs, recoil_v, rng
                )

            acceleration_now = (
                _total_acceleration(traps, positions[active], t, config.mass_kg)
            )
            next_positions = (
                positions[active]
                + active_velocities * dt
                + 0.5 * acceleration_now * dt * dt
            )
            acceleration_next = (
                _total_acceleration(traps, next_positions, t_next, config.mass_kg)
            )
            next_velocities = active_velocities + 0.5 * (
                acceleration_now + acceleration_next
            ) * dt

            positions[active] = next_positions
            velocities[active] = next_velocities
            active_energies_j = _mechanical_energy(
                traps,
                positions[active],
                velocities[active],
                config.mass_kg,
                time_s=t_next,
            )
            final_energies_j[active] = active_energies_j
            energy_lost = (
                active_energies_j >= 0.0
                if track_energy
                else np.zeros_like(active_energies_j, dtype=bool)
            )
            lost[active] = energy_lost | _outside_boundary(positions[active], config)

        t = t_next
        step_index += 1
        if config.store_trajectories and step_index % config.trajectory_stride == 0:
            _append_trajectory_sample(
                trajectory_times,
                trajectory_positions,
                trajectory_velocities,
                trajectory_lost,
                t,
                positions,
                velocities,
                lost,
            )

    if config.store_trajectories and (
        not trajectory_times or not np.isclose(trajectory_times[-1], config.duration_s)
    ):
        _append_trajectory_sample(
            trajectory_times,
            trajectory_positions,
            trajectory_velocities,
            trajectory_lost,
            config.duration_s,
            positions,
            velocities,
            lost,
        )

    survivors = ~lost
    initial_energies_uK = joule_to_microkelvin(initial_energies_j)
    final_energies_uK = joule_to_microkelvin(final_energies_j)
    energy_gains_uK = final_energies_uK - initial_energies_uK

    if np.any(survivors):
        survivor_gains = energy_gains_uK[survivors]
        mean_energy_gain_uK = float(np.mean(survivor_gains))
        median_energy_gain_uK = float(np.median(survivor_gains))
        final_temperature_uK_survivors = _kinetic_temperature_uK(
            velocities[survivors], config.mass_kg
        )
        initial_temperature_uK_survivors = _kinetic_temperature_uK(
            initial_velocities[survivors], config.mass_kg
        )
    else:
        mean_energy_gain_uK = float("nan")
        median_energy_gain_uK = float("nan")
        final_temperature_uK_survivors = float("nan")
        initial_temperature_uK_survivors = float("nan")

    final_temperature_uK_all = _kinetic_temperature_uK(velocities, config.mass_kg)
    initial_temperature_uK_all = _kinetic_temperature_uK(
        initial_velocities, config.mass_kg
    )

    survival_probability = float(np.mean(survivors))
    loss_fraction = 1.0 - survival_probability

    return SimulationResult(
        survival_probability=survival_probability,
        loss_fraction=loss_fraction,
        mean_energy_gain_uK=mean_energy_gain_uK,
        median_energy_gain_uK=median_energy_gain_uK,
        initial_temperature_uK_survivors=initial_temperature_uK_survivors,
        initial_temperature_uK_all=initial_temperature_uK_all,
        final_temperature_uK_survivors=final_temperature_uK_survivors,
        final_temperature_uK_all=final_temperature_uK_all,
        initial_energies_uK=initial_energies_uK,
        final_energies_uK=final_energies_uK,
        energy_gains_uK=energy_gains_uK,
        initial_positions_m=initial_positions,
        initial_velocities_m_per_s=initial_velocities,
        final_positions_m=positions,
        final_velocities_m_per_s=velocities,
        lost=lost,
        duration_s=float(config.duration_s),
        initial_rejected_count=initial_rejected_count,
        initial_rejection_fraction=(
            initial_rejected_count / (config.ensemble_size + initial_rejected_count)
            if initial_rejected_count > 0
            else 0.0
        ),
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
        scattered_photons=scattered_photons if scattering is not None else None,
        final_excited_fraction=(
            excited_fraction if scattering is not None else None
        ),
    )


def _append_trajectory_sample(
    trajectory_times: list[float],
    trajectory_positions: list[NDArray[np.float64]],
    trajectory_velocities: list[NDArray[np.float64]],
    trajectory_lost: list[NDArray[np.bool_]],
    time_s: float,
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    lost: NDArray[np.bool_],
) -> None:
    trajectory_times.append(time_s)
    trajectory_positions.append(positions_m.copy())
    trajectory_velocities.append(velocities_m_per_s.copy())
    trajectory_lost.append(lost.copy())


def _sample_initial_ensemble(
    traps: list[TrapConfig],
    config: SimulationConfig,
    count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample initial positions and velocities per the config.

    Positions: free Gaussian cloud if `initial_cloud_sigma_m` is set,
    otherwise the local harmonic approximation of the traps.
    Velocities: Maxwell-Boltzmann plus the configured mean drift.
    """

    if config.initial_cloud_sigma_m is not None:
        center = (
            config.initial_center_m
            if config.initial_center_m is not None
            else np.zeros(3)
        )
        positions = center + rng.normal(size=(count, 3)) * config.initial_cloud_sigma_m
    elif traps:
        positions = sample_thermal_positions_harmonic(
            traps,
            config.initial_temperature_uK,
            count,
            rng,
            center_m=config.initial_center_m,
            time_s=0.0,
        )
    else:
        raise ValueError(
            "Without traps, initial positions cannot come from a trap Hessian; "
            "set SimulationConfig.initial_cloud_sigma_m."
        )
    velocities = (
        sample_thermal_velocities(
            config.initial_temperature_uK, count, rng, mass_kg=config.mass_kg
        )
        + config.initial_mean_velocity_m_per_s
    )
    return positions, velocities


def _resample_initially_lost(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    config: SimulationConfig,
    rng: np.random.Generator,
    track_energy: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    rejected_count = 0
    lost = _initial_lost_flags(
        traps, positions_m, velocities_m_per_s, config, track_energy
    )
    for _ in range(config.max_initial_resampling_rounds):
        if not np.any(lost):
            return positions_m, velocities_m_per_s, rejected_count

        count = int(np.sum(lost))
        rejected_count += count
        positions_m[lost], velocities_m_per_s[lost] = _sample_initial_ensemble(
            traps, config, count, rng
        )
        lost = _initial_lost_flags(
            traps, positions_m, velocities_m_per_s, config, track_energy
        )

    remaining = int(np.sum(lost))
    raise RuntimeError(
        "Could not sample a fully bound initial ensemble after "
        f"{config.max_initial_resampling_rounds} resampling rounds; "
        f"{remaining} atoms remain initially lost. Increase trap depth, lower "
        "temperature, or set reject_initially_lost=False."
    )


def _initial_lost_flags(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    config: SimulationConfig,
    track_energy: bool,
) -> NDArray[np.bool_]:
    if track_energy:
        initial_energies_j = _mechanical_energy(
            traps, positions_m, velocities_m_per_s, config.mass_kg, time_s=0.0
        )
        energy_lost = initial_energies_j >= 0.0
    else:
        energy_lost = np.zeros(positions_m.shape[:-1], dtype=bool)
    return energy_lost | _outside_boundary(positions_m, config)


def _total_acceleration(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    time_s: float,
    mass_kg: float,
) -> NDArray[np.float64]:
    if not traps:
        return np.zeros_like(positions_m)
    return total_force(traps, positions_m, time_s=time_s) / mass_kg


def _mechanical_energy(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    mass_kg: float,
    time_s: float,
) -> NDArray[np.float64]:
    kinetic = 0.5 * mass_kg * np.sum(velocities_m_per_s * velocities_m_per_s, axis=-1)
    potential = total_potential(traps, positions_m, time_s=time_s)
    return kinetic + potential


def _kinetic_temperature_uK(
    velocities_m_per_s: NDArray[np.float64], mass_kg: float
) -> float:
    if velocities_m_per_s.size == 0:
        return float("nan")
    mean_speed_sq = float(np.mean(np.sum(velocities_m_per_s * velocities_m_per_s, axis=-1)))
    energy_per_axis_j = mass_kg * mean_speed_sq / 3.0
    return float(joule_to_microkelvin(energy_per_axis_j))


def _outside_boundary(
    positions_m: NDArray[np.float64], config: SimulationConfig
) -> NDArray[np.bool_]:
    if config.loss_radius_m is None:
        return np.zeros(positions_m.shape[:-1], dtype=bool)
    offsets = positions_m - config.boundary_center_m
    return np.linalg.norm(offsets, axis=-1) > config.loss_radius_m


def _trap_list(traps: TrapConfig | Iterable[TrapConfig]) -> list[TrapConfig]:
    if isinstance(traps, TrapConfig):
        return [traps]
    return list(traps)
