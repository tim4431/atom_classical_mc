"""Monte Carlo trajectory propagation and heating/loss analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import RB87_MASS_KG
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
    """Configuration for a Monte Carlo trajectory simulation."""

    initial_temperature_uK: float
    timestep_s: float
    duration_s: float
    ensemble_size: int
    random_seed: int | None = None
    mass_kg: float = RB87_MASS_KG
    loss_radius_m: float | None = None
    boundary_center_m: ArrayLike = (0.0, 0.0, 0.0)
    initial_center_m: ArrayLike | None = None
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


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate and per-particle simulation outputs.

    Temperature fields come in two flavors:

    - `*_survivors`: average over atoms not flagged lost at the relevant time.
    - `*_all`: average over the entire ensemble (including lost atoms).

    The legacy properties `final_temperature_uK` and `temperature_gain_uK`
    return the survivors-only values to match older callers.
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
      or an iterable of them. Time-dependent traps (e.g. `MovingGaussianTrap`,
      `AstigmaticAODTrap`) carry their own ramp internally.
    - Legacy: `run_simulation(static_trap, moving_trap_base, ramp, config)`.
      Builds a `MovingGaussianTrap` from `moving_trap_base + ramp` and calls
      the new form.
    """

    traps, config = _normalize_simulation_args(args, kwargs)
    return _run_simulation_core(traps, config)


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

    traps_list = _trap_list(traps_arg)
    if not traps_list:
        raise ValueError("At least one trap is required.")
    return traps_list, config


def _run_simulation_core(
    traps: list[TrapConfig], config: SimulationConfig
) -> SimulationResult:
    rng = np.random.default_rng(config.random_seed)

    positions = sample_thermal_positions_harmonic(
        traps,
        config.initial_temperature_uK,
        config.ensemble_size,
        rng,
        center_m=config.initial_center_m,
        time_s=0.0,
    )
    velocities = sample_thermal_velocities(
        config.initial_temperature_uK,
        config.ensemble_size,
        rng,
        mass_kg=config.mass_kg,
    )
    initial_rejected_count = 0
    if config.reject_initially_lost:
        positions, velocities, initial_rejected_count = _resample_initially_lost(
            traps, positions, velocities, config, rng
        )
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()

    initial_energies_j = _mechanical_energy(
        traps, positions, velocities, config.mass_kg, time_s=0.0
    )
    final_energies_j = initial_energies_j.copy()
    lost = _initial_lost_flags(traps, positions, velocities, config)

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
            acceleration_now = (
                total_force(traps, positions[active], time_s=t) / config.mass_kg
            )
            next_positions = (
                positions[active]
                + velocities[active] * dt
                + 0.5 * acceleration_now * dt * dt
            )
            acceleration_next = (
                total_force(traps, next_positions, time_s=t_next) / config.mass_kg
            )
            next_velocities = velocities[active] + 0.5 * (
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
                if config.track_energy_loss
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


def _resample_initially_lost(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    rejected_count = 0
    lost = _initial_lost_flags(traps, positions_m, velocities_m_per_s, config)
    for _ in range(config.max_initial_resampling_rounds):
        if not np.any(lost):
            return positions_m, velocities_m_per_s, rejected_count

        count = int(np.sum(lost))
        rejected_count += count
        positions_m[lost] = sample_thermal_positions_harmonic(
            traps,
            config.initial_temperature_uK,
            count,
            rng,
            center_m=config.initial_center_m,
            time_s=0.0,
        )
        velocities_m_per_s[lost] = sample_thermal_velocities(
            config.initial_temperature_uK,
            count,
            rng,
            mass_kg=config.mass_kg,
        )
        lost = _initial_lost_flags(traps, positions_m, velocities_m_per_s, config)

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
) -> NDArray[np.bool_]:
    if config.track_energy_loss:
        initial_energies_j = _mechanical_energy(
            traps, positions_m, velocities_m_per_s, config.mass_kg, time_s=0.0
        )
        energy_lost = initial_energies_j >= 0.0
    else:
        energy_lost = np.zeros(positions_m.shape[:-1], dtype=bool)
    return energy_lost | _outside_boundary(positions_m, config)


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
