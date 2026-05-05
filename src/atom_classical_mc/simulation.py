"""Monte Carlo trajectory propagation and heating/loss analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import RB87_MASS_KG
from .ramp import RampSequence
from .sampling import sample_thermal_positions_harmonic, sample_thermal_velocities
from .trap import TrapConfig, total_force, total_potential
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
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if self.loss_radius_m is not None and self.loss_radius_m <= 0.0:
            raise ValueError("loss_radius_m must be positive when provided.")
        if self.trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive.")

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
    """Aggregate and per-particle simulation outputs."""

    survival_probability: float
    loss_fraction: float
    mean_energy_gain_uK: float
    median_energy_gain_uK: float
    final_temperature_uK: float
    temperature_gain_uK: float
    initial_energies_uK: NDArray[np.float64]
    final_energies_uK: NDArray[np.float64]
    energy_gains_uK: NDArray[np.float64]
    final_positions_m: NDArray[np.float64]
    final_velocities_m_per_s: NDArray[np.float64]
    lost: NDArray[np.bool_]
    trajectory_times_s: NDArray[np.float64] | None = None
    trajectory_positions_m: NDArray[np.float64] | None = None
    trajectory_velocities_m_per_s: NDArray[np.float64] | None = None
    trajectory_lost: NDArray[np.bool_] | None = None


def run_simulation(
    static_trap: TrapConfig | Iterable[TrapConfig],
    moving_trap_base: TrapConfig,
    ramp: RampSequence,
    config: SimulationConfig,
) -> SimulationResult:
    """Run a classical Monte Carlo transfer simulation."""

    static_traps = _trap_list(static_trap)
    rng = np.random.default_rng(config.random_seed)

    moving_initial = _moving_trap_at(moving_trap_base, ramp, 0.0)
    initial_traps = static_traps + [moving_initial]
    positions = sample_thermal_positions_harmonic(
        initial_traps,
        config.initial_temperature_uK,
        config.ensemble_size,
        rng,
        center_m=config.initial_center_m,
    )
    velocities = sample_thermal_velocities(
        config.initial_temperature_uK,
        config.ensemble_size,
        rng,
        mass_kg=config.mass_kg,
    )

    initial_energies_j = _mechanical_energy(initial_traps, positions, velocities, config.mass_kg)
    final_energies_j = initial_energies_j.copy()
    lost = (initial_energies_j >= 0.0) | _outside_boundary(positions, config)

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
            traps_now = static_traps + [_moving_trap_at(moving_trap_base, ramp, t)]
            acceleration_now = total_force(traps_now, positions[active]) / config.mass_kg

            next_positions = (
                positions[active]
                + velocities[active] * dt
                + 0.5 * acceleration_now * dt * dt
            )

            traps_next = static_traps + [_moving_trap_at(moving_trap_base, ramp, t_next)]
            acceleration_next = total_force(traps_next, next_positions) / config.mass_kg
            next_velocities = velocities[active] + 0.5 * (
                acceleration_now + acceleration_next
            ) * dt

            positions[active] = next_positions
            velocities[active] = next_velocities
            active_energies_j = _mechanical_energy(
                traps_next,
                positions[active],
                velocities[active],
                config.mass_kg,
            )
            final_energies_j[active] = active_energies_j
            lost[active] = (active_energies_j >= 0.0) | _outside_boundary(
                positions[active], config
            )

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
        final_temperature_uK = _kinetic_temperature_uK(
            velocities[survivors], config.mass_kg
        )
    else:
        mean_energy_gain_uK = float("nan")
        median_energy_gain_uK = float("nan")
        final_temperature_uK = float("nan")

    survival_probability = float(np.mean(survivors))
    loss_fraction = 1.0 - survival_probability
    temperature_gain_uK = final_temperature_uK - config.initial_temperature_uK

    return SimulationResult(
        survival_probability=survival_probability,
        loss_fraction=loss_fraction,
        mean_energy_gain_uK=mean_energy_gain_uK,
        median_energy_gain_uK=median_energy_gain_uK,
        final_temperature_uK=final_temperature_uK,
        temperature_gain_uK=temperature_gain_uK,
        initial_energies_uK=initial_energies_uK,
        final_energies_uK=final_energies_uK,
        energy_gains_uK=energy_gains_uK,
        final_positions_m=positions,
        final_velocities_m_per_s=velocities,
        lost=lost,
        trajectory_times_s=(
            np.asarray(trajectory_times, dtype=float) if config.store_trajectories else None
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
            np.asarray(trajectory_lost, dtype=bool) if config.store_trajectories else None
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


def _moving_trap_at(
    moving_trap_base: TrapConfig, ramp: RampSequence, time_s: float
) -> TrapConfig:
    center, depth = ramp.at(time_s)
    return moving_trap_base.with_center_depth(center, depth)


def _mechanical_energy(
    traps: list[TrapConfig],
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    mass_kg: float,
) -> NDArray[np.float64]:
    kinetic = 0.5 * mass_kg * np.sum(velocities_m_per_s * velocities_m_per_s, axis=-1)
    potential = total_potential(traps, positions_m)
    return kinetic + potential


def _kinetic_temperature_uK(
    velocities_m_per_s: NDArray[np.float64], mass_kg: float
) -> float:
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
