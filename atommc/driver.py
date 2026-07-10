"""The multiphysics driver: `simulate(system, config)`.

The driver is the "study/solver" of the package: it samples an initial
ensemble, then advances it with velocity-Verlet under the summed
`ConservativeForce` modules, operator-splitting every `StochasticProcess`
into a per-step velocity kick applied before the Verlet update. Modules
are registered on the `AtomSystem`; the loop never special-cases any
particular physics.

Loss criteria:

- boundary: an atom leaving `loss_radius_m` (if set) is lost, always.
- energy: an atom whose conservative mechanical energy is >= 0 is lost.
  Controlled by `SimulationConfig.energy_loss`: `"auto"` (default) enables
  it only for purely conservative systems with at least one force;
  `"on"` demands it and raises if any stochastic process is attached
  (non-conservative physics invalidates the criterion); `"off"` disables
  it. Once lost, always lost; lost atoms are not advanced further.

Per-atom diagnostics declared by stochastic processes (via
`diagnostics_spec`) are accumulated into `SimulationResult.diagnostics`,
keyed by module name then channel key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ensemble import (
    EnsembleSource,
    sample_thermal_positions_harmonic,
    sample_thermal_velocities,
)
from .physics.base import DiagnosticSpec, StochasticProcess
from .system import AtomSystem
from .units import joule_to_microkelvin


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a Monte Carlo trajectory simulation.

    The atom species (and hence the mass) lives on the `AtomSystem`, not
    here; this object only describes the ensemble, the integrator, and
    the loss/recording knobs.

    Initial positions come from the local harmonic approximation of the
    system's conservative forces by default. Set `initial_cloud_sigma_m`
    (scalar or per-axis) to instead sample a free Gaussian cloud around
    `initial_center_m` — required when there are no conservative forces
    (pure light-force runs) and useful for capture studies.
    `initial_mean_velocity_m_per_s` adds a drift to the Maxwell-Boltzmann
    velocities (e.g. launched atoms).

    For a named physical source (a thermal cloud, a harmonic-trap
    equilibrium cloud, or an effusive oven beam) set `initial_source` to
    an `ensemble.EnsembleSource`; it supplies both positions and
    velocities and overrides the knobs above. Its per-sample `weight`
    (e.g. the flux fraction of a truncated beam window) is reported as
    `SimulationResult.initial_ensemble_weight`.

    For full control, pass explicit `initial_positions_m` and/or
    `initial_velocities_m_per_s_array` (shape `(ensemble_size, 3)`);
    each overrides the corresponding sampler. Explicit ensembles cannot
    be resampled, so they require `reject_initially_lost=False`.
    """

    initial_temperature_uK: float
    timestep_s: float
    duration_s: float
    ensemble_size: int
    random_seed: int | None = None
    loss_radius_m: float | None = None
    boundary_center_m: ArrayLike = (0.0, 0.0, 0.0)
    initial_center_m: ArrayLike | None = None
    initial_cloud_sigma_m: ArrayLike | None = None
    initial_mean_velocity_m_per_s: ArrayLike = (0.0, 0.0, 0.0)
    initial_positions_m: ArrayLike | None = None
    initial_velocities_m_per_s_array: ArrayLike | None = None
    initial_source: EnsembleSource | None = None
    reject_initially_lost: bool = True
    max_initial_resampling_rounds: int = 100
    store_trajectories: bool = False
    trajectory_stride: int = 1
    energy_loss: Literal["auto", "on", "off"] = "auto"

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
        if self.max_initial_resampling_rounds <= 0:
            raise ValueError("max_initial_resampling_rounds must be positive.")
        if self.energy_loss not in ("auto", "on", "off"):
            raise ValueError("energy_loss must be 'auto', 'on', or 'off'.")

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

        explicit = False
        for attr in ("initial_positions_m", "initial_velocities_m_per_s_array"):
            value = getattr(self, attr)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != (self.ensemble_size, 3):
                raise ValueError(
                    f"{attr} must have shape (ensemble_size, 3) = "
                    f"({self.ensemble_size}, 3); got {array.shape}."
                )
            object.__setattr__(self, attr, array)
            explicit = True
        if explicit and self.reject_initially_lost:
            raise ValueError(
                "Explicit initial ensembles cannot be resampled; set "
                "reject_initially_lost=False."
            )

        if self.initial_source is not None:
            if not isinstance(self.initial_source, EnsembleSource):
                raise TypeError("initial_source must be an ensemble.EnsembleSource.")
            if explicit:
                raise ValueError(
                    "initial_source and explicit initial_positions_m / "
                    "initial_velocities_m_per_s_array are mutually exclusive."
                )


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate and per-particle simulation outputs.

    Temperature fields come in two flavors:

    - `*_survivors`: average over atoms not flagged lost at the relevant time.
    - `*_all`: average over the entire ensemble (including lost atoms).

    The legacy properties `final_temperature_uK` and `temperature_gain_uK`
    return the survivors-only values to match older callers.

    `diagnostics` holds the per-atom channels declared by each stochastic
    process, keyed `diagnostics[module_name][channel_key]`. The
    `scattered_photons` and `final_excited_fraction` properties read the
    conventional `LightScattering` channels out of it (`None` on dark
    runs). Energy fields always measure the *conservative* mechanical
    energy (kinetic + summed potential), so with light attached the energy
    gains include recoil heating.
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
    initial_ensemble_weight: float = 1.0
    trajectory_times_s: NDArray[np.float64] | None = None
    trajectory_positions_m: NDArray[np.float64] | None = None
    trajectory_velocities_m_per_s: NDArray[np.float64] | None = None
    trajectory_lost: NDArray[np.bool_] | None = None
    diagnostics: dict[str, dict[str, NDArray]] = field(default_factory=dict)

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

    @property
    def scattered_photons(self) -> NDArray[np.int64] | None:
        """Total scattered photons per atom, summed over scattering modules."""

        arrays = [
            channels["scattered_photons"]
            for channels in self.diagnostics.values()
            if "scattered_photons" in channels
        ]
        if not arrays:
            return None
        total = arrays[0].copy()
        for extra in arrays[1:]:
            total = total + extra
        return total

    @property
    def final_excited_fraction(self) -> NDArray[np.float64] | None:
        """Latest excited-state population from the first scattering module."""

        for channels in self.diagnostics.values():
            if "excited_fraction" in channels:
                return channels["excited_fraction"]
        return None


def simulate(system: AtomSystem, config: SimulationConfig) -> SimulationResult:
    """Run a classical Monte Carlo simulation of `system`.

    Each timestep, every `StochasticProcess` on the system steps its
    per-atom internal state and applies its velocity kicks, then the
    ensemble is advanced by velocity-Verlet under the summed
    `ConservativeForce` modules. Lost atoms are masked out of both.
    """

    if not isinstance(system, AtomSystem):
        raise TypeError("system must be an AtomSystem.")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig.")
    if not system.modules:
        raise ValueError("AtomSystem has no physics modules; nothing to simulate.")

    track_energy = _resolve_energy_loss(system, config)
    rng = np.random.default_rng(config.random_seed)

    positions, velocities, initial_ensemble_weight = _sample_initial_ensemble(
        system, config, config.ensemble_size, rng
    )
    initial_rejected_count = 0
    if config.reject_initially_lost:
        positions, velocities, initial_rejected_count = _resample_initially_lost(
            system, positions, velocities, config, rng, track_energy
        )
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()

    initial_energies_j = system.mechanical_energy_j(
        positions, velocities, time_s=0.0
    )
    final_energies_j = initial_energies_j.copy()
    lost = _initial_lost_flags(system, positions, velocities, config, track_energy)

    process_states = [
        process.initialize(config.ensemble_size, rng)
        for process in system.processes
    ]
    diagnostics = _allocate_diagnostics(system.processes, config.ensemble_size)

    recorder = _TrajectoryRecorder(
        enabled=config.store_trajectories, stride=config.trajectory_stride
    )
    recorder.record(0.0, positions, velocities, lost)

    mass_kg = system.mass_kg
    t = 0.0
    step_index = 0
    while t < config.duration_s:
        dt = min(config.timestep_s, config.duration_s - t)
        t_next = t + dt
        active = np.flatnonzero(~lost)

        if active.size > 0:
            active_velocities = velocities[active]

            # Operator split: stochastic velocity kicks first, applied at
            # the start of the step, then the conservative Verlet update.
            for process, state in zip(system.processes, process_states):
                result = process.step(
                    state[active],
                    positions[active],
                    active_velocities,
                    t,
                    dt,
                    rng,
                )
                state[active] = result.state
                if result.velocity_kick_m_per_s is not None:
                    active_velocities = (
                        active_velocities + result.velocity_kick_m_per_s
                    )
                _accumulate_diagnostics(
                    diagnostics[process.name],
                    process.diagnostics_spec(),
                    active,
                    result.diagnostics,
                )

            acceleration_now = system.total_force(positions[active], t) / mass_kg
            next_positions = (
                positions[active]
                + active_velocities * dt
                + 0.5 * acceleration_now * dt * dt
            )
            acceleration_next = system.total_force(next_positions, t_next) / mass_kg
            next_velocities = active_velocities + 0.5 * (
                acceleration_now + acceleration_next
            ) * dt

            positions[active] = next_positions
            velocities[active] = next_velocities
            active_energies_j = system.mechanical_energy_j(
                next_positions, next_velocities, time_s=t_next
            )
            final_energies_j[active] = active_energies_j
            energy_lost = (
                active_energies_j >= 0.0
                if track_energy
                else np.zeros_like(active_energies_j, dtype=bool)
            )
            lost[active] = energy_lost | _outside_boundary(next_positions, config)

        t = t_next
        step_index += 1
        if step_index % config.trajectory_stride == 0:
            recorder.record(t, positions, velocities, lost)

    recorder.finalize(config.duration_s, positions, velocities, lost)

    return _assemble_result(
        config=config,
        mass_kg=mass_kg,
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        positions=positions,
        velocities=velocities,
        initial_energies_j=initial_energies_j,
        final_energies_j=final_energies_j,
        lost=lost,
        initial_rejected_count=initial_rejected_count,
        initial_ensemble_weight=initial_ensemble_weight,
        recorder=recorder,
        diagnostics=diagnostics,
    )


def _resolve_energy_loss(system: AtomSystem, config: SimulationConfig) -> bool:
    """Turn the `energy_loss` tri-state into a boolean, loudly."""

    if config.energy_loss == "off":
        return False
    if config.energy_loss == "on":
        if system.processes:
            names = ", ".join(process.name for process in system.processes)
            raise ValueError(
                "energy_loss='on' requires a purely conservative system, but "
                f"stochastic processes are attached ({names}); their "
                "non-conservative kicks invalidate the mechanical-energy loss "
                "criterion. Use energy_loss='auto'/'off' and loss_radius_m, "
                "then recover trap survival with analysis.bound_to_trap."
            )
        if not system.forces:
            raise ValueError(
                "energy_loss='on' requires at least one conservative force."
            )
        return True
    return system.is_conservative and bool(system.forces)


def _allocate_diagnostics(
    processes: tuple[StochasticProcess, ...], ensemble_size: int
) -> dict[str, dict[str, NDArray]]:
    diagnostics: dict[str, dict[str, NDArray]] = {}
    for process in processes:
        if process.name in diagnostics:
            raise ValueError(
                f"Two stochastic processes share the name '{process.name}'; "
                "give each a unique name so diagnostics stay separate."
            )
        diagnostics[process.name] = {
            spec.key: np.full(ensemble_size, spec.fill, dtype=spec.dtype)
            for spec in process.diagnostics_spec()
        }
    return diagnostics


def _accumulate_diagnostics(
    channels: dict[str, NDArray],
    specs: tuple[DiagnosticSpec, ...],
    active: NDArray[np.int64],
    values,
) -> None:
    for spec in specs:
        if spec.key not in values:
            continue
        sample = np.asarray(values[spec.key])
        if spec.reduce == "sum":
            channels[spec.key][active] += sample.astype(spec.dtype, copy=False)
        else:
            channels[spec.key][active] = sample


class _TrajectoryRecorder:
    """Collects stride-sampled `(t, x, v, lost)` snapshots when enabled."""

    def __init__(self, enabled: bool, stride: int) -> None:
        self.enabled = enabled
        self.stride = stride
        self.times: list[float] = []
        self.positions: list[NDArray[np.float64]] = []
        self.velocities: list[NDArray[np.float64]] = []
        self.lost: list[NDArray[np.bool_]] = []

    def record(
        self,
        time_s: float,
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        lost: NDArray[np.bool_],
    ) -> None:
        if not self.enabled:
            return
        self.times.append(time_s)
        self.positions.append(positions_m.copy())
        self.velocities.append(velocities_m_per_s.copy())
        self.lost.append(lost.copy())

    def finalize(
        self,
        duration_s: float,
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        lost: NDArray[np.bool_],
    ) -> None:
        """Force a final sample at `duration_s` if the stride missed it."""

        if not self.enabled:
            return
        if not self.times or not np.isclose(self.times[-1], duration_s):
            self.record(duration_s, positions_m, velocities_m_per_s, lost)

    def arrays(
        self,
    ) -> tuple[
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
        NDArray[np.bool_] | None,
    ]:
        if not self.enabled:
            return None, None, None, None
        return (
            np.asarray(self.times, dtype=float),
            np.asarray(self.positions, dtype=float),
            np.asarray(self.velocities, dtype=float),
            np.asarray(self.lost, dtype=bool),
        )


def _assemble_result(
    *,
    config: SimulationConfig,
    mass_kg: float,
    initial_positions: NDArray[np.float64],
    initial_velocities: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    initial_energies_j: NDArray[np.float64],
    final_energies_j: NDArray[np.float64],
    lost: NDArray[np.bool_],
    initial_rejected_count: int,
    initial_ensemble_weight: float,
    recorder: _TrajectoryRecorder,
    diagnostics: dict[str, dict[str, NDArray]],
) -> SimulationResult:
    survivors = ~lost
    initial_energies_uK = joule_to_microkelvin(initial_energies_j)
    final_energies_uK = joule_to_microkelvin(final_energies_j)
    energy_gains_uK = final_energies_uK - initial_energies_uK

    if np.any(survivors):
        survivor_gains = energy_gains_uK[survivors]
        mean_energy_gain_uK = float(np.mean(survivor_gains))
        median_energy_gain_uK = float(np.median(survivor_gains))
        final_temperature_uK_survivors = _kinetic_temperature_uK(
            velocities[survivors], mass_kg
        )
        initial_temperature_uK_survivors = _kinetic_temperature_uK(
            initial_velocities[survivors], mass_kg
        )
    else:
        mean_energy_gain_uK = float("nan")
        median_energy_gain_uK = float("nan")
        final_temperature_uK_survivors = float("nan")
        initial_temperature_uK_survivors = float("nan")

    trajectory_times, trajectory_positions, trajectory_velocities, trajectory_lost = (
        recorder.arrays()
    )

    survival_probability = float(np.mean(survivors))
    return SimulationResult(
        survival_probability=survival_probability,
        loss_fraction=1.0 - survival_probability,
        mean_energy_gain_uK=mean_energy_gain_uK,
        median_energy_gain_uK=median_energy_gain_uK,
        initial_temperature_uK_survivors=initial_temperature_uK_survivors,
        initial_temperature_uK_all=_kinetic_temperature_uK(
            initial_velocities, mass_kg
        ),
        final_temperature_uK_survivors=final_temperature_uK_survivors,
        final_temperature_uK_all=_kinetic_temperature_uK(velocities, mass_kg),
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
        initial_ensemble_weight=initial_ensemble_weight,
        trajectory_times_s=trajectory_times,
        trajectory_positions_m=trajectory_positions,
        trajectory_velocities_m_per_s=trajectory_velocities,
        trajectory_lost=trajectory_lost,
        diagnostics=diagnostics,
    )


def _sample_initial_ensemble(
    system: AtomSystem,
    config: SimulationConfig,
    count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Sample initial positions, velocities, and the ensemble weight.

    An `initial_source` (if set) supplies all three. Otherwise the weight
    is 1.0 and positions come from an explicit array, else a free Gaussian
    cloud if `initial_cloud_sigma_m` is set, else the local harmonic
    approximation of the conservative forces; velocities from an explicit
    array, else Maxwell-Boltzmann plus the configured mean drift.
    """

    if config.initial_source is not None:
        sample = config.initial_source.sample(count, rng)
        positions = np.asarray(sample.positions_m, dtype=float)
        velocities = np.asarray(sample.velocities_m_per_s, dtype=float)
        expected = (count, 3)
        if positions.shape != expected or velocities.shape != expected:
            raise ValueError(
                "initial_source produced arrays of shape "
                f"{positions.shape} / {velocities.shape}; expected {expected}."
            )
        return positions.copy(), velocities.copy(), float(sample.weight)

    if config.initial_positions_m is not None:
        positions = np.array(config.initial_positions_m, dtype=float, copy=True)
    elif config.initial_cloud_sigma_m is not None:
        center = (
            config.initial_center_m
            if config.initial_center_m is not None
            else np.zeros(3)
        )
        positions = center + rng.normal(size=(count, 3)) * config.initial_cloud_sigma_m
    elif system.forces:
        positions = sample_thermal_positions_harmonic(
            system.forces,
            config.initial_temperature_uK,
            count,
            rng,
            center_m=config.initial_center_m,
            time_s=0.0,
        )
    else:
        raise ValueError(
            "Without conservative forces, initial positions cannot come from "
            "a trap Hessian; set SimulationConfig.initial_cloud_sigma_m or "
            "initial_positions_m."
        )
    if config.initial_velocities_m_per_s_array is not None:
        velocities = np.array(
            config.initial_velocities_m_per_s_array, dtype=float, copy=True
        )
    else:
        velocities = (
            sample_thermal_velocities(
                config.initial_temperature_uK, count, rng, mass_kg=system.mass_kg
            )
            + config.initial_mean_velocity_m_per_s
        )
    return positions, velocities, 1.0


def _resample_initially_lost(
    system: AtomSystem,
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    config: SimulationConfig,
    rng: np.random.Generator,
    track_energy: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    rejected_count = 0
    lost = _initial_lost_flags(
        system, positions_m, velocities_m_per_s, config, track_energy
    )
    for _ in range(config.max_initial_resampling_rounds):
        if not np.any(lost):
            return positions_m, velocities_m_per_s, rejected_count

        count = int(np.sum(lost))
        rejected_count += count
        new_positions, new_velocities, _ = _sample_initial_ensemble(
            system, config, count, rng
        )
        positions_m[lost], velocities_m_per_s[lost] = new_positions, new_velocities
        lost = _initial_lost_flags(
            system, positions_m, velocities_m_per_s, config, track_energy
        )

    remaining = int(np.sum(lost))
    raise RuntimeError(
        "Could not sample a fully bound initial ensemble after "
        f"{config.max_initial_resampling_rounds} resampling rounds; "
        f"{remaining} atoms remain initially lost. Increase trap depth, lower "
        "temperature, or set reject_initially_lost=False."
    )


def _initial_lost_flags(
    system: AtomSystem,
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    config: SimulationConfig,
    track_energy: bool,
) -> NDArray[np.bool_]:
    if track_energy:
        initial_energies_j = system.mechanical_energy_j(
            positions_m, velocities_m_per_s, time_s=0.0
        )
        energy_lost = initial_energies_j >= 0.0
    else:
        energy_lost = np.zeros(positions_m.shape[:-1], dtype=bool)
    return energy_lost | _outside_boundary(positions_m, config)


def _kinetic_temperature_uK(
    velocities_m_per_s: NDArray[np.float64], mass_kg: float
) -> float:
    if velocities_m_per_s.size == 0:
        return float("nan")
    mean_speed_sq = float(
        np.mean(np.sum(velocities_m_per_s * velocities_m_per_s, axis=-1))
    )
    energy_per_axis_j = mass_kg * mean_speed_sq / 3.0
    return float(joule_to_microkelvin(energy_per_axis_j))


def _outside_boundary(
    positions_m: NDArray[np.float64], config: SimulationConfig
) -> NDArray[np.bool_]:
    if config.loss_radius_m is None:
        return np.zeros(positions_m.shape[:-1], dtype=bool)
    offsets = positions_m - config.boundary_center_m
    return np.linalg.norm(offsets, axis=-1) > config.loss_radius_m
