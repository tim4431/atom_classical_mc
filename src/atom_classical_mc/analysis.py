"""Analysis helpers for heating, loss, and transfer probability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import RB87_MASS_KG
from .simulation import SimulationResult
from .trap import TrapConfig
from .units import joule_to_microkelvin


@dataclass(frozen=True)
class TrapOccupation:
    """Mutually exclusive final trap occupation probabilities and masks."""

    slm_probability: float
    aod_probability: float
    ambiguous_probability: float
    unbound_survivor_probability: float
    loss_probability: float
    slm_mask: NDArray[np.bool_]
    aod_mask: NDArray[np.bool_]
    ambiguous_mask: NDArray[np.bool_]
    unbound_survivor_mask: NDArray[np.bool_]
    lost_mask: NDArray[np.bool_]


def kinetic_energy_uK(
    velocities_m_per_s: NDArray[np.float64],
    mass_kg: float = RB87_MASS_KG,
) -> NDArray[np.float64]:
    """Return per-particle kinetic energy in microkelvin energy units."""

    velocities = np.asarray(velocities_m_per_s, dtype=float)
    kinetic_j = 0.5 * mass_kg * np.sum(velocities * velocities, axis=-1)
    return joule_to_microkelvin(kinetic_j)


def bound_to_trap(
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    trap: TrapConfig,
    mass_kg: float = RB87_MASS_KG,
) -> NDArray[np.bool_]:
    """Classify particles as bound to a single trap in the lab frame."""

    kinetic_j = 0.5 * mass_kg * np.sum(velocities_m_per_s * velocities_m_per_s, axis=-1)
    single_trap_energy_j = kinetic_j + trap.potential(positions_m)
    return single_trap_energy_j < 0.0


def single_trap_energy_uK(
    positions_m: NDArray[np.float64],
    velocities_m_per_s: NDArray[np.float64],
    trap: TrapConfig,
    mass_kg: float = RB87_MASS_KG,
) -> NDArray[np.float64]:
    """Return mechanical energy against one trap only, in microkelvin units."""

    velocities = np.asarray(velocities_m_per_s, dtype=float)
    kinetic_j = 0.5 * mass_kg * np.sum(velocities * velocities, axis=-1)
    return joule_to_microkelvin(kinetic_j + trap.potential(positions_m))


def capture_probability(
    result: SimulationResult,
    trap: TrapConfig,
    mass_kg: float = RB87_MASS_KG,
    conditional_on_survival: bool = False,
) -> float:
    """Return final probability that an atom survived and is bound to `trap`.

    With `conditional_on_survival=False`, the denominator is the original
    ensemble size. With `True`, the denominator is the final survivor count.
    """

    survivors = ~result.lost
    captured = (
        survivors
        & bound_to_trap(
            result.final_positions_m,
            result.final_velocities_m_per_s,
            trap,
            mass_kg=mass_kg,
        )
    )
    if conditional_on_survival:
        survivor_count = int(np.sum(survivors))
        if survivor_count == 0:
            return float("nan")
        return float(np.sum(captured) / survivor_count)
    return float(np.mean(captured))


def classify_final_trap_occupation(
    result: SimulationResult,
    slm_trap: TrapConfig,
    aod_trap: TrapConfig,
    mass_kg: float = RB87_MASS_KG,
) -> TrapOccupation:
    """Classify final atoms as SLM-bound, AOD-bound, ambiguous, or unbound.

    A particle is bound to a trap when its lab-frame mechanical energy in that
    single-trap potential is negative. Particles bound to both final traps are
    reported as ambiguous instead of being forced into one trap.
    """

    survivors = ~result.lost
    slm_bound = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        slm_trap,
        mass_kg=mass_kg,
    )
    aod_bound = survivors & bound_to_trap(
        result.final_positions_m,
        result.final_velocities_m_per_s,
        aod_trap,
        mass_kg=mass_kg,
    )

    ambiguous = slm_bound & aod_bound
    slm_only = slm_bound & ~aod_bound
    aod_only = aod_bound & ~slm_bound
    unbound_survivor = survivors & ~slm_bound & ~aod_bound

    return TrapOccupation(
        slm_probability=float(np.mean(slm_only)),
        aod_probability=float(np.mean(aod_only)),
        ambiguous_probability=float(np.mean(ambiguous)),
        unbound_survivor_probability=float(np.mean(unbound_survivor)),
        loss_probability=float(np.mean(result.lost)),
        slm_mask=slm_only,
        aod_mask=aod_only,
        ambiguous_mask=ambiguous,
        unbound_survivor_mask=unbound_survivor,
        lost_mask=result.lost.copy(),
    )


def survival_probability_time_series(result: SimulationResult) -> NDArray[np.float64]:
    """Return survival probability at each stored trajectory time."""

    if result.trajectory_lost is None:
        raise ValueError("Simulation was run without stored trajectories.")
    return np.mean(~result.trajectory_lost, axis=1)


def loss_probability_time_series(result: SimulationResult) -> NDArray[np.float64]:
    """Return loss probability, `1 - survival probability`, at each stored time."""

    return 1.0 - survival_probability_time_series(result)


def mean_kinetic_energy_time_series_uK(
    result: SimulationResult,
    mass_kg: float = RB87_MASS_KG,
    survivors_only: bool = True,
) -> NDArray[np.float64]:
    """Return mean kinetic energy versus stored trajectory time."""

    if result.trajectory_velocities_m_per_s is None:
        raise ValueError("Simulation was run without stored trajectories.")

    kinetic = kinetic_energy_uK(result.trajectory_velocities_m_per_s, mass_kg=mass_kg)
    if not survivors_only:
        return np.mean(kinetic, axis=1)
    if result.trajectory_lost is None:
        raise ValueError("Trajectory lost masks are required for survivor-only analysis.")

    means = np.full(kinetic.shape[0], np.nan, dtype=float)
    survivor_masks = ~result.trajectory_lost
    for index, survivor_mask in enumerate(survivor_masks):
        if np.any(survivor_mask):
            means[index] = float(np.mean(kinetic[index, survivor_mask]))
    return means
