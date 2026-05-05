"""Thermal Monte Carlo initial state sampling."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import RB87_MASS_KG
from .trap import TrapConfig, total_hessian
from .units import microkelvin_to_joule


def sample_thermal_velocities(
    temperature_uK: float,
    ensemble_size: int,
    rng: np.random.Generator,
    mass_kg: float = RB87_MASS_KG,
) -> NDArray[np.float64]:
    """Sample Maxwell-Boltzmann velocities for a 3D gas."""

    if temperature_uK < 0.0:
        raise ValueError("temperature_uK must be non-negative.")
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")

    sigma = np.sqrt(float(microkelvin_to_joule(temperature_uK)) / mass_kg)
    return rng.normal(loc=0.0, scale=sigma, size=(ensemble_size, 3))


def sample_thermal_positions_harmonic(
    traps: TrapConfig | Iterable[TrapConfig],
    temperature_uK: float,
    ensemble_size: int,
    rng: np.random.Generator,
    center_m: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Sample positions from a local harmonic approximation to the trap.

    The covariance is `k_B T K^-1`, where `K` is the potential Hessian at
    `center_m`. If no center is provided, the deepest trap center is used.
    """

    trap_list = _trap_list(traps)
    if not trap_list:
        raise ValueError("At least one trap is required for position sampling.")
    if temperature_uK < 0.0:
        raise ValueError("temperature_uK must be non-negative.")
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive.")

    center = _default_center(trap_list) if center_m is None else np.asarray(center_m, dtype=float)
    if center.shape != (3,):
        raise ValueError("center_m must be a 3-vector in meters.")

    if temperature_uK == 0.0:
        return np.repeat(center[np.newaxis, :], ensemble_size, axis=0)

    stiffness = total_hessian(trap_list, center)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(
            "The local trap Hessian is not positive definite at the sampling center."
        )

    thermal_energy = float(microkelvin_to_joule(temperature_uK))
    covariance = eigenvectors @ np.diag(thermal_energy / eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return rng.multivariate_normal(mean=center, cov=covariance, size=ensemble_size)


def _default_center(traps: list[TrapConfig]) -> NDArray[np.float64]:
    deepest = max(traps, key=lambda trap: trap.depth_uK)
    if deepest.depth_uK <= 0.0:
        raise ValueError("At least one trap with positive depth is required.")
    return np.asarray(deepest.center_m, dtype=float)


def _trap_list(traps: TrapConfig | Iterable[TrapConfig]) -> list[TrapConfig]:
    if isinstance(traps, TrapConfig):
        return [traps]
    return list(traps)
