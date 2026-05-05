"""Small SI unit conversion helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import BOLTZMANN_CONSTANT_J_PER_K

MICROKELVIN_IN_KELVIN = 1.0e-6
MICROMETER_IN_METER = 1.0e-6
MILLISECOND_IN_SECOND = 1.0e-3


def microkelvin_to_joule(temperature_uK: ArrayLike) -> NDArray[np.float64]:
    """Convert a temperature-like energy scale in microkelvin to joules."""

    return (
        np.asarray(temperature_uK, dtype=float)
        * MICROKELVIN_IN_KELVIN
        * BOLTZMANN_CONSTANT_J_PER_K
    )


def joule_to_microkelvin(energy_j: ArrayLike) -> NDArray[np.float64]:
    """Convert joules to the equivalent temperature-like scale in microkelvin."""

    return np.asarray(energy_j, dtype=float) / (
        MICROKELVIN_IN_KELVIN * BOLTZMANN_CONSTANT_J_PER_K
    )


def um(value: ArrayLike) -> NDArray[np.float64]:
    """Convert micrometers to meters."""

    return np.asarray(value, dtype=float) * MICROMETER_IN_METER


def ms(value: ArrayLike) -> NDArray[np.float64]:
    """Convert milliseconds to seconds."""

    return np.asarray(value, dtype=float) * MILLISECOND_IN_SECOND
