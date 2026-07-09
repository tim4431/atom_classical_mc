"""Static and time-dependent magnetic field configurations.

`MagneticFieldConfig` mirrors the `TrapConfig` pattern: an abstract
`B(r, t)` returning the field vector in tesla, with concrete
implementations summed by `total_magnetic_field`.

- `UniformMagneticField` — constant bias field.
- `QuadrupoleMagneticField` — linear anti-Helmholtz quadrupole
  `B = b' * (x, y, -2 z)` for a symmetry axis along z (general axis
  supported), the standard MOT field. `gradient_T_per_m` is the radial
  gradient `b'`; the axial gradient is `2 b'`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MagneticFieldConfig(ABC):
    """Abstract magnetic field `B(r, t)` in tesla."""

    name: str = "field"

    @abstractmethod
    def vector(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Evaluate `B(r, t)` in tesla. Last axis of `positions_m` is 3."""


@dataclass(frozen=True)
class UniformMagneticField(MagneticFieldConfig):
    """Spatially uniform bias field."""

    field_T: ArrayLike = (0.0, 0.0, 0.0)
    name: str = "uniform_field"

    def __post_init__(self) -> None:
        field = np.asarray(self.field_T, dtype=float)
        if field.shape != (3,):
            raise ValueError("field_T must be a 3-vector in tesla.")
        object.__setattr__(self, "field_T", field)

    def vector(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        return np.broadcast_to(self.field_T, positions.shape).copy()


@dataclass(frozen=True)
class QuadrupoleMagneticField(MagneticFieldConfig):
    """Linear quadrupole (anti-Helmholtz) field.

    In the frame where `axis` is z: `B = b' * (x, y, -2 z)` with
    `b' = gradient_T_per_m` the transverse gradient. The field vanishes
    at `center_m`.
    """

    gradient_T_per_m: float = 0.1
    center_m: ArrayLike = (0.0, 0.0, 0.0)
    axis: ArrayLike = (0.0, 0.0, 1.0)
    name: str = "quadrupole_field"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_m must be a 3-vector in meters.")
        axis = np.asarray(self.axis, dtype=float)
        if axis.shape != (3,):
            raise ValueError("axis must be a 3-vector.")
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("axis must be non-zero.")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "axis", axis / norm)

    def vector(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        offsets = positions - self.center_m
        axial = np.sum(offsets * self.axis, axis=-1, keepdims=True)
        radial = offsets - axial * self.axis
        return self.gradient_T_per_m * (radial - 2.0 * axial * self.axis)


def total_magnetic_field(
    fields: MagneticFieldConfig | Iterable[MagneticFieldConfig],
    positions_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum field vectors of one or more configurations at `time_s`."""

    positions = _as_positions(positions_m)
    out = np.zeros_like(positions, dtype=float)
    for field in _field_list(fields):
        out = out + field.vector(positions, time_s=time_s)
    return out


def _field_list(
    fields: MagneticFieldConfig | Iterable[MagneticFieldConfig],
) -> list[MagneticFieldConfig]:
    if isinstance(fields, MagneticFieldConfig):
        return [fields]
    return list(fields)


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
