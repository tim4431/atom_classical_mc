"""Gaussian optical tweezer potentials and analytic forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .units import microkelvin_to_joule


@dataclass(frozen=True)
class TrapConfig:
    """A red-detuned attractive 3D Gaussian optical tweezer.

    The v1 model uses a Gaussian axial scale. If `rayleigh_length_m` is used,
    it is treated as that axial Gaussian scale rather than the exact
    diffraction formula for a focused beam.
    """

    center_m: ArrayLike
    waist_radial_m: float
    depth_uK: float
    waist_axial_m: float | None = None
    rayleigh_length_m: float | None = None
    name: str = "trap"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_m must be a 3-vector in meters.")
        if self.waist_radial_m <= 0.0:
            raise ValueError("waist_radial_m must be positive.")
        if self.depth_uK < 0.0:
            raise ValueError("depth_uK must be non-negative.")
        if (self.waist_axial_m is None) == (self.rayleigh_length_m is None):
            raise ValueError("Provide exactly one of waist_axial_m or rayleigh_length_m.")
        if self.axial_scale_m <= 0.0:
            raise ValueError("The axial scale must be positive.")
        object.__setattr__(self, "center_m", center)

    @property
    def axial_scale_m(self) -> float:
        """Return the axial Gaussian scale used by the v1 model."""

        if self.waist_axial_m is not None:
            return float(self.waist_axial_m)
        return float(self.rayleigh_length_m)

    @property
    def depth_joule(self) -> float:
        """Trap depth converted from microkelvin to joules."""

        return float(microkelvin_to_joule(self.depth_uK))

    @property
    def scales_m(self) -> NDArray[np.float64]:
        """Return `[radial_waist, radial_waist, axial_scale]` in meters."""

        return np.asarray(
            [self.waist_radial_m, self.waist_radial_m, self.axial_scale_m],
            dtype=float,
        )

    def with_center_depth(self, center_m: ArrayLike, depth_uK: float) -> "TrapConfig":
        """Return a copy with updated center and depth."""

        return TrapConfig(
            center_m=center_m,
            waist_radial_m=self.waist_radial_m,
            depth_uK=depth_uK,
            waist_axial_m=self.waist_axial_m,
            rayleigh_length_m=self.rayleigh_length_m,
            name=self.name,
        )

    def potential(self, positions_m: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the trap potential at one or more positions.

        `positions_m` must have final dimension 3 and may be shaped `(3,)` or
        `(n, 3)`. The returned array has shape `positions_m.shape[:-1]`.
        """

        positions = _as_positions(positions_m)
        offsets = positions - self.center_m
        scaled = offsets / self.scales_m
        exponent = -2.0 * np.sum(scaled * scaled, axis=-1)
        return -self.depth_joule * np.exp(exponent)

    def force(self, positions_m: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the force, `F = -grad U`, at one or more positions."""

        positions = _as_positions(positions_m)
        offsets = positions - self.center_m
        scaled = offsets / self.scales_m
        exponent = -2.0 * np.sum(scaled * scaled, axis=-1)
        exp_factor = np.asarray(np.exp(exponent))[..., np.newaxis]
        scale_sq = self.scales_m * self.scales_m
        return -4.0 * self.depth_joule * exp_factor * offsets / scale_sq

    def hessian(self, position_m: ArrayLike) -> NDArray[np.float64]:
        """Return the Hessian of the potential at a single position."""

        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,):
            raise ValueError("position_m must be a 3-vector in meters.")

        offsets = position - self.center_m
        scale_sq = self.scales_m * self.scales_m
        alpha = 2.0 / scale_sq
        exponent = -np.sum(alpha * offsets * offsets)
        exp_factor = np.exp(exponent)
        diag_term = np.diag(2.0 * self.depth_joule * alpha)
        outer_term = 4.0 * self.depth_joule * np.outer(alpha * offsets, alpha * offsets)
        return exp_factor * (diag_term - outer_term)


def total_potential(
    traps: TrapConfig | Iterable[TrapConfig], positions_m: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the summed potential from one or more traps."""

    positions = _as_positions(positions_m)
    potential = np.zeros(positions.shape[:-1], dtype=float)
    for trap in _trap_list(traps):
        potential = potential + trap.potential(positions)
    return potential


def total_force(
    traps: TrapConfig | Iterable[TrapConfig], positions_m: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the summed force from one or more traps."""

    positions = _as_positions(positions_m)
    force = np.zeros_like(positions, dtype=float)
    for trap in _trap_list(traps):
        force = force + trap.force(positions)
    return force


def total_hessian(
    traps: TrapConfig | Iterable[TrapConfig], position_m: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the summed potential Hessian from one or more traps."""

    hessian = np.zeros((3, 3), dtype=float)
    for trap in _trap_list(traps):
        hessian = hessian + trap.hessian(position_m)
    return hessian


def _trap_list(traps: TrapConfig | Iterable[TrapConfig]) -> list[TrapConfig]:
    if isinstance(traps, TrapConfig):
        return [traps]
    return list(traps)


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
