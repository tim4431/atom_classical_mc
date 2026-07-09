"""Laser beam definitions for radiation-pressure / MOT simulations.

A `LaserBeam` is a monochromatic traveling wave addressing one effective
two-level transition: propagation direction, detuning from the atomic
resonance, peak saturation parameter `s0 = I / I_sat`, polarization
helicity, and an optional collimated-Gaussian transverse profile.

Polarization is described by a single helicity number `h` in [-1, +1]:
the mean photon spin projection along the propagation direction `k`.
`h = +1` is pure sigma+ (left circular about k), `h = -1` pure sigma-,
and intermediate values are treated as an *incoherent* mixture of the
two circular components with weights `(1 + h)/2` and `(1 - h)/2`. This
is the appropriate level of description for the rate-equation internal
state models in `src.internal_state`; coherent (linear/elliptical)
polarization interference effects are out of scope.

`six_beam_mot` builds the standard retro-reflected three-axis MOT beam
set with the correct helicity signs for a quadrupole field whose axial
gradient is `-2 b'` along `axis` (see `QuadrupoleMagneticField`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LaserBeam:
    """One collimated traveling-wave cooling beam.

    - `direction`: propagation unit vector (normalized on construction).
    - `detuning_hz`: laser frequency minus atomic resonance, linear Hz
      (negative = red detuned).
    - `saturation`: peak on-axis saturation parameter `s0 = I / I_sat`.
    - `helicity`: photon spin projection along `direction`, in [-1, 1].
    - `waist_m`: optional Gaussian 1/e^2 intensity radius. `None` means
      a uniform plane wave (infinite aperture).
    - `center_m`: a point on the beam axis (only the transverse offset
      relative to this point matters for the Gaussian profile).
    - `profile`: optional dimensionless intensity profile
      `profile(positions_m) -> array` (leading shape of the input)
      multiplied onto the saturation. Use it for beams that are not
      Gaussian/uniform: apertured top-hats, diffracted grating-sector
      beams, shadowed regions, etc. Composes with `waist_m` if both are
      given.
    """

    direction: ArrayLike = (0.0, 0.0, 1.0)
    detuning_hz: float = 0.0
    saturation: float = 1.0
    helicity: float = 0.0
    waist_m: float | None = None
    center_m: ArrayLike = (0.0, 0.0, 0.0)
    profile: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    name: str = "beam"

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,):
            raise ValueError("direction must be a 3-vector.")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("direction must be non-zero.")
        object.__setattr__(self, "direction", direction / norm)

        center = np.asarray(self.center_m, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_m must be a 3-vector in meters.")
        object.__setattr__(self, "center_m", center)

        if self.saturation < 0.0:
            raise ValueError("saturation must be non-negative.")
        if not -1.0 <= self.helicity <= 1.0:
            raise ValueError("helicity must lie in [-1, 1].")
        if self.waist_m is not None and self.waist_m <= 0.0:
            raise ValueError("waist_m must be positive when provided.")

    def saturation_at(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Local saturation parameter `s(r) = s0 * I(r) / I(0)`."""

        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,):
            raise ValueError("positions must have final dimension 3.")
        if self.waist_m is None:
            saturation = np.full(positions.shape[:-1], float(self.saturation))
        else:
            offsets = positions - self.center_m
            axial = np.sum(offsets * self.direction, axis=-1, keepdims=True)
            radial = offsets - axial * self.direction
            rho_sq = np.sum(radial * radial, axis=-1)
            saturation = self.saturation * np.exp(-2.0 * rho_sq / (self.waist_m**2))
        if self.profile is not None:
            saturation = saturation * np.asarray(self.profile(positions), dtype=float)
        return saturation


def six_beam_mot(
    detuning_hz: float,
    saturation: float,
    waist_m: float | None = None,
    center_m: ArrayLike = (0.0, 0.0, 0.0),
    axis: ArrayLike = (0.0, 0.0, 1.0),
    radial_helicity: float = 1.0,
    axial_helicity: float = -1.0,
) -> list[LaserBeam]:
    """Build the standard six-beam MOT configuration.

    Two counter-propagating pairs transverse to `axis` with helicity
    `radial_helicity`, one pair along `axis` with `axial_helicity`.
    The defaults (`+1` radial, `-1` axial) restore atoms toward the
    field zero of a `QuadrupoleMagneticField` with positive
    `gradient_T_per_m` along the same `axis`, for a red-detuned laser
    and a positive effective magnetic moment.
    """

    axis_v = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_v))
    if norm == 0.0:
        raise ValueError("axis must be non-zero.")
    axis_v = axis_v / norm

    # Two unit vectors orthogonal to the axis.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis_v))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, axis_v) * axis_v
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis_v, e1)

    beams: list[LaserBeam] = []
    for label, direction, helicity in (
        ("mot_+e1", e1, radial_helicity),
        ("mot_-e1", -e1, radial_helicity),
        ("mot_+e2", e2, radial_helicity),
        ("mot_-e2", -e2, radial_helicity),
        ("mot_+axis", axis_v, axial_helicity),
        ("mot_-axis", -axis_v, axial_helicity),
    ):
        beams.append(
            LaserBeam(
                direction=direction,
                detuning_hz=detuning_hz,
                saturation=saturation,
                helicity=helicity,
                waist_m=waist_m,
                center_m=center_m,
                name=label,
            )
        )
    return beams
