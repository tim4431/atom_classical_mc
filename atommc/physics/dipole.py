"""Optical dipole potential of a far-detuned focused Gaussian beam.

`DipoleBeamPotential` is a `ConservativeForce` built from laboratory beam
parameters (power, focal waist, wavelength) instead of a hand-specified
`depth_uK`. It uses proper Gaussian-beam optics — the axial scale is the
Rayleigh length `z_R = pi w0^2 / lambda`, unlike `GaussianTrap` whose
`waist_axial_m` is a free model parameter.

Physics: the two-level rotating-wave light shift

    U(r) = (hbar delta / 2) * ln(1 + s(r) / D),   D = 1 + (2 delta / Gamma)^2

with `s = I / I_sat` and `delta` the (angular) detuning of the trap laser
from the species' resonance. Red detuning (`lambda > lambda_atom`,
`delta < 0`) gives an attractive trap; blue gives a repulsive barrier. In
the far-detuned limit this reduces to `U = hbar Gamma^2 s / (8 delta)`.

Model limits (documented, not enforced beyond the detuning check):

- Effective two-level atom: no counter-rotating term, no D1/D2 multi-line
  polarizability (roughly a 10% depth error for Rb at 1064 nm).
- Photon scattering by the trap light is NOT modeled here; the module is
  purely conservative. Construction requires `|delta| >= 100 Gamma` and
  `peak_scattering_rate_per_s` lets you check the residual rate. For
  near-resonant radiation pressure use `LightScattering`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import HBAR_J_S, SPEED_OF_LIGHT_M_PER_S
from ..species import AtomSpecies
from ..units import joule_to_microkelvin
from .base import ConservativeForce, _as_positions


@dataclass(frozen=True)
class DipoleBeamPotential(ConservativeForce):
    """Far-detuned focused Gaussian beam as a conservative potential."""

    species: AtomSpecies
    power_w: float = 1.0e-3
    waist_m: float = 1.0e-6
    wavelength_m: float = 8.5e-7
    focus_m: ArrayLike = (0.0, 0.0, 0.0)
    direction: ArrayLike = (0.0, 0.0, 1.0)
    name: str = "dipole_beam"

    _MIN_DETUNING_LINEWIDTHS = 100.0

    def __post_init__(self) -> None:
        if self.power_w <= 0.0:
            raise ValueError("power_w must be positive.")
        if self.waist_m <= 0.0:
            raise ValueError("waist_m must be positive.")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive.")
        focus = np.asarray(self.focus_m, dtype=float)
        if focus.shape != (3,):
            raise ValueError("focus_m must be a 3-vector in meters.")
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,):
            raise ValueError("direction must be a 3-vector.")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("direction must be non-zero.")
        object.__setattr__(self, "focus_m", focus)
        object.__setattr__(self, "direction", direction / norm)
        min_detuning = self._MIN_DETUNING_LINEWIDTHS * self.species.linewidth_rad_s
        if abs(self.detuning_rad_s) < min_detuning:
            raise ValueError(
                "DipoleBeamPotential is conservative-only and requires "
                f"|detuning| >= {self._MIN_DETUNING_LINEWIDTHS:g} linewidths "
                f"(got {abs(self.detuning_rad_s) / self.species.linewidth_rad_s:.1f}); "
                "nearer-resonant light scatters — use LightScattering instead."
            )

    # -- derived beam/atom quantities ------------------------------------

    @property
    def detuning_rad_s(self) -> float:
        """Angular detuning of the trap laser from resonance (< 0 = red)."""

        return (
            2.0
            * np.pi
            * SPEED_OF_LIGHT_M_PER_S
            * (1.0 / self.wavelength_m - 1.0 / self.species.wavelength_m)
        )

    @property
    def rayleigh_length_m(self) -> float:
        return np.pi * self.waist_m**2 / self.wavelength_m

    @property
    def peak_saturation(self) -> float:
        """On-axis focal `s0 = I(0) / I_sat = 2 P / (pi w0^2 I_sat)`."""

        return (
            2.0
            * self.power_w
            / (np.pi * self.waist_m**2 * self.species.saturation_intensity_w_per_m2)
        )

    @property
    def _lorentz_denominator(self) -> float:
        """`D = 1 + (2 delta / Gamma)^2`."""

        return 1.0 + (2.0 * self.detuning_rad_s / self.species.linewidth_rad_s) ** 2

    @property
    def depth_j(self) -> float:
        """`|U|` at the focus in joules."""

        return abs(
            0.5
            * HBAR_J_S
            * self.detuning_rad_s
            * np.log1p(self.peak_saturation / self._lorentz_denominator)
        )

    @property
    def depth_uK(self) -> float:
        return float(joule_to_microkelvin(self.depth_j))

    @property
    def peak_scattering_rate_per_s(self) -> float:
        """Residual photon-scattering rate at the focus (should be small)."""

        s0 = self.peak_saturation
        return (
            0.5
            * self.species.linewidth_rad_s
            * s0
            / (self._lorentz_denominator + s0)
        )

    # -- ConservativeForce interface --------------------------------------

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        return np.array(self.focus_m, dtype=float)

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        saturation = self._saturation(_as_positions(positions_m))
        return (
            0.5
            * HBAR_J_S
            * self.detuning_rad_s
            * np.log1p(saturation / self._lorentz_denominator)
        )

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Analytic `F = -(dU/ds) grad s` including the axial (Rayleigh) term."""

        positions = _as_positions(positions_m)
        rho_vec, axial, w_sq = self._beam_frame(positions)
        rho_sq = np.sum(rho_vec * rho_vec, axis=-1)
        saturation = self._saturation_from_frame(rho_sq, w_sq)

        du_ds = (
            0.5
            * HBAR_J_S
            * self.detuning_rad_s
            / (self._lorentz_denominator + saturation)
        )
        z_r_sq = self.rayleigh_length_m**2
        # grad s = s * [ -(4/w^2) rho_vec + (4 rho^2/w^2 - 2) z/(z_R^2 + z^2) n ]
        grad_s = saturation[..., None] * (
            -(4.0 / w_sq[..., None]) * rho_vec
            + (
                (4.0 * rho_sq / w_sq - 2.0)
                * axial
                / (z_r_sq + axial**2)
            )[..., None]
            * self.direction
        )
        return -du_ds[..., None] * grad_s

    def _finite_difference_step(self) -> float:
        return 1.0e-3 * self.waist_m

    # -- internals ---------------------------------------------------------

    def _beam_frame(
        self, positions: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (transverse offset vector, axial offset, local w^2)."""

        offsets = positions - self.focus_m
        axial = np.sum(offsets * self.direction, axis=-1)
        rho_vec = offsets - axial[..., None] * self.direction
        w_sq = self.waist_m**2 * (1.0 + (axial / self.rayleigh_length_m) ** 2)
        return rho_vec, axial, w_sq

    def _saturation(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        rho_vec, _, w_sq = self._beam_frame(positions)
        rho_sq = np.sum(rho_vec * rho_vec, axis=-1)
        return self._saturation_from_frame(rho_sq, w_sq)

    def _saturation_from_frame(
        self, rho_sq: NDArray[np.float64], w_sq: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        peak = (
            2.0
            * self.power_w
            / (np.pi * w_sq * self.species.saturation_intensity_w_per_m2)
        )
        return peak * np.exp(-2.0 * rho_sq / w_sq)
