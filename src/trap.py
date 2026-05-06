"""Trap potentials, possibly time-varying.

`TrapConfig` is the abstract interface for any 3D potential `U(r, t)`. Three
concrete implementations are provided:

- `GaussianTrap` — cylindrically symmetric, time-independent. Was the
  original `TrapConfig`. Use it for static SLM traps and for snapshots of
  moving traps frozen at one time.
- `MovingGaussianTrap` — cylindrically symmetric, with center and depth
  driven by a `RampSequence`.
- `AstigmaticAODTrap` — astigmatic Gaussian whose `x` and `y` axial focal
  points are velocity-coupled (`z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`).
  Models the AOD lensing effect from `aod_slm_movement_v2`.

`total_potential`, `total_force`, `total_hessian` accept any iterable of
`TrapConfig` and a time, and linearly sum the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ramp import RampSequence
from .units import microkelvin_to_joule


class TrapConfig(ABC):
    """Abstract trap potential `U(r, t)`.

    Subclasses must provide `potential` and `center_at`. Default `force` and
    `hessian` use central finite differences; subclasses override them when
    they have analytic gradients.
    """

    name: str = "trap"

    @abstractmethod
    def center_at(self, time_s: float) -> NDArray[np.float64]:
        """Return the natural anchor point (e.g. the trap minimum) at `t`."""

    @abstractmethod
    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Evaluate `U(r, t)` in joules. Last axis of `positions_m` is 3."""

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Evaluate `F = -grad U` in newtons. Default: central differences."""

        positions = _as_positions(positions_m)
        step = self._finite_difference_step()
        force = np.zeros_like(positions, dtype=float)
        basis = np.eye(3)
        for axis in range(3):
            delta = step * basis[axis]
            plus = self.potential(positions + delta, time_s=time_s)
            minus = self.potential(positions - delta, time_s=time_s)
            force[..., axis] = -(plus - minus) / (2.0 * step)
        return force

    def hessian(
        self, position_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Hessian `d^2 U / dr_i dr_j` in J/m^2. Default: central differences."""

        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,):
            raise ValueError("position_m must be a 3-vector in meters.")
        step = self._finite_difference_step()
        basis = np.eye(3)
        u_center = float(self.potential(position, time_s=time_s))
        h = np.zeros((3, 3), dtype=float)
        for i in range(3):
            plus_i = float(self.potential(position + step * basis[i], time_s=time_s))
            minus_i = float(self.potential(position - step * basis[i], time_s=time_s))
            h[i, i] = (plus_i - 2.0 * u_center + minus_i) / (step * step)
        for i in range(3):
            for j in range(i + 1, 3):
                d_i = step * basis[i]
                d_j = step * basis[j]
                mixed = (
                    float(self.potential(position + d_i + d_j, time_s=time_s))
                    - float(self.potential(position + d_i - d_j, time_s=time_s))
                    - float(self.potential(position - d_i + d_j, time_s=time_s))
                    + float(self.potential(position - d_i - d_j, time_s=time_s))
                ) / (4.0 * step * step)
                h[i, j] = mixed
                h[j, i] = mixed
        return h

    def _finite_difference_step(self) -> float:
        """Override if a class has a natural length scale (e.g. waist)."""

        return 1.0e-9


@dataclass(frozen=True)
class GaussianTrap(TrapConfig):
    """Cylindrically symmetric red-detuned 3D Gaussian, time-independent.

    `U(r) = -U0 exp(-2 (x^2 + y^2) / wr^2 - 2 z^2 / wz^2)`.
    `waist_axial_m` is the Gaussian axial 1/e^2 scale used directly by the
    model; it is **not** the diffraction Rayleigh length of a focused beam.
    """

    center_m: ArrayLike = (0.0, 0.0, 0.0)
    waist_radial_m: float = 1.0e-6
    waist_axial_m: float = 4.0e-6
    depth_uK: float = 0.0
    name: str = "gaussian"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float)
        if center.shape != (3,):
            raise ValueError("center_m must be a 3-vector in meters.")
        if self.waist_radial_m <= 0.0:
            raise ValueError("waist_radial_m must be positive.")
        if self.waist_axial_m <= 0.0:
            raise ValueError("waist_axial_m must be positive.")
        if self.depth_uK < 0.0:
            raise ValueError("depth_uK must be non-negative.")
        object.__setattr__(self, "center_m", center)

    @property
    def axial_scale_m(self) -> float:
        return float(self.waist_axial_m)

    @property
    def depth_joule(self) -> float:
        return float(microkelvin_to_joule(self.depth_uK))

    @property
    def scales_m(self) -> NDArray[np.float64]:
        return np.asarray(
            [self.waist_radial_m, self.waist_radial_m, self.waist_axial_m],
            dtype=float,
        )

    def with_center_depth(
        self, center_m: ArrayLike, depth_uK: float
    ) -> "GaussianTrap":
        return GaussianTrap(
            center_m=np.asarray(center_m, dtype=float),
            waist_radial_m=self.waist_radial_m,
            waist_axial_m=self.waist_axial_m,
            depth_uK=float(depth_uK),
            name=self.name,
        )

    def center_at(self, time_s: float = 0.0) -> NDArray[np.float64]:
        return np.asarray(self.center_m, dtype=float)

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        offsets = positions - self.center_m
        scaled = offsets / self.scales_m
        exponent = -2.0 * np.sum(scaled * scaled, axis=-1)
        return -self.depth_joule * np.exp(exponent)

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        offsets = positions - self.center_m
        scaled = offsets / self.scales_m
        exponent = -2.0 * np.sum(scaled * scaled, axis=-1)
        exp_factor = np.asarray(np.exp(exponent))[..., np.newaxis]
        scale_sq = self.scales_m * self.scales_m
        return -4.0 * self.depth_joule * exp_factor * offsets / scale_sq

    def hessian(
        self, position_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
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

    def _finite_difference_step(self) -> float:
        return float(min(self.waist_radial_m, self.waist_axial_m)) * 1.0e-3


@dataclass(frozen=True)
class MovingGaussianTrap(TrapConfig):
    """Cylindrically symmetric Gaussian whose center and depth follow a ramp."""

    template: GaussianTrap = field(default_factory=GaussianTrap)
    ramp: RampSequence = field(default=None)  # type: ignore[assignment]
    name: str = "moving_gaussian"

    def __post_init__(self) -> None:
        if self.ramp is None:
            raise ValueError("MovingGaussianTrap requires a RampSequence.")

    def snapshot(self, time_s: float) -> GaussianTrap:
        """Return a static `GaussianTrap` with this trap's parameters at `t`."""

        center, depth = self.ramp.at(time_s)
        return self.template.with_center_depth(center, depth)

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        return self.ramp.center_at(time_s)

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        return self.snapshot(time_s).potential(positions_m, time_s=time_s)

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        return self.snapshot(time_s).force(positions_m, time_s=time_s)

    def hessian(
        self, position_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        return self.snapshot(time_s).hessian(position_m, time_s=time_s)


@dataclass(frozen=True)
class AstigmaticAODTrap(TrapConfig):
    """Astigmatic Gaussian AOD trap with velocity-coupled focal lensing.

    Reproduces the `aod_slm_movement_v2` model: the radial waist is
    `waist_radial_m`, the wavelength is `wavelength_m`, and the
    Rayleigh length is computed as `zR = pi * w0^2 / lambda`. The trap
    center and depth come from `ramp`, evaluated smoothly. The per-axis
    axial focal points are `z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`,
    where `(vx, vy, vz) = ramp.velocity_at(t)`.

    Set `dxdt2z = 0` to model a pure cylindrically symmetric AOD without
    lensing (still time-dependent).
    """

    waist_radial_m: float = 5.0e-7
    wavelength_m: float = 8.5e-7
    ramp: RampSequence = field(default=None)  # type: ignore[assignment]
    dxdt2z: float = 0.0
    name: str = "astigmatic_aod"

    def __post_init__(self) -> None:
        if self.ramp is None:
            raise ValueError("AstigmaticAODTrap requires a RampSequence.")
        if self.waist_radial_m <= 0.0:
            raise ValueError("waist_radial_m must be positive.")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive.")

    @property
    def rayleigh_length_m(self) -> float:
        return float(np.pi * self.waist_radial_m**2 / self.wavelength_m)

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        return self.ramp.center_at(time_s)

    def _focal_offsets(self, time_s: float) -> tuple[float, float]:
        velocity = self.ramp.velocity_at(time_s)
        return (
            float(self.dxdt2z * velocity[0]),
            float(self.dxdt2z * velocity[1]),
        )

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        offsets = positions - self.center_at(time_s)
        x = offsets[..., 0]
        y = offsets[..., 1]
        z = offsets[..., 2]
        z01, z02 = self._focal_offsets(time_s)
        depth_joule = float(microkelvin_to_joule(self.ramp.depth_at(time_s)))
        w0 = self.waist_radial_m
        zR_sq = self.rayleigh_length_m**2
        qx_sq = 1.0 + (z - z01) ** 2 / zR_sq
        qy_sq = 1.0 + (z - z02) ** 2 / zR_sq
        expx = np.exp(-2.0 * x * x / (w0 * w0 * qx_sq))
        expy = np.exp(-2.0 * y * y / (w0 * w0 * qy_sq))
        profile = expx * expy / np.sqrt(qx_sq * qy_sq)
        return -depth_joule * profile

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        offsets = positions - self.center_at(time_s)
        x = offsets[..., 0]
        y = offsets[..., 1]
        z = offsets[..., 2]
        z01, z02 = self._focal_offsets(time_s)
        depth_joule = float(microkelvin_to_joule(self.ramp.depth_at(time_s)))
        w0 = self.waist_radial_m
        zR_sq = self.rayleigh_length_m**2
        qx_sq = 1.0 + (z - z01) ** 2 / zR_sq
        qy_sq = 1.0 + (z - z02) ** 2 / zR_sq
        expx = np.exp(-2.0 * x * x / (w0 * w0 * qx_sq))
        expy = np.exp(-2.0 * y * y / (w0 * w0 * qy_sq))
        profile = expx * expy / np.sqrt(qx_sq * qy_sq)
        u = -depth_joule * profile

        fx = u * 4.0 * x / (w0 * w0 * qx_sq)
        fy = u * 4.0 * y / (w0 * w0 * qy_sq)
        # dq^2/dz factors
        dqx2_dz = 2.0 * (z - z01) / zR_sq
        dqy2_dz = 2.0 * (z - z02) / zR_sq
        # d/dz of (-2 x^2 / w0^2 / qx^2) gives (2 x^2 / w0^2 / qx^4) * dqx2_dz
        d_logprofile_dz = (
            (2.0 * x * x / (w0 * w0)) * dqx2_dz / (qx_sq * qx_sq)
            + (2.0 * y * y / (w0 * w0)) * dqy2_dz / (qy_sq * qy_sq)
            - 0.5 * dqx2_dz / qx_sq
            - 0.5 * dqy2_dz / qy_sq
        )
        fz = -u * d_logprofile_dz
        return np.stack([fx, fy, fz], axis=-1)

    def _finite_difference_step(self) -> float:
        return float(self.waist_radial_m) * 1.0e-3


def total_potential(
    traps: TrapConfig | Iterable[TrapConfig],
    positions_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum potentials of one or more traps at `time_s`."""

    positions = _as_positions(positions_m)
    out = np.zeros(positions.shape[:-1], dtype=float)
    for trap in _trap_list(traps):
        out = out + trap.potential(positions, time_s=time_s)
    return out


def total_force(
    traps: TrapConfig | Iterable[TrapConfig],
    positions_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum forces of one or more traps at `time_s`."""

    positions = _as_positions(positions_m)
    out = np.zeros_like(positions, dtype=float)
    for trap in _trap_list(traps):
        out = out + trap.force(positions, time_s=time_s)
    return out


def total_hessian(
    traps: TrapConfig | Iterable[TrapConfig],
    position_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum Hessians of one or more traps at `time_s` and one position."""

    out = np.zeros((3, 3), dtype=float)
    for trap in _trap_list(traps):
        out = out + trap.hessian(position_m, time_s=time_s)
    return out


def _trap_list(traps: TrapConfig | Iterable[TrapConfig]) -> list[TrapConfig]:
    if isinstance(traps, TrapConfig):
        return [traps]
    return list(traps)


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
