"""Optical trap potentials, possibly time-varying.

All traps are `ConservativeForce` physics modules (see `physics.base`). Four
concrete implementations are provided:

- `GaussianTrap` — cylindrically symmetric, time-independent. Use it for
  static SLM traps and for snapshots of moving traps frozen at one time.
- `MovingGaussianTrap` — cylindrically symmetric, with center and depth
  driven by a `RampSequence`.
- `AstigmaticAODTrap` — astigmatic Gaussian whose `x` and `y` axial focal
  points are velocity-coupled (`z01 = dxdt2z * vx`, `z02 = dxdt2z * vy`).
  Models the AOD lensing effect from `aod_slm_movement_v2`.
- `GriddedTrap` — tabulated potential on a uniform 3D grid, sampled once
  at construction. Use for numerically-defined fields or to cache an
  expensive analytic potential.

`total_potential`, `total_force`, `total_hessian` live in `physics.base`
and linearly sum any iterable of `ConservativeForce` at a chosen time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..ramp import RampSequence
from ..units import microkelvin_to_joule
from .base import ConservativeForce, _as_positions

@dataclass(frozen=True)
class GaussianTrap(ConservativeForce):
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
class MovingGaussianTrap(ConservativeForce):
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
class AstigmaticAODTrap(ConservativeForce):
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


@dataclass(frozen=True)
class GriddedTrap(ConservativeForce):
    """Potential tabulated on a uniform 3D grid in the trap-local frame.

    The grid stores `U[i, j, k]` in joules on a regular lattice that spans
    `origin_local_m` to `origin_local_m + (shape - 1) * spacing_m`,
    expressed relative to the trap center. Force is computed as the
    analytical derivative of the interpolating kernel applied to that
    same potential grid — so force and potential are always exactly
    consistent (force is conservative on the interpolant).

    Two interpolation modes:

    - `"tricubic"` (default): Catmull-Rom cardinal cubic, 4 x 4 x 4 = 64
      stencil points. C^1 across cell boundaries: continuous force,
      bounded (piecewise C^0) Hessian.
    - `"trilinear"`: 2 x 2 x 2 = 8 stencil points. C^0 potential,
      piecewise-constant force per axis (discontinuous at cell faces),
      delta-like Hessian. Faster, useful as a baseline.

    Outside the grid both `potential` and `force` return zero. That is
    consistent with a localized trap and with the simulator's
    energy >= 0 loss criterion. Near the boundary the tricubic stencil
    is edge-clamped (edge replication), which is innocuous when the
    trap is sized so the field is near zero at the box edges.

    Pass `ramp` to make the trap translate rigidly through the lab; the
    stored field shape is fixed at construction.

    Direct construction takes the grid already built. The convenience
    classmethods `from_callable` and `from_trap` build it by sampling.
    """

    grid_potential_j: NDArray[np.float64]
    origin_local_m: NDArray[np.float64]
    spacing_m: NDArray[np.float64]
    center_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    ramp: RampSequence | None = None
    interpolation: str = "tricubic"
    name: str = "gridded"

    def __post_init__(self) -> None:
        u = np.asarray(self.grid_potential_j, dtype=float)
        if u.ndim != 3 or any(n < 2 for n in u.shape):
            raise ValueError(
                "grid_potential_j must have shape (Nx, Ny, Nz) with each N >= 2."
            )
        origin = np.asarray(self.origin_local_m, dtype=float)
        spacing = np.asarray(self.spacing_m, dtype=float)
        center = np.asarray(self.center_m, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin_local_m must be a 3-vector.")
        if spacing.shape != (3,):
            raise ValueError("spacing_m must be a 3-vector.")
        if center.shape != (3,):
            raise ValueError("center_m must be a 3-vector.")
        if np.any(spacing <= 0.0):
            raise ValueError("spacing_m must be positive on every axis.")
        if self.interpolation not in ("tricubic", "trilinear"):
            raise ValueError("interpolation must be 'tricubic' or 'trilinear'.")
        object.__setattr__(self, "grid_potential_j", u)
        object.__setattr__(self, "origin_local_m", origin)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "center_m", center)

    @classmethod
    def from_callable(
        cls,
        potential_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        bounds_m: ArrayLike,
        shape: tuple[int, int, int],
        center_m: ArrayLike = (0.0, 0.0, 0.0),
        ramp: RampSequence | None = None,
        interpolation: str = "tricubic",
        name: str = "gridded",
    ) -> "GriddedTrap":
        """Sample `potential_fn` on a uniform grid and cache the result.

        `potential_fn(r_local)` takes positions in the trap-local frame
        (last axis = 3) and returns the potential in joules with shape
        matching the leading axes of the input.
        """

        bounds = np.asarray(bounds_m, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError("bounds_m must have shape (3, 2): per-axis (min, max).")
        shape_t = tuple(int(s) for s in shape)
        if len(shape_t) != 3 or any(n < 2 for n in shape_t):
            raise ValueError("shape must be a 3-tuple of ints, each >= 2.")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("bounds_m max must exceed min on every axis.")

        axes = [
            np.linspace(bounds[i, 0], bounds[i, 1], shape_t[i]) for i in range(3)
        ]
        spacing = np.array(
            [(bounds[i, 1] - bounds[i, 0]) / (shape_t[i] - 1) for i in range(3)],
            dtype=float,
        )
        grid_x, grid_y, grid_z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        sample_points = np.stack([grid_x, grid_y, grid_z], axis=-1)
        potential_j = np.asarray(potential_fn(sample_points), dtype=float)
        if potential_j.shape != shape_t:
            raise ValueError(
                f"potential_fn returned shape {potential_j.shape}, expected {shape_t}."
            )
        origin = bounds[:, 0].copy()
        return cls(
            grid_potential_j=potential_j,
            origin_local_m=origin,
            spacing_m=spacing,
            center_m=np.asarray(center_m, dtype=float),
            ramp=ramp,
            interpolation=interpolation,
            name=name,
        )

    @classmethod
    def from_trap(
        cls,
        source: ConservativeForce,
        bounds_m: ArrayLike,
        shape: tuple[int, int, int],
        time_s: float = 0.0,
        ramp: RampSequence | None = None,
        interpolation: str = "tricubic",
        name: str | None = None,
    ) -> "GriddedTrap":
        """Cache an existing `ConservativeForce` as a gridded copy.

        Samples `source` at `time_s` over a box defined by `bounds_m`
        relative to `source.center_at(time_s)`. The resulting gridded
        trap's anchor is that same source center, unless `ramp` is passed
        to make it move independently.
        """

        anchor = np.asarray(source.center_at(time_s), dtype=float)

        def _pot_local(r_local: NDArray[np.float64]) -> NDArray[np.float64]:
            return source.potential(r_local + anchor, time_s=time_s)

        return cls.from_callable(
            potential_fn=_pot_local,
            bounds_m=bounds_m,
            shape=shape,
            center_m=anchor,
            ramp=ramp,
            interpolation=interpolation,
            name=name if name is not None else f"gridded_{source.name}",
        )

    def center_at(self, time_s: float = 0.0) -> NDArray[np.float64]:
        if self.ramp is None:
            return np.asarray(self.center_m, dtype=float)
        return self.ramp.center_at(time_s)

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        local = positions - self.center_at(time_s)
        return _evaluate_grid(
            self.grid_potential_j,
            self.origin_local_m,
            self.spacing_m,
            local,
            mode=self.interpolation,
            gradient=False,
        )

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        local = positions - self.center_at(time_s)
        gradient = _evaluate_grid(
            self.grid_potential_j,
            self.origin_local_m,
            self.spacing_m,
            local,
            mode=self.interpolation,
            gradient=True,
        )
        return -gradient

    def _finite_difference_step(self) -> float:
        return float(np.min(self.spacing_m)) * 2.0


def _kernel_weights_1d(
    t: NDArray[np.float64], mode: str, derivative: bool
) -> NDArray[np.float64]:
    """1D Catmull-Rom (mode='tricubic') or linear (mode='trilinear') weights.

    Returns shape `(K, M)` where `K = 4` for tricubic and `K = 2` for
    trilinear. If `derivative` is True, returns dweight/dt at the same
    knots (to be divided by axis spacing in the caller).
    """

    if mode == "tricubic":
        if not derivative:
            t2 = t * t
            t3 = t2 * t
            return 0.5 * np.stack(
                [
                    -t + 2.0 * t2 - t3,
                    2.0 - 5.0 * t2 + 3.0 * t3,
                    t + 4.0 * t2 - 3.0 * t3,
                    -t2 + t3,
                ],
                axis=0,
            )
        t2 = t * t
        return 0.5 * np.stack(
            [
                -1.0 + 4.0 * t - 3.0 * t2,
                -10.0 * t + 9.0 * t2,
                1.0 + 8.0 * t - 9.0 * t2,
                -2.0 * t + 3.0 * t2,
            ],
            axis=0,
        )
    # trilinear
    ones = np.ones_like(t)
    if not derivative:
        return np.stack([1.0 - t, t], axis=0)
    return np.stack([-ones, ones], axis=0)


def _evaluate_grid(
    grid: NDArray[np.float64],
    origin_m: NDArray[np.float64],
    spacing_m: NDArray[np.float64],
    query_positions_local_m: NDArray[np.float64],
    mode: str,
    gradient: bool,
) -> NDArray[np.float64]:
    """Vectorized interpolation or analytical gradient of `grid`.

    `grid` must be a scalar field of shape `(Nx, Ny, Nz)`. Query points
    have shape `(..., 3)` in the same local frame as `origin_m`.

    If `gradient=False`, returns the interpolated value with the leading
    shape of the query. If `gradient=True`, returns the analytical
    gradient of the interpolant with shape `(..., 3)`. Outside the grid
    both are zero. Out-of-bounds stencil indices are edge-clamped.
    """

    nx, ny, nz = grid.shape
    query = np.asarray(query_positions_local_m, dtype=float)
    leading_shape = query.shape[:-1]
    flat = query.reshape(-1, 3)

    u = (flat - origin_m) / spacing_m
    i_floor = np.floor(u).astype(np.int64)
    frac = u - i_floor

    valid = (
        (i_floor[:, 0] >= 0)
        & (i_floor[:, 0] <= nx - 2)
        & (i_floor[:, 1] >= 0)
        & (i_floor[:, 1] <= ny - 2)
        & (i_floor[:, 2] >= 0)
        & (i_floor[:, 2] <= nz - 2)
    )

    if mode == "tricubic":
        offsets = np.array([-1, 0, 1, 2], dtype=np.int64)
    elif mode == "trilinear":
        offsets = np.array([0, 1], dtype=np.int64)
    else:
        raise ValueError(f"Unknown interpolation mode: {mode!r}")

    sx = np.clip(i_floor[:, 0][None, :] + offsets[:, None], 0, nx - 1)
    sy = np.clip(i_floor[:, 1][None, :] + offsets[:, None], 0, ny - 1)
    sz = np.clip(i_floor[:, 2][None, :] + offsets[:, None], 0, nz - 1)

    vals = grid[
        sx[:, None, None, :], sy[None, :, None, :], sz[None, None, :, :]
    ]  # (K, K, K, M)

    wx = _kernel_weights_1d(frac[:, 0], mode, derivative=False)
    wy = _kernel_weights_1d(frac[:, 1], mode, derivative=False)
    wz = _kernel_weights_1d(frac[:, 2], mode, derivative=False)

    if not gradient:
        weights = (
            wx[:, None, None, :]
            * wy[None, :, None, :]
            * wz[None, None, :, :]
        )
        out = np.einsum("abcm,abcm->m", weights, vals)
        out = np.where(valid, out, 0.0)
        return out.reshape(leading_shape)

    dwx = _kernel_weights_1d(frac[:, 0], mode, derivative=True)
    dwy = _kernel_weights_1d(frac[:, 1], mode, derivative=True)
    dwz = _kernel_weights_1d(frac[:, 2], mode, derivative=True)

    w_x = (
        dwx[:, None, None, :]
        * wy[None, :, None, :]
        * wz[None, None, :, :]
    )
    w_y = (
        wx[:, None, None, :]
        * dwy[None, :, None, :]
        * wz[None, None, :, :]
    )
    w_z = (
        wx[:, None, None, :]
        * wy[None, :, None, :]
        * dwz[None, None, :, :]
    )

    gx = np.einsum("abcm,abcm->m", w_x, vals) / spacing_m[0]
    gy = np.einsum("abcm,abcm->m", w_y, vals) / spacing_m[1]
    gz = np.einsum("abcm,abcm->m", w_z, vals) / spacing_m[2]
    grad = np.stack([gx, gy, gz], axis=-1)
    grad = np.where(valid[:, None], grad, 0.0)
    return grad.reshape(leading_shape + (3,))
