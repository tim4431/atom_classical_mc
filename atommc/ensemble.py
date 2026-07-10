"""Initial-ensemble sources: where the atoms come from before the run.

A `run_simulation` needs starting positions and velocities. Three
physically distinct sources cover the cases this simulator cares about,
all behind one `EnsembleSource.sample(n, rng)` interface returning an
`EnsembleSample`:

- `ThermalCloud` — a cloud of atoms already localized near the trap
  region: a background vapor filling a capture volume, or a pre-cooled /
  pre-slowed cloud handed to the MOT. Positions are a Gaussian blob (or a
  uniform fill of a box/sphere); velocities are isotropic
  Maxwell-Boltzmann at temperature `T`, optionally plus a bulk drift.
- `HarmonicTrapCloud` — the equilibrium thermal cloud of a conservative
  trap: positions drawn from the local harmonic approximation (covariance
  `k_B T K^-1`), velocities Maxwell-Boltzmann. This is the default when a
  run has traps and no explicit source; wrapping it as a source just
  makes it selectable and composable.
- `EffusiveBeam` — an oven with a small hole. Atoms leak out through a
  thin aperture (hole diameter << mean free path, so no collisions in the
  aperture: the Knudsen / effusive regime). This is NOT the interior gas
  distribution: the escape rate of an atom is proportional to the flux it
  presents to the hole, `v * cos(theta)`, which reshapes both the speed
  and the angular distribution (see `EffusiveBeam`).

The `weight` field of an `EnsembleSample` carries the fraction of the
true physical source that the simulated ensemble represents. It is 1.0
for a cloud sampled in full, but for a beam whose slow capturable tail is
sampled over a truncated velocity window it is the analytic flux fraction
of that window, so an absolute capture fraction is `p_capture * weight`.

Out of scope (documented so callers don't mistake silence for support):
a *large*-hole oven / nozzle is hydrodynamic or supersonic — collisions
during the expansion narrow the velocity spread and boost the beam
forward. That needs atom-atom collisions this simulator does not model;
`EffusiveBeam` is the thin-aperture (small-hole) limit only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import BOLTZMANN_CONSTANT_J_PER_K, RB87_MASS_KG
from .sampling import (
    sample_thermal_positions_harmonic,
    sample_thermal_velocities,
)
from .trap import TrapConfig


@dataclass(frozen=True)
class EnsembleSample:
    """A sampled initial ensemble plus the physical weight it represents.

    `positions_m` and `velocities_m_per_s` are both `(n, 3)`. `weight` is
    the fraction of the underlying physical source captured by this
    ensemble: 1.0 for a cloud sampled in full, or the analytic flux
    fraction of the simulated velocity window for a truncated beam.
    """

    positions_m: NDArray[np.float64]
    velocities_m_per_s: NDArray[np.float64]
    weight: float = 1.0


class EnsembleSource(ABC):
    """Abstract source of an initial `(positions, velocities)` ensemble."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> EnsembleSample:
        """Draw `n` atoms. Must accept any positive `n` (used for resampling)."""


# --------------------------------------------------------------------------
# Localized clouds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalCloud(EnsembleSource):
    """Atoms already localized near the trap: Gaussian blob or uniform fill.

    Velocities are isotropic Maxwell-Boltzmann at `temperature_uK` plus an
    optional bulk `mean_velocity_m_per_s` drift (e.g. a launched or slowed
    cloud). Positions are one of:

    - a Gaussian blob of rms width `sigma_m` (scalar or per-axis) about
      `center_m` — the usual "pre-cooled cloud" / MOT starting point;
    - a uniform fill of a `box_half_widths_m` box or a `sphere_radius_m`
      sphere about `center_m` — a background vapor filling a region.

    Exactly one spatial specifier must be given.
    """

    temperature_uK: float
    sigma_m: ArrayLike | float | None = None
    box_half_widths_m: ArrayLike | None = None
    sphere_radius_m: float | None = None
    center_m: ArrayLike = (0.0, 0.0, 0.0)
    mean_velocity_m_per_s: ArrayLike = (0.0, 0.0, 0.0)
    mass_kg: float = RB87_MASS_KG

    def __post_init__(self) -> None:
        specifiers = [
            self.sigma_m is not None,
            self.box_half_widths_m is not None,
            self.sphere_radius_m is not None,
        ]
        if sum(specifiers) != 1:
            raise ValueError(
                "ThermalCloud needs exactly one of sigma_m, box_half_widths_m, "
                "or sphere_radius_m."
            )
        if self.temperature_uK < 0.0:
            raise ValueError("temperature_uK must be non-negative.")
        object.__setattr__(self, "center_m", _as_vec3(self.center_m, "center_m"))
        object.__setattr__(
            self,
            "mean_velocity_m_per_s",
            _as_vec3(self.mean_velocity_m_per_s, "mean_velocity_m_per_s"),
        )
        if self.sigma_m is not None:
            sigma = np.asarray(self.sigma_m, dtype=float)
            if sigma.ndim == 0:
                sigma = np.full(3, float(sigma))
            if sigma.shape != (3,) or np.any(sigma < 0.0):
                raise ValueError("sigma_m must be a non-negative scalar or 3-vector.")
            object.__setattr__(self, "sigma_m", sigma)
        if self.box_half_widths_m is not None:
            half = np.asarray(self.box_half_widths_m, dtype=float)
            if half.ndim == 0:
                half = np.full(3, float(half))
            if half.shape != (3,) or np.any(half < 0.0):
                raise ValueError(
                    "box_half_widths_m must be a non-negative scalar or 3-vector."
                )
            object.__setattr__(self, "box_half_widths_m", half)
        if self.sphere_radius_m is not None and self.sphere_radius_m < 0.0:
            raise ValueError("sphere_radius_m must be non-negative.")

    def sample(self, n: int, rng: np.random.Generator) -> EnsembleSample:
        _require_positive(n)
        center = self.center_m
        if self.sigma_m is not None:
            positions = center + rng.normal(size=(n, 3)) * self.sigma_m
        elif self.box_half_widths_m is not None:
            positions = center + rng.uniform(-1.0, 1.0, size=(n, 3)) * (
                self.box_half_widths_m
            )
        else:
            positions = center + _uniform_ball(n, float(self.sphere_radius_m), rng)
        velocities = (
            sample_thermal_velocities(
                self.temperature_uK, n, rng, mass_kg=self.mass_kg
            )
            + self.mean_velocity_m_per_s
        )
        return EnsembleSample(positions, velocities)


@dataclass(frozen=True)
class HarmonicTrapCloud(EnsembleSource):
    """Equilibrium thermal cloud of conservative traps (covariance k_B T K^-1).

    Wraps `sample_thermal_positions_harmonic`; this is exactly the default
    ensemble a trapped run uses when no explicit source is given, exposed
    as a source so it can be selected or swapped alongside the others.
    """

    traps: TrapConfig | Iterable[TrapConfig]
    temperature_uK: float
    center_m: ArrayLike | None = None
    time_s: float = 0.0
    mass_kg: float = RB87_MASS_KG

    def sample(self, n: int, rng: np.random.Generator) -> EnsembleSample:
        _require_positive(n)
        positions = sample_thermal_positions_harmonic(
            self.traps,
            self.temperature_uK,
            n,
            rng,
            center_m=self.center_m,
            time_s=self.time_s,
        )
        velocities = sample_thermal_velocities(
            self.temperature_uK, n, rng, mass_kg=self.mass_kg
        )
        return EnsembleSample(positions, velocities)


# --------------------------------------------------------------------------
# Effusive atomic beam (oven with a small hole)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EffusiveBeam(EnsembleSource):
    """Atomic beam leaking from a hot oven through a thin small aperture.

    In the effusive (Knudsen) regime the aperture is smaller than the mean
    free path, so escaping atoms do not collide on their way out. An atom
    of velocity ``v`` at angle ``theta`` from the aperture normal escapes
    at a rate proportional to the flux it presents to the hole,
    ``v * cos(theta)``. That single factor reshapes the interior gas
    distribution into the beam distribution in two ways:

    - **Speed.** The interior speed density ``~ v^2 exp(-v^2/2 sigma^2)``
      (``sigma^2 = k_B T / m``) gains an extra ``v``, giving the beam flux
      ``f(v) ~ v^3 exp(-v^2/2 sigma^2)`` — faster atoms are overrepresented,
      most-probable speed ``sqrt(3) sigma``.
    - **Angle.** Integrated over speed the rate per solid angle ``~ cos(theta)``
      — Lambert's cosine law, the ``A cos(theta)`` foreshortening of the
      hole seen from ``theta``.

    Angular model (`angular`): the default ``"cosine"`` is the bare
    thin-aperture emission pattern — Lambert's law ``dN/dOmega ~ cos(theta)``
    truncated to the divergence cone (set `divergence_half_angle_rad` near
    ``pi/2`` for the full hemisphere). Real capillary/channel ovens are
    *more* forward-peaked than cosine (Clausing), so cosine is the
    least-collimated baseline. For a beam aimed at a trap the practical
    collimation half-angle is small, where ``cos(theta) ~ 1`` and the cosine
    *shape* is a sub-percent correction; ``"uniform"`` fills that cone
    uniformly in solid angle instead, an idealized flat-top collimated beam.

    Only the slow tail of a hot beam is ever MOT-capturable, so restrict
    the simulated speeds to ``[v_min_m_per_s, v_max_m_per_s]``; the returned
    sample `weight` is the analytic fraction of the total beam flux carried
    by that window, ``F(v_max) - F(v_min)`` with
    ``F(v) = 1 - exp(-u)(1 + u)``, ``u = v^2 / 2 sigma^2``.

    Positions are launched uniformly over the `aperture_radius_m` disc in
    the plane normal to `direction`, centered at `launch_center_m`.
    """

    temperature_K: float
    mass_kg: float
    aperture_radius_m: float
    v_max_m_per_s: float
    v_min_m_per_s: float = 0.0
    direction: ArrayLike = (0.0, 0.0, 1.0)
    launch_center_m: ArrayLike = (0.0, 0.0, 0.0)
    divergence_half_angle_rad: float = 0.0
    angular: Literal["uniform", "cosine"] = "cosine"

    def __post_init__(self) -> None:
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if self.aperture_radius_m < 0.0:
            raise ValueError("aperture_radius_m must be non-negative.")
        if not 0.0 <= self.v_min_m_per_s < self.v_max_m_per_s:
            raise ValueError("Require 0 <= v_min_m_per_s < v_max_m_per_s.")
        if not 0.0 <= self.divergence_half_angle_rad <= np.pi:
            raise ValueError("divergence_half_angle_rad must be in [0, pi].")
        if self.angular not in ("uniform", "cosine"):
            raise ValueError("angular must be 'uniform' or 'cosine'.")
        axis = _as_vec3(self.direction, "direction")
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("direction must be a non-zero vector.")
        object.__setattr__(self, "direction", axis / norm)
        object.__setattr__(
            self, "launch_center_m", _as_vec3(self.launch_center_m, "launch_center_m")
        )

    @property
    def sigma_speed_m_per_s(self) -> float:
        """Thermal speed scale `sqrt(k_B T / m)`."""

        return float(
            np.sqrt(BOLTZMANN_CONSTANT_J_PER_K * self.temperature_K / self.mass_kg)
        )

    def flux_cdf(self, v_m_per_s: float) -> float:
        """Cumulative flux fraction below speed `v` for `f(v) ~ v^3 exp`."""

        u = v_m_per_s**2 / (2.0 * self.sigma_speed_m_per_s**2)
        return float(1.0 - np.exp(-u) * (1.0 + u))

    def window_fraction(self) -> float:
        """Flux fraction of the total beam carried by [v_min, v_max]."""

        return self.flux_cdf(self.v_max_m_per_s) - self.flux_cdf(self.v_min_m_per_s)

    def sample(self, n: int, rng: np.random.Generator) -> EnsembleSample:
        _require_positive(n)
        speeds = self._sample_speeds(n, rng)
        cos_theta = self._sample_cos_theta(n, rng)
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
        azimuth = rng.uniform(0.0, 2.0 * np.pi, n)

        e1, e2 = _orthonormal_frame(self.direction)
        directions = (
            cos_theta[:, None] * self.direction
            + (sin_theta * np.cos(azimuth))[:, None] * e1
            + (sin_theta * np.sin(azimuth))[:, None] * e2
        )
        velocities = speeds[:, None] * directions

        disc_phi = rng.uniform(0.0, 2.0 * np.pi, n)
        disc_r = self.aperture_radius_m * np.sqrt(rng.random(n))
        positions = (
            self.launch_center_m
            + (disc_r * np.cos(disc_phi))[:, None] * e1
            + (disc_r * np.sin(disc_phi))[:, None] * e2
        )
        return EnsembleSample(positions, velocities, weight=self.window_fraction())

    def _sample_speeds(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Rejection-sample `f(v) ~ v^3 exp(-v^2/2 sigma^2)` on the window.

        Envelope: the ``v^3`` term (exactly invertible on the window),
        accepted with the Gaussian factor. Acceptance is near 1 when
        ``v_max << sigma`` (the usual slow-tail window), so the loop
        rarely iterates more than once.
        """

        sigma = self.sigma_speed_m_per_s
        v_min4 = self.v_min_m_per_s**4
        v_max4 = self.v_max_m_per_s**4
        speeds = np.empty(0)
        while speeds.size < n:
            quantile = rng.random(2 * n)
            candidates = (v_min4 + quantile * (v_max4 - v_min4)) ** 0.25
            accept = rng.random(candidates.size) < np.exp(
                -(candidates**2) / (2.0 * sigma**2)
            )
            speeds = np.concatenate([speeds, candidates[accept]])
        return speeds[:n]

    def _sample_cos_theta(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Polar-angle cosines within the divergence cone."""

        cos_max = np.cos(self.divergence_half_angle_rad)
        u = rng.random(n)
        if self.angular == "uniform":
            # Uniform in solid angle over the cone: cos(theta) = 1 - U (1 - cos_max).
            return 1.0 - u * (1.0 - cos_max)
        # Lambert cosine law truncated to the cone: dN/dOmega ~ cos(theta),
        # so sin^2(theta) is uniform on [0, sin^2(theta_max)].
        sin2_max = 1.0 - cos_max**2
        sin_theta = np.sqrt(u * sin2_max)
        return np.sqrt(np.clip(1.0 - sin_theta**2, 0.0, 1.0))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _as_vec3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return vec


def _require_positive(n: int) -> None:
    if n <= 0:
        raise ValueError("Number of atoms to sample must be positive.")


def _uniform_ball(
    n: int, radius_m: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """`n` points uniformly inside a ball of the given radius, centered at 0."""

    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius_m * rng.random(n) ** (1.0 / 3.0)
    return radii[:, None] * directions


def _orthonormal_frame(
    axis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two unit vectors spanning the plane orthogonal to a unit `axis`."""

    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2
