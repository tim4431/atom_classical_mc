"""Smooth time-dependent ramp sequences for moving traps.

`RampSequence` is the time-vs-(center, depth) waypoint table that any
time-dependent trap can consume. Between waypoints, position and depth are
interpolated by a `PolynomialConnector` chosen at construction time. The
default is `LINEAR` so a ramp built without specifying a profile reproduces
plain piecewise-linear interpolation. Callers that want smooth motion pick a
named profile (e.g. `QUINTIC_MIN_JERK`) or build a custom one with
`arb_fifth_poly(beta)`.

Each connector exposes both the value and the derivative, so traps can ask
the ramp for `velocity_at(t)` (used by `AstigmaticAODTrap` for the
velocity-coupled focal shift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PolynomialConnector:
    """Smooth interpolation kernel between two waypoints.

    Given waypoints `(x0, y0)` and `(x1, y1)`, the connector evaluates
    `y(x) = y0 + (y1 - y0) * sum_n a_n u^n` with `u = (x - x0) / (x1 - x0)`.
    The polynomial coefficients should satisfy `a_0 = 0`, `sum a_n = 1` so the
    connector matches both endpoint values; smoothness at the joins comes from
    extra zero leading coefficients.
    """

    coefficients: Tuple[float, ...]
    name: str = "polynomial"

    def __post_init__(self) -> None:
        if len(self.coefficients) < 1:
            raise ValueError("coefficients must have at least one entry.")

    def value(self, u: ArrayLike) -> NDArray[np.float64]:
        """Normalized shape function on `u in [0, 1]`."""

        u_array = np.asarray(u, dtype=float)
        out = np.zeros_like(u_array)
        for n, a in enumerate(self.coefficients):
            if a == 0.0:
                continue
            out = out + a * u_array**n
        return out

    def derivative(self, u: ArrayLike) -> NDArray[np.float64]:
        """Derivative of the shape function with respect to `u`."""

        u_array = np.asarray(u, dtype=float)
        out = np.zeros_like(u_array)
        for n, a in enumerate(self.coefficients):
            if n == 0 or a == 0.0:
                continue
            out = out + a * n * u_array ** (n - 1)
        return out


LINEAR = PolynomialConnector((0.0, 1.0), name="linear")
CUBIC_SMOOTHSTEP = PolynomialConnector((0.0, 0.0, 3.0, -2.0), name="cubic_smoothstep")
QUINTIC_MIN_JERK = PolynomialConnector(
    (0.0, 0.0, 0.0, 10.0, -15.0, 6.0), name="quintic_minimum_jerk"
)


def arb_fifth_poly(beta: float) -> PolynomialConnector:
    """Quintic family from `aod_slm_movement_v2.trajectories`.

    `beta = 1` reproduces `QUINTIC_MIN_JERK`; the v2 work uses `beta = 1.5625`.
    """

    return PolynomialConnector(
        (
            0.0,
            0.0,
            15.0 - 8.0 * beta,
            -50.0 + 32.0 * beta,
            60.0 - 40.0 * beta,
            -24.0 + 16.0 * beta,
        ),
        name=f"arb_fifth_poly_beta_{beta:g}",
    )


def const_jerk() -> PolynomialConnector:
    """Cubic with constant jerk (v2's `const_jerk`)."""

    return PolynomialConnector((0.0, 0.0, 3.0, -2.0), name="const_jerk")


@dataclass(frozen=True)
class RampSequence:
    """Time-indexed center-and-depth waypoint table for one trap.

    `times_s` must be strictly increasing. Centers are 3-vectors per waypoint;
    depths are scalars in microkelvin. Outside `[times_s[0], times_s[-1]]`,
    `at(t)` clamps to the endpoint values and `velocity_at(t)` returns zero.

    `position_profile` and `depth_profile` choose the per-segment
    interpolation kernel. Both default to LINEAR so a plain waypoint list
    behaves like `np.interp`.
    """

    times_s: ArrayLike
    centers_m: ArrayLike
    depths_uK: ArrayLike
    position_profile: PolynomialConnector = field(default=LINEAR)
    depth_profile: PolynomialConnector = field(default=LINEAR)

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        centers = np.asarray(self.centers_m, dtype=float)
        depths = np.asarray(self.depths_uK, dtype=float)

        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times_s must be a 1D array with at least two points.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing.")
        if centers.shape != (len(times), 3):
            raise ValueError("centers_m must have shape (len(times_s), 3).")
        if depths.shape != (len(times),):
            raise ValueError("depths_uK must have shape (len(times_s),).")
        if np.any(depths < 0.0):
            raise ValueError("depths_uK must be non-negative.")

        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "centers_m", centers)
        object.__setattr__(self, "depths_uK", depths)

    @property
    def start_time_s(self) -> float:
        return float(self.times_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.times_s[-1])

    def at(self, time_s: float) -> tuple[NDArray[np.float64], float]:
        """Return `(center, depth)` interpolated at `time_s`."""

        return self.center_at(time_s), self.depth_at(time_s)

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return self._endpoint_center(time_s)
        shape = float(self.position_profile.value(u))
        c0 = self.centers_m[index]
        c1 = self.centers_m[index + 1]
        return c0 + shape * (c1 - c0)

    def depth_at(self, time_s: float) -> float:
        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return self._endpoint_depth(time_s)
        shape = float(self.depth_profile.value(u))
        d0 = float(self.depths_uK[index])
        d1 = float(self.depths_uK[index + 1])
        return d0 + shape * (d1 - d0)

    def velocity_at(self, time_s: float) -> NDArray[np.float64]:
        """Return `dcenter/dt` at `time_s`. Outside the table this is zero."""

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return np.zeros(3, dtype=float)
        dt = float(self.times_s[index + 1] - self.times_s[index])
        slope = float(self.position_profile.derivative(u)) / dt
        c0 = self.centers_m[index]
        c1 = self.centers_m[index + 1]
        return slope * (c1 - c0)

    def depth_rate_at(self, time_s: float) -> float:
        """Return `ddepth/dt` at `time_s` in uK/s."""

        index, u = self._segment_index_and_fraction(time_s)
        if index is None:
            return 0.0
        dt = float(self.times_s[index + 1] - self.times_s[index])
        slope = float(self.depth_profile.derivative(u)) / dt
        d0 = float(self.depths_uK[index])
        d1 = float(self.depths_uK[index + 1])
        return slope * (d1 - d0)

    def _endpoint_center(self, time_s: float) -> NDArray[np.float64]:
        if time_s <= self.start_time_s:
            return np.array(self.centers_m[0], dtype=float)
        return np.array(self.centers_m[-1], dtype=float)

    def _endpoint_depth(self, time_s: float) -> float:
        if time_s <= self.start_time_s:
            return float(self.depths_uK[0])
        return float(self.depths_uK[-1])

    def _segment_index_and_fraction(
        self, time_s: float
    ) -> tuple[int | None, float]:
        t = float(time_s)
        if t <= self.start_time_s or t >= self.end_time_s:
            return None, 0.0
        index = int(np.searchsorted(self.times_s, t, side="right") - 1)
        index = max(0, min(index, len(self.times_s) - 2))
        t0 = float(self.times_s[index])
        t1 = float(self.times_s[index + 1])
        u = (t - t0) / (t1 - t0)
        return index, u


PROFILES: dict[str, PolynomialConnector] = {
    "linear": LINEAR,
    "cubic_smoothstep": CUBIC_SMOOTHSTEP,
    "quintic_minimum_jerk": QUINTIC_MIN_JERK,
}


def named_profile(name: str) -> PolynomialConnector:
    """Look up a built-in profile by name."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown profile {name!r}. Choose from {list(PROFILES)} "
            "or build one with `arb_fifth_poly(beta)`."
        ) from exc


def build_waypoint_ramp(
    times_s: Sequence[float],
    centers_m: Sequence[Sequence[float]],
    depths_uK: Sequence[float],
    position_profile: PolynomialConnector = LINEAR,
    depth_profile: PolynomialConnector = LINEAR,
) -> RampSequence:
    """Convenience constructor — keyword-only smoothness selection."""

    return RampSequence(
        times_s=np.asarray(times_s, dtype=float),
        centers_m=np.asarray(centers_m, dtype=float),
        depths_uK=np.asarray(depths_uK, dtype=float),
        position_profile=position_profile,
        depth_profile=depth_profile,
    )
