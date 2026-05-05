"""Time-dependent AOD tweezer ramp sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RampSequence:
    """Piecewise-linear AOD center and depth sequence."""

    times_s: ArrayLike
    centers_m: ArrayLike
    depths_uK: ArrayLike

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
        """Interpolate the AOD center and depth at `time_s`.

        Times before the first point and after the last point are clamped to
        the nearest endpoint.
        """

        center = np.asarray(
            [np.interp(time_s, self.times_s, self.centers_m[:, axis]) for axis in range(3)],
            dtype=float,
        )
        depth = float(np.interp(time_s, self.times_s, self.depths_uK))
        return center, depth
