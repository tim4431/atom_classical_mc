"""Physics-module protocols: the two roles a module can play in a simulation.

A simulation couples an ensemble of classical point atoms to a list of
physics modules. Each module is one of:

- `ConservativeForce` — a potential `U(r, t)` in joules exerting
  `F = -grad U`. Optical tweezers, magnetic (Zeeman) potentials, and optical
  dipole traps are all conservative forces. The driver sums them for the
  velocity-Verlet integration, and the energy-based loss criterion is defined
  only when every module is conservative.
- `StochasticProcess` — per-step, possibly random, non-conservative physics
  (photon scattering, future: background-gas collisions). A process carries
  opaque per-atom state between steps and returns velocity kicks plus named
  per-atom diagnostics that the driver accumulates into the result.

`total_potential`, `total_force`, `total_hessian` accept one force or any
iterable of forces and linearly sum the results at a chosen time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConservativeForce(ABC):
    """Abstract conservative potential `U(r, t)` [J] with `F = -grad U` [N].

    Subclasses must provide `potential` and `center_at`. Default `force` and
    `hessian` use central finite differences; subclasses override them when
    they have analytic gradients.
    """

    name: str = "force"

    @abstractmethod
    def center_at(self, time_s: float) -> NDArray[np.float64]:
        """Return the natural anchor point (e.g. the potential minimum) at `t`."""

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
class DiagnosticSpec:
    """Declares one per-atom diagnostic channel a `StochasticProcess` emits.

    The driver allocates a full-ensemble array per spec (initialized to
    `fill`) and folds the per-step values in according to `reduce`:
    `"sum"` accumulates over steps, `"last"` keeps the latest value.
    """

    key: str
    reduce: Literal["sum", "last"] = "sum"
    dtype: Any = np.float64
    fill: float = 0.0


@dataclass(frozen=True)
class StochasticStepResult:
    """What one `StochasticProcess.step` call returns for the active atoms.

    `state` must have the same leading axis length as the state rows passed
    in; `velocity_kick_m_per_s` is an `(n, 3)` impulse (or `None` for no
    kick); `diagnostics` maps declared spec keys to `(n,)` per-atom values.
    """

    state: NDArray
    velocity_kick_m_per_s: NDArray[np.float64] | None = None
    diagnostics: Mapping[str, NDArray] = field(default_factory=dict)


class StochasticProcess(ABC):
    """Per-step non-conservative physics with opaque per-atom state.

    The state contract: `initialize` returns any ndarray whose *leading* axis
    is `n_atoms` (any dtype, any trailing shape). The driver never interprets
    it — it only slices active rows (`state[active]`) into `step` and writes
    the returned rows back, so lost atoms stop evolving. A richer backend
    (e.g. multilevel populations) can use an `(n_atoms, k)` state without any
    driver change.
    """

    name: str = "process"

    def diagnostics_spec(self) -> tuple[DiagnosticSpec, ...]:
        """Diagnostic channels this process emits (default: none)."""

        return ()

    @abstractmethod
    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray:
        """Return the initial per-atom state; leading axis is `n_atoms`."""

    @abstractmethod
    def step(
        self,
        state: NDArray,
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        time_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> StochasticStepResult:
        """Advance the active atoms' state by `dt_s` and report kicks/events.

        `positions_m` and `velocities_m_per_s` are `(n, 3)` rows of the
        currently active (not-lost) atoms only, matching `state`'s rows.
        """


def total_potential(
    forces: ConservativeForce | Iterable[ConservativeForce],
    positions_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum potentials of one or more conservative forces at `time_s`."""

    positions = _as_positions(positions_m)
    out = np.zeros(positions.shape[:-1], dtype=float)
    for force in _force_list(forces):
        out = out + force.potential(positions, time_s=time_s)
    return out


def total_force(
    forces: ConservativeForce | Iterable[ConservativeForce],
    positions_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum forces of one or more conservative forces at `time_s`."""

    positions = _as_positions(positions_m)
    out = np.zeros_like(positions, dtype=float)
    for force in _force_list(forces):
        out = out + force.force(positions, time_s=time_s)
    return out


def total_hessian(
    forces: ConservativeForce | Iterable[ConservativeForce],
    position_m: ArrayLike,
    time_s: float = 0.0,
) -> NDArray[np.float64]:
    """Sum Hessians of one or more conservative forces at one position."""

    out = np.zeros((3, 3), dtype=float)
    for force in _force_list(forces):
        out = out + force.hessian(position_m, time_s=time_s)
    return out


def _force_list(
    forces: ConservativeForce | Iterable[ConservativeForce],
) -> list[ConservativeForce]:
    if isinstance(forces, ConservativeForce):
        return [forces]
    return list(forces)


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
