"""Zeeman potential: the magnetic force on an atom in a fixed sublevel.

`ZeemanPotential` is a `ConservativeForce` with `U(r, t) = mu * |B(r, t)|`,
where `mu = m_F g_F mu_B` is the (signed) magnetic moment of the occupied
Zeeman sublevel. `mu > 0` is weak-field seeking (trapped at a field
minimum, e.g. a quadrupole node); `mu < 0` is strong-field seeking
(anti-trapped / Stern-Gerlach expulsion).

Model assumptions:

- The atom follows the local field direction adiabatically and stays in one
  sublevel; Majorana spin flips near field zeros are NOT modeled.
- Weak-field (linear) Zeeman regime.
- `|B|` is non-differentiable at a field zero. The force is set to exactly
  zero wherever `|B| < zero_field_threshold_t`; near a quadrupole node the
  potential is linear (conical), not harmonic, so Hessian-based tools
  (`HarmonicTrapCloud`, `approximate_harmonic_potential`) must not be
  centered there.

The module owns its full field list because magnitudes do not add:
`|B_1 + B_2| != |B_1| + |B_2|`. Share the same `MagneticFieldConfig`
instances with a `LightMatterSystem` to keep one coil geometry consistent
between Zeeman forces and scattering-rate Zeeman shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import BOHR_MAGNETON_J_PER_T
from ..geometry.fields import (
    MagneticFieldConfig,
    total_magnetic_field,
    total_magnetic_field_jacobian,
)
from .base import ConservativeForce, _as_positions


@dataclass(frozen=True)
class ZeemanPotential(ConservativeForce):
    """Conservative potential `U = moment_j_per_t * |B_total(r, t)|`.

    `moment_j_per_t` is the signed sublevel moment `m_F g_F mu_B`; use
    `ZeemanPotential.for_sublevel` to build it from quantum numbers.
    """

    fields: Sequence[MagneticFieldConfig] = ()
    moment_j_per_t: float = BOHR_MAGNETON_J_PER_T
    name: str = "zeeman"
    zero_field_threshold_t: float = 1.0e-15

    def __post_init__(self) -> None:
        if isinstance(self.fields, MagneticFieldConfig):
            object.__setattr__(self, "fields", (self.fields,))
        else:
            fields = tuple(self.fields)
            if not fields:
                raise ValueError("ZeemanPotential requires at least one field.")
            for field in fields:
                if not isinstance(field, MagneticFieldConfig):
                    raise TypeError(
                        "fields must contain MagneticFieldConfig instances, "
                        f"got {type(field).__name__}."
                    )
            object.__setattr__(self, "fields", fields)

    @classmethod
    def for_sublevel(
        cls,
        fields: MagneticFieldConfig | Sequence[MagneticFieldConfig],
        g_f: float,
        m_f: float,
        name: str = "zeeman",
    ) -> "ZeemanPotential":
        """Build from the sublevel quantum numbers: `moment = g_F m_F mu_B`."""

        return cls(
            fields=fields,
            moment_j_per_t=g_f * m_f * BOHR_MAGNETON_J_PER_T,
            name=name,
        )

    def center_at(self, time_s: float) -> NDArray[np.float64]:
        for field in self.fields:
            center = getattr(field, "center_m", None)
            if center is not None:
                return np.asarray(center, dtype=float)
        return np.zeros(3)

    def potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        b_vec = total_magnetic_field(self.fields, positions, time_s=time_s)
        return self.moment_j_per_t * np.linalg.norm(b_vec, axis=-1)

    def force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """`F = -mu grad|B| = -mu J^T B_hat`, zero at (near-)zero field."""

        positions = _as_positions(positions_m)
        b_vec = total_magnetic_field(self.fields, positions, time_s=time_s)
        b_mag = np.linalg.norm(b_vec, axis=-1)
        jac = total_magnetic_field_jacobian(self.fields, positions, time_s=time_s)
        safe_mag = np.where(b_mag > self.zero_field_threshold_t, b_mag, 1.0)
        b_hat = b_vec / safe_mag[..., None]
        # grad|B|_j = B_hat_i dB_i/dr_j
        grad_mag = np.einsum("...ij,...i->...j", jac, b_hat)
        force = -self.moment_j_per_t * grad_mag
        return np.where(
            (b_mag > self.zero_field_threshold_t)[..., None], force, 0.0
        )

    def _finite_difference_step(self) -> float:
        return 1.0e-8
