"""Near-resonant light-matter geometry: beams + magnetic fields -> rates.

`LightMatterSystem` is the general description of an atom interacting
with near-resonant light in a (possibly inhomogeneous) magnetic field.
It is not tied to any particular configuration: a MOT, an optical
molasses, a single probe/imaging beam, a Zeeman-slower-like beam, or
resonant light layered on top of conservative dipole traps are all just
different `LaserBeam` / `MagneticFieldConfig` sets.

Its single job is geometry reduction: collapse beam profiles, Doppler
shifts, Zeeman shifts, and polarization projections into the per-atom,
per-beam one-way stimulated rate matrix `W` (s^-1) consumed by the
internal-state backends in `physics.internal_state`:

- stimulated absorption from beam `b` occurs at rate `W_b * rho_gg`,
- stimulated emission into beam `b` at rate `W_b * rho_ee`,
- spontaneous emission at rate `Gamma * rho_ee`.

Zeeman/polarization model (effective two-level): for each beam, the
circular polarization content (`LaserBeam.helicity`) is decomposed
incoherently into sigma+, pi, sigma- components relative to the local
magnetic field direction. A component `q` in `(+1, 0, -1)` sees the
detuning

    delta_q = 2 pi detuning_hz - k . v - q * mu_eff * |B| / hbar,

i.e. laser detuning, Doppler shift, and the Zeeman shift of the
stretched cycling transition (`AtomSpecies.mu_eff_j_per_t`). The one-way
stimulated rate of beam `b` is

    W_b = (Gamma / 2) * sum_q s_b(r) f_q / (1 + (2 delta_q / Gamma)^2),

and saturation competition between beams is handled by the internal
state backend through `p = W_tot / (Gamma + 2 W_tot)`. Beams are
mutually incoherent: no standing-wave or polarization-gradient
(sub-Doppler) physics, and the dipole force of the near-resonant light
is not included (use `DipoleBeamPotential` for conservative dipole potentials).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import HBAR_J_S
from ..geometry.fields import MagneticFieldConfig, total_magnetic_field
from .internal_state import steady_state_excited_fraction
from ..geometry.laser import LaserBeam
from ..species import AtomSpecies


@dataclass(frozen=True)
class LightMatterSystem:
    """Species + laser beams + magnetic fields, with the rate engine.

    `species` may be left as `None` ("unbound"); dropping it into an
    `AtomSystem` injects the system species (see `bind_species`). The
    rate engine (`stimulated_rates`, `mean_radiation_force`) requires a
    bound species and raises a clear error otherwise.
    """

    species: Optional[AtomSpecies] = None
    beams: Sequence[LaserBeam] = ()
    magnetic_fields: Sequence[MagneticFieldConfig] = ()

    def __post_init__(self) -> None:
        beams = tuple(self.beams)
        if not beams:
            raise ValueError("LightMatterSystem requires at least one laser beam.")
        fields = (
            (self.magnetic_fields,)
            if isinstance(self.magnetic_fields, MagneticFieldConfig)
            else tuple(self.magnetic_fields)
        )
        object.__setattr__(self, "beams", beams)
        object.__setattr__(self, "magnetic_fields", fields)

        # Cache the per-beam scalars/vectors as stacked arrays so the
        # per-step rate engine can vectorize over beams instead of looping
        # in Python (the beams are immutable, so this is safe to precompute).
        object.__setattr__(
            self,
            "_beam_directions",
            np.stack([np.asarray(b.direction, dtype=float) for b in beams]),
        )
        object.__setattr__(
            self,
            "_beam_detunings_hz",
            np.array([b.detuning_hz for b in beams], dtype=float),
        )
        object.__setattr__(
            self,
            "_beam_helicities",
            np.array([b.helicity for b in beams], dtype=float),
        )

    @property
    def beam_directions(self) -> NDArray[np.float64]:
        """Unit propagation vectors, shape `(n_beams, 3)`."""

        return self._beam_directions

    @property
    def bound_species(self) -> AtomSpecies:
        """The species, or a clear error if this system is still unbound."""

        if self.species is None:
            raise ValueError(
                "LightMatterSystem has no species. Pass species=... when "
                "building it, or place it in an AtomSystem, which injects "
                "the system species."
            )
        return self.species

    def bind_species(self, species: AtomSpecies) -> "LightMatterSystem":
        """Return a copy bound to `species` (idempotent; verifies matches).

        Leaves an already-bound system unchanged when the species agrees,
        and raises when it conflicts, so the `AtomSystem` species stays the
        single source of truth.
        """

        if self.species is not None:
            if self.species.mass_kg != species.mass_kg:
                raise ValueError(
                    f"LightMatterSystem is built for {self.species.name} but "
                    f"the system species is {species.name}."
                )
            return self
        return replace(self, species=species)

    def magnetic_field_at(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        positions = _as_positions(positions_m)
        if not self.magnetic_fields:
            return np.zeros_like(positions)
        return total_magnetic_field(self.magnetic_fields, positions, time_s=time_s)

    def stimulated_rates(
        self,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """One-way stimulated rate matrix `W`, shape `(N, n_beams)`, in s^-1."""

        positions = _as_positions(positions_m)
        velocities = np.asarray(velocities_m_per_s, dtype=float)
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match the shape of positions.")

        species = self.bound_species
        gamma = species.linewidth_rad_s
        k = species.wavenumber_rad_per_m
        mu_over_hbar = species.mu_eff_j_per_t / HBAR_J_S

        b_vec = self.magnetic_field_at(positions, time_s=time_s)
        b_mag = np.linalg.norm(b_vec, axis=-1)
        # Where B ~ 0 the quantization axis is arbitrary and the Zeeman
        # shift vanishes for every component, so any unit vector works.
        safe = b_mag > 1.0e-15
        b_hat = np.where(
            safe[..., np.newaxis],
            b_vec / np.maximum(b_mag, 1.0e-300)[..., np.newaxis],
            np.array([0.0, 0.0, 1.0]),
        )
        zeeman_rad_s = mu_over_hbar * b_mag  # shape (N,)

        # Vectorize over beams: everything below is shape (N, n_beams).
        # Only the transverse-profile saturation stays a per-beam call,
        # since each beam's `profile` is an arbitrary Python callable.
        directions = self._beam_directions  # (n_beams, 3)
        s_local = np.stack(
            [beam.saturation_at(positions, time_s=time_s) for beam in self.beams],
            axis=-1,
        )
        cos_theta = b_hat @ directions.T  # (N, n_beams)
        doppler_rad_s = k * (velocities @ directions.T)
        base_detuning = (
            2.0 * np.pi * self._beam_detunings_hz - doppler_rad_s
        )  # broadcast (n_beams,) over (N, n_beams)

        f_plus, f_pi, f_minus = polarization_fractions(
            self._beam_helicities, cos_theta
        )
        zeeman = zeeman_rad_s[..., np.newaxis]  # (N, 1)
        w_beam = np.zeros_like(s_local)
        for q, fraction in ((+1.0, f_plus), (0.0, f_pi), (-1.0, f_minus)):
            delta_q = base_detuning - q * zeeman
            lorentz = 1.0 / (1.0 + (2.0 * delta_q / gamma) ** 2)
            w_beam += s_local * fraction * lorentz
        return 0.5 * gamma * w_beam

    def mean_radiation_force(
        self,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Steady-state mean radiation-pressure force, shape `(N, 3)`, newtons.

        `F = hbar k sum_b k_hat_b W_b (1 - 2 p)` with the adiabatic
        steady-state population. Deterministic; useful for computing
        damping/spring coefficients and for tests. Recoil diffusion is
        not included here — it emerges from the stochastic kicks in the
        full simulation.
        """

        species = self.bound_species
        w = self.stimulated_rates(positions_m, velocities_m_per_s, time_s=time_s)
        w_tot = np.sum(w, axis=-1)
        p = steady_state_excited_fraction(w_tot, species.linewidth_rad_s)
        net_rates = w * (1.0 - 2.0 * p)[..., np.newaxis]
        hbar_k = HBAR_J_S * species.wavenumber_rad_per_m
        return hbar_k * (net_rates @ self.beam_directions)


def polarization_fractions(
    helicity: float, cos_theta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Intensity fractions `(f_+1, f_0, f_-1)` relative to the B axis.

    Incoherent mixture of the two circular components with weights
    `(1 +/- h)/2`; a pure circular beam at angle `theta` to the field
    axis has `f_{+/-} = (1 +/- cos)^2 / 4`, `f_pi = sin^2 / 2`.
    """

    c = np.asarray(cos_theta, dtype=float)
    w_plus = 0.5 * (1.0 + helicity)
    w_minus = 0.5 * (1.0 - helicity)
    plus_sq = 0.25 * (1.0 + c) ** 2
    minus_sq = 0.25 * (1.0 - c) ** 2
    f_sigma_plus = w_plus * plus_sq + w_minus * minus_sq
    f_sigma_minus = w_plus * minus_sq + w_minus * plus_sq
    f_pi = 0.5 * (1.0 - c * c)
    return f_sigma_plus, f_pi, f_sigma_minus


def _as_positions(positions_m: ArrayLike) -> NDArray[np.float64]:
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape[-1:] != (3,):
        raise ValueError("positions must have final dimension 3.")
    return positions
