"""m_F-resolved hyperfine internal-state dynamics for photon scattering.

`HyperfineSpecies` extends an effective two-level `AtomSpecies` with the
full hyperfine sublevel structure of a D-line: every `|F, m_F>` ground
and excited sublevel, its hyperfine energy offset and linear Zeeman
shift, the relative excitation strength of every dipole-allowed
`|F, m_F> -> |F', m_F'>` transition, and the spontaneous branching
ratios. All angular factors are computed exactly from Wigner 3-j / 6-j
algebra (`physics.wigner`); only the measured hyperfine A/B constants
are inputs (Steck alkali data, cross-validated against the ARC package
in `tests/test_hyperfine.py`).

`HyperfineScattering` is the `StochasticProcess` using that structure.
Where `LightScattering` collapses each beam's sigma+/pi/sigma- content
into one scalar rate for an effective two-level atom,
`HyperfineScattering` keeps a rate per dipole-allowed transition:

    W_t,b = (Gamma/2) * s_b(r) * f_q(t)(b, r) * S_t
            / (1 + (2 delta_t,b / Gamma)^2),

    delta_t,b = 2 pi detuning_hz_b - k.v - (hyperfine offset of t)
                - (m_e g_e - m_g g_g) mu_B |B| / hbar,

with `S_t` the transition strength normalized to 1 on the stretched
cycling transition (so `LaserBeam.saturation` keeps its meaning: I over
the stretched-transition I_sat). Beam detunings stay referenced to the
cycling line; reference a beam to another line (e.g. a repumper on
F=2 -> F'=3) by adding `HyperfineSpecies.transition_offset_hz(f_g, f_e)`
to its `detuning_hz`.

Populations `p` over all sublevels evolve per atom under the rate-
equation generator (absorption, stimulated emission, spontaneous decay
with branching), integrated with one implicit-Euler step per timestep:
unconditionally stable at any `Gamma * dt`, exactly trace- and
positivity-preserving, with exact steady states. Fast (saturated) modes
relax to their quasi-steady values within a step; slow optical-pumping
transients are time-resolved and accurate while
`(pumping rate) * dt < ~0.3`. This makes dark ground states a real
dynamic: off-resonant excitation to lower F' levels leaks population
out of the cooling cycle unless a repump beam is present.

Still out of scope: Zeeman shifts are linear per sublevel (no
Paschen-Back regime), the nuclear g-factor term is neglected (~0.1%),
beams remain mutually incoherent (no coherences/dark superpositions, no
sub-Doppler mechanisms), and decay is a single fine-structure line
(D2: the excited J' = 3/2 manifold decays only to the ground J = 1/2).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..constants import BOHR_MAGNETON_J_PER_T, HBAR_J_S
from ..species import RB85_D2, RB87_D2, AtomSpecies
from .base import DiagnosticSpec, StochasticProcess, StochasticStepResult
from .internal_state import ScatteringEvents, sample_recoil_velocity_kicks
from .light_matter import LightMatterSystem, polarization_fractions
from .wigner import wigner_3j, wigner_6j

_ELECTRON_G_S = 2.00231930436256
_STRENGTH_CUTOFF = 1.0e-12


@dataclass(frozen=True)
class HyperfineSpecies:
    """Hyperfine sublevel structure layered on an `AtomSpecies`.

    - `base`: the effective two-level species (mass, wavelength, Gamma,
      stretched-transition I_sat) — shared with the rest of the library.
    - `i_nuclear`: nuclear spin `I`.
    - `hfs_a_*_hz` / `hfs_b_*_hz`: magnetic-dipole / electric-quadrupole
      hyperfine constants of the ground and excited fine-structure
      levels, in linear Hz (Steck convention).
    - `l_*` / `j_*`: fine-structure quantum numbers (defaults: a D2
      line, S1/2 -> P3/2).
    - `g_factor_ground_by_f` / `g_factor_excited_by_f`: optional
      per-hyperfine-level Lande g_F overrides (`{F: g_F}`). When absent,
      g_F comes from the exact electronic formula with the nuclear term
      neglected (~0.1%); `hyperfine_species_from_arc` fills these with
      ARC's values, which include it.

    Derived tables (built once, exposed as attributes): per-sublevel
    quantum numbers, hyperfine offsets [rad/s] relative to the cycling
    levels, linear Zeeman rates [rad/s per tesla], the dipole-allowed
    transition list with strengths normalized to the stretched cycling
    transition, and the spontaneous branching matrix.
    """

    base: AtomSpecies
    i_nuclear: float
    hfs_a_ground_hz: float
    hfs_a_excited_hz: float
    hfs_b_ground_hz: float = 0.0
    hfs_b_excited_hz: float = 0.0
    l_ground: int = 0
    j_ground: float = 0.5
    l_excited: int = 1
    j_excited: float = 1.5
    g_factor_ground_by_f: Mapping[float, float] | None = None
    g_factor_excited_by_f: Mapping[float, float] | None = None

    def __post_init__(self) -> None:
        if self.i_nuclear < 0 or round(2 * self.i_nuclear) != 2 * self.i_nuclear:
            raise ValueError("i_nuclear must be a non-negative half-integer.")
        f_cool = self.i_nuclear + self.j_ground
        f_cycle = self.i_nuclear + self.j_excited
        if self.base.f_ground != f_cool or self.base.f_excited != f_cycle:
            raise ValueError(
                "base species cycling levels must be the stretched hyperfine "
                f"levels F = I + J: expected F = {f_cool} -> F' = {f_cycle}, "
                f"got {self.base.f_ground} -> {self.base.f_excited}."
            )

        ground = _build_manifold(
            self.i_nuclear,
            self.l_ground,
            self.j_ground,
            self.hfs_a_ground_hz,
            self.hfs_b_ground_hz,
            self.g_factor_ground_by_f,
        )
        excited = _build_manifold(
            self.i_nuclear,
            self.l_excited,
            self.j_excited,
            self.hfs_a_excited_hz,
            self.hfs_b_excited_hz,
            self.g_factor_excited_by_f,
        )
        for key, value in ground.items():
            object.__setattr__(self, f"ground_{key}", value)
        for key, value in excited.items():
            object.__setattr__(self, f"excited_{key}", value)

        self._build_transitions()

    def _build_transitions(self) -> None:
        f_g, m_g = self.ground_f, self.ground_m
        f_e, m_e = self.excited_f, self.excited_m
        n_g, n_e = f_g.size, f_e.size

        # Squared dipole amplitudes up to a common reduced-matrix-element
        # factor: (2Fg+1)(2Fe+1) {Jg Je 1; Fe Fg I}^2 (Fe 1 Fg; me q -mg)^2.
        amp_sq = np.zeros((n_g, n_e))
        for g in range(n_g):
            for e in range(n_e):
                q = m_e[e] - m_g[g]
                if abs(q) > 1:
                    continue
                six = wigner_6j(
                    self.j_ground, self.j_excited, 1.0,
                    f_e[e], f_g[g], self.i_nuclear,
                )
                # The 3-j projection must close (m_e - q' - m_g = 0), so its
                # middle argument is -q; the polarization label stays q.
                three = wigner_3j(f_e[e], 1.0, f_g[g], m_e[e], -q, -m_g[g])
                amp_sq[g, e] = (
                    (2.0 * f_g[g] + 1.0)
                    * (2.0 * f_e[e] + 1.0)
                    * six**2
                    * three**2
                )

        # Every excited sublevel must decay at the same total rate Gamma;
        # its unnormalized amplitude sum is that rate up to a constant.
        row_sums = amp_sq.sum(axis=0)
        if not np.allclose(row_sums, row_sums[0], rtol=1.0e-10):
            raise AssertionError(
                "hyperfine table inconsistency: excited-state decay sums differ."
            )
        branching = (amp_sq / row_sums[np.newaxis, :]).T  # (n_e, n_g)

        g_stretch = int(np.argmax((f_g == f_g.max()) & (m_g == f_g.max())))
        e_stretch = int(np.argmax((f_e == f_e.max()) & (m_e == f_g.max() + 1)))
        strength = amp_sq / amp_sq[g_stretch, e_stretch]

        t_g, t_e = np.nonzero(strength > _STRENGTH_CUTOFF)
        object.__setattr__(self, "branching_ratios", branching)
        object.__setattr__(self, "transition_ground_index", t_g)
        object.__setattr__(self, "transition_excited_index", t_e)
        object.__setattr__(
            self, "transition_q", np.rint(m_e[t_e] - m_g[t_g]).astype(int)
        )
        object.__setattr__(self, "transition_strength", strength[t_g, t_e])
        object.__setattr__(self, "transition_branching", branching[t_e, t_g])
        object.__setattr__(
            self,
            "transition_offset_rad_s",
            self.excited_offset_rad_s[t_e] - self.ground_offset_rad_s[t_g],
        )
        object.__setattr__(
            self,
            "transition_zeeman_rad_s_per_t",
            self.excited_zeeman_rad_s_per_t[t_e]
            - self.ground_zeeman_rad_s_per_t[t_g],
        )

    @property
    def n_ground(self) -> int:
        return int(self.ground_f.size)

    @property
    def n_excited(self) -> int:
        return int(self.excited_f.size)

    @property
    def n_levels(self) -> int:
        return self.n_ground + self.n_excited

    @property
    def n_transitions(self) -> int:
        return int(self.transition_ground_index.size)

    def ground_level_index(self, f: float, m: float) -> int:
        """Index of ground sublevel `|F=f, m_F=m>` in the population vector."""

        return _level_index(self.ground_f, self.ground_m, f, m)

    def excited_level_index(self, f: float, m: float) -> int:
        """Index of excited `|F'=f, m_F=m>` within the excited manifold."""

        return _level_index(self.excited_f, self.excited_m, f, m)

    def transition_offset_hz(self, f_ground: float, f_excited: float) -> float:
        """Line-center offset of `F=f_ground -> F'=f_excited` vs the cycling line.

        Add this to a `LaserBeam.detuning_hz` to reference that beam's
        detuning to the given transition (e.g. a repumper): a beam with
        `detuning_hz = delta + transition_offset_hz(2, 3)` is detuned by
        `delta` from `F=2 -> F'=3`.
        """

        g = _first_index(self.ground_f, f_ground)
        e = _first_index(self.excited_f, f_excited)
        return float(
            (self.excited_offset_rad_s[e] - self.ground_offset_rad_s[g])
            / (2.0 * np.pi)
        )


def _build_manifold(
    i_nuclear: float,
    l: int,
    j: float,
    a_hz: float,
    b_hz: float,
    g_by_f: Mapping[float, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Per-sublevel tables of one fine-structure level's hyperfine manifold."""

    f_min, f_max = abs(j - i_nuclear), j + i_nuclear
    f_levels = [f_min + k for k in range(int(round(f_max - f_min)) + 1)]

    shifts = {f: _hfs_shift_hz(f, i_nuclear, j, a_hz, b_hz) for f in f_levels}
    g_j = _lande_g_j(l, j)
    if g_by_f is not None:
        missing = [f for f in f_levels if f not in g_by_f]
        if missing:
            raise ValueError(f"g-factor override is missing levels F = {missing}.")

    f_list, m_list, g_list, off_list = [], [], [], []
    for f in f_levels:
        if g_by_f is not None:
            g_f = float(g_by_f[f])
        else:
            g_f = 0.0 if f == 0 else _lande_g_f(g_j, f, i_nuclear, j)
        for k in range(int(round(2 * f)) + 1):
            m = -f + k
            f_list.append(f)
            m_list.append(m)
            g_list.append(g_f)
            off_list.append(2.0 * np.pi * (shifts[f] - shifts[f_max]))
    g_arr = np.array(g_list)
    m_arr = np.array(m_list)
    return {
        "f": np.array(f_list),
        "m": m_arr,
        "g_factor": g_arr,
        "offset_rad_s": np.array(off_list),
        "zeeman_rad_s_per_t": m_arr * g_arr * BOHR_MAGNETON_J_PER_T / HBAR_J_S,
    }


def _hfs_shift_hz(f: float, i: float, j: float, a_hz: float, b_hz: float) -> float:
    """Hyperfine shift `Delta E_hfs / h` of level F (standard A/B formula)."""

    k = f * (f + 1.0) - i * (i + 1.0) - j * (j + 1.0)
    shift = 0.5 * a_hz * k
    if b_hz != 0.0:
        if i <= 0.5 or j <= 0.5:
            raise ValueError("quadrupole B constant requires I > 1/2 and J > 1/2.")
        shift += b_hz * (
            (1.5 * k * (k + 1.0) - 2.0 * i * (i + 1.0) * j * (j + 1.0))
            / (2.0 * i * (2.0 * i - 1.0) * 2.0 * j * (2.0 * j - 1.0))
        )
    return shift


def _lande_g_j(l: int, j: float, s: float = 0.5) -> float:
    """Lande g_J with g_L = 1, g_S = 2.0023 (nuclear term neglected)."""

    jj, ll, ss = j * (j + 1.0), l * (l + 1.0), s * (s + 1.0)
    return (jj - ss + ll) / (2.0 * jj) + _ELECTRON_G_S * (jj + ss - ll) / (2.0 * jj)


def _lande_g_f(g_j: float, f: float, i: float, j: float) -> float:
    return g_j * (f * (f + 1.0) - i * (i + 1.0) + j * (j + 1.0)) / (
        2.0 * f * (f + 1.0)
    )


def _level_index(
    f_arr: NDArray[np.float64], m_arr: NDArray[np.float64], f: float, m: float
) -> int:
    match = np.flatnonzero((f_arr == f) & (m_arr == m))
    if match.size != 1:
        raise ValueError(f"no sublevel |F={f}, m_F={m}> in this manifold.")
    return int(match[0])


def _first_index(f_arr: NDArray[np.float64], f: float) -> int:
    match = np.flatnonzero(f_arr == f)
    if match.size == 0:
        raise ValueError(f"no hyperfine level F={f} in this manifold.")
    return int(match[0])


# Hyperfine constants from Steck's alkali D-line data (as in ARC).
RB85_D2_HFS = HyperfineSpecies(
    base=RB85_D2,
    i_nuclear=2.5,
    hfs_a_ground_hz=1.0119108130e9,
    hfs_a_excited_hz=25.0020e6,
    hfs_b_excited_hz=25.790e6,
)

RB87_D2_HFS = HyperfineSpecies(
    base=RB87_D2,
    i_nuclear=1.5,
    hfs_a_ground_hz=3.417341305e9,
    hfs_a_excited_hz=84.7185e6,
    hfs_b_excited_hz=12.4965e6,
)


_ARC_ISOTOPES = {
    "Li6": "Lithium6",
    "Li7": "Lithium7",
    "Na23": "Sodium",
    "K39": "Potassium39",
    "K40": "Potassium40",
    "K41": "Potassium41",
    "Rb85": "Rubidium85",
    "Rb87": "Rubidium87",
    "Cs133": "Caesium",
}


@lru_cache(maxsize=None)
def hyperfine_species_from_arc(isotope: str = "Rb85") -> HyperfineSpecies:
    """Build a D2-line `HyperfineSpecies` with every constant from ARC.

    Fetches the mass, transition wavelength, linewidth, stretched-line
    saturation intensity, hyperfine A/B coefficients, and per-level
    Lande g_F factors (including the nuclear term the built-in formula
    neglects) from the ARC package (`pip install
    ARC-Alkali-Rydberg-Calculator`) — an optional dependency imported
    lazily here and queried exactly once per isotope (the result is
    cached for the process lifetime; note ARC's own import takes a few
    seconds). The angular structure (strengths, branching) is still
    computed by the exact in-package Wigner algebra, which matches ARC
    to machine precision (`tests/test_hyperfine.py`).

    `isotope` is one of `Li6`, `Li7`, `Na23`, `K39`, `K40`, `K41`,
    `Rb85`, `Rb87`, `Cs133`. The hardcoded `RB85_D2_HFS` / `RB87_D2_HFS`
    presets (Steck constants) remain the ARC-free default path.
    """

    if isotope not in _ARC_ISOTOPES:
        raise ValueError(
            f"unknown isotope {isotope!r}; choose from {sorted(_ARC_ISOTOPES)}."
        )
    try:
        import arc
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "hyperfine_species_from_arc requires the optional ARC package: "
            "pip install ARC-Alkali-Rydberg-Calculator"
        ) from error

    atom = getattr(arc, _ARC_ISOTOPES[isotope])()
    n = atom.groundStateN
    i_nuc = float(atom.I)
    f_ground = i_nuc + 0.5
    f_excited = i_nuc + 1.5

    linewidth_rad_s = float(atom.getTransitionRate(n, 1, 1.5, n, 0, 0.5))
    wavelength_m = abs(float(atom.getTransitionWavelength(n, 0, 0.5, n, 1, 1.5)))
    isat_w_per_m2 = float(
        atom.getSaturationIntensity(
            n, 0, 0.5, f_ground, f_ground, n, 1, 1.5, f_excited, f_ground + 1.0
        )
    )
    a_ground_hz, b_ground_hz = atom.getHFSCoefficients(n, 0, 0.5)
    a_excited_hz, b_excited_hz = atom.getHFSCoefficients(n, 1, 1.5)
    b_ground_hz = 0.0  # undefined for J = 1/2; ARC reports 0

    def g_by_f(l: int, j: float) -> dict[float, float]:
        f_min = abs(j - i_nuc)
        levels = [f_min + k for k in range(int(round(j + i_nuc - f_min)) + 1)]
        return {
            f: 0.0 if f == 0 else float(atom.getLandegfExact(l, j, f))
            for f in levels
        }

    g_ground = g_by_f(0, 0.5)
    g_excited = g_by_f(1, 1.5)

    base = AtomSpecies(
        name=f"{isotope} D2 (ARC)",
        mass_kg=float(atom.mass),
        wavelength_m=wavelength_m,
        linewidth_rad_s=linewidth_rad_s,
        saturation_intensity_w_per_m2=isat_w_per_m2,
        g_ground=g_ground[f_ground],
        g_excited=g_excited[f_excited],
        f_ground=f_ground,
        f_excited=f_excited,
    )
    return HyperfineSpecies(
        base=base,
        i_nuclear=i_nuc,
        hfs_a_ground_hz=float(a_ground_hz),
        hfs_a_excited_hz=float(a_excited_hz),
        hfs_b_ground_hz=float(b_ground_hz),
        hfs_b_excited_hz=float(b_excited_hz),
        g_factor_ground_by_f=g_ground,
        g_factor_excited_by_f=g_excited,
    )


@dataclass(frozen=True)
class HyperfineScattering(StochasticProcess):
    """Photon scattering with m_F-resolved rate-equation populations.

    Drop-in alternative to `LightScattering` built from the same
    `LightMatterSystem` geometry (beams keep their meaning; detunings
    stay referenced to the cycling transition). Per step: resolve the
    per-atom, per-beam, per-transition stimulated rates; advance the
    sublevel populations by one unconditionally stable implicit-Euler
    step of the rate-equation generator; Poisson-sample photon events
    from the mid-step populations; and convert them to recoil kicks.

    `initial_populations` selects the starting distribution:
    `"ground-uniform"` (every ground sublevel equally, i.e. degeneracy-
    weighted over both hyperfine ground levels), `"cooling-uniform"`
    (uniform over the upper cooling level F = I + J), or `"stretched"`
    (all atoms in |F = I + J, m_F = +F>).

    Emits the conventional `scattered_photons` / `excited_fraction`
    diagnostics plus one `ground_f<F>_population` channel per ground
    hyperfine level (`reduce="last"`), so dark-state population is
    directly observable.
    """

    light: LightMatterSystem
    hyperfine: HyperfineSpecies
    initial_populations: str = "ground-uniform"
    name: str = "scattering"

    def __post_init__(self) -> None:
        if not isinstance(self.light, LightMatterSystem):
            raise TypeError("light must be a LightMatterSystem.")
        if not isinstance(self.hyperfine, HyperfineSpecies):
            raise TypeError("hyperfine must be a HyperfineSpecies.")
        # The hyperfine species already fixes the isotope, so an unbound
        # LightMatterSystem simply inherits `hyperfine.base` — the light
        # geometry never needs its own species= for the hyperfine backend.
        if self.light.species is None:
            object.__setattr__(
                self, "light", self.light.bind_species(self.hyperfine.base)
            )
        if self.hyperfine.base != self.light.species:
            raise ValueError(
                "hyperfine.base must be the same species as light.species."
            )
        if self.initial_populations not in (
            "ground-uniform",
            "cooling-uniform",
            "stretched",
        ):
            raise ValueError(
                "initial_populations must be 'ground-uniform', "
                "'cooling-uniform', or 'stretched'."
            )
        hf = self.hyperfine
        onehot_g = np.zeros((hf.n_transitions, hf.n_ground))
        onehot_g[np.arange(hf.n_transitions), hf.transition_ground_index] = 1.0
        onehot_e = np.zeros((hf.n_transitions, hf.n_excited))
        onehot_e[np.arange(hf.n_transitions), hf.transition_excited_index] = 1.0
        object.__setattr__(self, "_onehot_ground", onehot_g)
        object.__setattr__(self, "_onehot_excited", onehot_e)

        beams = self.light.beams
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
        channels = tuple(
            (float(f), f"ground_f{_format_f(f)}_population")
            for f in sorted(set(hf.ground_f.tolist()))
        )
        object.__setattr__(self, "_ground_f_channels", channels)

    @property
    def species(self) -> AtomSpecies:
        return self.light.species

    def bind_species(self, species: AtomSpecies) -> "HyperfineScattering":
        """Return a copy whose `LightMatterSystem` is bound to `species`.

        Raises if `species` disagrees with `hyperfine.base`.
        """

        return replace(self, light=self.light.bind_species(species))

    def diagnostics_spec(self) -> tuple[DiagnosticSpec, ...]:
        specs = [
            DiagnosticSpec("scattered_photons", reduce="sum", dtype=np.int64),
            DiagnosticSpec("excited_fraction", reduce="last"),
        ]
        specs.extend(
            DiagnosticSpec(key, reduce="last") for _, key in self._ground_f_channels
        )
        return tuple(specs)

    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        hf = self.hyperfine
        populations = np.zeros((n_atoms, hf.n_levels), dtype=float)
        if self.initial_populations == "ground-uniform":
            populations[:, : hf.n_ground] = 1.0 / hf.n_ground
        elif self.initial_populations == "cooling-uniform":
            cooling = np.flatnonzero(hf.ground_f == hf.ground_f.max())
            populations[:, cooling] = 1.0 / cooling.size
        else:  # stretched
            f_max = hf.ground_f.max()
            populations[:, hf.ground_level_index(f_max, f_max)] = 1.0
        return populations

    def step(
        self,
        state: NDArray[np.float64],
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        time_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> StochasticStepResult:
        hf = self.hyperfine
        gamma = self.species.linewidth_rad_s
        populations = np.asarray(state, dtype=float)

        w_beam = self.stimulated_rates_resolved(
            positions_m, velocities_m_per_s, time_s=time_s
        )
        w_total = np.sum(w_beam, axis=1)  # (N, n_transitions)
        populations_next = self._advance_populations(
            populations, w_total, gamma, dt_s
        )
        # Photon means use the mid-step populations (trapezoidal average).
        populations_mid = 0.5 * (populations + populations_next)
        events = self._sample_photons(w_beam, populations_mid, gamma, dt_s, rng)
        kicks = sample_recoil_velocity_kicks(
            events,
            self._beam_directions,
            self.species.recoil_velocity_m_per_s,
            rng,
        )

        diagnostics = {
            "scattered_photons": events.total_scattered,
            "excited_fraction": np.sum(populations_next[:, hf.n_ground :], axis=-1),
        }
        for f_value, key in self._ground_f_channels:
            level_idx = np.flatnonzero(hf.ground_f == f_value)
            diagnostics[key] = np.sum(populations_next[:, level_idx], axis=-1)
        return StochasticStepResult(
            state=populations_next,
            velocity_kick_m_per_s=kicks,
            diagnostics=diagnostics,
        )

    def stimulated_rates_resolved(
        self,
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Per-transition rate tensor, shape `(N, n_beams, n_transitions)`.

        The m_F-resolved analog of `LightMatterSystem.stimulated_rates`:
        same geometry reduction (profiles, Doppler, local-field
        polarization decomposition), but each dipole-allowed transition
        keeps its own strength, hyperfine offset, and Zeeman shift.
        """

        hf = self.hyperfine
        light = self.light
        gamma = self.species.linewidth_rad_s
        k = self.species.wavenumber_rad_per_m

        positions = np.asarray(positions_m, dtype=float)
        velocities = np.asarray(velocities_m_per_s, dtype=float)

        b_vec = light.magnetic_field_at(positions, time_s=time_s)
        b_mag = np.linalg.norm(b_vec, axis=-1)
        safe = b_mag > 1.0e-15
        b_hat = np.where(
            safe[..., np.newaxis],
            b_vec / np.maximum(b_mag, 1.0e-300)[..., np.newaxis],
            np.array([0.0, 0.0, 1.0]),
        )

        directions = self._beam_directions
        s_local = np.stack(
            [beam.saturation_at(positions, time_s=time_s) for beam in light.beams],
            axis=-1,
        )
        cos_theta = b_hat @ directions.T
        doppler_rad_s = k * (velocities @ directions.T)
        f_plus, f_pi, f_minus = polarization_fractions(
            self._beam_helicities, cos_theta
        )
        # Index by q + 1: [sigma-, pi, sigma+].
        f_stack = np.stack([f_minus, f_pi, f_plus], axis=-1)
        f_q = f_stack[..., hf.transition_q + 1]  # (N, n_beams, n_t)

        detuning = (
            (2.0 * np.pi * self._beam_detunings_hz - doppler_rad_s)[..., np.newaxis]
            - hf.transition_offset_rad_s
            - hf.transition_zeeman_rad_s_per_t * b_mag[..., np.newaxis, np.newaxis]
        )
        lorentz = 1.0 / (1.0 + (2.0 * detuning / gamma) ** 2)
        return (
            0.5
            * gamma
            * s_local[..., np.newaxis]
            * f_q
            * hf.transition_strength
            * lorentz
        )

    def _advance_populations(
        self,
        populations: NDArray[np.float64],
        w_total: NDArray[np.float64],
        gamma: float,
        dt_s: float,
    ) -> NDArray[np.float64]:
        """One implicit-Euler step of `dp/dt = M p` per atom.

        `M` is the rate-equation generator over all sublevels; its
        columns sum to zero and its off-diagonal entries are
        non-negative, so `(I - dt M)^{-1}` conserves the population sum
        exactly and never produces negative populations.
        """

        hf = self.hyperfine
        n_ground, n_levels = hf.n_ground, hf.n_levels

        # Atoms in the dark with an empty excited manifold are stationary.
        active = (np.sum(w_total, axis=-1) > 0.0) | (
            np.sum(populations[:, n_ground:], axis=-1) > 1.0e-15
        )
        if not np.any(active):
            return populations.copy()

        w = w_total[active]
        rows_e = n_ground + hf.transition_excited_index
        cols_g = hf.transition_ground_index

        generator = np.zeros((w.shape[0], n_levels, n_levels))
        # Each (g, e) pair appears in exactly one transition (q is fixed
        # by m_e - m_g), so plain fancy assignment covers all entries.
        generator[:, rows_e, cols_g] = w
        generator[:, cols_g, rows_e] = w + gamma * hf.transition_branching
        pump_out = w @ self._onehot_ground  # (n, n_ground)
        pump_in = w @ self._onehot_excited  # (n, n_excited)
        diag_idx = np.arange(n_levels)
        generator[:, diag_idx, diag_idx] = -np.concatenate(
            [pump_out, gamma + pump_in], axis=1
        )

        system = -dt_s * generator
        system[:, diag_idx, diag_idx] += 1.0
        solved = np.linalg.solve(system, populations[active][..., np.newaxis])

        out = populations.copy()
        out[active] = solved[..., 0]
        return out

    def _sample_photons(
        self,
        w_beam: NDArray[np.float64],
        populations: NDArray[np.float64],
        gamma: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> ScatteringEvents:
        hf = self.hyperfine
        p_ground_t = populations[:, hf.transition_ground_index]
        p_excited_t = populations[:, hf.n_ground + hf.transition_excited_index]
        absorbed_mean = np.einsum("nbt,nt->nb", w_beam, p_ground_t) * dt_s
        stimulated_mean = np.einsum("nbt,nt->nb", w_beam, p_excited_t) * dt_s
        spontaneous_mean = (
            gamma * np.sum(populations[:, hf.n_ground :], axis=-1) * dt_s
        )
        return ScatteringEvents(
            absorbed_per_beam=rng.poisson(absorbed_mean).astype(np.int64),
            stimulated_per_beam=rng.poisson(stimulated_mean).astype(np.int64),
            spontaneous=rng.poisson(spontaneous_mean).astype(np.int64),
        )


def _format_f(f: float) -> str:
    return str(int(f)) if float(f).is_integer() else f"{f:.1f}".replace(".", "_")
