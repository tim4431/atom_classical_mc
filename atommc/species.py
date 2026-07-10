"""Atomic species and optical transition data for light-force simulations.

`AtomSpecies` bundles the mass and the (effective two-level) cycling
transition parameters needed by the MOT / radiation-pressure model:
wavelength, natural linewidth, saturation intensity, and the effective
magnetic moment of the cycling transition.

The Zeeman model used by the rate engine is the standard textbook effective
two-level treatment: a `q = +1, 0, -1` polarization component driving
`m -> m + q` sees a Zeeman detuning `q * mu_eff * B / hbar`, where
`mu_eff = (g_e * m_e - g_g * m_g) * mu_B` evaluated on the stretched
cycling transition (`m_g = F_g`, `m_e = F_g + 1`). For both Rb85
(F=3 -> F'=4) and Rb87 (F=2 -> F'=3) on the D2 line this evaluates to
exactly one Bohr magneton.

Presets `RB85_D2` and `RB87_D2` use values from Steck's alkali data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    BOHR_MAGNETON_J_PER_T,
    HBAR_J_S,
    RB85_MASS_KG,
    RB87_MASS_KG,
)


@dataclass(frozen=True)
class AtomSpecies:
    """Mass plus effective two-level cycling-transition data.

    - `wavelength_m`: transition wavelength (vacuum).
    - `linewidth_rad_s`: natural linewidth Gamma (angular, rad/s).
    - `saturation_intensity_w_per_m2`: I_sat for the cycling transition
      with circularly polarized light (sigma+/- stretched).
    - `g_ground` / `g_excited`: Lande g_F factors of the cooling ground
      and excited hyperfine levels.
    - `f_ground` / `f_excited`: hyperfine quantum numbers of the cycling
      transition (used to compute the stretched-state effective moment).
    """

    name: str
    mass_kg: float
    wavelength_m: float
    linewidth_rad_s: float
    saturation_intensity_w_per_m2: float
    g_ground: float
    g_excited: float
    f_ground: float
    f_excited: float

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive.")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive.")
        if self.linewidth_rad_s <= 0.0:
            raise ValueError("linewidth_rad_s must be positive.")
        if self.saturation_intensity_w_per_m2 <= 0.0:
            raise ValueError("saturation_intensity_w_per_m2 must be positive.")

    @property
    def wavenumber_rad_per_m(self) -> float:
        """Magnitude of the transition wavevector `k = 2 pi / lambda`."""

        return float(2.0 * np.pi / self.wavelength_m)

    @property
    def recoil_velocity_m_per_s(self) -> float:
        """Single-photon recoil velocity `hbar k / m`."""

        return float(HBAR_J_S * self.wavenumber_rad_per_m / self.mass_kg)

    @property
    def mu_eff_j_per_t(self) -> float:
        """Effective magnetic moment of the stretched cycling transition.

        `mu_eff = (g_e * (F_g + 1) - g_g * F_g) * mu_B`, the Zeeman shift
        per unit field of the `m = F_g -> m' = F_g + 1` transition.
        """

        return float(
            (self.g_excited * (self.f_ground + 1.0) - self.g_ground * self.f_ground)
            * BOHR_MAGNETON_J_PER_T
        )

    @property
    def doppler_temperature_uK(self) -> float:
        """Doppler cooling limit `T_D = hbar Gamma / (2 k_B)` in microkelvin."""

        from .constants import BOLTZMANN_CONSTANT_J_PER_K

        return float(
            HBAR_J_S
            * self.linewidth_rad_s
            / (2.0 * BOLTZMANN_CONSTANT_J_PER_K)
            / 1.0e-6
        )


# Rb85 D2 cooling transition: 5S1/2 F=3 -> 5P3/2 F'=4 (Steck, Rb85 D line data).
RB85_D2 = AtomSpecies(
    name="Rb85 D2",
    mass_kg=RB85_MASS_KG,
    wavelength_m=780.241368271e-9,
    linewidth_rad_s=2.0 * np.pi * 6.0666e6,
    saturation_intensity_w_per_m2=16.6933,  # 1.66933 mW/cm^2
    g_ground=1.0 / 3.0,
    g_excited=1.0 / 2.0,
    f_ground=3.0,
    f_excited=4.0,
)

# Rb87 D2 cooling transition: 5S1/2 F=2 -> 5P3/2 F'=3 (Steck, Rb87 D line data).
RB87_D2 = AtomSpecies(
    name="Rb87 D2",
    mass_kg=RB87_MASS_KG,
    wavelength_m=780.241209686e-9,
    linewidth_rad_s=2.0 * np.pi * 6.0666e6,
    saturation_intensity_w_per_m2=16.6933,  # 1.66933 mW/cm^2
    g_ground=1.0 / 2.0,
    g_excited=2.0 / 3.0,
    f_ground=2.0,
    f_excited=3.0,
)
