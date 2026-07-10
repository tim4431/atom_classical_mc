"""Photon-scattering physics module: radiation pressure with recoil noise.

`LightScattering` is the `StochasticProcess` that couples near-resonant
light to the atomic motion. Each step it reduces the field geometry
(`LightMatterSystem.stimulated_rates`) to a per-atom, per-beam rate matrix,
lets an `InternalStateModel` backend evolve the internal state and sample
photon events, and converts the events into recoil velocity kicks.

A MOT, an optical molasses, or a resonant probe on top of tweezers are all
just different `LightMatterSystem` geometries attached through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..species import AtomSpecies
from .base import DiagnosticSpec, StochasticProcess, StochasticStepResult
from .internal_state import (
    AdiabaticSteadyState,
    InternalStateModel,
    sample_recoil_velocity_kicks,
)
from .light_matter import LightMatterSystem


@dataclass(frozen=True)
class LightScattering(StochasticProcess):
    """Stochastic recoil kicks from near-resonant photon scattering.

    `light` bundles the species, laser beams, and magnetic fields; `model`
    selects the internal-state backend (adiabatic steady state by default).
    Emits two per-atom diagnostics: `scattered_photons` (total absorbed
    photons, summed over the run) and `excited_fraction` (latest excited
    population).
    """

    light: LightMatterSystem
    model: InternalStateModel = field(default_factory=AdiabaticSteadyState)
    name: str = "scattering"

    def __post_init__(self) -> None:
        if not isinstance(self.light, LightMatterSystem):
            raise TypeError("light must be a LightMatterSystem.")
        if not isinstance(self.model, InternalStateModel):
            raise TypeError("model must be an InternalStateModel.")

    @property
    def species(self) -> AtomSpecies:
        return self.light.species

    def diagnostics_spec(self) -> tuple[DiagnosticSpec, ...]:
        return (
            DiagnosticSpec("scattered_photons", reduce="sum", dtype=np.int64),
            DiagnosticSpec("excited_fraction", reduce="last"),
        )

    def initialize(
        self, n_atoms: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.asarray(self.model.initialize(n_atoms, rng), dtype=float)

    def step(
        self,
        state: NDArray[np.float64],
        positions_m: NDArray[np.float64],
        velocities_m_per_s: NDArray[np.float64],
        time_s: float,
        dt_s: float,
        rng: np.random.Generator,
    ) -> StochasticStepResult:
        rates = self.light.stimulated_rates(
            positions_m, velocities_m_per_s, time_s=time_s
        )
        new_state, events = self.model.step(
            state, rates, self.light.species.linewidth_rad_s, dt_s, rng
        )
        kicks = sample_recoil_velocity_kicks(
            events,
            self.light.beam_directions,
            self.light.species.recoil_velocity_m_per_s,
            rng,
        )
        return StochasticStepResult(
            state=new_state,
            velocity_kick_m_per_s=kicks,
            diagnostics={
                "scattered_photons": events.total_scattered,
                "excited_fraction": self.model.excited_fraction(new_state),
            },
        )
