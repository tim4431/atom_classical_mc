"""`AtomSystem`: the multiphysics container the driver simulates.

An `AtomSystem` is a species plus a list of physics modules, in the spirit
of a COMSOL model: each module is one physics interface and the driver
(`atommc.driver.simulate`) is the study/solver. Modules come in two kinds
(see `atommc.physics.base`):

- `ConservativeForce` — optical traps, Zeeman potentials, dipole beams.
  Summed into the velocity-Verlet integration.
- `StochasticProcess` — photon scattering (and future non-conservative
  physics). Applied as per-step velocity kicks with per-atom state.

Example::

    system = AtomSystem(species=RB85_D2, modules=[
        ZeemanPotential.for_sublevel([quad], g_f=1/3, m_f=3),
        LightScattering(LightMatterSystem(RB85_D2, beams, [quad])),
    ])
    result = simulate(system, SimulationConfig(...))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .physics.base import (
    ConservativeForce,
    StochasticProcess,
    total_force,
    total_hessian,
    total_potential,
)
from .species import AtomSpecies

PhysicsModule = Union[ConservativeForce, StochasticProcess]


@dataclass(frozen=True)
class AtomSystem:
    """A species plus the physics modules acting on it."""

    species: AtomSpecies
    modules: Sequence[PhysicsModule] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.species, AtomSpecies):
            raise TypeError("species must be an AtomSpecies.")
        modules = tuple(self.modules)
        forces = []
        processes = []
        for module in modules:
            if isinstance(module, ConservativeForce):
                forces.append(module)
            elif isinstance(module, StochasticProcess):
                processes.append(module)
            else:
                raise TypeError(
                    "Each module must be a ConservativeForce or a "
                    f"StochasticProcess; got {type(module).__name__}."
                )
            module_species = getattr(module, "species", None)
            if (
                isinstance(module_species, AtomSpecies)
                and module_species.mass_kg != self.species.mass_kg
            ):
                raise ValueError(
                    f"Module '{module.name}' is built for "
                    f"{module_species.name} but the system species is "
                    f"{self.species.name}."
                )
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "_forces", tuple(forces))
        object.__setattr__(self, "_processes", tuple(processes))

    @property
    def mass_kg(self) -> float:
        return self.species.mass_kg

    @property
    def forces(self) -> tuple[ConservativeForce, ...]:
        """The conservative modules, in registration order."""

        return self._forces  # type: ignore[attr-defined]

    @property
    def processes(self) -> tuple[StochasticProcess, ...]:
        """The stochastic modules, in registration order."""

        return self._processes  # type: ignore[attr-defined]

    @property
    def is_conservative(self) -> bool:
        """True when every module is a conservative force."""

        return not self.processes

    def total_potential(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Summed conservative potential [J]; zeros when there are no forces."""

        positions = np.asarray(positions_m, dtype=float)
        if not self.forces:
            return np.zeros(positions.shape[:-1], dtype=float)
        return total_potential(self.forces, positions, time_s=time_s)

    def total_force(
        self, positions_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Summed conservative force [N]; zeros when there are no forces."""

        positions = np.asarray(positions_m, dtype=float)
        if not self.forces:
            return np.zeros_like(positions)
        return total_force(self.forces, positions, time_s=time_s)

    def total_hessian(
        self, position_m: ArrayLike, time_s: float = 0.0
    ) -> NDArray[np.float64]:
        """Summed conservative Hessian [J/m^2] at one position."""

        return total_hessian(self.forces, position_m, time_s=time_s)

    def mechanical_energy_j(
        self,
        positions_m: ArrayLike,
        velocities_m_per_s: ArrayLike,
        time_s: float = 0.0,
    ) -> NDArray[np.float64]:
        """Kinetic plus summed conservative potential energy [J]."""

        velocities = np.asarray(velocities_m_per_s, dtype=float)
        kinetic = 0.5 * self.mass_kg * np.sum(velocities * velocities, axis=-1)
        return kinetic + self.total_potential(positions_m, time_s=time_s)
