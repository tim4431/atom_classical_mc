"""atommc: classical Monte Carlo multiphysics simulator for cold atoms.

The top-level namespace re-exports the full modeling surface: species,
field/beam geometry, physics modules (conservative forces + stochastic
processes), ensemble sources, and the `AtomSystem`/`simulate` driver.
Post-processing (`analysis`, `harmonic`, `visualization`) stays behind
`atommc.postprocess` so matplotlib remains an optional dependency.
"""

from .constants import (
    BOHR_MAGNETON_J_PER_T,
    BOLTZMANN_CONSTANT_J_PER_K,
    HBAR_J_S,
    RB85_MASS_KG,
    RB87_MASS_KG,
)
from .driver import SimulationConfig, SimulationResult, simulate
from .ensemble import (
    EffusiveBeam,
    EnsembleSample,
    EnsembleSource,
    HarmonicTrapCloud,
    ThermalCloud,
    sample_thermal_positions_harmonic,
    sample_thermal_velocities,
)
from .geometry.fields import (
    MagneticFieldConfig,
    QuadrupoleMagneticField,
    UniformMagneticField,
    total_magnetic_field,
    total_magnetic_field_jacobian,
)
from .geometry.laser import LaserBeam, six_beam_mot
from .physics.base import (
    ConservativeForce,
    DiagnosticSpec,
    StochasticProcess,
    StochasticStepResult,
    total_force,
    total_hessian,
    total_potential,
)
from .physics.dipole import DipoleBeamPotential
from .physics.internal_state import (
    AdiabaticSteadyState,
    InternalStateModel,
    RateEquationPopulations,
    ScatteringEvents,
    StochasticJumpState,
    sample_recoil_velocity_kicks,
)
from .physics.light_matter import LightMatterSystem, polarization_fractions
from .physics.scattering import LightScattering
from .physics.traps import (
    AstigmaticAODTrap,
    GaussianTrap,
    GriddedTrap,
    MovingGaussianTrap,
)
from .physics.zeeman import ZeemanPotential
from .ramp import (
    CONST_JERK,
    LINEAR,
    QUINTIC_MIN_JERK,
    PolynomialConnector,
    RampSequence,
    arb_fifth_poly,
    const_jerk,
)
from .species import RB85_D2, RB87_D2, AtomSpecies
from .system import AtomSystem, PhysicsModule
from .units import (
    gauss,
    gauss_per_cm,
    joule_to_microkelvin,
    mhz,
    microkelvin_to_joule,
    ms,
    um,
)

__all__ = [
    # constants
    "BOHR_MAGNETON_J_PER_T",
    "BOLTZMANN_CONSTANT_J_PER_K",
    "HBAR_J_S",
    "RB85_MASS_KG",
    "RB87_MASS_KG",
    # driver
    "SimulationConfig",
    "SimulationResult",
    "simulate",
    # system
    "AtomSystem",
    "PhysicsModule",
    # physics protocols
    "ConservativeForce",
    "StochasticProcess",
    "StochasticStepResult",
    "DiagnosticSpec",
    "total_force",
    "total_hessian",
    "total_potential",
    # conservative modules
    "AstigmaticAODTrap",
    "DipoleBeamPotential",
    "GaussianTrap",
    "GriddedTrap",
    "MovingGaussianTrap",
    "ZeemanPotential",
    # stochastic modules + backends
    "LightScattering",
    "LightMatterSystem",
    "polarization_fractions",
    "InternalStateModel",
    "AdiabaticSteadyState",
    "RateEquationPopulations",
    "StochasticJumpState",
    "ScatteringEvents",
    "sample_recoil_velocity_kicks",
    # geometry
    "LaserBeam",
    "six_beam_mot",
    "MagneticFieldConfig",
    "UniformMagneticField",
    "QuadrupoleMagneticField",
    "total_magnetic_field",
    "total_magnetic_field_jacobian",
    # ramps
    "RampSequence",
    "PolynomialConnector",
    "LINEAR",
    "CONST_JERK",
    "QUINTIC_MIN_JERK",
    "arb_fifth_poly",
    "const_jerk",
    # species
    "AtomSpecies",
    "RB85_D2",
    "RB87_D2",
    # ensembles
    "EnsembleSource",
    "EnsembleSample",
    "ThermalCloud",
    "HarmonicTrapCloud",
    "EffusiveBeam",
    "sample_thermal_positions_harmonic",
    "sample_thermal_velocities",
    # units
    "um",
    "ms",
    "mhz",
    "gauss",
    "gauss_per_cm",
    "microkelvin_to_joule",
    "joule_to_microkelvin",
]
