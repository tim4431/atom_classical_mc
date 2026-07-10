"""Harmonic trap approximation and motional-state decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import HBAR_J_S, RB87_MASS_KG
from ..physics.base import ConservativeForce, total_force, total_hessian, total_potential
from ..units import joule_to_microkelvin


@dataclass(frozen=True)
class HarmonicApproximation:
    """Second-order approximation of a potential near a chosen center."""

    center_m: NDArray[np.float64]
    potential_offset_joule: float
    gradient_j_per_m: NDArray[np.float64]
    hessian_j_per_m2: NDArray[np.float64]
    mass_kg: float
    angular_frequencies_rad_s: NDArray[np.float64]
    mode_axes: NDArray[np.float64]
    mode_labels: tuple[str, str, str]

    @property
    def frequencies_hz(self) -> NDArray[np.float64]:
        """Normal-mode frequencies in Hz."""

        return self.angular_frequencies_rad_s / (2.0 * np.pi)

    @property
    def gradient_norm_n(self) -> float:
        """Magnitude of the local force imbalance in newtons."""

        return float(np.linalg.norm(self.gradient_j_per_m))

    def quadratic_potential(
        self,
        positions_m: ArrayLike,
        include_linear_term: bool = True,
    ) -> NDArray[np.float64]:
        """Evaluate the Taylor-expanded potential near the center."""

        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,):
            raise ValueError("positions_m must have final dimension 3.")

        offsets = positions - self.center_m
        quadratic = 0.5 * np.einsum(
            "...i,ij,...j->...",
            offsets,
            self.hessian_j_per_m2,
            offsets,
        )
        if include_linear_term:
            linear = np.einsum("...i,i->...", offsets, self.gradient_j_per_m)
        else:
            linear = 0.0
        return self.potential_offset_joule + linear + quadratic


@dataclass(frozen=True)
class MotionalDecomposition:
    """Classical phase-space coordinates decomposed into harmonic modes."""

    approximation: HarmonicApproximation
    positions_mode_m: NDArray[np.float64]
    velocities_mode_m_per_s: NDArray[np.float64]
    mode_energies_joule: NDArray[np.float64]
    mode_energies_uK: NDArray[np.float64]
    coherent_amplitudes: NDArray[np.complex128]
    mean_occupations: NDArray[np.float64]
    nearest_quantum_numbers: NDArray[np.int64]

    @property
    def mode_labels(self) -> tuple[str, str, str]:
        return self.approximation.mode_labels

    @property
    def angular_frequencies_rad_s(self) -> NDArray[np.float64]:
        return self.approximation.angular_frequencies_rad_s

    @property
    def frequencies_hz(self) -> NDArray[np.float64]:
        return self.approximation.frequencies_hz


def approximate_harmonic_potential(
    traps: ConservativeForce | Iterable[ConservativeForce],
    center_m: ArrayLike,
    mass_kg: float = RB87_MASS_KG,
    mode_labels: Sequence[str] | None = None,
    require_positive_definite: bool = True,
    time_s: float = 0.0,
) -> HarmonicApproximation:
    """Approximate one or more traps by a harmonic potential near `center_m`.

    The Taylor expansion is
    `U(r) ~= U(c) + grad U(c).(r-c) + 1/2 (r-c)^T K (r-c)`.
    Motional-state decomposition is physically meaningful when `center_m` is a
    stable equilibrium, so `grad U(c)` should be close to zero and `K` should be
    positive definite. For time-dependent traps, `time_s` selects the snapshot
    used for the expansion.
    """

    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,):
        raise ValueError("center_m must be a 3-vector in meters.")
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")

    hessian = total_hessian(traps, center, time_s=time_s)
    hessian = 0.5 * (hessian + hessian.T)
    return _build_harmonic_approximation(
        center_m=center,
        potential_offset_joule=float(total_potential(traps, center, time_s=time_s)),
        gradient_j_per_m=-np.asarray(
            total_force(traps, center, time_s=time_s), dtype=float
        ),
        hessian_j_per_m2=hessian,
        mass_kg=mass_kg,
        mode_labels=mode_labels,
        require_positive_definite=require_positive_definite,
    )


def approximate_harmonic_potential_from_callable(
    potential_joule_fn: Callable[[NDArray[np.float64]], ArrayLike],
    center_m: ArrayLike,
    finite_difference_step_m: float | ArrayLike,
    mass_kg: float = RB87_MASS_KG,
    mode_labels: Sequence[str] | None = None,
    require_positive_definite: bool = True,
) -> HarmonicApproximation:
    """Finite-difference harmonic approximation for a generic potential.

    `potential_joule_fn` should accept positions with final dimension 3 and
    return potential energy in joules. This helper uses central finite
    differences for the gradient and Hessian.
    """

    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,):
        raise ValueError("center_m must be a 3-vector in meters.")
    step = np.asarray(finite_difference_step_m, dtype=float)
    if step.shape == ():
        step = np.full(3, float(step), dtype=float)
    if step.shape != (3,) or np.any(step <= 0.0):
        raise ValueError("finite_difference_step_m must be positive scalar or 3-vector.")
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")

    potential_offset = _evaluate_potential_scalar(potential_joule_fn, center)
    gradient = np.zeros(3, dtype=float)
    hessian = np.zeros((3, 3), dtype=float)
    basis = np.eye(3)

    for axis in range(3):
        plus = center + step[axis] * basis[axis]
        minus = center - step[axis] * basis[axis]
        potential_plus = _evaluate_potential_scalar(potential_joule_fn, plus)
        potential_minus = _evaluate_potential_scalar(potential_joule_fn, minus)
        gradient[axis] = (potential_plus - potential_minus) / (2.0 * step[axis])
        hessian[axis, axis] = (
            potential_plus - 2.0 * potential_offset + potential_minus
        ) / (step[axis] * step[axis])

    for axis_i in range(3):
        for axis_j in range(axis_i + 1, 3):
            delta_i = step[axis_i] * basis[axis_i]
            delta_j = step[axis_j] * basis[axis_j]
            mixed = (
                _evaluate_potential_scalar(potential_joule_fn, center + delta_i + delta_j)
                - _evaluate_potential_scalar(potential_joule_fn, center + delta_i - delta_j)
                - _evaluate_potential_scalar(potential_joule_fn, center - delta_i + delta_j)
                + _evaluate_potential_scalar(potential_joule_fn, center - delta_i - delta_j)
            ) / (4.0 * step[axis_i] * step[axis_j])
            hessian[axis_i, axis_j] = mixed
            hessian[axis_j, axis_i] = mixed

    return _build_harmonic_approximation(
        center_m=center,
        potential_offset_joule=potential_offset,
        gradient_j_per_m=gradient,
        hessian_j_per_m2=hessian,
        mass_kg=mass_kg,
        mode_labels=mode_labels,
        require_positive_definite=require_positive_definite,
    )


def _build_harmonic_approximation(
    center_m: NDArray[np.float64],
    potential_offset_joule: float,
    gradient_j_per_m: NDArray[np.float64],
    hessian_j_per_m2: NDArray[np.float64],
    mass_kg: float,
    mode_labels: Sequence[str] | None,
    require_positive_definite: bool,
) -> HarmonicApproximation:
    hessian = 0.5 * (hessian_j_per_m2 + hessian_j_per_m2.T)
    stiffness, mode_axes = np.linalg.eigh(hessian)
    if require_positive_definite and np.any(stiffness <= 0.0):
        raise ValueError("The potential Hessian is not positive definite at center_m.")

    positive_stiffness = np.maximum(stiffness, 0.0)
    angular_frequencies = np.sqrt(positive_stiffness / mass_kg)
    labels = (
        tuple(mode_labels)
        if mode_labels is not None
        else _default_mode_labels(mode_axes)
    )
    if len(labels) != 3:
        raise ValueError("mode_labels must contain exactly three labels.")

    return HarmonicApproximation(
        center_m=center_m,
        potential_offset_joule=float(potential_offset_joule),
        gradient_j_per_m=np.asarray(gradient_j_per_m, dtype=float),
        hessian_j_per_m2=hessian,
        mass_kg=mass_kg,
        angular_frequencies_rad_s=angular_frequencies,
        mode_axes=mode_axes,
        mode_labels=(str(labels[0]), str(labels[1]), str(labels[2])),
    )


def _evaluate_potential_scalar(
    potential_joule_fn: Callable[[NDArray[np.float64]], ArrayLike],
    position_m: NDArray[np.float64],
) -> float:
    value = np.asarray(potential_joule_fn(np.asarray(position_m, dtype=float)), dtype=float)
    if value.shape == ():
        return float(value)
    if value.size == 1:
        return float(np.reshape(value, ())[()])
    raise ValueError("potential_joule_fn must return a scalar for one position.")


def decompose_motion_into_harmonic_modes(
    approximation: HarmonicApproximation,
    positions_m: ArrayLike,
    velocities_m_per_s: ArrayLike,
    trap_velocity_m_per_s: ArrayLike = (0.0, 0.0, 0.0),
) -> MotionalDecomposition:
    """Decompose atom motion into harmonic normal modes.

    The returned `mean_occupations` are coherent-state/semi-classical motional
    quanta, `nbar_i = E_i / (hbar omega_i)`, where `E_i` is the classical
    oscillator energy above the harmonic minimum for mode `i`.
    """

    positions = np.asarray(positions_m, dtype=float)
    velocities = np.asarray(velocities_m_per_s, dtype=float)
    trap_velocity = np.asarray(trap_velocity_m_per_s, dtype=float)
    if positions.shape[-1:] != (3,) or velocities.shape[-1:] != (3,):
        raise ValueError("positions_m and velocities_m_per_s must have final dimension 3.")
    if positions.shape != velocities.shape:
        raise ValueError("positions_m and velocities_m_per_s must have the same shape.")
    if trap_velocity.shape != (3,):
        raise ValueError("trap_velocity_m_per_s must be a 3-vector.")
    if np.any(approximation.angular_frequencies_rad_s <= 0.0):
        raise ValueError("All harmonic frequencies must be positive for decomposition.")

    offsets = positions - approximation.center_m
    relative_velocities = velocities - trap_velocity
    positions_mode = offsets @ approximation.mode_axes
    velocities_mode = relative_velocities @ approximation.mode_axes
    omega = approximation.angular_frequencies_rad_s
    mass = approximation.mass_kg

    potential_energy = 0.5 * mass * omega * omega * positions_mode * positions_mode
    kinetic_energy = 0.5 * mass * velocities_mode * velocities_mode
    mode_energies_joule = potential_energy + kinetic_energy
    coherent_amplitudes = (
        np.sqrt(mass * omega / (2.0 * HBAR_J_S)) * positions_mode
        + 1j * np.sqrt(mass / (2.0 * HBAR_J_S * omega)) * velocities_mode
    )
    mean_occupations = mode_energies_joule / (HBAR_J_S * omega)
    nearest_quantum_numbers = np.rint(mean_occupations).astype(np.int64)

    return MotionalDecomposition(
        approximation=approximation,
        positions_mode_m=positions_mode,
        velocities_mode_m_per_s=velocities_mode,
        mode_energies_joule=mode_energies_joule,
        mode_energies_uK=joule_to_microkelvin(mode_energies_joule),
        coherent_amplitudes=coherent_amplitudes,
        mean_occupations=mean_occupations,
        nearest_quantum_numbers=nearest_quantum_numbers,
    )


def coherent_fock_probabilities(
    mean_occupations: ArrayLike,
    max_n: int,
) -> NDArray[np.float64]:
    """Return Fock-state probabilities for coherent states with mean `nbar`.

    The output shape is `mean_occupations.shape + (max_n + 1,)`.
    """

    if max_n < 0:
        raise ValueError("max_n must be non-negative.")
    nbar = np.asarray(mean_occupations, dtype=float)
    probabilities = np.empty(nbar.shape + (max_n + 1,), dtype=float)
    probabilities[..., 0] = np.exp(-nbar)
    for n in range(1, max_n + 1):
        probabilities[..., n] = probabilities[..., n - 1] * nbar / n
    return probabilities


def summarize_mode_occupations(
    decomposition: MotionalDecomposition,
    mask: ArrayLike | None = None,
) -> dict[str, dict[str, float]]:
    """Return mean/median/std motional occupation for each mode."""

    occupations = decomposition.mean_occupations
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        occupations = occupations[mask_array]
    if occupations.size == 0:
        return {
            label: {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
            for label in decomposition.mode_labels
        }

    return {
        label: {
            "mean": float(np.mean(occupations[:, mode_index])),
            "median": float(np.median(occupations[:, mode_index])),
            "std": float(np.std(occupations[:, mode_index])),
        }
        for mode_index, label in enumerate(decomposition.mode_labels)
    }


def _default_mode_labels(mode_axes: NDArray[np.float64]) -> tuple[str, str, str]:
    axis_names = ("x", "y", "z")
    used: dict[str, int] = {}
    labels: list[str] = []
    for mode_index in range(3):
        dominant_axis = int(np.argmax(np.abs(mode_axes[:, mode_index])))
        axis_name = axis_names[dominant_axis]
        base_label = "axial_z" if axis_name == "z" else f"radial_{axis_name}"
        count = used.get(base_label, 0)
        used[base_label] = count + 1
        labels.append(base_label if count == 0 else f"{base_label}_{count + 1}")
    return (labels[0], labels[1], labels[2])
