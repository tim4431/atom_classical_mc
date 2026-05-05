import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import (  # noqa: E402
    capture_probability,
    classify_final_trap_occupation,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
    survival_probability_time_series,
)
from src.constants import (  # noqa: E402
    BOLTZMANN_CONSTANT_J_PER_K,
    HBAR_J_S,
    RB87_MASS_KG,
)
from src.harmonic import (  # noqa: E402
    approximate_harmonic_potential,
    approximate_harmonic_potential_from_callable,
    coherent_fock_probabilities,
    decompose_motion_into_harmonic_modes,
    summarize_mode_occupations,
)
from src.ramp import RampSequence  # noqa: E402
from src.sampling import sample_thermal_velocities  # noqa: E402
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.trap import TrapConfig, total_force, total_potential  # noqa: E402
from src.units import ms, um  # noqa: E402


class CorePhysicsTests(unittest.TestCase):
    def test_gaussian_potential_and_force_symmetry(self):
        trap = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(1.0)),
            waist_axial_m=float(um(4.0)),
            depth_uK=100.0,
        )

        center_potential = total_potential(trap, np.array([0.0, 0.0, 0.0]))
        offset_potential = total_potential(trap, np.array([0.2e-6, 0.0, 0.0]))
        self.assertLess(center_potential, offset_potential)

        forces = total_force(
            trap,
            np.array(
                [
                    [0.2e-6, 0.0, 0.0],
                    [-0.2e-6, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        self.assertLess(forces[0, 0], 0.0)
        self.assertGreater(forces[1, 0], 0.0)
        np.testing.assert_allclose(forces[2], np.zeros(3), atol=0.0)
        np.testing.assert_allclose(forces[0, 0], -forces[1, 0], rtol=1e-12)

    def test_ramp_interpolation_and_clamping(self):
        ramp = RampSequence(
            times_s=[0.0, 1.0e-3, 2.0e-3],
            centers_m=[[0.0, 0.0, 0.0], [2.0e-6, 0.0, 0.0], [4.0e-6, 2.0e-6, 0.0]],
            depths_uK=[10.0, 20.0, 30.0],
        )

        center, depth = ramp.at(0.5e-3)
        np.testing.assert_allclose(center, [1.0e-6, 0.0, 0.0])
        self.assertAlmostEqual(depth, 15.0)

        start_center, start_depth = ramp.at(-1.0)
        np.testing.assert_allclose(start_center, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(start_depth, 10.0)

        end_center, end_depth = ramp.at(10.0)
        np.testing.assert_allclose(end_center, [4.0e-6, 2.0e-6, 0.0])
        self.assertAlmostEqual(end_depth, 30.0)

    def test_velocity_sampling_temperature(self):
        rng = np.random.default_rng(123)
        temperature_uK = 12.0
        velocities = sample_thermal_velocities(
            temperature_uK=temperature_uK,
            ensemble_size=40000,
            rng=rng,
            mass_kg=RB87_MASS_KG,
        )

        estimated_temperature_k = (
            RB87_MASS_KG
            * np.mean(np.sum(velocities * velocities, axis=-1))
            / (3.0 * BOLTZMANN_CONSTANT_J_PER_K)
        )
        estimated_temperature_uK = estimated_temperature_k / 1.0e-6
        self.assertLess(abs(estimated_temperature_uK - temperature_uK), 0.4)

    def test_static_trap_has_low_numerical_heating(self):
        static_trap = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=120.0,
        )
        moving_base = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=0.0,
        )
        ramp = RampSequence(
            times_s=[0.0, float(ms(0.04))],
            centers_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            depths_uK=[0.0, 0.0],
        )
        config = SimulationConfig(
            initial_temperature_uK=5.0,
            timestep_s=2.0e-8,
            duration_s=float(ms(0.04)),
            ensemble_size=128,
            random_seed=7,
        )

        result = run_simulation(static_trap, moving_base, ramp, config)
        self.assertGreater(result.survival_probability, 0.98)
        self.assertLess(abs(result.mean_energy_gain_uK), 0.15)
        self.assertEqual(result.initial_positions_m.shape, (128, 3))
        self.assertEqual(result.initial_velocities_m_per_s.shape, (128, 3))
        self.assertTrue(np.all(result.initial_energies_uK < 0.0))

    def test_harmonic_approximation_and_mode_decomposition(self):
        trap = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(1.0)),
            waist_axial_m=float(um(5.0)),
            depth_uK=100.0,
        )
        approximation = approximate_harmonic_potential(
            trap,
            trap.center_m,
            mass_kg=RB87_MASS_KG,
        )

        expected_stiffness = 4.0 * trap.depth_joule / trap.waist_radial_m**2
        expected_omega = np.sqrt(expected_stiffness / RB87_MASS_KG)
        radial_indices = [
            index
            for index, label in enumerate(approximation.mode_labels)
            if label.startswith("radial")
        ]
        self.assertEqual(len(radial_indices), 2)
        np.testing.assert_allclose(
            approximation.angular_frequencies_rad_s[radial_indices],
            expected_omega,
            rtol=1.0e-12,
        )
        self.assertLess(approximation.gradient_norm_n, 1.0e-30)

        positions = np.zeros((2, 3), dtype=float)
        velocities = np.zeros((2, 3), dtype=float)
        radial_index = radial_indices[0]
        one_quantum_amplitude = np.sqrt(
            2.0 * HBAR_J_S / (RB87_MASS_KG * expected_omega)
        )
        positions[1] = one_quantum_amplitude * approximation.mode_axes[:, radial_index]
        decomposition = decompose_motion_into_harmonic_modes(
            approximation,
            positions,
            velocities,
        )

        np.testing.assert_allclose(decomposition.mean_occupations[0], np.zeros(3))
        self.assertAlmostEqual(decomposition.mean_occupations[1, radial_index], 1.0)
        self.assertEqual(decomposition.nearest_quantum_numbers[1, radial_index], 1)

        summary = summarize_mode_occupations(decomposition)
        self.assertIn(approximation.mode_labels[radial_index], summary)
        probabilities = coherent_fock_probabilities(
            decomposition.mean_occupations[1, radial_index],
            max_n=4,
        )
        expected = np.exp(-1.0) * np.array([1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0])
        np.testing.assert_allclose(probabilities, expected)

    def test_callable_harmonic_approximation_matches_quadratic_potential(self):
        stiffness = np.diag([2.0e-18, 3.0e-18, 5.0e-19])
        center = np.array([0.2e-6, -0.1e-6, 0.05e-6])

        def potential(position_m):
            offset = np.asarray(position_m, dtype=float) - center
            return 1.0e-30 + 0.5 * offset @ stiffness @ offset

        approximation = approximate_harmonic_potential_from_callable(
            potential,
            center,
            finite_difference_step_m=1.0e-8,
            mass_kg=RB87_MASS_KG,
            mode_labels=("x", "z", "y"),
        )

        np.testing.assert_allclose(
            np.sort(approximation.angular_frequencies_rad_s),
            np.sort(np.sqrt(np.diag(stiffness) / RB87_MASS_KG)),
            rtol=1.0e-10,
        )
        self.assertLess(approximation.gradient_norm_n, 1.0e-32)

    def test_stored_trajectories_include_visualization_series(self):
        static_trap = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=80.0,
        )
        moving_base = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=0.0,
        )
        ramp = RampSequence(
            times_s=[0.0, 5.0e-6],
            centers_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            depths_uK=[0.0, 0.0],
        )
        result = run_simulation(
            static_trap,
            moving_base,
            ramp,
            SimulationConfig(
                initial_temperature_uK=4.0,
                timestep_s=1.0e-7,
                duration_s=5.0e-6,
                ensemble_size=32,
                random_seed=5,
                store_trajectories=True,
                trajectory_stride=10,
            ),
        )

        self.assertIsNotNone(result.trajectory_times_s)
        self.assertIsNotNone(result.trajectory_positions_m)
        self.assertIsNotNone(result.trajectory_velocities_m_per_s)
        self.assertIsNotNone(result.trajectory_lost)
        self.assertEqual(result.trajectory_positions_m.shape[-2:], (32, 3))
        self.assertEqual(result.trajectory_velocities_m_per_s.shape[-2:], (32, 3))
        self.assertFalse(np.any(result.trajectory_lost[0]))

        survival = survival_probability_time_series(result)
        loss = loss_probability_time_series(result)
        kinetic = mean_kinetic_energy_time_series_uK(result)
        self.assertEqual(survival.shape, result.trajectory_times_s.shape)
        self.assertEqual(loss.shape, result.trajectory_times_s.shape)
        np.testing.assert_allclose(loss, 1.0 - survival)
        self.assertEqual(kinetic.shape, result.trajectory_times_s.shape)
        self.assertGreaterEqual(capture_probability(result, static_trap), 0.0)
        occupation = classify_final_trap_occupation(result, static_trap, moving_base)
        total_probability = (
            occupation.slm_probability
            + occupation.aod_probability
            + occupation.ambiguous_probability
            + occupation.unbound_survivor_probability
            + occupation.loss_probability
        )
        self.assertAlmostEqual(total_probability, 1.0)

    def test_aggressive_pullout_loses_more_than_slow_pullout(self):
        static_trap = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=0.0,
        )
        moving_base = TrapConfig(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(1.5)),
            waist_axial_m=float(um(6.0)),
            depth_uK=60.0,
        )

        slow_ramp = RampSequence(
            times_s=[0.0, float(ms(0.5))],
            centers_m=[[0.0, 0.0, 0.0], [6.0e-6, 0.0, 0.0]],
            depths_uK=[60.0, 60.0],
        )
        fast_ramp = RampSequence(
            times_s=[0.0, 2.0e-6],
            centers_m=[[0.0, 0.0, 0.0], [6.0e-6, 0.0, 0.0]],
            depths_uK=[60.0, 60.0],
        )

        common = dict(
            initial_temperature_uK=3.0,
            ensemble_size=96,
            random_seed=11,
            loss_radius_m=20.0e-6,
        )
        slow = run_simulation(
            static_trap,
            moving_base,
            slow_ramp,
            SimulationConfig(
                timestep_s=2.5e-7,
                duration_s=float(ms(0.5)),
                **common,
            ),
        )
        fast = run_simulation(
            static_trap,
            moving_base,
            fast_ramp,
            SimulationConfig(
                timestep_s=2.0e-8,
                duration_s=2.0e-6,
                **common,
            ),
        )

        self.assertGreater(slow.survival_probability, fast.survival_probability)


if __name__ == "__main__":
    unittest.main()
