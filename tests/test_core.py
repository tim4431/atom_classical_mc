import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from atommc.postprocess.analysis import (  # noqa: E402
    capture_probability,
    classify_final_trap_occupation,
    loss_probability_time_series,
    mean_kinetic_energy_time_series_uK,
    survival_probability_time_series,
)
from atommc.constants import (  # noqa: E402
    BOLTZMANN_CONSTANT_J_PER_K,
    HBAR_J_S,
    RB87_MASS_KG,
)
from atommc.postprocess.harmonic import (  # noqa: E402
    approximate_harmonic_potential,
    approximate_harmonic_potential_from_callable,
    coherent_fock_probabilities,
    decompose_motion_into_harmonic_modes,
    summarize_mode_occupations,
)
from atommc import (  # noqa: E402
    QUINTIC_MIN_JERK,
    AstigmaticAODTrap,
    AtomSystem,
    GaussianTrap,
    GriddedTrap,
    MovingGaussianTrap,
    RB87_D2,
    RampSequence,
    SimulationConfig,
    arb_fifth_poly,
    microkelvin_to_joule,
    ms,
    sample_thermal_velocities,
    simulate,
    total_force,
    total_potential,
    um,
)


def simulate_traps(traps, config):
    """Run the given conservative traps as an Rb87 AtomSystem."""

    return simulate(AtomSystem(species=RB87_D2, modules=list(traps)), config)


class CorePhysicsTests(unittest.TestCase):
    def test_gaussian_potential_and_force_symmetry(self):
        trap = GaussianTrap(
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

    def test_ramp_linear_interpolation_and_clamping(self):
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

        # Linear position profile -> constant velocity within a segment.
        velocity = ramp.velocity_at(0.5e-3)
        expected_velocity = np.array([2.0e-6, 0.0, 0.0]) / 1.0e-3
        np.testing.assert_allclose(velocity, expected_velocity)
        np.testing.assert_allclose(ramp.velocity_at(-0.1), np.zeros(3))
        np.testing.assert_allclose(ramp.velocity_at(5.0), np.zeros(3))

    def test_smooth_position_profile_zero_velocity_at_endpoints(self):
        ramp = RampSequence(
            times_s=[0.0, 1.0e-3],
            centers_m=[[0.0, 0.0, 0.0], [4.0e-6, 0.0, 0.0]],
            depths_uK=[100.0, 100.0],
            position_profile=QUINTIC_MIN_JERK,
        )
        # Min-jerk profile has zero velocity at u = 0 and u = 1 (endpoint
        # smoothness), and peaks at u = 0.5.
        v_start = ramp.velocity_at(1.0e-9)
        v_end = ramp.velocity_at(1.0e-3 - 1.0e-9)
        v_mid = ramp.velocity_at(0.5e-3)
        self.assertAlmostEqual(np.linalg.norm(v_start), 0.0, places=2)
        self.assertAlmostEqual(np.linalg.norm(v_end), 0.0, places=2)
        # Peak velocity for quintic min-jerk = 1.875 * (delta / T).
        peak = 1.875 * 4.0e-6 / 1.0e-3
        np.testing.assert_allclose(v_mid[0], peak, rtol=1.0e-3)

    def test_arb_fifth_poly_at_min_jerk_beta_matches_quintic_min_jerk(self):
        # arb_fifth_poly(beta) reduces to QUINTIC_MIN_JERK at beta = 15/8
        # (the unique quintic with zero endpoint velocity AND acceleration).
        arb = arb_fifth_poly(15.0 / 8.0)
        u = np.linspace(0.0, 1.0, 9)
        np.testing.assert_allclose(
            arb.value(u), QUINTIC_MIN_JERK.value(u), atol=1.0e-12
        )
        np.testing.assert_allclose(
            arb.derivative(u), QUINTIC_MIN_JERK.derivative(u), atol=1.0e-12
        )

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
        static_trap = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=120.0,
        )
        moving_base = GaussianTrap(
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

        moving = MovingGaussianTrap(template=moving_base, ramp=ramp)
        result = simulate_traps([static_trap, moving], config)
        self.assertGreater(result.survival_probability, 0.98)
        self.assertLess(abs(result.mean_energy_gain_uK), 0.15)
        self.assertEqual(result.initial_positions_m.shape, (128, 3))
        self.assertEqual(result.initial_velocities_m_per_s.shape, (128, 3))
        self.assertTrue(np.all(result.initial_energies_uK < 0.0))

    def test_simulate_accepts_multiple_forces(self):
        static_trap = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=120.0,
        )
        moving = MovingGaussianTrap(
            template=GaussianTrap(
                waist_radial_m=float(um(2.0)),
                waist_axial_m=float(um(8.0)),
                depth_uK=0.0,
            ),
            ramp=RampSequence(
                times_s=[0.0, float(ms(0.04))],
                centers_m=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                depths_uK=[0.0, 0.0],
            ),
        )
        config = SimulationConfig(
            initial_temperature_uK=5.0,
            timestep_s=2.0e-8,
            duration_s=float(ms(0.04)),
            ensemble_size=64,
            random_seed=7,
        )
        result = simulate_traps([static_trap, moving], config)
        self.assertGreater(result.survival_probability, 0.98)
        # Both temperature variants should be populated.
        self.assertFalse(np.isnan(result.final_temperature_uK_survivors))
        self.assertFalse(np.isnan(result.final_temperature_uK_all))
        # Survivors-only and all-atoms gains should both be finite small numbers.
        gain_survivors = result.temperature_gain_uK_at(survivors_only=True)
        gain_all = result.temperature_gain_uK_at(survivors_only=False)
        self.assertTrue(np.isfinite(gain_survivors))
        self.assertTrue(np.isfinite(gain_all))

    def test_harmonic_approximation_and_mode_decomposition(self):
        trap = GaussianTrap(
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
        static_trap = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=80.0,
        )
        moving_base = GaussianTrap(
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
        result = simulate_traps(
            [static_trap, MovingGaussianTrap(template=moving_base, ramp=ramp)],
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
        static_trap = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(2.0)),
            waist_axial_m=float(um(8.0)),
            depth_uK=0.0,
        )
        moving_base = GaussianTrap(
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
        slow = simulate_traps(
            [static_trap, MovingGaussianTrap(template=moving_base, ramp=slow_ramp)],
            SimulationConfig(
                timestep_s=2.5e-7,
                duration_s=float(ms(0.5)),
                **common,
            ),
        )
        fast = simulate_traps(
            [static_trap, MovingGaussianTrap(template=moving_base, ramp=fast_ramp)],
            SimulationConfig(
                timestep_s=2.0e-8,
                duration_s=2.0e-6,
                **common,
            ),
        )

        self.assertGreater(slow.survival_probability, fast.survival_probability)


class AstigmaticAODTests(unittest.TestCase):
    def test_astigmatic_force_matches_finite_difference(self):
        ramp = RampSequence(
            times_s=[0.0, 1.0e-3],
            centers_m=[[0.0, 0.0, 0.0], [4.0e-6, 0.0, 0.0]],
            depths_uK=[200.0, 200.0],
            position_profile=QUINTIC_MIN_JERK,
        )
        trap = AstigmaticAODTrap(
            waist_radial_m=500.0e-9,
            wavelength_m=487.0e-9,
            ramp=ramp,
            dxdt2z=0.025 / 4.0 / 650.0,
        )
        rng = np.random.default_rng(0)
        for _ in range(20):
            t = float(rng.uniform(0.0, 1.0e-3))
            r0 = trap.center_at(t) + rng.uniform(-3.0e-7, 3.0e-7, size=3)
            f_analytic = trap.force(r0, time_s=t)

            step = 5.0e-11
            f_fdiff = np.zeros(3)
            for axis in range(3):
                delta = np.zeros(3)
                delta[axis] = step
                u_plus = trap.potential(r0 + delta, time_s=t)
                u_minus = trap.potential(r0 - delta, time_s=t)
                f_fdiff[axis] = -(u_plus - u_minus) / (2.0 * step)
            np.testing.assert_allclose(f_analytic, f_fdiff, rtol=5.0e-3, atol=1.0e-30)

    def test_astigmatic_zero_velocity_is_cylindrically_symmetric(self):
        ramp = RampSequence(
            times_s=[0.0, 1.0e-3, 2.0e-3],
            centers_m=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            depths_uK=[150.0, 150.0, 150.0],
        )
        astig = AstigmaticAODTrap(
            waist_radial_m=500.0e-9,
            wavelength_m=487.0e-9,
            ramp=ramp,
            dxdt2z=1.0e-5,
        )
        # At rest, z01 = z02, so swapping x <-> y (or sign flips) leave U unchanged.
        rng = np.random.default_rng(7)
        for _ in range(8):
            r = rng.uniform(-3.0e-7, 3.0e-7, size=3)
            r_swapped = np.array([r[1], r[0], r[2]])
            u_r = astig.potential(r, time_s=0.5e-3)
            u_swap = astig.potential(r_swapped, time_s=0.5e-3)
            np.testing.assert_allclose(u_r, u_swap, rtol=1.0e-12)
            # And in the radial plane (z = 0) the profile is a true 2D Gaussian
            # in (x, y) with the bare radial waist.
            r_xy = np.array([r[0], r[1], 0.0])
            expected = -float(microkelvin_to_joule(150.0)) * np.exp(
                -2.0 * (r[0] ** 2 + r[1] ** 2) / 500.0e-9**2
            )
            np.testing.assert_allclose(
                astig.potential(r_xy, time_s=0.5e-3), expected, rtol=1.0e-12
            )

    def test_lensing_increases_axial_heating(self):
        # AOD ramp drags the trap with a smooth quintic profile; with dxdt2z>0
        # the velocity-coupled focal shift should lift the axial energy by more
        # than the cylindrically symmetric baseline.
        ramp = RampSequence(
            times_s=[0.0, 5.0e-5, 1.5e-4, 2.0e-4],
            centers_m=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [4.0e-6, 0.0, 0.0],
                [4.0e-6, 0.0, 0.0],
            ],
            depths_uK=[400.0, 400.0, 400.0, 400.0],
            position_profile=QUINTIC_MIN_JERK,
        )
        common_config = SimulationConfig(
            initial_temperature_uK=5.0,
            timestep_s=2.0e-8,
            duration_s=2.0e-4,
            ensemble_size=128,
            random_seed=3,
            loss_radius_m=20.0e-6,
        )
        cyl_trap = AstigmaticAODTrap(
            waist_radial_m=500.0e-9,
            wavelength_m=487.0e-9,
            ramp=ramp,
            dxdt2z=0.0,
        )
        lensing_trap = AstigmaticAODTrap(
            waist_radial_m=500.0e-9,
            wavelength_m=487.0e-9,
            ramp=ramp,
            dxdt2z=0.025 / 4.0 / 650.0,
        )
        cyl_result = simulate_traps([cyl_trap], common_config)
        lensing_result = simulate_traps([lensing_trap], common_config)
        self.assertGreater(
            lensing_result.mean_energy_gain_uK, cyl_result.mean_energy_gain_uK
        )



class GriddedTrapTests(unittest.TestCase):
    def test_gridded_trap_matches_source_gaussian(self):
        source = GaussianTrap(
            center_m=[1.0e-6, -0.5e-6, 0.0],
            waist_radial_m=float(um(1.0)),
            waist_axial_m=float(um(4.0)),
            depth_uK=200.0,
        )
        half = np.array([3.0e-6, 3.0e-6, 1.2e-5])
        bounds = np.stack([-half, half], axis=-1)
        grid = GriddedTrap.from_trap(source, bounds_m=bounds, shape=(81, 81, 121))

        rng = np.random.default_rng(0)
        offsets = rng.uniform(-0.6, 0.6, size=(200, 3)) * half
        query = offsets + source.center_at(0.0)

        u_source = source.potential(query)
        u_grid = grid.potential(query)
        scale = float(np.max(np.abs(u_source))) + 1.0e-30
        np.testing.assert_allclose(u_grid, u_source, atol=5.0e-3 * scale)

        f_source = source.force(query)
        f_grid = grid.force(query)
        f_scale = float(np.max(np.abs(f_source))) + 1.0e-30
        np.testing.assert_allclose(f_grid, f_source, atol=2.0e-2 * f_scale)

        far = source.center_at(0.0) + np.array(
            [[10.0e-6, 0.0, 0.0], [0.0, 0.0, 50.0e-6]]
        )
        np.testing.assert_allclose(grid.potential(far), np.zeros(2))
        np.testing.assert_allclose(grid.force(far), np.zeros((2, 3)))

    def test_tricubic_beats_trilinear_on_smooth_field(self):
        source = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(1.0)),
            waist_axial_m=float(um(4.0)),
            depth_uK=200.0,
        )
        half = np.array([3.0e-6, 3.0e-6, 1.2e-5])
        bounds = np.stack([-half, half], axis=-1)
        # A deliberately coarse grid where the trilinear error is visible.
        shape = (21, 21, 31)
        cubic = GriddedTrap.from_trap(source, bounds, shape, interpolation="tricubic")
        linear = GriddedTrap.from_trap(source, bounds, shape, interpolation="trilinear")

        rng = np.random.default_rng(2)
        offsets = rng.uniform(-0.5, 0.5, size=(400, 3)) * half
        u_ref = source.potential(offsets)
        f_ref = source.force(offsets)

        u_err_cubic = np.max(np.abs(cubic.potential(offsets) - u_ref))
        u_err_linear = np.max(np.abs(linear.potential(offsets) - u_ref))
        f_err_cubic = np.max(np.linalg.norm(cubic.force(offsets) - f_ref, axis=-1))
        f_err_linear = np.max(np.linalg.norm(linear.force(offsets) - f_ref, axis=-1))

        # Tricubic should be at least ~4x better on this smooth field at this grid.
        self.assertLess(u_err_cubic, 0.25 * u_err_linear)
        self.assertLess(f_err_cubic, 0.5 * f_err_linear)

    def test_gridded_trap_with_ramp_translates_rigidly(self):
        source = GaussianTrap(
            center_m=[0.0, 0.0, 0.0],
            waist_radial_m=float(um(0.8)),
            waist_axial_m=float(um(3.0)),
            depth_uK=150.0,
        )
        half = np.array([2.5e-6, 2.5e-6, 1.0e-5])
        bounds = np.stack([-half, half], axis=-1)
        ramp = RampSequence(
            times_s=[0.0, 1.0e-3],
            centers_m=[[0.0, 0.0, 0.0], [4.0e-6, 0.0, 0.0]],
            depths_uK=[150.0, 150.0],
            position_profile=QUINTIC_MIN_JERK,
        )
        grid = GriddedTrap.from_trap(
            source, bounds_m=bounds, shape=(61, 61, 61), ramp=ramp
        )

        # At t = end of ramp the trap center should be (4 um, 0, 0); the
        # gridded potential at that center should equal -depth_joule.
        t_end = 1.0e-3
        center_t = grid.center_at(t_end)
        np.testing.assert_allclose(center_t, [4.0e-6, 0.0, 0.0], atol=1.0e-15)
        u_at_center = grid.potential(center_t, time_s=t_end)
        self.assertAlmostEqual(
            float(u_at_center), -source.depth_joule, delta=1.0e-3 * abs(source.depth_joule)
        )

        # A point 4 um offset from the new center matches the source at the
        # equivalent local offset (rigid-translation invariance).
        offset = np.array([0.4e-6, 0.3e-6, -1.0e-6])
        u_grid = grid.potential(center_t + offset, time_s=t_end)
        u_source_local = source.potential(offset)
        scale = abs(float(u_source_local)) + 1.0e-30
        self.assertAlmostEqual(float(u_grid), float(u_source_local), delta=5.0e-3 * scale)


if __name__ == "__main__":
    unittest.main()
