import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from atommc.constants import (  # noqa: E402
    BOLTZMANN_CONSTANT_J_PER_K,
    RB85_MASS_KG,
    RB87_MASS_KG,
)
from atommc.ensemble import (  # noqa: E402
    EffusiveBeam,
    HarmonicTrapCloud,
    ThermalCloud,
)
from atommc import (  # noqa: E402
    AtomSystem,
    GaussianTrap,
    RB85_D2,
    SimulationConfig,
    simulate,
    um,
)


class ThermalCloudTests(unittest.TestCase):
    def test_gaussian_cloud_moments_and_drift(self) -> None:
        rng = np.random.default_rng(0)
        cloud = ThermalCloud(
            temperature_uK=200.0,
            sigma_m=(1.0e-3, 2.0e-3, 3.0e-3),
            center_m=(1.0e-3, 0.0, 0.0),
            mean_velocity_m_per_s=(0.0, 0.5, 0.0),
            mass_kg=RB87_MASS_KG,
        )
        sample = cloud.sample(200000, rng)
        self.assertEqual(sample.positions_m.shape, (200000, 3))
        self.assertEqual(sample.weight, 1.0)
        np.testing.assert_allclose(
            sample.positions_m.std(axis=0), [1e-3, 2e-3, 3e-3], rtol=0.05
        )
        np.testing.assert_allclose(
            sample.positions_m.mean(axis=0), [1e-3, 0.0, 0.0], atol=2e-5
        )
        # Velocity spread matches Maxwell-Boltzmann sigma = sqrt(kT/m).
        sigma_v = np.sqrt(
            BOLTZMANN_CONSTANT_J_PER_K * 200e-6 / RB87_MASS_KG
        )
        np.testing.assert_allclose(
            sample.velocities_m_per_s.std(axis=0), sigma_v, rtol=0.05
        )
        np.testing.assert_allclose(
            sample.velocities_m_per_s.mean(axis=0), [0.0, 0.5, 0.0], atol=5e-4
        )

    def test_uniform_sphere_fill(self) -> None:
        rng = np.random.default_rng(1)
        cloud = ThermalCloud(temperature_uK=0.0, sphere_radius_m=2.0e-3)
        sample = cloud.sample(200000, rng)
        radii = np.linalg.norm(sample.positions_m, axis=1)
        self.assertLessEqual(radii.max(), 2.0e-3 + 1e-9)
        # Uniform ball: <r>/R = 3/4.
        self.assertAlmostEqual((radii / 2.0e-3).mean(), 0.75, places=2)

    def test_requires_exactly_one_spatial_specifier(self) -> None:
        with self.assertRaises(ValueError):
            ThermalCloud(temperature_uK=1.0)
        with self.assertRaises(ValueError):
            ThermalCloud(temperature_uK=1.0, sigma_m=1e-3, sphere_radius_m=1e-3)


class HarmonicTrapCloudTests(unittest.TestCase):
    def test_matches_trap_hessian_covariance(self) -> None:
        rng = np.random.default_rng(2)
        trap = GaussianTrap(
            center_m=(0.0, 0.0, 0.0),
            depth_uK=1000.0,
            waist_radial_m=um(1.0),
            waist_axial_m=um(5.0),
        )
        cloud = HarmonicTrapCloud(
            forces=trap, temperature_uK=50.0, mass_kg=RB87_MASS_KG
        )
        sample = cloud.sample(100000, rng)
        # Radial spread tighter than axial (stiffer radial curvature).
        stds = sample.positions_m.std(axis=0)
        self.assertLess(stds[0], stds[2])
        sigma_v = np.sqrt(BOLTZMANN_CONSTANT_J_PER_K * 50e-6 / RB87_MASS_KG)
        np.testing.assert_allclose(
            sample.velocities_m_per_s.std(axis=0), sigma_v, rtol=0.05
        )


class EffusiveBeamTests(unittest.TestCase):
    def _beam(self, **overrides) -> EffusiveBeam:
        params = dict(
            temperature_K=600.0,
            mass_kg=RB85_MASS_KG,
            aperture_radius_m=1.0e-3,
            v_min_m_per_s=0.0,
            v_max_m_per_s=40.0,
            direction=(0.0, 0.0, 1.0),
            launch_center_m=(0.0, 0.0, -30.0e-3),
        )
        params.update(overrides)
        return EffusiveBeam(**params)

    def test_flux_speed_distribution_moments(self) -> None:
        # For v_max << sigma the Gaussian factor ~ 1, so the flux law is the
        # pure v^3 envelope on [0, v_max]: <v> = 4/5 v_max, <v^2> = 2/3 v_max^2.
        rng = np.random.default_rng(3)
        beam = self._beam(v_max_m_per_s=40.0)
        speeds = np.linalg.norm(beam.sample(300000, rng).velocities_m_per_s, axis=1)
        self.assertLessEqual(speeds.max(), 40.0 + 1e-6)
        self.assertAlmostEqual(speeds.mean() / 40.0, 0.8, places=2)
        self.assertAlmostEqual(np.mean(speeds**2) / 40.0**2, 2.0 / 3.0, places=2)

    def test_window_fraction_matches_analytic_cdf(self) -> None:
        beam = self._beam(v_min_m_per_s=5.0, v_max_m_per_s=45.0)
        sigma = np.sqrt(
            BOLTZMANN_CONSTANT_J_PER_K * 600.0 / RB85_MASS_KG
        )

        def cdf(v: float) -> float:
            u = v**2 / (2.0 * sigma**2)
            return 1.0 - np.exp(-u) * (1.0 + u)

        expected = cdf(45.0) - cdf(5.0)
        self.assertAlmostEqual(beam.window_fraction(), expected, places=12)
        rng = np.random.default_rng(4)
        self.assertAlmostEqual(beam.sample(10, rng).weight, expected, places=12)

    def test_speeds_respect_v_min(self) -> None:
        rng = np.random.default_rng(5)
        beam = self._beam(v_min_m_per_s=20.0, v_max_m_per_s=40.0)
        speeds = np.linalg.norm(beam.sample(50000, rng).velocities_m_per_s, axis=1)
        self.assertGreaterEqual(speeds.min(), 20.0 - 1e-6)
        self.assertLessEqual(speeds.max(), 40.0 + 1e-6)

    def test_positions_lie_on_aperture_disc(self) -> None:
        rng = np.random.default_rng(6)
        beam = self._beam(aperture_radius_m=1.5e-3)
        pos = beam.sample(50000, rng).positions_m
        radial = np.hypot(pos[:, 0], pos[:, 1])
        self.assertLessEqual(radial.max(), 1.5e-3 + 1e-9)
        np.testing.assert_allclose(pos[:, 2], -30.0e-3, atol=1e-12)

    def test_cosine_law_angular_statistics(self) -> None:
        # Full-hemisphere Lambert law: <cos theta> = 2/3 over dN ~ cos theta dOmega.
        rng = np.random.default_rng(7)
        beam = self._beam(divergence_half_angle_rad=np.pi / 2.0, angular="cosine")
        vel = beam.sample(300000, rng).velocities_m_per_s
        cos_theta = vel[:, 2] / np.linalg.norm(vel, axis=1)
        self.assertGreater(cos_theta.min(), -1e-9)  # forward hemisphere only
        self.assertAlmostEqual(cos_theta.mean(), 2.0 / 3.0, places=2)

    def test_uniform_solid_angle_angular_statistics(self) -> None:
        # Uniform in solid angle over a hemisphere: <cos theta> = 1/2.
        rng = np.random.default_rng(8)
        beam = self._beam(divergence_half_angle_rad=np.pi / 2.0, angular="uniform")
        vel = beam.sample(300000, rng).velocities_m_per_s
        cos_theta = vel[:, 2] / np.linalg.norm(vel, axis=1)
        self.assertAlmostEqual(cos_theta.mean(), 0.5, places=2)

    def test_off_axis_direction_is_normalized(self) -> None:
        rng = np.random.default_rng(9)
        beam = self._beam(direction=(0.0, 3.0, 4.0), divergence_half_angle_rad=0.0)
        vel = beam.sample(10000, rng).velocities_m_per_s
        speeds = np.linalg.norm(vel, axis=1)
        # Perfectly collimated: every velocity parallel to the unit axis.
        axis = np.array([0.0, 0.6, 0.8])
        projections = vel @ axis
        np.testing.assert_allclose(projections, speeds, rtol=1e-9)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._beam(v_min_m_per_s=40.0, v_max_m_per_s=40.0)
        with self.assertRaises(ValueError):
            self._beam(temperature_K=0.0)
        with self.assertRaises(ValueError):
            self._beam(angular="banana")
        with self.assertRaises(ValueError):
            self._beam(direction=(0.0, 0.0, 0.0))


class SimulationIntegrationTests(unittest.TestCase):
    def test_initial_source_flows_through_simulate(self) -> None:
        trap = GaussianTrap(
            center_m=(0.0, 0.0, 0.0),
            depth_uK=500.0,
            waist_radial_m=um(50.0),
            waist_axial_m=um(200.0),
        )
        beam = EffusiveBeam(
            temperature_K=600.0,
            mass_kg=RB85_MASS_KG,
            aperture_radius_m=1.0e-3,
            v_min_m_per_s=0.0,
            v_max_m_per_s=40.0,
            launch_center_m=(0.0, 0.0, -100.0e-6),
            divergence_half_angle_rad=5.0e-3,
        )
        config = SimulationConfig(
            initial_temperature_uK=0.0,
            timestep_s=1.0e-6,
            duration_s=1.0e-4,
            ensemble_size=100,
            initial_source=beam,
            reject_initially_lost=False,
            loss_radius_m=50.0e-3,
            random_seed=3,
        )
        result = simulate(AtomSystem(species=RB85_D2, modules=[trap]), config)
        self.assertAlmostEqual(
            result.initial_ensemble_weight, beam.window_fraction(), places=12
        )
        # Beam launched along +z: mean initial vz is positive.
        self.assertGreater(result.initial_velocities_m_per_s[:, 2].mean(), 0.0)

    def test_source_conflicts_with_explicit_arrays(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(
                initial_temperature_uK=0.0,
                timestep_s=1.0e-6,
                duration_s=1.0e-4,
                ensemble_size=2,
                initial_source=ThermalCloud(temperature_uK=1.0, sigma_m=1e-3),
                initial_positions_m=np.zeros((2, 3)),
                reject_initially_lost=False,
            )


if __name__ == "__main__":
    unittest.main()
