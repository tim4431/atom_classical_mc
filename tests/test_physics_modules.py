"""Tests for the Zeeman and dipole-beam conservative physics modules."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atommc import (  # noqa: E402
    AtomSystem,
    BOHR_MAGNETON_J_PER_T,
    DipoleBeamPotential,
    HBAR_J_S,
    QuadrupoleMagneticField,
    RB87_D2,
    SimulationConfig,
    UniformMagneticField,
    ZeemanPotential,
    gauss_per_cm,
    simulate,
    um,
)


def _finite_difference_force(module, positions, step=1e-9):
    positions = np.asarray(positions, dtype=float)
    force = np.zeros_like(positions)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        plus = module.potential(positions + delta)
        minus = module.potential(positions - delta)
        force[..., axis] = -(plus - minus) / (2.0 * step)
    return force


class MagneticJacobianTests(unittest.TestCase):
    def test_uniform_field_jacobian_is_zero(self):
        field = UniformMagneticField(field_T=(1e-4, -2e-4, 5e-5))
        positions = np.array([[0.0, 0.0, 0.0], [1e-3, -2e-3, 4e-4]])
        self.assertTrue(np.allclose(field.jacobian(positions), 0.0))

    def test_quadrupole_jacobian_matches_finite_difference(self):
        field = QuadrupoleMagneticField(
            gradient_T_per_m=gauss_per_cm(12.0),
            center_m=(1e-4, -3e-4, 2e-4),
            axis=(1.0, 1.0, 1.0),
        )
        positions = np.array([[2e-3, 1e-3, -5e-4], [0.0, 4e-3, 1e-3]])
        analytic = field.jacobian(positions)
        numeric = super(QuadrupoleMagneticField, field).jacobian(positions)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-10)

    def test_quadrupole_jacobian_is_traceless_symmetric(self):
        field = QuadrupoleMagneticField(gradient_T_per_m=0.1, axis=(0.0, 1.0, 0.0))
        jac = field.jacobian(np.zeros(3))
        self.assertAlmostEqual(float(np.trace(jac)), 0.0, places=12)
        np.testing.assert_allclose(jac, jac.T)


class ZeemanPotentialTests(unittest.TestCase):
    def setUp(self):
        self.gradient = gauss_per_cm(50.0)
        self.quadrupole = QuadrupoleMagneticField(gradient_T_per_m=self.gradient)
        # Rb87 |F=2, m_F=2>: g_F = 1/2, weak-field seeking.
        self.trap = ZeemanPotential.for_sublevel(self.quadrupole, g_f=0.5, m_f=2.0)

    def test_for_sublevel_moment(self):
        self.assertAlmostEqual(
            self.trap.moment_j_per_t, BOHR_MAGNETON_J_PER_T, delta=1e-30
        )

    def test_potential_is_linear_in_field_magnitude(self):
        r_radial = np.array([1e-3, 0.0, 0.0])
        expected = self.trap.moment_j_per_t * self.gradient * 1e-3
        self.assertAlmostEqual(
            float(self.trap.potential(r_radial)), expected, delta=1e-9 * expected
        )
        # Axial gradient is 2 b'.
        r_axial = np.array([0.0, 0.0, 1e-3])
        self.assertAlmostEqual(
            float(self.trap.potential(r_axial)), 2.0 * expected,
            delta=1e-9 * expected,
        )

    def test_force_matches_finite_difference(self):
        positions = np.array(
            [[1e-3, 2e-3, -1e-3], [-4e-4, 3e-4, 8e-4], [0.0, 1e-3, 0.0]]
        )
        analytic = self.trap.force(positions)
        numeric = _finite_difference_force(self.trap, positions)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-28)

    def test_restoring_force_magnitudes(self):
        # |F| = mu b' radially, 2 mu b' axially, pointing back to the node.
        mu = self.trap.moment_j_per_t
        f_radial = self.trap.force(np.array([1e-3, 0.0, 0.0]))
        self.assertAlmostEqual(
            float(f_radial[0]), -mu * self.gradient, delta=1e-9 * mu * self.gradient
        )
        f_axial = self.trap.force(np.array([0.0, 0.0, 1e-3]))
        self.assertAlmostEqual(
            float(f_axial[2]),
            -2.0 * mu * self.gradient,
            delta=1e-9 * mu * self.gradient,
        )

    def test_force_is_zero_at_field_node(self):
        np.testing.assert_allclose(self.trap.force(np.zeros(3)), 0.0)

    def test_anti_trapped_sublevel_is_expelled(self):
        anti = ZeemanPotential.for_sublevel(self.quadrupole, g_f=0.5, m_f=-2.0)
        force = anti.force(np.array([1e-3, 0.0, 0.0]))
        self.assertGreater(float(force[0]), 0.0)

    def test_bias_field_shifts_quadrupole_node(self):
        # A uniform bias along x moves the |B| = 0 point; the combined
        # magnitude at the shifted node must vanish (|B1+B2| != |B1|+|B2|).
        bias_t = 1e-4
        shifted = ZeemanPotential(
            fields=(
                self.quadrupole,
                UniformMagneticField(field_T=(bias_t, 0.0, 0.0)),
            ),
            moment_j_per_t=BOHR_MAGNETON_J_PER_T,
        )
        node = np.array([-bias_t / self.gradient, 0.0, 0.0])
        self.assertAlmostEqual(float(shifted.potential(node)), 0.0, delta=1e-40)

    def test_requires_at_least_one_field(self):
        with self.assertRaises(ValueError):
            ZeemanPotential(fields=())


class ZeemanSimulationTests(unittest.TestCase):
    def _run(self, m_f):
        # 100 uK cloud, 50 G/cm quadrupole. Trapped sublevel stays inside
        # the loss radius over 10 ms; the anti-trapped one is expelled.
        quadrupole = QuadrupoleMagneticField(gradient_T_per_m=gauss_per_cm(50.0))
        zeeman = ZeemanPotential.for_sublevel(quadrupole, g_f=0.5, m_f=m_f)
        system = AtomSystem(species=RB87_D2, modules=[zeeman])
        config = SimulationConfig(
            initial_temperature_uK=100.0,
            timestep_s=2.0e-6,
            duration_s=1.0e-2,
            ensemble_size=64,
            random_seed=9,
            initial_cloud_sigma_m=2.0e-4,
            loss_radius_m=3.0e-3,
            reject_initially_lost=False,
            energy_loss="off",
        )
        return simulate(system, config)

    def test_trapped_sublevel_is_confined(self):
        result = self._run(m_f=2.0)
        self.assertGreater(result.survival_probability, 0.9)
        final_rms = float(
            np.sqrt(np.mean(np.sum(result.final_positions_m**2, axis=-1)))
        )
        self.assertLess(final_rms, 1.5e-3)

    def test_anti_trapped_sublevel_is_expelled(self):
        result = self._run(m_f=-2.0)
        self.assertGreater(result.loss_fraction, 0.5)


class DipoleBeamPotentialTests(unittest.TestCase):
    def setUp(self):
        self.beam = DipoleBeamPotential(
            species=RB87_D2,
            power_w=5.0e-3,
            waist_m=um(1.0),
            wavelength_m=8.5e-7,
        )

    def test_red_detuned_beam_is_attractive(self):
        self.assertLess(self.beam.detuning_rad_s, 0.0)
        self.assertLess(float(self.beam.potential(np.zeros(3))), 0.0)

    def test_depth_matches_far_detuned_asymptote(self):
        gamma = RB87_D2.linewidth_rad_s
        delta = self.beam.detuning_rad_s
        asymptote = abs(
            HBAR_J_S * gamma**2 * self.beam.peak_saturation / (8.0 * delta)
        )
        self.assertAlmostEqual(
            self.beam.depth_j, asymptote, delta=1e-6 * asymptote
        )
        self.assertGreater(self.beam.depth_uK, 0.0)

    def test_force_matches_finite_difference(self):
        rng = np.random.default_rng(7)
        positions = rng.normal(
            scale=[0.5 * self.beam.waist_m] * 2 + [0.5 * self.beam.rayleigh_length_m],
            size=(6, 3),
        )
        analytic = self.beam.force(positions)
        numeric = _finite_difference_force(
            self.beam, positions, step=1e-4 * self.beam.waist_m
        )
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-30)

    def test_harmonic_frequencies_match_gaussian_beam_theory(self):
        mass = RB87_D2.mass_kg
        depth = self.beam.depth_j
        omega_r_expected = np.sqrt(4.0 * depth / (mass * self.beam.waist_m**2))
        omega_z_expected = np.sqrt(
            2.0 * depth / (mass * self.beam.rayleigh_length_m**2)
        )
        hessian = self.beam.hessian(np.zeros(3))
        omegas = np.sqrt(np.linalg.eigvalsh(hessian) / mass)
        np.testing.assert_allclose(
            np.sort(omegas),
            np.sort([omega_z_expected, omega_r_expected, omega_r_expected]),
            rtol=1e-3,
        )

    def test_tilted_beam_force_matches_finite_difference(self):
        beam = DipoleBeamPotential(
            species=RB87_D2,
            power_w=2.0e-3,
            waist_m=um(1.5),
            wavelength_m=1.064e-6,
            focus_m=(um(2.0), -um(1.0), um(3.0)),
            direction=(1.0, 1.0, 0.5),
        )
        rng = np.random.default_rng(11)
        positions = np.asarray(beam.focus_m) + rng.normal(
            scale=um(1.0), size=(6, 3)
        )
        analytic = beam.force(positions)
        numeric = _finite_difference_force(beam, positions, step=1e-4 * beam.waist_m)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-30)

    def test_near_resonant_wavelength_rejected(self):
        with self.assertRaises(ValueError):
            DipoleBeamPotential(
                species=RB87_D2,
                power_w=1.0e-6,
                waist_m=um(1.0),
                wavelength_m=RB87_D2.wavelength_m * (1.0 + 1e-9),
            )

    def test_peak_scattering_rate_is_small_when_far_detuned(self):
        rate = self.beam.peak_scattering_rate_per_s
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1e-6 * RB87_D2.linewidth_rad_s)


if __name__ == "__main__":
    unittest.main()
