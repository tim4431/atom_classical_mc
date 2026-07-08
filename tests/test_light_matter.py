import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis import kinetic_temperature_time_series_uK  # noqa: E402
from src.constants import (  # noqa: E402
    BOHR_MAGNETON_J_PER_T,
    HBAR_J_S,
)
from src.fields import (  # noqa: E402
    QuadrupoleMagneticField,
    UniformMagneticField,
    total_magnetic_field,
)
from src.internal_state import (  # noqa: E402
    AdiabaticSteadyState,
    RateEquationPopulations,
    ScatteringEvents,
    StochasticJumpState,
    sample_recoil_velocity_kicks,
)
from src.laser import LaserBeam, six_beam_mot  # noqa: E402
from src.light_matter import (  # noqa: E402
    LightMatterSystem,
    polarization_fractions,
)
from src.simulation import SimulationConfig, run_simulation  # noqa: E402
from src.species import RB85_D2, RB87_D2  # noqa: E402
from src.trap import GaussianTrap  # noqa: E402
from src.units import gauss_per_cm, um  # noqa: E402


def _molasses_system(species=RB85_D2, detuning_hz=None, saturation=2.0):
    if detuning_hz is None:
        detuning_hz = -0.5 * species.linewidth_rad_s / (2.0 * np.pi)
    beams = six_beam_mot(detuning_hz=detuning_hz, saturation=saturation)
    return LightMatterSystem(species=species, beams=beams)


def _mot_system(species=RB85_D2, gradient_g_per_cm=10.0, saturation=2.0):
    detuning_hz = -1.0 * species.linewidth_rad_s / (2.0 * np.pi)
    beams = six_beam_mot(detuning_hz=detuning_hz, saturation=saturation)
    quad = QuadrupoleMagneticField(
        gradient_T_per_m=float(gauss_per_cm(gradient_g_per_cm))
    )
    return LightMatterSystem(species=species, beams=beams, magnetic_fields=[quad])


class SpeciesTests(unittest.TestCase):
    def test_rb85_and_rb87_effective_moment_is_one_bohr_magneton(self):
        for species in (RB85_D2, RB87_D2):
            self.assertAlmostEqual(
                species.mu_eff_j_per_t / BOHR_MAGNETON_J_PER_T, 1.0, places=12
            )

    def test_rb85_recoil_and_doppler_scales(self):
        # Steck Rb85: v_rec ~ 6.02 mm/s, T_D ~ 145.6 uK.
        self.assertAlmostEqual(
            RB85_D2.recoil_velocity_m_per_s, 6.02e-3, delta=0.05e-3
        )
        self.assertAlmostEqual(RB85_D2.doppler_temperature_uK, 145.6, delta=1.0)

    def test_rb85_lighter_than_rb87(self):
        self.assertLess(RB85_D2.mass_kg, RB87_D2.mass_kg)


class MagneticFieldTests(unittest.TestCase):
    def test_quadrupole_field_shape(self):
        quad = QuadrupoleMagneticField(gradient_T_per_m=0.1)
        b = quad.vector(np.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0e-3]]))
        np.testing.assert_allclose(b[0], [1.0e-4, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(b[1], [0.0, 0.0, -2.0e-4], atol=1e-12)

    def test_quadrupole_is_divergence_free_sum(self):
        quad = QuadrupoleMagneticField(gradient_T_per_m=0.25)
        # Trace of the gradient matrix must vanish: b' + b' - 2 b' = 0.
        eps = 1.0e-6
        div = 0.0
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = eps
            plus = quad.vector(delta)[axis]
            minus = quad.vector(-delta)[axis]
            div += (plus - minus) / (2.0 * eps)
        self.assertAlmostEqual(div, 0.0, places=9)

    def test_uniform_plus_quadrupole_sums(self):
        fields = [
            UniformMagneticField(field_T=(0.0, 0.0, 1.0e-4)),
            QuadrupoleMagneticField(gradient_T_per_m=0.1),
        ]
        b = total_magnetic_field(fields, np.zeros(3))
        np.testing.assert_allclose(b, [0.0, 0.0, 1.0e-4], atol=1e-15)


class LaserBeamTests(unittest.TestCase):
    def test_direction_normalized(self):
        beam = LaserBeam(direction=(3.0, 0.0, 4.0))
        np.testing.assert_allclose(np.linalg.norm(beam.direction), 1.0)

    def test_gaussian_profile_falls_off(self):
        beam = LaserBeam(direction=(0.0, 0.0, 1.0), saturation=4.0, waist_m=1.0e-2)
        on_axis = beam.saturation_at(np.array([0.0, 0.0, 0.5]))
        at_waist = beam.saturation_at(np.array([1.0e-2, 0.0, 0.0]))
        self.assertAlmostEqual(float(on_axis), 4.0)
        self.assertAlmostEqual(float(at_waist), 4.0 * np.exp(-2.0), places=10)

    def test_six_beam_mot_geometry(self):
        beams = six_beam_mot(detuning_hz=-6.0e6, saturation=1.0)
        self.assertEqual(len(beams), 6)
        total = np.sum([b.direction for b in beams], axis=0)
        np.testing.assert_allclose(total, np.zeros(3), atol=1e-12)


class PolarizationTests(unittest.TestCase):
    def test_fractions_sum_to_one(self):
        cos_theta = np.linspace(-1.0, 1.0, 21)
        for helicity in (-1.0, -0.3, 0.0, 0.7, 1.0):
            f_plus, f_pi, f_minus = polarization_fractions(helicity, cos_theta)
            np.testing.assert_allclose(f_plus + f_pi + f_minus, 1.0, atol=1e-12)
            self.assertTrue(np.all(f_plus >= 0.0))
            self.assertTrue(np.all(f_pi >= 0.0))
            self.assertTrue(np.all(f_minus >= 0.0))

    def test_pure_sigma_plus_along_field(self):
        f_plus, f_pi, f_minus = polarization_fractions(1.0, np.array([1.0]))
        np.testing.assert_allclose([f_plus[0], f_pi[0], f_minus[0]], [1.0, 0.0, 0.0])

    def test_helicity_flips_when_counterpropagating(self):
        f_plus, _, f_minus = polarization_fractions(1.0, np.array([-1.0]))
        np.testing.assert_allclose([f_plus[0], f_minus[0]], [0.0, 1.0])


class RateEngineTests(unittest.TestCase):
    def test_resonant_rate_matches_two_level_formula(self):
        species = RB85_D2
        beam = LaserBeam(direction=(0.0, 0.0, 1.0), detuning_hz=0.0, saturation=1.0)
        system = LightMatterSystem(species=species, beams=[beam])
        w = system.stimulated_rates(np.zeros((1, 3)), np.zeros((1, 3)))
        # W = Gamma/2 * s on resonance with B = 0.
        self.assertAlmostEqual(float(w[0, 0]), 0.5 * species.linewidth_rad_s, delta=1.0)

    def test_doppler_shift_moves_resonance(self):
        species = RB85_D2
        gamma = species.linewidth_rad_s
        detuning_hz = -gamma / (2.0 * np.pi)  # one linewidth red
        beam = LaserBeam(direction=(0.0, 0.0, 1.0), detuning_hz=detuning_hz, saturation=1.0)
        system = LightMatterSystem(species=species, beams=[beam])
        # Atom moving against the beam is Doppler-shifted toward resonance.
        v_resonant = -gamma / species.wavenumber_rad_per_m
        w_moving = system.stimulated_rates(
            np.zeros((1, 3)), np.array([[0.0, 0.0, v_resonant]])
        )
        w_static = system.stimulated_rates(np.zeros((1, 3)), np.zeros((1, 3)))
        self.assertGreater(float(w_moving[0, 0]), float(w_static[0, 0]))
        self.assertAlmostEqual(float(w_moving[0, 0]), 0.5 * gamma, delta=1.0)

    def test_zeeman_shift_moves_resonance(self):
        species = RB85_D2
        gamma = species.linewidth_rad_s
        detuning_hz = -gamma / (2.0 * np.pi)
        # sigma- beam along +z; B along +z shifts the q=-1 component by
        # +mu B / hbar, moving a red-detuned laser toward resonance.
        beam = LaserBeam(
            direction=(0.0, 0.0, 1.0),
            detuning_hz=detuning_hz,
            saturation=1.0,
            helicity=-1.0,
        )
        b_resonant = gamma * HBAR_J_S / species.mu_eff_j_per_t
        system_b = LightMatterSystem(
            species=species,
            beams=[beam],
            magnetic_fields=[UniformMagneticField(field_T=(0.0, 0.0, b_resonant))],
        )
        system_0 = LightMatterSystem(species=species, beams=[beam])
        w_b = system_b.stimulated_rates(np.zeros((1, 3)), np.zeros((1, 3)))
        w_0 = system_0.stimulated_rates(np.zeros((1, 3)), np.zeros((1, 3)))
        self.assertGreater(float(w_b[0, 0]), float(w_0[0, 0]))
        self.assertAlmostEqual(float(w_b[0, 0]), 0.5 * gamma, delta=1.0)

    def test_molasses_force_damps_velocity(self):
        system = _molasses_system()
        v = np.array([[3.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]])
        f = system.mean_radiation_force(np.zeros((3, 3)), v)
        # Force must oppose velocity component-wise.
        self.assertLess(f[0, 0], 0.0)
        self.assertGreater(f[1, 1], 0.0)
        self.assertLess(f[2, 2], 0.0)

    def test_mot_force_restores_position(self):
        system = _mot_system()
        r = 2.0e-3 * np.eye(3)  # displaced along +x, +y, +z
        f = system.mean_radiation_force(r, np.zeros((3, 3)))
        for axis in range(3):
            self.assertLess(
                f[axis, axis],
                0.0,
                msg=f"MOT force must restore along axis {axis}",
            )

    def test_molasses_force_zero_at_rest(self):
        system = _molasses_system()
        f = system.mean_radiation_force(np.zeros((1, 3)), np.zeros((1, 3)))
        np.testing.assert_allclose(f, np.zeros((1, 3)), atol=1e-30)


class InternalStateBackendTests(unittest.TestCase):
    def _rates(self, n_atoms, w_value, n_beams=2):
        return np.full((n_atoms, n_beams), w_value, dtype=float)

    def test_steady_state_population(self):
        model = AdiabaticSteadyState()
        rng = np.random.default_rng(0)
        gamma = RB85_D2.linewidth_rad_s
        w = self._rates(4, 0.25 * gamma)
        state = model.initialize(4, rng)
        state, _ = model.step(state, w, gamma, 1.0e-7, rng)
        expected = 0.5 * gamma / (gamma + 2.0 * 0.5 * gamma)
        np.testing.assert_allclose(model.excited_fraction(state), expected)

    def test_rate_equation_relaxes_to_steady_state(self):
        model = RateEquationPopulations()
        rng = np.random.default_rng(0)
        gamma = RB85_D2.linewidth_rad_s
        w = self._rates(3, 0.25 * gamma)
        state = model.initialize(3, rng)
        for _ in range(200):
            state, _ = model.step(state, w, gamma, 1.0e-8, rng)
        expected = 0.5 * gamma / (gamma + 2.0 * 0.5 * gamma)
        np.testing.assert_allclose(model.excited_fraction(state), expected, rtol=1e-6)

    def test_stochastic_jump_rejects_large_timestep(self):
        model = StochasticJumpState()
        rng = np.random.default_rng(0)
        gamma = RB85_D2.linewidth_rad_s
        state = model.initialize(2, rng)
        with self.assertRaises(ValueError):
            model.step(state, self._rates(2, 0.1 * gamma), gamma, 1.0e-6, rng)

    def test_stochastic_jump_mean_population_matches_steady_state(self):
        model = StochasticJumpState()
        rng = np.random.default_rng(1)
        gamma = RB85_D2.linewidth_rad_s
        n_atoms = 2000
        w = self._rates(n_atoms, 0.25 * gamma)
        state = model.initialize(n_atoms, rng)
        dt = 0.02 / gamma
        # Equilibrate for many lifetimes, then time-average.
        for _ in range(500):
            state, _ = model.step(state, w, gamma, dt, rng)
        samples = []
        for _ in range(200):
            state, _ = model.step(state, w, gamma, dt, rng)
            samples.append(np.mean(model.excited_fraction(state)))
        expected = 0.5 * gamma / (gamma + 2.0 * 0.5 * gamma)
        self.assertAlmostEqual(float(np.mean(samples)), expected, delta=0.02)

    def test_backend_scattering_rates_agree_in_cw_conditions(self):
        rng = np.random.default_rng(2)
        gamma = RB85_D2.linewidth_rad_s
        n_atoms = 500
        w = self._rates(n_atoms, 0.2 * gamma)
        dt = 1.0e-7
        n_steps = 100
        totals = {}
        for model in (AdiabaticSteadyState(), RateEquationPopulations()):
            state = model.initialize(n_atoms, rng)
            scattered = 0
            for _ in range(n_steps):
                state, events = model.step(state, w, gamma, dt, rng)
                scattered += int(np.sum(events.total_scattered))
            totals[model.name] = scattered / (n_atoms * n_steps * dt)
        ratio = totals["adiabatic_steady_state"] / totals["rate_equation"]
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_photon_budget_balances_in_steady_state(self):
        model = AdiabaticSteadyState()
        rng = np.random.default_rng(3)
        gamma = RB85_D2.linewidth_rad_s
        n_atoms = 2000
        w = self._rates(n_atoms, 0.3 * gamma)
        state = model.initialize(n_atoms, rng)
        absorbed = emitted = 0
        for _ in range(50):
            state, events = model.step(state, w, gamma, 1.0e-7, rng)
            absorbed += int(np.sum(events.absorbed_per_beam))
            emitted += int(
                np.sum(events.stimulated_per_beam) + np.sum(events.spontaneous)
            )
        self.assertAlmostEqual(absorbed / emitted, 1.0, delta=0.02)


class RecoilTests(unittest.TestCase):
    def test_directed_kicks(self):
        rng = np.random.default_rng(0)
        dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        events = ScatteringEvents(
            absorbed_per_beam=np.array([[3, 0], [0, 2]], dtype=np.int64),
            stimulated_per_beam=np.array([[1, 0], [0, 0]], dtype=np.int64),
            spontaneous=np.array([0, 0], dtype=np.int64),
        )
        kicks = sample_recoil_velocity_kicks(events, dirs, 6.0e-3, rng)
        np.testing.assert_allclose(kicks[0], [2 * 6.0e-3, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(kicks[1], [0.0, 2 * 6.0e-3, 0.0], atol=1e-15)

    def test_spontaneous_kicks_have_recoil_magnitude_and_zero_mean(self):
        rng = np.random.default_rng(4)
        n_atoms = 4000
        dirs = np.array([[1.0, 0.0, 0.0]])
        events = ScatteringEvents(
            absorbed_per_beam=np.zeros((n_atoms, 1), dtype=np.int64),
            stimulated_per_beam=np.zeros((n_atoms, 1), dtype=np.int64),
            spontaneous=np.ones(n_atoms, dtype=np.int64),
        )
        v_rec = 6.0e-3
        kicks = sample_recoil_velocity_kicks(events, dirs, v_rec, rng)
        np.testing.assert_allclose(np.linalg.norm(kicks, axis=-1), v_rec, rtol=1e-12)
        mean_kick = np.mean(kicks, axis=0)
        self.assertLess(float(np.linalg.norm(mean_kick)), 3.0 * v_rec / np.sqrt(n_atoms) * 3)


class ScatteringSimulationTests(unittest.TestCase):
    def test_molasses_cools_hot_cloud_toward_doppler_scale(self):
        system = _molasses_system(saturation=1.0)
        config = SimulationConfig(
            initial_temperature_uK=2000.0,
            timestep_s=2.0e-7,
            duration_s=6.0e-3,
            ensemble_size=200,
            mass_kg=RB85_D2.mass_kg,
            initial_cloud_sigma_m=1.0e-4,
            random_seed=7,
        )
        result = run_simulation([], config, scattering=system)
        self.assertLess(
            result.final_temperature_uK_all, 0.3 * result.initial_temperature_uK_all
        )
        # Should approach the Doppler scale but not go below the recoil scale.
        self.assertLess(result.final_temperature_uK_all, 800.0)
        self.assertGreater(result.final_temperature_uK_all, 10.0)
        self.assertGreater(float(np.mean(result.scattered_photons)), 100.0)

    def test_mot_confines_cloud(self):
        system = _mot_system(gradient_g_per_cm=15.0, saturation=2.0)
        config = SimulationConfig(
            initial_temperature_uK=500.0,
            timestep_s=2.0e-7,
            duration_s=10.0e-3,
            ensemble_size=100,
            mass_kg=RB85_D2.mass_kg,
            initial_cloud_sigma_m=0.5e-3,
            random_seed=11,
            store_trajectories=True,
            trajectory_stride=500,
        )
        result = run_simulation([], config, scattering=system)
        initial_rms = float(
            np.sqrt(np.mean(np.sum(result.initial_positions_m**2, axis=-1)))
        )
        final_rms = float(
            np.sqrt(np.mean(np.sum(result.final_positions_m**2, axis=-1)))
        )
        self.assertLess(final_rms, initial_rms)
        self.assertEqual(result.survival_probability, 1.0)
        temperatures = kinetic_temperature_time_series_uK(
            result, mass_kg=RB85_D2.mass_kg
        )
        self.assertEqual(temperatures.shape, result.trajectory_times_s.shape)
        self.assertLess(temperatures[-1], temperatures[0])

    def test_hot_atoms_without_light_fly_away(self):
        # Sanity check the loss criterion with a far-detuned dark system
        # (negligible scattering): free flight out of the loss radius.
        system = _molasses_system(detuning_hz=-5.0e9, saturation=1.0e-6)
        config = SimulationConfig(
            initial_temperature_uK=1000.0,
            timestep_s=1.0e-6,
            duration_s=5.0e-3,
            ensemble_size=50,
            mass_kg=RB85_D2.mass_kg,
            initial_cloud_sigma_m=1.0e-5,
            loss_radius_m=1.0e-3,
            random_seed=3,
        )
        result = run_simulation([], config, scattering=system)
        self.assertGreater(result.loss_fraction, 0.5)
        self.assertLess(float(np.mean(result.scattered_photons)), 1.0)

    def test_stochastic_backend_runs_and_cools(self):
        # dt must satisfy (Gamma + W_tot) dt < 0.1 for the jump backend.
        system = _molasses_system(saturation=0.5)
        config = SimulationConfig(
            initial_temperature_uK=2000.0,
            timestep_s=1.2e-9,
            duration_s=0.1e-3,
            ensemble_size=40,
            mass_kg=RB85_D2.mass_kg,
            initial_cloud_sigma_m=1.0e-4,
            random_seed=5,
        )
        result = run_simulation(
            [], config, scattering=system, internal_model=StochasticJumpState()
        )
        self.assertLess(
            result.final_temperature_uK_all, result.initial_temperature_uK_all
        )
        self.assertTrue(np.all(result.final_excited_fraction >= 0.0))
        self.assertTrue(np.all(result.final_excited_fraction <= 1.0))

    def test_rate_equation_backend_matches_adiabatic_temperature(self):
        system = _molasses_system(saturation=1.0)
        base = dict(
            initial_temperature_uK=1500.0,
            timestep_s=2.0e-7,
            duration_s=4.0e-3,
            ensemble_size=150,
            mass_kg=RB85_D2.mass_kg,
            initial_cloud_sigma_m=1.0e-4,
            random_seed=13,
        )
        result_a = run_simulation(
            [],
            SimulationConfig(**base),
            scattering=system,
            internal_model=AdiabaticSteadyState(),
        )
        result_b = run_simulation(
            [],
            SimulationConfig(**base),
            scattering=system,
            internal_model=RateEquationPopulations(),
        )
        self.assertLess(
            abs(
                result_a.final_temperature_uK_all - result_b.final_temperature_uK_all
            ),
            0.5 * result_a.final_temperature_uK_all + 50.0,
        )


class MixedPhysicsTests(unittest.TestCase):
    """Conservative traps and scattering light in the same simulation."""

    def _tweezer(self):
        return GaussianTrap(
            center_m=(0.0, 0.0, 0.0),
            waist_radial_m=float(um(1.0)),
            waist_axial_m=float(um(5.0)),
            depth_uK=1000.0,
        )

    def test_resonant_light_heats_atom_in_tweezer(self):
        # Rb87 tweezer + one weak resonant probe: recoil heating must show
        # up as a temperature increase relative to the dark tweezer.
        trap = self._tweezer()
        beam = LaserBeam(direction=(0.0, 0.0, 1.0), detuning_hz=0.0, saturation=0.01)
        system = LightMatterSystem(species=RB87_D2, beams=[beam])
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=2.0e-3,
            ensemble_size=100,
            random_seed=21,
        )
        lit = run_simulation(trap, config, scattering=system)
        dark = run_simulation(trap, config)
        self.assertGreater(
            lit.final_temperature_uK_all, 2.0 * dark.final_temperature_uK_all
        )
        self.assertGreater(float(np.mean(lit.scattered_photons)), 10.0)
        # Energy bookkeeping still uses the conservative potential, so the
        # recoil heating appears as a mechanical energy gain.
        self.assertGreater(lit.mean_energy_gain_uK, 0.0)

    def test_scattering_disables_energy_loss_criterion(self):
        # Same run, but atoms heated above trap depth must not be flagged
        # lost by the (disabled) energy criterion; no loss radius is set.
        trap = self._tweezer()
        beam = LaserBeam(direction=(0.0, 0.0, 1.0), detuning_hz=0.0, saturation=0.5)
        system = LightMatterSystem(species=RB87_D2, beams=[beam])
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=1.0e-3,
            ensemble_size=30,
            random_seed=22,
        )
        result = run_simulation(trap, config, scattering=system)
        self.assertEqual(result.loss_fraction, 0.0)


class ValidationTests(unittest.TestCase):
    def test_no_traps_and_no_scattering_raises(self):
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=1.0e-6,
            ensemble_size=2,
            initial_cloud_sigma_m=1.0e-6,
        )
        with self.assertRaises(ValueError):
            run_simulation([], config)

    def test_mass_mismatch_raises(self):
        system = _molasses_system(species=RB85_D2)
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=1.0e-6,
            ensemble_size=2,
            initial_cloud_sigma_m=1.0e-6,
        )  # mass_kg defaults to Rb87
        with self.assertRaises(ValueError):
            run_simulation([], config, scattering=system)

    def test_internal_model_without_scattering_raises(self):
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=1.0e-6,
            ensemble_size=2,
            initial_cloud_sigma_m=1.0e-6,
        )
        with self.assertRaises(TypeError):
            run_simulation([], config, internal_model=AdiabaticSteadyState())

    def test_no_traps_requires_cloud_sigma(self):
        system = _molasses_system()
        config = SimulationConfig(
            initial_temperature_uK=10.0,
            timestep_s=1.0e-7,
            duration_s=1.0e-6,
            ensemble_size=2,
            mass_kg=RB85_D2.mass_kg,
        )
        with self.assertRaises(ValueError):
            run_simulation([], config, scattering=system)


if __name__ == "__main__":
    unittest.main()
