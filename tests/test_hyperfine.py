"""Tests for the m_F-resolved hyperfine scattering module.

Covers the exact angular-algebra tables (Wigner symbols, branching
ratios, transition strengths, hyperfine splittings — spot-checked
against Steck's alkali data), the multilevel rate-equation dynamics
(two-level limit, optical pumping, dark states, repumping,
conservation), a full `simulate()` molasses cooling run, and an
optional element-by-element cross-validation against the ARC package
(skipped when ARC is not installed).
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atommc import (  # noqa: E402
    AtomSystem,
    HyperfineScattering,
    LaserBeam,
    LightMatterSystem,
    RB85_D2_HFS,
    RB87_D2_HFS,
    SimulationConfig,
    UniformMagneticField,
    hyperfine_species_from_arc,
    simulate,
    six_beam_mot,
)
from atommc.physics.internal_state import (  # noqa: E402
    steady_state_excited_fraction,
)
from atommc.physics.wigner import wigner_3j, wigner_6j  # noqa: E402

GAMMA_RAD_S = RB85_D2_HFS.base.linewidth_rad_s
GAMMA_HZ = GAMMA_RAD_S / (2.0 * np.pi)

# A tiny uniform field fixes the quantization axis without Zeeman shifts.
AXIS_FIELD = UniformMagneticField(field_T=(0.0, 0.0, 1.0e-9))


def _cycling_process(saturation, detuning_gamma, hyperfine=RB85_D2_HFS, **kwargs):
    """Single sigma+ beam along +z / B, referenced to the cycling line."""

    beam = LaserBeam(
        direction=(0.0, 0.0, 1.0),
        detuning_hz=detuning_gamma * GAMMA_HZ,
        saturation=saturation,
        helicity=+1.0,
    )
    light = LightMatterSystem(
        species=hyperfine.base, beams=[beam], magnetic_fields=[AXIS_FIELD]
    )
    return HyperfineScattering(light=light, hyperfine=hyperfine, **kwargs)


def _run_static(process, n_atoms, n_steps, dt_s, rng):
    """Step atoms at rest at the origin; return final state + photon total."""

    state = process.initialize(n_atoms, rng)
    positions = np.zeros((n_atoms, 3))
    velocities = np.zeros((n_atoms, 3))
    photons = np.zeros(n_atoms, dtype=np.int64)
    time_s = 0.0
    for _ in range(n_steps):
        result = process.step(state, positions, velocities, time_s, dt_s, rng)
        state = result.state
        photons += result.diagnostics["scattered_photons"]
        time_s += dt_s
    return state, photons


class WignerSymbolTests(unittest.TestCase):
    def test_known_3j_values(self):
        self.assertAlmostEqual(wigner_3j(1, 1, 0, 0, 0, 0), -1.0 / np.sqrt(3), 12)
        self.assertAlmostEqual(wigner_3j(2, 1, 1, 0, 0, 0), np.sqrt(2.0 / 15.0), 12)
        self.assertAlmostEqual(
            wigner_3j(0.5, 0.5, 1, 0.5, 0.5, -1), -1.0 / np.sqrt(3), 12
        )

    def test_known_6j_values(self):
        self.assertAlmostEqual(wigner_6j(1, 1, 1, 1, 1, 1), 1.0 / 6.0, 12)
        self.assertAlmostEqual(
            wigner_6j(0.5, 0.5, 1, 0.5, 0.5, 1), 1.0 / 6.0, 12
        )

    def test_selection_rules_return_zero(self):
        self.assertEqual(wigner_3j(1, 1, 3, 0, 0, 0), 0.0)  # triangle
        self.assertEqual(wigner_3j(1, 1, 1, 1, 1, -1), 0.0)  # m sum
        self.assertEqual(wigner_6j(1, 1, 3, 1, 1, 1), 0.0)  # triangle

    def test_rejects_non_half_integers(self):
        with self.assertRaises(ValueError):
            wigner_3j(0.3, 1, 1, 0, 0, 0)


class HyperfineTableTests(unittest.TestCase):
    def test_manifold_sizes(self):
        self.assertEqual(RB85_D2_HFS.n_ground, 12)  # F = 2, 3
        self.assertEqual(RB85_D2_HFS.n_excited, 24)  # F' = 1..4
        self.assertEqual(RB87_D2_HFS.n_ground, 8)  # F = 1, 2
        self.assertEqual(RB87_D2_HFS.n_excited, 16)  # F' = 0..3

    def test_hyperfine_splittings_match_steck(self):
        def split(hf, manifold, f_lo, f_hi):
            f = getattr(hf, f"{manifold}_f")
            off = getattr(hf, f"{manifold}_offset_rad_s")
            lo = off[np.flatnonzero(f == f_lo)[0]]
            hi = off[np.flatnonzero(f == f_hi)[0]]
            return (hi - lo) / (2.0 * np.pi)

        self.assertAlmostEqual(
            split(RB85_D2_HFS, "ground", 2, 3) / 1e6, 3035.732, delta=0.01
        )
        self.assertAlmostEqual(
            split(RB85_D2_HFS, "excited", 3, 4) / 1e6, 120.640, delta=0.01
        )
        self.assertAlmostEqual(
            split(RB85_D2_HFS, "excited", 2, 3) / 1e6, 63.401, delta=0.01
        )
        self.assertAlmostEqual(
            split(RB87_D2_HFS, "ground", 1, 2) / 1e6, 6834.683, delta=0.01
        )
        self.assertAlmostEqual(
            split(RB87_D2_HFS, "excited", 2, 3) / 1e6, 266.650, delta=0.01
        )

    def test_lande_g_factors(self):
        def g_of(hf, manifold, f):
            f_arr = getattr(hf, f"{manifold}_f")
            g_arr = getattr(hf, f"{manifold}_g_factor")
            return float(g_arr[np.flatnonzero(f_arr == f)[0]])

        self.assertAlmostEqual(g_of(RB85_D2_HFS, "ground", 3), 1.0 / 3.0, delta=1e-3)
        self.assertAlmostEqual(g_of(RB85_D2_HFS, "ground", 2), -1.0 / 3.0, delta=1e-3)
        self.assertAlmostEqual(g_of(RB85_D2_HFS, "excited", 4), 0.5, delta=1e-3)
        self.assertAlmostEqual(g_of(RB87_D2_HFS, "ground", 2), 0.5, delta=1e-3)
        self.assertAlmostEqual(g_of(RB87_D2_HFS, "excited", 3), 2.0 / 3.0, delta=1e-3)
        self.assertEqual(g_of(RB87_D2_HFS, "excited", 0), 0.0)

    def test_branching_rows_sum_to_one(self):
        for hf in (RB85_D2_HFS, RB87_D2_HFS):
            np.testing.assert_allclose(
                hf.branching_ratios.sum(axis=1), 1.0, rtol=1e-12
            )
            self.assertTrue(np.all(hf.branching_ratios >= 0.0))

    def test_branching_spot_values(self):
        hf = RB85_D2_HFS
        b = hf.branching_ratios
        # Stretched cycling decay is closed.
        e44 = hf.excited_level_index(4, 4)
        g33 = hf.ground_level_index(3, 3)
        self.assertAlmostEqual(b[e44, g33], 1.0, places=12)
        # |F'=3, +3> -> |2, +2> = 4/9 (Steck / ARC).
        e33 = hf.excited_level_index(3, 3)
        g22 = hf.ground_level_index(2, 2)
        self.assertAlmostEqual(b[e33, g22], 4.0 / 9.0, places=12)
        # F' = 1 decays only into F = 2 (Delta F selection).
        f1_rows = np.flatnonzero(hf.excited_f == 1)
        f3_cols = np.flatnonzero(hf.ground_f == 3)
        self.assertEqual(float(np.abs(b[np.ix_(f1_rows, f3_cols)]).max()), 0.0)

    def test_strength_spot_values(self):
        hf = RB87_D2_HFS
        strength = np.zeros((hf.n_ground, hf.n_excited))
        strength[hf.transition_ground_index, hf.transition_excited_index] = (
            hf.transition_strength
        )
        self.assertAlmostEqual(
            strength[hf.ground_level_index(2, 2), hf.excited_level_index(3, 3)],
            1.0,
            places=12,
        )
        # F=2, m=0 -> F'=3, m=+1 sigma+ relative strength = 2/5 (Steck).
        self.assertAlmostEqual(
            strength[hf.ground_level_index(2, 0), hf.excited_level_index(3, 1)],
            0.4,
            places=12,
        )
        # Delta F = 2 is dipole-forbidden: no F=2 -> F'=0 transitions.
        g_f2 = np.isin(hf.transition_ground_index, np.flatnonzero(hf.ground_f == 2))
        e_f0 = np.isin(hf.transition_excited_index, np.flatnonzero(hf.excited_f == 0))
        self.assertFalse(np.any(g_f2 & e_f0))

    def test_repump_offsets(self):
        self.assertAlmostEqual(
            RB85_D2_HFS.transition_offset_hz(2, 3) / 1e9, 2.9151, delta=0.001
        )
        self.assertAlmostEqual(
            RB87_D2_HFS.transition_offset_hz(1, 2) / 1e9, 6.5680, delta=0.001
        )

    def test_base_species_consistency_enforced(self):
        with self.assertRaises(ValueError):
            # Rb87 base with Rb85 nuclear spin: cycling levels disagree.
            type(RB85_D2_HFS)(
                base=RB87_D2_HFS.base,
                i_nuclear=2.5,
                hfs_a_ground_hz=1.0e9,
                hfs_a_excited_hz=25.0e6,
            )

    def test_g_factor_override_is_applied_and_validated(self):
        overrides = {2.0: -0.3, 3.0: 0.3}
        hf = type(RB85_D2_HFS)(
            base=RB85_D2_HFS.base,
            i_nuclear=2.5,
            hfs_a_ground_hz=RB85_D2_HFS.hfs_a_ground_hz,
            hfs_a_excited_hz=RB85_D2_HFS.hfs_a_excited_hz,
            hfs_b_excited_hz=RB85_D2_HFS.hfs_b_excited_hz,
            g_factor_ground_by_f=overrides,
        )
        self.assertEqual(
            float(hf.ground_g_factor[hf.ground_level_index(3, 0)]), 0.3
        )
        with self.assertRaises(ValueError):
            type(RB85_D2_HFS)(
                base=RB85_D2_HFS.base,
                i_nuclear=2.5,
                hfs_a_ground_hz=RB85_D2_HFS.hfs_a_ground_hz,
                hfs_a_excited_hz=RB85_D2_HFS.hfs_a_excited_hz,
                g_factor_ground_by_f={3.0: 0.3},  # F=2 missing
            )

    def test_from_arc_rejects_unknown_isotope(self):
        with self.assertRaises(ValueError):
            hyperfine_species_from_arc("Fr210")


class HyperfineDynamicsTests(unittest.TestCase):
    def test_two_level_limit_on_stretched_transition(self):
        rng = np.random.default_rng(7)
        for s0, det in ((0.5, 0.0), (2.0, -1.0), (10.0, -0.5)):
            process = _cycling_process(s0, det, initial_populations="stretched")
            state, _ = _run_static(process, 3, 400, 5.0e-9, rng)
            p_exc = state[:, RB85_D2_HFS.n_ground :].sum(axis=-1)
            w = 0.5 * GAMMA_RAD_S * s0 / (1.0 + (2.0 * det) ** 2)
            ref = float(steady_state_excited_fraction(np.array([w]), GAMMA_RAD_S)[0])
            np.testing.assert_allclose(p_exc, ref, rtol=1e-3)

    def test_large_timestep_reaches_exact_steady_state(self):
        # Gamma * dt ~ 7.6: implicit Euler must stay stable and converge
        # to the same steady state as the resolved-timestep run.
        rng = np.random.default_rng(8)
        process = _cycling_process(1.0, 0.0, initial_populations="stretched")
        state, _ = _run_static(process, 2, 50, 2.0e-7, rng)
        p_exc = state[:, RB85_D2_HFS.n_ground :].sum(axis=-1)
        np.testing.assert_allclose(p_exc, 0.25, rtol=1e-6)

    def test_sigma_plus_pumps_to_stretched_state(self):
        rng = np.random.default_rng(9)
        process = _cycling_process(1.0, 0.0, initial_populations="cooling-uniform")
        state, _ = _run_static(process, 2, 3000, 2.0e-8, rng)
        idx = RB85_D2_HFS.ground_level_index(3, 3)
        self.assertGreater(float(state[0, idx]), 0.6)

    def test_dark_state_accumulation_without_repump(self):
        rng = np.random.default_rng(10)
        hf = RB85_D2_HFS
        beam = LaserBeam(
            direction=(0.0, 0.0, 1.0),
            detuning_hz=hf.transition_offset_hz(3, 3),
            saturation=1.0,
            helicity=+1.0,
        )
        light = LightMatterSystem(
            species=hf.base, beams=[beam], magnetic_fields=[AXIS_FIELD]
        )
        process = HyperfineScattering(
            light=light, hyperfine=hf, initial_populations="cooling-uniform"
        )
        state, _ = _run_static(process, 2, 2000, 2.0e-8, rng)
        f2 = state[0, np.flatnonzero(hf.ground_f == 2)].sum()
        excited = state[0, hf.n_ground :].sum()
        self.assertGreater(float(f2), 0.5)
        self.assertLess(float(excited), 1e-3)

    def test_repump_prevents_dark_accumulation(self):
        rng = np.random.default_rng(11)
        hf = RB85_D2_HFS
        pump = LaserBeam(
            direction=(0.0, 0.0, 1.0),
            detuning_hz=hf.transition_offset_hz(3, 3),
            saturation=1.0,
            helicity=+1.0,
        )
        repump = LaserBeam(
            direction=(0.0, 0.0, 1.0),
            detuning_hz=hf.transition_offset_hz(2, 3),
            saturation=0.5,
            helicity=+1.0,
        )
        light = LightMatterSystem(
            species=hf.base, beams=[pump, repump], magnetic_fields=[AXIS_FIELD]
        )
        process = HyperfineScattering(
            light=light, hyperfine=hf, initial_populations="cooling-uniform"
        )
        state, _ = _run_static(process, 2, 2000, 2.0e-8, rng)
        f2 = state[0, np.flatnonzero(hf.ground_f == 2)].sum()
        self.assertLess(float(f2), 0.05)

    def test_populations_conserved_and_positive(self):
        rng = np.random.default_rng(12)
        process = _cycling_process(3.0, -0.7, initial_populations="ground-uniform")
        state, _ = _run_static(process, 4, 500, 5.0e-8, rng)
        self.assertGreaterEqual(float(state.min()), 0.0)
        np.testing.assert_allclose(state.sum(axis=-1), 1.0, atol=1e-10)

    def test_photon_rate_matches_excited_population(self):
        rng = np.random.default_rng(13)
        process = _cycling_process(1.0, 0.0, initial_populations="stretched")
        duration = 500 * 1.0e-8
        _, photons = _run_static(process, 200, 500, 1.0e-8, rng)
        # `scattered_photons` counts absorptions: rate W * p_ground with
        # W = Gamma/2 (s = 1, resonant) and p_ground = 0.75. In steady
        # state absorbed = stimulated + spontaneous.
        expected = 0.5 * GAMMA_RAD_S * 0.75 * duration  # ~71.5 photons
        self.assertAlmostEqual(
            float(np.mean(photons)), expected, delta=0.05 * expected
        )

    def test_dark_atoms_are_skipped_cheaply(self):
        # Atoms far outside every beam must keep their state exactly.
        rng = np.random.default_rng(14)
        beam = LaserBeam(
            direction=(0.0, 0.0, 1.0),
            detuning_hz=0.0,
            saturation=1.0,
            helicity=+1.0,
            waist_m=1.0e-3,
        )
        light = LightMatterSystem(
            species=RB85_D2_HFS.base, beams=[beam], magnetic_fields=[AXIS_FIELD]
        )
        process = HyperfineScattering(light=light, hyperfine=RB85_D2_HFS)
        state = process.initialize(2, rng)
        far = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        result = process.step(state, far, np.zeros((2, 3)), 0.0, 1e-7, rng)
        np.testing.assert_array_equal(result.state, state)
        self.assertEqual(int(result.diagnostics["scattered_photons"].sum()), 0)


class HyperfineSimulateTests(unittest.TestCase):
    def test_molasses_cools_with_repump(self):
        hf = RB87_D2_HFS
        beams = six_beam_mot(detuning_hz=-0.5 * GAMMA_HZ, saturation=1.0)
        beams.append(
            LaserBeam(
                direction=(0.0, 0.0, 1.0),
                detuning_hz=hf.transition_offset_hz(1, 2),
                saturation=0.3,
                helicity=+1.0,
            )
        )
        light = LightMatterSystem(
            species=hf.base, beams=beams, magnetic_fields=[AXIS_FIELD]
        )
        system = AtomSystem(
            species=hf.base,
            modules=[HyperfineScattering(light=light, hyperfine=hf)],
        )
        config = SimulationConfig(
            initial_temperature_uK=400.0,
            timestep_s=2.5e-7,
            duration_s=1.5e-3,
            ensemble_size=30,
            initial_cloud_sigma_m=(1e-4, 1e-4, 1e-4),
            loss_radius_m=2.0e-2,
            random_seed=3,
        )
        result = simulate(system, config)

        self.assertEqual(int(np.sum(result.lost)), 0)
        self.assertGreater(float(np.mean(result.scattered_photons)), 100.0)
        self.assertIsNotNone(result.final_excited_fraction)

        temp_uK = (
            hf.base.mass_kg
            * float(np.mean(np.sum(result.final_velocities_m_per_s**2, axis=-1)))
            / 3.0
            / 1.380649e-23
            * 1e6
        )
        self.assertLess(temp_uK, 300.0)

        # The repumper must keep the dark F=1 ground level empty (the
        # rest of the population is split between ground F=2 and the
        # excited manifold at this saturation).
        f1 = result.diagnostics["scattering"]["ground_f1_population"]
        self.assertLess(float(np.mean(f1)), 0.1)


@unittest.skipUnless(
    importlib.util.find_spec("arc") is not None,
    "ARC (alkali-rydberg-calculator) not installed",
)
class ArcCrossValidationTests(unittest.TestCase):
    """Element-by-element comparison against the ARC package."""

    def test_tables_match_arc(self):
        from arc import Rubidium85, Rubidium87

        for hf, atom in ((RB85_D2_HFS, Rubidium85()), (RB87_D2_HFS, Rubidium87())):
            # Branching ratios.
            for e in range(hf.n_excited):
                fe, me = hf.excited_f[e], hf.excited_m[e]
                for g in range(hf.n_ground):
                    fg, mg = hf.ground_f[g], hf.ground_m[g]
                    if abs(me - mg) > 1:
                        continue
                    ref = atom.getBranchingRatio(0.5, fg, mg, 1.5, fe, me)
                    self.assertAlmostEqual(
                        float(hf.branching_ratios[e, g]), ref, places=12
                    )
            # Excited hyperfine offsets.
            a_e, b_e = atom.getHFSCoefficients(5, 1, 1.5)
            f_max = hf.excited_f.max()
            ref_top = atom.getHFSEnergyShift(1.5, f_max, a_e, b_e)
            for f in sorted(set(hf.excited_f.tolist())):
                e = int(np.flatnonzero(hf.excited_f == f)[0])
                got_hz = hf.excited_offset_rad_s[e] / (2.0 * np.pi)
                ref_hz = atom.getHFSEnergyShift(1.5, f, a_e, b_e) - ref_top
                self.assertAlmostEqual(got_hz / 1e6, ref_hz / 1e6, places=6)

    def test_from_arc_matches_presets_and_is_cached(self):
        for isotope, preset in (("Rb85", RB85_D2_HFS), ("Rb87", RB87_D2_HFS)):
            hf = hyperfine_species_from_arc(isotope)
            # Cached: a second call returns the identical object.
            self.assertIs(hyperfine_species_from_arc(isotope), hf)
            base, ref = hf.base, preset.base
            self.assertAlmostEqual(
                base.linewidth_rad_s / ref.linewidth_rad_s, 1.0, delta=1e-3
            )
            self.assertAlmostEqual(
                base.saturation_intensity_w_per_m2
                / ref.saturation_intensity_w_per_m2,
                1.0,
                delta=1e-3,
            )
            self.assertAlmostEqual(base.mass_kg / ref.mass_kg, 1.0, delta=1e-6)
            self.assertAlmostEqual(
                hf.hfs_a_ground_hz / preset.hfs_a_ground_hz, 1.0, delta=1e-6
            )
            # Same Wigner algebra behind both: identical branching.
            np.testing.assert_allclose(
                hf.branching_ratios, preset.branching_ratios, atol=1e-14
            )
            # ARC g_F include the nuclear term: close to, but not exactly,
            # the electronic-only preset values.
            np.testing.assert_allclose(
                hf.ground_g_factor, preset.ground_g_factor, atol=2e-3
            )

    def test_from_arc_builds_other_alkalis(self):
        cs = hyperfine_species_from_arc("Cs133")
        self.assertEqual(cs.n_ground, 16)  # F = 3, 4
        self.assertEqual(cs.n_excited, 32)  # F' = 2..5
        self.assertAlmostEqual(cs.base.wavelength_m * 1e9, 852.35, delta=0.01)
        # Cs repump offset F=3 -> F'=4 is ~8.94 GHz blue of the cycling line.
        self.assertAlmostEqual(
            cs.transition_offset_hz(3, 4) / 1e9, 8.94, delta=0.02
        )
        # K40: inverted hyperfine structure must still build consistently.
        k40 = hyperfine_species_from_arc("K40")
        np.testing.assert_allclose(k40.branching_ratios.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
