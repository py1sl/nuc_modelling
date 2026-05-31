"""
Tests for nuclear_interaction.py

Covers all four layers:
  1. Potential forms
  2. Cross-section utilities
  3. NuclearTarget class
  4. Uncertainty propagation and model metadata
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nuclear_interaction import (
    # Constants
    HBAR_C_MEV_FM,
    FINE_STRUCTURE_CONSTANT,
    NUCLEAR_RADIUS_PARAM_FM,
    FM2_TO_MB,
    MB_TO_FM2,
    # Potentials
    yukawa_potential,
    square_well_potential,
    gaussian_potential,
    woods_saxon_potential,
    # Cross sections
    rutherford_differential_cross_section,
    rutherford_total_cross_section,
    geometric_cross_section,
    nuclear_reaction_cross_section,
    effective_range_cross_section,
    born_approximation_cross_section,
    # Target profile
    NuclearTarget,
    # Uncertainty / metadata
    propagate_relative_uncertainty,
    model_result,
    energy_loss_bethe,
)


# ===========================================================================
# 1. Potential forms
# ===========================================================================

class TestYukawaPotential:
    """Tests for yukawa_potential."""

    def test_negative_attractive(self):
        """Yukawa potential should be attractive (negative)."""
        v = yukawa_potential(2.0, 40.0, 0.71)
        assert v < 0

    def test_decreases_with_r(self):
        """Yukawa potential magnitude decreases with distance."""
        v1 = yukawa_potential(1.0, 40.0, 0.71)
        v2 = yukawa_potential(3.0, 40.0, 0.71)
        assert abs(v1) > abs(v2)

    def test_increases_with_V0(self):
        """Deeper well should give larger magnitude."""
        v_shallow = yukawa_potential(2.0, 20.0, 0.71)
        v_deep = yukawa_potential(2.0, 80.0, 0.71)
        assert abs(v_deep) > abs(v_shallow)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            yukawa_potential(0.0, 40.0, 0.71)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            yukawa_potential(-1.0, 40.0, 0.71)

    def test_negative_V0_raises(self):
        with pytest.raises(ValueError):
            yukawa_potential(2.0, -40.0, 0.71)

    def test_zero_mu_raises(self):
        with pytest.raises(ValueError):
            yukawa_potential(2.0, 40.0, 0.0)

    def test_large_r_approaches_zero(self):
        """Yukawa potential should vanish at large r."""
        v = yukawa_potential(100.0, 40.0, 0.71)
        assert abs(v) < 1e-20


class TestSquareWellPotential:
    """Tests for square_well_potential."""

    def test_inside_well(self):
        """Potential inside the well equals -V0."""
        assert square_well_potential(1.0, 50.0, 2.5) == pytest.approx(-50.0)

    def test_at_edge(self):
        """At r == R the potential is still -V0 (inside)."""
        assert square_well_potential(2.5, 50.0, 2.5) == pytest.approx(-50.0)

    def test_outside_well(self):
        """Potential outside the well is zero."""
        assert square_well_potential(3.0, 50.0, 2.5) == pytest.approx(0.0)

    def test_at_zero(self):
        """At r=0 the potential is -V0."""
        assert square_well_potential(0.0, 50.0, 2.5) == pytest.approx(-50.0)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            square_well_potential(-0.5, 50.0, 2.5)

    def test_negative_V0_raises(self):
        with pytest.raises(ValueError):
            square_well_potential(1.0, -50.0, 2.5)

    def test_zero_R_raises(self):
        with pytest.raises(ValueError):
            square_well_potential(1.0, 50.0, 0.0)


class TestGaussianPotential:
    """Tests for gaussian_potential."""

    def test_at_origin_equals_minus_V0(self):
        """At r=0 the Gaussian potential equals -V0."""
        assert gaussian_potential(0.0, 50.0, 1.5) == pytest.approx(-50.0)

    def test_negative_attractive(self):
        """Potential should be attractive (negative)."""
        assert gaussian_potential(1.0, 50.0, 1.5) < 0

    def test_decreases_with_r(self):
        v1 = gaussian_potential(0.5, 50.0, 1.5)
        v2 = gaussian_potential(2.0, 50.0, 1.5)
        assert abs(v1) > abs(v2)

    def test_at_r_equals_a_magnitude(self):
        """At r=a, magnitude should be V0 * exp(-1)."""
        v = gaussian_potential(1.5, 50.0, 1.5)
        assert v == pytest.approx(-50.0 * math.exp(-1.0), rel=1e-10)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            gaussian_potential(-1.0, 50.0, 1.5)

    def test_zero_a_raises(self):
        with pytest.raises(ValueError):
            gaussian_potential(1.0, 50.0, 0.0)

    def test_zero_V0_raises(self):
        with pytest.raises(ValueError):
            gaussian_potential(1.0, 0.0, 1.5)


class TestWoodsSaxonPotential:
    """Tests for woods_saxon_potential."""

    def test_at_surface_half_depth(self):
        """At r=R the potential is -V0/2."""
        v = woods_saxon_potential(5.0, 50.0, 5.0, 0.65)
        assert v == pytest.approx(-25.0, abs=0.01)

    def test_attractive(self):
        v = woods_saxon_potential(3.0, 50.0, 6.0, 0.65)
        assert v < 0

    def test_approaches_minus_V0_at_origin(self):
        """Deep inside the nucleus, potential approaches -V0."""
        v = woods_saxon_potential(0.0, 50.0, 6.0, 0.65)
        assert v == pytest.approx(-50.0, rel=0.01)

    def test_approaches_zero_far_outside(self):
        """Far outside, potential should approach zero."""
        v = woods_saxon_potential(30.0, 50.0, 6.0, 0.65)
        assert abs(v) < 0.01

    def test_decreases_with_r(self):
        v_inner = woods_saxon_potential(2.0, 50.0, 6.0, 0.65)
        v_outer = woods_saxon_potential(9.0, 50.0, 6.0, 0.65)
        assert abs(v_inner) > abs(v_outer)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            woods_saxon_potential(-1.0, 50.0, 6.0, 0.65)

    def test_zero_R_raises(self):
        with pytest.raises(ValueError):
            woods_saxon_potential(1.0, 50.0, 0.0, 0.65)

    def test_zero_a_raises(self):
        with pytest.raises(ValueError):
            woods_saxon_potential(1.0, 50.0, 6.0, 0.0)

    def test_zero_V0_raises(self):
        with pytest.raises(ValueError):
            woods_saxon_potential(1.0, 0.0, 6.0, 0.65)


# ===========================================================================
# 2. Cross-section utilities
# ===========================================================================

class TestRutherfordDifferentialCrossSection:
    """Tests for rutherford_differential_cross_section."""

    def test_positive_result(self):
        dcs = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 2)
        assert dcs > 0

    def test_diverges_at_small_angle(self):
        """Smaller angle → larger cross section (Coulomb divergence)."""
        dcs_small = rutherford_differential_cross_section(10.0, 2, 79, 0.1)
        dcs_large = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 2)
        assert dcs_small > dcs_large

    def test_increases_with_charge(self):
        """Higher product z1*z2 → larger cross section."""
        dcs_light = rutherford_differential_cross_section(10.0, 1, 1, math.pi / 2)
        dcs_heavy = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 2)
        assert dcs_heavy > dcs_light

    def test_decreases_with_energy(self):
        """Higher kinetic energy → smaller Rutherford cross section."""
        dcs_low = rutherford_differential_cross_section(5.0, 2, 79, math.pi / 2)
        dcs_high = rutherford_differential_cross_section(20.0, 2, 79, math.pi / 2)
        assert dcs_low > dcs_high

    def test_zero_energy_raises(self):
        with pytest.raises(ValueError):
            rutherford_differential_cross_section(0.0, 2, 79, math.pi / 2)

    def test_zero_angle_raises(self):
        with pytest.raises(ValueError):
            rutherford_differential_cross_section(10.0, 2, 79, 0.0)

    def test_angle_above_pi_raises(self):
        with pytest.raises(ValueError):
            rutherford_differential_cross_section(10.0, 2, 79, math.pi + 0.1)

    def test_zero_z1_raises(self):
        with pytest.raises(ValueError):
            rutherford_differential_cross_section(10.0, 0, 79, math.pi / 2)

    def test_z1_z2_symmetry(self):
        """Swapping z1 and z2 should give the same cross section."""
        dcs_12 = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 4)
        dcs_21 = rutherford_differential_cross_section(10.0, 79, 2, math.pi / 4)
        assert dcs_12 == pytest.approx(dcs_21, rel=1e-10)

    def test_known_value_sin4_scaling(self):
        """Cross section should scale as 1/sin^4(theta/2)."""
        T, z1, z2 = 10.0, 2, 79
        theta1, theta2 = math.pi / 4, math.pi / 2
        dcs1 = rutherford_differential_cross_section(T, z1, z2, theta1)
        dcs2 = rutherford_differential_cross_section(T, z1, z2, theta2)
        ratio = dcs1 / dcs2
        expected_ratio = (math.sin(theta2 / 2) / math.sin(theta1 / 2)) ** 4
        assert ratio == pytest.approx(expected_ratio, rel=1e-8)


class TestRutherfordTotalCrossSection:
    """Tests for rutherford_total_cross_section."""

    def test_positive_result(self):
        sigma = rutherford_total_cross_section(10.0, 2, 79, 0.1)
        assert sigma > 0

    def test_decreases_with_theta_min(self):
        """Larger minimum angle → less solid angle → smaller σ."""
        sigma_small = rutherford_total_cross_section(10.0, 2, 79, 0.05)
        sigma_large = rutherford_total_cross_section(10.0, 2, 79, 0.5)
        assert sigma_small > sigma_large

    def test_zero_angle_raises(self):
        with pytest.raises(ValueError):
            rutherford_total_cross_section(10.0, 2, 79, 0.0)

    def test_zero_energy_raises(self):
        with pytest.raises(ValueError):
            rutherford_total_cross_section(0.0, 2, 79, 0.1)

    def test_charge_symmetry(self):
        sigma_12 = rutherford_total_cross_section(10.0, 2, 79, 0.1)
        sigma_21 = rutherford_total_cross_section(10.0, 79, 2, 0.1)
        assert sigma_12 == pytest.approx(sigma_21, rel=1e-10)

    def test_angle_above_pi_raises(self):
        with pytest.raises(ValueError):
            rutherford_total_cross_section(10.0, 2, 79, math.pi + 0.1)


class TestGeometricCrossSection:
    """Tests for geometric_cross_section."""

    def test_positive(self):
        assert geometric_cross_section(6.86) > 0

    def test_scales_as_r_squared(self):
        sig1 = geometric_cross_section(1.0)
        sig2 = geometric_cross_section(2.0)
        assert sig2 == pytest.approx(4.0 * sig1, rel=1e-10)

    def test_formula(self):
        R = 5.0
        expected_mb = math.pi * R ** 2 * FM2_TO_MB
        assert geometric_cross_section(R) == pytest.approx(expected_mb, rel=1e-10)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError):
            geometric_cross_section(0.0)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            geometric_cross_section(-1.0)


class TestNuclearReactionCrossSection:
    """Tests for nuclear_reaction_cross_section."""

    def test_positive(self):
        assert nuclear_reaction_cross_section(1, 12) > 0

    def test_larger_than_geometric_single_nucleus(self):
        """Reaction cross section (R1+R2)² > (R1)² + (R2)² individually."""
        sigma_r = nuclear_reaction_cross_section(12, 12)
        sigma_g = geometric_cross_section(
            NUCLEAR_RADIUS_PARAM_FM * 12 ** (1.0 / 3.0)
        )
        # (2R)^2 = 4R^2 > pi R^2 per nucleus
        assert sigma_r > sigma_g

    def test_symmetric(self):
        assert nuclear_reaction_cross_section(1, 12) == pytest.approx(
            nuclear_reaction_cross_section(12, 1), rel=1e-10
        )

    def test_proton_proton(self):
        assert nuclear_reaction_cross_section(1, 1) > 0

    def test_non_integer_a1_raises(self):
        with pytest.raises((ValueError, TypeError)):
            nuclear_reaction_cross_section(1.5, 12)

    def test_zero_a1_raises(self):
        with pytest.raises(ValueError):
            nuclear_reaction_cross_section(0, 12)


class TestEffectiveRangeCrossSection:
    """Tests for effective_range_cross_section."""

    def test_zero_k_gives_4pi_a_squared(self):
        """At k=0, σ = 4π a²."""
        a = 1.0  # fm
        sigma = effective_range_cross_section(0.0, a)
        expected = 4.0 * math.pi * a ** 2 * FM2_TO_MB
        assert sigma == pytest.approx(expected, rel=1e-6)

    def test_positive_result(self):
        assert effective_range_cross_section(0.5, 2.0) > 0

    def test_decreases_with_k(self):
        """Cross section should decrease as k increases (above the resonance)."""
        sig_low = effective_range_cross_section(0.1, 2.0)
        sig_high = effective_range_cross_section(2.0, 2.0)
        assert sig_low > sig_high

    def test_effective_range_correction_changes_result(self):
        sigma_bare = effective_range_cross_section(0.5, 2.0, r0_fm=0.0)
        sigma_er = effective_range_cross_section(0.5, 2.0, r0_fm=1.5)
        assert sigma_bare != pytest.approx(sigma_er)

    def test_zero_a_raises(self):
        with pytest.raises(ValueError):
            effective_range_cross_section(0.5, 0.0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            effective_range_cross_section(-0.1, 2.0)

    def test_negative_r0_raises(self):
        with pytest.raises(ValueError):
            effective_range_cross_section(0.5, 2.0, r0_fm=-0.1)


class TestBornApproximationCrossSection:
    """Tests for born_approximation_cross_section."""

    def _gaussian(self, r_fm):
        return gaussian_potential(r_fm, 10.0, 1.5)

    def _square_well(self, r_fm):
        return square_well_potential(r_fm, 10.0, 2.0)

    def test_positive_result_gaussian(self):
        sigma = born_approximation_cross_section(
            self._gaussian, 0.2, 0.2, reduced_mass_mev_c2=469.0
        )
        assert sigma > 0

    def test_positive_result_square_well(self):
        sigma = born_approximation_cross_section(
            self._square_well, 0.2, 0.2, reduced_mass_mev_c2=469.0
        )
        assert sigma > 0

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            born_approximation_cross_section(
                self._gaussian, 0.0, 0.2, reduced_mass_mev_c2=469.0
            )

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError):
            born_approximation_cross_section(
                self._gaussian, 0.2, 0.2, reduced_mass_mev_c2=0.0
            )

    def test_deeper_potential_gives_larger_cross_section(self):
        """Deeper potential should give larger Born cross section."""
        def shallow(r):
            return gaussian_potential(r, 5.0, 1.5)

        def deep(r):
            return gaussian_potential(r, 50.0, 1.5)

        sig_shallow = born_approximation_cross_section(
            shallow, 0.2, 0.2, reduced_mass_mev_c2=469.0
        )
        sig_deep = born_approximation_cross_section(
            deep, 0.2, 0.2, reduced_mass_mev_c2=469.0
        )
        assert sig_deep > sig_shallow

    def test_n_points_below_2_raises(self):
        with pytest.raises(ValueError):
            born_approximation_cross_section(
                self._gaussian, 0.2, 0.2, reduced_mass_mev_c2=469.0, n_points=1
            )


# ===========================================================================
# 3. NuclearTarget class
# ===========================================================================

class TestNuclearTargetInit:
    """Tests for NuclearTarget initialisation."""

    def test_tungsten_radius(self):
        """W-184 nuclear radius should be ~6.86 fm."""
        target = NuclearTarget(z=74, a=184)
        expected = NUCLEAR_RADIUS_PARAM_FM * 184 ** (1.0 / 3.0)
        assert target.radius_fm == pytest.approx(expected, rel=1e-10)

    def test_fractions(self):
        target = NuclearTarget(z=74, a=184)
        assert target.z_fraction == pytest.approx(74 / 184, rel=1e-10)
        assert target.n_fraction == pytest.approx(110 / 184, rel=1e-10)
        assert target.z_fraction + target.n_fraction == pytest.approx(1.0)

    def test_density_positive(self):
        assert NuclearTarget(z=26, a=56).density_fm3 > 0

    def test_volume_formula(self):
        target = NuclearTarget(z=6, a=12)
        expected_v = (4.0 / 3.0) * math.pi * target.radius_fm ** 3
        assert target.volume_fm3 == pytest.approx(expected_v, rel=1e-10)

    def test_neutron_count(self):
        target = NuclearTarget(z=6, a=12)
        assert target.n == 6

    def test_zero_a_raises(self):
        with pytest.raises(ValueError):
            NuclearTarget(z=0, a=0)

    def test_negative_a_raises(self):
        with pytest.raises(ValueError):
            NuclearTarget(z=0, a=-1)

    def test_z_greater_than_a_raises(self):
        with pytest.raises(ValueError):
            NuclearTarget(z=100, a=50)

    def test_negative_z_raises(self):
        with pytest.raises(ValueError):
            NuclearTarget(z=-1, a=10)

    def test_z_zero_is_valid(self):
        """Pure neutron matter (Z=0) should be constructible."""
        target = NuclearTarget(z=0, a=1)
        assert target.z == 0

    def test_repr_contains_Z_and_A(self):
        target = NuclearTarget(z=74, a=184)
        r = repr(target)
        assert "74" in r and "184" in r


class TestNuclearTargetWoodsSaxonDensity:
    """Tests for NuclearTarget.woods_saxon_density."""

    def test_central_density_near_bulk(self):
        """Central density should be approximately equal to the average density."""
        target = NuclearTarget(z=74, a=184)
        rho_centre = target.woods_saxon_density(0.0)
        # For large A, R >> a, so the WS central value ≈ density_fm3 (within 0.1%)
        assert rho_centre == pytest.approx(target.density_fm3, rel=0.001)

    def test_half_density_at_surface(self):
        """At r = R the WS density should be approximately half the central value."""
        target = NuclearTarget(z=74, a=184)
        rho_centre = target.woods_saxon_density(0.0)
        rho_surface = target.woods_saxon_density(target.radius_fm)
        assert rho_surface == pytest.approx(rho_centre / 2.0, rel=0.05)

    def test_decreases_with_r(self):
        target = NuclearTarget(z=74, a=184)
        rho1 = target.woods_saxon_density(2.0)
        rho2 = target.woods_saxon_density(8.0)
        assert rho1 > rho2

    def test_negative_r_raises(self):
        target = NuclearTarget(z=74, a=184)
        with pytest.raises(ValueError):
            target.woods_saxon_density(-1.0)

    def test_zero_diffuseness_raises(self):
        target = NuclearTarget(z=74, a=184)
        with pytest.raises(ValueError):
            target.woods_saxon_density(2.0, a_diffuse_fm=0.0)


class TestNuclearTargetGeometric:
    """Tests for NuclearTarget geometric cross sections."""

    def test_geometric_cross_section_positive(self):
        assert NuclearTarget(z=74, a=184).geometric_cross_section_mb() > 0

    def test_geometric_cross_section_matches_function(self):
        target = NuclearTarget(z=74, a=184)
        expected = geometric_cross_section(target.radius_fm)
        assert target.geometric_cross_section_mb() == pytest.approx(expected)

    def test_reaction_cross_section_positive(self):
        assert NuclearTarget(z=74, a=184).reaction_cross_section_mb(1) > 0

    def test_reaction_cross_section_larger_than_geometric(self):
        """(R_p + R_T)² > R_T² for any positive projectile."""
        target = NuclearTarget(z=74, a=184)
        assert target.reaction_cross_section_mb(1) > target.geometric_cross_section_mb()

    def test_reaction_cross_section_invalid_a_raises(self):
        target = NuclearTarget(z=74, a=184)
        with pytest.raises(ValueError):
            target.reaction_cross_section_mb(0)


class TestNuclearTargetInteraction:
    """Tests for NuclearTarget interaction-probability helpers."""

    def test_interaction_probability_zero_areal_density(self):
        """At zero areal density, interaction probability is zero."""
        target = NuclearTarget(z=74, a=184)
        assert target.interaction_probability(0.0) == pytest.approx(0.0)

    def test_interaction_probability_increases_with_areal_density(self):
        target = NuclearTarget(z=74, a=184)
        p1 = target.interaction_probability(0.01)
        p2 = target.interaction_probability(0.1)
        assert p2 > p1

    def test_interaction_probability_bounded_by_one(self):
        target = NuclearTarget(z=74, a=184)
        for rho_a in [0.0, 0.001, 0.01, 1.0, 100.0]:
            p = target.interaction_probability(rho_a)
            assert 0.0 <= p <= 1.0

    def test_interaction_probability_negative_raises(self):
        target = NuclearTarget(z=74, a=184)
        with pytest.raises(ValueError):
            target.interaction_probability(-0.01)

    def test_mean_free_path_positive(self):
        target = NuclearTarget(z=74, a=184)
        mfp = target.mean_free_path_fm(45.0)  # ~45 mb pp cross section
        assert mfp > 0

    def test_mean_free_path_decreases_with_sigma(self):
        target = NuclearTarget(z=74, a=184)
        mfp_small = target.mean_free_path_fm(100.0)
        mfp_large = target.mean_free_path_fm(10.0)
        assert mfp_large > mfp_small

    def test_mean_free_path_zero_sigma_raises(self):
        target = NuclearTarget(z=74, a=184)
        with pytest.raises(ValueError):
            target.mean_free_path_fm(0.0)


class TestNuclearTargetPhysics:
    """Tests for NuclearTarget physics helpers."""

    def test_coulomb_barrier_positive(self):
        target = NuclearTarget(z=79, a=197)  # gold
        V_c = target.coulomb_barrier_mev(z_projectile=2, a_projectile=4)  # alpha
        assert V_c > 0

    def test_coulomb_barrier_zero_charge_is_zero(self):
        target = NuclearTarget(z=79, a=197)
        V_c = target.coulomb_barrier_mev(z_projectile=0, a_projectile=1)
        assert V_c == pytest.approx(0.0)

    def test_coulomb_barrier_increases_with_z_product(self):
        target = NuclearTarget(z=79, a=197)
        V_proton = target.coulomb_barrier_mev(z_projectile=1, a_projectile=1)
        V_alpha = target.coulomb_barrier_mev(z_projectile=2, a_projectile=4)
        # Alpha has z=2 and slightly larger radius, so barrier increases
        assert V_alpha > V_proton

    def test_coulomb_barrier_invalid_z_projectile_raises(self):
        target = NuclearTarget(z=79, a=197)
        with pytest.raises(ValueError):
            target.coulomb_barrier_mev(z_projectile=-1, a_projectile=1)

    def test_coulomb_barrier_zero_a_projectile_raises(self):
        target = NuclearTarget(z=79, a=197)
        with pytest.raises(ValueError):
            target.coulomb_barrier_mev(z_projectile=1, a_projectile=0)

    def test_fermi_momentum_positive(self):
        target = NuclearTarget(z=74, a=184)
        pf = target.fermi_momentum_mev_c()
        assert pf > 0

    def test_fermi_momentum_saturation_density_range(self):
        """Fermi momentum should be in 220–280 MeV/c for typical nuclei."""
        target = NuclearTarget(z=26, a=56)  # iron
        pf = target.fermi_momentum_mev_c()
        assert 200.0 < pf < 320.0


# ===========================================================================
# 4. Uncertainty propagation and model metadata
# ===========================================================================

class TestPropagateRelativeUncertainty:
    """Tests for propagate_relative_uncertainty."""

    def test_single_component(self):
        """Single 10% error on 50 mb → 5 mb uncertainty."""
        delta = propagate_relative_uncertainty(50.0, 0.10)
        assert delta == pytest.approx(5.0, rel=1e-10)

    def test_two_components_quadrature(self):
        """Two independent 10% and 5% errors combined in quadrature."""
        delta = propagate_relative_uncertainty(50.0, 0.10, 0.05)
        expected = 50.0 * math.sqrt(0.10 ** 2 + 0.05 ** 2)
        assert delta == pytest.approx(expected, rel=1e-10)

    def test_zero_uncertainty(self):
        delta = propagate_relative_uncertainty(50.0, 0.0)
        assert delta == pytest.approx(0.0)

    def test_zero_value(self):
        delta = propagate_relative_uncertainty(0.0, 0.10)
        assert delta == pytest.approx(0.0)

    def test_multiple_components(self):
        errs = [0.05, 0.08, 0.03]
        delta = propagate_relative_uncertainty(100.0, *errs)
        expected = 100.0 * math.sqrt(sum(e ** 2 for e in errs))
        assert delta == pytest.approx(expected, rel=1e-10)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            propagate_relative_uncertainty(-50.0, 0.10)

    def test_negative_relative_error_raises(self):
        with pytest.raises(ValueError):
            propagate_relative_uncertainty(50.0, -0.10)


class TestModelResult:
    """Tests for model_result."""

    def test_basic_fields(self):
        result = model_result(42.5, 3.0, "mb", "Rutherford")
        assert result["value"] == 42.5
        assert result["uncertainty"] == 3.0
        assert result["units"] == "mb"
        assert result["model"] == "Rutherford"
        assert result["valid_range"] is None
        assert result["notes"] is None

    def test_with_valid_range_and_notes(self):
        vr = {"T_mev": (1, 100)}
        result = model_result(42.5, 3.0, "mb", "Rutherford",
                              valid_range=vr, notes="Below Coulomb barrier")
        assert result["valid_range"] == vr
        assert "Coulomb" in result["notes"]

    def test_zero_uncertainty_allowed(self):
        result = model_result(100.0, 0.0, "fm", "formula")
        assert result["uncertainty"] == 0.0

    def test_negative_uncertainty_raises(self):
        with pytest.raises(ValueError):
            model_result(100.0, -1.0, "fm", "formula")

    def test_returns_dict(self):
        assert isinstance(model_result(1.0, 0.1, "mb", "test"), dict)


class TestEnergyLossBethe:
    """Tests for energy_loss_bethe (Bethe–Bloch stopping power)."""

    # Typical parameters: 200 MeV proton in iron
    _Z_PROJ = 1
    _T_MEV = 200.0
    _M_MEV = 938.272
    _I_EV = 286.0      # mean excitation energy of iron in eV (approximate)
    # Electron density of iron: Z=26, A=56, ρ≈0.17 fm^-3 nucleons
    # n_e = Z/A * ρ_nucl ≈ 0.4643 * 0.17 ≈ 0.079 fm^-3
    _N_E = 0.079

    def test_positive_result(self):
        dedx = energy_loss_bethe(
            self._Z_PROJ, self._T_MEV, self._M_MEV, self._I_EV, self._N_E
        )
        assert dedx > 0

    def test_increases_with_z_squared(self):
        """Stopping power ∝ z²."""
        dedx_p = energy_loss_bethe(1, self._T_MEV, self._M_MEV, self._I_EV, self._N_E)
        dedx_a = energy_loss_bethe(2, self._T_MEV, 3727.0, self._I_EV, self._N_E)
        # Alpha has z=2, so 4× larger prefactor (roughly)
        assert dedx_a > dedx_p

    def test_increases_with_electron_density(self):
        dedx_low = energy_loss_bethe(
            self._Z_PROJ, self._T_MEV, self._M_MEV, self._I_EV, 0.01
        )
        dedx_high = energy_loss_bethe(
            self._Z_PROJ, self._T_MEV, self._M_MEV, self._I_EV, 0.1
        )
        assert dedx_high > dedx_low

    def test_zero_energy_raises(self):
        with pytest.raises(ValueError):
            energy_loss_bethe(1, 0.0, self._M_MEV, self._I_EV, self._N_E)

    def test_zero_z_raises(self):
        with pytest.raises(ValueError):
            energy_loss_bethe(0, self._T_MEV, self._M_MEV, self._I_EV, self._N_E)

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError):
            energy_loss_bethe(1, self._T_MEV, 0.0, self._I_EV, self._N_E)

    def test_zero_I_raises(self):
        with pytest.raises(ValueError):
            energy_loss_bethe(1, self._T_MEV, self._M_MEV, 0.0, self._N_E)

    def test_zero_electron_density_raises(self):
        with pytest.raises(ValueError):
            energy_loss_bethe(1, self._T_MEV, self._M_MEV, self._I_EV, 0.0)


# ===========================================================================
# 5. Cross-module consistency tests
# ===========================================================================

class TestCrossModuleConsistency:
    """Verify that nuclear_interaction results are consistent with other modules."""

    def test_nuclear_target_radius_consistent_with_bertini(self):
        """NuclearTarget radius formula matches bertini_cascade.nuclear_radius."""
        from bertini_cascade import nuclear_radius as bertini_radius

        target = NuclearTarget(z=74, a=184)
        assert target.radius_fm == pytest.approx(bertini_radius(184), rel=1e-10)

    def test_nuclear_target_fermi_momentum_consistent_with_bertini(self):
        """NuclearTarget Fermi momentum matches bertini_cascade.fermi_momentum."""
        from bertini_cascade import fermi_momentum as bertini_fp

        target = NuclearTarget(z=74, a=184)
        assert target.fermi_momentum_mev_c() == pytest.approx(
            bertini_fp(target.density_fm3), rel=1e-10
        )

    def test_mean_free_path_consistent_with_bertini(self):
        """NuclearTarget.mean_free_path_fm is consistent with bertini mean_free_path."""
        from bertini_cascade import mean_free_path as bertini_mfp, nucleon_nucleon_cross_section

        target = NuclearTarget(z=74, a=184)
        T_mev = 500.0
        sigma_pp = nucleon_nucleon_cross_section(T_mev, is_pp=True)
        sigma_pn = nucleon_nucleon_cross_section(T_mev, is_pp=False)
        sigma_eff = target.z_fraction * sigma_pp + target.n_fraction * sigma_pn

        mfp_bertini = bertini_mfp(T_mev, target.density_fm3, target.z_fraction)
        mfp_target = target.mean_free_path_fm(sigma_eff)
        assert mfp_target == pytest.approx(mfp_bertini, rel=1e-8)

    def test_coulomb_barrier_kinematic_threshold(self):
        """Coulomb barrier should be below typical Bertini energy range (100–3000 MeV)."""
        target = NuclearTarget(z=74, a=184)  # tungsten
        V_c = target.coulomb_barrier_mev(z_projectile=1, a_projectile=1)
        # Proton-tungsten Coulomb barrier is typically ~13 MeV, well below 100 MeV
        assert V_c < 100.0
        assert V_c > 0.0


# ===========================================================================
# 6. Benchmark / regression tests (canonical cases)
# ===========================================================================

class TestBenchmarks:
    """Canonical benchmark values for regression detection."""

    def test_rutherford_alpha_gold_90deg(self):
        """Rutherford dσ/dΩ for 10 MeV alpha on gold at θ=90°.

        Expected ≈ 1.7e4 mb/sr (order of magnitude check).
        """
        dcs = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 2)
        assert 1e3 < dcs < 1e6  # mb/sr

    def test_geometric_cross_section_carbon12(self):
        """Geometric cross section of carbon-12 (R ≈ 2.74 fm) ≈ 236 mb."""
        R_c12 = NUCLEAR_RADIUS_PARAM_FM * 12 ** (1.0 / 3.0)
        sigma = geometric_cross_section(R_c12)
        assert 200 < sigma < 270  # mb

    def test_nuclear_reaction_cs_proton_carbon(self):
        """p+¹²C reaction cross section from geometric formula (R1+R2)²."""
        sigma = nuclear_reaction_cross_section(1, 12)
        # Geometric formula gives ~490 mb; empirical values vary with energy
        assert 400 < sigma < 600  # mb

    def test_woods_saxon_depth_at_deep_interior(self):
        """WS potential at r=0 for V0=50 MeV, R=6 fm, a=0.65 fm → ~−50 MeV."""
        v = woods_saxon_potential(0.0, 50.0, 6.0, 0.65)
        assert v == pytest.approx(-50.0, rel=0.01)

    def test_fermi_momentum_tungsten(self):
        """Tungsten Fermi momentum should be ~265–275 MeV/c."""
        target = NuclearTarget(z=74, a=184)
        pf = target.fermi_momentum_mev_c()
        assert 250.0 < pf < 290.0
