"""
Tests for kinematics.py — classical and relativistic collision kinematics.
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kinematics import (
    # classical
    classical_cm_energy,
    classical_scattered_energy_cm,
    classical_recoil_energy_cm,
    classical_max_energy_transfer,
    classical_scattered_energy_lab,
    classical_recoil_energy_lab,
    classical_cm_to_lab_angle,
    classical_lab_to_cm_angle,
    classical_recoil_angle_from_cm,
    # neutron helpers
    neutron_scattered_energy,
    neutron_recoil_energy,
    neutron_max_energy_transfer,
    neutron_energy_after_moderation,
    # relativistic
    relativistic_invariant_mass,
    relativistic_cm_momentum,
    relativistic_cm_lorentz_boost,
    relativistic_scattered_energy,
    relativistic_recoil_energy,
    relativistic_scattered_lab_angle,
    relativistic_recoil_lab_angle,
    relativistic_max_energy_transfer,
    # reaction
    q_value,
    threshold_energy,
)


# ---------------------------------------------------------------------------
# Classical kinematics
# ---------------------------------------------------------------------------

class TestClassicalCmEnergy:
    def test_equal_masses(self):
        """Half the lab energy goes into CoM for equal masses."""
        assert classical_cm_energy(1.0, 1.0, 1.0) == pytest.approx(0.5)

    def test_heavy_target(self):
        """Neutron (m=1) on carbon-12 (M=12)."""
        assert classical_cm_energy(13.0, 1.0, 12.0) == pytest.approx(12.0)

    def test_zero_energy(self):
        assert classical_cm_energy(0.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            classical_cm_energy(-1.0, 1.0, 1.0)

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError):
            classical_cm_energy(1.0, 0.0, 1.0)


class TestClassicalScatteredEnergyCm:
    def test_forward_scattering(self):
        """θ_cm=0 → no energy loss for any mass ratio."""
        assert classical_scattered_energy_cm(1.0, 1.0, 12.0, 0.0) == pytest.approx(1.0)

    def test_backscatter_equal_masses(self):
        """θ_cm=π, equal masses → zero energy for scattered particle."""
        assert classical_scattered_energy_cm(1.0, 1.0, 1.0, math.pi) == pytest.approx(0.0)

    def test_backscatter_neutron_carbon12(self):
        """θ_cm=π, n on ¹²C → known result ((A-1)/(A+1))² × T."""
        A = 12.0
        T = 1.0
        expected = T * ((A - 1) / (A + 1)) ** 2
        assert classical_scattered_energy_cm(T, 1.0, A, math.pi) == pytest.approx(expected)

    def test_energy_conservation_at_all_angles(self):
        """T_scattered + T_recoil = T for all CoM angles."""
        T, m1, m2 = 2.5, 1.0, 4.0
        for theta in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]:
            t_scat = classical_scattered_energy_cm(T, m1, m2, theta)
            t_rec = classical_recoil_energy_cm(T, m1, m2, theta)
            assert t_scat + t_rec == pytest.approx(T, rel=1e-12)

    def test_invalid_angle(self):
        with pytest.raises(ValueError):
            classical_scattered_energy_cm(1.0, 1.0, 1.0, -0.1)

    def test_angle_above_pi_raises(self):
        with pytest.raises(ValueError):
            classical_scattered_energy_cm(1.0, 1.0, 1.0, math.pi + 0.1)


class TestClassicalRecoilEnergyCm:
    def test_forward_scattering_zero_recoil(self):
        assert classical_recoil_energy_cm(1.0, 1.0, 12.0, 0.0) == pytest.approx(0.0)

    def test_backscatter_equal_masses_full_transfer(self):
        """Head-on n-p collision: all kinetic energy transfers to proton."""
        assert classical_recoil_energy_cm(1.0, 1.0, 1.0, math.pi) == pytest.approx(1.0)

    def test_backscatter_neutron_carbon12(self):
        A = 12.0
        T = 1.0
        expected = T * 4.0 * A / (1.0 + A) ** 2
        assert classical_recoil_energy_cm(T, 1.0, A, math.pi) == pytest.approx(expected)


class TestClassicalMaxEnergyTransfer:
    def test_equal_masses_full_transfer(self):
        assert classical_max_energy_transfer(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_neutron_carbon12(self):
        A = 12.0
        expected = 4.0 * A / (1.0 + A) ** 2
        assert classical_max_energy_transfer(1.0, 1.0, A) == pytest.approx(expected)

    def test_matches_recoil_at_pi(self):
        T, m1, m2 = 3.0, 1.0, 6.0
        assert classical_max_energy_transfer(T, m1, m2) == pytest.approx(
            classical_recoil_energy_cm(T, m1, m2, math.pi)
        )


class TestClassicalScatteredEnergyLab:
    def test_forward_no_energy_loss(self):
        """θ_lab=0 → full energy retained."""
        result = classical_scattered_energy_lab(1.0, 1.0, 12.0, 0.0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_right_angle_equal_masses(self):
        """Equal masses: θ_lab=π/2 → zero scattered energy (kinematically)."""
        result = classical_scattered_energy_lab(1.0, 1.0, 1.0, math.pi / 2)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_heavy_projectile_raises_at_max_angle(self):
        """m1 > m2: raises beyond max allowed lab angle."""
        with pytest.raises(ValueError):
            classical_scattered_energy_lab(1.0, 4.0, 1.0, math.pi / 2)

    def test_negative_angle_raises(self):
        with pytest.raises(ValueError):
            classical_scattered_energy_lab(1.0, 1.0, 1.0, -0.1)

    def test_energy_consistency_light_projectile(self):
        """Cross-check lab angle formula against CoM formula via angle conversion."""
        T, m1, m2 = 1.0, 1.0, 4.0
        theta_cm = math.pi / 3
        theta_lab = classical_cm_to_lab_angle(theta_cm, m1, m2)
        T_from_lab = classical_scattered_energy_lab(T, m1, m2, theta_lab)
        T_from_cm = classical_scattered_energy_cm(T, m1, m2, theta_cm)
        assert T_from_lab == pytest.approx(T_from_cm, rel=1e-6)


class TestClassicalRecoilEnergyLab:
    def test_head_on_full_transfer_equal_masses(self):
        """φ_lab=0, equal masses → full energy transfer."""
        assert classical_recoil_energy_lab(1.0, 1.0, 1.0, 0.0) == pytest.approx(1.0)

    def test_ninety_degrees_zero_recoil(self):
        """φ_lab=π/2 → zero recoil energy."""
        assert classical_recoil_energy_lab(1.0, 1.0, 12.0, math.pi / 2) == pytest.approx(0.0)

    def test_invalid_angle_raises(self):
        with pytest.raises(ValueError):
            classical_recoil_energy_lab(1.0, 1.0, 1.0, math.pi)


class TestClassicalAngleConversions:
    def test_cm_to_lab_equal_masses_pi_over_2(self):
        """Equal masses: θ_cm=π/2 → θ_lab=π/4."""
        result = classical_cm_to_lab_angle(math.pi / 2, 1.0, 1.0)
        assert result == pytest.approx(math.pi / 4, rel=1e-12)

    def test_cm_to_lab_forward(self):
        """θ_cm=0 → θ_lab=0."""
        assert classical_cm_to_lab_angle(0.0, 1.0, 4.0) == pytest.approx(0.0)

    def test_cm_to_lab_backward(self):
        """θ_cm=π, light projectile (m1<m2) → θ_lab=π (projectile can backscatter)."""
        result = classical_cm_to_lab_angle(math.pi, 1.0, 4.0)
        assert result == pytest.approx(math.pi)

    def test_lab_to_cm_round_trip(self):
        """lab_to_cm(cm_to_lab(θ_cm)) == θ_cm for various angles."""
        m1, m2 = 1.0, 3.0
        for theta_cm in [0.1, 0.5, 1.0, math.pi / 2, 2.0, math.pi - 0.01]:
            theta_lab = classical_cm_to_lab_angle(theta_cm, m1, m2)
            recovered = classical_lab_to_cm_angle(theta_lab, m1, m2)
            assert recovered == pytest.approx(theta_cm, abs=1e-8), (
                f"Round-trip failed at theta_cm={theta_cm}"
            )

    def test_lab_to_cm_equal_masses_pi_over_4(self):
        """Equal masses: θ_lab=π/4 → θ_cm=π/2."""
        result = classical_lab_to_cm_angle(math.pi / 4, 1.0, 1.0)
        assert result == pytest.approx(math.pi / 2, abs=1e-8)

    def test_invalid_cm_angle_raises(self):
        with pytest.raises(ValueError):
            classical_cm_to_lab_angle(-0.1, 1.0, 1.0)


class TestClassicalRecoilAngle:
    def test_head_on_forward_recoil(self):
        """θ_cm=π → φ_lab=0 (recoil straight forward)."""
        assert classical_recoil_angle_from_cm(math.pi) == pytest.approx(0.0)

    def test_forward_scatter_recoil_pi_over_2(self):
        """θ_cm=0 (glancing) → φ_lab=π/2."""
        assert classical_recoil_angle_from_cm(0.0) == pytest.approx(math.pi / 2)

    def test_pi_over_2_gives_pi_over_4(self):
        assert classical_recoil_angle_from_cm(math.pi / 2) == pytest.approx(math.pi / 4)


# ---------------------------------------------------------------------------
# Neutron convenience functions
# ---------------------------------------------------------------------------

class TestNeutronScatteredEnergy:
    def test_hydrogen_backscatter_zero(self):
        """Neutron backscattered from ¹H → zero energy."""
        assert neutron_scattered_energy(1.0, 1, math.pi) == pytest.approx(0.0)

    def test_forward_scatter_full_energy(self):
        assert neutron_scattered_energy(2.0, 12, 0.0) == pytest.approx(2.0)

    def test_matches_classical_function(self):
        T, A, theta = 5.0, 56.0, math.pi / 3
        expected = classical_scattered_energy_cm(T, 1.0, A, theta)
        assert neutron_scattered_energy(T, A, theta) == pytest.approx(expected)


class TestNeutronRecoilEnergy:
    def test_hydrogen_backscatter_full_transfer(self):
        assert neutron_recoil_energy(1.0, 1, math.pi) == pytest.approx(1.0)

    def test_forward_scatter_zero_recoil(self):
        assert neutron_recoil_energy(1.0, 12, 0.0) == pytest.approx(0.0)


class TestNeutronMaxEnergyTransfer:
    def test_hydrogen_full_transfer(self):
        assert neutron_max_energy_transfer(1.0, 1) == pytest.approx(1.0)

    def test_carbon12(self):
        A = 12.0
        expected = 4.0 * A / (1 + A) ** 2
        assert neutron_max_energy_transfer(1.0, A) == pytest.approx(expected)

    def test_heavy_nucleus_small_transfer(self):
        """Heavy nucleus: small energy transfer per collision."""
        assert neutron_max_energy_transfer(1.0, 238) < 0.02


class TestNeutronEnergyAfterModeration:
    def test_zero_collisions_unchanged(self):
        assert neutron_energy_after_moderation(1.0, 12, 0) == pytest.approx(1.0)

    def test_hydrogen_one_collision(self):
        """Isotropic scattering from ¹H: average 50 % energy loss per collision."""
        assert neutron_energy_after_moderation(1.0, 1, 1) == pytest.approx(0.5)

    def test_hydrogen_two_collisions(self):
        assert neutron_energy_after_moderation(1.0, 1, 2) == pytest.approx(0.25)

    def test_anisotropic_scattering(self):
        """avg_cos_cm > 0 → less energy loss (more forward-peaked scattering)."""
        iso = neutron_energy_after_moderation(1.0, 4, 1, avg_cos_cm=0.0)
        fwd = neutron_energy_after_moderation(1.0, 4, 1, avg_cos_cm=0.5)
        assert fwd > iso

    def test_invalid_n_collisions(self):
        with pytest.raises(ValueError):
            neutron_energy_after_moderation(1.0, 12, -1)

    def test_invalid_cos_cm(self):
        with pytest.raises(ValueError):
            neutron_energy_after_moderation(1.0, 12, 1, avg_cos_cm=1.5)


# ---------------------------------------------------------------------------
# Relativistic kinematics
# ---------------------------------------------------------------------------

NEUTRON_MEV = 939.56542052
PROTON_MEV = 938.27208816
DEUTERON_MEV = 1875.612928


class TestRelativisticInvariantMass:
    def test_zero_energy_at_rest(self):
        """T=0: √s = m1 + m2."""
        sqrt_s = relativistic_invariant_mass(0.0, NEUTRON_MEV, PROTON_MEV)
        assert sqrt_s == pytest.approx(NEUTRON_MEV + PROTON_MEV)

    def test_increases_with_energy(self):
        s1 = relativistic_invariant_mass(100.0, NEUTRON_MEV, NEUTRON_MEV)
        s2 = relativistic_invariant_mass(1000.0, NEUTRON_MEV, NEUTRON_MEV)
        assert s2 > s1

    def test_negative_T_raises(self):
        with pytest.raises(ValueError):
            relativistic_invariant_mass(-1.0, NEUTRON_MEV, PROTON_MEV)


class TestRelativisticCmMomentum:
    def test_zero_T_nonzero_unequal_masses(self):
        """At T=0 the CoM momentum is zero."""
        p = relativistic_cm_momentum(0.0, NEUTRON_MEV, PROTON_MEV)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_increases_with_energy(self):
        p1 = relativistic_cm_momentum(100.0, NEUTRON_MEV, PROTON_MEV)
        p2 = relativistic_cm_momentum(1000.0, NEUTRON_MEV, PROTON_MEV)
        assert p2 > p1


class TestRelativisticLorentzBoost:
    def test_beta_in_range(self):
        beta, gamma = relativistic_cm_lorentz_boost(100.0, NEUTRON_MEV, PROTON_MEV)
        assert 0.0 < beta < 1.0

    def test_gamma_greater_than_one(self):
        beta, gamma = relativistic_cm_lorentz_boost(100.0, NEUTRON_MEV, PROTON_MEV)
        assert gamma > 1.0

    def test_beta_gamma_relation(self):
        """β² + 1/γ² = 1."""
        beta, gamma = relativistic_cm_lorentz_boost(500.0, NEUTRON_MEV, PROTON_MEV)
        assert beta**2 + 1.0 / gamma**2 == pytest.approx(1.0, rel=1e-10)


class TestRelativisticScatteredEnergy:
    def test_forward_no_energy_loss(self):
        """θ_cm=0 → projectile retains its full kinetic energy."""
        T = 100.0
        result = relativistic_scattered_energy(T, NEUTRON_MEV, PROTON_MEV, 0.0)
        assert result == pytest.approx(T, rel=1e-8)

    def test_energy_conservation(self):
        """T_scattered + T_recoil = T for all CoM angles."""
        T, m1, m2 = 200.0, NEUTRON_MEV, PROTON_MEV
        for theta in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]:
            t_s = relativistic_scattered_energy(T, m1, m2, theta)
            t_r = relativistic_recoil_energy(T, m1, m2, theta)
            assert t_s + t_r == pytest.approx(T, rel=1e-8), (
                f"Energy not conserved at theta_cm={theta:.3f}"
            )

    def test_monotone_decrease_with_angle(self):
        """Scattered energy decreases as CoM angle increases."""
        T, m1, m2 = 100.0, NEUTRON_MEV, PROTON_MEV
        angles = [0.0, math.pi / 6, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi]
        energies = [relativistic_scattered_energy(T, m1, m2, a) for a in angles]
        assert all(energies[i] >= energies[i + 1] for i in range(len(energies) - 1))

    def test_backscatter_equals_classical_low_energy(self):
        """At very low energies (T << mc²) relativistic ≈ classical."""
        T = 0.1  # MeV (much less than ~939 MeV)
        m = NEUTRON_MEV
        theta_cm = math.pi
        rel = relativistic_scattered_energy(T, m, m, theta_cm)
        clas = classical_scattered_energy_cm(T, 1.0, 1.0, theta_cm)
        # Both should be near zero for equal masses at backscatter
        assert rel == pytest.approx(clas, rel=1e-3)

    def test_invalid_angle_raises(self):
        with pytest.raises(ValueError):
            relativistic_scattered_energy(100.0, NEUTRON_MEV, PROTON_MEV, -0.1)


class TestRelativisticRecoilEnergy:
    def test_forward_zero_recoil(self):
        """θ_cm=0 → no recoil."""
        result = relativistic_recoil_energy(100.0, NEUTRON_MEV, PROTON_MEV, 0.0)
        assert result == pytest.approx(0.0, abs=1e-7)

    def test_backscatter_max_recoil(self):
        """θ_cm=π → maximum recoil."""
        T = 100.0
        r_pi = relativistic_recoil_energy(T, NEUTRON_MEV, PROTON_MEV, math.pi)
        r_half = relativistic_recoil_energy(T, NEUTRON_MEV, PROTON_MEV, math.pi / 2)
        assert r_pi > r_half


class TestRelativisticMaxEnergyTransfer:
    def test_larger_than_zero(self):
        result = relativistic_max_energy_transfer(100.0, NEUTRON_MEV, PROTON_MEV)
        assert result > 0.0

    def test_matches_recoil_at_pi(self):
        T, m1, m2 = 500.0, NEUTRON_MEV, PROTON_MEV
        assert relativistic_max_energy_transfer(T, m1, m2) == pytest.approx(
            relativistic_recoil_energy(T, m1, m2, math.pi)
        )


class TestRelativisticLabAngles:
    def test_forward_scatter_zero_lab_angle(self):
        assert relativistic_scattered_lab_angle(
            100.0, NEUTRON_MEV, PROTON_MEV, 0.0
        ) == pytest.approx(0.0)

    def test_backscatter_zero_recoil_lab_angle(self):
        """θ_cm=π → target recoils straight forward (φ_lab=0)."""
        result = relativistic_recoil_lab_angle(
            100.0, NEUTRON_MEV, PROTON_MEV, math.pi
        )
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_lab_angles_in_range(self):
        T, m1, m2 = 100.0, NEUTRON_MEV, PROTON_MEV
        for theta in [0.1, 0.5, 1.0, 1.5, 2.0, math.pi - 0.1]:
            scat_angle = relativistic_scattered_lab_angle(T, m1, m2, theta)
            rec_angle = relativistic_recoil_lab_angle(T, m1, m2, theta)
            assert 0.0 <= scat_angle <= math.pi / 2
            assert 0.0 <= rec_angle <= math.pi / 2


# ---------------------------------------------------------------------------
# Q-value and threshold energy
# ---------------------------------------------------------------------------

class TestQValue:
    def test_elastic_scattering_zero_q(self):
        """Elastic scattering: products same as reactants → Q = 0."""
        assert q_value(
            [NEUTRON_MEV, PROTON_MEV], [NEUTRON_MEV, PROTON_MEV]
        ) == pytest.approx(0.0)

    def test_exothermic_positive_q(self):
        """Positive Q when product masses are less than reactant masses."""
        # Use simple numbers
        q = q_value([10.0, 5.0], [12.0])
        assert q == pytest.approx(3.0)

    def test_endothermic_negative_q(self):
        q = q_value([10.0], [12.0])
        assert q == pytest.approx(-2.0)

    def test_neutron_capture_approximate(self):
        """n + ¹H → ²H: Q ≈ 2.22 MeV."""
        # Masses: n=939.565, p=938.272, d=1875.613
        q = q_value([939.56542052, 938.27208816], [1875.612928, 0.0])
        assert abs(q - 2.224) < 0.01

    def test_empty_initial_raises(self):
        with pytest.raises(ValueError):
            q_value([], [NEUTRON_MEV])

    def test_empty_final_raises(self):
        with pytest.raises(ValueError):
            q_value([NEUTRON_MEV], [])

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError):
            q_value([NEUTRON_MEV], [-1.0])


class TestThresholdEnergy:
    def test_elastic_scattering_zero_threshold(self):
        """Elastic: products = reactants → no threshold."""
        assert threshold_energy(
            NEUTRON_MEV, PROTON_MEV, [NEUTRON_MEV, PROTON_MEV]
        ) == pytest.approx(0.0)

    def test_exothermic_zero_threshold(self):
        """Exothermic reaction always has zero threshold."""
        assert threshold_energy(1.0, 1.0, [1.5]) == pytest.approx(0.0)

    def test_endothermic_positive_threshold(self):
        """p + p → p + p + π⁰ requires threshold."""
        pi0_mev = 134.976
        T_th = threshold_energy(PROTON_MEV, PROTON_MEV, [PROTON_MEV, PROTON_MEV, pi0_mev])
        assert T_th > 0.0

    def test_known_pion_threshold(self):
        """p + p → p + p + π⁰: T_th = 2 m_pi + m_pi²/(2 m_p) ≈ 280 MeV (approx)."""
        pi0_mev = 134.976
        T_th = threshold_energy(PROTON_MEV, PROTON_MEV, [PROTON_MEV, PROTON_MEV, pi0_mev])
        # Standard textbook value ~280 MeV
        assert 270.0 < T_th < 290.0

    def test_zero_product_mass_raises(self):
        with pytest.raises(ValueError):
            threshold_energy(NEUTRON_MEV, PROTON_MEV, [0.0, PROTON_MEV])

    def test_empty_products_raises(self):
        with pytest.raises(ValueError):
            threshold_energy(NEUTRON_MEV, PROTON_MEV, [])
