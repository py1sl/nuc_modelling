"""
Tests for basic_nuclear_structure.py
"""

import pytest
import sys
import os

# Add parent directory to path to import basic_nuclear_structure module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_nuclear_structure import (
    SEMF,
    binding_energy,
    neutron_separation_energy,
    proton_separation_energy,
    bohr_hydrogen_state,
    hydrogen_transition_wavelength,
)

class TestSEMF:
    """Tests for Semi-Empirical Mass Formula"""
    
    def test_semf_basic_case(self):
        """Test SEMF with n=10, z=10 (Ne-20)"""
        result = SEMF(10, 10)
        # Expected value should be positive and reasonable
        assert isinstance(result, (int, float))
        assert result > 0
        assert result < 20  # Should be less than volume term
    
    def test_semf_even_even_nucleus(self):
        """Test SEMF with even-even nucleus (both n and z are even)"""
        # Carbon-12: n=6, z=6
        result = SEMF(6, 6)
        assert isinstance(result, (int, float))
        # Even-even nuclei have positive pairing term
        assert result > 0
    
    def test_semf_odd_odd_nucleus(self):
        """Test SEMF with odd-odd nucleus (both n and z are odd)"""
        # Deuterium-like: n=1, z=1
        result = SEMF(1, 1)
        assert isinstance(result, (int, float))
        # Odd-odd nuclei have negative pairing term, may result in lower binding energy
    
    def test_semf_even_odd_nucleus(self):
        """Test SEMF with even-odd nucleus"""
        # n=6, z=7 (N-13)
        result = SEMF(6, 7)
        assert isinstance(result, (int, float))
        # Even-odd nuclei have zero pairing term
    
    def test_semf_increasing_mass_number(self):
        """Test SEMF with increasing mass numbers"""
        # As mass number increases, binding energy per nucleon should change
        result_small = SEMF(1, 1)
        result_medium = SEMF(10, 10)
        result_large = SEMF(50, 50)
        
        assert isinstance(result_small, (int, float))
        assert isinstance(result_medium, (int, float))
        assert isinstance(result_large, (int, float))
    
    def test_semf_asymmetric_nucleus(self):
        """Test SEMF with asymmetric nucleus (different n and z)"""
        # Heavy nucleus with more neutrons than protons
        result = SEMF(82, 50)  # More neutrons
        assert isinstance(result, (int, float))
        # Asymmetry term should reduce binding energy
    
    def test_semf_zero_neutrons(self):
        """Test SEMF edge case with zero neutrons"""
        # This is not physical but tests the formula
        result = SEMF(0, 1)
        assert isinstance(result, (int, float))
    
    def test_semf_zero_protons(self):
        """Test SEMF edge case with zero protons"""
        # This is not physical but tests the formula
        result = SEMF(1, 0)
        assert isinstance(result, (int, float))

    def test_semf_negative_inputs_raise(self):
        """SEMF should reject negative neutron/proton counts."""
        with pytest.raises(ValueError, match="non-negative integer"):
            SEMF(-1, 1)
        with pytest.raises(ValueError, match="non-negative integer"):
            SEMF(1, -1)


class TestBindingEnergy:
    """Tests for the total binding_energy function"""

    def test_binding_energy_equals_semf_times_A(self):
        """Total binding energy should be SEMF * A"""
        n, z = 6, 6
        assert binding_energy(n, z) == pytest.approx(SEMF(n, z) * (n + z))

    def test_binding_energy_positive(self):
        """Binding energy should be positive for typical nuclei"""
        assert binding_energy(10, 10) > 0

    def test_binding_energy_zero_A(self):
        """Binding energy for A=0 should be 0"""
        assert binding_energy(0, 0) == 0

    def test_binding_energy_heavy_nucleus(self):
        """Test binding energy for a heavy nucleus"""
        result = binding_energy(82, 50)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_binding_energy_negative_inputs_raise(self):
        """binding_energy should reject negative neutron/proton counts."""
        with pytest.raises(ValueError, match="non-negative integer"):
            binding_energy(-1, 6)
        with pytest.raises(ValueError, match="non-negative integer"):
            binding_energy(6, -1)


class TestSeparationEnergies:
    """Tests for neutron and proton separation energy functions"""

    def test_neutron_separation_energy_positive(self):
        """Neutron separation energy should be positive for stable-ish nuclei"""
        result = neutron_separation_energy(6, 6)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_proton_separation_energy_positive(self):
        """Proton separation energy should be positive for stable-ish nuclei"""
        result = proton_separation_energy(6, 6)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_neutron_separation_energy_definition(self):
        """S_n(N,Z) = B(N,Z) - B(N-1,Z)"""
        n, z = 8, 8
        expected = binding_energy(n, z) - binding_energy(n - 1, z)
        assert neutron_separation_energy(n, z) == pytest.approx(expected)

    def test_proton_separation_energy_definition(self):
        """S_p(N,Z) = B(N,Z) - B(N,Z-1)"""
        n, z = 8, 8
        expected = binding_energy(n, z) - binding_energy(n, z - 1)
        assert proton_separation_energy(n, z) == pytest.approx(expected)

    def test_neutron_separation_energy_zero_neutrons(self):
        """S_n should return 0 when n < 1"""
        assert neutron_separation_energy(0, 6) == 0

    def test_proton_separation_energy_zero_protons(self):
        """S_p should return 0 when z < 1"""
        assert proton_separation_energy(6, 0) == 0

    def test_separation_energy_negative_inputs_raise(self):
        """Separation-energy helpers should reject negative inputs."""
        with pytest.raises(ValueError, match="non-negative integer"):
            neutron_separation_energy(-1, 6)
        with pytest.raises(ValueError, match="non-negative integer"):
            neutron_separation_energy(6, -1)
        with pytest.raises(ValueError, match="non-negative integer"):
            proton_separation_energy(-1, 6)
        with pytest.raises(ValueError, match="non-negative integer"):
            proton_separation_energy(6, -1)


class TestBohrHydrogenModel:
    """Tests for Bohr-model helpers for hydrogen."""

    def test_ground_state_values_are_reasonable(self):
        """Hydrogen n=1 should match known Bohr-scale values."""
        state = bohr_hydrogen_state(1)
        assert state["radius_m"] == pytest.approx(5.29177e-11, rel=1e-4)
        assert state["speed_m_s"] == pytest.approx(2.18769e6, rel=1e-4)
        assert state["energy_ev"] == pytest.approx(-13.6057, rel=1e-4)

    def test_n2_radius_scales_with_n_squared(self):
        """r_n should scale as n² for hydrogen."""
        r1 = bohr_hydrogen_state(1)["radius_m"]
        r2 = bohr_hydrogen_state(2)["radius_m"]
        assert r2 == pytest.approx(4.0 * r1, rel=1e-12)

    def test_halpha_transition_wavelength(self):
        """H-alpha (n=3 -> 2) should be ~656.3 nm."""
        wavelength_m = hydrogen_transition_wavelength(3, 2)
        assert wavelength_m == pytest.approx(656.3e-9, rel=2e-3)

    def test_invalid_levels_raise(self):
        """Bohr helpers should reject invalid principal quantum numbers."""
        with pytest.raises(ValueError):
            bohr_hydrogen_state(0)
        with pytest.raises(ValueError):
            hydrogen_transition_wavelength(2, 2)
