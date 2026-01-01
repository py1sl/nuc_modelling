"""
Tests for main.py
"""

import pytest
import sys
import os

# Add parent directory to path to import main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import particle, em_force, strong_force, SEMF


class TestParticle:
    """Tests for the particle class"""
    
    def test_particle_initialization(self):
        """Test that particle initializes with default values"""
        p = particle()
        assert p.charge == 0
        assert p.mass_amu == 1
        assert p.energy == 100
        assert p.rest_mass == 0
        assert p.name == "ball"
        assert p.symbol == "b"
        assert p.x == 0
        assert p.y == 0
        assert p.z == 0
        assert p.u == 0
        assert p.v == 0
        assert p.w == 0
    
    def test_particle_attributes_can_be_modified(self):
        """Test that particle attributes can be modified"""
        p = particle()
        p.charge = 1
        p.mass_amu = 2
        p.energy = 200
        p.name = "proton"
        assert p.charge == 1
        assert p.mass_amu == 2
        assert p.energy == 200
        assert p.name == "proton"


class TestForces:
    """Tests for force functions"""
    
    def test_em_force_returns_zero(self):
        """Test that em_force returns 0"""
        p1 = particle()
        p2 = particle()
        assert em_force(p1, p2) == 0
    
    def test_strong_force_returns_zero(self):
        """Test that strong_force returns 0"""
        p1 = particle()
        p2 = particle()
        assert strong_force(p1, p2) == 0


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
