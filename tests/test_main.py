"""
Tests for main module
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import particle, em_force, strong_force, SEMF


def test_particle_creation():
    """Test that a particle can be created with default values"""
    p = particle()
    assert p.charge == 0
    assert p.mass_amu == 1
    assert p.energy == 100
    assert p.name == "ball"
    assert p.symbol == "b"


def test_em_force():
    """Test electromagnetic force calculation"""
    p1 = particle()
    p2 = particle()
    force = em_force(p1, p2)
    assert force == 0


def test_strong_force():
    """Test strong force calculation"""
    p1 = particle()
    p2 = particle()
    force = strong_force(p1, p2)
    assert force == 0


def test_SEMF():
    """Test semi-empirical mass formula"""
    # Test for a simple nucleus (e.g., 20Ne: 10 neutrons, 10 protons)
    bind_erg = SEMF(10, 10)
    assert isinstance(bind_erg, float)
    assert bind_erg > 0  # Binding energy should be positive
