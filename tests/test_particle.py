"""
Tests for particle.py
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from particle import particle, neutron, proton, electron
from nuclear_constants import PROTON_MASS, NEUTRON_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE


class TestParticle:
    """Tests for the particle base class"""

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

    def test_particle_str(self):
        """Test that __str__ returns expected string"""
        p = particle()
        result = str(p)
        assert "ball" in result
        assert "b" in result
        assert "charge=0" in result


class TestNeutron:
    """Tests for the neutron child class"""

    def test_neutron_initialization(self):
        """Test neutron initializes with correct properties"""
        n = neutron()
        assert n.charge == 0
        assert n.rest_mass == NEUTRON_MASS
        assert n.name == "neutron"
        assert n.symbol == "n"

    def test_neutron_is_particle(self):
        """Test neutron is an instance of particle"""
        n = neutron()
        assert isinstance(n, particle)

    def test_neutron_str(self):
        """Test neutron __str__ includes name and symbol"""
        n = neutron()
        result = str(n)
        assert "neutron" in result
        assert "n" in result


class TestProton:
    """Tests for the proton child class"""

    def test_proton_initialization(self):
        """Test proton initializes with correct properties"""
        p = proton()
        assert p.charge == ELEMENTARY_CHARGE
        assert p.rest_mass == PROTON_MASS
        assert p.name == "proton"
        assert p.symbol == "p"

    def test_proton_is_particle(self):
        """Test proton is an instance of particle"""
        p = proton()
        assert isinstance(p, particle)

    def test_proton_str(self):
        """Test proton __str__ includes name and symbol"""
        p = proton()
        result = str(p)
        assert "proton" in result
        assert "p" in result


class TestElectron:
    """Tests for the electron child class"""

    def test_electron_initialization(self):
        """Test electron initializes with correct properties"""
        e = electron()
        assert e.charge == pytest.approx(-ELEMENTARY_CHARGE)
        assert e.rest_mass == ELECTRON_MASS
        assert e.name == "electron"
        assert e.symbol == "e"

    def test_electron_is_particle(self):
        """Test electron is an instance of particle"""
        e = electron()
        assert isinstance(e, particle)

    def test_electron_str(self):
        """Test electron __str__ includes name and symbol"""
        e = electron()
        result = str(e)
        assert "electron" in result
        assert "e" in result
