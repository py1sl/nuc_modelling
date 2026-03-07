"""
Tests for particle.py
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from particle import particle, neutron, proton, electron, load_particle_data
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


class TestLoadParticleData:
    """Tests for the load_particle_data function"""

    def test_returns_list(self):
        """Test that load_particle_data returns a list"""
        data = load_particle_data()
        assert isinstance(data, list)

    def test_list_is_not_empty(self):
        """Test that the particle list is not empty"""
        data = load_particle_data()
        assert len(data) > 0

    def test_each_entry_has_required_keys(self):
        """Test that every particle entry contains the required fields"""
        required_keys = {
            "name", "pdg_symbol", "pdg_number",
            "mass_kg", "mass_mev", "charge", "spin", "type"
        }
        for entry in load_particle_data():
            assert required_keys.issubset(entry.keys()), (
                f"Particle '{entry.get('name', '?')}' is missing required keys"
            )

    def test_known_particles_present(self):
        """Test that key particles are present in the data"""
        names = {p["name"] for p in load_particle_data()}
        for expected in ("neutron", "proton", "electron", "photon", "alpha"):
            assert expected in names


class TestParticleFromName:
    """Tests for the particle.from_name() classmethod"""

    def test_from_name_returns_particle_instance(self):
        """Test that from_name returns a particle instance"""
        p = particle.from_name("neutron")
        assert isinstance(p, particle)

    def test_from_name_neutron(self):
        """Test neutron created from JSON has correct properties"""
        p = particle.from_name("neutron")
        assert p.name == "neutron"
        assert p.pdg_symbol == "n"
        assert p.pdg_number == 2112
        assert p.charge == pytest.approx(0.0)
        assert p.rest_mass == pytest.approx(1.67492749804e-27)
        assert p.mass_mev == pytest.approx(939.56542052)
        assert p.spin == pytest.approx(0.5)
        assert p.particle_type == "baryon"

    def test_from_name_proton(self):
        """Test proton created from JSON has correct properties"""
        p = particle.from_name("proton")
        assert p.name == "proton"
        assert p.pdg_symbol == "p"
        assert p.pdg_number == 2212
        assert p.charge == pytest.approx(ELEMENTARY_CHARGE)
        assert p.rest_mass == pytest.approx(PROTON_MASS)
        assert p.mass_mev == pytest.approx(938.27208816)
        assert p.particle_type == "baryon"

    def test_from_name_electron(self):
        """Test electron created from JSON has correct properties"""
        p = particle.from_name("electron")
        assert p.name == "electron"
        assert p.pdg_symbol == "e-"
        assert p.pdg_number == 11
        assert p.charge == pytest.approx(-ELEMENTARY_CHARGE)
        assert p.rest_mass == pytest.approx(ELECTRON_MASS)
        assert p.particle_type == "lepton"

    def test_from_name_photon(self):
        """Test photon created from JSON has zero mass and charge"""
        p = particle.from_name("photon")
        assert p.name == "photon"
        assert p.charge == pytest.approx(0.0)
        assert p.rest_mass == pytest.approx(0.0)
        assert p.mass_mev == pytest.approx(0.0)
        assert p.particle_type == "boson"

    def test_from_name_alpha(self):
        """Test alpha particle created from JSON has charge +2"""
        p = particle.from_name("alpha")
        assert p.name == "alpha"
        assert p.charge == pytest.approx(2 * ELEMENTARY_CHARGE)
        assert p.particle_type == "nucleus"

    def test_from_name_unknown_raises_value_error(self):
        """Test that an unknown particle name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown particle 'quark'"):
            particle.from_name("quark")

    def test_from_name_symbol_set(self):
        """Test that symbol is set to pdg_symbol from JSON"""
        p = particle.from_name("muon")
        assert p.symbol == "mu-"
        assert p.pdg_symbol == "mu-"

    def test_from_name_preserves_default_phase_space(self):
        """Test that phase-space coordinates default to zero"""
        p = particle.from_name("proton")
        assert p.x == 0
        assert p.y == 0
        assert p.z == 0
        assert p.u == 0
        assert p.v == 0
        assert p.w == 0
