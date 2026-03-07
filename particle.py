"""
Particle module for nuclear modelling.

This module provides:
- particle: A base class representing a nuclear particle.
- neutron: A neutron particle with properties from nuclear constants.
- proton: A proton particle with properties from nuclear constants.
- electron: An electron particle with properties from nuclear constants.
- load_particle_data: Load particle definitions from the bundled JSON data file.
"""

import json
import os

from nuclear_constants import PROTON_MASS, NEUTRON_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE

_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "particles.json")
_PARTICLE_DATA_CACHE = None


def load_particle_data():
    """Load and return all particle definitions from the JSON data file.

    The result is cached after the first read so subsequent calls do not
    incur repeated file I/O.

    Returns
    -------
    list[dict]
        A list of particle dictionaries, each containing keys: name,
        pdg_symbol, pdg_number, mass_kg, mass_mev, charge, spin, type.

    Raises
    ------
    FileNotFoundError
        If the particles.json data file cannot be found.
    """
    global _PARTICLE_DATA_CACHE
    if _PARTICLE_DATA_CACHE is None:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _PARTICLE_DATA_CACHE = data["particles"]
    return _PARTICLE_DATA_CACHE


class particle:
    """Represents a nuclear particle with basic properties and phase-space coordinates."""

    def __init__(self):
        self.charge = 0
        self.mass_amu = 1
        self.energy = 100
        self.rest_mass = 0
        self.name = "ball"
        self.symbol = "b"
        self.pdg_symbol = ""
        self.pdg_number = 0
        self.mass_mev = 0.0
        self.spin = 0.0
        self.particle_type = ""
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.v = 0
        self.w = 0

    def __str__(self):
        return f"{self.name} ({self.symbol}), charge={self.charge}, mass={self.mass_amu} amu"

    @classmethod
    def from_name(cls, name):
        """Create a particle instance populated with data from the JSON data file.

        Parameters
        ----------
        name : str
            The particle name as it appears in particles.json (e.g. "neutron",
            "proton", "electron", "alpha").

        Returns
        -------
        particle
            A new particle instance with properties set from the JSON data.
            The ``rest_mass`` attribute is set to ``mass_kg``, ``symbol`` to
            ``pdg_symbol``, and ``charge`` is in Coulombs (elementary charge
            units multiplied by ELEMENTARY_CHARGE).

        Raises
        ------
        ValueError
            If no particle with the given name is found in the data file.
        """
        particles = load_particle_data()
        entry = next((p for p in particles if p["name"] == name), None)
        if entry is None:
            available = ", ".join(p["name"] for p in particles)
            raise ValueError(
                f"Unknown particle '{name}'. Available particles: {available}"
            )
        instance = cls()
        instance.name = entry["name"]
        instance.symbol = entry["pdg_symbol"]
        instance.pdg_symbol = entry["pdg_symbol"]
        instance.pdg_number = entry["pdg_number"]
        instance.rest_mass = entry["mass_kg"]
        instance.mass_mev = entry["mass_mev"]
        instance.charge = entry["charge"] * ELEMENTARY_CHARGE
        instance.spin = entry["spin"]
        instance.particle_type = entry["type"]
        return instance


class neutron(particle):
    """Represents a neutron using nuclear constants."""

    def __init__(self):
        super().__init__()
        self.charge = 0
        self.rest_mass = NEUTRON_MASS
        self.name = "neutron"
        self.symbol = "n"


class proton(particle):
    """Represents a proton using nuclear constants."""

    def __init__(self):
        super().__init__()
        self.charge = ELEMENTARY_CHARGE
        self.rest_mass = PROTON_MASS
        self.name = "proton"
        self.symbol = "p"


class electron(particle):
    """Represents an electron using nuclear constants."""

    def __init__(self):
        super().__init__()
        self.charge = -ELEMENTARY_CHARGE
        self.rest_mass = ELECTRON_MASS
        self.name = "electron"
        self.symbol = "e"
