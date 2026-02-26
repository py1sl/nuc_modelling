"""
Particle module for nuclear modelling.

This module provides:
- particle: A base class representing a nuclear particle.
- neutron: A neutron particle with properties from nuclear constants.
- proton: A proton particle with properties from nuclear constants.
- electron: An electron particle with properties from nuclear constants.
"""

from nuclear_constants import PROTON_MASS, NEUTRON_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE


class particle:
    """Represents a nuclear particle with basic properties and phase-space coordinates."""

    def __init__(self):
        self.charge = 0
        self.mass_amu = 1
        self.energy = 100
        self.rest_mass = 0
        self.name = "ball"
        self.symbol = "b"
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.v = 0
        self.w = 0

    def __str__(self):
        return f"{self.name} ({self.symbol}), charge={self.charge}, mass={self.mass_amu} amu"


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
