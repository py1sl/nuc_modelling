"""
Basic nuclear structure module for nuclear modelling.

This module provides:
- SEMF: Semi-Empirical Mass Formula (Bethe-Weizsäcker) for binding energy per nucleon.
- binding_energy: Total binding energy of a nucleus using SEMF.
- neutron_separation_energy: Energy required to remove one neutron from a nucleus.
- proton_separation_energy: Energy required to remove one proton from a nucleus.
- bohr_hydrogen_state: Radius, speed, and energy for hydrogen Bohr levels.
- hydrogen_transition_wavelength: Emission wavelength between hydrogen levels.
- main: Example usage demonstrating the module.
"""

import math

from particle import particle
from nuclear_constants import (
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    HBAR,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
)


BOHR_RADIUS = (
    4 * math.pi * VACUUM_PERMITTIVITY * HBAR**2
    / (ELECTRON_MASS * ELEMENTARY_CHARGE**2)
)
FINE_STRUCTURE_CONSTANT = (
    ELEMENTARY_CHARGE**2
    / (4 * math.pi * VACUUM_PERMITTIVITY * HBAR * SPEED_OF_LIGHT)
)
RYDBERG_ENERGY_EV = (
    ELECTRON_MASS * ELEMENTARY_CHARGE**4
    / (8 * VACUUM_PERMITTIVITY**2 * PLANCK_CONSTANT**2 * ELEMENTARY_CHARGE)
)
RYDBERG_ENERGY_J = RYDBERG_ENERGY_EV * ELEMENTARY_CHARGE


def _validate_positive_integer(value, name):
    """Validate that an input is a positive integer."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def SEMF(n, z):
    """
    Semi-Empirical Mass Formula (Bethe-Weizsäcker formula).

    Calculates the binding energy per nucleon for a nucleus with n neutrons
    and z protons using the liquid-drop model coefficients (in MeV).

    Args:
        n: Number of neutrons.
        z: Number of protons.

    Returns:
        Binding energy per nucleon in MeV.
    """
    # SEMF coefficients (MeV)
    av = 15.8    # volume term
    as_ = 18.3   # surface term
    ac = 0.714   # Coulomb term
    aa = 23.2    # asymmetry term
    ap = 12.0    # pairing term

    A = n + z    # mass number

    if A == 0:
        return 0

    # Volume term
    binding_energy = av * A

    # Surface term
    binding_energy -= as_ * A ** (2.0 / 3.0)

    # Coulomb term
    binding_energy -= ac * z * (z - 1) / A ** (1.0 / 3.0)

    # Asymmetry term
    binding_energy -= aa * (A - 2 * z) ** 2 / A

    # Pairing term
    if n % 2 == 0 and z % 2 == 0:
        # Even-even: positive contribution
        binding_energy += ap / A ** 0.5
    elif n % 2 != 0 and z % 2 != 0:
        # Odd-odd: negative contribution
        binding_energy -= ap / A ** 0.5
    # else: even-odd or odd-even: no pairing contribution

    return binding_energy / A


def binding_energy(n, z):
    """
    Calculate the total binding energy of a nucleus using the SEMF.

    Args:
        n: Number of neutrons.
        z: Number of protons.

    Returns:
        Total binding energy in MeV.
    """
    A = n + z
    return SEMF(n, z) * A


def neutron_separation_energy(n, z):
    """
    Calculate the neutron separation energy (S_n) of a nucleus.

    S_n is the energy required to remove one neutron from the nucleus:
        S_n(N, Z) = B(N, Z) - B(N-1, Z)

    Args:
        n: Number of neutrons (must be >= 1).
        z: Number of protons.

    Returns:
        Neutron separation energy in MeV, or 0 if n < 1.
    """
    if n < 1:
        return 0
    return binding_energy(n, z) - binding_energy(n - 1, z)


def proton_separation_energy(n, z):
    """
    Calculate the proton separation energy (S_p) of a nucleus.

    S_p is the energy required to remove one proton from the nucleus:
        S_p(N, Z) = B(N, Z) - B(N, Z-1)

    Args:
        n: Number of neutrons.
        z: Number of protons (must be >= 1).

    Returns:
        Proton separation energy in MeV, or 0 if z < 1.
    """
    if z < 1:
        return 0
    return binding_energy(n, z) - binding_energy(n, z - 1)


def bohr_hydrogen_state(n, z=1):
    """
    Return Bohr-model properties for a hydrogen-like atom in level n.

    Args:
        n: Principal quantum number (n >= 1).
        z: Atomic number for a one-electron ion (z >= 1). Default is 1.

    Returns:
        Dictionary with:
            - radius_m: orbital radius in meters.
            - speed_m_s: orbital speed in m/s.
            - energy_ev: total bound-state energy in eV (negative).
    """
    _validate_positive_integer(n, "n")
    _validate_positive_integer(z, "z")

    radius_m = BOHR_RADIUS * (n**2) / z
    speed_m_s = SPEED_OF_LIGHT * FINE_STRUCTURE_CONSTANT * z / n
    energy_ev = -RYDBERG_ENERGY_EV * (z**2) / (n**2)

    return {
        "radius_m": radius_m,
        "speed_m_s": speed_m_s,
        "energy_ev": energy_ev,
    }


def hydrogen_transition_wavelength(n_initial, n_final):
    """
    Emission wavelength (meters) for hydrogen transition n_initial -> n_final.

    Args:
        n_initial: Initial principal quantum number (must be > n_final).
        n_final: Final principal quantum number (must be >= 1).

    Returns:
        Emission wavelength in meters.
    """
    _validate_positive_integer(n_initial, "n_initial")
    _validate_positive_integer(n_final, "n_final")
    if n_initial <= n_final:
        raise ValueError("n_initial must be greater than n_final for emission")

    initial_energy_j = -RYDBERG_ENERGY_J / (n_initial**2)
    final_energy_j = -RYDBERG_ENERGY_J / (n_final**2)
    delta_e_j = initial_energy_j - final_energy_j
    return PLANCK_CONSTANT * SPEED_OF_LIGHT / delta_e_j


def main():
    """Demonstrate basic nuclear structure calculations."""
    p = particle()
    print(p)
    h1 = bohr_hydrogen_state(1)
    print(
        "Hydrogen Bohr n=1: "
        f"r={h1['radius_m']:.3e} m, "
        f"v={h1['speed_m_s']:.3e} m/s, "
        f"E={h1['energy_ev']:.4f} eV"
    )
    h_alpha_nm = hydrogen_transition_wavelength(3, 2) * 1e9
    print(f"Hydrogen H-alpha (3->2): {h_alpha_nm:.2f} nm")

    # Binding energy per nucleon for some example nuclei
    for (n, z, label) in [(6, 6, "C-12"), (10, 10, "Ne-20"), (82, 50, "Sn-132")]:
        be = SEMF(n, z)
        print(f"SEMF({n}, {z}) [{label}]: {be:.4f} MeV/nucleon")
        total_be = binding_energy(n, z)
        print(f"Total binding energy ({label}): {total_be:.4f} MeV")
        sn = neutron_separation_energy(n, z)
        print(f"Neutron separation energy ({label}): {sn:.4f} MeV")
        sp = proton_separation_energy(n, z)
        print(f"Proton separation energy ({label}): {sp:.4f} MeV")


if __name__ == "__main__":
    main()
