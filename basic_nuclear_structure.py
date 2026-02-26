"""
Basic nuclear structure module for nuclear modelling.

This module provides:
- SEMF: Semi-Empirical Mass Formula (Bethe-Weizsäcker) for binding energy per nucleon.
- binding_energy: Total binding energy of a nucleus using SEMF.
- neutron_separation_energy: Energy required to remove one neutron from a nucleus.
- proton_separation_energy: Energy required to remove one proton from a nucleus.
- main: Example usage demonstrating the module.
"""

from particle import particle


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


def main():
    """Demonstrate basic nuclear structure calculations."""
    p = particle()
    print(p)

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
