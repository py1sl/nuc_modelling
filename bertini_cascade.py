"""
Bertini Intra-Nuclear Cascade (INC) model for proton-tungsten interactions.

This module implements a simplified Bertini INC model for simulating the
interaction of protons with a tungsten target. The model is valid for proton
kinetic energies between 100 MeV and 3 GeV.

Reference:
    H. W. Bertini, Phys. Rev. 131, 1801 (1963).
"""

import math

# Tungsten (W-184) target properties
TUNGSTEN_Z = 74                          # atomic number
TUNGSTEN_A = 184                         # mass number (most abundant isotope)
TUNGSTEN_N = TUNGSTEN_A - TUNGSTEN_Z    # number of neutrons (110)

# Bertini model valid energy range (MeV)
BERTINI_MIN_ENERGY_MEV = 100.0           # 100 MeV
BERTINI_MAX_ENERGY_MEV = 3000.0          # 3 GeV

# Nuclear radius parameter (fm)
NUCLEAR_RADIUS_PARAM_FM = 1.2

# ℏc (MeV·fm), used for Fermi momentum calculation
HBAR_C_MEV_FM = 197.3269804

# Cross-section unit conversion: 1 mb = 0.1 fm²
MB_TO_FM2 = 0.1


def nuclear_radius(a):
    """
    Calculate nuclear radius using the empirical formula R = r₀ · A^(1/3).

    Args:
        a: Mass number of the nucleus.

    Returns:
        Nuclear radius in fm.
    """
    if a <= 0:
        raise ValueError("Mass number must be positive")
    return NUCLEAR_RADIUS_PARAM_FM * (a ** (1.0 / 3.0))


def fermi_momentum(density):
    """
    Calculate the Fermi momentum for nucleons at a given nuclear density.

    Uses the free Fermi gas approximation:
        p_F = ℏ · (3π² ρ / 2)^(1/3)

    where ρ is the nucleon number density in fm⁻³ and the result is in MeV/c.

    Args:
        density: Nucleon number density in fm⁻³.

    Returns:
        Fermi momentum in MeV/c.
    """
    if density <= 0:
        raise ValueError("Density must be positive")
    return HBAR_C_MEV_FM * (3.0 * math.pi ** 2 * density / 2.0) ** (1.0 / 3.0)


def nucleon_nucleon_cross_section(kinetic_energy_mev, is_pp=True):
    """
    Empirical total nucleon-nucleon cross section.

    Provides an approximate total cross section for pp or pn interactions
    in the energy range 100 MeV to 3 GeV, based on empirical parametrisation.

    Args:
        kinetic_energy_mev: Proton kinetic energy in MeV (100–3000 MeV).
        is_pp: If True, return the pp cross section; otherwise return pn.

    Returns:
        Total cross section in mb (millibarns).
    """
    T = kinetic_energy_mev
    if is_pp:
        # pp: ~50 mb at 100 MeV, falling toward ~40 mb at high energy
        sigma = 40.0 + 10.0 * math.exp(-T / 500.0)
    else:
        # pn: higher at low energies (~70 mb at 100 MeV), ~40 mb at high energy
        sigma = 40.0 + 30.0 * math.exp(-T / 400.0)
    return sigma


def mean_free_path(kinetic_energy_mev, density, z_fraction):
    """
    Calculate the mean free path of a proton in nuclear matter.

    λ = 1 / (σ_eff · ρ)

    where σ_eff is the cross section weighted by the proton/neutron content
    of the target nucleus.

    Args:
        kinetic_energy_mev: Proton kinetic energy in MeV.
        density: Nuclear number density in fm⁻³.
        z_fraction: Proton fraction of the nucleus (Z/A).

    Returns:
        Mean free path in fm.
    """
    if density <= 0:
        raise ValueError("Density must be positive")
    if z_fraction < 0 or z_fraction > 1:
        raise ValueError("z_fraction must be between 0 and 1")

    sigma_pp_fm2 = nucleon_nucleon_cross_section(kinetic_energy_mev, is_pp=True) * MB_TO_FM2
    sigma_pn_fm2 = nucleon_nucleon_cross_section(kinetic_energy_mev, is_pp=False) * MB_TO_FM2
    sigma_eff = z_fraction * sigma_pp_fm2 + (1.0 - z_fraction) * sigma_pn_fm2

    return 1.0 / (sigma_eff * density)


class BertiniCascade:
    """
    Bertini Intra-Nuclear Cascade model for proton interactions on a nucleus.

    Implements a simplified Bertini INC model to simulate the cascade of
    secondary particles produced when a proton strikes a nucleus.
    The model is valid for proton kinetic energies between 100 MeV and 3 GeV.

    By default the target is tungsten (W-184, Z=74, A=184).

    Reference:
        H. W. Bertini, Phys. Rev. 131, 1801 (1963).
    """

    def __init__(self, z=TUNGSTEN_Z, a=TUNGSTEN_A):
        """
        Initialise the Bertini cascade model.

        Args:
            z: Atomic number of the target nucleus (default: 74 for tungsten).
            a: Mass number of the target nucleus (default: 184 for W-184).

        Raises:
            ValueError: If a <= 0 or z is outside [0, a].
        """
        if a <= 0:
            raise ValueError("Mass number must be positive")
        if z < 0 or z > a:
            raise ValueError("Atomic number must be between 0 and A")

        self.z = z
        self.a = a
        self.n = a - z
        self.radius_fm = nuclear_radius(a)
        self.density = (3.0 * a) / (4.0 * math.pi * self.radius_fm ** 3)
        self.z_fraction = z / a
        self.fermi_momentum_mev_c = fermi_momentum(self.density)

    def nuclear_thickness(self):
        """
        Return the nuclear diameter (2R) as a simple estimate of path length.

        Returns:
            Nuclear diameter in fm.
        """
        return 2.0 * self.radius_fm

    def average_number_of_collisions(self, kinetic_energy_mev):
        """
        Estimate the average number of primary nucleon-nucleon collisions.

        Calculated as nuclear thickness / mean free path, giving the expected
        number of interactions along a central trajectory.

        Args:
            kinetic_energy_mev: Proton kinetic energy in MeV.

        Returns:
            Average number of primary NN collisions (dimensionless).
        """
        mfp = mean_free_path(kinetic_energy_mev, self.density, self.z_fraction)
        return self.nuclear_thickness() / mfp

    def run(self, kinetic_energy_mev):
        """
        Run the Bertini cascade for an incoming proton.

        Args:
            kinetic_energy_mev: Kinetic energy of the incoming proton in MeV.
                Must be in the range [100 MeV, 3000 MeV].

        Returns:
            dict with keys:
                - 'kinetic_energy_mev': Input proton kinetic energy (MeV).
                - 'nuclear_radius_fm': Target nuclear radius (fm).
                - 'nuclear_density_fm3': Average nuclear density (fm⁻³).
                - 'fermi_momentum_mev_c': Nucleon Fermi momentum (MeV/c).
                - 'mean_free_path_fm': Proton mean free path in the nucleus (fm).
                - 'average_collisions': Expected number of NN collisions.

        Raises:
            ValueError: If kinetic_energy_mev is outside [100, 3000] MeV.
        """
        if kinetic_energy_mev < BERTINI_MIN_ENERGY_MEV:
            raise ValueError(
                f"Proton kinetic energy {kinetic_energy_mev} MeV is below the "
                f"minimum valid energy {BERTINI_MIN_ENERGY_MEV} MeV for the "
                "Bertini model."
            )
        if kinetic_energy_mev > BERTINI_MAX_ENERGY_MEV:
            raise ValueError(
                f"Proton kinetic energy {kinetic_energy_mev} MeV exceeds the "
                f"maximum valid energy {BERTINI_MAX_ENERGY_MEV} MeV for the "
                "Bertini model."
            )

        mfp = mean_free_path(kinetic_energy_mev, self.density, self.z_fraction)
        avg_collisions = self.average_number_of_collisions(kinetic_energy_mev)

        return {
            'kinetic_energy_mev': kinetic_energy_mev,
            'nuclear_radius_fm': self.radius_fm,
            'nuclear_density_fm3': self.density,
            'fermi_momentum_mev_c': self.fermi_momentum_mev_c,
            'mean_free_path_fm': mfp,
            'average_collisions': avg_collisions,
        }
