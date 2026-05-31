"""
Nuclear interaction module for nuclear modelling.

This module provides a consistent set of two-body nuclear interaction tools,
organised into four layers:

1. **Potential forms** — Yukawa, square-well, Gaussian, and Woods-Saxon
   nuclear potentials evaluated at a given radial distance.

2. **Cross-section utilities** — Rutherford differential and integrated
   cross sections, geometric / reaction cross sections, and a low-energy
   effective-range cross-section helper.

3. **Target nucleus profile** — ``NuclearTarget`` class that encapsulates
   the geometrical and density properties of a target nucleus, provides a
   Woods-Saxon radial density profile, and exposes interaction-probability
   helpers for use with ``bertini_cascade`` results.

4. **Uncertainty propagation** — lightweight helpers that attach relative
   uncertainty and model-validity metadata to computed observables.

Unit conventions
----------------
* Distances in **fm** (femtometres).
* Energies in **MeV**.
* Cross sections in **mb** (millibarns); 1 mb = 0.1 fm².
* Differential cross sections in **mb/sr**.
* Potentials in **MeV**.

References
----------
- Krane, K. S. *Introductory Nuclear Physics* (Wiley, 1988).
- Bohr, A. & Mottelson, B. R. *Nuclear Structure* Vol. 1 (World Scientific, 1998).
- Blatt, J. M. & Weisskopf, V. F. *Theoretical Nuclear Physics* (Springer, 1952).
"""

import math

# ---------------------------------------------------------------------------
# Module-level constants (natural units, fm / MeV)
# ---------------------------------------------------------------------------

#: ℏc in MeV·fm
HBAR_C_MEV_FM = 197.3269804

#: Proton mass in MeV/c²
PROTON_MASS_MEV = 938.272046

#: Neutron mass in MeV/c²
NEUTRON_MASS_MEV = 939.565379

#: Nuclear radius parameter r₀ (fm); R = r₀ · A^(1/3)
NUCLEAR_RADIUS_PARAM_FM = 1.2

#: Fine-structure constant (dimensionless)
FINE_STRUCTURE_CONSTANT = 1.0 / 137.035999084

#: Conversion factor mb → fm²: 1 mb = 0.1 fm²
MB_TO_FM2 = 0.1

#: Conversion factor fm² → mb
FM2_TO_MB = 1.0 / MB_TO_FM2


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(value, name):
    """Raise ValueError if *value* is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value, name):
    """Raise ValueError if *value* is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_angle(theta, name):
    """Raise ValueError if *theta* is outside [0, π]."""
    if not (0.0 <= theta <= math.pi):
        raise ValueError(f"{name} must be in [0, π], got {theta}")


def _validate_mass_number(a, name="a"):
    """Raise ValueError if *a* is not a positive integer."""
    if not isinstance(a, int) or a < 1:
        raise ValueError(f"{name} must be a positive integer, got {a}")


# ===========================================================================
# 1. Potential forms
# ===========================================================================

def yukawa_potential(r_fm, V0_mev, mu_inv_fm):
    """Evaluate the Yukawa (meson-exchange) nuclear potential at radius *r_fm*.

    The Yukawa potential is:

    V(r) = -V₀ · exp(-μr) / (μr)

    It is the one-pion-exchange (OPE) prototype potential used in nuclear
    structure and reaction calculations.

    Parameters
    ----------
    r_fm : float
        Radial distance in fm.  Must be positive.
    V0_mev : float
        Potential depth (positive value, MeV).  Must be positive.
    mu_inv_fm : float
        Inverse range parameter μ = m_π c / ℏ in fm⁻¹.  Must be positive.
        For one-pion exchange, μ ≈ 1/1.41 fm⁻¹ ≈ 0.71 fm⁻¹.

    Returns
    -------
    float
        Potential value V(r) in MeV (negative, attractive).

    Raises
    ------
    ValueError
        If any argument is non-positive.

    Examples
    --------
    One-pion-exchange Yukawa at r = 2 fm, depth 40 MeV, μ = 0.71 fm⁻¹:

    >>> abs(yukawa_potential(2.0, 40.0, 0.71)) > 0
    True
    """
    _validate_positive(r_fm, "r_fm")
    _validate_positive(V0_mev, "V0_mev")
    _validate_positive(mu_inv_fm, "mu_inv_fm")
    return -V0_mev * math.exp(-mu_inv_fm * r_fm) / (mu_inv_fm * r_fm)


def square_well_potential(r_fm, V0_mev, R_fm):
    """Evaluate a finite square-well nuclear potential at radius *r_fm*.

    V(r) = -V₀   for r ≤ R
    V(r) =  0     for r > R

    Parameters
    ----------
    r_fm : float
        Radial distance in fm.  Must be non-negative.
    V0_mev : float
        Potential depth (positive, MeV).  Must be positive.
    R_fm : float
        Well radius in fm.  Must be positive.

    Returns
    -------
    float
        Potential value V(r) in MeV.

    Raises
    ------
    ValueError
        If *r_fm* is negative or *V0_mev* / *R_fm* are non-positive.

    Examples
    --------
    Inside the well:

    >>> square_well_potential(1.0, 50.0, 2.5)
    -50.0

    Outside the well:

    >>> square_well_potential(3.0, 50.0, 2.5)
    0.0
    """
    _validate_non_negative(r_fm, "r_fm")
    _validate_positive(V0_mev, "V0_mev")
    _validate_positive(R_fm, "R_fm")
    return -V0_mev if r_fm <= R_fm else 0.0


def gaussian_potential(r_fm, V0_mev, a_fm):
    """Evaluate a Gaussian nuclear potential at radius *r_fm*.

    V(r) = -V₀ · exp(-(r/a)²)

    The Gaussian potential is often used as a soft, finite-range approximation
    to the nuclear force.

    Parameters
    ----------
    r_fm : float
        Radial distance in fm.  Must be non-negative.
    V0_mev : float
        Potential depth (positive, MeV).  Must be positive.
    a_fm : float
        Range parameter (diffuseness) in fm.  Must be positive.

    Returns
    -------
    float
        Potential value V(r) in MeV (negative).

    Raises
    ------
    ValueError
        If *r_fm* is negative or *V0_mev* / *a_fm* are non-positive.

    Examples
    --------
    >>> gaussian_potential(0.0, 50.0, 1.5)
    -50.0
    """
    _validate_non_negative(r_fm, "r_fm")
    _validate_positive(V0_mev, "V0_mev")
    _validate_positive(a_fm, "a_fm")
    return -V0_mev * math.exp(-(r_fm / a_fm) ** 2)


def woods_saxon_potential(r_fm, V0_mev, R_fm, a_fm):
    """Evaluate a Woods-Saxon (WS) nuclear potential at radius *r_fm*.

    The standard WS form is widely used for nuclear mean-field potentials:

    V(r) = -V₀ / (1 + exp((r - R) / a))

    Parameters
    ----------
    r_fm : float
        Radial distance in fm.  Must be non-negative.
    V0_mev : float
        Potential depth (positive, MeV).  Must be positive.
    R_fm : float
        Half-density radius in fm.  Must be positive.
    a_fm : float
        Surface diffuseness in fm.  Must be positive.

    Returns
    -------
    float
        Potential value V(r) in MeV (negative).

    Raises
    ------
    ValueError
        If *r_fm* is negative or any other parameter is non-positive.

    Examples
    --------
    At the nuclear surface (r = R) the potential is half the depth:

    >>> ws = woods_saxon_potential(5.0, 50.0, 5.0, 0.65)
    >>> abs(ws + 25.0) < 0.01
    True
    """
    _validate_non_negative(r_fm, "r_fm")
    _validate_positive(V0_mev, "V0_mev")
    _validate_positive(R_fm, "R_fm")
    _validate_positive(a_fm, "a_fm")
    return -V0_mev / (1.0 + math.exp((r_fm - R_fm) / a_fm))


# ===========================================================================
# 2. Cross-section utilities
# ===========================================================================

def rutherford_differential_cross_section(T_mev, z1, z2, theta_rad):
    """Rutherford differential cross section dσ/dΩ in the lab frame.

    The classical Coulomb-scattering formula (centre-of-mass frame, assuming
    the target is much heavier or applying the appropriate CoM transformation):

    dσ/dΩ = (z₁ z₂ e² / 4T)² · 1 / sin⁴(θ/2)

    where T is the **lab-frame** kinetic energy and the formula is exact in
    the CoM frame for identical total energy.  For a heavy target (m₂ ≫ m₁)
    the lab and CoM cross sections coincide.

    Parameters
    ----------
    T_mev : float
        Projectile kinetic energy in the lab frame (MeV).  Must be positive.
    z1 : int or float
        Charge number of the projectile.  Must be positive.
    z2 : int or float
        Charge number of the target.  Must be positive.
    theta_rad : float
        Scattering angle in radians (0, π].  Must be in (0, π].

    Returns
    -------
    float
        Differential cross section dσ/dΩ in mb/sr.

    Raises
    ------
    ValueError
        If *T_mev* ≤ 0, charges are non-positive, or *theta_rad* is outside
        (0, π].

    Examples
    --------
    Alpha on gold at 10 MeV, θ = π/2:

    >>> dcs = rutherford_differential_cross_section(10.0, 2, 79, math.pi / 2)
    >>> dcs > 0
    True
    """
    _validate_positive(T_mev, "T_mev")
    _validate_positive(z1, "z1")
    _validate_positive(z2, "z2")
    if not (0.0 < theta_rad <= math.pi):
        raise ValueError(f"theta_rad must be in (0, π], got {theta_rad}")

    # Coulomb parameter: k = z1*z2*e²/(4T) in natural units
    # e² in MeV·fm = ℏc * α ≈ 197.327 * (1/137.036) ≈ 1.43996 MeV·fm
    e2_mev_fm = HBAR_C_MEV_FM * FINE_STRUCTURE_CONSTANT
    k_fm = z1 * z2 * e2_mev_fm / (4.0 * T_mev)

    sin_half = math.sin(theta_rad / 2.0)
    # dσ/dΩ = k² / sin⁴(θ/2)  in fm²/sr → convert to mb/sr
    dsigma_fm2_per_sr = (k_fm ** 2) / (sin_half ** 4)
    return dsigma_fm2_per_sr * FM2_TO_MB


def rutherford_total_cross_section(T_mev, z1, z2, theta_min_rad):
    """Integrated Rutherford cross section above a minimum angle *theta_min*.

    The Rutherford cross section diverges as θ → 0, so it is always
    integrated above a finite lower cutoff angle θ_min:

    σ(θ > θ_min) = π (z₁ z₂ e² / 2T)² · cos²(θ_min/2) / sin²(θ_min/2)

    Parameters
    ----------
    T_mev : float
        Projectile lab-frame kinetic energy in MeV.  Must be positive.
    z1 : int or float
        Projectile charge number.  Must be positive.
    z2 : int or float
        Target charge number.  Must be positive.
    theta_min_rad : float
        Minimum scattering angle (rad) above which to integrate.
        Must be in (0, π].

    Returns
    -------
    float
        Integrated cross section in mb.

    Raises
    ------
    ValueError
        If *T_mev* ≤ 0, charges are non-positive, or *theta_min_rad* is
        outside (0, π].

    Examples
    --------
    >>> sigma = rutherford_total_cross_section(10.0, 2, 79, 0.1)
    >>> sigma > 0
    True
    """
    _validate_positive(T_mev, "T_mev")
    _validate_positive(z1, "z1")
    _validate_positive(z2, "z2")
    if not (0.0 < theta_min_rad <= math.pi):
        raise ValueError(f"theta_min_rad must be in (0, π], got {theta_min_rad}")

    e2_mev_fm = HBAR_C_MEV_FM * FINE_STRUCTURE_CONSTANT
    k_fm = z1 * z2 * e2_mev_fm / (2.0 * T_mev)

    sin_half = math.sin(theta_min_rad / 2.0)
    cos_half = math.cos(theta_min_rad / 2.0)
    sigma_fm2 = math.pi * k_fm ** 2 * (cos_half / sin_half) ** 2
    return sigma_fm2 * FM2_TO_MB


def geometric_cross_section(a_fm):
    """Geometric cross section of a sphere with radius *a_fm*.

    σ_geo = π R²

    Parameters
    ----------
    a_fm : float
        Nuclear radius in fm.  Must be positive.

    Returns
    -------
    float
        Geometric cross section in mb.

    Raises
    ------
    ValueError
        If *a_fm* is non-positive.

    Examples
    --------
    >>> geometric_cross_section(6.86) > 0  # W-184 radius ≈ 6.86 fm
    True
    """
    _validate_positive(a_fm, "a_fm")
    return math.pi * a_fm ** 2 * FM2_TO_MB


def nuclear_reaction_cross_section(a1, a2):
    """Geometric nuclear reaction cross section for two nuclei.

    Uses the empirical radius formula R = r₀ · A^(1/3):

    σ_R = π (R₁ + R₂)²

    This gives an order-of-magnitude estimate of the reaction cross section
    for nucleus-nucleus collisions.

    Parameters
    ----------
    a1 : int
        Mass number of nucleus 1.  Must be a positive integer.
    a2 : int
        Mass number of nucleus 2.  Must be a positive integer.

    Returns
    -------
    float
        Nuclear reaction cross section in mb.

    Raises
    ------
    ValueError
        If either mass number is not a positive integer.

    Examples
    --------
    Proton (A=1) on carbon-12 (A=12):

    >>> nuclear_reaction_cross_section(1, 12) > 0
    True
    """
    _validate_mass_number(a1, "a1")
    _validate_mass_number(a2, "a2")
    R1_fm = NUCLEAR_RADIUS_PARAM_FM * a1 ** (1.0 / 3.0)
    R2_fm = NUCLEAR_RADIUS_PARAM_FM * a2 ** (1.0 / 3.0)
    return math.pi * (R1_fm + R2_fm) ** 2 * FM2_TO_MB


def effective_range_cross_section(k_inv_fm, a_fm, r0_fm=0.0):
    """Low-energy s-wave cross section from the effective range expansion.

    At low energies, the s-wave elastic cross section is well described by:

    σ = 4π a² / ((1 - a r₀ k²/2)² + a² k²)

    where a is the scattering length, r₀ is the effective range, and k is
    the centre-of-mass wave number.  For k → 0 this reduces to σ = 4π a².

    Parameters
    ----------
    k_inv_fm : float
        Centre-of-mass wave number k in fm⁻¹.  Must be non-negative.
    a_fm : float
        Scattering length a in fm.  Must be non-zero (raises if zero).
    r0_fm : float, optional
        Effective range r₀ in fm.  Default is 0 (no effective range correction).
        Must be non-negative.

    Returns
    -------
    float
        Elastic cross section in mb.

    Raises
    ------
    ValueError
        If k is negative, *a_fm* is zero, or *r0_fm* is negative.

    Examples
    --------
    Zero-energy limit gives 4π a²:

    >>> import math
    >>> sigma0 = effective_range_cross_section(0.0, 1.0)
    >>> abs(sigma0 - 4 * math.pi * FM2_TO_MB) < 1e-6
    True
    """
    _validate_non_negative(k_inv_fm, "k_inv_fm")
    if a_fm == 0.0:
        raise ValueError("Scattering length a_fm must not be zero")
    _validate_non_negative(r0_fm, "r0_fm")

    k2 = k_inv_fm ** 2
    numerator = 4.0 * math.pi * a_fm ** 2
    denom_real = 1.0 - a_fm * r0_fm * k2 / 2.0
    denom = denom_real ** 2 + (a_fm * k_inv_fm) ** 2
    return (numerator / denom) * FM2_TO_MB


def born_approximation_cross_section(potential_fn, k_in_inv_fm, k_out_inv_fm,
                                     reduced_mass_mev_c2,
                                     r_max_fm=20.0, n_points=1000):
    """Total elastic cross section in the first Born approximation.

    Numerically integrates the form factor:

    f(q) = -(m / 2π ℏ²) ∫ V(r) exp(i q·r) d³r

    using the partial-wave Born formula for a central potential:

    dσ/dΩ = |f(θ)|²,   f(θ) = -(2m/ℏ²) ∫₀^∞ V(r) j₀(qr) r² dr

    where q = |k_in - k_out| = 2k sin(θ/2) for elastic scattering
    (|k_in| = |k_out| = k).

    The total cross section is obtained by integrating dσ/dΩ over all angles.
    This uses a simple trapezoidal integration.

    Parameters
    ----------
    potential_fn : callable
        Function V(r_fm) → MeV.  Must accept a single float argument (radius
        in fm) and return the potential in MeV (negative for attractive).
    k_in_inv_fm : float
        Incident wave number in fm⁻¹.  Must be positive.
    k_out_inv_fm : float
        Outgoing wave number in fm⁻¹.  For elastic scattering this equals
        *k_in_inv_fm*.
    reduced_mass_mev_c2 : float
        Reduced mass of the projectile–target system in MeV/c².  Must be
        positive.
    r_max_fm : float, optional
        Upper radial integration limit in fm (default 20 fm).
    n_points : int, optional
        Number of radial points for numerical integration (default 1000).

    Returns
    -------
    float
        Total elastic cross section in mb.

    Raises
    ------
    ValueError
        If wave numbers or reduced mass are non-positive.

    Notes
    -----
    This is a simplified Born calculation valid when the potential is weak
    compared to the kinetic energy.  For strong potentials or near-threshold
    energies use a full phase-shift calculation.
    """
    _validate_positive(k_in_inv_fm, "k_in_inv_fm")
    _validate_positive(k_out_inv_fm, "k_out_inv_fm")
    _validate_positive(reduced_mass_mev_c2, "reduced_mass_mev_c2")
    if r_max_fm <= 0:
        raise ValueError("r_max_fm must be positive")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Pre-factor: (2m / ℏ²)² · (4π)
    # m in MeV/c², ℏc in MeV·fm → (m / (ℏc)²) in fm⁻² MeV⁻¹
    m_over_hbarc2 = reduced_mass_mev_c2 / (HBAR_C_MEV_FM ** 2)
    prefactor = (m_over_hbarc2 / (2.0 * math.pi)) ** 2 * (4.0 * math.pi)

    k = (k_in_inv_fm + k_out_inv_fm) / 2.0  # elastic: k_in = k_out
    dr = r_max_fm / (n_points - 1)

    # Integrate ∫ dΩ over scattering angle, exploiting azimuthal symmetry
    # σ = 2π ∫₀^π |f(θ)|² sin θ dθ
    # f(θ) = -(2m/ℏ²) ∫₀^∞ V(r) j₀(qr) r² dr  with q = 2k sin(θ/2)
    # We perform the angular integral numerically using n_points/10 angle steps
    n_theta = max(50, n_points // 10)
    dtheta = math.pi / (n_theta - 1)

    sigma_fm2 = 0.0
    for i_theta in range(n_theta):
        theta = i_theta * dtheta
        sin_half = math.sin(theta / 2.0)
        q = 2.0 * k * sin_half

        # Radial form-factor integral: I(q) = ∫ V(r) j₀(qr) r² dr
        # j₀(x) = sin(x)/x for x > 0, 1 for x = 0
        I_q = 0.0
        for j in range(n_points):
            r = (j + 0.5) * dr  # midpoint rule
            V = potential_fn(r)
            qr = q * r
            j0 = math.sin(qr) / qr if qr > 1e-14 else 1.0
            I_q += V * j0 * r ** 2 * dr

        f_sq = prefactor * I_q ** 2
        # Trapezoid weight for angle integration: sin θ dθ
        weight = math.sin(theta) * dtheta
        if i_theta == 0 or i_theta == n_theta - 1:
            weight *= 0.5
        sigma_fm2 += 2.0 * math.pi * f_sq * weight

    return sigma_fm2 * FM2_TO_MB


# ===========================================================================
# 3. Target nucleus profile
# ===========================================================================

class NuclearTarget:
    """Target nucleus profile for use in interaction calculations.

    Encapsulates the geometry, density, and interaction properties of a
    target nucleus, providing a reusable description that can feed
    ``bertini_cascade``, ``kinematics``, and ``basic_nuclear_structure``
    calculations.

    Parameters
    ----------
    z : int
        Atomic number (number of protons).
    a : int
        Mass number (total nucleons).

    Attributes
    ----------
    z : int
    a : int
    n : int
        Number of neutrons (a - z).
    z_fraction : float
        Proton fraction Z/A.
    n_fraction : float
        Neutron fraction N/A.
    radius_fm : float
        RMS nuclear radius (fm), using R = r₀ A^(1/3).
    volume_fm3 : float
        Nuclear volume (fm³), 4π R³/3.
    density_fm3 : float
        Average nucleon number density (fm⁻³).

    Raises
    ------
    ValueError
        If *a* < 1, *z* < 0, or *z* > *a*.

    Examples
    --------
    Tungsten W-184:

    >>> target = NuclearTarget(z=74, a=184)
    >>> round(target.radius_fm, 2)
    6.86
    """

    def __init__(self, z, a):
        if not isinstance(a, int) or a < 1:
            raise ValueError("a must be a positive integer")
        if not isinstance(z, int) or z < 0 or z > a:
            raise ValueError("z must be an integer in [0, a]")

        self.z = z
        self.a = a
        self.n = a - z
        self.z_fraction = z / a
        self.n_fraction = (a - z) / a
        self.radius_fm = NUCLEAR_RADIUS_PARAM_FM * a ** (1.0 / 3.0)
        self.volume_fm3 = (4.0 / 3.0) * math.pi * self.radius_fm ** 3
        self.density_fm3 = a / self.volume_fm3

    # ------------------------------------------------------------------
    # Density profiles
    # ------------------------------------------------------------------

    def woods_saxon_density(self, r_fm, a_diffuse_fm=0.54):
        """Nucleon number density using the Woods-Saxon profile.

        ρ(r) = ρ₀ / (1 + exp((r - R) / a))

        where ρ₀ is the central density, R is the half-density radius, and
        *a_diffuse_fm* is the surface diffuseness (~0.54 fm for most nuclei).

        Parameters
        ----------
        r_fm : float
            Radial distance in fm.  Must be non-negative.
        a_diffuse_fm : float, optional
            Surface diffuseness in fm (default 0.54 fm).

        Returns
        -------
        float
            Nucleon density at *r_fm* in fm⁻³.

        Raises
        ------
        ValueError
            If *r_fm* < 0 or *a_diffuse_fm* ≤ 0.
        """
        _validate_non_negative(r_fm, "r_fm")
        _validate_positive(a_diffuse_fm, "a_diffuse_fm")
        return self.density_fm3 / (1.0 + math.exp((r_fm - self.radius_fm) / a_diffuse_fm))

    # ------------------------------------------------------------------
    # Geometric cross sections
    # ------------------------------------------------------------------

    def geometric_cross_section_mb(self):
        """Return the geometric cross section π R² in mb.

        Returns
        -------
        float
            Geometric cross section in mb.
        """
        return geometric_cross_section(self.radius_fm)

    def reaction_cross_section_mb(self, projectile_a):
        """Geometric reaction cross section for a given projectile mass number.

        σ_R = π (R_projectile + R_target)²

        Parameters
        ----------
        projectile_a : int
            Mass number of the projectile nucleus.  Must be a positive integer.

        Returns
        -------
        float
            Reaction cross section in mb.

        Raises
        ------
        ValueError
            If *projectile_a* is not a positive integer.
        """
        return nuclear_reaction_cross_section(projectile_a, self.a)

    # ------------------------------------------------------------------
    # Interaction probability helpers
    # ------------------------------------------------------------------

    def interaction_probability(self, areal_density_fm2):
        """Probability that a projectile interacts within an areal density.

        P = 1 - exp(-σ_R · ρ_areal)

        where σ_R is the geometric reaction cross section for a proton
        (A=1) projectile and ρ_areal is the areal number density
        (nucleons per fm²) of the target.

        Parameters
        ----------
        areal_density_fm2 : float
            Areal density of the target in nucleons per fm².  Must be
            non-negative.

        Returns
        -------
        float
            Interaction probability in [0, 1].

        Raises
        ------
        ValueError
            If *areal_density_fm2* is negative.
        """
        _validate_non_negative(areal_density_fm2, "areal_density_fm2")
        sigma_fm2 = self.reaction_cross_section_mb(1) * MB_TO_FM2
        return 1.0 - math.exp(-sigma_fm2 * areal_density_fm2)

    def mean_free_path_fm(self, sigma_total_mb):
        """Mean free path of a projectile in the target material.

        λ = 1 / (σ_total · ρ)

        Parameters
        ----------
        sigma_total_mb : float
            Total projectile-nucleon cross section in mb.  Must be positive.

        Returns
        -------
        float
            Mean free path in fm.

        Raises
        ------
        ValueError
            If *sigma_total_mb* is non-positive.
        """
        _validate_positive(sigma_total_mb, "sigma_total_mb")
        sigma_fm2 = sigma_total_mb * MB_TO_FM2
        return 1.0 / (sigma_fm2 * self.density_fm3)

    # ------------------------------------------------------------------
    # Connection to nuclear structure
    # ------------------------------------------------------------------

    def coulomb_barrier_mev(self, z_projectile, a_projectile):
        """Estimate the Coulomb barrier for a projectile on this target.

        V_C = z₁ z₂ e² / (R₁ + R₂)

        Parameters
        ----------
        z_projectile : int
            Charge number of the projectile.  Must be non-negative.
        a_projectile : int
            Mass number of the projectile.  Must be a positive integer.

        Returns
        -------
        float
            Coulomb barrier height in MeV.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if not isinstance(z_projectile, int) or z_projectile < 0:
            raise ValueError("z_projectile must be a non-negative integer")
        _validate_mass_number(a_projectile, "a_projectile")
        e2_mev_fm = HBAR_C_MEV_FM * FINE_STRUCTURE_CONSTANT
        R_proj_fm = NUCLEAR_RADIUS_PARAM_FM * a_projectile ** (1.0 / 3.0)
        R_total_fm = R_proj_fm + self.radius_fm
        return z_projectile * self.z * e2_mev_fm / R_total_fm

    def fermi_momentum_mev_c(self):
        """Fermi momentum of nucleons inside this target nucleus.

        Uses the free Fermi gas approximation:
        p_F = ℏc · (3π² ρ / 2)^(1/3)

        Returns
        -------
        float
            Fermi momentum in MeV/c.
        """
        return HBAR_C_MEV_FM * (3.0 * math.pi ** 2 * self.density_fm3 / 2.0) ** (1.0 / 3.0)

    def __repr__(self):
        return (
            f"NuclearTarget(Z={self.z}, A={self.a}, "
            f"R={self.radius_fm:.2f} fm, ρ={self.density_fm3:.3f} fm⁻³)"
        )


# ===========================================================================
# 4. Uncertainty propagation and model metadata
# ===========================================================================

def propagate_relative_uncertainty(value, *rel_uncertainties):
    """Propagate independent relative uncertainties in quadrature.

    For a result y = f(x₁, x₂, …) where each xᵢ has a relative uncertainty
    δᵢ = Δxᵢ/xᵢ, the combined relative uncertainty is:

    Δy/y = sqrt(Σᵢ δᵢ²)

    Parameters
    ----------
    value : float
        Central value of the observable.
    *rel_uncertainties : float
        One or more relative uncertainties (dimensionless fractions, e.g.
        0.05 for 5 %).  Each must be non-negative.

    Returns
    -------
    float
        Absolute uncertainty Δy = y × sqrt(Σᵢ δᵢ²).

    Raises
    ------
    ValueError
        If *value* is negative or any relative uncertainty is negative.

    Examples
    --------
    Cross section of 50 mb with 10% and 5% independent contributions:

    >>> delta = propagate_relative_uncertainty(50.0, 0.10, 0.05)
    >>> abs(delta - 50.0 * (0.10**2 + 0.05**2)**0.5) < 1e-10
    True
    """
    _validate_non_negative(value, "value")
    for i, du in enumerate(rel_uncertainties):
        if du < 0:
            raise ValueError(f"rel_uncertainty[{i}] must be non-negative, got {du}")
    combined_rel = math.sqrt(sum(du ** 2 for du in rel_uncertainties))
    return value * combined_rel


def model_result(value, abs_uncertainty, units, model_name,
                 valid_range=None, notes=None):
    """Package an observable with uncertainty and model validity metadata.

    Returns a dictionary that bundles the computed value with its uncertainty,
    unit label, the name of the model used, the energy/parameter range over
    which the model is valid, and optional free-text notes.

    Parameters
    ----------
    value : float
        Central value of the observable.
    abs_uncertainty : float
        Absolute uncertainty (same units as *value*).  Must be non-negative.
    units : str
        Unit string, e.g. ``"mb"``, ``"MeV"``, ``"fm"``.
    model_name : str
        Short identifier for the model or formula used, e.g.
        ``"Rutherford"``, ``"Born approximation"``.
    valid_range : dict, optional
        Dictionary describing the validity domain of the model, e.g.
        ``{"T_mev": (100, 3000), "description": "Bertini energy range"}``.
    notes : str, optional
        Free-text notes or caveats about the result.

    Returns
    -------
    dict
        Keys: ``value``, ``uncertainty``, ``units``, ``model``,
        ``valid_range`` (or ``None``), ``notes`` (or ``None``).

    Raises
    ------
    ValueError
        If *abs_uncertainty* is negative.

    Examples
    --------
    >>> result = model_result(42.5, 3.0, "mb", "Rutherford",
    ...     valid_range={"T_mev": (1, 100)},
    ...     notes="Valid below nuclear Coulomb barrier")
    >>> result["value"]
    42.5
    """
    _validate_non_negative(abs_uncertainty, "abs_uncertainty")
    return {
        "value": value,
        "uncertainty": abs_uncertainty,
        "units": units,
        "model": model_name,
        "valid_range": valid_range,
        "notes": notes,
    }


def energy_loss_bethe(z_projectile, T_mev, m_projectile_mev_c2,
                      mean_excitation_ev, electron_density_inv_fm3):
    """Relativistic Bethe–Bloch stopping power dE/dx (MeV/fm).

    The Bethe–Bloch formula gives the mean energy loss per unit path length
    of a charged particle traversing a medium:

    -dE/dx = (4π z² e⁴ n_e) / (m_e c² β²) · [ln(2 m_e c² β² γ² / I) - β²]

    where n_e is the electron number density, I is the mean excitation energy,
    β = v/c, and γ is the Lorentz factor.

    Parameters
    ----------
    z_projectile : int or float
        Charge number of the projectile.  Must be positive.
    T_mev : float
        Kinetic energy of the projectile in MeV.  Must be positive.
    m_projectile_mev_c2 : float
        Rest mass of the projectile in MeV/c².  Must be positive.
    mean_excitation_ev : float
        Mean excitation energy I of the target medium in eV.  Must be
        positive.  Typical values: ~13.5 eV for hydrogen, ~727 eV for lead.
    electron_density_inv_fm3 : float
        Electron number density in fm⁻³.  Must be positive.

    Returns
    -------
    float
        Stopping power -dE/dx in MeV/fm (positive value).

    Raises
    ------
    ValueError
        If any argument is non-positive.

    Notes
    -----
    This is the relativistic Bethe formula without shell, density, or
    Barkas corrections.  It is accurate to ~1–2 % for heavy particles
    (protons, alphas) at energies well above the Bragg-peak region.

    References
    ----------
    - Bethe, H. A. *Ann. Phys.* **5**, 325 (1930).
    - PDG Review of Particle Physics, Sec. 34 (Passage of particles through
      matter).
    """
    _validate_positive(z_projectile, "z_projectile")
    _validate_positive(T_mev, "T_mev")
    _validate_positive(m_projectile_mev_c2, "m_projectile_mev_c2")
    _validate_positive(mean_excitation_ev, "mean_excitation_ev")
    _validate_positive(electron_density_inv_fm3, "electron_density_inv_fm3")

    # Electron mass in MeV/c²
    m_e_mev = 0.510998950  # MeV/c²

    # Lorentz factors
    gamma = 1.0 + T_mev / m_projectile_mev_c2
    beta2 = 1.0 - 1.0 / gamma ** 2
    if beta2 <= 0:
        raise ValueError("Computed beta² ≤ 0; check T_mev and m_projectile_mev_c2")

    # e² in MeV·fm
    e2_mev_fm = HBAR_C_MEV_FM * FINE_STRUCTURE_CONSTANT

    # Prefactor: 4π z² e⁴ n_e / (m_e c² β²)
    # e⁴ = e² · e²  (in MeV·fm units, e² ≈ 1.44 MeV·fm)
    prefactor = (
        4.0 * math.pi
        * z_projectile ** 2
        * e2_mev_fm ** 2
        * electron_density_inv_fm3
        / (m_e_mev * beta2)
    )

    # Mean excitation energy in MeV
    I_mev = mean_excitation_ev * 1e-6

    # Bethe logarithm
    bethe_log = math.log(2.0 * m_e_mev * beta2 * gamma ** 2 / I_mev) - beta2

    if bethe_log <= 0:
        # Below minimum-ionizing region; return zero to avoid unphysical negative dE/dx
        return 0.0

    return prefactor * bethe_log
