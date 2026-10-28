"""
Kinematics module for nuclear modelling.

This module provides functions for both classical (non-relativistic) and
relativistic two-body collision kinematics, with a focus on nuclear
scattering problems such as a neutron striking a nucleus.

**Classical kinematics** assumes kinetic energies much smaller than
rest-mass energies (T << mc²) and works with masses directly.

**Relativistic kinematics** uses four-momentum conservation and is valid
at all energies.  Masses and energies are expressed in MeV/c² and MeV
respectively (natural units with c = 1).

Coordinate conventions
----------------------
* The projectile travels along the positive z-axis in the laboratory frame.
* The target is at rest in the laboratory frame.
* Scattering angles (``theta``) are measured from the beam axis (forward
  direction) and lie in [0, π].
* CoM angles (``theta_cm``) follow the same convention; θ_cm = 0 means
  forward scattering, θ_cm = π means backward scattering in the CoM frame.

References
----------
- Krane, K. S. *Introductory Nuclear Physics* (Wiley, 1988), Ch. 4.
- Burcham, W. E. & Jobes, M. *Nuclear and Particle Physics* (Longman, 1995).
- Griffiths, D. J. *Introduction to Elementary Particles* (Wiley, 2008), App. B.
"""

import math


# ---------------------------------------------------------------------------
# Classical (non-relativistic) two-body elastic scattering
# ---------------------------------------------------------------------------

def classical_cm_energy(T, m1, m2):
    """Calculate the kinetic energy available in the centre-of-mass (CoM) frame.

    T_cm = T × m2 / (m1 + m2)

    This is the energy available for scattering in the non-relativistic
    centre-of-mass frame.

    Parameters
    ----------
    T : float
        Kinetic energy of the projectile in the laboratory frame (any
        consistent energy unit, e.g. MeV or eV).
    m1 : float
        Mass of the projectile (any consistent mass unit).
    m2 : float
        Mass of the target (same unit as *m1*).

    Returns
    -------
    float
        CoM frame kinetic energy in the same unit as *T*.

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    1 MeV neutron (m=1 amu) on hydrogen-1 (M=1 amu):

    >>> classical_cm_energy(1.0, 1.0, 1.0)
    0.5
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    return T * m2 / (m1 + m2)


def classical_scattered_energy_cm(T, m1, m2, theta_cm):
    """Kinetic energy of the scattered projectile as a function of CoM angle.

    After elastic scattering at centre-of-mass angle *theta_cm*, the
    scattered projectile kinetic energy in the **laboratory** frame is:

    T1' = T × (m1² + m2² + 2 m1 m2 cos θ_cm) / (m1 + m2)²

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in the lab frame.
    m1 : float
        Projectile mass (any consistent unit).
    m2 : float
        Target mass (same unit as *m1*).
    theta_cm : float
        CoM scattering angle of the projectile in radians [0, π].

    Returns
    -------
    float
        Scattered projectile kinetic energy in the lab frame (same unit as
        *T*).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    1 MeV neutron on carbon-12 (A=12), backscatter (θ_cm = π):

    >>> classical_scattered_energy_cm(1.0, 1.0, 12.0, math.pi)
    0.7160493827160494
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    _validate_angle(theta_cm, "theta_cm")
    numerator = m1**2 + m2**2 + 2.0 * m1 * m2 * math.cos(theta_cm)
    return T * numerator / (m1 + m2) ** 2


def classical_recoil_energy_cm(T, m1, m2, theta_cm):
    """Kinetic energy of the recoiling target nucleus as a function of CoM angle.

    T2' = T × 2 m1 m2 (1 − cos θ_cm) / (m1 + m2)²

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in the lab frame.
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Recoil kinetic energy in the lab frame (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    Maximum recoil energy of hydrogen-1 struck by a 1 MeV neutron (θ_cm = π):

    >>> classical_recoil_energy_cm(1.0, 1.0, 1.0, math.pi)
    1.0
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    _validate_angle(theta_cm, "theta_cm")
    return T * 2.0 * m1 * m2 * (1.0 - math.cos(theta_cm)) / (m1 + m2) ** 2


def classical_max_energy_transfer(T, m1, m2):
    """Maximum kinetic energy that can be transferred from projectile to target.

    The maximum energy transfer occurs at θ_cm = π (head-on collision):

    ΔT_max = T × 4 m1 m2 / (m1 + m2)²

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy.
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).

    Returns
    -------
    float
        Maximum transferable kinetic energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    Neutron on hydrogen-1 — full energy transfer is possible:

    >>> classical_max_energy_transfer(1.0, 1.0, 1.0)
    1.0

    Neutron on carbon-12:

    >>> classical_max_energy_transfer(1.0, 1.0, 12.0)
    0.28402366863905326
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    return T * 4.0 * m1 * m2 / (m1 + m2) ** 2


def classical_scattered_energy_lab(T, m1, m2, theta_lab):
    """Kinetic energy of the scattered projectile at a given **lab** angle.

    For elastic scattering with m1 ≤ m2 there is a unique solution:

    T1' = T/(m1+m2)² × [m1 cos θ_lab + sqrt(m2²−m1² sin² θ_lab)]²

    When m1 > m2 the projectile cannot scatter beyond a maximum lab angle
    θ_max = arcsin(m2/m1); two solutions exist below this angle (only the
    kinematically reachable one with positive square-root is returned).

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in the lab frame.
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).
    theta_lab : float
        Lab-frame scattering angle of the projectile in radians [0, π/2]
        for m1 ≤ m2, or [0, arcsin(m2/m1)] for m1 > m2.

    Returns
    -------
    float
        Scattered projectile kinetic energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_lab* is
        outside the kinematically allowed range.

    Examples
    --------
    1 MeV neutron forward-scattered (θ_lab = 0) on carbon-12 — no energy loss:

    >>> abs(classical_scattered_energy_lab(1.0, 1.0, 12.0, 0.0) - 1.0) < 1e-12
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    if theta_lab < 0.0:
        raise ValueError("theta_lab must be non-negative")

    sin_theta = math.sin(theta_lab)
    cos_theta = math.cos(theta_lab)
    discriminant = m2**2 - m1**2 * sin_theta**2
    if discriminant < 0.0:
        max_angle = math.asin(m2 / m1)
        raise ValueError(
            f"theta_lab={theta_lab:.4f} rad exceeds the maximum allowed lab angle "
            f"{max_angle:.4f} rad for m1={m1}, m2={m2}"
        )
    factor = m1 * cos_theta + math.sqrt(discriminant)
    return T * factor**2 / (m1 + m2) ** 2


def classical_recoil_energy_lab(T, m1, m2, phi_lab):
    """Kinetic energy of the recoiling target at a given **lab** recoil angle.

    The recoil angle *phi_lab* is measured from the beam axis and lies in
    [0, π/2] (recoil is always in the forward hemisphere):

    T2' = T × 4 m1 m2 cos²φ_lab / (m1 + m2)²

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in the lab frame.
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).
    phi_lab : float
        Lab-frame recoil angle in radians [0, π/2].

    Returns
    -------
    float
        Recoil kinetic energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *phi_lab* is
        outside [0, π/2].

    Examples
    --------
    Neutron head-on collision with proton — full energy transfer at φ = 0:

    >>> classical_recoil_energy_lab(1.0, 1.0, 1.0, 0.0)
    1.0
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    if not 0.0 <= phi_lab <= math.pi / 2.0:
        raise ValueError("phi_lab must be in [0, π/2]")
    return T * 4.0 * m1 * m2 * math.cos(phi_lab) ** 2 / (m1 + m2) ** 2


def classical_cm_to_lab_angle(theta_cm, m1, m2):
    """Convert a projectile CoM scattering angle to the equivalent lab angle.

    tan θ_lab = sin θ_cm / (m1/m2 + cos θ_cm)

    For m1 < m2: θ_lab is uniquely defined for all θ_cm in [0, π].
    For m1 = m2: θ_lab = θ_cm / 2.
    For m1 > m2: a maximum lab angle exists; the function returns the
    physically meaningful (forward-hemisphere) solution.

    Parameters
    ----------
    theta_cm : float
        CoM scattering angle in radians [0, π].
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).

    Returns
    -------
    float
        Lab scattering angle in radians [0, π/2] (or [0, π] for m1 < m2).

    Raises
    ------
    ValueError
        If *theta_cm* is outside [0, π] or either mass is non-positive.

    Examples
    --------
    Equal masses, θ_cm = π/2 → θ_lab = π/4:

    >>> import math
    >>> abs(classical_cm_to_lab_angle(math.pi/2, 1.0, 1.0) - math.pi/4) < 1e-12
    True
    """
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    _validate_angle(theta_cm, "theta_cm")
    ratio = m1 / m2
    return math.atan2(math.sin(theta_cm), ratio + math.cos(theta_cm))


def classical_lab_to_cm_angle(theta_lab, m1, m2):
    """Convert a projectile lab scattering angle to the equivalent CoM angle.

    This is the inverse of :func:`classical_cm_to_lab_angle`.  The
    relationship is solved analytically by rearranging:

    tan θ_lab = sin θ_cm / (r + cos θ_cm),  r = m1/m2

    into the quadratic in x = cos θ_cm:

    (1 + t²) x² + 2r t² x + (r²t² − 1) = 0,  t = tan θ_lab

    Both roots are evaluated and the one that round-trips back to *theta_lab*
    via :func:`classical_cm_to_lab_angle` is returned.

    Parameters
    ----------
    theta_lab : float
        Lab scattering angle of the projectile in radians.
    m1 : float
        Projectile mass.
    m2 : float
        Target mass (same unit as *m1*).

    Returns
    -------
    float
        CoM scattering angle in radians [0, π].

    Raises
    ------
    ValueError
        If *theta_lab* is outside the kinematically allowed range, or
        either mass is non-positive.

    Examples
    --------
    Equal masses, θ_lab = π/4 → θ_cm = π/2:

    >>> import math
    >>> abs(classical_lab_to_cm_angle(math.pi/4, 1.0, 1.0) - math.pi/2) < 1e-10
    True
    """
    _validate_positive(m1, "m1")
    _validate_positive(m2, "m2")
    if theta_lab < 0.0:
        raise ValueError("theta_lab must be non-negative")

    ratio = m1 / m2  # r = m1/m2
    # From: tan(theta_lab) = sin(theta_cm)/(r + cos(theta_cm))
    # Let s = sin(theta_cm), c = cos(theta_cm)
    # tan(theta_lab) * (r + c) = s
    # Squaring and using s^2 + c^2 = 1:
    # t^2*(r+c)^2 = 1-c^2  where t = tan(theta_lab)
    # t^2*r^2 + 2t^2*r*c + t^2*c^2 + c^2 = 1
    # (1+t^2)*c^2 + 2t^2*r*c + (t^2*r^2 - 1) = 0
    # c^2/cos^2 + 2t^2*r/cos^2 * cos + (t^2*r^2-1)/cos^2 = 0 (dividing by cos^2(theta_lab))
    # This simplifies to a quadratic in cos(theta_cm):
    #   (1 + t^2) x^2 + 2 r t^2 x + (r^2 t^2 - 1) = 0
    # which becomes (since 1+t^2 = 1/cos^2(theta_lab)):
    #   x^2 + 2 r sin^2(theta_lab)/cos^2(theta_lab) * cos^2(theta_lab) x + ...
    # Let me redo with exact trig:
    sin_l = math.sin(theta_lab)
    cos_l = math.cos(theta_lab)

    if abs(sin_l) < 1e-15:
        # theta_lab = 0 → theta_cm = 0
        return 0.0

    # Quadratic: (1 + t²) x² + 2r t² x + (r²t² - 1) = 0, x = cos(theta_cm)
    t_sq = math.tan(theta_lab) ** 2
    a_coef = 1.0 + t_sq
    b_coef = 2.0 * ratio * t_sq
    c_coef = ratio**2 * t_sq - 1.0
    discriminant = b_coef**2 - 4.0 * a_coef * c_coef
    if discriminant < 0.0:
        raise ValueError(
            f"theta_lab={theta_lab:.4f} rad is not kinematically reachable "
            f"for m1={m1}, m2={m2}"
        )
    sqrt_disc = math.sqrt(discriminant)
    # Two solutions; pick the one in [0, π] consistent with positive sin(theta_cm)
    x1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    x2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    # Clamp to [-1, 1] for numerical safety
    x1 = max(-1.0, min(1.0, x1))
    x2 = max(-1.0, min(1.0, x2))
    # Choose solution where sin(theta_cm) has same sign as sin(theta_lab) (>= 0)
    theta_cm_1 = math.acos(x1)  # always in [0, π]
    theta_cm_2 = math.acos(x2)
    # Verify which solution reproduces theta_lab
    err1 = abs(classical_cm_to_lab_angle(theta_cm_1, m1, m2) - theta_lab)
    err2 = abs(classical_cm_to_lab_angle(theta_cm_2, m1, m2) - theta_lab)
    return theta_cm_1 if err1 <= err2 else theta_cm_2


def classical_recoil_angle_from_cm(theta_cm):
    """Lab recoil angle of the target as a function of CoM angle.

    The recoil nucleus always goes off in the forward hemisphere:

    φ_lab = (π − θ_cm) / 2

    Parameters
    ----------
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Lab recoil angle in radians [0, π/2].

    Raises
    ------
    ValueError
        If *theta_cm* is outside [0, π].

    Examples
    --------
    θ_cm = π (head-on) → φ_lab = 0 (recoil along beam):

    >>> import math
    >>> classical_recoil_angle_from_cm(math.pi)
    0.0
    """
    _validate_angle(theta_cm, "theta_cm")
    return (math.pi - theta_cm) / 2.0


# ---------------------------------------------------------------------------
# Neutron-nucleus convenience functions (classical)
# ---------------------------------------------------------------------------

def neutron_scattered_energy(T, mass_number, theta_cm):
    """Scattered neutron kinetic energy after elastic collision with a nucleus.

    Uses the classical elastic-scattering formula with m_neutron ≈ 1 amu
    and M_nucleus = A amu (mass-number approximation).

    T_n' = T × (1 + A² + 2A cos θ_cm) / (A + 1)²

    Parameters
    ----------
    T : float
        Incident neutron kinetic energy (any unit).
    mass_number : int or float
        Mass number *A* of the target nucleus.
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Scattered neutron energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, *mass_number* is non-positive, or *theta_cm*
        is outside [0, π].

    Examples
    --------
    1 MeV neutron backscattered (θ_cm = π) from ¹H (A=1):

    >>> neutron_scattered_energy(1.0, 1, math.pi)
    0.0
    """
    return classical_scattered_energy_cm(T, 1.0, float(mass_number), theta_cm)


def neutron_recoil_energy(T, mass_number, theta_cm):
    """Recoil kinetic energy of a nucleus struck by a neutron (classical).

    T_A' = T × 2A(1 − cos θ_cm) / (A + 1)²

    Parameters
    ----------
    T : float
        Incident neutron kinetic energy.
    mass_number : int or float
        Mass number *A* of the target nucleus.
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Recoil nuclear kinetic energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, *mass_number* is non-positive, or *theta_cm*
        is outside [0, π].

    Examples
    --------
    Maximum recoil for neutron on ¹H (θ_cm = π) — full energy transfer:

    >>> neutron_recoil_energy(1.0, 1, math.pi)
    1.0
    """
    return classical_recoil_energy_cm(T, 1.0, float(mass_number), theta_cm)


def neutron_max_energy_transfer(T, mass_number):
    """Maximum kinetic energy transferable from a neutron to a nucleus.

    ΔT_max = T × 4A / (A + 1)²

    Parameters
    ----------
    T : float
        Incident neutron kinetic energy.
    mass_number : int or float
        Mass number *A* of the target nucleus.

    Returns
    -------
    float
        Maximum transferable energy (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative or *mass_number* is non-positive.

    Examples
    --------
    >>> neutron_max_energy_transfer(1.0, 1)   # ¹H: full transfer
    1.0
    >>> neutron_max_energy_transfer(1.0, 12)  # ¹²C
    0.28402366863905326
    """
    return classical_max_energy_transfer(T, 1.0, float(mass_number))


def neutron_energy_after_moderation(T, mass_number, n_collisions, avg_cos_cm=0.0):
    """Average neutron energy after *n* elastic scattering collisions.

    Assumes isotropic scattering in the CoM frame (average cos θ_cm = 0)
    unless *avg_cos_cm* is specified.  The average energy-loss factor per
    collision is:

    <T'/T> = (1 + α) / 2  where  α = ((A−1)/(A+1))²

    with the generalisation for anisotropic scattering:

    <T'/T> = (m1²+m2²)/(m1+m2)² + 2 m1 m2/(m1+m2)² × avg_cos_cm

    Parameters
    ----------
    T : float
        Initial neutron kinetic energy.
    mass_number : int or float
        Mass number *A* of the moderator nucleus.
    n_collisions : int
        Number of elastic scattering collisions.
    avg_cos_cm : float, optional
        Average value of cos θ_cm (default 0 = isotropic in CoM frame).

    Returns
    -------
    float
        Average neutron energy after *n* collisions (same unit as *T*).

    Raises
    ------
    ValueError
        If *T* is negative, *mass_number* is non-positive, *n_collisions*
        is negative, or *avg_cos_cm* is outside [−1, 1].

    Examples
    --------
    1 MeV neutron after 1 isotropic scatter from ¹H (A=1): 50 % energy loss:

    >>> neutron_energy_after_moderation(1.0, 1, 1)
    0.5
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(float(mass_number), "mass_number")
    if n_collisions < 0:
        raise ValueError("n_collisions must be non-negative")
    if not -1.0 <= avg_cos_cm <= 1.0:
        raise ValueError("avg_cos_cm must be in [-1, 1]")
    A = float(mass_number)
    m1, m2 = 1.0, A
    mean_ratio = (m1**2 + m2**2) / (m1 + m2) ** 2 + 2.0 * m1 * m2 / (m1 + m2) ** 2 * avg_cos_cm
    return T * mean_ratio**n_collisions


# ---------------------------------------------------------------------------
# Relativistic two-body elastic scattering
# ---------------------------------------------------------------------------

def relativistic_invariant_mass(T, m1_mev, m2_mev):
    """Lorentz-invariant Mandelstam variable √s (centre-of-mass energy).

    For projectile 1 (rest mass m1, kinetic energy T) on target 2 (rest mass
    m2, at rest in lab):

    s = (T + m1 + m2)² − (T² + 2 T m1)  =  (m1 + m2)² + 2 m2 T

    √s is the total CoM energy including rest masses.

    Parameters
    ----------
    T : float
        Projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².

    Returns
    -------
    float
        Invariant mass √s in MeV.

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    1 MeV neutron on ¹H at rest (m ≈ 938.272 MeV):

    >>> import math
    >>> sqrt_s = relativistic_invariant_mass(1.0, 939.565, 938.272)
    >>> sqrt_s > 939.565 + 938.272
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    s = (m1_mev + m2_mev) ** 2 + 2.0 * m2_mev * T
    return math.sqrt(s)


def relativistic_cm_momentum(T, m1_mev, m2_mev):
    """Magnitude of the CoM-frame three-momentum for each particle.

    In elastic scattering both particles have the same CoM momentum magnitude
    before and after the collision:

    p*² = [s − (m1 + m2)²][s − (m1 − m2)²] / (4 s)

    Parameters
    ----------
    T : float
        Projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².

    Returns
    -------
    float
        CoM momentum magnitude p* in MeV/c.

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    >>> p_star = relativistic_cm_momentum(100.0, 939.565, 938.272)
    >>> p_star > 0
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    s = sqrt_s**2
    p_star_sq = (s - (m1_mev + m2_mev) ** 2) * (s - (m1_mev - m2_mev) ** 2) / (4.0 * s)
    if p_star_sq < 0.0:
        p_star_sq = 0.0
    return math.sqrt(p_star_sq)


def relativistic_cm_lorentz_boost(T, m1_mev, m2_mev):
    """Lorentz boost parameters (β_cm, γ_cm) of the lab frame in the CoM frame.

    β_cm = |p_lab| / (E1_lab + m2)
    γ_cm = (E1_lab + m2) / √s

    where E1_lab = T + m1 is the total energy of the projectile in the lab.

    Parameters
    ----------
    T : float
        Projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².

    Returns
    -------
    tuple of float
        ``(beta_cm, gamma_cm)`` — the CoM boost velocity (0 ≤ β < 1) and
        Lorentz factor (γ ≥ 1).

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    >>> beta, gamma = relativistic_cm_lorentz_boost(100.0, 939.565, 938.272)
    >>> 0 < beta < 1 and gamma > 1
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    E1_lab = T + m1_mev
    p1_lab = math.sqrt(E1_lab**2 - m1_mev**2)
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    total_lab_energy = E1_lab + m2_mev
    beta_cm = p1_lab / total_lab_energy
    gamma_cm = total_lab_energy / sqrt_s
    return beta_cm, gamma_cm


def relativistic_scattered_energy(T, m1_mev, m2_mev, theta_cm):
    """Relativistic lab kinetic energy of the scattered projectile.

    Boost the CoM-frame four-momentum back to the lab frame after scattering
    at CoM angle *theta_cm*:

    E1*_cm = (s + m1² − m2²) / (2√s)
    E1_lab  = γ_cm (E1*_cm + β_cm p* cos θ_cm)
    T1_lab  = E1_lab − m1

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Scattered projectile kinetic energy in the lab frame (MeV).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    Forward scattering (θ_cm = 0) — no energy loss:

    >>> T = 100.0
    >>> m = 939.565
    >>> abs(relativistic_scattered_energy(T, m, m, 0.0) - T) < 1e-8
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    _validate_angle(theta_cm, "theta_cm")
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    s = sqrt_s**2
    p_star = relativistic_cm_momentum(T, m1_mev, m2_mev)
    beta_cm, gamma_cm = relativistic_cm_lorentz_boost(T, m1_mev, m2_mev)
    # CoM energy of projectile after elastic scatter
    E1_star = (s + m1_mev**2 - m2_mev**2) / (2.0 * sqrt_s)
    # Boost back to lab
    E1_lab = gamma_cm * (E1_star + beta_cm * p_star * math.cos(theta_cm))
    return E1_lab - m1_mev


def relativistic_recoil_energy(T, m1_mev, m2_mev, theta_cm):
    """Relativistic lab kinetic energy of the recoiling target nucleus.

    The recoil angle in the CoM frame is π − θ_cm (back-to-back with the
    projectile):

    E2*_cm = (s + m2² − m1²) / (2√s)
    E2_lab  = γ_cm (E2*_cm − β_cm p* cos θ_cm)
    T2_lab  = E2_lab − m2

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Recoil kinetic energy in the lab frame (MeV).

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    Forward scattering (θ_cm = 0) — zero recoil:

    >>> relativistic_recoil_energy(100.0, 939.565, 939.565, 0.0) < 1e-8
    True
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    _validate_angle(theta_cm, "theta_cm")
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    s = sqrt_s**2
    p_star = relativistic_cm_momentum(T, m1_mev, m2_mev)
    beta_cm, gamma_cm = relativistic_cm_lorentz_boost(T, m1_mev, m2_mev)
    # CoM energy of target after elastic scatter (recoil back at π − theta_cm)
    E2_star = (s + m2_mev**2 - m1_mev**2) / (2.0 * sqrt_s)
    # Boost back to lab (target moves backward in CoM, hence −cos θ_cm)
    E2_lab = gamma_cm * (E2_star - beta_cm * p_star * math.cos(theta_cm))
    return E2_lab - m2_mev


def relativistic_scattered_lab_angle(T, m1_mev, m2_mev, theta_cm):
    """Lab scattering angle of the projectile corresponding to a CoM angle.

    tan θ_lab = p* sin θ_cm / [γ_cm (β_cm E1*_cm + p* cos θ_cm)]

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Lab scattering angle in radians [0, π/2].

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    Forward scattering: θ_cm = 0 → θ_lab = 0:

    >>> relativistic_scattered_lab_angle(100.0, 939.565, 938.272, 0.0)
    0.0
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    _validate_angle(theta_cm, "theta_cm")
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    s = sqrt_s**2
    p_star = relativistic_cm_momentum(T, m1_mev, m2_mev)
    beta_cm, gamma_cm = relativistic_cm_lorentz_boost(T, m1_mev, m2_mev)
    E1_star = (s + m1_mev**2 - m2_mev**2) / (2.0 * sqrt_s)
    p_perp = p_star * math.sin(theta_cm)
    p_par = gamma_cm * (beta_cm * E1_star + p_star * math.cos(theta_cm))
    return math.atan2(p_perp, p_par)


def relativistic_recoil_lab_angle(T, m1_mev, m2_mev, theta_cm):
    """Lab recoil angle of the target nucleus corresponding to a CoM angle.

    tan φ_lab = p* sin θ_cm / [γ_cm (β_cm E2*_cm − p* cos θ_cm)]

    Note: the denominator uses the target's CoM momentum which points
    *opposite* to the projectile in the CoM frame, giving the recoil in
    the forward hemisphere.

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².
    theta_cm : float
        CoM scattering angle in radians [0, π].

    Returns
    -------
    float
        Lab recoil angle in radians [0, π/2].

    Raises
    ------
    ValueError
        If *T* is negative, either mass is non-positive, or *theta_cm* is
        outside [0, π].

    Examples
    --------
    Backscatter (θ_cm = π) → target recoils straight forward (φ_lab = 0):

    >>> relativistic_recoil_lab_angle(100.0, 939.565, 938.272, math.pi)
    0.0
    """
    _validate_positive(T, "T", allow_zero=True)
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    _validate_angle(theta_cm, "theta_cm")
    sqrt_s = relativistic_invariant_mass(T, m1_mev, m2_mev)
    s = sqrt_s**2
    p_star = relativistic_cm_momentum(T, m1_mev, m2_mev)
    beta_cm, gamma_cm = relativistic_cm_lorentz_boost(T, m1_mev, m2_mev)
    E2_star = (s + m2_mev**2 - m1_mev**2) / (2.0 * sqrt_s)
    p_perp = p_star * math.sin(theta_cm)
    # Target recoil goes backward in CoM → its lab-frame parallel momentum:
    p_par = gamma_cm * (beta_cm * E2_star - p_star * math.cos(theta_cm))
    return math.atan2(p_perp, p_par)


def relativistic_max_energy_transfer(T, m1_mev, m2_mev):
    """Maximum kinetic energy transferable to the target in a relativistic collision.

    Occurs at θ_cm = π (head-on):

    ΔT_max = 2 m2 p1² / [(E1 + m2)² − p1² cos²0]  (simplified)

    Computed exactly as: T2_max = relativistic_recoil_energy(T, m1, m2, π).

    Parameters
    ----------
    T : float
        Incident projectile kinetic energy in MeV.
    m1_mev : float
        Projectile rest-mass energy in MeV/c².
    m2_mev : float
        Target rest-mass energy in MeV/c².

    Returns
    -------
    float
        Maximum recoil kinetic energy in MeV.

    Raises
    ------
    ValueError
        If *T* is negative or either mass is non-positive.

    Examples
    --------
    >>> relativistic_max_energy_transfer(1.0, 939.565, 939.565) > 0
    True
    """
    return relativistic_recoil_energy(T, m1_mev, m2_mev, math.pi)


# ---------------------------------------------------------------------------
# Nuclear reaction Q-value and threshold energy
# ---------------------------------------------------------------------------

def q_value(initial_masses_mev, final_masses_mev):
    """Compute the Q-value for a nuclear reaction.

    Q = (Σ m_initial − Σ m_final) × c²

    With masses expressed in MeV/c² the result is directly in MeV.

    Q > 0: exothermic (energy is released).
    Q < 0: endothermic (energy must be supplied).

    Parameters
    ----------
    initial_masses_mev : sequence of float
        Rest-mass energies of the reactants in MeV/c².
    final_masses_mev : sequence of float
        Rest-mass energies of the products in MeV/c².

    Returns
    -------
    float
        Q-value in MeV.

    Raises
    ------
    ValueError
        If either sequence is empty or any mass is negative.

    Examples
    --------
    n + ¹H → ²H + γ  (neutron capture, Q ≈ 2.22 MeV):

    >>> q = q_value([939.565, 938.272], [1875.613, 0.0])
    >>> abs(q - 2.224) < 0.01
    True
    """
    initial_masses_mev = list(initial_masses_mev)
    final_masses_mev = list(final_masses_mev)
    if not initial_masses_mev:
        raise ValueError("initial_masses_mev must not be empty")
    if not final_masses_mev:
        raise ValueError("final_masses_mev must not be empty")
    for m in initial_masses_mev + final_masses_mev:
        if m < 0.0:
            raise ValueError("All masses must be non-negative")
    return sum(initial_masses_mev) - sum(final_masses_mev)


def threshold_energy(m1_mev, m2_mev, product_masses_mev):
    """Minimum lab kinetic energy for an endothermic reaction to occur.

    For reaction 1 + 2 → products (with 2 at rest in the lab):

    T_th = [(Σ m_products)² − (m1 + m2)²] / (2 m2)

    Returns 0 for exothermic reactions (no threshold).

    Parameters
    ----------
    m1_mev : float
        Rest-mass energy of the projectile in MeV/c².
    m2_mev : float
        Rest-mass energy of the target in MeV/c².
    product_masses_mev : sequence of float
        Rest-mass energies of all products in MeV/c².

    Returns
    -------
    float
        Threshold kinetic energy in MeV (0 if Q ≥ 0).

    Raises
    ------
    ValueError
        If any mass is non-positive or *product_masses_mev* is empty.

    Examples
    --------
    Endothermic reaction: threshold > 0:

    >>> threshold_energy(938.272, 938.272, [938.272, 938.272, 134.976]) > 0
    True

    Elastic scattering (products = reactants): threshold = 0:

    >>> threshold_energy(939.565, 938.272, [939.565, 938.272])
    0.0
    """
    _validate_positive(m1_mev, "m1_mev")
    _validate_positive(m2_mev, "m2_mev")
    product_masses_mev = list(product_masses_mev)
    if not product_masses_mev:
        raise ValueError("product_masses_mev must not be empty")
    for m in product_masses_mev:
        _validate_positive(m, "product mass")
    sum_initial = m1_mev + m2_mev
    sum_final = sum(product_masses_mev)
    if sum_final <= sum_initial:
        return 0.0
    return (sum_final**2 - sum_initial**2) / (2.0 * m2_mev)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_positive(value, name, allow_zero=False):
    """Raise ValueError if *value* is negative (or zero when not allowed)."""
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")


def _validate_angle(theta, name):
    """Raise ValueError if *theta* is outside [0, π]."""
    if not 0.0 <= theta <= math.pi:
        raise ValueError(
            f"{name} must be in [0, π] radians, got {theta}"
        )
