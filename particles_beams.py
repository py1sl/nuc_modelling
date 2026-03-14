"""
Particle beams module for nuclear modelling.

This module provides functions for calculating basic particle beam parameters:

- Beam power (average and peak)
- Particle flux (particles per second)
- Particles per pulse
- Average and peak current relationships
- Duty factor
- Relativistic beam parameters (β, γ, velocity, magnetic rigidity)

Functions accepting a :class:`~particle.particle` instance are provided
alongside standalone versions that take explicit numeric arguments.
"""

import math

from nuclear_constants import ELEMENTARY_CHARGE, MEV_TO_J, SPEED_OF_LIGHT
from particle import particle


# ---------------------------------------------------------------------------
# Current / power relationships
# ---------------------------------------------------------------------------

def beam_power(current, energy_mev, charge_number=1, duty_factor=1.0):
    """Calculate average beam power.

    For a pulsed beam the average power is:

    P = I_peak × duty_factor × E / q

    where E is the kinetic energy per particle in Joules and q is the particle
    charge in Coulombs.  For a continuous-wave (CW) beam set
    ``duty_factor=1.0`` (the default).

    Parameters
    ----------
    current : float
        Beam current in Amperes.  For pulsed beams this is the *peak*
        current; the duty factor is applied internally.
    energy_mev : float
        Particle kinetic energy in MeV.
    charge_number : int or float, optional
        Particle charge in units of elementary charge (e.g. 1 for protons,
        2 for alpha particles, -1 for electrons).  Default is 1.
    duty_factor : float, optional
        Duty factor (dimensionless, 0 < duty_factor <= 1).  For a CW beam
        use the default of 1.0.

    Returns
    -------
    float
        Average beam power in Watts.

    Raises
    ------
    ValueError
        If *current* or *energy_mev* is negative, *duty_factor* is not in
        (0, 1], or *charge_number* is zero.

    Examples
    --------
    100 MeV proton beam at 1 mA average current:

    >>> beam_power(1e-3, 100.0)
    100000.0
    """
    if current < 0:
        raise ValueError("Current must be non-negative")
    if energy_mev < 0:
        raise ValueError("Energy must be non-negative")
    if not (0 < duty_factor <= 1.0):
        raise ValueError("Duty factor must be in (0, 1]")
    if charge_number == 0:
        raise ValueError("Charge number must not be zero")

    energy_j = energy_mev * MEV_TO_J
    charge_c = abs(charge_number) * ELEMENTARY_CHARGE
    avg_current = current * duty_factor
    return avg_current * energy_j / charge_c


def particles_per_second(current, charge_number=1):
    """Calculate the number of particles per second from average beam current.

    N = I / q

    Parameters
    ----------
    current : float
        Average beam current in Amperes.
    charge_number : int or float, optional
        Particle charge in units of elementary charge.  Default is 1.

    Returns
    -------
    float
        Number of particles per second.

    Raises
    ------
    ValueError
        If *current* is negative or *charge_number* is zero.

    Examples
    --------
    >>> particles_per_second(1e-3)  # 1 mA singly-charged beam
    6.241509074460763e+15
    """
    if current < 0:
        raise ValueError("Current must be non-negative")
    if charge_number == 0:
        raise ValueError("Charge number must not be zero")

    charge_c = abs(charge_number) * ELEMENTARY_CHARGE
    return current / charge_c


def particles_per_pulse(peak_current, pulse_length_s, charge_number=1):
    """Calculate the number of particles delivered in a single pulse.

    N_pulse = I_peak × t_pulse / q

    Parameters
    ----------
    peak_current : float
        Peak beam current during the pulse in Amperes.
    pulse_length_s : float
        Pulse duration in seconds.
    charge_number : int or float, optional
        Particle charge in units of elementary charge.  Default is 1.

    Returns
    -------
    float
        Number of particles per pulse.

    Raises
    ------
    ValueError
        If *peak_current* or *pulse_length_s* is negative, or
        *charge_number* is zero.

    Examples
    --------
    >>> particles_per_pulse(0.1, 1e-3)  # 100 mA peak, 1 ms pulse
    6.241509074460763e+14
    """
    if peak_current < 0:
        raise ValueError("Peak current must be non-negative")
    if pulse_length_s < 0:
        raise ValueError("Pulse length must be non-negative")
    if charge_number == 0:
        raise ValueError("Charge number must not be zero")

    charge_c = abs(charge_number) * ELEMENTARY_CHARGE
    return peak_current * pulse_length_s / charge_c


def average_current(peak_current, duty_factor_val):
    """Calculate the time-averaged current from peak current and duty factor.

    I_avg = I_peak × duty_factor

    Parameters
    ----------
    peak_current : float
        Peak beam current in Amperes.
    duty_factor_val : float
        Duty factor (dimensionless, 0 < duty_factor_val <= 1).

    Returns
    -------
    float
        Average current in Amperes.

    Raises
    ------
    ValueError
        If *peak_current* is negative or *duty_factor_val* is not in (0, 1].

    Examples
    --------
    >>> average_current(0.1, 0.05)  # 100 mA peak, 5 % duty factor
    0.005
    """
    if peak_current < 0:
        raise ValueError("Peak current must be non-negative")
    if not (0 < duty_factor_val <= 1.0):
        raise ValueError("Duty factor must be in (0, 1]")
    return peak_current * duty_factor_val


def duty_factor(pulse_length_s, repetition_rate_hz):
    """Calculate the duty factor from pulse length and repetition rate.

    df = t_pulse × f_rep

    Parameters
    ----------
    pulse_length_s : float
        Pulse duration in seconds.
    repetition_rate_hz : float
        Pulse repetition rate in Hz.

    Returns
    -------
    float
        Duty factor (dimensionless, between 0 and 1).

    Raises
    ------
    ValueError
        If either argument is negative or if the resulting duty factor
        exceeds 1 (i.e. overlapping pulses).

    Examples
    --------
    >>> duty_factor(1e-3, 50.0)  # 1 ms pulses at 50 Hz
    0.05
    """
    if pulse_length_s < 0:
        raise ValueError("Pulse length must be non-negative")
    if repetition_rate_hz < 0:
        raise ValueError("Repetition rate must be non-negative")
    df = pulse_length_s * repetition_rate_hz
    if df > 1.0:
        raise ValueError(
            "Duty factor exceeds 1. Check pulse_length_s and repetition_rate_hz."
        )
    return df


def peak_current_from_average(average_current_a, duty_factor_val):
    """Calculate the peak current from average current and duty factor.

    I_peak = I_avg / duty_factor

    Parameters
    ----------
    average_current_a : float
        Average beam current in Amperes.
    duty_factor_val : float
        Duty factor (dimensionless, 0 < duty_factor_val <= 1).

    Returns
    -------
    float
        Peak current in Amperes.

    Raises
    ------
    ValueError
        If *average_current_a* is negative or *duty_factor_val* is not in
        (0, 1].

    Examples
    --------
    >>> peak_current_from_average(5e-3, 0.05)  # 5 mA average, 5 % duty factor
    0.1
    """
    if average_current_a < 0:
        raise ValueError("Average current must be non-negative")
    if not (0 < duty_factor_val <= 1.0):
        raise ValueError("Duty factor must be in (0, 1]")
    return average_current_a / duty_factor_val


# ---------------------------------------------------------------------------
# Relativistic beam parameters
# ---------------------------------------------------------------------------

def lorentz_gamma(kinetic_energy_mev, rest_mass_mev):
    """Calculate the Lorentz factor γ for a particle.

    γ = 1 + KE / (m₀c²)

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy of the particle in MeV.
    rest_mass_mev : float
        Rest mass energy of the particle in MeV (m₀c²).

    Returns
    -------
    float
        Lorentz factor γ (>= 1).

    Raises
    ------
    ValueError
        If *kinetic_energy_mev* is negative or *rest_mass_mev* is
        non-positive.

    Examples
    --------
    >>> lorentz_gamma(100.0, 938.272)  # 100 MeV proton
    1.1065878...
    """
    if kinetic_energy_mev < 0:
        raise ValueError("Kinetic energy must be non-negative")
    if rest_mass_mev <= 0:
        raise ValueError("Rest mass energy must be positive")
    return 1.0 + kinetic_energy_mev / rest_mass_mev


def lorentz_beta(kinetic_energy_mev, rest_mass_mev):
    """Calculate the normalised velocity β = v/c for a particle.

    β = sqrt(1 - 1/γ²)

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy of the particle in MeV.
    rest_mass_mev : float
        Rest mass energy of the particle in MeV (m₀c²).

    Returns
    -------
    float
        Normalised velocity β (0 <= β < 1).

    Raises
    ------
    ValueError
        If *kinetic_energy_mev* is negative or *rest_mass_mev* is
        non-positive.

    Examples
    --------
    >>> lorentz_beta(100.0, 938.272)  # 100 MeV proton
    0.42837...
    """
    gamma = lorentz_gamma(kinetic_energy_mev, rest_mass_mev)
    return math.sqrt(1.0 - 1.0 / gamma**2)


def beam_velocity(kinetic_energy_mev, rest_mass_mev):
    """Calculate the velocity of particles in a beam.

    v = β × c

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy of the particle in MeV.
    rest_mass_mev : float
        Rest mass energy of the particle in MeV (m₀c²).

    Returns
    -------
    float
        Particle velocity in m/s.

    Raises
    ------
    ValueError
        If *kinetic_energy_mev* is negative or *rest_mass_mev* is
        non-positive.

    Examples
    --------
    >>> beam_velocity(100.0, 938.272)  # 100 MeV proton
    1.28447...e+08
    """
    beta = lorentz_beta(kinetic_energy_mev, rest_mass_mev)
    return beta * SPEED_OF_LIGHT


def magnetic_rigidity(kinetic_energy_mev, rest_mass_mev, charge_number=1):
    """Calculate the magnetic rigidity Bρ of a particle beam.

    Bρ = p / q  (in T·m)

    The relativistic momentum *p* is obtained from:

    pc = sqrt((KE + m₀c²)² − (m₀c²)²)  [in MeV]

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy of the particle in MeV.
    rest_mass_mev : float
        Rest mass energy of the particle in MeV (m₀c²).
    charge_number : int or float, optional
        Particle charge in units of elementary charge.  Default is 1.

    Returns
    -------
    float
        Magnetic rigidity Bρ in Tesla·metres (T·m).

    Raises
    ------
    ValueError
        If *kinetic_energy_mev* is negative, *rest_mass_mev* is
        non-positive, or *charge_number* is zero.

    Examples
    --------
    >>> magnetic_rigidity(100.0, 938.272)  # 100 MeV proton
    1.4...
    """
    if kinetic_energy_mev < 0:
        raise ValueError("Kinetic energy must be non-negative")
    if rest_mass_mev <= 0:
        raise ValueError("Rest mass energy must be positive")
    if charge_number == 0:
        raise ValueError("Charge number must not be zero")

    # Total energy in MeV
    total_energy_mev = kinetic_energy_mev + rest_mass_mev
    # Relativistic momentum × c in MeV
    pc_mev = math.sqrt(total_energy_mev**2 - rest_mass_mev**2)
    # Convert to SI: p = pc_mev × (e×10⁶) / c
    p_si = pc_mev * MEV_TO_J / SPEED_OF_LIGHT
    charge_c = abs(charge_number) * ELEMENTARY_CHARGE
    return p_si / charge_c


# ---------------------------------------------------------------------------
# Convenience wrappers that accept a particle instance
# ---------------------------------------------------------------------------

def beam_power_from_particle(p, current, duty_factor=1.0):
    """Calculate average beam power using a :class:`~particle.particle` instance.

    The particle's ``energy`` attribute (kinetic energy in MeV) and
    ``charge`` attribute (in Coulombs) are used automatically.

    Parameters
    ----------
    p : particle
        A particle instance with ``energy`` (MeV) and ``charge``
        (Coulombs) attributes set.
    current : float
        Beam current in Amperes.  For pulsed beams this is the *peak*
        current; the duty factor is applied internally.
    duty_factor : float, optional
        Duty factor (dimensionless, 0 < duty_factor <= 1).  Default is 1.0.

    Returns
    -------
    float
        Average beam power in Watts.

    Raises
    ------
    TypeError
        If *p* is not a :class:`~particle.particle` instance.
    ValueError
        If *current* is negative, *duty_factor* is not in (0, 1], or the
        particle charge is zero.
    """
    if not isinstance(p, particle):
        raise TypeError("p must be an instance of particle")
    if p.charge == 0:
        raise ValueError("Particle charge must not be zero")
    charge_number = p.charge / ELEMENTARY_CHARGE
    return beam_power(current, p.energy, charge_number=charge_number,
                      duty_factor=duty_factor)


def particles_per_second_from_particle(p, current):
    """Calculate particle flux from average beam current using a particle instance.

    Parameters
    ----------
    p : particle
        A particle instance with ``charge`` (Coulombs) attribute set.
    current : float
        Average beam current in Amperes.

    Returns
    -------
    float
        Number of particles per second.

    Raises
    ------
    TypeError
        If *p* is not a :class:`~particle.particle` instance.
    ValueError
        If *current* is negative or the particle charge is zero.
    """
    if not isinstance(p, particle):
        raise TypeError("p must be an instance of particle")
    if p.charge == 0:
        raise ValueError("Particle charge must not be zero")
    charge_number = p.charge / ELEMENTARY_CHARGE
    return particles_per_second(current, charge_number=charge_number)


def lorentz_gamma_from_particle(p):
    """Calculate the Lorentz factor γ using a particle instance.

    Uses the particle's ``energy`` (kinetic energy in MeV) and
    ``mass_mev`` (rest-mass energy in MeV) attributes.

    Parameters
    ----------
    p : particle
        A particle instance with ``energy`` and ``mass_mev`` attributes set.

    Returns
    -------
    float
        Lorentz factor γ.

    Raises
    ------
    TypeError
        If *p* is not a :class:`~particle.particle` instance.
    ValueError
        If the particle ``mass_mev`` is not positive (e.g. a bare
        ``particle()`` default instance).
    """
    if not isinstance(p, particle):
        raise TypeError("p must be an instance of particle")
    if p.mass_mev <= 0:
        raise ValueError(
            "Particle rest mass (mass_mev) must be positive. "
            "Use particle.from_name() or a typed subclass."
        )
    return lorentz_gamma(p.energy, p.mass_mev)


def lorentz_beta_from_particle(p):
    """Calculate the normalised velocity β = v/c using a particle instance.

    Uses the particle's ``energy`` (kinetic energy in MeV) and
    ``mass_mev`` (rest-mass energy in MeV) attributes.

    Parameters
    ----------
    p : particle
        A particle instance with ``energy`` and ``mass_mev`` attributes set.

    Returns
    -------
    float
        Normalised velocity β.

    Raises
    ------
    TypeError
        If *p* is not a :class:`~particle.particle` instance.
    ValueError
        If the particle ``mass_mev`` is not positive.
    """
    if not isinstance(p, particle):
        raise TypeError("p must be an instance of particle")
    if p.mass_mev <= 0:
        raise ValueError(
            "Particle rest mass (mass_mev) must be positive. "
            "Use particle.from_name() or a typed subclass."
        )
    return lorentz_beta(p.energy, p.mass_mev)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    """Demonstrate particle beam parameter calculations."""
    print("=== Particle Beam Parameter Calculations ===\n")

    # Example 1: 100 MeV proton beam, 1 mA CW
    print("Example 1: 100 MeV proton beam, 1 mA average current (CW)")
    I = 1e-3    # A
    E = 100.0   # MeV
    P = beam_power(I, E)
    N = particles_per_second(I)
    print(f"  Current:            {I * 1e3:.1f} mA")
    print(f"  Energy:             {E:.0f} MeV")
    print(f"  Beam power:         {P / 1e3:.1f} kW")
    print(f"  Particles/s:        {N:.3e}")
    print()

    # Example 2: Pulsed beam – 100 mA peak, 1 ms pulses at 50 Hz
    print("Example 2: Pulsed proton beam – 100 mA peak, 1 ms pulse, 50 Hz")
    I_peak = 0.1    # A
    t_pulse = 1e-3  # s
    f_rep = 50.0    # Hz
    df = duty_factor(t_pulse, f_rep)
    I_avg = average_current(I_peak, df)
    N_pulse = particles_per_pulse(I_peak, t_pulse)
    P_avg = beam_power(I_peak, E, duty_factor=df)
    print(f"  Peak current:       {I_peak * 1e3:.0f} mA")
    print(f"  Pulse length:       {t_pulse * 1e3:.1f} ms")
    print(f"  Repetition rate:    {f_rep:.0f} Hz")
    print(f"  Duty factor:        {df:.4f}")
    print(f"  Average current:    {I_avg * 1e6:.1f} µA")
    print(f"  Particles/pulse:    {N_pulse:.3e}")
    print(f"  Average power:      {P_avg:.2f} W")
    print()

    # Example 3: Relativistic parameters for a 100 MeV proton
    print("Example 3: Relativistic parameters for a 100 MeV proton")
    p = particle.from_name("proton")
    p.energy = 100.0  # MeV kinetic energy
    gamma = lorentz_gamma_from_particle(p)
    beta = lorentz_beta_from_particle(p)
    v = beam_velocity(p.energy, p.mass_mev)
    Br = magnetic_rigidity(p.energy, p.mass_mev)
    print(f"  Kinetic energy:     {p.energy:.0f} MeV")
    print(f"  Rest mass:          {p.mass_mev:.4f} MeV/c²")
    print(f"  γ (Lorentz factor): {gamma:.6f}")
    print(f"  β (v/c):            {beta:.6f}")
    print(f"  Velocity:           {v:.4e} m/s")
    print(f"  Magnetic rigidity:  {Br:.4f} T·m")
    print()


if __name__ == "__main__":
    main()
