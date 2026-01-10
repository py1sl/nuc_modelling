"""
Radiation physics functions for nuclear modelling.

This module provides functions for calculating:
- Decay constants and half-lives
- Radioactive activity
- Radiation flux from various source geometries
"""

import numpy as np


def decay_constant(half_life):
    """
    Calculate the decay constant from half-life.
    
    The decay constant (λ) relates to half-life (t₁/₂) by:
    λ = ln(2) / t₁/₂
    
    Args:
        half_life: Half-life of the radioactive isotope (in any time unit)
    
    Returns:
        Decay constant (in inverse of the same time unit as half_life)
    
    Example:
        >>> decay_constant(10.0)  # half-life of 10 seconds
        0.06931471805599453  # decay constant in s^-1
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return np.log(2) / half_life


def half_life(decay_constant_value):
    """
    Calculate the half-life from decay constant.
    
    The half-life (t₁/₂) relates to decay constant (λ) by:
    t₁/₂ = ln(2) / λ
    
    Args:
        decay_constant_value: Decay constant (in inverse time unit)
    
    Returns:
        Half-life (in the same time unit as decay_constant)
    
    Example:
        >>> half_life(0.0693)  # decay constant of 0.0693 s^-1
        10.003341995507592  # half-life in seconds
    """
    if decay_constant_value <= 0:
        raise ValueError("Decay constant must be positive")
    return np.log(2) / decay_constant_value


def activity(decay_constant_value, num_nuclei):
    """
    Calculate radioactive activity.
    
    Activity (A) is the rate of decay given by:
    A = λN
    
    where λ is the decay constant and N is the number of radioactive nuclei.
    
    Args:
        decay_constant_value: Decay constant (in inverse time unit, e.g., s^-1)
        num_nuclei: Number of radioactive nuclei
    
    Returns:
        Activity (in Becquerels if decay_constant in s^-1, i.e., decays per second)
    
    Example:
        >>> activity(0.0693, 1e10)  # 0.0693 s^-1, 10 billion nuclei
        693000000.0  # activity in Bq
    """
    if decay_constant_value < 0:
        raise ValueError("Decay constant must be non-negative")
    if num_nuclei < 0:
        raise ValueError("Number of nuclei must be non-negative")
    return decay_constant_value * num_nuclei


def radiation_point_source(activity_value, distance):
    """
    Calculate radiation flux from a point source.
    
    For a point source, the radiation flux follows an inverse square law:
    Φ = A / (4πr²)
    
    where A is the activity and r is the distance from the source.
    
    Args:
        activity_value: Source activity (in Bq or other activity unit)
        distance: Distance from the source (in meters or other length unit)
    
    Returns:
        Radiation flux (particles per unit area per unit time)
        Units: activity_unit / length_unit² (e.g., Bq/m² or particles/(cm²·s))
    
    Example:
        >>> radiation_point_source(1e6, 1.0)  # 1 MBq at 1 meter
        79577.47154594767  # flux in particles/(m²·s)
    """
    if activity_value < 0:
        raise ValueError("Activity must be non-negative")
    if distance <= 0:
        raise ValueError("Distance must be positive")
    return activity_value / (4 * np.pi * distance**2)


def radiation_infinite_line_source(activity_per_length, distance):
    """
    Calculate radiation flux from an infinite line source.
    
    For an infinite line source, the radiation flux follows:
    Φ = λ_L / (2πr)
    
    where λ_L is the activity per unit length and r is the perpendicular
    distance from the line.
    
    Args:
        activity_per_length: Activity per unit length (e.g., Bq/m)
        distance: Perpendicular distance from the line source (in same length unit)
    
    Returns:
        Radiation flux (particles per unit area per unit time)
        Units: activity_per_length / length (e.g., Bq/m²)
    
    Example:
        >>> radiation_infinite_line_source(1e6, 1.0)  # 1 MBq/m at 1 m distance
        159154.94309189535  # flux in particles/(m²·s)
    """
    if activity_per_length < 0:
        raise ValueError("Activity per length must be non-negative")
    if distance <= 0:
        raise ValueError("Distance must be positive")
    return activity_per_length / (2 * np.pi * distance)


def radiation_semi_infinite_plane(activity_per_area, distance=None):
    """
    Calculate radiation flux from a semi-infinite plane source.
    
    For a semi-infinite plane source, the radiation flux at a point directly
    above the plane is:
    Φ = σ / 2
    
    where σ is the activity per unit area. The flux is independent of distance
    for an infinite plane (assuming no attenuation).
    
    Args:
        activity_per_area: Activity per unit area (e.g., Bq/m²)
        distance: Distance from the plane (optional, included for API consistency
                  but not used in calculation for infinite plane)
    
    Returns:
        Radiation flux (particles per unit area per unit time)
        Units: same as activity_per_area (e.g., Bq/m²)
    
    Example:
        >>> radiation_semi_infinite_plane(1e6)  # 1 MBq/m² surface activity
        500000.0  # flux in particles/(m²·s)
    
    Note:
        For a truly semi-infinite plane, half the radiation goes into the half-space
        above the plane, resulting in Φ = σ/2. This assumes no attenuation and
        isotropic emission.
    """
    if activity_per_area < 0:
        raise ValueError("Activity per area must be non-negative")
    if distance is not None and distance < 0:
        raise ValueError("Distance must be non-negative")
    return activity_per_area / 2
