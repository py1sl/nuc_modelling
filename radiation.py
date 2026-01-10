"""
Radiation physics functions for nuclear modelling.

This module provides functions for calculating:
- Decay constants and half-lives
- Radioactive activity
- Radiation flux from various source geometries
"""

import numpy as np


def decay_constant(half_life):
    """Calculate the decay constant from half-life: λ = ln(2) / t₁/₂"""
    if half_life <= 0:
        raise ValueError("Half-life must be positive")
    return np.log(2) / half_life


def half_life(decay_constant_value):
    """Calculate the half-life from decay constant: t₁/₂ = ln(2) / λ"""
    if decay_constant_value <= 0:
        raise ValueError("Decay constant must be positive")
    return np.log(2) / decay_constant_value


def activity(decay_constant_value, num_nuclei):
    """Calculate radioactive activity: A = λN"""
    if decay_constant_value < 0:
        raise ValueError("Decay constant must be non-negative")
    if num_nuclei < 0:
        raise ValueError("Number of nuclei must be non-negative")
    return decay_constant_value * num_nuclei


def activity_at_time(initial_activity, decay_constant_value, time):
    """
    Calculate radioactive activity at a future time.
    
    The activity decays exponentially according to:
    A(t) = A₀ * e^(-λt)
    
    where A₀ is the initial activity, λ is the decay constant, and t is time.
    
    Args:
        initial_activity: Initial activity at t=0 (in Bq or other activity unit)
        decay_constant_value: Decay constant (in inverse time unit, e.g., s^-1)
        time: Time elapsed (in same time unit as decay_constant)
    
    Returns:
        Activity at the specified time (same units as initial_activity)
    
    Example:
        >>> activity_at_time(1000.0, 0.0693, 10.0)  # 1000 Bq, λ=0.0693 s^-1, t=10s
        500.0  # approximately half after one half-life
    """
    if initial_activity < 0:
        raise ValueError("Initial activity must be non-negative")
    if decay_constant_value < 0:
        raise ValueError("Decay constant must be non-negative")
    if time < 0:
        raise ValueError("Time must be non-negative")
    return initial_activity * np.exp(-decay_constant_value * time)


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
