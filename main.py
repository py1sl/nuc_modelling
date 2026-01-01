"""
Short demo code for nucleus simulation
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import glob
import os


class particle():
    """ particle info """
    
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
        
      
def em_force(p1, p2):
    return 0

def strong_force(p1, p2):
    return 0


# ============================================================================
# SUGGESTED FUNCTIONS FOR BASIC NUCLEAR MODELS
# ============================================================================
# The following are suggestions for functions that could be implemented to
# model historical atomic and nuclear models. These are educational models
# that helped develop our understanding of atomic structure.
#
# PLUM PUDDING MODEL (J.J. Thomson, 1904)
# ---------------------------------------
# This model depicted the atom as a sphere of positive charge with electrons
# embedded throughout, like plums in a pudding.
#
# Suggested functions:
#
# def plum_pudding_potential(r, R, Z, e):
#     """
#     Calculate the electric potential at radius r in the plum pudding model.
#     
#     Parameters:
#     -----------
#     r : float
#         Distance from the center of the atom (in meters)
#     R : float
#         Radius of the positive charge sphere (in meters)
#     Z : int
#         Atomic number (number of protons/positive charges)
#     e : float
#         Elementary charge (1.602e-19 C)
#     
#     Returns:
#     --------
#     float : Electric potential in Volts
#     
#     Note: For r < R (inside the sphere), the potential would be calculated
#     assuming uniform positive charge distribution. For r >= R, it would
#     follow Coulomb's law for a point charge.
#     """
#     pass
#
# def plum_pudding_electric_field(r, R, Z, e):
#     """
#     Calculate the electric field at radius r in the plum pudding model.
#     
#     Parameters:
#     -----------
#     r : float
#         Distance from the center of the atom (in meters)
#     R : float
#         Radius of the positive charge sphere (in meters)
#     Z : int
#         Atomic number (number of protons/positive charges)
#     e : float
#         Elementary charge (1.602e-19 C)
#     
#     Returns:
#     --------
#     float : Electric field strength in V/m
#     
#     Note: Inside the sphere (r < R), the field increases linearly with r.
#     Outside (r >= R), it follows the inverse square law.
#     """
#     pass
#
# def plum_pudding_electron_oscillation_frequency(R, Z, m_e, e):
#     """
#     Calculate the natural oscillation frequency of an electron in the
#     plum pudding model when displaced from equilibrium.
#     
#     Parameters:
#     -----------
#     R : float
#         Radius of the positive charge sphere (in meters)
#     Z : int
#         Atomic number
#     m_e : float
#         Electron mass (9.109e-31 kg)
#     e : float
#         Elementary charge (1.602e-19 C)
#     
#     Returns:
#     --------
#     float : Oscillation frequency in Hz
#     
#     Note: This could be compared with spectral line frequencies to test
#     the model's validity.
#     """
#     pass
#
#
# RUTHERFORD MODEL (Ernest Rutherford, 1911)
# -------------------------------------------
# Based on the gold foil experiment, this model proposed a small, dense,
# positively charged nucleus with electrons orbiting around it.
#
# Suggested functions:
#
# def rutherford_scattering_angle(b, Z_projectile, Z_target, E_kinetic):
#     """
#     Calculate the scattering angle for a charged particle in Rutherford
#     scattering (Coulomb scattering from a nucleus).
#     
#     Parameters:
#     -----------
#     b : float
#         Impact parameter (perpendicular distance from the nucleus, in meters)
#     Z_projectile : int
#         Atomic number of the projectile (e.g., 2 for alpha particle)
#     Z_target : int
#         Atomic number of the target nucleus
#     E_kinetic : float
#         Kinetic energy of the projectile (in Joules)
#     
#     Returns:
#     --------
#     float : Scattering angle theta in radians
#     
#     Note: Uses the classical Coulomb scattering formula derived by Rutherford.
#     """
#     pass
#
# def rutherford_differential_cross_section(theta, Z_projectile, Z_target, E_kinetic):
#     """
#     Calculate the Rutherford differential scattering cross section.
#     
#     Parameters:
#     -----------
#     theta : float
#         Scattering angle in radians
#     Z_projectile : int
#         Atomic number of the projectile
#     Z_target : int
#         Atomic number of the target nucleus
#     E_kinetic : float
#         Kinetic energy of the projectile (in Joules)
#     
#     Returns:
#     --------
#     float : Differential cross section d(sigma)/d(Omega) in m^2/sr
#     
#     Note: This is the famous Rutherford formula that agrees with
#     experimental gold foil results.
#     """
#     pass
#
# def rutherford_closest_approach(Z_projectile, Z_target, E_kinetic):
#     """
#     Calculate the distance of closest approach in a head-on collision
#     between a charged projectile and a nucleus.
#     
#     Parameters:
#     -----------
#     Z_projectile : int
#         Atomic number of the projectile (e.g., 2 for alpha particle)
#     Z_target : int
#         Atomic number of the target nucleus
#     E_kinetic : float
#         Kinetic energy of the projectile (in Joules)
#     
#     Returns:
#     --------
#     float : Distance of closest approach in meters
#     
#     Note: At closest approach, all kinetic energy is converted to
#     electrostatic potential energy. This gives an upper bound on nuclear size.
#     """
#     pass
#
# def rutherford_impact_parameter(theta, Z_projectile, Z_target, E_kinetic):
#     """
#     Calculate the impact parameter needed to achieve a given scattering angle.
#     
#     Parameters:
#     -----------
#     theta : float
#         Desired scattering angle in radians
#     Z_projectile : int
#         Atomic number of the projectile
#     Z_target : int
#         Atomic number of the target nucleus
#     E_kinetic : float
#         Kinetic energy of the projectile (in Joules)
#     
#     Returns:
#     --------
#     float : Impact parameter in meters
#     
#     Note: Inverse of rutherford_scattering_angle function.
#     """
#     pass
#
# def rutherford_orbital_radius(n, Z, m_e, e, h_bar):
#     """
#     Calculate the orbital radius in the Rutherford-Bohr model for a given
#     quantum number (this bridges to Bohr's refinement of Rutherford's model).
#     
#     Parameters:
#     -----------
#     n : int
#         Principal quantum number (1, 2, 3, ...)
#     Z : int
#         Atomic number of the nucleus
#     m_e : float
#         Electron mass (9.109e-31 kg)
#     e : float
#         Elementary charge (1.602e-19 C)
#     h_bar : float
#         Reduced Planck constant (1.055e-34 J·s)
#     
#     Returns:
#     --------
#     float : Orbital radius in meters
#     
#     Note: This extends Rutherford's model with Bohr's quantum conditions.
#     """
#     pass
#


def SEMF(n, z):
    """ semi empirical mass formula
        n = neutron number
        z = proton number
    """
    
    # atomic mass
    a = n + z

    # parameters from J. W. Rohlf, "Modern Physics from alpha to Z0", Wiley (1994).
    aV = 15.75
    aS = 17.8
    aC = 0.711
    aA = 23.7
    delta = 11.18
    
    # need to know if odd or even numbers of neutrons and protons
    if ((n%2) == 0) and ((z%2) == 0):
        sgn = 1
    elif ((n%2) != 0) and ((z%2) != 0):
        sgn = -1
    else:
        sgn = 0

    # The SEMF for the average binding energy per nucleon.
    E = (aV - aS / a**(1/3) - aC * z**2 / a**(4/3) -
         aA * (a-2*z)**2/a**2 + sgn * delta/a**(3/2))

    return E

def main():
    # set logging up
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    logging.info('Started')
    logging.info("Reading data")

    bind_erg = SEMF(10, 10)
    print(bind_erg)

    logging.info("Analysis complete")


if __name__ == "__main__":
    main()

