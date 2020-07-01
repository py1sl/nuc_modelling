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

