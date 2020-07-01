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

def setup():
    """ """

    np = 10
    plist= []
    i = 0 
    while i < np:
        p = particle()
        p.x = i
        plist.append(p)
        i = i + 1
    return plist

def main():
    # set logging up
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    logging.info('Started')
    logging.info("Reading data")

    initial_plist = setup()
    initial_plist[0].v = 1
    t=0
    maxt= 10
    plist = initial_plist
    while t < maxt:
        for i, p in enumerate(plist):
            p.x = p.x + p.u
            p.y = p.y + p.v
            p.z = p.z + p.w
            plist[i] = p
        t = t + 1
    
    print(plist[0].y)

    logging.info("Analysis complete")


if __name__ == "__main__":
    main()

