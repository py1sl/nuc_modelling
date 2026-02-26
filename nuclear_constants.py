"""
Nuclear and physical constants for nuclear modelling.

All values are in SI units unless otherwise noted.
"""

import math

# Elementary charge (Coulombs)
ELEMENTARY_CHARGE = 1.602176634e-19  # C

# Particle masses (kg)
PROTON_MASS = 1.67262192369e-27     # kg
NEUTRON_MASS = 1.67492749804e-27    # kg
ELECTRON_MASS = 9.1093837015e-31    # kg

# Speed of light (m/s)
SPEED_OF_LIGHT = 2.99792458e8       # m/s
C = SPEED_OF_LIGHT

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT = 1.380649e-23   # J/K
K_B = BOLTZMANN_CONSTANT

# Avogadro constant (1/mol)
AVOGADRO_CONSTANT = 6.02214076e23   # 1/mol
N_A = AVOGADRO_CONSTANT

# Planck constant (J·s)
PLANCK_CONSTANT = 6.62607015e-34    # J·s
H = PLANCK_CONSTANT

# Reduced Planck constant (J·s)
HBAR = PLANCK_CONSTANT / (2 * math.pi)  # J·s

# --- Energy conversions ---
# 1 eV in Joules
EV_TO_J = ELEMENTARY_CHARGE         # J/eV  (1 eV = 1.602e-19 J)
J_TO_EV = 1.0 / EV_TO_J             # eV/J

# keV, MeV, GeV helpers (in Joules)
KEV_TO_J = EV_TO_J * 1e3            # J/keV
MEV_TO_J = EV_TO_J * 1e6            # J/MeV
GEV_TO_J = EV_TO_J * 1e9            # J/GeV

J_TO_KEV = 1.0 / KEV_TO_J           # keV/J
J_TO_MEV = 1.0 / MEV_TO_J           # MeV/J
J_TO_GEV = 1.0 / GEV_TO_J           # GeV/J

# --- Activity conversions ---
# 1 Curie in Becquerel
CURIE_TO_BQ = 3.7e10                # Bq/Ci
BQ_TO_CURIE = 1.0 / CURIE_TO_BQ    # Ci/Bq

MILLICURIE_TO_BQ = CURIE_TO_BQ * 1e-3   # Bq/mCi
MICROCURIE_TO_BQ = CURIE_TO_BQ * 1e-6   # Bq/µCi

# --- Cross-section conversions ---
# 1 barn in m²
BARN_TO_M2 = 1e-28                  # m²/barn
M2_TO_BARN = 1.0 / BARN_TO_M2      # barn/m²

MILLIBARN_TO_M2 = BARN_TO_M2 * 1e-3    # m²/mb
MICROBARN_TO_M2 = BARN_TO_M2 * 1e-6   # m²/µb
NANOBARN_TO_M2 = BARN_TO_M2 * 1e-9    # m²/nb
PICOBARN_TO_M2 = BARN_TO_M2 * 1e-12   # m²/pb
