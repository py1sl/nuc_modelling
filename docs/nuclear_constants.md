# Nuclear and Physical Constants

This document describes the physical constants and unit conversions defined in
`nuclear_constants.py` and used throughout the nuclear modelling codebase.

---

## Fundamental Constants

| Symbol | Name | Value | Unit |
|--------|------|-------|------|
| $e$ | Elementary charge | $1.602176634 \times 10^{-19}$ | C |
| $m_p$ | Proton mass | $1.67262192369 \times 10^{-27}$ | kg |
| $m_n$ | Neutron mass | $1.67492749804 \times 10^{-27}$ | kg |
| $m_e$ | Electron mass | $9.1093837015 \times 10^{-31}$ | kg |
| $c$ | Speed of light | $2.99792458 \times 10^{8}$ | m/s |
| $k_B$ | Boltzmann constant | $1.380649 \times 10^{-23}$ | J/K |
| $N_A$ | Avogadro constant | $6.02214076 \times 10^{23}$ | mol$^{-1}$ |
| $h$ | Planck constant | $6.62607015 \times 10^{-34}$ | J·s |
| $\hbar$ | Reduced Planck constant | $h / (2\pi)$ | J·s |

All values are exact 2018 CODATA recommended values (where applicable) or exact SI definitions.

---

## Energy–Mass Relation

The rest-mass energy of a particle with mass $m$ is given by Einstein's relation:

$$
E_0 = mc^2
$$

Nuclear energies are conventionally quoted in **electronvolts** (eV) or its multiples. The
conversion between joules and electronvolts uses the elementary charge:

$$
1\ \text{eV} = e \times 1\ \text{V} = 1.602176634 \times 10^{-19}\ \text{J}
$$

| Multiple | Abbreviation | Value in joules |
|----------|-------------|-----------------|
| 1 electronvolt | eV | $1.602 \times 10^{-19}$ J |
| 1 kiloelectronvolt | keV | $1.602 \times 10^{-16}$ J |
| 1 megaelectronvolt | MeV | $1.602 \times 10^{-13}$ J |
| 1 gigaelectronvolt | GeV | $1.602 \times 10^{-10}$ J |

---

## Particle Rest-Mass Energies

Using $E_0 = mc^2$:

| Particle | Rest mass energy |
|----------|----------------|
| Proton | $\approx 938.272\ \text{MeV}$ |
| Neutron | $\approx 939.565\ \text{MeV}$ |
| Electron | $\approx 0.511\ \text{MeV}$ |

The neutron is slightly heavier than the proton, which is why a free neutron decays to a proton
via $\beta^-$ decay with a mean lifetime of about 879 s.

---

## Activity Units

Radioactive activity is measured in **becquerels** (Bq) in the SI system:

$$
1\ \text{Bq} = 1\ \text{decay/s}
$$

The historical unit is the **curie** (Ci), defined as the activity of 1 gram of radium-226:

$$
1\ \text{Ci} = 3.7 \times 10^{10}\ \text{Bq}
$$

| Unit | In Bq |
|------|-------|
| 1 Ci | $3.7 \times 10^{10}$ |
| 1 mCi | $3.7 \times 10^{7}$ |
| 1 µCi | $3.7 \times 10^{4}$ |

---

## Cross-Section Units

Nuclear and particle physics cross sections are measured in **barns** (b):

$$
1\ \text{b} = 10^{-28}\ \text{m}^2 = 100\ \text{fm}^2
$$

The choice of unit reflects the approximate geometrical cross-sectional area of a heavy nucleus
($A \sim 100$), which is $\pi R^2 \approx \pi (r_0 A^{1/3})^2 \sim 1\ \text{b}$.

| Unit | In m² | In fm² |
|------|-------|--------|
| 1 b | $10^{-28}$ | $10^{2}$ |
| 1 mb | $10^{-31}$ | $10^{-1}$ |
| 1 µb | $10^{-34}$ | $10^{-4}$ |
| 1 nb | $10^{-37}$ | $10^{-7}$ |
| 1 pb | $10^{-40}$ | $10^{-10}$ |

---

## References

- CODATA 2018 recommended values: https://physics.nist.gov/cuu/Constants/
- K. S. Krane, *Introductory Nuclear Physics*, Wiley (1988), Appendix.
