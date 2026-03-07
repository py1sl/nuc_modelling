# Particle Properties

This document describes the physical properties of the fundamental particles represented in
`particle.py` and the nuclear physics quantities associated with them.

---

## Particle Classification

The particles in this module are the basic constituents of atomic nuclei and atoms:

| Particle | Symbol | Type | Charge | Spin | Rest mass |
|----------|--------|------|--------|------|-----------|
| Proton | $p$ | Baryon (fermion) | $+e$ | $\frac{1}{2}$ | $938.272\ \text{MeV}/c^2$ |
| Neutron | $n$ | Baryon (fermion) | $0$ | $\frac{1}{2}$ | $939.565\ \text{MeV}/c^2$ |
| Electron | $e^-$ | Lepton (fermion) | $-e$ | $\frac{1}{2}$ | $0.511\ \text{MeV}/c^2$ |
| Alpha | $\alpha$ | Nucleus ($^4$He) | $+2e$ | $0$ | $3727.4\ \text{MeV}/c^2$ |

where $e = 1.602176634 \times 10^{-19}\ \text{C}$ is the elementary charge.

---

## Proton

The proton is a **spin-$\frac{1}{2}$ baryon** composed of two up quarks and one down quark
(uud). It is the nucleus of the hydrogen atom and a constituent of all heavier nuclei.

$$
m_p = 1.67262 \times 10^{-27}\ \text{kg} = 938.272\ \text{MeV}/c^2
$$

$$
\text{charge} = +e = +1.602176634 \times 10^{-19}\ \text{C}
$$

The proton is stable with no experimentally observed decay. Within nuclear physics, the proton
number $Z$ defines the chemical element.

---

## Neutron

The neutron is a **spin-$\frac{1}{2}$ baryon** composed of one up quark and two down quarks
(udd). It is electrically neutral and, together with the proton, forms the nucleus.

$$
m_n = 1.67493 \times 10^{-27}\ \text{kg} = 939.565\ \text{MeV}/c^2
$$

$$
\text{charge} = 0
$$

The **free neutron** is unstable and undergoes $\beta^-$ decay:

$$
n \;\longrightarrow\; p + e^- + \bar{\nu}_e
$$

with a mean lifetime of $\tau_n \approx 879\ \text{s}$, corresponding to a half-life of
$t_{1/2} \approx 609\ \text{s}$. Neutrons bound inside a stable nucleus do not decay because
the decay is energetically forbidden by the nuclear binding energy.

The neutron number $N = A - Z$ (where $A$ is the mass number) determines the nuclear isotope.

---

## Electron

The electron is a **fundamental spin-$\frac{1}{2}$ lepton** and the carrier of negative charge.
It is not a nuclear constituent but governs the atomic and chemical properties of matter.

$$
m_e = 9.10938 \times 10^{-31}\ \text{kg} = 0.511\ \text{MeV}/c^2
$$

$$
\text{charge} = -e = -1.602176634 \times 10^{-19}\ \text{C}
$$

The electron is believed to be stable. It appears in nuclear physics as a product of
$\beta^-$ decay and in atomic physics as the orbital constituent of atoms.

---

## Nuclear Binding and the Mass Excess

The rest mass of a nucleus is **less** than the sum of the masses of its constituent nucleons.
This mass deficit is called the **mass excess** (or mass defect) $\Delta m$:

$$
\Delta m = Z\,m_p + N\,m_n - m_{\text{nucleus}}
$$

The corresponding energy — the **binding energy** — is released when free nucleons combine to
form a nucleus:

$$
B = \Delta m\, c^2
$$

A large binding energy per nucleon ($B/A$) indicates a more tightly bound, more stable nucleus.
The maximum of $B/A$ occurs near $A \approx 56$ (iron-group nuclei).

---

## Phase-Space Coordinates

Each particle carries six phase-space coordinates representing its position and direction of
travel, used in transport and cascade simulations:

| Coordinate | Description |
|-----------|-------------|
| $x, y, z$ | Cartesian position (cm or m, depending on context) |
| $u, v, w$ | Direction cosines satisfying $u^2 + v^2 + w^2 = 1$ |

Direction cosines are defined with respect to the $x$-, $y$-, and $z$-axes respectively:

$$
u = \cos\theta_x, \quad v = \cos\theta_y, \quad w = \cos\theta_z
$$

---

## References

- K. S. Krane, *Introductory Nuclear Physics*, Wiley (1988), Chapter 1.
- Particle Data Group, *Review of Particle Physics*, Prog. Theor. Exp. Phys. (2022), https://pdg.lbl.gov.
- CODATA 2018 recommended values: https://physics.nist.gov/cuu/Constants/
