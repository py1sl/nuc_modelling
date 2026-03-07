# Bertini Intra-Nuclear Cascade Model

This document describes the physics and equations implemented in `bertini_cascade.py`.

The Bertini Intra-Nuclear Cascade (INC) model simulates the interaction of a high-energy
projectile (here a proton) with a nucleus by treating the nucleus as a continuous nuclear medium
through which secondary particles travel in straight-line trajectories between successive
nucleon–nucleon (NN) collisions.

The implementation targets proton–tungsten ($p$+$^{184}$W) interactions and is valid for
incident proton kinetic energies in the range $100\ \text{MeV} \leq T_p \leq 3000\ \text{MeV}$.

**Reference:** H. W. Bertini, *Phys. Rev.* **131**, 1801 (1963).

---

## Nuclear Radius

The nuclear charge/matter radius is described by the empirical **$A^{1/3}$ scaling law**:

$$
R = r_0 \, A^{1/3}
$$

| Symbol | Value | Description |
|--------|-------|-------------|
| $r_0$ | $1.2\ \text{fm}$ | Nuclear radius parameter |
| $A$ | — | Mass number of the nucleus |
| $R$ | — | Nuclear radius (fm) |

This relation reflects the approximately constant density of nuclear matter: because the nuclear
volume $V \propto A$ and $V \propto R^3$, it follows that $R \propto A^{1/3}$.

---

## Average Nuclear Density

Assuming the nucleus is a uniformly filled sphere of radius $R$, the average nucleon number
density is:

$$
\rho = \frac{3A}{4\pi R^3} = \frac{3}{4\pi r_0^3}
$$

The density is therefore approximately independent of $A$ (nuclear matter saturation).

---

## Fermi Momentum

Inside the nucleus, nucleons occupy momentum states up to the **Fermi momentum** $p_F$. In the
free Fermi gas approximation, nucleons fill all momentum states up to $p_F$ with two spin states
per momentum cell. For equal numbers of protons and neutrons the Fermi momentum is:

$$
p_F = \hbar \left( \frac{3\pi^2 \rho}{2} \right)^{1/3}
$$

In practical units, using $\hbar c = 197.327\ \text{MeV·fm}$:

$$
p_F\ [\text{MeV/c}] = \hbar c \left( \frac{3\pi^2 \rho}{2} \right)^{1/3}
$$

where $\rho$ is the nucleon number density in fm$^{-3}$.

The Fermi momentum determines the **Pauli blocking** condition: a secondary nucleon produced in
a NN collision cannot be scattered into an already-occupied momentum state (i.e. the recoil
momentum must exceed $p_F$).

---

## Nucleon–Nucleon Cross Sections

The model uses an empirical parametrisation of the total nucleon–nucleon cross sections as a
function of the incident proton kinetic energy $T$ (in MeV):

**Proton–proton (pp):**
$$
\sigma_{pp}(T) = 40 + 10\,e^{-T/500}\ \text{mb}
$$

**Proton–neutron (pn):**
$$
\sigma_{pn}(T) = 40 + 30\,e^{-T/400}\ \text{mb}
$$

Both cross sections approach $\sim 40\ \text{mb}$ at high energies and rise at lower energies,
which is consistent with measured total NN cross sections.

---

## Effective Cross Section in Nuclear Matter

For a nucleus with proton fraction $f_Z = Z/A$, the effective cross section seen by the
incident proton is a weighted average of the pp and pn cross sections:

$$
\sigma_{\text{eff}} = f_Z\,\sigma_{pp} + (1 - f_Z)\,\sigma_{pn}
$$

Cross sections are converted from millibarns to fm² using $1\ \text{mb} = 0.1\ \text{fm}^2$.

---

## Mean Free Path

The **mean free path** $\lambda$ is the average distance a proton travels in nuclear matter
before undergoing a NN collision:

$$
\lambda = \frac{1}{\sigma_{\text{eff}}\,\rho}
$$

where $\sigma_{\text{eff}}$ is in fm² and $\rho$ is in fm$^{-3}$, giving $\lambda$ in fm.

---

## Average Number of Primary Collisions

The average number of primary NN collisions experienced by the incident proton along a central
(diameter) trajectory through the nucleus is estimated as:

$$
\langle n_{\text{coll}} \rangle = \frac{2R}{\lambda}
$$

where $2R$ is the nuclear diameter (an approximation to the path length through a uniform sphere
along a central trajectory) and $\lambda$ is the mean free path.

---

## Summary of Key Quantities for Tungsten (W-184)

| Quantity | Symbol | Typical value |
|----------|--------|---------------|
| Atomic number | $Z$ | 74 |
| Mass number | $A$ | 184 |
| Nuclear radius | $R$ | $\approx 6.8\ \text{fm}$ |
| Average density | $\rho$ | $\approx 0.14\ \text{fm}^{-3}$ |
| Fermi momentum | $p_F$ | $\approx 270\ \text{MeV/c}$ |

---

## References

- H. W. Bertini, *Low-Energy Intranuclear Cascade Calculation*, Phys. Rev. **131**, 1801 (1963).
- H. W. Bertini, *Intranuclear-Cascade Calculation of the Secondary Nucleon Spectra from Nucleon-Nucleus Interactions in the Energy Range 340 to 2900 MeV*, Phys. Rev. **188**, 1711 (1969).
- R. Serber, *Nuclear Reactions at High Energies*, Phys. Rev. **72**, 1114 (1947).
