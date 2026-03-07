# Basic Nuclear Structure

This document describes the physics and equations implemented in `basic_nuclear_structure.py`.

---

## Semi-Empirical Mass Formula (SEMF)

The Semi-Empirical Mass Formula, also known as the **Bethe–Weizsäcker formula**, estimates the
binding energy of an atomic nucleus using a liquid-drop analogy. It treats the nucleus as an
incompressible fluid of nucleons.

For a nucleus with mass number $A = N + Z$ (where $N$ is the number of neutrons and $Z$ the
number of protons), the total binding energy $B(N, Z)$ is:

$$
B(N, Z) = a_V A \;-\; a_S A^{2/3} \;-\; a_C \frac{Z(Z-1)}{A^{1/3}} \;-\; a_A \frac{(A - 2Z)^2}{A} \;+\; \delta(N, Z)
$$

The **binding energy per nucleon** is then:

$$
\frac{B(N, Z)}{A}
$$

### Terms

| Term | Expression | Coefficient | Physical origin |
|------|-----------|-------------|-----------------|
| Volume | $a_V A$ | $a_V = 15.8\ \text{MeV}$ | Each nucleon interacts with its nearest neighbours; binding grows with volume |
| Surface | $-a_S A^{2/3}$ | $a_S = 18.3\ \text{MeV}$ | Nucleons on the surface have fewer neighbours; correction proportional to surface area $\propto R^2 \propto A^{2/3}$ |
| Coulomb | $-a_C \dfrac{Z(Z-1)}{A^{1/3}}$ | $a_C = 0.714\ \text{MeV}$ | Electrostatic repulsion between all pairs of protons; $Z(Z-1)/2$ pairs spread over a sphere of radius $R \propto A^{1/3}$ |
| Asymmetry | $-a_A \dfrac{(A-2Z)^2}{A}$ | $a_A = 23.2\ \text{MeV}$ | Quantum (Pauli exclusion) penalty for unequal numbers of protons and neutrons |
| Pairing | $\delta(N,Z)$ | $a_P = 12.0\ \text{MeV}$ | Nucleons pair with opposite spins; even–even nuclei are more tightly bound than odd–odd |

### Pairing term $\delta(N, Z)$

$$
\delta(N, Z) =
\begin{cases}
+\dfrac{a_P}{\sqrt{A}} & \text{if both } N \text{ and } Z \text{ are even (even–even)} \\[6pt]
-\dfrac{a_P}{\sqrt{A}} & \text{if both } N \text{ and } Z \text{ are odd (odd–odd)} \\[6pt]
0 & \text{otherwise (even–odd or odd–even)}
\end{cases}
$$

---

## Total Binding Energy

The total binding energy of a nucleus is simply the SEMF value per nucleon multiplied by the
mass number:

$$
B_{\text{total}}(N, Z) = \frac{B(N, Z)}{A} \times A = B(N, Z)
$$

In practice the module computes the per-nucleon SEMF first and recovers the total:

$$
B_{\text{total}} = \text{SEMF}(N, Z) \times A
$$

---

## Neutron Separation Energy

The **neutron separation energy** $S_n$ is the minimum energy required to remove one neutron
from a nucleus $(N, Z)$, leaving the daughter nucleus $(N-1, Z)$:

$$
S_n(N, Z) = B(N, Z) - B(N-1, Z)
$$

A large $S_n$ indicates that the nucleus is particularly stable against neutron removal (e.g.
closed neutron shells produce characteristic jumps in $S_n$).

---

## Proton Separation Energy

The **proton separation energy** $S_p$ is the minimum energy required to remove one proton from
a nucleus $(N, Z)$, leaving the daughter nucleus $(N, Z-1)$:

$$
S_p(N, Z) = B(N, Z) - B(N, Z-1)
$$

Analogously to $S_n$, closed proton shells produce discontinuities in $S_p$ across isotopes.

---

## References

- C. F. von Weizsäcker, *Zur Theorie der Kernmassen*, Z. Phys. **96**, 431 (1935).
- H. A. Bethe and R. F. Bacher, *Nuclear Physics A: Stationary States of Nuclei*, Rev. Mod. Phys. **8**, 82 (1936).
- K. S. Krane, *Introductory Nuclear Physics*, Wiley (1988), Chapter 3.
