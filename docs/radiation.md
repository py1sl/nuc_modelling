# Radiation Physics

This document describes the physics and equations implemented in `radiation.py`.

---

## Radioactive Decay Law

Radioactive decay is a stochastic process: each unstable nucleus has a constant probability per
unit time of decaying, characterised by the **decay constant** $\lambda$.

The number of undecayed nuclei $N(t)$ at time $t$ follows an exponential law:

$$
N(t) = N_0\,e^{-\lambda t}
$$

where $N_0$ is the number of nuclei at $t = 0$.

### Decay Constant and Half-Life

The **half-life** $t_{1/2}$ is the time after which exactly half of the original nuclei have
decayed:

$$
N(t_{1/2}) = \frac{N_0}{2} \implies e^{-\lambda t_{1/2}} = \frac{1}{2}
$$

Solving gives:

$$
\boxed{t_{1/2} = \frac{\ln 2}{\lambda}}
$$

Equivalently, the decay constant in terms of half-life:

$$
\boxed{\lambda = \frac{\ln 2}{t_{1/2}}}
$$

---

## Radioactive Activity

The **activity** $A$ of a sample is the number of decays per unit time:

$$
A = -\frac{dN}{dt} = \lambda N
$$

The SI unit of activity is the **becquerel** (Bq), where $1\ \text{Bq} = 1\ \text{decay/s}$.
The older unit is the **curie** (Ci), where $1\ \text{Ci} = 3.7 \times 10^{10}\ \text{Bq}$.

### Activity as a Function of Time

Because $N(t) = N_0 e^{-\lambda t}$, the activity also decays exponentially:

$$
\boxed{A(t) = A_0\,e^{-\lambda t}}
$$

where $A_0 = \lambda N_0$ is the activity at $t = 0$.

---

## Radiation Flux from Source Geometries

The **radiation flux** $\Phi$ (fluence rate) is the number of particles crossing a unit area per
unit time. For an isotropically emitting source, the geometry of the source determines how the
flux falls off with distance.

### Point Source — Inverse Square Law

For a point source of activity $A$ emitting radiation isotropically, the flux at a distance $r$
is distributed uniformly over the surface of a sphere of area $4\pi r^2$:

$$
\boxed{\Phi = \frac{A}{4\pi r^2}}
$$

The flux falls off as $1/r^2$ (the **inverse square law**).

### Infinite Line Source — Inverse Distance Law

For an infinitely long uniform line source with activity per unit length $\lambda_L$, the flux
at a perpendicular distance $r$ from the line is:

$$
\boxed{\Phi = \frac{\lambda_L}{2\pi r}}
$$

The flux falls off as $1/r$ because radiation spreads over cylindrical shells of area $2\pi r
\ell$ (where $\ell$ is the length element), rather than spherical shells.

### Semi-Infinite Plane Source — Distance-Independent Flux

For a uniform plane source (infinite in two dimensions) with activity per unit area $\sigma$,
half the emitted radiation escapes into the half-space above the plane. The flux at any height
above the plane (assuming no attenuation) is:

$$
\boxed{\Phi = \frac{\sigma}{2}}
$$

The flux is **independent of distance** because the solid angle subtended by an infinite plane
as seen from any point above it is always $2\pi$ steradians (a full hemisphere), regardless of
height.

---

## Summary of Flux Geometries

| Source geometry | Flux $\Phi$ | Distance dependence |
|-----------------|-------------|---------------------|
| Point source | $A / (4\pi r^2)$ | $\propto r^{-2}$ |
| Infinite line source | $\lambda_L / (2\pi r)$ | $\propto r^{-1}$ |
| Infinite plane source | $\sigma / 2$ | independent of $r$ |

---

## References

- K. S. Krane, *Introductory Nuclear Physics*, Wiley (1988), Chapter 6.
- G. F. Knoll, *Radiation Detection and Measurement*, 4th ed., Wiley (2010), Chapter 2.
- IAEA, *Radiation Protection and Safety of Radiation Sources: International Basic Safety Standards*, Safety Standards Series No. GSR Part 3 (2014).
