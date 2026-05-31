# Nuclear Modelling

Codes and tools around nuclear modelling, ranging from simple models of hydrogen through to a range of inter and intra cascade models.  
Based on papers, codes, books and educational resources

---

## Module overview

| Module | Layer | Purpose |
|--------|-------|---------|
| `nuclear_constants.py` | Constants | Physical constants in SI units |
| `basic_nuclear_structure.py` | Structure | Binding/separation energies, liquid-drop model, nuclear radii, magic numbers |
| `nuclear_interaction.py` | Interaction | Two-body potentials, cross-section utilities, target geometry, uncertainty propagation |
| `kinematics.py` | Transport | Relativistic kinematics, Lorentz boosts, threshold energies |
| `particles_beams.py` | Beams | Particle/beam definitions, beam-phase-space utilities |
| `bertini_cascade.py` | Cascade | Intra-nuclear cascade (INC) model: Fermi momentum, mean-free-path, cascade stepping |
| `radiation.py` | Radiation | Photon/charged-particle energy-loss and radiative-process utilities |

### When to use each layer

- **Structure** — nuclear ground-state properties (mass, radius, binding energy, shell structure).  
  Use before setting up a collision to obtain threshold energies and Q-values.
- **Interaction** — two-body physics: potential forms, differential/total cross sections, Coulomb barriers,  
  target geometry (Woods-Saxon density, mean-free-path, interaction probability), and uncertainty propagation.  
  Entry point for building analytic interaction-level estimates.
- **Transport / Kinematics** — relativistic event kinematics, frame transforms, and phase-space geometry.  
  Use to map beam settings to CM-frame quantities or to compute recoil observables.
- **Beams** — define projectile and beam distributions; compose with the Interaction layer to produce  
  beam-averaged observables.
- **Cascade** — Monte-Carlo INC stepping inside a nucleus; compare against analytic Interaction-layer  
  estimates on the same scenario for cross-checks.
- **Radiation** — energy-loss rates and radiative corrections for charged particles and photons inside matter.

### Units convention

All physics modules use **MeV** for energies, **fm** for distances, and **mb** (millibarn) for cross sections  
unless otherwise stated in the function docstring.
