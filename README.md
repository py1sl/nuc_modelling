# Nuclear Modelling

Codes and tools around nuclear modelling, ranging from simple models of hydrogen through to a range of inter and intra cascade models.  
Based on papers, codes, books and educational resources

---

## Installation

Install the project and its development tools from the repository root:

```bash
python -m pip install -e ".[dev]"
```

The package uses a `src` layout and exposes modules under the `nuc_modelling`
namespace.

## Module overview

| Package | Modules | Purpose |
|---------|---------|---------|
| `nuc_modelling.basic_nuclear` | `nuclear_constants`, `basic_nuclear_structure`, `nuclear_interaction` | Physical constants, nuclear structure, potentials, cross sections, and target geometry |
| `nuc_modelling.spallation` | `bertini_cascade` | Intra-nuclear cascade (INC) model: Fermi momentum, mean-free-path, cascade stepping |
| `nuc_modelling.utilities` | `kinematics`, `particle`, `particles_beams`, `radiation` | Shared particle, beam, kinematics, and radiation utilities |

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

Modules can be imported from their package paths, for example:

```python
from nuc_modelling.basic_nuclear.basic_nuclear_structure import binding_energy
from nuc_modelling.spallation.bertini_cascade import BertiniCascade
from nuc_modelling.utilities.kinematics import neutron_recoil_energy
```

### Units convention

All physics modules use **MeV** for energies, **fm** for distances, and **mb** (millibarn) for cross sections  
unless otherwise stated in the function docstring.
