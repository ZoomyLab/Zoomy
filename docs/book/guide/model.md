# Model

A `Model` is a symbolic PDE system: state, parameters, equations, boundary and initial conditions.
It has no mesh and no numerics.
`SystemModel.from_model(model)` lifts it to the [SystemModel](system-model.md).

## What a Model has

`zoomy_core.model.basemodel.Model`, configured by keyword arguments.

| setting | meaning |
|---|---|
| `dimension` | spatial dimension. The derived models (`SME`, `VAM`, multilayer) count the vertical: `dimension=2` is (x, z) |
| `variables` | state fields, a name or `(name, "nonnegative")` |
| `aux_variables` | auxiliary fields, for example derivatives a closure needs |
| `parameters` | `{name: default}`, overridden per instance by `parameters={...}` |
| `boundary_conditions` | one condition per mesh tag, below |
| `initial_conditions` | `zoomy_core.model.initial_conditions`: `Constant`, `RP`, `RP2d`, `RadialDambreak`, ... |
| `aux_boundary_conditions`, `aux_initial_conditions` | the same for the auxiliary fields |
| `eigenvalue_mode` | `"symbolic"` or `"numerical"` wave speeds |
| `equilibrium_reconstruction` | well-balanced reconstruction: `"none"`, `"audusse"`, `"bernoulli"`, `"projected_bernoulli"` |

The equations come in one of two forms.

- **Operators**: the model overrides `flux`, `nonconservative_matrix`, `source`, `diffusion_matrix`, ... (`SWE`).
- **Derived rows**: `derive_model` builds the equations symbolically from a full system and an ansatz (`SME`, `VAM`, the multilayer models). `model.derivation.describe()` prints them.

Closures live on the derived models: `closures=[...]` takes pieces of `zoomy_core.model.models.closures`.

| closes | closures |
|---|---|
| bulk stress | `Newtonian`, `Bingham`, `ElderViscosity`, `KEpsilonViscosity`, `QRViscosity` |
| bed | `NavierSlip`, `RoughWall`, `WallFunctionBed`, `ManningFriction` |
| free surface | `StressFree` |
| layer interfaces | `MeanInterface`, `UpwindInterface` |

## The smallest model: SWE

`SWE()` in `zoomy_core/model/models/swe.py`, written as operators.
Generated from the code by `SWE().summarize_model()`:

```{include} ../../_generated/model_swe.md
```

## Implemented models

`zoomy_core.model.models`:

| model | |
|---|---|
| `SWE` | shallow water with bed and Manning friction, written as operators |
| `MalpassetSWE` | viscous wetting and drying SWE in two horizontal directions |
| `SME` | shallow moment equations, `level` velocity moments above the mean. [Details](models/sme.md) |
| `ElderSME` | SME with the Elder eddy viscosity and a rough-wall bed |
| `KESME` | SME with transported k and epsilon |
| `QRKESME` | KESME with `k = sk^2` and `epsilon = se^2`, nonnegative by construction |
| `VAM` | non-hydrostatic vertically averaged moments, u, w and the non-hydrostatic pressure in one basis |
| `VAMWeakViscosity` | VAM in the weak-viscosity limit |
| `MLSWE` | multilayer SWE, layers moving with the free surface |
| `MLSME` | multilayer SME, a moment column per layer |
| `MLVAM` | multilayer VAM, a VAM column per layer. [Details](models/mlvam.md) |

The settings of a model are `param` parameters:

```python
from zoomy_core.model.models import SME

list(SME.param)                 # every setting
SME.param["stress"].objects     # allowed values: [Stress.shallow, Stress.full]
SME.param["closures"].doc       # documentation of one setting
```

An unknown setting raises a `TypeError` naming the closest match.
An option given as a plain string raises with the typed value to use (`stress="full"`: use `Stress.full`).

## Boundary conditions

`zoomy_core.model.boundary_conditions`, each on one mesh tag, collected in `BoundaryConditions([...])`.

| class | meaning |
|---|---|
| `Wall(tag)` | reflecting free-slip wall |
| `FromModel(tag, definition="wall")` | a boundary the model derives, for example the free-slip wall of every moment |
| `Dirichlet(tag, on=field, value=v)` | prescribed value of a field |
| `Flux(tag, on=field, gradient=g)` | prescribed normal gradient |
| `Extrapolation(tag)` | zero gradient, transmissive |
| `InflowOutflow(tag, prescribe_fields={...})` | prescribe some fields, extrapolate the rest |
| `Periodic(tag, periodic_to_physical_tag=partner)` | periodic pair |
| `Lambda(tag, prescribe_fields={...})` | face values from a user function |
| `Coupled(tag)` | preCICE interface, see [Backends](backends.md) |

Tutorial: {doc}`Model and SystemModel <../tutorials/ipynb/components/model_system_model>`.

API: {doc}`zoomy_core.model <../api/_autodoc2/zoomy_core/zoomy_core.model>`.
