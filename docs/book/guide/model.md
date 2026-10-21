# Model

A `Model` is the symbolic PDE: state, parameters, closures and boundary conditions, derived from a vertical ansatz.
It has no mesh and no numerics.

| choice | options |
|---|---|
| family (`zoomy_core.model.models`) | `SME` (shallow moments, `level=0` is shallow water), `VAM` (non-hydrostatic moments), `VAMWeakViscosity`, `MLSWE` / `MLSME` / `MLVAM` (multilayer), `KESME`, `QRKESME`, `ElderSME` |
| `level` | number of velocity moments above the depth average |
| `dimension` | spatial dimension including the vertical: 2 is (x, z), 3 is (x, y, z) |
| `closures` | stress closures: `Newtonian`, `NavierSlip`, `StressFree`, `ManningFriction`, `RoughWall`, `Bingham`, `KEpsilonViscosity`, ... |
| `boundary_conditions` | one per mesh tag, below |

## Boundary conditions

`zoomy_core.model.boundary_conditions`, each on one mesh tag, collected in `BoundaryConditions([...])`.

| class | meaning |
|---|---|
| `Wall(tag)` | reflecting free-slip wall |
| `Dirichlet(tag, on=field, value=v)` | prescribed value of a field |
| `Flux(tag, on=field, gradient=g)` | prescribed normal gradient |
| `Extrapolation(tag)` | zero gradient, transmissive |
| `InflowOutflow(tag, prescribe_fields={...})` | prescribe some fields, extrapolate the rest |
| `Periodic(tag, periodic_to_physical_tag=partner)` | periodic pair |
| `Lambda(tag, prescribe_fields={...})` | face values from a user function |
| `Coupled(tag)` | preCICE interface, see [Backends](backends.md) |

`SystemModel.from_model(model)` derives the [SystemModel](system-model.md), `model.describe()` shows the equations.
Tutorial: {doc}`Model and SystemModel <../tutorials/ipynb/components/model_system_model>`.

API: {doc}`zoomy_core.model <../api/_autodoc2/zoomy_core/zoomy_core.model>`.
