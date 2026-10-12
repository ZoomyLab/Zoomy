# Model

A `Model` is the symbolic PDE: state, parameters, closures and boundary
conditions, derived from a vertical ansatz.  It has no mesh and no numerics.

**What you write:** a model family with its options.

```python
from zoomy_core.model.models import SME
model = SME(level=0, dimension=3, closures=[...], boundary_conditions=...)
```

| choice | options (default) |
|---|---|
| family (`zoomy_core.model.models`) | `SME` (shallow moments; `level=0` is shallow water), `VAM` (non-hydrostatic moments, pressure modes `P_i`), `MLSWE` / `MLSME` / `MLVAM` (multilayer), `KESME`, `QRKESME`, `ElderSME` |
| `level` | number of velocity moments above the depth average (`SME`: 2) |
| `dimension` | spatial dimension including the vertical: 2 = (x, z), 3 = (x, y, z) |
| `closures` | stress pieces, one per stress component: `Newtonian`, `NavierSlip`, `StressFree`, `ManningFriction`, `RoughWall`, `Bingham`, `KEpsilonViscosity`, ... (`[]`: unclosed) |
| `boundary_conditions` | per mesh tag (below) |
| `small_slope` | `True`: the small-slope frame (the standard shallow model) |

Boundary conditions (`zoomy_core.model.boundary_conditions`), each on one mesh
tag, collected in `BoundaryConditions([...])` or passed as a list:

| class | meaning |
|---|---|
| `Wall` / `FromModel(definition="wall")` | reflecting (free-slip) wall; `FromModel` takes the model's own definition |
| `Dirichlet(tag, on=field, value=v)` | prescribed value of a field |
| `Flux(tag, on=field, gradient=g)` | prescribed normal gradient |
| `Extrapolation(tag)` | zero gradient (transmissive) |
| `InflowOutflow(tag, prescribe_fields={...})` | prescribe some fields, extrapolate the rest |
| `Periodic(tag, periodic_to_physical_tag=partner)` | periodic seam |
| `Lambda(tag, prescribe_fields={...})` | face values from a user function |
| `Coupled(tag)` | preCICE interface (see [Backends](backends.md)) |

**Connects to:** `SystemModel.from_model(model)` ([SystemModel](system-model.md)).
`model.describe()` shows what was built.  Models without a vertical ansatz
(the compressible Euler equations, `zoomy_core.systemmodel.euler.compressible_euler`)
are built directly as a SystemModel.

API: {doc}`zoomy_core.model <../api/_apidoc_zoomy_core/zoomy_core.model>`.
