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
| family (`zoomy_core.model.models`) | `SME` (shallow moments; `level=0` is shallow water), `VAM` (non-hydrostatic moments, pressure modes `P_i`), `VAMWeakViscosity` (`VAM` in the weak-viscosity limit, the thesis VAM), `MLSWE` / `MLSME` / `MLVAM` (multilayer), `KESME`, `QRKESME`, `ElderSME` |
| `level` | number of velocity moments above the depth average (`SME`: 2) |
| `dimension` | spatial dimension including the vertical: 2 = (x, z), 3 = (x, y, z) |
| `closures` | stress pieces, one per stress component: `Newtonian`, `NavierSlip`, `StressFree`, `ManningFriction`, `RoughWall`, `Bingham`, `KEpsilonViscosity`, ... (`[]`: unclosed) |
| `boundary_conditions` | per mesh tag (below) |
| `small_slope` | `True`: the small-slope frame (the standard shallow model) |
| `stress`, `scaling` | which stress terms the model keeps: truncated by order from the non-dimensional balance laws, `scaling=ShallowScaling(horizontal, vertical, regime=Regime(...))`; default the regime `Fr, De = O(1)` (`SME` / `MLSME` orders (0, 0), `VAM` (0, 2): its vertical row keeps `-nu d_x d_z u`; `VAMWeakViscosity`: `1/De = O(eps)`, orders (1, 2)); `stress=Stress.full` keeps the in-plane stress |

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

Every BC feeds three channels, the data structures its value goes to:

| channel | what | consumed by |
|---|---|---|
| `face` | the state on the boundary face | the numerical flux (Riemann problem) |
| `ghost` | the value at the mirror image of the inner cell centre; by default from the face value (`2 face - Q` for a face value) | gradients, limiters, derivative aux, diffusion |
| `normal_gradient` | a prescribed dQ/dn on the face (Neumann data; 0 unless set) | diffusion, the LSQ boundary closure |

Any BC can take any channel from another BC, on the fields of that BC's `on`:
`Wall(tag="w", ghost=Dirichlet(on="momentum", value=0))` (reflecting Riemann state, no-slip
stencils), `Wall(tag="w", normal_gradient=Flux(on="h", gradient=2.0))`.

A new BC subclasses `BoundaryCondition` and implements only what differs:
`compute_boundary_condition` (the face value; set `boundary_value_kind =
BoundaryValueKind.face` when it is a true face value), `compute_ghost_state` (only if the
stencils need something else), `compute_boundary_gradient` (Neumann data).  `Wall` is the
template: about 20 lines building the reflected state from `normal` and the momentum groups.
The kernels are symbolic, so every backend gets them.

Model parameters are `param` parameters, like the solver settings ([Solvers](solvers.md)): the
class declares them, the constructor's arguments set them, and a misspelt name gets the closest
one (`SME(levl=1)`: `TypeError: SME has no setting 'levl'; did you mean 'level'?`).

**Connects to:** `SystemModel.from_model(model)` ([SystemModel](system-model.md)).
`model.describe()` shows what was built.  Models without a vertical ansatz
(the compressible Euler equations, `zoomy_core.systemmodel.euler.compressible_euler`)
are built directly as a SystemModel.

API: {doc}`zoomy_core.model <../api/_autodoc2/zoomy_core/zoomy_core.model>`.
