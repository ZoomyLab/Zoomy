# NumericalSystemModel

The `NumericalSystemModel` (NSM) is the SystemModel plus the scheme choices
that are not a solver option: what is reconstructed and how, the Riemann
solver, the wave speed.  It also carries the kernels (`nsm.call.<slot>`:
flux, fluctuations, source, boundary states, ...) that every backend prints
from the same expressions.

**What you write:**

```python
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
nsm = NumericalSystemModel.from_system_model(
    sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
```

| option | default | meaning |
|---|---|---|
| `reconstruction.order` | `1` | 1: piecewise constant; 2: LSQ gradients, limited |
| `reconstruction.limiter` | `Limiter.venkatakrishnan` | also `minmod`, `barth_jespersen`, `none`; per row: `limiters={"b": Limiter.none}` |
| `reconstruction.variables` | `ReconstructedVariables.model` | `model`: the model's reconstruction variables (SWE/SME: $\eta = b + h$, $u = q/h$); `conservative`: $Q$ |
| `reconstruction.bed` | `BedReconstruction.derived` | bed at the faces from $\eta - h$ (lake at rest, $h_f \ge 0$); `reconstructed` |
| `reconstruction.ties` | `{}` | `{row: lead}`: a row is limited at least as much as its lead |
| `riemann` | from the model's tags, else `NonconservativeRusanov` (C1 bump: `PositiveNonconservativeRusanov`) | `zoomy_core.fvm.riemann_solvers`: `Rusanov`, `NonconservativeRusanov`, `HLL`, `HLLC`, ... |
| `diffusion` | on if `diffusion_matrix` is nonzero | `DiffusionSpec` |
| `spectral_radius_method` | `zoomy_core.symsolver.WAVE_SPEED` = `"power"` | wave speed of a model without closed-form eigenvalues: power iteration on $A^2$, the same on every backend and device |

MOOD (the a-posteriori order fallback) is a solver option
([Solvers](solvers.md)); what it tests is the model's admissibility
(e.g. $h \ge 0$; Euler: $\rho, p > 0$).

**Connects to:** every solver takes the NSM unchanged
(`solver.setup(mesh, nsm)` on JAX, `solver.run(mesh, nsm, Q0, workdir)` on
foam / AMReX).  The mesh is an `LSQMesh` (`zoomy_core.mesh`:
`LSQMesh.create_1d/2d/3d`, or from a gmsh file); its least-squares stencils
serve gradients and boundary closures.

API: {doc}`zoomy_core.numerics <../api/_autodoc2/zoomy_core/zoomy_core.numerics>`,
{doc}`zoomy_core.mesh <../api/_autodoc2/zoomy_core/zoomy_core.mesh>`.
