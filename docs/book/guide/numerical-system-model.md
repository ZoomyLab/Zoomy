# NumericalSystemModel

The `NumericalSystemModel` (NSM) is the SystemModel plus the scheme choices that are not a solver option.
Every backend prints its kernels (flux, fluctuations, source, boundary states) from the same NSM.
`NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2), riemann=...)` builds it.

| option | default | meaning |
|---|---|---|
| `reconstruction.order` | `1` | 1 is piecewise constant, 2 uses limited least-squares gradients |
| `reconstruction.limiter` | `Limiter.venkatakrishnan` | also `minmod`, `barth_jespersen`, `none` |
| `riemann` | from the model, else `NonconservativeRusanov` | a class of `zoomy_core.fvm.riemann_solvers`: `Rusanov`, `NonconservativeRusanov`, `HLL`, `HLLC`, ... |
| `diffusion` | on for a nonzero `diffusion_matrix` | `DiffusionSpec` |
| `spectral_radius_method` | `SpectralRadius.gelfand` | wave speed of a model without closed-form eigenvalues: an upper bound, or `SpectralRadius.eig` for numerical eigenvalues |

The mesh is an `LSQMesh` (`zoomy_core.mesh`), created with `LSQMesh.create_1d/2d/3d` or from a gmsh file.
Every solver takes the same NSM and mesh ([Backends](backends.md)).
Tutorials: {doc}`NumericalSystemModel <../tutorials/ipynb/components/numerical_system_model>`, {doc}`a 1-D mesh <../tutorials/ipynb/mesh/mesh_1d>`, {doc}`a 2-D gmsh mesh <../tutorials/ipynb/mesh/mesh_2d_gmsh>`.

API: {doc}`zoomy_core.numerics <../api/_autodoc2/zoomy_core/zoomy_core.numerics>`,
{doc}`zoomy_core.mesh <../api/_autodoc2/zoomy_core/zoomy_core.mesh>`.
