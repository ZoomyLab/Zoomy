# Solver backends

Zoomy's symbolic layer compiles to several numerical backends. The same
`Model` → `SystemModel` → `NumericalSystemModel` definition drives all of them,
and a **solver template** (`zoomy_core.symsolver`, see [Solvers and options](../guide/solvers.md))
is printed once per backend: as Python for NumPy and JAX, as C++ for OpenFOAM and AMReX.
Switching backend is a change of solver import, not a change of model.

| Backend | Package | Solver classes | Runs in | Page |
| --- | --- | --- | --- | --- |
| NumPy | `zoomy_core.symsolver.numpy_backend` | `SymbolicHyperbolicSolverNumpy`, `SymbolicChorinSolverNumpy` | the Python process, serial | [NumPy](numpy.md) |
| JAX | `zoomy_jax.symsolver.solver` | `SymbolicHyperbolicSolverJax`, `SymbolicChorinSolverJax` | the Python process; CPU, GPU, MPI | [JAX](jax.md) |
| AMReX | `zoomy_amrex.symsolver` | `SymbolicHyperbolicSolverAmrex`, `SymbolicChorinSolverAmrex` | apptainer image; block-structured AMR | [AMReX](amrex.md) |
| OpenFOAM | `zoomy_foam.symsolver` | `SymbolicHyperbolicSolverFoam`, `SymbolicChorinSolverFoam` | apptainer image; unstructured meshes, preCICE | [OpenFOAM](openfoam.md) |
| Firedrake | `zoomy_firedrake` | `FiredrakeHyperbolicSolver` | Firedrake install; finite element | [Firedrake](firedrake.md) |

A hyperbolic solver (`SymbolicHyperbolic...`) marches balance laws; a Chorin solver
(`SymbolicChorin...`) adds the pressure projection of a pressure-split model (`VAM`).
All of them take the settings of the table `Options` in [Solvers and options](../guide/solvers.md).
The [test matrix](../testing/symsolver.md) runs the same cases on every backend that prints the
templates (convergence orders, exact solutions, conservation, symmetry, parallel = serial,
GPU = CPU), so a disagreement between two backends is a bug in one of them.

Backends contain no numerical constants of their own. Well-balancing, positivity,
truncation and every other scheme decision lives at the symbolic level
([NumericalSystemModel](../guide/numerical-system-model.md)) and reaches each backend through the
printer, so they cannot drift apart.

```{note}
**PETSc/DMPlex** and **FEniCSx** are early prototypes: the packages are
skeletons with no test coverage and only placeholder CI containers, so their
pages sit under *Work in progress*. Do not plan work against them yet.
Firedrake has its own driver and does not print the symbolic templates.
```

## Which one should I use?

- **Starting out, or a 1-D/2-D case that fits in memory** → NumPy; nothing to install but `zoomy_core`.
- **You need speed, GPUs, MPI or gradients** → JAX.
- **Large domains needing adaptive refinement** → AMReX.
- **Unstructured meshes in OpenFOAM, or coupling a depth-averaged model to a resolved 3-D free
  surface** → OpenFOAM via preCICE.

The compiled backends run in containers; the container path is the supported one, see
[Installation](../installation.md). What differs between the backends in settings and in how
they are called is on [Backends](../guide/backends.md).
