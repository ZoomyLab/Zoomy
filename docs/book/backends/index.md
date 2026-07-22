# Solver Backends

Zoomy's symbolic layer compiles to several numerical backends. The same
`Model` → `SystemModel` → `NumericalSystemModel` definition drives all of them,
so switching backend is a change of solver import, not a change of model.

| Backend | Package | Container | Status | Page |
| --- | --- | --- | --- | --- |
| NumPy | `library/zoomy_core` | `zoomy_numpy` | reference implementation | [NumPy](numpy.md) |
| JAX | `library/zoomy_jax` | `zoomy_jax` | verification backend; CPU + GPU | [JAX](jax.md) |
| AMReX | `library/zoomy_amrex` | `zoomy_amrex` | block-structured AMR | [AMReX](amrex.md) |
| OpenFOAM | `library/zoomy_foam` | `zoomy_openfoam` | 3-D coupling via preCICE | [OpenFOAM](openfoam.md) |
| Firedrake | `library/zoomy_firedrake` | `zoomy_firedrake` | finite element | [Firedrake](firedrake.md) |

**NumPy is the reference.** Every other backend is ported from it, and a
disagreement with NumPy is treated as a bug in the port. **JAX is the
verification backend** for numerical checks.

Backends are deliberately simple loops. Well-balancing, positivity, truncation
and every other scheme decision lives at the symbolic level
([Numerical System Model](../authoring/numerics.md)) and is emitted through the
code printers — so no backend carries its own numerical constants, and they
cannot drift apart.

```{note}
**PETSc/DMPlex** and **FEniCSx** are early prototypes: the packages are
skeletons with no test coverage and only placeholder CI containers, so their
pages sit under *Work in progress*. Do not plan work against them yet.
```

## Which one should I use?

- **Starting out, or a 1-D/2-D case that fits in memory** → NumPy.
- **You need speed, GPUs, or gradients** → JAX.
- **Large domains needing adaptive refinement** → AMReX.
- **Coupling a depth-averaged model to a resolved 3-D free surface** →
  OpenFOAM via preCICE.

Installation for each is on the [Installation](../installation.md) page —
the container path is the supported one for the compiled backends.
