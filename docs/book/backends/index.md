# Solver Backends

Zoomy's symbolic modeling layer compiles to several numerical backends. Each
backend lives in its own sub-repository (`library/zoomy_<name>`) and is
installable on its own.

| Backend | Sub-repository | Page |
|---|---|---|
| NumPy (reference) | `library/zoomy_core` | [NumPy](numpy.md) |
| JAX | `library/zoomy_jax` | [JAX](jax.md) |
| Firedrake | `library/zoomy_firedrake` | [Firedrake](firedrake.md) |
| FEniCSx | `library/zoomy_fenicsx` | [FEniCSx](fenicsx.md) |
| PETSc DMPlex | `library/zoomy_dmplex` | [DMPlex](dmplex.md) |
| AMReX | `library/zoomy_amrex` | [AMReX](amrex.md) |
| OpenFOAM (via PreCICE) | `library/zoomy_foam` | [OpenFOAM](openfoam.md) |

The backends share the same `Model` / `SystemModel` API, so the same symbolic
model definition is used to drive any of them.
