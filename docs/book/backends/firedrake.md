# Firedrake backend

```{note}
Firedrake has its own driver, `FiredrakeHyperbolicSolver` (`zoomy_firedrake.firedrake_solver`), which
takes the `SystemModel` of the symbolic layer but does not print the symbolic solver templates of
the [user guide](../guide/solvers.md); the [test matrix](../testing/symsolver.md) does not run it.
```

`zoomy_firedrake` drives [Firedrake](https://www.firedrakeproject.org/) (a
Python FEM framework built on PETSc and FInAT) from the Zoomy symbolic model.

**Highlights**
- Generic DG solver: arbitrary polynomial degree, IP-DG diffusion.
- Free-surface variant with p-weighted DG limiter (Li et al. 2020).
- AMR-ready scaffolding.

**Install**: requires a working Firedrake installation (see
[firedrakeproject.org/install.html](https://www.firedrakeproject.org/install.html))
plus

```bash
pip install zoomy_core
pip install -e library/zoomy_firedrake
```

**Repository**: [`library/zoomy_firedrake`](https://github.com/ZoomyLab/zoomy-firedrake)  
**API reference**: see [firedrake](../api/firedrake.md).
