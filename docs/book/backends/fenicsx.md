# FEniCSx backend

```{note}
FEniCSx has its own driver, `DolfinxHyperbolicSolver` (`library/zoomy_fenicsx/dolfinx_solver.py`),
which does not print the symbolic solver templates of the [user guide](../guide/solvers.md); the
[test matrix](../developer/tests.md) does not run it.
```

`zoomy_fenicsx` drives [FEniCSx](https://fenicsproject.org/) — the next-gen
FEniCS, built around DOLFINx + UFL + Basix.

**Status**: experimental.

**Install**: requires a working FEniCSx installation, then

```bash
pip install zoomy_core
pip install -e library/zoomy_fenicsx
```

**Repository**: [`library/zoomy_fenicsx`](https://github.com/ZoomyLab/zoomy-fenicsx)
