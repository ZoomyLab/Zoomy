# FEniCSx backend

```{warning}
Legacy: written for the previous solvers (`zoomy_core.fvm`, `zoomy_jax.fvm`), not the symbolic solvers; see the [user guide](../guide/index.md).
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
