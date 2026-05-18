# Firedrake backend

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
