# Shallow moments

```{warning}
Legacy: written for the previous solvers (`zoomy_core.fvm`, `zoomy_jax.fvm`), not the symbolic solvers; see the [user guide](../guide/index.md).
```

Shallow moments in 1-D recover the vertical structure of a 2-D flow; in 2-D they
are essentially our 3-D model. The dimensionality is set by the mesh.

| | What it covers |
|---|---|
| [Moments in 2-D](ipynb/sme/moments_2d.ipynb) | Write your **own closure**, inject it into `SME`, run in **2-D**, and read the vertical velocity profile back out. |

Runs on the reference **NumPy** backend — `pip install zoomy_core`, no container.
