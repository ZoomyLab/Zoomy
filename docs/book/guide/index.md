# User guide

A simulation is built in three symbolic layers and run by a solver template on
a backend:

```
Model ──────────▶ SystemModel ─────────▶ NumericalSystemModel ──▶ solver template ──▶ backend
PDE, closures,     operator slots          + reconstruction,        one sympy graph      JAX · foam · AMReX
BCs                (flux, source, ...)     Riemann solver, kernels  per time step        (printed, compiled)
```

| page | what you choose there |
|---|---|
| [Quick start](quickstart.md) | one case, run on NumPy and JAX |
| [Model](model.md) | model family, closures, boundary conditions |
| [SystemModel](system-model.md) | nothing, usually: it is derived; inspect it, set initial conditions |
| [NumericalSystemModel](numerical-system-model.md) | reconstruction order and limiter, Riemann solver |
| [Solvers and options](solvers.md) | hyperbolic or Chorin, CFL / `dt`, time order, MOOD, IMEX, pressure solver |
| [Backends](../backends/index.md) | where it runs: CPU, MPI, GPU; one page per backend |
| [Tutorials and use cases](../tutorials/index.md) | executable notebooks, filtered by tag |
| [GUI](gui.md) | compose and run a case in the browser |

The NSM is backend-neutral: the same object is handed to every backend, and the
[test matrix](../developer/tests.md) checks that the backends agree.  Install first: [Install](../installation.md).
