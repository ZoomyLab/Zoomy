# Architecture

```
Model ──────────▶ SystemModel ─────────▶ NumericalSystemModel ──▶ solver template ──▶ backend
PDE, closures,     operator slots          + reconstruction,        one sympy graph      NumPy · JAX · foam · AMReX
BCs                (flux, source, ...)     Riemann solver, kernels  per time step        (printed, compiled)
```

- [Model](model.md): model family, closures, boundary conditions
- [SystemModel](system-model.md): initial conditions, the derived operators
- [NumericalSystemModel](numerical-system-model.md): reconstruction order and limiter, Riemann solver
- [Solvers and options](solvers.md): hyperbolic or Chorin, CFL and `dt`, time order, MOOD, IMEX, restart
- [Backends](backends.md): CPU, GPU, MPI, AD, coupling, code printing
- [GUI](gui.md): compose and run a case in the browser
