# JAX backend

`zoomy_jax` runs the same finite-volume solvers on JAX — JIT compilation,
vectorisation, autodiff, and GPU/TPU acceleration — against the **same** `Model`
/ `SystemModel` / `NumericalSystemModel` contract as the
[NumPy reference](numpy.md).

## How it is built — subclass the NumPy solver

The JAX backend does **not** re-implement the solver; it *subclasses* the NumPy
one and overrides only the compute methods with JAX kernels
(`zoomy_jax/fvm/solver_jax.py:167`):

```python
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy

class HyperbolicSolver(HyperbolicSolverNumpy):
    """JAX HyperbolicSolver — JIT-compiled explicit time stepping.
    Inherits param definitions from the NumPy base class."""
```

So everything structural is reused from `zoomy_core`:

- the `param` knobs (`time_end`, `min_dt`, `compute_dt`, reconstruction order /
  limiter),
- the `setup_simulation → step → run_simulation → solve` skeleton and
  `_coerce_to_nsm`,
- the `NumericalSystemModel` / `ReconstructionSpec` specs and the symbolic
  Riemann/`Numerics` objects,
- the `timestepping` CFL closures (deliberately written JIT-safe so both
  backends share them).

What JAX **reimplements** is only the execution layer: the runtime
(`JaxRuntime.from_nsm`), reconstruction (`reconstruction_jax.py`), the ODE
integrators (`zoomy_jax/fvm/ode.py`), and the time loop. The symbolic operators
are the same — `zoomy_jax` lowers them through its own printer instead of
[`NumpyRuntimeModel`](../authoring/numerics.md).

## JAX mechanics

- Per-cell / per-face operators are `@jax.jit` + `jax.vmap` over the cell or face
  axis (the BC kernel is vmapped over boundary faces, the Riemann flux over the
  face axis).
- `step(self, dt, time, Q, Qaux)` re-applies the boundary conditions at each
  RK stage.
- `run_simulation` is a single `@jax.jit` `jax.lax.while_loop` over the time
  step, with `io_callback` for logging and snapshots — the whole march compiles
  into one XLA program. (It uses `while_loop`, not `lax.scan`.)

## Solver variants

| Class | File | What it adds |
| --- | --- | --- |
| `HyperbolicSolver` | `fvm/solver_jax.py:167` | JIT explicit RK1/SSP-RK2 + free-surface-aware reconstruction (`reconstruction_variables = conservative / eta / xz` — Audusse-Bouchut, Kurganov-Petrova, Xing-Zhang well-balanced recipes). |
| `PoissonSolver` | `fvm/solver_jax.py:949` | JIT elliptic residual solve. |
| `IMEXSourceSolverJax` | `fvm/solver_imex_jax.py:100` | Full IMEX compiled into one XLA program; CN implicit diffusion; local `fori_loop` Newton / global `while_loop` Newton-GMRES; `jv_backend="ad"` autodiff Jacobian-vector products. |
| `ChorinSplitVAMSolverJax` | `fvm/solver_chorin_vam_jax.py:61` | JAX port of the Chorin VAM split (same 3-sub-system constructor as NumPy). |

The `jv_backend="ad"` option is the JAX-specific win: where NumPy forms the
source Jacobian-vector product analytically or by finite difference, JAX takes it
by automatic differentiation.

```{note}
`zoomy_jax` also ships an optional preCICE integration
(`PreciceHyperbolicSolver`, `fvm/precice_solver.py`). That is a separate coupled
path; the OpenFOAM + preCICE coupling is documented on
[its own page](openfoam.md). This page covers the standalone JAX solver.
```

## How a user runs it

The call site mirrors NumPy — same `solve(mesh, model_or_nsm, …)` signature and
the same specs; you swap the solver import and pass a pre-built
`NumericalSystemModel`. From `zoomy_jax/tests/test_swe_dambreak_jax.py`:

```python
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel, ReconstructionSpec)
from zoomy_core.fvm import timestepping
from zoomy_jax.fvm.solver_jax import HyperbolicSolver   # JAX entry point

mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=N)
nsm  = NumericalSystemModel.from_system_model(
    sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod"))

solver  = HyperbolicSolver(time_end=t_end, compute_dt=timestepping.adaptive(CFL=0.9))
Q, Qaux = solver.solve(mesh, nsm, write_output=False)
```

Because the model, the `SystemModel`, the specs, and the solver API are shared,
a model verified on NumPy runs on JAX without change — which is exactly how the
backends are cross-checked against each other: a result that diverges between
backends is a bug in one of them.

## Authoring checklist

- Verify the model on the [NumPy reference](numpy.md) first; JAX should
  reproduce it.
- Build the discretisation via `NumericalSystemModel.from_system_model(sm, …)` —
  the JAX solver carries no numerical knobs of its own.
- Choose `jv_backend` for the IMEX path (`"ad"` autodiff is the JAX default).
- Keep `compute_dt` a `timestepping` closure (JIT-safe); a Python callable that
  closes over NumPy arrays will not trace.

**Install**

```bash
pip install zoomy_core zoomy_jax
```

**Repository**: [`library/zoomy_jax`](https://github.com/ZoomyLab/zoomy-jax)
**API reference**: see [zoomy_jax](../api/_apidoc_zoomy_jax/zoomy_jax.rst).
