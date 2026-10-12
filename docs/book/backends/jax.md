# JAX backend

```{warning}
Legacy: written for the previous solvers (`zoomy_core.fvm`, `zoomy_jax.fvm`), not the symbolic solvers; see the [user guide](../guide/index.md).
```

`zoomy_jax` runs the same finite-volume solvers on JAX — JIT compilation,
vectorisation, autodiff, and GPU/TPU acceleration — against the **same** `Model`
/ `SystemModel` / `NumericalSystemModel` contract as the
[NumPy reference](numpy.md).

## How it is built — subclass the NumPy solver

The JAX backend does **not** re-implement the solver; it *subclasses* the NumPy
one and overrides only the compute methods with JAX kernels
(`zoomy_jax/fvm/solver_jax.py:331`):

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
- `run_simulation` is a single `@jax.jit` while loop over the time step
  (`jax.lax.while_loop`, or the checkpointed equinox loop for
  `ad_mode="reverse"`, see [below](#automatic-differentiation)), with
  `io_callback` for snapshots — the whole march compiles into one XLA program.
  (It uses `while_loop`, not `lax.scan`.)

## Multi-device (SPMD)

The same `HyperbolicSolver` scales across devices (multi-GPU/TPU or multi-host)
with **no MPI** — just `jax.shard_map` + `jax.lax.ppermute` halo exchange. The
domain is partitioned along the cell axis
(`zoomy_jax/mesh/partition_jax.py`: `partition_1d_contiguous`,
`partition_xaxis_structured`); each device owns a contiguous strip plus halo
cells, and the flux / reconstruction operators compose with `shard_map`
**unchanged** — you refill the halo (one `ppermute`) before each stage and call
the ordinary flux operator. The sharded march is **bit-identical** to a
replicated single-device run, so a model verified on one device is correct on
many.

```python
spmd_mesh = Mesh(np.array(jax.devices()), axis_names=("cells",))

@partial(shard_map, mesh=spmd_mesh,
         in_specs=(P(None, "cells"), P(None, "cells")),
         out_specs=P(None, "cells"), check_rep=False)
def run(Q_pad, Qaux_pad):
    Q_pad = halo_exchange(Q_pad)            # ppermute along "cells"
    dQ    = flux_op(dt, t, Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad))
    return Q_pad + dt * dQ
```

Bit-identity is regression-guarded in `tests/unit/zoomy_jax/test_spmd_sme0.py`
(SME(0), 4 devices, orders 1–2, live parameter passing through the shard_map
call); the full recipe + halo-exchange helpers live in
[`zoomy_jax/README.md`](https://github.com/ZoomyLab/zoomy-jax) and
`fvm/halo_exchange_jax.py`. Simulate N devices on one CPU with
`XLA_FLAGS="--xla_force_host_platform_device_count=N"`.

## Solver variants

| Class | File | What it adds |
| --- | --- | --- |
| `HyperbolicSolver` | `fvm/solver_jax.py:333` | JIT explicit RK1/SSP-RK2 + free-surface-aware reconstruction (`reconstruction_variables = conservative / eta / xz` — Audusse-Bouchut, Kurganov-Petrova, Xing-Zhang well-balanced recipes). |
| `PoissonSolver` | `fvm/solver_jax.py:2279` | JIT elliptic residual solve. |
| `IMEXSourceSolverJax` | `fvm/solver_imex_jax.py:31` | Base `HyperbolicSolver` time loop (adaptive CFL, output, `ad_mode`) with two implicit stages appended to the explicit flux step: Crank-Nicolson diffusion of `diffusion_matrix`, then implicit Euler on the source (`source_mode="local"` cell-wise Newton / `"global"` Newton-GMRES). Both are Newton roots under `jax.lax.custom_root` (`fvm/implicit_ad.py`), so the step is differentiable forward and reverse; `jv_backend="ad"/"fd"/"analytic"` picks the Newton Jacobian action. |
| `ChorinSplitVAMSolverJax` | `fvm/solver_chorin_vam_jax.py:105` | JAX port of the Chorin VAM split (same 3-sub-system constructor as NumPy) on the base time loop: `step` is one hyperbolic Euler, pressure (matrix-free GMRES), corrector cycle (SSP-RK2 over it for `time_order=2`), so `run_simulation`, `run_observed` and `ad_mode` work as for the base solver. See [below](#chorin-split-vam). |

The `jv_backend="ad"` option is the JAX-specific win: where NumPy forms the
source Jacobian-vector product analytically or by finite difference, JAX takes it
by automatic differentiation. Derivatives of the solution itself always use the
exact AD linearization of the converged root (implicit function theorem),
whatever `jv_backend` the Newton loop uses.

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

## Automatic differentiation

The explicit `HyperbolicSolver` march is differentiable end to end — through the
adaptive time step, w.r.t. the physical parameters and the initial state. The
solver parameter `ad_mode` picks the loop primitive before tracing:

| `ad_mode` | Loop | Use | Memory |
| --- | --- | --- | --- |
| `"forward"` (default) | `jax.lax.while_loop` | `jax.jvp`, `jax.jacfwd` | tangent carried, nothing extra |
| `"reverse"` | `equinox.internal.while_loop(kind="checkpointed")` | `jax.grad`, `jax.vjp` | `ad_checkpoints` (default 64) copies of the loop state; steps between them are recomputed on the backward pass |

Each mode supports only its own direction (`grad` through the forward loop, or
`jvp` through the reverse loop, raises). Reverse mode needs equinox:
`pip install -e "library/zoomy_jax[ad]"`.

```python
import jax, jax.numpy as jnp
from zoomy_core.fvm import timestepping as ts
from zoomy_jax.fvm.solver_jax import HyperbolicSolver

solver = HyperbolicSolver(time_end=T, compute_dt=ts.adaptive(CFL=0.9, dimension=2),
                          ad_mode="reverse", ad_checkpoints=64)
Q0, _ = solver.setup_simulation(mesh, nsm)
p0 = solver.parameters                        # runtime parameter vector

def J(p, Q0):
    Q, _ = solver.run_simulation(Q0, write_output=False, parameters=p)
    return jnp.max(Q[ih])

J_val, (dJ_dp, dJ_dQ0) = jax.value_and_grad(J, argnums=(0, 1))(p0, Q0)
```

`run_simulation(Q0)` without `Qaux` rebuilds the aux state from `Q0` inside the
differentiated path, so `dJ_dQ0` includes `dQaux0/dQ0`.

`run_simulation(..., time0=t0)` (and `run_observed`) starts the march at `t0`;
`time_end` stays absolute. A chunked driver continues a run with the previous
chunk's end state, its `Qaux`, and `time0` = the previous `time_end`. `time0` is
traced, not part of the compiled-loop cache key; snapshots are spread evenly
over `[time0, time_end]`.

The solver keeps one compiled loop per configuration (`ad_mode`,
`ad_checkpoints`, `time_end`, output settings, observer). Switching `ad_mode`
back and forth on one solver compiles each loop once and keeps both: on the
40-cell bump of `tests/runtime/test_ad.py` a second reverse `grad` after a
forward `jvp` takes 0.28 s instead of 4.2 s. `setup_simulation` empties the
cache.

### Quantities of interest over the whole run

`run_observed(Q0, observer, Qaux=None, parameters=None, write_output=False, time0=0.0)`
runs the same march and returns `(Q, Qaux, acc)`. An observer
(`zoomy_jax.fvm.observers.Observer`) is two pure functions:
`init(Q, Qaux, parameters) -> acc` on the initial state, and
`update(acc, time, dt, Q, Qaux, parameters) -> acc` after every step, on the
post-step state at the new `time`. `acc` is any pytree of fixed shapes. It is
carried in the loop state, so it is differentiable in both modes, including the
checkpointed reverse loop. The final `Q, Qaux` are bitwise those of
`run_simulation`.

Two observers ship ready-made:

- `running_max(row)`: maximum of a state row over all cells and all time
  levels; `acc` is a scalar.
- `gauge_series(row, cells, times)`: a state row at `cells`, sampled at
  `times`. The value at `t_k` interpolates linearly in time between the two
  levels of the step that crosses it, so it moves continuously with the
  adaptive time grid. `acc.values` has shape `(len(times), len(cells))`; a time
  the run did not reach stays NaN.

```python
from zoomy_jax.fvm.observers import running_max, gauge_series

hmax = running_max(ih)                         # build ONCE, reuse
def J(p):
    return solver.run_observed(Q0, hmax, parameters=p)[2]
dJ_dp = jax.grad(J)(p0)                        # ad_mode="reverse"

gauges = gauge_series(iq, cells=[5, 20, 33], times=np.linspace(0.0, T, 7))
Q, Qaux, acc = solver.run_observed(Q0, gauges)
series = acc.values                            # (7, 3)
```

The observer is part of the cache key, compared by the identity of its two
functions. Reusing the same observer object reuses the compiled loop; every
`running_max(...)` call builds a new one and compiles again.

- **Time grid.** With `timestepping.adaptive` the derivative includes
  `d(dt)/d(input)`. For a frozen time grid use `timestepping.constant(dt)`.
- **Fractional powers.** `d/dx x**r` is infinite at `x = 0` for `r < 1`; a
  velocity norm `sqrt(q_x**2 + q_y**2)` in a friction source gives `0*inf = NaN`
  at rest. Opt in to the NSM operation
  `extra_operations=[regularize_fractional_powers(1e-6)]`
  (`zoomy_core.systemmodel.operations`), which maps `b**r -> (b + eps**2)**r`.
  The model itself stays analytic.
- **Limiters.** `minmod` and `barth_jespersen` make the discrete map piecewise
  smooth. AD returns the exact derivative of the active branch; finite
  differences cross switching branches and can disagree badly with it. Forward
  and reverse can also differ where roundoff flips a branch at a tie. Use
  `venkatakrishnan` for sensitivity work: forward, reverse and finite
  differences agree.
- **Dry cells.** Initially dry cells give large gradient spikes (no NaN with
  order 2 + venkatakrishnan + MOOD); treat gradients at wet/dry fronts with
  care.
- **Unconverged Newton (IMEX).** The implicit stages differentiate the EXACT
  root by the implicit function theorem. If Newton stops at `implicit_maxiter`
  with `||R||_2 > implicit_tol`, the result is the last iterate and its
  derivative belongs to a different map. `newton_root` then logs a warning
  naming the residual norm and the tolerance (`newton_root: not converged
  ...`), also under `jit`, `jvp` and the reverse loop, where the recomputation
  can log it again. Raise `implicit_maxiter` (or lower the CFL) until it is
  quiet.

### Chorin split VAM

`ChorinSplitVAMSolverJax` binds its three sub-models at construction, so
`setup_simulation(mesh)` takes no model and returns the state `Q` only. The
split sub-models carry no initial condition: write the initial state into that
`Q`, then march it on the base loop. One step is `step(dt, time, Q, Qaux,
parameters=None)` (pure, returns `Q`; `coupled` is not supported), and
`post_step` rebuilds the predictor aux pool. `parameters` is the predictor
parameter vector; the pressure and corrector vectors are built from it plus
`dt` by name, so a derivative w.r.t. `g` reaches all three stages.

```python
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax

solver = ChorinSplitVAMSolverJax(SM_pred, SM_press, SM_corr, time_end=T,
                                 cfl=0.9, ad_mode="reverse")
Q0 = setup_ic(solver.setup_simulation(mesh))      # write the IC into Q

def J(p, Q0):
    Q, _ = solver.run_simulation(Q0, write_output=False, parameters=p)
    return jnp.sum(Q[ih] ** 2)

dJ_dp, dJ_dQ0 = jax.grad(J, argnums=(0, 1))(solver.parameters, Q0)
```

Time step: `cfl` set gives the Chorin local-CFL law
`dt = min_f cfl*2*r_f / (dim*(2*deg+1)*|lam|_f)` with `deg = order - 1`, the
mesh dimension and the NSM `dt_max` cap. It equals
`timestepping.adaptive(CFL=cfl, dimension=dim, degree=deg)` where every face has
a wave; a NaN wave speed propagates into `dt` instead of being dropped, and the
division runs on a safe denominator so the reverse cotangent of a wave-free
face is 0, not NaN. `cfl=None` uses `compute_dt` (e.g.
`timestepping.constant(dt)`).

`run_jit_steps(dt, n_steps, Q, Qaux_pred=None, ..., t_start=0.0, cfl=None,
dt_max=None, parameters=None)` stays as the forward-only fast path: exactly
`n_steps` of the same `step` + `post_step` in one cached `lax.scan` (one compile
per `(n_steps, adaptive)`), no `time_end` clamp; it refuses
`ad_mode="reverse"`. On the 30-cell bump with constant `dt` it reproduces
`run_simulation` bitwise (`tests/runtime/test_ad_chorin.py`), and the two cost
the same per step (warm, CPU: 1.85 ms constant `dt`, 2.4 ms adaptive). Pass
`Qaux` to `run_simulation`: `Qaux=None` rebuilds it eagerly before the loop,
about 45 ms per call there.

## Authoring checklist

- Verify the model on the [NumPy reference](numpy.md) first; JAX should
  reproduce it.
- Build the discretisation via `NumericalSystemModel.from_system_model(sm, …)` —
  the JAX solver carries no numerical knobs of its own.
- Choose `jv_backend` for the IMEX path (`"ad"` autodiff is the JAX default).
- Keep `compute_dt` a `timestepping` closure (JIT-safe); a Python callable that
  closes over NumPy arrays will not trace.

**Install** — `zoomy_jax` is not on PyPI; it is installed from the clone or
used through the `zoomy_jax` container (see [Installation](../installation.md)):

```bash
pip install zoomy_core
pip install -e library/zoomy_jax
```

**Repository**: [`library/zoomy_jax`](https://github.com/ZoomyLab/zoomy-jax)
**API reference**: see [zoomy_jax](../api/_apidoc_zoomy_jax/zoomy_jax.rst).
