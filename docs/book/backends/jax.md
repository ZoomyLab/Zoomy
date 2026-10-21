# JAX backend

`zoomy_jax` runs the solver templates on JAX: JIT compilation, vectorisation, autodiff, and
CPU / GPU / multi-rank execution, against the **same** `NumericalSystemModel` as the
[NumPy backend](numpy.md).

## How it is built: print, then compile

The JAX backend does not re-implement a solver.  `zoomy_jax.symsolver.solver` prints the
template (`HyperbolicTemplate`, `ChorinTemplate`) to Python source with `PyPrinter`, the same
source the NumPy backend prints, and runs it against `zoomy_jax.symsolver.runtime`: the
building blocks (gradients, Riemann problems on all faces, reductions, linear solves) written
as jitted, vectorised JAX functions.  The printed march is jitted.
Only the runtime differs between NumPy and JAX; the arithmetic is the template's.

| Class | Template | Call |
| --- | --- | --- |
| `SymbolicHyperbolicSolverJax` | `HyperbolicTemplate`: SSP-RK step, MOOD, IMEX | `setup(mesh, nsm)`, then `run(Q, Qaux)` |
| `SymbolicChorinSolverJax` | `ChorinTemplate`: predictor, pressure solve, correction (`VAM`) | `setup(mesh, pred, press, corr)`, then `run(Q, Qaux)` |

`setup` prints the template, compiles, and returns the initial `(Q, Qaux)`; `run` returns
`(time, Q, Qaux)`.  Both are JAX arrays of shape `(n_state, n_inner_cells)`.

## Running

`import zoomy_jax` switches JAX to double precision (`ZOOMY_JAX_ENABLE_X64=0` keeps float32).
One NSM on JAX and on NumPy: [Quick start: a 2-D bump on NumPy and JAX](../tutorials/ipynb/swe/bump_2d.ipynb).

## Settings

The settings shared by all backends are the tables of [Solvers and options](../guide/solvers.md).
JAX adds:

| option | default | meaning |
|---|---|---|
| `comm` | `None` | an mpi4py communicator: a distributed run, one rank per device; `setup` takes the global mesh and returns this rank's piece, `solver.partition.gather(Q)` collects the state |
| `ad_mode`, `ad_checkpoints` | `AdMode.forward`, `64` | `AdMode.reverse`: checkpointed loop for `jax.grad` |
| `fold_geometry` | `False` | bake the mesh into the compiled loop as constants (XLA folds them; compile time grows with the mesh) |
| `output`, `on_write` | `None` | HDF5 snapshots at `dt_write`; a callback `on_write(time, Q, Qaux)` per write (forward mode only) |
| `couple`, `subcycling` | `None`, `False` | preCICE participant (a dict `adapter=`, `tag=`, `side=`, optionally `face=`, `frozen_mass_row=`); one step per window or adaptive steps inside it |

A GPU run needs a CUDA `jaxlib` (`JAX_PLATFORMS=cuda`); a distributed one is started by
`mpiexec -n N python -m zoomy_jax.gpurank run.py`, which binds one GPU per rank before JAX
loads CUDA (see [Backends](../guide/backends.md)).

## Automatic differentiation

The march is differentiable end to end: through the time loop, with respect to the parameters
(`solver.parameters`, named in `solver.rt.parameter_names`) and the initial state.
`ad_mode` picks the loop primitive before tracing:

| `ad_mode` | Loop | Use | Memory |
| --- | --- | --- | --- |
| `AdMode.forward` (default) | `jax.lax.while_loop` | `jax.jvp`, `jax.jacfwd` | tangent carried, nothing extra |
| `AdMode.reverse` | `equinox.internal.while_loop(kind="checkpointed")` | `jax.grad`, `jax.vjp` | `ad_checkpoints` copies of the loop state; the steps between them are recomputed on the backward pass |

Each mode supports only its own direction: `jax.grad` through the forward loop raises
(`Reverse-mode differentiation does not work for lax.while_loop`), `jax.jvp` through the reverse
loop raises (`can't apply forward-mode autodiff (jvp) to a custom_vjp function`).  Reverse mode
needs equinox: `pip install "zoomy-jax[ad]"`.  The mode is read once, at `setup`:
`run` raises `ValueError` when `ad_mode` was changed afterwards (set it before `setup`), and
`setup` raises for `AdMode.reverse` with `on_write` or `output` (snapshots are forward mode only).

`run(Q, Qaux, parameters=p, time0=t0)` takes the parameter vector and the start time as traced
arguments; `initial_aux(Q, Qaux, p)` rebuilds the aux state of `Q` inside the differentiated
path, so a derivative with respect to the initial state includes `dQaux0/dQ0`.  The example
differentiates a scalar of the final state with respect to Manning's $n$, forward, reverse and by
central differences: [Forward-mode AD](../tutorials/ipynb/ad/forward_ad.ipynb) and
[Reverse-mode AD](../tutorials/ipynb/ad/reverse_ad.ipynb).

A constant `dt` freezes the
time grid; with `cfl` the derivative also contains $\partial\,\mathrm{d}t/\partial(\text{input})$.
Limiters that switch branches (`minmod`) make the discrete map piecewise smooth: AD returns the
derivative of the active branch and finite differences can cross branches, so use
`Limiter.venkatakrishnan` for sensitivity work.  The mesh vertices are differentiable too:
`solver.X` holds their coordinates, shape `(dim, n_vertices)`, and `initial_aux(..., X=X)` and
`run(..., X=X)` compute the geometry from `X` inside the differentiated path
(`docs/dev/symbolic_solver/gates/gate5_mesh_ad.py`).

## Chorin projection: `VAM`

`SymbolicChorinSolverJax` takes the predictor NSM and the pressure and corrector systems of the
model's pressure split (`predictor_nsm` builds the predictor's numerics); the pressure solve
options (`pressure_solve`, `projection`, `pressure_linear_solver`, `pressure_tol`) are in
[Solvers and options](../guide/solvers.md).  The VAM(1) channel (matrix case C8):
[Chorin projection: a VAM channel](../tutorials/ipynb/vam/chorin_channel.ipynb).  The step is
differentiable like the hyperbolic one (`ad_mode=AdMode.reverse`).

## preCICE

`couple=dict(adapter=, tag=, side=)` makes the march one preCICE participant (the adapter
imports the `precice` bindings and `petsc4py`).  `adapter` is a `PreciceAdapter`
(`zoomy_jax.symsolver.runtime`), the host side of the template's `couple.*` effects: the preCICE
API calls only.  `tag` names the coupled boundary and `side` is 0 if this domain is the low
side of the interface.  `face`, a member of `CoupledFace`
(`zoomy_core.symsolver.strategies.hyperbolic`), picks the interface answer:

- `CoupledFace.shared` (default): one Riemann problem on (own, peer) that both sides solve alike,
  for two participants of the same model;
- `CoupledFace.characteristic`: the Riemann problem (own, ghost) towards a peer of another model,
  the own cells' states and aux exchanged; `frozen_mass_row=True` holds the depth row of the
  coupled faces over the window at the value the peer charges.

`subcycling=False` takes one step per window and conserves mass exactly; `True` takes adaptive
steps inside the window.  Towards a VOF peer (`zoomy_foam.vof`),
`zoomy_jax.symsolver.coupling.column_participant` builds the participant with the
`ColumnPreciceAdapter` (the six canonical fields `[b, h, u, v, w, p]` on `nz` heights, through
`interpolate_to_3d` / `project_from_3d`).  The Chorin solver refuses `couple`.  The C++
participants take another dict, see the [OpenFOAM page](openfoam.md).

## Earlier JAX solvers

`zoomy_jax.fvm` still contains the earlier solvers (`HyperbolicSolver`, `IMEXSourceSolverJax`,
`ChorinSplitVAMSolverJax`, `PreciceHyperbolicSolver`).  The symbolic solvers superseded them;
`gate1_vs_solver_jax.py` and `gate7_chorin.py` in `docs/dev/symbolic_solver/gates/` compare the
two.  New work should not start there.

## Authoring checklist

- Verify the model on the [NumPy backend](numpy.md) first when something looks wrong; JAX
  prints the same template.
- Build the discretisation through `NumericalSystemModel.from_system_model(sm, ...)`: the JAX
  solver carries no numerical knobs of the scheme.
- Set `jax_enable_x64` (or import `zoomy_jax` first) before comparing with other backends.

**Install**: [Install](../installation.md), or the `zoomy_jax` [container](../containers.md).

**Repository**: [`library/zoomy_jax`](https://github.com/ZoomyLab/zoomy-jax)
**API reference**: {doc}`zoomy_jax.symsolver <../api/_autodoc2/zoomy_jax/zoomy_jax.symsolver>`.
