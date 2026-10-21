# Solvers and options

Every backend has two solver classes ([Backends](backends.md)).

| solver | for |
|---|---|
| `SymbolicHyperbolicSolver<Backend>` | balance laws: SWE, SME, Euler, diffusion |
| `SymbolicChorinSolver<Backend>` | pressure-split models (`VAM`): predictor, pressure solve, correction |

Settings are keyword arguments of the solver class, declared with `param`.
A choice is a member of an enum from `zoomy_core.symsolver.settings` (`Imex.ssp2`, `Projection.increment`).
A wrong value or a misspelt name raises an error that names the closest valid one.

## Options

| option | default | meaning |
|---|---|---|
| `time_end` | - | end time |
| `cfl` | `0.9` | CFL number of the adaptive step |
| `dt` | `None` | constant step, `None` is adaptive |
| `time_order` | `None` | SSP-RK order 1, 2 or 3, `None` is the reconstruction order (Chorin: 1) |
| `mood` | `True` | order 2: cells that fail the model's admissibility, such as $h \ge 0$, are redone at order 1 |
| `imex` | `False` | `Imex.lie` or `Imex.ssp2`: the implicit slots of the [SystemModel](system-model.md) are solved implicitly |
| `dt_write` | `None` | output interval, `None` is the final state only |
| `output`, `on_write` | `None` | HDF5 snapshots and a callback per snapshot (NumPy, JAX) |
| `functions` | `None` | implementations of the opaque functions the model calls (NumPy, JAX) |
| `couple`, `subcycling` | `None`, `False` | preCICE participant, see [Backends](backends.md) |

## Restart

| option | default | meaning |
|---|---|---|
| `restart_from` | `None` | a restart state written by the same backend |
| `dt_checkpoint` | `None` | interval of the restart states, `None` is one at the end |
| `checkpoint` | `None` | where they go, `None` is next to the output |

## Chorin

| option | default | meaning |
|---|---|---|
| `pressure_solve` | `PressureSolve.linear` | `PressureSolve.newton` for a constraint nonlinear in the pressure |
| `projection` | `Projection.standard` | `Projection.increment` solves for the increment over the lagged pressure |
| `pressure_tol` | - | relative tolerance of the pressure solve |
| `pressure_linear_solver` | `LinearSolver.auto` | NumPy, JAX: `pyamg`, `amgx` on a GPU, or matrix-free `gmres` |
| `linear_solver` | `LinearSolver.hypre` | OpenFOAM, AMReX: `hypre`, `gmres`, OpenFOAM also `schur_gamg` |

Tutorials: {doc}`End to end: 1-D dam break <../tutorials/ipynb/components/end_to_end>`, {doc}`Chorin projection: a VAM channel <../tutorials/ipynb/vam/chorin_channel>`, {doc}`Function hooks <../tutorials/ipynb/hooks/function_hook>`.

API: {doc}`zoomy_core.symsolver <../api/_autodoc2/zoomy_core/zoomy_core.symsolver>`,
{doc}`zoomy_jax.symsolver <../api/_autodoc2/zoomy_jax/zoomy_jax.symsolver>`.
