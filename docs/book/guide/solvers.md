# Solvers and options

A solver is a **template**: a Python function that, run once on sympy symbols,
returns one sympy graph of a whole time step (`zoomy_core.symsolver`).  Each
backend prints that graph in its language and supplies the building blocks
(gradients, Riemann problems on all faces, reductions, linear solves).  Two
templates exist:

| template | solver classes | for |
|---|---|---|
| `HyperbolicTemplate` (`strategies/hyperbolic.py`) | `SymbolicHyperbolicSolverNumpy`, `...Jax`, `...Foam`, `...Amrex` | balance laws: SWE, SME, Euler, diffusion |
| `ChorinTemplate` (`strategies/chorin.py`) | `SymbolicChorinSolverNumpy`, `...Jax`, `...Foam`, `...Amrex` | pressure split (`VAM`): predictor, pressure solve, correction |

The NumPy classes are in `zoomy_core.symsolver.numpy_backend` (pure Python, also under Pyodide), the
JAX ones in `zoomy_jax.symsolver.solver`, the foam ones in `zoomy_foam.symsolver`, the AMReX ones in
`zoomy_amrex.symsolver`.  The settings shared by all of them are declared once, in
`zoomy_core.symsolver.settings` (`HyperbolicSettings`, `ChorinSettings`).

## Settings: typed options, one hierarchy

Every choice is a typed option (a str-`Enum`, `zoomy_core.misc.options.Option`): the IDE completes
`Imex.`, `Projection.`, `LinearSolver.`, and a wrong value names the closest one.  Every solver
takes its settings from three places, each overriding the one before:

1. the defaults declared on the class (the tables below);
2. the `solver` section of a settings file (`.yml` or `.json`), passed as `settings=`;
3. keyword arguments.

```yaml
# run.yml
solver:
  imex: ssp2
  projection: increment
  cfl: 0.5
```

```python
from zoomy_core.symsolver.discretization.time import Imex
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

s = SymbolicChorinSolverJax(settings="run.yml", cfl=0.3)    # cfl 0.3: the argument wins
s.imex                     # Imex.ssp2 (a string from the file becomes the member)
SymbolicChorinSolverJax(imex=Imex.lie)                    # typed, as in code
SymbolicChorinSolverJax(projection="incremnt")
# ValueError: 'incremnt' is not a valid Projection; did you mean 'increment'? (one of: standard, increment)
SymbolicChorinSolverJax(pressur_tol=1e-8)
# TypeError: SymbolicChorinSolverJax has no setting 'pressur_tol'; did you mean 'pressure_tol'?
```

A backend that supports only some members says which (`linear_solver="pyamg"` on AMReX:
`'pyamg' is not supported here (one of: hypre, gmres)`).  The numerics are typed the same way:
`ReconstructionSpec(limiter=Limiter.minmod)`, `nsm.spectral_radius_method = SpectralRadius.power`.

## Options (all backends)

| option | default | meaning |
|---|---|---|
| `time_end` | - | end time |
| `cfl` | `0.9` | CFL number of the adaptive step (Chorin: divided by $2\,\mathrm{order} - 1$) |
| `dt` | `None` | constant step; `None`: adaptive.  A constant step above the CFL step warns |
| `cfl_check_every` | `10` | constant `dt`: test it against the CFL step every N steps (0: never) |
| `cfl_wave_speed` | `CflWaveSpeed.faces` | `faces`: speeds of the first stage's Riemann problems (no extra eigensolve); `cells`: from the cell states |
| `time_order` | `None` | SSP-RK order 1, 2, 3; `None`: the reconstruction order.  Chorin: outer stages, default 1 |
| `mood` | `True` | a-posteriori MOOD at order 2: cells failing the model's admissibility redo at order 1 |
| `imex` | `False` | `Imex.lie`: explicit RK step, then the implicit slots (`source`, `diffusion_matrix`) by implicit Euler; `Imex.ssp2`: IMEX-SSP2(2,2,2), the implicit source in every stage |
| `implicit_tol`, `implicit_maxiter` | `1e-8`, `6` | Newton of the implicit solves |
| `gmres_tol`, `gmres_maxiter` | `1e-7`, `30` | GMRES inside Newton |
| `dt_write` | `None` | output interval; `None`: final state only |
| `couple`, `subcycling` | `None`, `False` | preCICE participant (dict); one step per window or adaptive steps inside it |

Which terms are implicit is the SystemModel's classification (the `source` and
`diffusion_matrix` slots), never the solver's.

## Chorin options

| option | default | meaning |
|---|---|---|
| `pressure_solve` | `PressureSolve.linear` | `newton` for a constraint nonlinear in the pressure (`pressure_newton_tol` 1e-10, `pressure_newton_maxit` 10) |
| `projection` | `Projection.standard` | `increment`: solve for the pressure increment over the lagged pressure (time order 2 on a fixed mesh; opt-in) |
| `pressure_tol`, `pressure_maxit` | `ELLIPTIC_RTOL`, `100` | relative tolerance and GMRES restarts of the pressure solve |
| `pressure_linear_solver` (NumPy, JAX) | `LinearSolver.auto` | `pyamg` on CPU, `amgx` on GPU (JAX) when installed, else matrix-free `gmres` |
| `linear_solver` (foam, AMReX) | `LinearSolver.hypre` | hypre FlexGMRES + BoomerAMG on the operator assembled by colouring and scaled by its inverse cell blocks; foam `schur_gamg` (GMRES, OpenFOAM's GAMG on the Schur complement) or `gmres`; AMReX `gmres` |

`docs/book/examples/chorin_vam.py` runs VAM(1) in a channel (matrix case C8):
the model's `chorin_split`, the predictor's numerics, 8 steps of
`dt` $= 2^{-10}$:

```text
state ['b', 'h', 'q_x_0', 'q_x_1', 'q_y_0', 'q_y_1', 'r_0', 'r_1', 'P_0', 'P_1']
t = 0.007812, cells = 360, h in [0.01500, 0.34274]
```

## JAX only

| option | default | meaning |
|---|---|---|
| `comm` | `None` | an mpi4py communicator: a distributed run, one rank per device |
| `ad_mode`, `ad_checkpoints` | `AdMode.forward`, `64` | `reverse`: checkpointed loop for `jax.grad` |
| `fold_geometry` | `False` | bake the mesh into the compiled loop as constants |
| `output`, `on_write` | `None` | HDF5 snapshots; a callback per write |

Backend-specific settings (ranks, GPU, AMR, containers): [Backends](backends.md).
What each option is tested by: the [test matrix](../testing/symsolver.md).

API: {doc}`zoomy_core.symsolver <../api/_autodoc2/zoomy_core/zoomy_core.symsolver>`,
{doc}`zoomy_jax.symsolver <../api/_autodoc2/zoomy_jax/zoomy_jax.symsolver>`.
