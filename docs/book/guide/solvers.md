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

## Settings: declared with `param`, typed options

Every setting is a `param` parameter declared on the solver class: its default, documentation and
bounds sit in one place (the tables below list them), and the constructor's keyword arguments are
the only way in.  A choice is a member of a str-`Enum` (`zoomy_core.misc.options.Option`) held by
that parameter, so the IDE completes `Imex.`, `Projection.`, `LinearSolver.`.  A plain string is
refused with the member to use, a typo with the closest one, and a misspelt argument name with the
closest setting:

```python
from zoomy_core.symsolver.settings import Imex, Projection
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

s = SymbolicChorinSolverJax(imex=Imex.ssp2, projection=Projection.increment, cfl=0.5)
SymbolicChorinSolverJax(projection="increment")
# TypeError: projection='increment': use Projection.increment
SymbolicChorinSolverJax(projection="incremnt")
# TypeError: projection='incremnt': did you mean Projection.increment? (one of: Projection.standard, Projection.increment)
SymbolicChorinSolverJax(pressur_tol=1e-8)
# TypeError: SymbolicChorinSolverJax has no setting 'pressur_tol'; did you mean 'pressure_tol'?
```

Assignment is checked the same way (`s.imex = "lie"`: `TypeError: imex='lie': use Imex.lie`).  A
string from a command line or an environment variable is converted explicitly,
`LinearSolver(os.environ["SOLVER"])`.  A backend that supports only some members says which:
`SymbolicChorinSolverAmrex(linear_solver=LinearSolver.pyamg)` raises `ValueError:
linear_solver=LinearSolver.pyamg is not supported here (one of: LinearSolver.hypre,
LinearSolver.gmres)`.

| options | module |
|---|---|
| `Imex`, `Projection`, `CflWaveSpeed`, `PressureSolve`, `LinearSolver` | `zoomy_core.symsolver.settings` |
| `Limiter`, `ReconstructedVariables`, `BedReconstruction` | `zoomy_core.numerics` |
| `AdMode` | `zoomy_jax.symsolver.solver` |
| `Decomposition` | `zoomy_foam.symsolver` |

The numerics are typed the same way: `ReconstructionSpec(limiter=Limiter.minmod)`,
`nsm.spectral_radius_method = SpectralRadius.eig`,
`UpwindInterface(quantity=InterfaceQuantity.stress)`.

## Options (all backends)

| option | default | meaning |
|---|---|---|
| `time_end` | - | end time |
| `cfl` | `0.9` | CFL number of the adaptive step (Chorin: divided by $2\,\mathrm{order} - 1$) |
| `dt` | `None` | constant step; `None`: adaptive.  A constant step above the CFL step warns |
| `cfl_check_every` | `10` | constant `dt`: test it against the CFL step every N steps (0: never) |
| `cfl_wave_speed` | `CflWaveSpeed.faces` | `faces`: speeds of the first stage's Riemann problems, at order 2 with an edge source (the NCP inside each cell) also the cells' own; `cells`: from the cell states |
| `time_order` | `None` | SSP-RK order 1, 2, 3; `None`: the reconstruction order.  Chorin: outer stages, default 1 |
| `mood` | `True` | a-posteriori MOOD at order 2: cells failing the model's admissibility redo at order 1 (with `faces`, at most the step of their own first-order face speeds) |
| `imex` | `False` | `Imex.lie`: explicit RK step, then the implicit slots (`source`, `diffusion_matrix`) by implicit Euler; `Imex.ssp2`: IMEX-SSP2(2,2,2), the implicit source in every stage |
| `implicit_tol`, `implicit_maxiter` | `1e-8`, `6` | Newton of the implicit solves |
| `gmres_tol`, `gmres_maxiter` | `1e-7`, `30` | GMRES inside Newton |
| `dt_write` | `None` | output interval; `None`: final state only |
| `couple`, `subcycling` | `None`, `False` | preCICE participant (dict); one step per window or adaptive steps inside it |

Which terms are implicit is the SystemModel's classification (the `source` and
`diffusion_matrix` slots), never the solver's.

## Restart

| option | default | meaning |
|---|---|---|
| `time_start` | `0.0` | start time when not restarting |
| `restart_from` | `None` | a restart state written by the same backend; its time overrides `time_start` |
| `dt_checkpoint` | `None` | interval of the restart states (the march lands on them, as on `dt_write`); `None`: one at the end |
| `checkpoint` | `None` | where they go; `None`: next to the output |

A restart state is the template's march state `(time, Q, Qaux)`, written by the effect
`io.save_state` and read by `io.load_state` (the same pair on every backend).  It is the whole
loop carry: the pressure rows of `Q` hold the lagged pressure of `Projection.increment`, and the
CFL-check cadence and the write / checkpoint counters are functions of the time.  So a run
0 -> T with `dt_checkpoint=T/2` and a run 0 -> T/2 restarted to T end in the same state, bit for
bit, also with the adaptive step (matrix case C35).  The lagged solver caches (the `pyamg`
hierarchy, NumPy's Jacobian LU) restart at every save, as in a restarted run.

| backend | restart state | `restart_from` | MPI |
|---|---|---|---|
| NumPy, JAX | one HDF5 file: `time`, `Q`, `Qaux` in the mesh's cell order; default `<output stem>.restart.h5`, none without `output` or `checkpoint` | the file | JAX: gathered to and written by rank 0, every rank reads its cells (any rank count) |
| foam | the time directory `<t>/`: `Q<i>`, `Qaux<i>`, `uniform/zoomyState` (the exact time), in the case or in `checkpoint`; the run starts from it (`startFrom startTime`) | a time directory, or a case / checkpoint directory (its latest) | `processor<k>/<t>`; the same decomposition |
| AMReX | `chk<t>/`: `Header` and the MultiFabs `Q`, `Qaux` (VisMF), in `out/` or in `checkpoint` | a `chk<t>` directory, or one holding them (its latest) | distributed MultiFabs; another rank count is redistributed |

```python
s = SymbolicHyperbolicSolverJax(time_end=1.25, checkpoint="bump.restart.h5")
Q, A = s.setup(mesh, nsm)
s.run(Q, A)                         # writes bump.restart.h5 at t = 1.25
s = SymbolicHyperbolicSolverJax(time_end=2.5, restart_from="bump.restart.h5")
Q, A = s.setup(mesh, nsm)
t, Q, A = s.run(Q, A)               # continues from t = 1.25
SymbolicHyperbolicSolverJax(time_end=2.5, restart_from="bump.restart.h").setup(mesh, nsm)
# FileNotFoundError: restart_from 'bump.restart.h': no such file; did you mean 'bump.restart.h5'?
```

Not with preCICE coupling or AMReX AMR (`NotImplementedError`).

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
t = 0.007812, cells = 360, h in [0.01500, 0.34276]
```

## JAX only

| option | default | meaning |
|---|---|---|
| `comm` | `None` | an mpi4py communicator: a distributed run, one rank per device |
| `ad_mode`, `ad_checkpoints` | `AdMode.forward`, `64` | `reverse`: checkpointed loop for `jax.grad` |
| `fold_geometry` | `False` | bake the mesh into the compiled loop as constants |
| `output`, `on_write` | `None` | HDF5 snapshots; a callback per write |

Backend-specific settings (ranks, GPU, AMR, containers): [Backends](backends.md).
What each option is tested by: the [test matrix](../developer/tests.md).

API: {doc}`zoomy_core.symsolver <../api/_autodoc2/zoomy_core/zoomy_core.symsolver>`,
{doc}`zoomy_jax.symsolver <../api/_autodoc2/zoomy_jax/zoomy_jax.symsolver>`.
