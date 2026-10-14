# NumPy backend

The NumPy backend runs the solver templates in plain Python: no JAX, no compiler, serial.
It lives **inside** `zoomy_core`, in `zoomy_core.symsolver.numpy_backend`; there is no separate
package.  It takes the same `NumericalSystemModel` as every other backend and, being the
lightest, also runs where nothing else installs (Pyodide, the browser).

- The template is printed to Python source by `PyPrinter` and run against
  `numpy_backend.runtime` (the building blocks: gradients, Riemann problems on all faces,
  reductions, linear solves) and `NumpyKernels` (the NSM's kernels).
- Pure Python / NumPy / SciPy; [PyAMG](https://github.com/pyamg/pyamg) is optional.
- MPI: `comm=` (mpi4py) runs distributed on JAX's partition (PETSc solves via petsc4py, else a
  distributed GMRES).  No AD and no preCICE; use [JAX](jax.md).

## Install

```bash
pip install zoomy_core
```

## Solvers

| Class | Template | Call |
| --- | --- | --- |
| `SymbolicHyperbolicSolverNumpy` | `HyperbolicTemplate`: SSP-RK step of the flux, nonconservative and source terms, MOOD, IMEX | `setup(mesh, nsm)`, then `run(Q, Qaux)` |
| `SymbolicChorinSolverNumpy` | `ChorinTemplate`: predictor, pressure solve, correction (`VAM`) | `setup(mesh, pred, press, corr)`, then `run(Q, Qaux)` |

`setup` returns the initial state `(Q, Qaux)` from the SystemModel's initial conditions (a
SystemModel is lifted to an NSM with the defaults); `run` returns `(time, Q, Qaux)`.  `Q` has the
shape `(n_state, n_inner_cells)`; find a row by name, `[str(s) for s in nsm.state].index("h")`.
`initial_aux(Q, Qaux)` recomputes the aux variables (the pointwise formulas and the derivative aux) of
a state you wrote into `Q` yourself, as the Chorin example below does.

## Settings

The settings shared by all backends (`time_end`, `cfl`, `dt`, `time_order`, `mood`, `imex`,
`dt_write`, restart, and for Chorin `pressure_solve`, `projection`, `pressure_tol`) are the tables of
[Solvers and options](../guide/solvers.md); they are `param` parameters declared on the class
and checked at construction: a choice is a member of an `Option` enum, a plain string raises
`TypeError`.  A solver carries no numerical knobs of the scheme: reconstruction order and
limiter, the Riemann solver and the eigenvalue treatment belong to the
`NumericalSystemModel` ([NumericalSystemModel](../guide/numerical-system-model.md)), so the same
solver runs any discretisation.

NumPy adds:

| option | default | meaning |
|---|---|---|
| `output` | `None` | HDF5 file for the snapshots of `dt_write` |
| `on_write` | `None` | callback `on_write(time, Q, Qaux)` per snapshot |
| `pressure_linear_solver` (Chorin) | `LinearSolver.auto` | `pyamg` (when installed) or matrix-free `gmres` |
| `pressure_amg_rebuild`, `pressure_amg_growth` (Chorin) | `10`, `2.0` | rebuild the AMG hierarchy at least every N solves, or earlier once the GMRES iterations exceed `growth` times those after the last rebuild |

## End-to-end example: shallow water dam break

Shallow water is `SME(level=0)`: the level-0 member of the shallow-moment hierarchy, derived by
the same operators as every other level, with no hand-written system anywhere.  A 1-D case
runs the `dimension=2` model on a 1-D mesh; the state is then `[b, h, q_0]`.  The case is
`docs/book/examples/dambreak.py`, shared by the backend pages:

```{literalinclude} ../examples/dambreak.py
:language: python
```

Run it from `docs/book/examples/` after `source env.sh`:

```{literalinclude} ../examples/numpy_dambreak.py
:language: python
```

```text
t = 0.500, cells = 300, h in [1.00000, 2.00000], mass = 15.000000
```

About 5 s.  At $t = 0.5$ the waves have not reached the open ends, so the extremes are the
initial depths and the mass is that of the initial state, $2\cdot5 + 1\cdot5$.

## Chorin projection: `VAM`

`SymbolicChorinSolverNumpy` takes the three systems of the model's pressure split: the predictor
NSM on the full state (`predictor_nsm` pads it and builds the numerics of the predictor), the
pressure system and the corrector.  The VAM(1) channel of
[Solvers and options](../guide/solvers.md) (matrix case C8), on NumPy:

```{literalinclude} ../examples/chorin_vam_numpy.py
:language: python
```

```text
t = 0.007812, cells = 360, h in [0.01500, 0.34276]
```

About 15 s; the JAX solver of `chorin_vam.py` prints the same extremes.

## Authoring checklist: what the solver expects

- Every closure the model declares is supplied: a declared closure that reaches the SystemModel
  unbound raises `UnclosedClosureError` (`SME(level=0)` needs `[Newtonian(), ManningFriction(),
  StressFree()]` or another bed and surface stress).
- Boundary-condition tags match the mesh's boundary tags; periodic pairs are resolved on the mesh
  before the BC kernel is built.
- The initial condition is set on the SystemModel (`sm.initial_conditions`), as the NSM reads it
  at `setup`.
- Choose the limiter to match the smoothness (`Limiter.venkatakrishnan` for smooth flow,
  `Limiter.minmod` with shocks) on the `ReconstructionSpec`.
- A free-surface model declares its state roles (`sm.roles`: depth, bed, momentum, gravity).
  Generic code reads them instead of guessing from field names, and a model that declares none
  gets no free-surface specialisation.  The derived models (`SME`, `VAM`, ...) declare them; a
  hand-written SystemModel must.

**API reference**: {doc}`zoomy_core.symsolver <../api/_autodoc2/zoomy_core/zoomy_core.symsolver>`.
