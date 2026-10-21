# Templates and backends

## A template is a Python function

`zoomy_core.symsolver.strategies.hyperbolic.HyperbolicTemplate.march` and
`...chorin.ChorinTemplate.march` are ordinary Python.  Called once with sympy
symbols for `Q`, `Qaux`, `p`, `time` and `mesh`, they return one sympy graph
of the whole march.  Three kinds of names appear in it
(`zoomy_core/symsolver/core.py`):

| name | what | defined in |
|---|---|---|
| kernel `nsm.<slot>` | the NSM's pointwise functions (flux, fluctuations, source, BC states) | printed by each backend's kernel printer |
| `Block` | a mesh-wide operation or solve: `cell_traces`, `faces`, `divergence`, `reduce_min`, `halo`, `linsolve`, `root`, `while_loop`, `cond` | declared in `core.py` and `vocabulary.py`, implemented once per runtime |
| `Effect` | talks to the outside: `io.write`, preCICE `couple.*` | ordered by a token threaded through the graph |

Python `if` and `for` in a template run at build time, so an RK tableau is
unrolled.  Run-time control flow is `While` and `Cond`.  The pieces the
templates compose (`residual`, `explicit_step`, `mood_step`, `imex_step`,
reconstruction, time steps) are in `zoomy_core/symsolver/discretization/`.

## One graph, two printers

`PyPrinter` emits Python for NumPy and JAX, `CxxPrinter` emits C++ for OpenFOAM and AMReX
(`zoomy_core/symsolver/backend_cxx.py` prints the host program both C++ runtimes compile).
A change to a template changes every backend.  A backend decides only how a block is computed.
[Code printer: AMReX](../tutorials/ipynb/components/printer_amrex.ipynb) prints a template as C++.

## Two templates

A backend prints the graph in its language and supplies the building blocks the graph names
(gradients, Riemann problems on all faces, reductions, linear solves).
Two templates exist, each with one solver class per backend.

| template | solver classes | for |
|---|---|---|
| `HyperbolicTemplate` (`symsolver/strategies/hyperbolic.py`) | `SymbolicHyperbolicSolver{Numpy,Jax,Foam,Amrex}` | balance laws: SWE, SME, Euler, diffusion |
| `ChorinTemplate` (`symsolver/strategies/chorin.py`) | `SymbolicChorinSolver{Numpy,Jax,Foam,Amrex}` | pressure split (`VAM`): predictor, pressure solve, correction |

## Where to change what

| to change | edit | reaches |
|---|---|---|
| an equation, a closure, a boundary condition | the model, its derivation or `zoomy_core/model/boundary_conditions.py` | every backend, printed as kernels |
| reconstruction, Riemann solver, wave speed | `zoomy_core/numerics` and `zoomy_core/fvm/riemann_solvers.py` | every backend |
| time stepping, MOOD, IMEX, Chorin stages | the template or `symsolver/discretization/` | every backend |
| a setting shared by all backends | `zoomy_core/symsolver/settings.py` | every backend |
| a mesh-wide operation | declare the `Block` in `symsolver/vocabulary.py`, implement it in every runtime | every backend |
| a setting of one backend | the driver class of that backend | that backend |
| how a backend builds, launches or reads back | the driver (`solver.py`, `symsolver.py`) | that backend |

Backends hold no numerical constants of their own. A run that needs different operators is fixed in
the derivation, never in a printer or a runtime.

## What a backend does

| backend | step | driver |
|---|---|---|
| NumPy | `PyPrinter` prints the template as Python, run against `numpy_backend.runtime` | `zoomy_core/symsolver/numpy_backend/solver.py` |
| JAX | the same Python source, run against `zoomy_jax/symsolver/runtime.py` and jitted | `zoomy_jax/symsolver/solver.py` |
| foam | `CxxPrinter` prints `Solver.H` and `Kernels.H`, `wmake` builds `symFoam`, cached by a hash of the sources | `zoomy_foam/zoomy_foam/symsolver.py` |
| AMReX | the same `Solver.H` plus device kernels, GNU make builds the binary, cached by a hash of the sources | `zoomy_amrex/zoomy_amrex/symsolver.py` |

The steps to add a block, a case or a backend are in [Extending](extending.md).

## Design notes

| note | topic |
|---|---|
| [`ANALYSIS.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/ANALYSIS.md) | templates, decisions, gates |
| [`PARALLEL.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/PARALLEL.md) | distributed runs: partition, halos, reductions |
| [`AMREX.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMREX.md) | the AMReX runtime, AMR |
| [`AMG.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMG.md), [`amg_comparison.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/amg_comparison.md) | pressure solvers and preconditioners |
| [`matrix/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/matrix/README.md) | the test matrix: cases, thresholds, environment |
| [`tests/symsolver/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/tests/symsolver/README.md) | the timing suite |
| [`tests/qr_cases/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/tests/qr_cases/README.md) | the thesis QR cases |
