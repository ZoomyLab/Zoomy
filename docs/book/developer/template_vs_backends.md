# Template versus backends

A solver is a **template**: a Python function that, run once on sympy symbols, returns one sympy
graph of a whole time step (`zoomy_core.symsolver`). A backend prints that graph in its language and
supplies the building blocks the graph names (gradients, Riemann problems on all faces, reductions,
linear solves). Two templates exist, each with one solver class per backend.

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

The graph and the printers are described in [Architecture](templates.md), the steps to add a block,
a case or a backend in [Extending](extending.md).
