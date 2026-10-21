# Architecture: templates and printers

## A template is a Python function

`zoomy_core.symsolver.strategies.hyperbolic.HyperbolicTemplate.march` and
`...chorin.ChorinTemplate.march` are ordinary Python.  Called once with sympy
symbols for `Q`, `Qaux`, `p`, `time`, `mesh`, they return **one sympy graph**
of the whole march.  Three kinds of names appear in it
(`zoomy_core/symsolver/core.py`):

| name | what | defined in |
|---|---|---|
| kernel `nsm.<slot>` | the NSM's pointwise functions (flux, fluctuations, source, BC states) | printed by each backend's kernel printer |
| `Block` | a mesh-wide operation or solve: `cell_traces`, `faces`, `divergence`, `reduce_min`, `halo`, `linsolve`, `root`, `while_loop`, `cond` | declared in `core.py` / `vocabulary.py`, implemented once per runtime |
| `Effect` | talks to the outside: `io.write`, preCICE `couple.*` | ordered by a token threaded through the graph |

Python `if` / `for` in a template run at build time (an RK tableau is
unrolled); run-time control flow is `While` / `Cond`.  The pieces the
templates compose (`residual`, `explicit_step`, `mood_step`, `imex_step`,
reconstruction, time steps) are in `zoomy_core/symsolver/discretization/`.

## One graph, two printers

`PyPrinter` emits Python for JAX, `CxxPrinter` C++ for foam and AMReX
(`zoomy_core/symsolver/backend_cxx.py` prints the host program `Solver.H`
both C++ runtimes compile).  [Code printer: AMReX](../tutorials/ipynb/components/printer_amrex.ipynb)
prints a template as C++ and reads it.

The schemes therefore cannot drift apart between backends: a change to a
template changes every backend; a backend only decides how a block is
computed.

## The runtimes

A runtime gives every printed name its meaning.

| backend | blocks | kernels | driver (build, case, launch, read-back) |
|---|---|---|---|
| JAX | `zoomy_jax/symsolver/runtime.py` (`runtime()` returns the name table); `parallel.py` (halo, reductions over MPI) | `zoomy_jax/transformation/jax_runtime.py` (`JaxRuntime.from_nsm`) | `zoomy_jax/symsolver/solver.py` |
| foam | `zoomy_foam/symsolver/rt.H` + `zoomy_core/symsolver/rt_common.H` | `zoomy_foam/zoomy_foam/symkernels.py` -> `Kernels.H` | `zoomy_foam/zoomy_foam/symsolver.py`; app `symsolver/symFoam.C` |
| AMReX | `zoomy_amrex/Source/symsolver/rt_amrex.H` (+ `amr_amrex.H`) + `rt_common.H` | `zoomy_amrex/zoomy_amrex/symkernels.py` -> `Kernels.H` | `zoomy_amrex/zoomy_amrex/symsolver.py`; app `Source/symsolver/symAmrex.cpp` |

The C++ kernels are functors on fixed-size stack storage, so a cell or face
call allocates nothing: foam's take `zs::Col` views of a field column and
return a row-major `std::array`, AMReX's take and return `amrex::SmallMatrix`.

`rt_common.H` holds what both C++ runtimes share: elementwise functors, the
`configure()` settings, effects, the Newton skeletons and the preCICE call
sequence.  A solver class combines a driver with a strategy by inheritance,
driver first: `class SymbolicChorinSolverFoam(FoamDriver, CxxChorinBackend)`.
