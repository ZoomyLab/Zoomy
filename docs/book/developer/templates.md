# Architecture

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
