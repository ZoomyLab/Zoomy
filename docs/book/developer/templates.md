# Templates and printers

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
both C++ runtimes compile).  `docs/book/examples/print_template.py` prints
the first-order C1 march both ways:

```{literalinclude} ../examples/print_template.py
:language: python
```

```text
Python: 24 lines; C++: 61 lines

def hyperbolic(Q, Qaux, p, time, time_end, mesh, tok):
    def f1(time_2, Q_2, Qaux_2, n_write, tok_2):
        return (time_2 < time_end)
    x2 = face_inradius(mesh)
    def f3(time_3, Q_3, Qaux_3, n_write_2, tok_3):
        WL, WR = cell_traces(Q_3, mesh)
        QR = boundary_traces(nsm.boundary_conditions, Q_3, WL, WR, Qaux_3, p, time_3, mesh)
        AL, AR = face_aux(Qaux_3, mesh)
        Wm, Wp, S = faces(nsm.numerical_face_sides_speed, WL, QR, AL, AR, p, mesh)
        dt = reduce_min(0.9*x2/S)
        x14 = minimum(dt, 5.0)
        dt_2 = minimum(x14, -time_3 + time_end)
        time_4 = time_3 + dt_2
        x15 = divergence(Wm, Wp, mesh)
...

    auto x2 = face_inradius(mesh);
    auto f3 = [&](auto time_3, auto Q_3, auto Qaux_3, auto n_write_2, auto tok_3) {
        auto x4 = cell_traces(Q_3, mesh);
        auto WL = std::get<0>(x4);
        auto WR = std::get<1>(x4);
        auto QR = boundary_traces(nsm.boundary_conditions, Q_3, WL, WR, Qaux_3, p, time_3, mesh);
        auto x5 = face_aux(Qaux_3, mesh);
        auto AL = std::get<0>(x5);
        auto AR = std::get<1>(x5);
        auto x6 = faces<3>(nsm.numerical_face_sides_speed, WL, QR, AL, AR, p, mesh);
        auto Wm = std::get<0>(x6);
        auto Wp = std::get<1>(x6);
```

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

`rt_common.H` holds what both C++ runtimes share: elementwise functors, the
`configure()` settings, effects, the Newton skeletons and the preCICE call
sequence.  A solver class combines a driver with a strategy by inheritance,
driver first: `class SymbolicChorinSolverFoam(FoamDriver, CxxChorinBackend)`.
