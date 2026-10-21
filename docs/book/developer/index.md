# Developer guide

For those who change Zoomy: how the symbolic solvers are built, where to change them, and how it is
tested.  To use Zoomy, see the [user guide](../guide/index.md).

| page | content |
|---|---|
| [Developer installation](installation.md) | clone with submodules, editable installs, local images |
| [Architecture](templates.md) | a solver written once as a sympy graph, printed per backend; the layout of the runtimes |
| [Extending](extending.md) | add a backend operation, a test-matrix case, a timing case, a backend |
| [Adding a tutorial](tutorials.md) | the notebook metadata and the tag vocabulary |
| [Tests and timings](tests.md) | the cross-backend matrix and the solver timings |
| [CI reports](ci-reports.md) | the test suites per push and per week |
| [GUI](gui.md) | how the browser GUI is built and deployed |
| [Writing the docs](docs.md) | Markdown only, and how the site is built |

Design notes (not part of the book; they record decisions and measurements):

| note | topic |
|---|---|
| [`ANALYSIS.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/ANALYSIS.md) | design of the templates, decisions, gates |
| [`PARALLEL.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/PARALLEL.md) | distributed runs: partition, halos, reductions |
| [`AMREX.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMREX.md) | the AMReX runtime, AMR |
| [`AMG.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMG.md), [`amg_comparison.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/amg_comparison.md) | pressure solvers and preconditioners |
| [`matrix/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/matrix/README.md) | the cross-backend test matrix: cases, thresholds, environment |
| [`tests/symsolver/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/tests/symsolver/README.md) | the timing suite |
| [`tests/qr_cases/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/tests/qr_cases/README.md) | the thesis QR-code cases, run in CPython and Pyodide on every commit (CI "QR cases") |

The scripts under `docs/dev/symbolic_solver/gates/` are the acceptance gates
the matrix wraps; they are scratch, not library tests.
