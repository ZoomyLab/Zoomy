# Developer guide

How the symbolic solvers are built, and where to change them.

| page | content |
|---|---|
| [Templates and printers](templates.md) | a solver written once as a sympy graph, printed per backend; the layout of the runtimes |
| [Extending](extending.md) | add a backend operation, a test-matrix case, a timing case, a backend |

Design notes (not part of the book; they record decisions and measurements):

| note | topic |
|---|---|
| [`ANALYSIS.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/ANALYSIS.md) | design of the templates, decisions, gates |
| [`PARALLEL.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/PARALLEL.md) | distributed runs: partition, halos, reductions |
| [`AMREX.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMREX.md) | the AMReX runtime, AMR |
| [`AMG.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/AMG.md), [`amg_comparison.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/amg_comparison.md) | pressure solvers and preconditioners |
| [`matrix/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/dev/symbolic_solver/matrix/README.md) | the cross-backend test matrix: cases, thresholds, environment |
| [`tests/symsolver/README.md`](https://github.com/ZoomyLab/Zoomy/blob/main/tests/symsolver/README.md) | the timing suite |

The scripts under `docs/dev/symbolic_solver/gates/` are the acceptance gates
the matrix wraps; they are scratch, not library tests.

## Writing and building the docs

- **Markdown (MyST), never reStructuredText**: pages are `.md`, and so are the docstrings. The API reference ([sphinx-autodoc2](https://sphinx-autodoc2.readthedocs.io)) reads them as MyST: backticks for code, `[text](url)` for links, `` ```{note} `` for admonitions; `Args:` / `Returns:` / `References:` sections stay readable (`docs/book/_ext/zoomy_docstrings.py`). Old rST markup in docstrings is rewritten by `python docs/scripts/docstrings_rst_to_md.py PATH` (a dry run unless `--write`).
- **The site is built by CI from `main`** (`.github/workflows/render-webpage.yml`, in the image of `docs/Dockerfile.docs`): the scripts of `docs/scripts/` write the report and API pages, then `jupyter-book build docs/book`. **Never commit built or generated output** (`_build`, `api/_autodoc2`, `docs/_generated`); preview in that image, do not push a local build.
