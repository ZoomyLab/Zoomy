# Tutorial notebooks

[![Tutorials](https://github.com/ZoomyLab/Zoomy/actions/workflows/tutorials.yml/badge.svg)](https://github.com/ZoomyLab/Zoomy/actions/workflows/tutorials.yml)

- **What:** every `tutorials/**/*.ipynb` is one test of `test_notebooks.py` (marker `tutorial`, opt-in: nothing runs without `-m tutorial`). The cells run top to bottom in a fresh kernel (house rule: the last cell is an `assert`; not enforced here). A failing cell, a timeout or bad `metadata.zoomy` fails the test.
- **Metadata:** `metadata.zoomy` (`title`, `summary`, `tags`) is validated by `docs/scripts/generate_tutorial_index.py`, which also writes the tutorials page (one card per notebook, one dropdown per tag category). The vocabulary is `TAGS` in that script; `type:tutorial|use-case` is required; an unknown tag fails.
- **`requires:`** (`cpu` default, `gpu`, `mpi`, `container`) is not a filter on the page; it tells this harness what the notebook needs. A notebook that needs more than `cpu` is collected and **skipped with that reason** (`-rs` lists them), never silently dropped. A runner for those is on the November list (gaia).
- **Kernel:** this interpreter (`sys.executable`, via a temporary kernelspec), `PYTHONNOUSERSITE=1`, `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=""`, `MPLBACKEND=Agg`; working directory = the notebook's folder. Outputs are not written back.
- **Timeout:** one budget per notebook, all cells together: `TUTORIAL_TIMEOUT` seconds (default 900).
- **Run:** `pip install -e library/zoomy_core[full] -e library/zoomy_plotting pytest nbclient nbformat ipykernel; pytest tests/tutorials -m tutorial -rs`; pin cores with `taskset -c 0-7`. One notebook: `-k pyodide`.
- **CI:** `.github/workflows/tutorials.yml` runs it, called by `ci.yml` when `tutorials/`, core, jax, plotting or this folder change and by `weekly.yml`; the Smart Tests report (`tests-report.yml`, group *notebooks*) runs the same folder in the core image. `render-webpage.yml` runs the index generator, so a bad tag also fails the docs build.
- **Add a notebook:** put it under `tutorials/` with `metadata.zoomy`; nothing else is registered (no toc entry, no test list). The book page is built from a copy in `docs/book/tutorials/ipynb/` (generated, git-ignored) and reached through its card.
