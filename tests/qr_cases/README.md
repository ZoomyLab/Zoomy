# QR cases

[![QR cases](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml/badge.svg)](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml)

- **What:** every URL in the thesis QR manifest (`thesis/scripts/animations.yml`, snapshot `animations.yml` here). Each GUI session a code opens is composed as the GUI composes it (model, steps, mesh, settings, run, visualization) and executed through `library/zoomy_gui/engine.py`.
- **CPython** (`-m qr`): `zoomy_core` from the checkout, jax blocked for browser-local sessions; the JAX session (`-m qr_jax`, Malpasset, `time_end` 2000 s cut to 2 s) adds `zoomy_jax`. Pass = all cells succeed, every float in every HDF5 store finite.
- **Pyodide** (`-m qr`): the same cells in Pyodide 0.29.3 under node, booted like `pyodide-worker.js`, with a wheel of the checkout's `zoomy_core`; plus the C1 bump on `SymbolicHyperbolicSolverNumpy` (t = 0.5, mass to 1e-12).
- **Live** (`-m qr_live`, after each deploy in `render-webpage.yml`): every target answers 200; every project zip is served with the repo's file list.
- **Run:** `pip install -r tests/qr_cases/requirements.txt; npm ci --prefix tests/qr_cases/pyodide; pytest tests/qr_cases -m qr -n 4` (`-m qr_live` for the site).
- **Env:** `QR_MANIFEST` (manifest path), `QR_PYODIDE_DIR` / `QR_NODE` (npm pyodide elsewhere), `QR_PYODIDE_CACHE`, `QR_STRICT=1` (CI: missing tools fail).
