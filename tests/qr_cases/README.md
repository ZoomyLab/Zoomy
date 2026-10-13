# QR cases

[![QR cases](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml/badge.svg)](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml)

- **What:** every URL in the thesis QR manifest (`thesis/scripts/animations.yml`, snapshot `animations.yml` here). Each GUI session a code opens is composed as the GUI composes it (model, steps, mesh, settings, run, visualization) and executed through `library/zoomy_gui/engine.py`.
- **Wheels:** `docs/scripts/build_gui_wheels.py DIR` builds the `zoomy_core` / `zoomy_plotting` wheels the deployed GUI installs (`gui/gui/wheels/index.json`); CI installs the same build in CPython and Pyodide (`QR_WHEELS=DIR`). Without `QR_WHEELS` the tests build them for Pyodide and CPython reads the checkout.
- **CPython** (`-m qr`): jax blocked for browser-local sessions; the JAX session (`-m qr_jax`, Malpasset, `time_end` 2000 s cut to 2 s) adds `zoomy_jax` from the checkout. Pass = all cells succeed, every float in every HDF5 store finite.
- **Pyodide** (`-m qr`): the same cells in Pyodide 0.29.3 under node, booted like `pyodide-worker.js` (a synchronous `XMLHttpRequest` is shimmed for `zoomy_core.misc.download`); plus the C1 bump on `SymbolicHyperbolicSolverNumpy` (t = 0.5, mass to 1e-12) and an import of every `zoomy_core` module.
- **Not in the browser:** Malpasset (jax) and Malpasset (AMReX) are submitted to a backend by the GUI; their Pyodide tests skip. The AMReX solve needs the backend container, so its CPython test runs the model and mesh cells only.
- **Live** (`render-webpage.yml`, after each deploy): `-m qr_live`, every target answers 200 and every project zip is served with the repo's file list; `-m qr_deploy`, the GUI serves this commit's wheels and the mirrored session data (`gui/gui/data/`).
- **Run:** `pip install -r tests/qr_cases/requirements.txt; npm ci --prefix tests/qr_cases/pyodide; pytest tests/qr_cases -m qr -n 4` (`-m qr_live` for the site).
- **Env:** `QR_MANIFEST` (manifest path), `QR_WHEELS`, `QR_PYODIDE_DIR` / `QR_NODE` (npm pyodide elsewhere), `QR_PYODIDE_CACHE`, `QR_STRICT=1` (CI: missing tools fail).
