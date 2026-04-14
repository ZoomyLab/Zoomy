# Architecture Overview

## System Layers

Zoomy is organized in three layers:

```
                  ┌──────────────────────────────────┐
  User layer      │     GUI (browser)    CLI (node)   │
                  ├──────────────────────────────────┤
  Configuration   │         core.js (shared)          │
                  │   Cards / Sessions / Selections   │
                  ├──────────────────────────────────┤
  Execution       │  Pyodide  │  zoomy-server (HTTP)  │
                  │ (browser) │  ┌──────────────────┐ │
                  │           │  │ SolverAdapter     │ │
                  │           │  │  ├─ NumPy         │ │
                  │           │  │  ├─ JAX           │ │
                  │           │  │  ├─ DMPlex (C++)  │ │
                  │           │  │  └─ AMReX (C++)   │ │
                  │           │  └──────────────────┘ │
                  └──────────────────────────────────┘
```

**User layer** -- two interfaces (GUI and CLI) that present the same card-based configuration system. The GUI runs in the browser as a static site; the CLI runs in Node.js.

**Configuration layer** -- `core.js` is a pure-logic module (no DOM) shared by both GUI and CLI. It manages the data model: cards, selections, sessions, case building, and project serialization. This ensures both interfaces produce identical case configurations.

**Execution layer** -- simulations run either in-browser (Pyodide for NumPy) or on a server (zoomy-server with adapter pattern for all backends).

## Components

### zoomy_gui (static web app)

```
library/zoomy_gui/
  index.html              HTML shell (loads scripts + CDN libs)
  core.js                 Shared business logic (CardState, SelectionManager, SessionManager, Project)
  app.js                  UI rendering, event handlers, save/load, display() output cells
  backend.js              Backend connection management, job submission/polling
  param_widgets.js        Parameter form rendering from JSON schema
  engine.py               Python execution engine (loaded into Pyodide)
  pyodide-worker.js       Web Worker: Pyodide runtime, param extraction, code execution
  style.css               All styling (design tokens, card layout, output cells)
  cards/                  Card definitions (default.json, generated.json, user.json per category)
  tutorials/              Tutorial ZIP bundles + index.json manifest
```

**Key design decisions:**
- Pure static site -- no server-side rendering, deployable to GitHub Pages
- Pyodide Web Worker -- Python runs in a background thread, never blocks UI
- Card-based configuration -- models, meshes, solvers are JSON card descriptors
- Per-session state -- each session snapshots its own selections + card overrides

### zoomy_cli (Node.js CLI)

```
library/zoomy_cli/
  cli.js                  Full CLI entry point (requires core.js from zoomy_gui)
  package.json            Node dependencies (jszip, inherits)
  examples/               Tutorial batch scripts
```

**Key design decisions:**
- Requires `core.js` directly from `zoomy_gui/` -- single source of truth for business logic
- State persisted in `.zoomy/state.json` in the project directory
- `zoomy run --local` generates inline Python from card configs, executes via `python -c`
- `zoomy run` submits case JSON to a backend server via HTTP

### zoomy_server (FastAPI)

```
library/zoomy_server/zoomy_server/
  routes.py               REST API: /api/v1/health, /api/v1/jobs, /api/v1/registry
  adapter.py              SolverAdapter base class with shared preprocessing
  jobs.py                 Job manager (ProcessPoolExecutor, progress tracking)
  registry.py             Card discovery: scans zoomy_core for models/solvers, user session files
  adapters/
    numpy.py              In-process FVM solving with NumPy
    jax.py                JIT-compiled solving with JAX
    dmplex.py             C++ codegen + PETSc compile + MPI execution
    mesh.py               Mesh preprocessing (gmsh, converters)
```

**Key design decisions:**
- Adapter pattern -- each backend implements `solve(case_dir, output_dir, on_progress)`
- Case folder format -- `model.py`, `mesh.py`, `numerics.py`, `settings.json`
- Registry API -- auto-discovers cards from Python packages and user files
- CORS enabled -- GUI can connect from any origin

### zoomy_core (Python library)

The symbolic/numerical core. Not covered in detail here (see API docs), but key modules:

| Module | Purpose |
|--------|---------|
| `model` | Symbolic model definitions (SymPy-based) |
| `mesh` | BaseMesh, FVMMesh, LSQMesh hierarchy |
| `fvm` | FVM solvers, timestepping, Riemann solvers |
| `transformation` | Code generators (to_c, to_numpy, to_jax) |

## Data Flow

### GUI workflow

```
cards/*.json ──→ Project.fromConfig() ──→ CardState + Selections
                                              │
                                     User edits cards/params
                                              │
                             ┌────────────────┴────────────────┐
                             │                                  │
                    [NumPy / Pyodide]                   [Backend server]
                             │                                  │
                   resolveCode() per card              buildCase() → JSON
                   concatenate into script              POST /api/v1/jobs
                             │                                  │
                   pyodide-worker.js                  adapter.solve()
                   engine.py::process_code()                    │
                             │                         progress.json
                        display() ──→ output cells     simulation.h5
                        fig/plot ──→ dashboard
```

### CLI workflow

```
cards/*.json ──→ Project.fromConfig() ──→ CardState + Selections
                                              │
                                     zoomy select/status
                                              │
                             ┌────────────────┴────────────────┐
                             │                                  │
                    [zoomy run --local]                 [zoomy run]
                             │                                  │
                   Generate inline Python              buildCase() → JSON
                   python -c "<script>"                POST /api/v1/jobs
                             │                                  │
                        stdout output                  zoomy watch <id>
                                                       poll /api/v1/jobs/<id>
```

### Session save/load format

```
zoomy-project.zip
  project.json            version, sessions[], activeSession
    sessions[]:
      id, title, description
      selections: { model: "card-id", mesh: "card-id", solver: "card-id" }
      cardOverrides: { "card-id": { params: {...}, code: "..." } }
  <session-title>/
    <tab>/<card-title>/
      card.json           Card metadata (id, title, params, tab)
      code.py             User-edited code (optional)
```

Format version 1.1 stores per-session state. Version 1.0 (legacy) stores global state and is loaded with backwards compatibility.

## How GUI and CLI Stay in Sync

The critical insight: **`core.js` is the single source of truth**.

- `app.js` (GUI) calls `ZoomyCore.Project.fromConfig(config)` and uses `Project.buildCase()`, `buildSaveData()`, `applySaveData()`
- `cli.js` (CLI) does `var ZoomyCore = require("../zoomy_gui/core.js")` and calls the same methods

Both load cards from the same `cards/` directory structure. Both serialize/deserialize using the same ZIP format. A project saved by the GUI can be loaded by the CLI and vice versa.

## Deployment

### GitHub Pages (static GUI)

The `render-webpage.yml` workflow copies `library/zoomy_gui/*` into the Jupyter Book build output at `/gui/`. This deploys alongside the documentation:

- Docs: `https://mbd-rwth.github.io/Zoomy/`
- GUI: `https://mbd-rwth.github.io/Zoomy/gui/`

### Backend servers (Docker)

Each solver backend runs as a Docker container with `zoomy-server`:

```bash
docker run -p 8080:8080 ghcr.io/zoomylab/zoomy-numpy:latest
```

Multiple backends can run on different ports. The GUI/CLI connects to each independently.

### Tutorial distribution

Tutorial ZIPs live in `library/zoomy_gui/tutorials/`. The `tutorials/index.json` manifest describes available tutorials. They are deployed alongside the GUI on GitHub Pages.

For persistent distribution, upload ZIPs to Zenodo. The GUI resolves `zenodo:` URLs via the Zenodo REST API.
