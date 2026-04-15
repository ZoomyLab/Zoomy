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

**Configuration layer** -- `core.js` is a pure-logic module (no DOM) shared by both GUI and CLI. It manages the data model: cards, selections, sessions, case building, and project serialization.

**Execution layer** -- simulations run either in-browser (Pyodide for NumPy) or on a server (zoomy-server with adapter pattern for all backends).

## Key Modules

### zoomy_core (Python library)

The symbolic/numerical core:

| Module | Purpose |
|--------|---------|
| `model` | Symbolic model definitions (SymPy-based) |
| `mesh` | BaseMesh, FVMMesh, LSQMesh hierarchy |
| `fvm` | FVM solvers, timestepping, Riemann solvers |
| `transformation` | Code generators (to_c, to_numpy, to_jax) |

### zoomy_server (FastAPI)

```
library/zoomy_server/zoomy_server/
  routes.py               REST API: /api/v1/health, /api/v1/jobs, /api/v1/registry
  adapter.py              SolverAdapter base class with shared preprocessing
  jobs.py                 Job manager (ProcessPoolExecutor, progress tracking)
  registry.py             Card discovery from Python packages and user files
  adapters/               NumPy, JAX, DMPlex, AMReX, Mesh adapters
```

### zoomy_gui + zoomy_cli

See [GUI and CLI Architecture](dev-gui-cli.md) for details on the card system, `display()` pipeline, session management, and deployment.

## Deployment

### GitHub Pages

The `render-webpage.yml` workflow copies `library/zoomy_gui/*` into the Jupyter Book build at `/gui/`:

- Docs: `https://zoomylab.github.io/Zoomy/`
- GUI: `https://zoomylab.github.io/Zoomy/gui/`

### Backend servers (Docker)

Each solver backend runs as a Docker container:

```bash
docker run -p 8080:8080 ghcr.io/zoomylab/zoomy-numpy:latest
```

Multiple backends can run on different ports. The GUI/CLI connects to each independently.
