# Component Reference

## Submodule Map

| Module | Purpose | Language | Stability |
|--------|---------|----------|-----------|
| `zoomy_core` | Symbolic models, FVM numerics, mesh hierarchy, code generators | Python | Stable |
| `zoomy_jax` | JAX-accelerated solvers, JIT compilation | Python | Evolving |
| `zoomy_gui` | Web-based GUI (static site), cards, sessions, Pyodide execution | JS/HTML/CSS/Python | Active |
| `zoomy_cli` | Command-line interface (mirrors GUI workflow) | Node.js | Active |
| `zoomy_server` | FastAPI backend, solver adapters, job management, card registry | Python | Active |
| `zoomy_mesh` | Mesh format converters (HDF5, Gmsh) | Python | Stable |
| `zoomy_dmplex` | PETSc/DMPlex backend (C++ codegen, MPI) | Python/C++ | Stable |
| `zoomy_amrex` | AMReX backend (C++ codegen, AMR) | Python/C++ | Stable |
| `zoomy_firedrake` | Firedrake/FEniCSx backends (UFL, DG) | Python | Stable |
| `zoomy_client` | Lightweight HTTP client for server API | Python | Minimal |
| `zoomy_foam` | OpenFOAM code generation | Python | Stable |
| `zoomy_js` | JavaScript utilities (shared) | JS | Minimal |
| `zoomy_tests` | Shared test infrastructure | Python | Stable |

## Card System

Cards are the fundamental configuration unit shared by GUI and CLI. Each card describes a model, mesh, solver, or visualization.

**Card structure:**
```json
{
  "id": "sme-l0",
  "title": "SWE (SME L0)",
  "class": "zoomy_core.model.models.sme_model.SMEInviscid",
  "init": {"level": 0},
  "description": "Shallow water equations...",
  "requires_tag": "numpy",
  "template": "from ... import ...\nmodel = ...",
  "snippet": "snippets/custom.py",
  "preview": "previews/image.svg"
}
```

**Card sources** (loaded in priority order):
1. `cards/<category>/default.json` -- built-in cards
2. `cards/<category>/generated.json` -- auto-discovered from `zoomy_core`
3. `cards/<category>/user.json` -- user-created cards
4. Server registry (`/api/v1/registry`) -- additional cards from running backends

**Code resolution** (for execution):
1. User-edited code (from editor) -- highest priority
2. `template` field with `{placeholder}` substitution
3. Auto-generated from `class` + `init` fields

## Server Adapter Pattern

Each solver backend implements the `SolverAdapter` interface:

```python
class SolverAdapter:
    tag = "numpy"  # identifier for backend routing

    def solve(self, case_dir, output_dir, on_progress):
        """Run simulation from case folder to output folder."""
        # 1. Execute mesh.py (optional preprocessing)
        # 2. Load model from model.py
        # 3. Load mesh from case files
        # 4. Create solver and run
        # 5. Write results to output_dir/simulation.h5
        # 6. Report progress via on_progress(iteration, time, dt)
```

**Case folder format** (what adapters receive):
```
case_dir/
  model.py          Python file defining `model` instance
  numerics.py       Optional: numerical scheme configuration
  mesh.py           Optional: preprocessing script (generates mesh)
  settings.json     Solver parameters (time_end, cfl, output_snapshots)
  mesh.h5 / mesh.msh   Mesh file (from mesh.py or pre-existing)
```

**Job lifecycle:**
1. Client POSTs to `/api/v1/jobs` with case data
2. Server creates job ID, spawns worker in process pool
3. Worker calls `adapter.solve(case_dir, output_dir, on_progress)`
4. Client polls `GET /api/v1/jobs/{id}` for status
5. On completion, results available at `/api/v1/jobs/{id}/results/hdf5`

## display() Pipeline

The `display()` function provides Jupyter-like output cells in the GUI code editor.

**Data flow:**

```
User code:  display(obj)
     │
     ▼
engine.py:  ZoomyDisplay.__call__(obj)
     │      Detects type: text/plotly/matplotlib/mermaid/latex/html
     │      Serializes to {mime, content} dict
     ▼
engine.py:  sys._zoomy_display_callback(cell)
     │
     ▼
pyodide-worker.js:  _zoomyDisplayBridge(cellJson)
     │              postMessage({type: "display", cell: cellJson})
     ▼
app.js:  addEventListener("message") handler
     │   Parses cell JSON
     ▼
app.js:  renderOutputCell(cell, container)
         Creates DOM element based on MIME type
         Renders: text/<pre>, mermaid.js, KaTeX, Plotly, SVG, HTML
```

**CPython fallback:** When `sys._zoomy_display_callback` is not set (outside Pyodide), `display()` prints a text representation. No code changes needed for scripts to work in both environments.
