# Session Summary: Zoomy GUI Buildout (2026-04-02 to 2026-04-03)

## What was built (in order)

### 1. Panel GUI fixes
- Fixed missing `misc` import in `visu.py`
- Re-added `register_sympy_attribute` and `eigenvalue_dict_to_matrix` to `basemodel.py`
- Made mesh loading graceful when files missing

### 2. Standalone static webapp (`library/zoomy_gui/standalone/`)
- Replaced Panel-based GUI with lightweight static HTML/JS/CSS
- **Milligram CSS** (~2KB) for responsive layout
- **Pyodide** for in-browser Python execution (Web Worker for zero UI freeze)
- **Ace editor** for code editing
- **KaTeX** for math rendering in descriptions
- Card-based UI with CardManager system

### 3. Card architecture
- Unified `createCard()` function handles all card types: model, mesh, solver, visualization, session
- Feature flags per type: gear (params), edit (code), refresh (run), timeline, preview, maximize
- `CardManager` with layout modes: "stack" (full width), "grid" (multi-column)
- `collapseUnselected` for visualization tab (only selected card expands)
- Slot-based cards (`createSlotCard`) for dashboard
- Card variants via CSS classes (`card--log`, `card--backends`)

### 4. State management
- `core.js` — pure business logic (Card, Session, Project, save/load, selections), works in browser AND Node.js
- `cardState` / `cardDefaults` for tracking modifications
- `appState` with shared JSON state bridge
- Title/description editable via gear panel
- Code syncs to `cardState` on every editor change

### 5. Project save/load
- ZIP format: `session_name/tab/subtab/card_name/{card.json, code.py}`
- Only modified cards exported (delta from defaults)
- Import matches by card ID first, then by title
- JSZip library for browser ZIP handling
- Handles manually re-zipped files (finds project.json anywhere in tree)

### 6. Docker solver server (`containers/zoomy_solver_server/`)
- FastAPI REST API with CORS
- Endpoints: health, models, jobs (submit/poll/results/cancel)
- `SolverAdapter` pattern — one class with one `solve()` method to integrate any backend
- Built-in numpy and jax adapters
- `ProcessPoolExecutor` for job isolation
- Registry auto-scans zoomy_core model classes at startup
- Dockerfiles for numpy and jax backends
- Server tag configurable via `ZOOMY_SERVER_TAG` env var

### 7. `zoomy-server` package (`library/zoomy_server/`)
- PyPI-installable: `pip install zoomy-server`
- CLI: `zoomy-server --adapter numpy --port 8080`
- Integrated into zoomy_jax as default dependency

### 8. `zoomy-client` package (`library/zoomy_client/`)
- Pure Python HTTP client, zero dependencies (stdlib urllib)
- `ZoomyClient.submit()`, `.status()`, `.wait()`, `.results()`, `.cancel()`

### 9. Multi-backend connection system
- `backend.js` supports multiple simultaneous connections
- Each server has a `tag` (numpy, jax, amrex)
- Heartbeat every 5s, auto-disconnect on failure
- Solver cards show ✓/✗ connection status
- Navbar shows all connected backends
- Dashboard "Backends" card with ✕ disconnect buttons

### 10. Web Worker for Pyodide
- `pyodide-worker.js` — loads Pyodide in background thread
- Pre-installs all packages on startup (param, zoomy-core, numpy, plotly, matplotlib)
- Pre-extracts param schemas for all model cards (cache in worker)
- Zero UI freeze — ever

### 11. Debug log panel
- Dashboard shows scrollable dark log panel
- `logDebug(level, msg)` for errors, warnings, info
- Backend connection events, job status, param extraction timing

### 12. Node.js CLI (`library/zoomy_cli/`)
- `core.js` is single source of truth (shared with browser)
- Commands: start, overview, status, list, select, show, session, connect, disconnect, run, jobs, case, save, load
- `zoomy select model smt` / `zoomy select solver numpy` — tab then shortname
- `zoomy run --wait` — submit and poll with progress
- `zoomy jobs <id>` — check job status
- Shell completion for zsh and bash (auto-installed on `zoomy start`)
- Session management: new, switch, rename, list
- Backend connections persisted in `.zoomy/state.json`

### 13. `save_session()` in zoomy_core
- `from zoomy_core.misc import save_session`
- `save_session("dam_break", model=model, mesh=mesh, solver=solver)`
- Extracts class path, init params, physical params, code from live objects
- Produces ZIP compatible with UI and CLI
- Notebook → session one-way bridge

### 14. Dashboard improvements
- Slot-based cards with CardManager grid layout
- Live job tracking with progress bar and ETA
- `updateDashboardSummary()` shows selected model/mesh/solver
- Session card with gear for name editing

## Key files

### Standalone GUI
- `library/zoomy_gui/standalone/index.html` — app shell
- `library/zoomy_gui/standalone/style.css` — design tokens + all styles
- `library/zoomy_gui/standalone/core.js` — pure business logic (portable)
- `library/zoomy_gui/standalone/app.js` — DOM rendering + event handling
- `library/zoomy_gui/standalone/backend.js` — multi-backend HTTP client
- `library/zoomy_gui/standalone/pyodide-worker.js` — background Pyodide
- `library/zoomy_gui/standalone/param_widgets.js` — auto-widget generation from param schema
- `library/zoomy_gui/standalone/param_extract.py` — param introspection for Pyodide
- `library/zoomy_gui/standalone/engine.py` — code execution engine
- `library/zoomy_gui/standalone/cards.json` — card definitions

### Server
- `library/zoomy_server/` — PyPI package with FastAPI server + adapter pattern
- `library/zoomy_client/` — PyPI package with HTTP client
- `containers/zoomy_solver_server/` — Dockerfiles + legacy server code

### CLI
- `library/zoomy_cli/cli.js` — Node.js CLI
- `library/zoomy_cli/package.json` — npm package

### Core
- `library/zoomy_core/zoomy_core/misc/save_session.py` — notebook export

## Design decisions
- **JS is source of truth** for business logic (core.js), not Python
- **REST + SSE** for client-server communication (no gRPC, no WebSocket)
- **Pyodide in Web Worker** — never freezes UI
- **Lazy loading** — Pyodide/Ace/Plotly only load when needed
- **Delta-based save** — only modified cards in ZIP
- **Adapter pattern** for solver backends — one class to integrate any solver
- **No framework** — vanilla JS, Milligram CSS, no React/Vue

## How to run

```bash
# GUI
cd library/zoomy_gui/standalone && python -m http.server 8000

# Solver server
zoomy-server --adapter numpy --port 8080
# or: ZOOMY_SERVER_TAG=jax uvicorn server.main:app --port 8080

# CLI
cd library/zoomy_cli && npm install && npm link
zoomy start && zoomy overview

# Notebook
from zoomy_core.misc import save_session
save_session("my_sim", model=model, mesh=mesh)
```
