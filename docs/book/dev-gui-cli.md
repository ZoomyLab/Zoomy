# GUI and CLI Architecture

This section documents the architecture specific to the web GUI and CLI interfaces.

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

## How GUI and CLI Stay in Sync

`core.js` is the single source of truth, shared by both interfaces:

- `app.js` (GUI) calls `ZoomyCore.Project.fromConfig(config)` and uses `buildCase()`, `buildSaveData()`, `applySaveData()`
- `cli.js` (CLI) does `var ZoomyCore = require("../zoomy_gui/core.js")` and calls the same methods

Both load cards from the same `cards/` directory. Both serialize using the same ZIP format. A project saved by the GUI can be loaded by the CLI and vice versa.

## `display()` Pipeline

The `display()` function provides Jupyter-like output cells in the GUI code editor.

```
User code:  display(obj)
     |
engine.py:  ZoomyDisplay.__call__(obj)
     |      Detects type: text/plotly/matplotlib/mermaid/latex/html
     |      Serializes to {mime, content} dict
     |
engine.py:  sys._zoomy_display_callback(cell)
     |
pyodide-worker.js:  _zoomyDisplayBridge(cellJson)
     |              postMessage({type: "display", cell: cellJson})
     |
app.js:  renderOutputCell(cell, container)
         Creates DOM element based on MIME type
```

**CPython fallback:** When `sys._zoomy_display_callback` is not set (outside Pyodide), `display()` prints a text representation.

## Session Architecture

Each session stores its own card selections and parameter overrides:

```
session = {
    id, title, description,
    selections: { model: "card-id", mesh: "card-id", solver: "card-id" },
    cardOverrides: { "card-id": { params: {...}, code: "..." } }
}
```

Switching sessions snapshots the departing session's state and restores the arriving session. Save format v1.1 stores per-session data in the ZIP.

## Deployment

### GitHub Pages

The `render-webpage.yml` workflow copies `library/zoomy_gui/*` into the Jupyter Book build at `/gui/`. The GUI is accessible at `https://zoomylab.github.io/Zoomy/gui/`.

### URL-based project loading

The GUI supports `?project=` with three source types:
- Relative path: `?project=tutorials/dam-break.zip`
- Full URL: `?project=https://github.com/.../file.zip`
- Zenodo: `?project=zenodo:12345` or `?project=zenodo:12345/file.zip`
