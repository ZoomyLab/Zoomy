# Theia + Baukasten prototype (experiment)

```{warning}
Work-in-progress experiment. This is a **backend-less Eclipse Theia** running
entirely in your browser — the foundation for a future GUI that is the *same*
code on web, VS Code, Theia and Electron. It is being built in stages; this page
tracks what is live.
```

**[▶ Open the Theia prototype](theia-preview/index.html)** — it opens on a
**Baukasten start page** in a clean, full-window GUI (the VS Code chrome — menus,
activity bar, tabs, status bar — is hidden). *Open code editor* drops you into
Theia's Monaco editor with the full IDE revealed; *Open Pyodide notebook* opens a
**native Theia notebook** that runs the classic Zoomy notebook on an **in-browser
Pyodide kernel**. The **🏠 Zoomy start** item in the status bar returns you to the
GUI from anywhere. Python surfaces have **jedi autocomplete** (type `np.` or, after
a cell runs, `model.`). All of this with **no backend and no server**. First load
is a large download (Theia is heavy) and the kernel warms `zoomy-core` + jedi in
the background — give it a moment.

![Backend-less Theia](images/theia-preview.png)

## What this proves

Eclipse Theia's official **`browser-only`** target builds the whole IDE frontend
to **static files** — no Node backend, no editor host, and (crucially) no plugin
host. So VS Code–style `contributes.*` plugins do not run here. Instead, a small
**native Theia extension** registers everything the standard way: a notebook
serializer + type, an in-browser Pyodide `NotebookKernel`, and — because the
iframe output webview is a backend feature — a DOM output surface that renders
text, matplotlib PNGs and rich `describe()` output directly under each cell.

The kernel installs `zoomy-core` + `zoomy-plotting` from PyPI with micropip,
exactly like the JupyterLite page, and runs the *same* classic notebook
(`SME(level=2)` → `SystemModel` → C++ codegen → NumPy solve → vertical velocity
profile). It is Theia's **native** notebook UI, not an embedded JupyterLite page.

## The plan (three targets, one GUI)

The point is to write the Zoomy GUI **once** in [Baukasten](baukasten-preview.md)
and run it everywhere, changing only the host + one stylesheet:

- **web** — this backend-less Theia + Baukasten GUI + notebooks on a Pyodide kernel
- **local** — a VS Code / Theia extension: install it and you have the GUI
- **desktop** — a Theia Electron app

## Milestones

| | Status |
|---|---|
| 1. Backend-less Theia builds + runs | **live** |
| 2. Baukasten start page as the opening full-window GUI view | **live** |
| 3. Click → Theia code editor (full IDE), with a way back | **live** |
| 4. Native Theia notebook on a **Pyodide kernel**, running the classic Zoomy notebook | **live** |
| 5. Pyodide off the main thread (Web Worker) + jedi autocomplete | **live** |
| 6. App view vs IDE view — chrome only appears when editing code | **live** |

The kernel now runs in a **Web Worker** so the UI never blocks, and reuses the
Zoomy GUI's proven machinery: tiered background installs warmed at boot, a parso
AST cache on IndexedDB (jedi cold-start drops from ~20 s to <1 s on the 2nd
visit), and the GUI's `complete_code` (jedi) driving a Monaco completion provider
across the editor and every notebook cell. Next: fold in the rest of the GUI's
worker (interrupts, the full package set) and swap the start page's theme-token
styling for the real `baukasten-ui` React components (already proven on the
[Baukasten preview](baukasten-preview.md)).

Source: `apps/theia-preview/` — a standard `@theia/cli` browser-only app plus the
`zoomy-theia-ext/` native extension. CI builds it non-blocking and serves it at
`/theia-preview/`.
