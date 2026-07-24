# Theia + Baukasten prototype (experiment)

```{warning}
Work-in-progress experiment. This is a **backend-less Eclipse Theia** running
entirely in your browser — the foundation for a future GUI that is the *same*
code on web, VS Code, Theia and Electron. It is being built in stages; this page
tracks what is live.
```

**[▶ Open the Theia preview](theia-preview/index.html)** — a full Theia
workbench (menu bar, activity bar, editor, status bar) with **no backend and no
server**. First load is a large download (Theia is heavy); give it a moment.

![Backend-less Theia](images/theia-preview.png)

## What this proves

Eclipse Theia's official **`browser-only`** target builds the whole IDE frontend
to **static files** — no Node backend, no editor host. It builds and runs here
(node 22, `theia build`), which settles the first open question: **a
backend-less Theia is viable now** (the browser-only work you previously had to
do by hand is upstream). The filesystem is backed by the browser (IndexedDB).

## The plan (three targets, one GUI)

The point is to write the Zoomy GUI **once** in [Baukasten](baukasten-preview.md)
and run it everywhere, changing only the host + one stylesheet:

- **web** — this backend-less Theia + Baukasten GUI + notebooks on a Pyodide kernel
- **local** — a VS Code / Theia extension: install it and you have the GUI
- **desktop** — a Theia Electron app

## Milestones

| | Status |
|---|---|
| 1. Backend-less Theia builds + runs (this page) | **live** |
| 2. Baukasten start page as the opening Theia view | building |
| 3. Click → Theia code editor, with a way back | building |
| 4. Native Theia notebook on a **Pyodide kernel**, running the classic Zoomy notebook | building |

The notebook (milestone 4) uses a Theia `NotebookController` driving the **same
Pyodide worker the GUI already runs** — so its jedi autocomplete and everything
built there carries over, rather than a throwaway kernel. It is Theia's *native*
notebook UI, not an embedded JupyterLite page.

Source: `apps/theia-preview/` (a standard `@theia/cli` browser-only app —
`npm install && npx theia build`). CI builds it non-blocking and serves it at
`/theia-preview/`.
