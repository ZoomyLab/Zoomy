# Theia + Baukasten prototype (experiment)

```{warning}
Work-in-progress experiment. This is a **backend-less Eclipse Theia** running
entirely in your browser — the foundation for a future GUI that is the *same*
code on web, VS Code, Theia and Electron. It is being built in stages; this page
tracks what is live.
```

**[▶ Open the Theia prototype](theia-preview/index.html)** — it opens on a
**Baukasten start page**. From there: *Open code editor* drops you into Theia's
Monaco editor, and *Open Pyodide notebook* opens a **native Theia notebook**
that runs the classic Zoomy notebook on an **in-browser Pyodide kernel**. The
**🏠 Zoomy start** item in the status bar takes you back from anywhere. All of
this with **no backend and no server**. First load is a large download (Theia is
heavy) and the kernel installs `zoomy-core` on first run — give it a moment.

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
| 2. Baukasten start page as the opening Theia view | **live** |
| 3. Click → Theia code editor, with a way back | **live** |
| 4. Native Theia notebook on a **Pyodide kernel**, running the classic Zoomy notebook | **live** |

The Pyodide kernel is a thin wrapper today; the next step is to fold in the
**same worker the GUI already runs** — its jedi autocomplete, interrupts and
package set — so everything built there carries over rather than being a
throwaway kernel. The start page is styled with Theia/Baukasten theme tokens;
swapping in the real `baukasten-ui` React components (already proven on the
[Baukasten preview](baukasten-preview.md)) is the remaining polish.

Source: `apps/theia-preview/` — a standard `@theia/cli` browser-only app plus the
`zoomy-theia-ext/` native extension. CI builds it non-blocking and serves it at
`/theia-preview/`.
