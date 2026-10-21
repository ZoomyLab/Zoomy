# GUI

How the page [Using the GUI](../guide/gui.md) is built.  Source: `library/zoomy_gui/`, a `@theia/cli`
browser-only app plus the `zoomy-theia-ext/` native extension and the `gui/` brain.  CI builds it
and serves it at `/gui/`.

Eclipse Theia's **`browser-only`** target builds the whole IDE frontend to **static files**: no
Node backend, no plugin host.  A small **native Theia extension** (`zoomy-theia-ext`) registers
everything the standard way: the card GUI view, a native notebook + in-browser Pyodide
`NotebookKernel`, and a DOM output surface for text, matplotlib and rich `describe()` output.  The
kernel reuses the Zoomy GUI "brain" (`zoomy_cli`) for the card catalog, case composition, param
extraction and backend submission.

The goal is **one GUI, three targets**: the same code on the **web** (this backend-less build), as a
**VS Code / Theia extension** (local), and as a **Theia Electron app** (desktop), changing only the
host and one stylesheet.
