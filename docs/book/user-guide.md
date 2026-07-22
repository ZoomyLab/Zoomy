# User Guide

Zoomy provides two interfaces for setting up and running simulations:
a **web-based GUI** and a **command-line interface (CLI)**.
Both work from the same card-based configuration system and produce identical results.

Choose the interface that fits your workflow:

| | GUI | CLI |
|---|---|---|
| **Best for** | Exploring models, interactive debugging, sharing via URL | Batch runs, scripting, CI pipelines |
| **Runs in** | Browser (GitHub Pages, localhost) | Terminal (Node.js) |
| **Solver execution** | In-browser via Pyodide (NumPy), or a connected backend server | Local Python subprocess, or a connected backend server |

Neither needs an installation to get started: the GUI runs the NumPy solver
entirely in the browser. Other backends are reached by connecting to a
[backend container](installation.md#prebuilt-containers), which both interfaces
talk to over the same HTTP API.
