# User Guide

Zoomy provides two interfaces for setting up and running simulations:
a **web-based GUI** and a **command-line interface (CLI)**.
Both work from the same card-based configuration system and produce identical results.

Choose the interface that fits your workflow:

| | GUI | CLI |
|---|---|---|
| **Best for** | Exploring models, interactive debugging, sharing via URL | Batch runs, scripting, CI pipelines |
| **Runs in** | Browser (GitHub Pages, localhost) | Terminal (Node.js) |
| **Solver execution** | In-browser (Pyodide) or Docker backend | Local Python subprocess or remote backend |
