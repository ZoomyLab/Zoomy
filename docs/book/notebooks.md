# Notebooks

## Run Zoomy in your browser

The full modeling stack — `Model` → `SystemModel` → `NumericalSystemModel` →
solve — runs in a **JupyterLite / Pyodide** notebook with nothing installed
locally.

::::{grid} 1
:gutter: 3

:::{grid-item-card} Open the Pyodide notebook
:link: jupyter-lite/_output/lab/index.html?path=pyodide.ipynb
:link-type: url
No installation, no backend — `zoomy_core` and `zoomy_plotting` are bundled into
the browser runtime.
:::

::::

The notebook source is `tutorials/pyodide/pyodide.ipynb`; the JupyterLite site
is built from `docs/jupyterlite/` on every deploy, with `zoomy-core` and
`zoomy-plotting` pinned into the Pyodide lock.

Because it runs in the browser, only the pure-Python NumPy solver is available.
For JAX, AMReX or OpenFOAM, install locally or use a
[container](installation.md).

## Structured tutorials

For the guided material — shallow water from first model to custom closure, and
the AMReX walkthrough — see [Tutorials](tutorials/swe.md).
