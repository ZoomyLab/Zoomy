# In the browser

Nothing to install: the GUI and a JupyterLite notebook run the NumPy solver in the browser
(Pyodide).

- <a href="gui/index.html">The GUI</a>: compose and run a case, share it via URL
  ([Using the GUI](guide/gui.md)).
- <a href="jupyter-lite/_output/lab/index.html?path=pyodide.ipynb">JupyterLite notebook</a>: the
  whole stack, `Model` → `SystemModel` → `NumericalSystemModel` → C++ codegen → NumPy solve.
- <a href="playground/swe-game/index.html">SWE game</a>: a shallow-water game that runs in the
  page.

Other backends (JAX, AMReX, OpenFOAM) run on your machine: [Install](installation.md) or
[Containers](containers.md), then connect them to the GUI.
