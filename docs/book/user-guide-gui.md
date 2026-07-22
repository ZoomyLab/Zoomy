# Using the GUI

## Opening the GUI

The Zoomy GUI is a static web application:

::::{grid} 2
:gutter: 3

:::{grid-item-card} Open the GUI
:link: gui/index.html
:link-type: url
Launch the Zoomy GUI in your browser.
:::

:::{grid-item-card} Open with Tutorial
:link: gui/index.html?project=tutorials/getting-started.zip
:link-type: url
Load the Getting Started project with three pre-configured test cases.
:::

::::

**Run locally** (from the repository):
```bash
cd library/zoomy_gui
python -m http.server 8000
# Open http://localhost:8000
```

## The Workflow

The simulation workflow consists of four steps:

### 1. Select a Model

Switch to the **Model** tab. Each card represents a mathematical model (e.g., Shallow Water Equations, Scalar Advection). Click a card to select it.

Click the **gear icon** to view and edit model parameters (e.g., polynomial level, dimension). Click the **pencil icon** to edit the Python code directly.

### 2. Select a Mesh

Switch to the **Mesh** tab. Choose a mesh type:
- **Create 1D** -- uniform 1D grid with configurable domain and cell count
- **Create 2D** -- structured 2D grid with configurable bounds and resolution
- **Create 3D** -- structured 3D grid

Edit mesh parameters via the gear icon (domain bounds, cell counts).

### 3. Select a Solver

Switch to the **Solver** tab. Available solvers:
- **NumPy** (Hyperbolic / IMEX / Split) -- run directly in the browser via
  Pyodide, always available, **no backend needed**
- **JAX**, **AMReX**, **DMPlex**, **OpenFOAM** -- require a connected backend
  server (see [Connecting to Backends](#connecting-to-backends))

Pick the solver that matches the model: hyperbolic models run on the plain
hyperbolic solver, while the non-hydrostatic VAM family needs the split
(Chorin) solver.

### 4. Run the Simulation

Click **Run simulation** in the sidebar. For the NumPy solvers the simulation
executes in-browser via Pyodide — nothing to install. Results (plots, output)
appear on the Dashboard tab.

For backend solvers, connect to a backend first (enter URL, click Connect), then
run. Job progress is polled automatically.

## Sessions

Sessions let you manage multiple configurations in one project. Each session stores its own card selections and parameter overrides.

- **Create a session**: click "+ New session" in the sidebar
- **Switch sessions**: click a session name in the sidebar -- selections and parameters are saved/restored automatically
- **Edit session name**: select a session, edit its name on the Dashboard

### Use case: comparison studies

Create one session per test case (e.g., "1D Dam Break", "2D Dam Break", "Fine mesh"). Switch between them to run and compare results without losing configuration state.

## The Code Editor and `display()`

When you open a card's code editor (pencil icon), you get an Ace editor with the Python source and an **output panel** below it.

### Manual execution

Click the **Run** button in the output panel toolbar to execute the editor code and see results.

### `display()` for rich output

Inside the editor code, use `display()` to send rich output to individual cells in the output panel:

```python
from zoomy_core.model.models import SME

model = SME(level=0)

# Text output
display(model.describe())

# Mermaid diagram
display(mermaid="graph TD; A[Model] --> B[Mesh] --> C[Solver] --> D[Results]")

# LaTeX equation
display(latex=r"\frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) = 0")

# Matplotlib figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 0])
display(fig)
```

Each `display()` call creates its own output cell. Supported types:
- **Text** -- plain strings, `model.describe()`, arrays
- **Mermaid diagrams** -- `display(mermaid="graph TD; ...")`
- **LaTeX math** -- `display(latex=r"\frac{a}{b}")`
- **Matplotlib figures** -- `display(fig)` (rendered as SVG)
- **Plotly figures** -- `display(fig)` (interactive)
- **HTML** -- `display(html="<table>...</table>")`

### Auto-run mode

Check the **auto** checkbox. When you finish typing a new `display()` call (balanced parentheses), the entire script re-runs automatically and all output cells refresh. This gives a notebook-like experience inside the code editor.

### Portability

`display()` is injected into the Pyodide scope automatically. When the same script runs in a normal Python environment (CLI, batch, server), `display` falls back to `print()` -- no code changes needed.

## Saving and Loading Projects

**Save**: click "Save project" in the sidebar. Downloads a `zoomy-project.zip` containing all sessions, card selections, parameters, and edited code.

**Load**: click "Load project", select a `.zip` file. Sessions and card states are restored.

## Loading Projects from URLs

The GUI supports loading projects via URL parameters:

| Source | URL format |
|--------|-----------|
| Same-origin (tutorials) | `?project=tutorials/getting-started.zip` |
| GitHub release | `?project=https://github.com/org/repo/releases/download/v1/project.zip` |
| Zenodo (with filename) | `?project=zenodo:12345/project.zip` |
| Zenodo (auto-detect) | `?project=zenodo:12345` |

Additional parameters:
- `?session=1D+Dam+Break` -- auto-switch to a named session after loading
- `#run` -- auto-execute the simulation after loading

**Example:** share a reproducible simulation with a colleague:
```
https://mbd-rwth.github.io/Zoomy/gui/?project=zenodo:12345/dam-break.zip&session=1D+Dam+Break#run
```

## Connecting to Backends

For solvers other than NumPy you need a running backend server. The simplest
route is the prebuilt container for that backend — every image serves the solver
API on `:8080` by default:

```bash
apptainer pull zoomy_jax.sif oras://ghcr.io/zoomylab/zoomy_jax_sif:latest
apptainer run zoomy_jax.sif 8080
# or: docker run --rm -p 8080:8080 ghcr.io/zoomylab/zoomy_jax:latest
```

From a source checkout instead: `zoomy-server --adapter jax --port 8080`.

Then, in the GUI sidebar, enter `http://localhost:8080` and click **Connect**.
Connected backends appear in the navbar, and solver cards that require them
become selectable. Multiple backends can be connected simultaneously.

See [Installation](installation.md) for the full container list.
