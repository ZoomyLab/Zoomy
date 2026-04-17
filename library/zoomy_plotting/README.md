# zoomy-plotting

Publication-ready plotting for Zoomy simulation outputs, with first-class
support for 1D/2D/3D meshes and very large timeline datasets.

## Why

- **Lazy**: an HDF5 file with hundreds of snapshots opens cheaply. Only the
  cells needed for the current plot are read from disk.
- **Class-based, one `store` per session**. Pass the store once to a
  `MatplotlibPlotter`; plot methods take only `(ax, time_step, field, ...)`.
- **Config-driven styling**. Every color, linewidth, colormap, and marker
  size lives in `zoomy_plotting.CONFIG`. No visual parameter is hardcoded
  inside the plot functions.
- **One API across dimensions**. `plot_1d` / `plot_2d` / `plot_3d` share
  the same signature modulo the 3D viewpoint.

## Quick start

```python
import zoomy_plotting as zp
import matplotlib.pyplot as plt

store = zp.read_hdf5("simulation.h5")          # reads metadata only
plotter = zp.MatplotlibPlotter(store)

with zp.apply_style():
    fig, ax = plt.subplots()
    plotter.plot_1d(ax, time_step=0, field=store.field.h)
    plt.show()

store.close()
```

## Field references

All three forms resolve to the same integer index:

```python
plotter.plot_2d(ax, time_step=0, field=store.field.h)   # attribute on Zstruct
plotter.plot_2d(ax, time_step=0, field="h")             # string lookup
plotter.plot_2d(ax, time_step=0, field=0)               # raw index
```

## Styling

Three ways to restyle:

```python
# 1. per-call kwarg override
plotter.plot_2d(ax, time_step=0, field="h", cmap="plasma")

# 2. mutate the singleton config for the rest of the session
zp.CONFIG.cmap = "plasma"
zp.CONFIG.mesh_edgecolor = "gray"

# 3. context manager that reverts on exit
with zp.apply_style(cmap="plasma", mesh_edgecolor="gray"):
    plotter.plot_2d(ax, time_step=0, field="h")
```

## Install

```bash
pip install -e .[testing]       # for tests
pip install -e .[plotly]        # optional plotly backend (future PR)
pip install -e .[pyvista]       # optional pyvista backend (future PR)
```
