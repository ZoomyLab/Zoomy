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

## Template — the case-visualization house style

`zoomy_plotting` is **THE** template every case figure uses. A case
`visualize.py` owns *zero* styling — no hardcoded font size, figure size,
color, linewidth, alpha, dpi, grid, or legend placement. All of it lives here
and is selected with one call.

### Size presets (switch the format, never the case)

Pick a medium; the fonts *and* the figure size come from it. Thesis-format
figures come from switching the preset, never from editing the case:

```python
import zoomy_plotting as zp

fig, axes = zp.subplots(1, 3, preset="publication")   # journal 9 pt
fig, axes = zp.subplots(1, 3, preset="thesis")        # thesis text column
fig, axes = zp.subplots(1, 3, preset="screen")        # slides / GIFs
```

`subplots` activates the preset globally (`zp.use`) and sizes the figure from
`SIZES` — per-axes *cell* × the `(ncols, nrows)` grid. For a fixed single-panel
width pass a token: `zp.subplots(preset="thesis", width="2col")`
(`"1col"` / `"2col"` / `"full"`). `zp.figsize(preset, ncols, nrows, width)`
returns the size without making a figure. Presets: `publication` (alias
`print`), `thesis`, `screen`, `presentation` (14 pt floor). `zp.use(preset)`
also reflows a bare `plt.subplots()` default size.

### Semantic colors (fixed) vs the data cycle

Three cross-figure roles stay identical in every figure — never cycle them:

```python
zp.colors.experiment   # measured data      (vermillion)
zp.colors.reference    # ground-truth bench (grey)
zp.colors.analytic     # closed-form / asymptotic (black)
zp.colors.cycle        # the Okabe–Ito rotation for model SERIES (SME levels …)
```

Model series ride the Okabe–Ito `prop_cycle` automatically. `zp.line_plot`
accepts `role=`/`color=` per series; `zp.colors["anything"]` falls through
`resolve_color`.

### Legend below + small setup titles

- One legend under the figure: `zp.figure_legend(fig)`; under a single row:
  `zp.row_legend(fig, axes_row)`. Never put `ax.legend(...)` in a case.
- Titles are **setup-only and small** — say *what the figure shows*, never a
  conclusion. Axis titles render at body size, the `fig.suptitle` one point up
  (`figure.titlesize`); the case passes no `fontsize`.

### Recurring shapes

`zp.subplots` (grid) · `zp.line_plot(ax, series)` (line / multi-line overlay) ·
`zp.profile_plot(ax, profiles)` (vertical profiles) · `zp.figure_legend` /
`zp.row_legend` (legend below) · `zp.animate(draw, frames, out)` (GIF/MP4 — the
per-frame composer draws under the active preset, so it needs no styling atoms
either). Reference: `examples/template_showcase.py`.

## Install

```bash
pip install -e .[testing]       # for tests
pip install -e .[plotly]        # optional plotly backend (future PR)
pip install -e .[pyvista]       # optional pyvista backend (future PR)
```
