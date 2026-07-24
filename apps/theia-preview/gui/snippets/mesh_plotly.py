"""Plotly mesh viewer — interactive 1D/2D/3D rendering of the current store.

Reads cell data from the ``zoomy_plotting.SimulationStore`` installed in
scope as ``store``. No fallback — raises if the store is missing or the
requested field/time step are invalid. Plotly has no counterpart in
``zoomy_plotting`` yet, so the snippet builds the figures directly
against the store's public API (``get_cell``, ``vertices``, ``cells``).

The GUI injects ``time_step`` (from the timeline slider) and
``field_name`` (from the field selector) before exec.
"""
from collections import Counter

import numpy as np
import plotly.graph_objects as go

import zoomy_plotting as zp

if store is None:
    raise RuntimeError(
        "store is not populated — run a simulation first. The solver "
        "template writes /tmp/zoomy_sim/sim.h5 and calls open_hdf5()."
    )
if not isinstance(store, zp.SimulationStore):
    raise TypeError(
        f"store must be zoomy_plotting.SimulationStore, got "
        f"{type(store).__name__}"
    )

field_name = field_name if "field_name" in dir() and field_name else next(iter(store.field.keys()))
time_step = int(time_step) if "time_step" in dir() else 0
colormap = "Viridis"

time_step = max(0, min(time_step, store.n_snapshots - 1))
values = np.asarray(store.get_cell(time_step, field_name))
vertices = np.asarray(store.vertices)
cells = np.asarray(store.cells)
dim = store.dim
# Prefer real simulation time in the title; fall back to step number.
if store.times is not None and len(store.times):
    t_label = f"t = {float(store.times[time_step]):.3f}"
else:
    t_label = f"step {time_step}"


def _triangulate(cells_arr):
    ii, jj, kk = [], [], []
    for cell in cells_arr:
        if len(cell) < 3:
            continue
        for t in range(1, len(cell) - 1):
            ii.append(int(cell[0]))
            jj.append(int(cell[t]))
            kk.append(int(cell[t + 1]))
    return ii, jj, kk


def _boundary_faces_3d(cells_arr):
    n_per = cells_arr.shape[1]
    if n_per == 4:
        face_defs = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif n_per == 8:
        face_defs = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                     [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
    else:
        raise ValueError(f"unsupported 3-D cell size: {n_per}")
    count, fmap, parent = Counter(), {}, {}
    for ci, cell in enumerate(cells_arr):
        for fd in face_defs:
            nodes = [int(cell[k]) for k in fd]
            key = tuple(sorted(nodes))
            count[key] += 1
            fmap[key] = nodes
            parent[key] = ci
    faces, parents = [], []
    for k, c in count.items():
        if c == 1:
            faces.append(fmap[k])
            parents.append(parent[k])
    return faces, parents


def _cell_to_vert_values(n_vert, cells_arr, cell_values):
    vv = np.zeros(n_vert)
    vc = np.zeros(n_vert)
    for ci, cell in enumerate(cells_arr):
        val = float(cell_values[ci])
        for vi in cell:
            vv[int(vi)] += val
            vc[int(vi)] += 1
    vc[vc == 0] = 1
    return vv / vc


if dim == 1:
    x = vertices[cells].mean(axis=1)[:, 0]
    order = np.argsort(x)
    fig = go.Figure(go.Scatter(
        x=x[order].tolist(), y=values[order].tolist(),
        mode="lines+markers",
    ))
    fig.update_layout(
        title=f"{field_name}  —  {t_label}",
        xaxis_title="x", yaxis_title=field_name,
        margin=dict(l=40, r=20, t=40, b=40),
    )

elif dim == 2:
    vert_vals = _cell_to_vert_values(store.n_vertices, cells, values)
    ii, jj, kk = _triangulate(cells)
    z = [0.0] * store.n_vertices
    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(), z=z,
        i=ii, j=jj, k=kk,
        intensity=vert_vals.tolist(),
        colorscale=colormap, showscale=True,
        colorbar=dict(title=field_name), flatshading=False,
    ))
    fig.update_layout(
        title=f"{field_name}  —  {t_label}",
        scene=dict(aspectmode="data",
                   camera=dict(eye=dict(x=0, y=0, z=2.2),
                               up=dict(x=0, y=1, z=0))),
        margin=dict(l=0, r=0, t=40, b=0),
    )

elif dim == 3:
    faces, parents = _boundary_faces_3d(cells)
    if not faces:
        raise RuntimeError(
            f"no 3-D boundary faces extracted from {store.n_cells} cells "
            f"of type {store.cell_type!r}"
        )
    ii, jj, kk = _triangulate(np.asarray(faces))
    face_vals = np.array([float(values[int(p)]) for p in parents])
    vert_vals = _cell_to_vert_values(store.n_vertices, np.asarray(faces), face_vals)
    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(),
        z=vertices[:, 2].tolist(),
        i=ii, j=jj, k=kk,
        intensity=vert_vals.tolist(),
        colorscale=colormap, showscale=True,
        colorbar=dict(title=field_name),
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.8),
        lightposition=dict(x=100, y=200, z=300),
    ))
    fig.update_layout(
        title=f"{field_name}  —  {t_label}",
        scene=dict(aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        margin=dict(l=0, r=0, t=40, b=0),
    )

else:
    raise ValueError(f"unsupported store.dim={dim}")

print(f"[plotly] {store.cell_type} dim={dim} field={field_name!r} "
      f"snaps={store.n_snapshots} cells={store.n_cells}")

# Single output convention — display(fig) is the only way a snippet
# publishes a plot; the GUI clears the output list at the start of
# every run, so the figure shown is always just this one.
display(fig)
