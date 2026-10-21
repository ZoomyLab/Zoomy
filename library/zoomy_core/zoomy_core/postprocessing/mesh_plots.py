"""Unstructured-mesh postprocessing (2D/3D) — companion to column_plots.

Adopts the zoomy_plotting base (`SimulationStore` + `MatplotlibPlotter`,
the same machinery the Zoomy GUI uses) and adds what the coupling/thesis
figures need: a VTK-series reader producing a `SimulationStore`, segment
and plane slicing, the slice -> ColumnField bridge (a slice of a reduced
run lifted through the model IS a column field, so every column_plots
function works on slices), and a pure-matplotlib 3D viewpoint chooser.

pyvista is used for FILE LOADING and GEOMETRIC slicing only — all
rendering is matplotlib (headless-safe, style-governed).

Layer 1 (tools)
---------------
- `store_from_vtk`      pvd / vtk.series / single vtu -> SimulationStore
- `plotter`             store -> styled MatplotlibPlotter (1D/2D/3D)
- `mark_slice`          slice-segment marker (cf. mark_stations)
- `slice_store`         sample cell fields along a segment (nearest cell)
- `slice_plane`         plane cut of a 3D store -> in-plane 2D store
- `columns_from_slice`  slice of a reduced run -> ColumnField (lifted
  through the MODEL'S interpolate_to_3d, like column_plots.read_zoomyfoam)
- `preview_3d`          viewpoint chooser: grid of (elev, azim) views
- `plot_topo_field`     2D plan view: gray topography + field on top
- `plot_topo_field_3d`  3D surfaces: gray topo + colored free surface

Layer 2 (composed figures)
--------------------------
- `fig_mesh_slice`      field map + marked slice + lifted water column
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from zoomy_plotting.store import SimulationStore, Zstruct
from zoomy_plotting.plot.matplotlib import MatplotlibPlotter

from . import style
from .plotting import list_series_frames

__all__ = [
    "store_from_vtk", "plotter", "mark_slice", "slice_store", "slice_plane",
    "columns_from_slice", "preview_3d", "plot_topo_field",
    "plot_topo_field_3d", "fig_mesh_slice",
]


# ── VTK reader -> SimulationStore ────────────────────────────────────────────

#: VTK cell code -> (cell-type name, topological dimension)
_VTK_CELLS = {
    3: ("line", 1),
    5: ("triangle", 2),
    9: ("quad", 2),
    10: ("tetra", 3),
    12: ("hexahedron", 3),
    13: ("wedge", 3),
}

_NAME_TO_CODE = {name: code for code, (name, _) in _VTK_CELLS.items()}


def store_from_vtk(path) -> SimulationStore:
    """Read a pvd / vtk.series / single vtu|vtk file into a SimulationStore.

    pyvista handles the file formats (incl. VTU 2.2 that meshio rejects);
    point data is averaged onto cells so every field is cell-centered.
    Frames load lazily (one full frame cached per visited time step).
    """
    import pyvista as pv

    paths, times = list_series_frames(path)
    base = Path(path).parent
    paths = [p if Path(p).is_absolute() else str(base / p) for p in paths]

    g0 = pv.read(paths[0])
    counts = {c: len(conn) for c, conn in g0.cells_dict.items()
              if c in _VTK_CELLS}
    if not counts:
        raise ValueError(
            f"no supported cell type in {paths[0]} "
            f"(found VTK codes {sorted(g0.cells_dict)}, "
            f"supported {sorted(_VTK_CELLS)})")
    code = max(counts, key=counts.get)
    cell_type, dim = _VTK_CELLS[code]
    celltypes = np.asarray(g0.celltypes)
    mask = celltypes == code
    cells = np.asarray(g0.cells_dict[code], dtype=int)
    vertices = np.asarray(g0.points, dtype=float)[:, :dim]

    def _frame_data(g):
        """All fields of one frame as cell arrays restricted to `mask`."""
        data = {k: np.asarray(g.cell_data[k], dtype=float)
                for k in g.cell_data.keys()}
        if len(g.point_data.keys()):
            gc = g.point_data_to_cell_data()
            for k in g.point_data.keys():
                data.setdefault(k, np.asarray(gc.cell_data[k], dtype=float))
        return {k: v[mask] for k, v in data.items()}

    first = _frame_data(g0)
    # Vector fields expand to per-component entries (reader contract: 1-D).
    index = []                       # field index -> (raw name, component)
    names = []
    for nm in sorted(first):
        arr = first[nm]
        if arr.ndim == 1:
            index.append((nm, None))
            names.append(nm)
        else:
            for c in range(arr.shape[1]):
                index.append((nm, c))
                names.append(f"{nm}_{'xyz'[c] if c < 3 else c}")
    field = Zstruct({nm: i for i, nm in enumerate(names)})

    cache = {0: first}

    def _read_cell(ts: int, idx: int) -> np.ndarray:
        if ts not in cache:
            cache[ts] = _frame_data(pv.read(paths[ts]))
        nm, comp = index[idx]
        arr = cache[ts][nm]
        return arr if comp is None else arr[:, comp]

    return SimulationStore(
        dim=dim, cell_type=cell_type, vertices=vertices, cells=cells,
        times=np.asarray(times, dtype=float), field=field,
        _cell_reader=_read_cell, source_path=str(path))


def plotter(store: SimulationStore) -> MatplotlibPlotter:
    """The styled mesh plotter (`plot_1d/plot_2d/plot_3d/plot`)."""
    return MatplotlibPlotter(store)


# ── slicing ──────────────────────────────────────────────────────────────────

def mark_slice(ax, p0, p1, color=None, label=None, ls="--"):
    """Draw a slice-segment marker (the 2D analogue of mark_stations)."""
    color = color or style.COLORS["interface"]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, ls=ls, marker="")
    if label:
        ax.annotate(label, xy=(p1[0], p1[1]), xytext=(-3, 5),
                    textcoords="offset points", color=color, ha="right")
    return ax


def slice_store(store, fields, p0, p1, time_step=0, n=100):
    """Sample cell fields along the segment p0 -> p1 (nearest cell center).

    Works in 2D and 3D (the dimension is taken from len(p0)). Returns
    `(s, {name: values})` with `s` the arc length along the slice.
    """
    from scipy.spatial import cKDTree
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = len(p0)
    tree = cKDTree(store.cell_centers[:, :d])
    s = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    _, idx = tree.query(pts)
    out = {nm: np.asarray(store.get_cell(time_step, nm))[idx]
           for nm in fields}
    return s * float(np.linalg.norm(p1 - p0)), out


def slice_plane(store, origin, normal, time_step=0, fields=None):
    """Plane cut of a 3D store -> a 2D SimulationStore in plane coordinates.

    pyvista does the geometric cut (no rendering); parent-cell data rides
    along, so `plotter(slice_plane(...)).plot_2d` gives the in-plane
    field map. Axes are (s1, s2): an orthonormal in-plane basis.
    """
    import pyvista as pv
    if store.dim != 3:
        raise ValueError(f"slice_plane needs a 3D store, got dim={store.dim}")
    fields = list(fields) if fields is not None else list(store.field.keys())

    code = _NAME_TO_CODE[store.cell_type]
    pts3 = np.zeros((store.n_vertices, 3))
    pts3[:, :store.dim] = store.vertices
    grid = pv.UnstructuredGrid({code: store.cells}, pts3)
    for nm in fields:
        grid.cell_data[nm] = np.asarray(store.get_cell(time_step, nm))

    cut = grid.slice(normal=normal, origin=origin).triangulate()
    if cut.n_cells == 0:
        raise ValueError("plane misses the mesh: empty slice")
    tris = cut.faces.reshape(-1, 4)[:, 1:]

    # In-plane basis aligned with the global axes (so a y-normal cut plots
    # as (x, z), not some rotated/mirrored frame).
    nvec = np.asarray(normal, dtype=float)
    nvec /= np.linalg.norm(nvec)
    a = np.eye(3)[int(np.argmin(np.abs(nvec)))]
    e1 = a - (a @ nvec) * nvec
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nvec, e1)
    if e2[int(np.argmax(np.abs(e2)))] < 0:
        e2 = -e2
    rel = np.asarray(cut.points) - np.asarray(origin, dtype=float)[None, :]
    verts2 = np.stack([rel @ e1, rel @ e2], axis=1)

    def _axname(e):
        i = int(np.argmax(np.abs(e)))
        return "xyz"[i] if abs(e[i]) > 0.99 else "s"

    data = {nm: np.asarray(cut.cell_data[nm], dtype=float) for nm in fields}
    field = Zstruct({nm: i for i, nm in enumerate(fields)})
    t0 = (float(store.times[time_step])
          if store.times is not None else 0.0)

    def _read_cell(ts, idx):
        return data[fields[idx]]

    return SimulationStore(
        dim=2, cell_type="triangle", vertices=verts2, cells=tris,
        times=np.asarray([t0]), field=field, _cell_reader=_read_cell,
        source_path=store.source_path,
        extras={"origin": np.asarray(origin, float), "normal": nvec,
                "basis": (e1, e2),
                "axis_labels": (_axname(e1), _axname(e2))})


# ── slice -> ColumnField (the dimensionally-reduced view IN the slice) ──────

def _state_field_names(store, model, state_prefix):
    nq = len(model.state)
    for prefix in (state_prefix, "Q", "q"):
        names = [f"{prefix}{i}" for i in range(nq)]
        if all(nm in store.field for nm in names):
            return names
    have = sorted(store.field.keys())
    raise KeyError(
        f"state fields {state_prefix}0..{state_prefix}{nq - 1} not in store "
        f"(available: {have})")


def columns_from_slice(store, model, p0, p1, time_step=0, n=100, K=24,
                       state_prefix="Q", label=""):
    """Slice a REDUCED-model run and lift it through the model's own
    `interpolate_to_3d` -> `ColumnField`.

    The state is read from cell fields `Q0..Qn` (zoomy VTK convention;
    the hdf5 reader's `q0..qn` is detected too). Every column_plots
    function (water, profiles, stations) then works on the slice.
    """
    from .column_plots import ColumnField, CANONICAL
    import sympy as sp

    names = _state_field_names(store, model, state_prefix)
    nq = len(names)
    s, data = slice_store(store, names, p0, p1, time_step=time_step, n=n)
    Q = np.array([data[nm] for nm in names])          # (nq, n)

    zeta = (np.arange(K) + 0.5) / K

    def _unwrap(e):
        while hasattr(e, "__len__") and not hasattr(e, "free_symbols"):
            e = e[0]
        return sp.sympify(e)

    rows = [_unwrap(e) for e in model.interpolate_to_3d]
    syms = (list(model.state)
            + list(getattr(model, "aux_state", []) or [])
            + list(model.parameters) + [sp.Symbol("z")])
    fns = [sp.lambdify(syms, r, "numpy") for r in rows]
    pvals_raw = model.parameter_values
    pvals = [float(v) for v in (pvals_raw.values()
                                if hasattr(pvals_raw, "values") else pvals_raw)]
    naux = len(getattr(model, "aux_state", []) or [])

    fields = {nm: np.zeros((1, n, K)) for nm in CANONICAL}
    args = [Q[q][:, None] for q in range(nq)]
    args += [np.zeros((n, 1))] * naux
    args += [np.full((n, 1), v) for v in pvals]
    args += [zeta[None, :]]
    for ci, fn in enumerate(fns[:len(CANONICAL)]):
        val = np.squeeze(np.asarray(fn(*args), dtype=float))
        if val.ndim == 0:
            val = np.full((n, K), float(val))
        elif val.ndim == 1:
            val = (np.repeat(val[:, None], K, axis=1) if len(val) == n
                   else np.repeat(val[None, :], n, axis=0))
        fields[CANONICAL[ci]][0] = val
    t0 = (float(store.times[time_step])
          if store.times is not None else 0.0)
    return ColumnField(np.array([t0]), s, zeta, fields, label)


# ── 3D viewpoint chooser (pure matplotlib) ───────────────────────────────────

def preview_3d(fig, store, views=((20, -60), (20, 30), (20, 120), (60, -45)),
               field=0, time_step=0, **kw):
    """Viewpoint chooser: the mesh (edges on) from several (elev, azim)
    angles, so a final view can be picked for `plot_3d(viewpoint=...)`."""
    pl = plotter(store)
    nv = len(views)
    for k, view in enumerate(views):
        ax = fig.add_subplot(1, nv, k + 1, projection="3d")
        pl.plot_3d(ax, time_step=time_step, field=field, show_mesh=True,
                   viewpoint=view, colorbar=False, **kw)
        ax.set_title(f"elev={view[0]}, azim={view[1]}")
    return fig


# ── topography + flow-feature overlays ───────────────────────────────────────

def _norm_and_cmap(values, cmap, vlim):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    vmin, vmax = (vlim if vlim is not None
                  else (float(np.min(values)), float(np.max(values))))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap)


def _vertex_values(store, cell_values):
    from zoomy_plotting.mesh.faces import cell_to_vert_values
    return cell_to_vert_values(store.n_vertices, store.cells,
                               np.asarray(cell_values, dtype=float))


def _topo_contours(ax, store, topo_vert, contours, **kw):
    """Contour lines of the topography (vertex-averaged, fan-triangulated).
    `contours`: int (number of levels) or an explicit level sequence."""
    import matplotlib.tri as mtri
    from zoomy_plotting.mesh.faces import triangulate
    ii, jj, kk, _ = triangulate(store.cells)
    tri = mtri.Triangulation(store.vertices[:, 0], store.vertices[:, 1],
                             np.stack([ii, jj, kk], axis=1))
    levels = int(contours) if isinstance(contours, (int, np.integer)) \
        else list(contours)
    return ax.tricontour(tri, topo_vert, levels=levels,
                         colors=style.COLORS["reference"],
                         linewidths=0.8 * _lw(), **kw)


def _lw():
    import matplotlib as mpl
    return mpl.rcParams["lines.linewidth"]


def plot_topo_field(ax, store, topo, field, time_step=0, hide_below=None,
                    topo_cmap=None, cmap=None, vlim=None, topo_vlim=None,
                    colorbar=True, topo_colorbar=False, contours=None,
                    alpha=None):
    """Plan view: bottom topography in grayscale, `field` colored on top.

    - `topo`/`field`: cell-field names (e.g. "Q0"/"Q1").
    - `hide_below`: cells with field <= threshold are NOT drawn, so the
      gray topography shows through (dry-area masking).
    - `colorbar` (field, right) and `topo_colorbar` (grayscale, left)
      are independent; `contours` adds topography iso-lines (int = number
      of levels, or an explicit level list).
    """
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection
    if store.dim != 2:
        raise ValueError(f"plot_topo_field needs a 2D store, got "
                         f"dim={store.dim}")
    b = np.asarray(store.get_cell(time_step, topo), dtype=float)
    f = np.asarray(store.get_cell(time_step, field), dtype=float)
    verts2 = store.vertices[:, :2]
    polys = [verts2[c] for c in store.cells]

    tnorm, tcmap = _norm_and_cmap(b, topo_cmap or style.CMAP_TOPO, topo_vlim)
    ax.add_collection(PolyCollection(
        polys, facecolors=tcmap(tnorm(b)), edgecolors="none"))

    keep = np.ones(len(f), dtype=bool) if hide_below is None \
        else f > float(hide_below)
    fnorm, fcmap = _norm_and_cmap(f[keep] if keep.any() else f,
                                  cmap or style.CMAP_CONTINUOUS, vlim)
    if keep.any():
        ax.add_collection(PolyCollection(
            [p for p, k in zip(polys, keep) if k],
            facecolors=fcmap(fnorm(f[keep])), edgecolors="none",
            alpha=alpha))

    out = {}
    if contours is not None:
        out["contours"] = _topo_contours(ax, store, _vertex_values(store, b),
                                         contours)
    # The field ScalarMappable is always exposed (independent of ``colorbar``)
    # so overlay helpers — e.g. ``overlays.inset_wedge_colorbar`` — can attach
    # a colorbar even when the default right-side one is suppressed.
    sm = mpl.cm.ScalarMappable(cmap=fcmap, norm=fnorm)
    sm.set_array([])
    out["mappable"] = sm
    if colorbar:
        out["colorbar"] = ax.figure.colorbar(
            sm, ax=ax, shrink=style.CONFIG.colorbar_shrink,
            pad=style.CONFIG.colorbar_pad, label=str(field))
    if topo_colorbar:
        sm = mpl.cm.ScalarMappable(cmap=tcmap, norm=tnorm)
        sm.set_array([])
        out["topo_colorbar"] = ax.figure.colorbar(
            sm, ax=ax, location="left",
            shrink=style.CONFIG.colorbar_shrink,
            pad=style.CONFIG.colorbar_pad, label=str(topo))
    ax.autoscale()
    ax.set_aspect("equal")
    return out


def plot_topo_field_3d(ax, store, topo, field, time_step=0, surface=None,
                       hide_below=None, topo_cmap=None, cmap=None, vlim=None,
                       topo_vlim=None, colorbar=True, topo_colorbar=False,
                       contours=None, viewpoint="auto", alpha=None):
    """Perspective view of a 2D run: gray topography surface z=b, plus the
    surface z=`surface` (default topo+field, i.e. the free surface b+h)
    colored by `field`.  Same options as `plot_topo_field`;
    `contours` draws topography iso-lines at their true height.
    `viewpoint`: "auto" | (elev, azim)."""
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from zoomy_plotting.mesh.faces import triangulate
    from zoomy_plotting.plot._common import resolve_viewpoint
    if store.dim != 2:
        raise ValueError(f"plot_topo_field_3d lifts a 2D store, got "
                         f"dim={store.dim}")
    b = np.asarray(store.get_cell(time_step, topo), dtype=float)
    f = np.asarray(store.get_cell(time_step, field), dtype=float)
    z = b + f if surface is None \
        else np.asarray(store.get_cell(time_step, surface), dtype=float)

    ii, jj, kk, parents = triangulate(store.cells)
    tris = np.stack([ii, jj, kk], axis=1)
    xy = store.vertices[:, :2]
    bv, zv = _vertex_values(store, b), _vertex_values(store, z)

    def _surf(zvert, cell_vals, cmap_, norm_, alpha_):
        coords = [np.column_stack([xy[t, 0], xy[t, 1], zvert[t]])
                  for t in tris]
        return Poly3DCollection(
            coords, facecolors=cmap_(norm_(cell_vals[parents])),
            edgecolors="none", alpha=alpha_)

    tnorm, tcmap = _norm_and_cmap(b, topo_cmap or style.CMAP_TOPO, topo_vlim)
    ax.add_collection3d(_surf(bv, b, tcmap, tnorm, None))

    keep = np.ones(len(f), dtype=bool) if hide_below is None \
        else f > float(hide_below)
    fnorm, fcmap = _norm_and_cmap(f[keep] if keep.any() else f,
                                  cmap or style.CMAP_CONTINUOUS, vlim)
    if keep.any():
        wet = keep[parents]
        coords = [np.column_stack([xy[t, 0], xy[t, 1], zv[t]])
                  for t, w in zip(tris, wet) if w]
        ax.add_collection3d(Poly3DCollection(
            coords, facecolors=fcmap(fnorm(f[parents[wet]])),
            edgecolors="none", alpha=alpha))

    out = {}
    if contours is not None:
        import matplotlib.tri as mtri
        levels = int(contours) if isinstance(contours, (int, np.integer)) \
            else list(contours)
        out["contours"] = ax.tricontour(
            mtri.Triangulation(xy[:, 0], xy[:, 1], tris), bv, levels=levels,
            colors=style.COLORS["reference"], linewidths=0.8 * _lw())
    if colorbar:
        sm = mpl.cm.ScalarMappable(cmap=fcmap, norm=fnorm)
        sm.set_array([])
        out["colorbar"] = ax.figure.colorbar(
            sm, ax=ax, shrink=style.CONFIG.colorbar_shrink,
            pad=style.CONFIG.colorbar_pad, label=str(field))
    if topo_colorbar:
        sm = mpl.cm.ScalarMappable(cmap=tcmap, norm=tnorm)
        sm.set_array([])
        out["topo_colorbar"] = ax.figure.colorbar(
            sm, ax=ax, location="left",
            shrink=style.CONFIG.colorbar_shrink,
            pad=style.CONFIG.colorbar_pad, label=str(topo))

    ax.set_xlim(float(xy[:, 0].min()), float(xy[:, 0].max()))
    ax.set_ylim(float(xy[:, 1].min()), float(xy[:, 1].max()))
    zall = np.concatenate([bv, zv])
    ax.set_zlim(float(zall.min()), float(zall.max()))
    elev, azim = resolve_viewpoint(viewpoint, store.vertices)
    ax.view_init(elev=elev, azim=azim)
    return out


# ── layer 2 ──────────────────────────────────────────────────────────────────

def fig_mesh_slice(fig, store, model, p0, p1, field="Q1", time_step=0,
                   n=120, K=16, slice_label="slice", title=None):
    """Field map with the slice marked + the lifted water column along it."""
    from .column_plots import plot_water_columns
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    plotter(store).plot_2d(ax, time_step=time_step, field=field)
    mark_slice(ax, p0, p1, label=slice_label)
    if title:
        ax.set_title(title)

    a2 = fig.add_subplot(gs[1])
    cf = columns_from_slice(store, model, p0, p1, time_step=time_step,
                            n=n, K=K)
    plot_water_columns(a2, cf, float(cf.t[0]), style="fill",
                       color=style.COLORS["water"])
    a2.plot(cf.x, cf.fields["b"][0, :, 0],
            color=style.COLORS["reference"], marker="")
    a2.set_xlabel("s along slice")
    a2.set_ylabel("b, b+h")
    style.figure_legend(fig, extra=[
        (slice_label, style.line("interface", ls="--")),
        ("water", style.line("water", lw=5)),
    ])
    return fig
