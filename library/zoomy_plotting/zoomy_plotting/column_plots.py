"""Column-field postprocessing: one data object, disk-based readers, six plots.

The canonical plotting format IS the coupling contract: every solver exposes
its state through ``interpolate_to_3d`` as water-column profiles
``[b, h, u, v, w, p]`` on a unit-zeta grid.  Everything here consumes that.

Data object
-----------
``ColumnField``: t (T,), x (X,), zeta (K,), fields name -> (T, X, K).

Readers (always from disk — solvers write, plots read)
------------------------------------------------------
- :func:`read_columns`       universal ``columns/t_*.csv`` (written by the
  coupling adapters / solvers; exactly the exchanged data).
- :func:`read_zoomyfoam`     zoomyFoam Q-frames lifted through the MODEL'S
  OWN symbolic ``interpolate_to_3d`` (no re-implementation of the ansatz).
- :func:`read_vof_raw`       raw alpha/U frames of an OpenFOAM VoF case:
  alpha field for :func:`plot_water_vof`, plus column resampling at chosen
  stations for the profile/xt functions.

Plot functions (source-type-free; differentiation lives in the readers)
-----------------------------------------------------------------------
- :func:`plot_water_columns`  water region/surface from column data
- :func:`plot_water_vof`      raw alpha-field rendering (VOF only)
- :func:`plot_profiles`       u(zeta) at a station, multiple sources overlaid
- :func:`plot_xt`             x-t map of a scalar (transforms: raw,
  plateau-deviation, difference-of-two-sources)
- :func:`plot_series`         time series of scalar reductions (+ read_ledger)
- :func:`animate`             panel compositor -> gif
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ColumnField", "read_columns", "read_zoomyfoam", "read_vof_raw",
    "read_ledger", "read_of_frames", "read_of_field", "read_of_states",
    "concat_columns", "mass_of_columns", "mass_of_vof",
    "plot_water_columns", "plot_water_vof", "plot_profiles",
    "plot_xt", "plot_series", "mark_stations", "frame_color",
    "fig_coupling", "fig_reduced_coupling", "fig_mass_deviation", "animate",
]

CANONICAL = ("b", "h", "u", "v", "w", "p")


@dataclass
class ColumnField:
    """Water-column profiles on a unit-zeta grid: fields[name] -> (T, X, K)."""
    t: np.ndarray
    x: np.ndarray
    zeta: np.ndarray
    fields: dict = field(default_factory=dict)
    label: str = ""

    def at(self, tq):
        """Frame index nearest to time ``tq``."""
        return int(np.argmin(np.abs(self.t - tq)))

    def station(self, xq):
        """Column index nearest to position ``xq``."""
        return int(np.argmin(np.abs(self.x - xq)))


# ── readers ──────────────────────────────────────────────────────────────────

def _frame_dirs(case, fname):
    case = Path(case)
    out = []
    for d in case.iterdir():
        if d.is_dir() and re.fullmatch(r"[0-9]+(\.[0-9]+)?([eE][-+][0-9]+)?",
                                       d.name) and (d / fname).exists():
            out.append((float(d.name), d))
    return sorted(out)


def _read_internal(path, n):
    txt = open(path).read()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\)\s*;", txt, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", txt)
        return np.full(n, float(u.group(1)))
    return np.fromstring(m.group(1).replace("\n", " "), sep=" ")[:n]


def read_of_frames(case, field="Q1"):
    """Sorted ``[(t, dir)]`` of OpenFOAM write-time directories holding
    ``field`` — the universal frame index for raw-state access."""
    return _frame_dirs(case, field)


def read_of_field(path, n):
    """One OpenFOAM scalar ``internalField`` (uniform or nonuniform) ->
    array of length ``n``."""
    return _read_internal(path, n)


def read_of_states(case, nq, n, prefix="Q"):
    """Raw zoomyFoam state over all write times -> ``(T, Q[t, q, x])``."""
    frames = _frame_dirs(case, f"{prefix}1")
    T = np.array([t for t, _ in frames])
    Q = np.array([[_read_internal(d / f"{prefix}{q}", n) for q in range(nq)]
                  for _, d in frames])
    return T, Q


def concat_columns(cfs, label="", atol=1e-9):
    """Join coupled-participant ColumnFields along x (e.g. part1 + part2 of
    a split domain) at their shared write times (nearest match within
    ``atol``; participants on the aligned time grid share names exactly)."""
    base = cfs[0]
    keep = [[] for _ in cfs]
    ts = []
    for t in base.t:
        js = [int(np.argmin(np.abs(cf.t - t))) for cf in cfs]
        if all(abs(cf.t[j] - t) <= atol for cf, j in zip(cfs, js)):
            ts.append(t)
            for k, j in enumerate(js):
                keep[k].append(j)
    fields = {nm: np.concatenate(
        [cf.fields[nm][keep[k]] for k, cf in enumerate(cfs)], axis=1)
        for nm in CANONICAL}
    return ColumnField(np.array(ts), np.concatenate([cf.x for cf in cfs]),
                       base.zeta, fields, label)


def _read_internal_vec_x(path, n):
    txt = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\d+\s*\((.*?)\)\s*;",
                  txt, re.S)
    if m:
        vals = re.findall(r"\(([^)]*)\)", m.group(1))
        return np.array([float(v.split()[0]) for v in vals])[:n]
    u = re.search(r"internalField\s+uniform\s+\(([^)]*)\)", txt)
    return np.full(n, float(u.group(1).split()[0]))


def read_columns(outdir, label=""):
    """Universal reader for ``columns/t_*.csv`` (x, zeta, b, h, u, v, w, p)."""
    outdir = Path(outdir)
    cdir = outdir / "columns" if (outdir / "columns").exists() else outdir
    frames = []
    for f in cdir.glob("t_*.csv"):
        frames.append((float(f.stem[2:]), f))
    frames.sort()
    if not frames:
        raise FileNotFoundError(f"no columns/t_*.csv under {outdir}")
    t = np.array([tf for tf, _ in frames])
    first = np.array([[float(v) for v in row] for row in
                      list(csv.reader(open(frames[0][1])))[1:]])
    xs = np.unique(first[:, 0])
    zs = np.unique(first[:, 1])
    X, K = len(xs), len(zs)
    fields = {n: np.zeros((len(t), X, K)) for n in CANONICAL}
    for fi, (_, f) in enumerate(frames):
        a = np.array([[float(v) for v in row] for row in
                      list(csv.reader(open(f)))[1:]])
        ix = np.searchsorted(xs, a[:, 0])
        iz = np.searchsorted(zs, a[:, 1])
        for ci, n in enumerate(CANONICAL):
            fields[n][fi, ix, iz] = a[:, 2 + ci]
    return ColumnField(t, xs, zs, fields, label)


def read_zoomyfoam(case, model, n_cells, x, K=24, label=""):
    """Lift zoomyFoam Q-frames through the MODEL'S symbolic interpolate_to_3d.

    ``model``: a zoomy SystemModel (e.g. ``SME(level=L, ...).system_model``);
    its ``interpolate_to_3d`` rows are lambdified once — the SAME function
    the coupling exchanges, no re-implementation.
    """
    import sympy as sp
    frames = _frame_dirs(case, "Q1")
    nq = len(model.state)
    zeta = (np.arange(K) + 0.5) / K
    def _unwrap(e):
        # ZArray rows may wrap the expression in a 1-element list/Matrix
        while hasattr(e, "__len__") and not hasattr(e, "free_symbols"):
            e = e[0]
        return sp.sympify(e)
    rows = [_unwrap(e) for e in model.interpolate_to_3d]
    syms = (list(model.state)
            + list(getattr(model, "aux_state", []) or [])
            + list(model.parameters)
            + [sp.Symbol("z")])
    fns = [sp.lambdify(syms, r, "numpy") for r in rows]
    pv = model.parameter_values
    pvals = [float(v) for v in (pv.values() if hasattr(pv, "values") else pv)]
    naux = len(getattr(model, "aux_state", []) or [])
    t = np.array([tf for tf, _ in frames])
    fields = {n: np.zeros((len(t), n_cells, K)) for n in CANONICAL}
    for fi, (_, d) in enumerate(frames):
        Q = np.array([_read_internal(d / f"Q{q}", n_cells) for q in range(nq)])
        args = [Q[q][:, None] for q in range(nq)]
        args += [np.zeros((n_cells, 1))] * naux
        args += [np.full((n_cells, 1), v) for v in pvals]
        args += [zeta[None, :]]
        for ci, fn in enumerate(fns[:6]):
            val = np.asarray(fn(*args), dtype=float)
            val = np.squeeze(val)
            if val.ndim == 0:
                val = np.full((n_cells, K), float(val))
            elif val.ndim == 1:
                val = (np.repeat(val[:, None], K, axis=1) if len(val) == n_cells
                       else np.repeat(val[None, :], n_cells, axis=0))
            fields[CANONICAL[ci]][fi] = val
    return ColumnField(t, np.asarray(x), zeta, fields, label)


def read_vof_raw(case, nx, ny, lx, ly, stations=None, label=""):
    """Raw VoF reader: alpha frames (for plot_water_vof) + a ColumnField of
    water-relative u(zeta) profiles resampled at the given x stations."""
    dz, dy = lx / nx, ly / ny
    ycc = (np.arange(ny) + 0.5) * dy
    frames = _frame_dirs(case, "alpha.water")
    t = np.array([tf for tf, _ in frames])
    alphas = np.zeros((len(t), ny, nx))
    for fi, (_, d) in enumerate(frames):
        alphas[fi] = _read_internal(d / "alpha.water", nx * ny).reshape(ny, nx)
    raw = {"t": t, "alpha": alphas, "lx": lx, "ly": ly}
    if stations is None:
        return raw, None
    cols = [int(np.clip(s / dz, 0, nx - 1)) for s in stations]
    K = ny
    zeta = (np.arange(K) + 0.5) / K
    fields = {n: np.zeros((len(t), len(cols), K)) for n in CANONICAL}
    for fi, (_, d) in enumerate(frames):
        Ux = _read_internal_vec_x(d / "U", nx * ny).reshape(ny, nx)
        for si, c in enumerate(cols):
            a = alphas[fi, :, c]
            hcol = a.sum() * dy
            fields["h"][fi, si, :] = hcol
            if hcol > 1e-9:
                wet = a > 0.5
                zw = np.clip(ycc[wet] / hcol, 0, 1)
                if wet.sum() >= 2:
                    order = np.argsort(zw)
                    fields["u"][fi, si, :] = np.interp(zeta, zw[order],
                                                       Ux[wet, c][order])
    cf = ColumnField(t, np.array([(c + 0.5) * dz for c in cols]), zeta,
                     fields, label)
    return raw, cf


def mass_of_columns(cf, dx):
    """Total water mass (area) M(t) = sum h(x,t) dx of a column source."""
    return cf.t, cf.fields["h"][:, :, 0].sum(axis=1) * dx


def mass_of_vof(raw, dx, dy):
    """Total water mass (area) M(t) = sum alpha dx dy of a raw VoF source."""
    return raw["t"], raw["alpha"].sum(axis=(1, 2)) * dx * dy


def read_ledger(csv_path):
    """Coupling-adapter ledger CSV -> dict of named columns.

    The 'debt' column is an IN-HOUSE diagnostic of our conservative coupling
    scheme (no literature equivalent): debt(t) = cumulative (q*·dt − realized
    intake) = mass the reduced model has given up that the resolved solver
    has not yet absorbed.  Healthy runs: hovers near zero, spikes at violent
    events, decays over the repayment horizon; the final value is the only
    mass still in transit between the domains (bounded, non-accumulating)."""
    rows = list(csv.reader(open(csv_path)))
    hdr, data = rows[0], np.array([[float(v) for v in r] for r in rows[1:]])
    return {name: data[:, i] for i, name in enumerate(hdr)}


# ── plot functions ───────────────────────────────────────────────────────────

def plot_water_columns(ax, cf, tq, style="fill", color="tab:blue",
                       grid=False, grid_kw=None, field=None, cmap="viridis",
                       vlim=None, **kw):
    """Water region (fill) / surface line h(x), or a FIELD on the true 3-D
    ``(x, z)`` quad mesh, from column data.

    ``field=<name>`` (e.g. ``"u"``) colours the actual ``(x, ζ→z)`` cells via
    ``pcolormesh`` on the cell-EDGE grid — x edges × physical-z edges
    ``z = b + h·ζ`` with ζ running the full ``[0, 1]`` (bed→surface).  Because
    the mesh is built from EDGES (not cell centres) it fills the entire water
    column with NO white gap between the top cell row and the free surface.
    Returns the ``QuadMesh`` mappable (for ``fig.colorbar``); otherwise ``ax``.

    ``grid=True`` overlays the σ-mesh wireframe: the K+1 layer interfaces
    ``z = b + h·ζ`` AND the vertical column edges (the ``(x, ζ→z)`` quad cells).
    ``grid_kw`` overrides the line style (default thin grey)."""
    i = cf.at(tq)
    h = cf.fields["h"][i, :, 0]
    b = cf.fields["b"][i, :, 0]
    X = np.asarray(cf.x); K = cf.fields["h"].shape[2]
    # Cell EDGES in x and ζ → the true quad-mesh vertices (shared by field + grid).
    xe = np.empty(len(X) + 1)
    xe[1:-1] = 0.5 * (X[:-1] + X[1:])
    xe[0] = X[0] - (X[1] - X[0]) / 2
    xe[-1] = X[-1] + (X[-1] - X[-2]) / 2
    ze = np.linspace(0.0, 1.0, K + 1)                  # ζ edges 0→1 (bed→surface)
    be = np.interp(xe, X, b); he = np.interp(xe, X, h)
    out = ax
    if field is not None:
        F = cf.fields[field][i]                        # (X, K) cell-centred
        XX = np.repeat(xe[:, None], K + 1, axis=1)     # (X+1, K+1) vertices
        ZZ = be[:, None] + he[:, None] * ze[None, :]   # physical z = b + h·ζ
        vmin, vmax = (vlim if vlim is not None else (float(F.min()), float(F.max())))
        out = ax.pcolormesh(XX, ZZ, F, cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="flat", **kw)
    elif style == "fill":
        ax.fill_between(cf.x, b, b + h, color=color, alpha=0.8, **kw)
    else:
        ax.plot(cf.x, b + h, color=color, **kw)
    if grid:
        gk = {"color": "k", "lw": 0.4, "alpha": 0.35}
        gk.update(grid_kw or {})
        # FULL quad-mesh wireframe on the cell EDGES: K+1 horizontal σ-layer
        # interfaces z=b+h·ζ AND every vertical x-cell boundary, bed→surface.
        for zk in ze:
            ax.plot(xe, be + he * zk, **gk)
        ax.vlines(xe, be, be + he, colors=gk["color"],
                  linewidths=gk["lw"], alpha=gk["alpha"])
    return out


def plot_water_vof(ax, raw, tq, smooth=True, extent=None, cmap="Blues",
                   color="tab:blue", **kw):
    """VOF water rendering.

    smooth=True (default): column-average alpha -> water depth h(x) and draw
    a smooth filled surface (same visual language as the reduced side).
    smooth=False: the raw alpha field (the VOF's own interface resolution),
    useful when the interface STRUCTURE itself is the point."""
    i = int(np.argmin(np.abs(raw["t"] - tq)))
    ext = extent or [0, raw["lx"], 0, raw["ly"]]
    if smooth:
        ny, nx = raw["alpha"][i].shape
        dx, dy = (ext[1] - ext[0]) / nx, raw["ly"] / ny
        x = ext[0] + (np.arange(nx) + 0.5) * dx
        h = raw["alpha"][i].sum(axis=0) * dy
        ax.fill_between(x, 0.0, h, color=color, alpha=0.8, **kw)
    else:
        ax.imshow(raw["alpha"][i], origin="lower", extent=ext, aspect="auto",
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest", **kw)
    return ax


def plot_profiles(ax, sources, xq, tq, colors=None, labels=None, **kw):
    """u(zeta) at station xq: every source (ColumnField) overlaid.
    A gap between curves at a coupling interface = interface mismatch."""
    from .plot import style as _st
    colors = colors or [f"C{i}" for i in range(len(sources))]
    for k, cf in enumerate(sources):
        i, s = cf.at(tq), cf.station(xq)
        lab = (labels[k] if labels else cf.label) or None
        ax.plot(cf.fields["u"][i, s, :], cf.zeta, color=colors[k],
                marker=kw.get("marker", _st.MARKERS[k % len(_st.MARKERS)]),
                markevery=kw.get("markevery", _st.MARKEVERY),
                label=lab,
                **{a: b for a, b in kw.items()
                   if a not in ("marker", "markevery")})
    ax.set_xlabel("u")
    ax.set_ylabel(r"$\zeta = z/h$")
    ax.set_ylim(0, 1)
    return ax


def plot_xt(ax, cf, fld="h", transform=None, other=None, cmap="RdBu_r", **kw):
    """Space-time (x-t) diagram — a HOVMOLLER DIAGRAM in the geophysics
    literature / characteristics diagram in hyperbolic-PDE texts.  Waves
    appear as diagonal stripes (slope = 1/speed); stripes emanating from a
    coupling interface and travelling back into the domain are reflections.

    transform:
    None        raw field (zeta-mean)
    'plateau'   detrended: minus the instantaneous spatial mean and the
                temporal mean per position.  NOT a literature method — an
                in-house high-pass diagnostic that isolates propagating
                disturbances from slow drifts (label plots accordingly).
    'diff'      this source minus ``other`` (e.g. coupled - monolithic)
    """
    F = cf.fields[fld].mean(axis=2)
    if transform == "plateau":
        F = F - F.mean(axis=1, keepdims=True)
        F = F - F.mean(axis=0, keepdims=True)
    elif transform == "diff":
        G = other.fields[fld].mean(axis=2)
        n = min(len(F), len(G))
        F = F[:n] - G[:n]
    v = kw.pop("vmax", np.abs(F).max() or 1)
    im = ax.pcolormesh(cf.x, cf.t[:len(F)], F, cmap=cmap, vmin=-v, vmax=v,
                       shading="nearest", **kw)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return im


def mark_stations(ax, xs, colors=None, labels=None, label_y=0.92,
                  ls="--", lw=1.2, alpha=0.9, fontsize=None):
    """Vertical marker lines (interface position, profile stations, ...)
    with optional small text labels at axes-fraction height ``label_y``.
    Use the same colors as the linked panels (see frame_color)."""
    import matplotlib.transforms as mtransforms
    colors = colors or [f"C{i}" for i in range(len(xs))]
    for k, (x, c) in enumerate(zip(xs, colors)):
        ax.axvline(x, color=c, ls=ls, lw=lw, alpha=alpha)
        if labels and labels[k]:
            trans = mtransforms.blended_transform_factory(ax.transData,
                                                          ax.transAxes)
            import matplotlib as mpl
            fs = fontsize or mpl.rcParams["legend.fontsize"]
            ax.text(x, label_y, " " + labels[k], transform=trans,
                    fontsize=fs, color=c, ha="left", va="top",
                    rotation=0, clip_on=True)
    return ax


def frame_color(ax, color, lw=2.0):
    """Color a subplot's outer frame — links a panel to its marker line."""
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(lw)
    return ax


def plot_series(ax, curves, **kw):
    """curves: list of (t, y, label).  Mass audits, ledger traces, etc."""
    from .plot import style as _st
    for t, y, lab in curves:
        ax.plot(t, y, label=lab,
                markevery=kw.get("markevery", _st.MARKEVERY),
                **{a: b for a, b in kw.items() if a != "markevery"})
    ax.set_xlabel("t")
    if any(lab for _, _, lab in curves):
        ax.legend()
    return ax


# ── layer 2: ready-made composed figures (built from the tools above) ──────

def fig_coupling(fig, tq, reduced_cf, vof_raw, panels, interface_x=0.0,
                 xlim=None, ylim=None, ulim=None, title=None):
    """The standard coupled-domain figure: top = joint water surface with
    interface + station markers; bottom = one u(zeta) panel per station,
    each panel's FRAME colored like its marker line in the top view.

    panels: list of (title, station_x, [ColumnField, ...], color)
    """
    from .plot import style as _st
    gs = fig.add_gridspec(2, max(len(panels), 1), height_ratios=[1.2, 1.0],
                          hspace=0.45, wspace=0.4)
    ax = fig.add_subplot(gs[0, :])
    plot_water_vof(ax, vof_raw, tq, color=_st.COLORS["water"])
    plot_water_columns(ax, reduced_cf, tq, style="fill",
                       color=_st.COLORS["water"])
    ax.axvline(interface_x, color=_st.COLORS["interface"], lw=1.6)
    mark_stations(ax, [s for _, s, _, _ in panels],
                  [c for _, _, _, c in panels],
                  labels=[n for n, _, _, _ in panels])
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title or f"t = {tq:5.2f} s")
    for k, (name, xq, sources, color) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        plot_profiles(a, sources, xq, tq,
                      colors=[_st.COLORS["reduced"], _st.COLORS["resolved"],
                              _st.COLORS["reference"]][:len(sources)])
        a.set_title(name)
        if ulim: a.set_xlim(*ulim)
        frame_color(a, color)
    _st.figure_legend(fig, extra=[
        ("interface", _st.line("interface")),
        ("water", _st.line("water", lw=5)),
    ])
    return fig


def fig_reduced_coupling(fig, tq, coupled, reference=(), panels=(),
                         interface_x=None, ylim=None, ulim=None, title=None):
    """Reduced<->reduced coupling figure (the SME|SME analogue of
    :func:`fig_coupling`): top = free surface b+h(x) of every source
    (coupled solid + markers, reference dashed gray); bottom = one u(zeta)
    panel per station, frames colored like their marker lines.

    coupled / reference: list of (ColumnField, label)
    panels: list of (title, station_x, [ColumnField, ...], color)

    Returns ``{"water": ax, "profiles": [ax, ...]}`` so callers can overlay
    extra curves (e.g. an analytic solution) before saving.
    """
    from .plot import style as _st
    gs = fig.add_gridspec(2 if panels else 1, max(len(panels), 1),
                          height_ratios=[1.2, 1.0] if panels else [1.0],
                          hspace=0.45, wspace=0.4)
    ax = fig.add_subplot(gs[0, :])
    color_of = {}
    grays = ["0.55", "0.25", "0.7"]
    for k, (cf, lab) in enumerate(reference):
        i = cf.at(tq)
        color_of[id(cf)] = grays[k % len(grays)]
        ax.plot(cf.x, cf.fields["b"][i, :, 0] + cf.fields["h"][i, :, 0],
                color=color_of[id(cf)], ls="--", marker="",
                label=lab or cf.label)
    for k, (cf, lab) in enumerate(coupled):
        i = cf.at(tq)
        color_of[id(cf)] = _st.CYCLE[k % len(_st.CYCLE)]
        ax.plot(cf.x, cf.fields["b"][i, :, 0] + cf.fields["h"][i, :, 0],
                color=color_of[id(cf)], marker=_st.MARKERS[k % len(_st.MARKERS)],
                markevery=_st.MARKEVERY, label=lab or cf.label)
    if interface_x is not None:
        ax.axvline(interface_x, color=_st.COLORS["interface"], lw=1.6)
    if panels:
        mark_stations(ax, [s for _, s, _, _ in panels],
                      [c for _, _, _, c in panels],
                      labels=[n for n, _, _, _ in panels])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("b + h")
    ax.set_title(title or f"t = {tq:5.2f} s")
    profs = []
    for k, (name, xq, sources, color) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        plot_profiles(a, sources, xq, tq,
                      colors=[color_of.get(id(s), f"C{j}")
                              for j, s in enumerate(sources)])
        a.set_title(name)
        if ulim:
            a.set_xlim(*ulim)
        frame_color(a, color)
        profs.append(a)
    extra = ([("interface", _st.line("interface"))]
             if interface_x is not None else [])
    _st.figure_legend(fig, extra=extra)
    return {"water": ax, "profiles": profs}


def fig_mass_deviation(ax, sources, total_label="total"):
    """Conservation figure: M(t) − M(0) per source and for the sum — only
    the DEVIATION is shown.  sources: list of (t, M, label)."""
    base_t = sources[0][0]
    tot = np.zeros_like(np.asarray(base_t, dtype=float))
    for t_, M, lab in sources:
        M = np.asarray(M, dtype=float)
        ax.plot(t_, M - M[0], label=lab)
        tot += np.interp(base_t, t_, M - M[0])
    ax.plot(base_t, tot, "k--", lw=1.6, label=total_label)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$M(t) - M(0)$")
    ax.legend()
    return ax


def animate(layout, times, out, figsize=(10, 6), fps=8, dpi=100):
    """Compose panels into a gif (thin wrapper of the generic engine,
    :func:`zoomy_plotting.animate`).

    ``layout``: whole-figure composer ``fn(fig, t)`` (layer 2), OR a list
    of (gridspec_slice, draw_fn) where draw_fn(ax, t) draws one panel at
    time t using the plot functions above.
    """
    from .plot.video import animate as _animate

    def draw(fig, tq):
        if callable(layout):
            layout(fig, tq)          # whole-figure composer (layer 2)
        else:
            gs = fig.add_gridspec(2, max(len(layout) - 1, 1),
                                  height_ratios=[1.2, 1.0],
                                  hspace=0.45, wspace=0.4)
            for spec, draw_fn in layout:
                ax = fig.add_subplot(gs[spec] if not isinstance(spec, tuple)
                                     else gs[spec[0], spec[1]])
                draw_fn(ax, tq)

    return _animate(draw, times, out, fps=fps, figsize=figsize, dpi=dpi)
