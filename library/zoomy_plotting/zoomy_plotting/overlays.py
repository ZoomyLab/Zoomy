"""Domain-overlay + comparison primitives (REQ-135, malpasset-flagged).

Generalizations of the case-local helpers in
``thesis/cases/malpasset_jax/postprocess.py`` — the reusable gaps around
:func:`zoomy_plotting.mesh_plots.plot_topo_field` plus two composed helpers.
Everything follows the house style (Okabe–Ito ``CYCLE``, semantic
``COLORS``, below-figure :func:`figure_legend`).

- :func:`station_markers`      ringed + halo-labelled points on a plan view.
- :func:`inset_wedge_colorbar` horizontal colorbar tucked into an empty
  corner ("wedge") of the domain axes.
- :func:`bars`                 grouped-bar chart (one bar per series per
  station) with the legend-below house style.
- :func:`compare_animate`      frame-locked side-by-side animation of the
  same field across N stores with a shared vlim.

Matplotlib is imported lazily inside every function so ``import
zoomy_plotting`` stays cheap (Pyodide-safe).
"""
from __future__ import annotations

import numpy as np

from .plot import style as _st
from .plot.video import animate as _animate

__all__ = [
    "station_markers", "inset_wedge_colorbar", "bars", "compare_animate",
]


# ── plot_topo_field overlays ────────────────────────────────────────────────

def station_markers(ax, points, labels=None, color=None, size=42, lw=1.6,
                    halo=True, halo_color="white", fontsize=None,
                    offset=(5, 5), zorder=6):
    """Ringed survey/gauge markers with halo-outlined labels on a plan view.

    ``points``: sequence of ``(x, y)`` (or a mapping ``{label: (x, y, ...)}``,
    in which case its keys become the labels and only the first two entries of
    each value are used as coordinates).  Rings are hollow (``facecolors
    none``) so the underlying field/topography shows through; labels get a
    white stroke ``halo`` so they stay legible over any background — the exact
    gap that forced the bespoke ``scatter``/``annotate`` block in the Malpasset
    gauge figure.
    """
    import matplotlib as mpl
    import matplotlib.patheffects as pe

    if isinstance(points, dict):
        labels = list(points.keys()) if labels is None else labels
        pts = [(v[0], v[1]) for v in points.values()]
    else:
        pts = [(p[0], p[1]) for p in points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    c = _st.resolve_color(color) if color is not None else _st.COLORS["station"]
    fs = fontsize or mpl.rcParams["legend.fontsize"]

    ax.scatter(xs, ys, s=size, facecolors="none", edgecolors=c,
               linewidths=lw, zorder=zorder - 1)
    if labels is not None:
        effects = ([pe.withStroke(linewidth=2.2, foreground=halo_color)]
                   if halo else [])
        for (x, y), lab in zip(pts, labels):
            if not lab:
                continue
            ax.annotate(str(lab), (x, y), textcoords="offset points",
                        xytext=offset, color=c, fontsize=fs, zorder=zorder,
                        path_effects=effects)
    return ax


def inset_wedge_colorbar(ax, mappable, loc="lower right", width=0.32,
                         height=0.035, margin=0.04, label=None,
                         orientation="horizontal", **kw):
    """A horizontal colorbar tucked into an empty corner ("wedge") of ``ax``.

    Reduced-flow plan views (e.g. Malpasset) fill only part of a rectangular
    axes; a side colorbar wastes the empty wedge.  This drops the colorbar
    INTO that wedge via an inset axes placed in axes-fraction coordinates.

    ``mappable``: any ScalarMappable — e.g. ``plot_topo_field(...,
    colorbar=False)["mappable"]``.
    ``loc``: one of ``lower right / lower left / upper right / upper left``.
    """
    corners = {
        "lower right": (1 - margin - width, margin),
        "lower left":  (margin,             margin),
        "upper right": (1 - margin - width, 1 - margin - height),
        "upper left":  (margin,             1 - margin - height),
    }
    if loc not in corners:
        raise ValueError(f"loc must be one of {sorted(corners)}, got {loc!r}")
    x0, y0 = corners[loc]
    import matplotlib as mpl
    cax = ax.inset_axes([x0, y0, width, height])
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation=orientation, **kw)
    cax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"])
    if label:
        cbar.set_label(label, fontsize=mpl.rcParams["axes.labelsize"])
    return cbar


# ── grouped bars ────────────────────────────────────────────────────────────

def bars(ax, stations, series, colors=None, width=0.8, ylabel=None,
         legend=True, grid=True):
    """Grouped-bar chart: one bar per ``series`` per ``station``.

    ``stations``: category labels (x groups).  ``series``: ``{name: values}``
    where each ``values`` aligns with ``stations``.  Bars use the Okabe–Ito
    ``CYCLE``; the legend goes BELOW the whole figure (house style) rather
    than crammed in-axes — the generalization of the Malpasset gauge bar
    block (Observed | TELEMAC | order1 | order2).
    """
    names = list(series)
    n = max(len(names), 1)
    x = np.arange(len(stations))
    w = width / n
    off0 = -(n - 1) / 2 * w
    for i, name in enumerate(names):
        c = (colors[i] if colors is not None
             else _st.CYCLE[i % len(_st.CYCLE)])
        ax.bar(x + off0 + i * w, np.asarray(series[name], dtype=float), w,
               label=name, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stations])
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis="y", alpha=0.3)
    if legend:
        _st.figure_legend(ax.figure)
    return ax


# ── side-by-side comparison animation ───────────────────────────────────────

def _shared_vlim(stores, field, hide_below=None, pct=97):
    """Percentile-capped color limits shared across every store and frame."""
    vals = []
    for s in stores:
        for k in range(s.n_snapshots):
            a = np.asarray(s.get_cell(k, field), dtype=float)
            vals.append(a[a > hide_below] if hide_below is not None else a)
    vals = [v for v in vals if v.size]
    allv = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    lo = 0.0 if hide_below is not None else float(np.min(allv))
    return lo, float(np.percentile(allv, pct))


def compare_animate(stores, draw, times, out, titles=None, vlim=None,
                    field=None, hide_below=None, figsize=None, fps=8,
                    dpi=100, suptitle=None):
    """Frame-locked side-by-side animation of the SAME field across N stores.

    Builds a ``1 x N`` panel per frame and calls ``draw(ax, store, t, vlim)``
    for each store, so every panel shares ONE color scale ``vlim`` and the
    same time ``t`` — the honest way to compare runs (Malpasset 1st-vs-2nd
    order).  ``vlim`` is auto-computed as a shared percentile cap when
    ``field`` is given and ``vlim is None``.

    ``titles``: per-panel titles (defaults to each store's ``label``).
    ``suptitle``: a string or a callable ``t -> str`` drawn over the row.
    """
    n = len(stores)
    if vlim is None and field is not None:
        vlim = _shared_vlim(stores, field, hide_below)
    if titles is None:
        titles = [getattr(s, "label", "") or f"run {i}"
                  for i, s in enumerate(stores)]
    figsize = figsize or (7 * n, 6)

    def _draw(fig, t):
        axes = np.atleast_1d(fig.subplots(1, n, squeeze=False)[0])
        for ax, s, ttl in zip(axes, stores, titles):
            draw(ax, s, t, vlim)
            if ttl:
                ax.set_title(ttl)
        if suptitle is not None:
            fig.suptitle(suptitle(t) if callable(suptitle) else suptitle)

    return _animate(_draw, times, out, fps=fps, figsize=figsize, dpi=dpi)
