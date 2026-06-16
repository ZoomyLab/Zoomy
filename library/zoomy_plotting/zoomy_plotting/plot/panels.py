"""Granular, composable single-axes plotting building blocks.

These are deliberately *primitive*: each draws into an ``ax`` you already
own (from ``plt.subplots`` / ``plt.subplot_mosaic``), never creates a
figure, and never draws its own legend — labels are attached to the
artists so a *shared* legend (:func:`figure_legend` / :func:`row_legend`)
can collect them. Compose them freely into multi-panel layouts.

Design rules (same contract as the rest of ``zoomy_plotting.plot``):

1. Every color routes through :func:`~zoomy_plotting.plot.style.resolve_color`
   (the single ``COLORS`` lookup); line widths / sizes fall through to the
   active rcParams (set by :func:`~zoomy_plotting.plot.style.apply_style`)
   when the caller does not override them. No palette is duplicated here.
2. **Lazy matplotlib import**: importing this module must NOT load
   matplotlib (checked by ``tests/test_lazy.py``). Every function here
   operates on ``ax`` / ``fig`` objects the caller already created, so no
   ``import matplotlib`` is needed at all.
"""

from __future__ import annotations

from typing import Optional

from . import style
from .style import CONFIG


# --------------------------------------------------------------------------
# Entry normalisation + styling
# --------------------------------------------------------------------------
def _normalize(entry, xkey, ykey):
    """Coerce a series/profile item into a dict with ``xkey``/``ykey`` set.

    Accepts either a dict (used as-is) or a ``(x, y)`` / ``(x, y, label)``
    tuple/list. ``xkey``/``ykey`` name the two data axes for this kind of
    panel (``"x"``/``"y"`` for lines, ``"values"``/``"coord"`` for
    profiles).
    """
    if isinstance(entry, dict):
        return entry
    seq = list(entry)
    if len(seq) < 2:
        raise ValueError(
            f"tuple series entry needs at least ({xkey}, {ykey}); got {entry!r}"
        )
    d = {xkey: seq[0], ykey: seq[1]}
    if len(seq) >= 3:
        d["label"] = seq[2]
    return d


def _style_kwargs(entry):
    """Build matplotlib ``ax.plot`` style kwargs from a normalized entry.

    ``role`` / ``color`` -> color (via :func:`style.resolve_color`),
    ``ls`` -> linestyle, ``marker`` -> marker, ``lw`` -> linewidth,
    ``label`` -> label. Anything the entry omits is left out so the active
    rcParams / prop_cycle supply the default.
    """
    kw = {}
    ls = entry.get("ls", "-")
    if ls is not None:
        kw["linestyle"] = ls
    marker = entry.get("marker")
    if marker is not None:
        kw["marker"] = marker
    lw = entry.get("lw")
    if lw is not None:
        kw["linewidth"] = lw
    spec = entry.get("color", entry.get("role"))
    if spec is not None:
        kw["color"] = style.resolve_color(spec)
    label = entry.get("label")
    if label is not None:
        kw["label"] = label
    return kw


def _apply_grid(ax, grid):
    show = CONFIG.axes_grid_1d if grid is None else grid
    if show:
        ax.grid(True, alpha=CONFIG.grid_alpha)


# --------------------------------------------------------------------------
# line_plot
# --------------------------------------------------------------------------
def line_plot(
    ax,
    series,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    logx: bool = False,
    logy: bool = False,
    title: Optional[str] = None,
    grid: Optional[bool] = None,
):
    """Draw one or more labelled curves into ``ax``.

    Parameters
    ----------
    ax
        A matplotlib ``Axes`` the caller owns. Not created here.
    series
        List of entries. Each is a dict ``{x, y, label?, role|color?,
        ls?, marker?, lw?}`` or a ``(x, y[, label])`` tuple. ``role`` is a
        semantic key of :data:`~zoomy_plotting.plot.style.COLORS`
        (e.g. ``"water"``); ``color`` is any matplotlib color spec.
    logx, logy
        Set log scale on the respective axis.

    Does NOT add a legend — labels live on the artists for a shared
    legend (:func:`figure_legend` / :func:`row_legend`).

    Returns a list of the created ``Line2D`` artists.
    """
    lines = []
    for raw in series:
        entry = _normalize(raw, "x", "y")
        kw = _style_kwargs(entry)
        (ln,) = ax.plot(entry["x"], entry["y"], **kw)
        lines.append(ln)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    _apply_grid(ax, grid)
    return lines


# --------------------------------------------------------------------------
# profile_plot
# --------------------------------------------------------------------------
def profile_plot(
    ax,
    profiles,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = "vertical",
    grid: Optional[bool] = None,
):
    """Draw vertical (or horizontal) profiles into ``ax``.

    Parameters
    ----------
    profiles
        List of entries. Each is a dict ``{values, coord, label?,
        role|color?, ls?, marker?, lw?}`` or a ``(values, coord[, label])``
        tuple. ``values`` is the field (e.g. ``u``); ``coord`` is the
        profile coordinate (e.g. depth fraction ``sigma``).
    orientation
        ``"vertical"`` (default) puts ``coord`` on the y-axis and
        ``values`` on the x-axis (the usual ``u`` vs depth picture);
        ``"horizontal"`` swaps them.

    Same styling/legend contract as :func:`line_plot`. Returns the list of
    ``Line2D`` artists.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )

    lines = []
    for raw in profiles:
        entry = _normalize(raw, "values", "coord")
        kw = _style_kwargs(entry)
        values, coord = entry["values"], entry["coord"]
        if orientation == "vertical":
            (ln,) = ax.plot(values, coord, **kw)
        else:
            (ln,) = ax.plot(coord, values, **kw)
        lines.append(ln)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    _apply_grid(ax, grid)
    return lines


# --------------------------------------------------------------------------
# row_legend
# --------------------------------------------------------------------------
def row_legend(
    fig,
    axes_row,
    *,
    extra=None,
    ncol: Optional[int] = None,
    y: Optional[float] = None,
    reserve: float = 0.04,
):
    """ONE legend centered under a ROW of axes.

    The missing counterpart to :func:`figure_legend`: instead of one
    legend under the *whole* figure, this places one legend under a single
    row. Handles/labels are gathered from ``axes_row`` only (de-duplicated,
    in-axes legends removed) and merged with ``extra``
    ``[(label, handle), ...]`` proxies, exactly like
    :func:`figure_legend` (shared via
    :func:`~zoomy_plotting.plot.style._collect_handles_labels` and
    :func:`~zoomy_plotting.plot.style._frame_legend`).

    The legend is horizontally centered on the union of the row's axes and
    placed just below them. ``reserve`` is the vertical gap (figure
    fraction) between the row's bottom and the legend's top; pass ``y`` to
    override the anchor's y-coordinate (figure fraction) outright.

    Returns the ``Legend`` (or ``None`` if the row has no labelled
    artists).
    """
    handles, labels = style._collect_handles_labels(axes_row, extra)
    if not handles:
        return None
    ncol = ncol or min(len(handles), 5)

    positions = [ax.get_position() for ax in axes_row]
    x0 = min(p.x0 for p in positions)
    x1 = max(p.x1 for p in positions)
    y0 = min(p.y0 for p in positions)
    xcenter = 0.5 * (x0 + x1)
    yanchor = y if y is not None else (y0 - reserve)

    leg = fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(xcenter, yanchor), ncol=ncol,
        frameon=True, fancybox=True, borderpad=0.6,
    )
    return style._frame_legend(leg)
