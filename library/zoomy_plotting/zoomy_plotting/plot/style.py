"""Styling configuration — the single source of truth for every visual
parameter used by the plotters.

Design rules:

1. Nothing inside ``plot_1d / plot_2d / plot_3d`` or their helpers may
   hardcode a color, linewidth, alpha, markersize, or colormap. All such
   values come from :data:`CONFIG`. Plot functions may accept matching
   kwargs that, when non-``None``, override the config for that one call.
2. **Lazy matplotlib import**: importing this module must NOT load
   matplotlib. ``import zoomy_plotting`` should be cheap even in Pyodide
   where matplotlib's init takes ~1–2 s. Every ``import matplotlib...``
   in this file lives inside a function body.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields as _fields
from typing import Optional


# ── the Zoomy color/marker scheme (canonical home) ──────────────────────────

#: Okabe–Ito (colorblind-safe) discrete rotation
CYCLE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

#: semantic roles used across the coupling/reduced-model figures.
#:
#: The three cross-figure DATA roles a case should reach for so the same kind
#: of curve looks identical in every figure (never cycle these):
#:   * ``experiment`` — measured data (points / dashed markers);
#:   * ``reference``  — a trusted ground-truth benchmark (TELEMAC, VOF, DNS,
#:     refined semi-analytic);
#:   * ``analytic``   — a closed-form / asymptotic reference line.
#: Model SERIES (SME levels, our runs) keep the Okabe–Ito :data:`CYCLE`.
COLORS = {
    # --- cross-figure data roles ---
    "experiment": "#D55E00",   # measured data (vermillion, Okabe–Ito)
    "reference":  "#666666",   # ground-truth benchmark (grey)
    "analytic":   "#000000",   # closed-form / asymptotic (black)
    # --- coupling / reduced-model roles ---
    "water":     "#56B4E9",
    "reduced":   "#0072B2",
    "resolved":  "#D55E00",
    "interface": "#C8102E",
    "station":   "#009E73",
}

#: marker rotation, paired index-wise with CYCLE
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: marker thinning for dense lines — pass markevery=MARKEVERY
MARKEVERY = 0.15

CMAP_CONTINUOUS = "viridis"
CMAP_DIVERGING = "RdBu_r"
#: grayscale base layer for topography-under-field overlays
CMAP_TOPO = "Greys"

#: output-medium profiles: "print" = fonts for FINAL PRINT dimensions
#: (publication norm 9pt at true figsize); "screen" = gifs/slides on large
#: canvases need proportionally larger fonts.
#:
#: The SELECTABLE TEMPLATES a case opts into (REQ-98 / CASE_STYLE §7). Pick one
#: per render with :func:`use` or :func:`subplots`; NEVER edit a case to change
#: format — switch the preset instead:
#:   * ``publication`` — final-print journal sizes (9 pt at true figsize).
#:   * ``thesis``      — final-print sizes; differs from ``publication`` only in
#:     the figure WIDTHS (thesis text-column, see :data:`SIZES`).
#:   * ``screen``      — slides/GIFs: larger fonts on a big canvas.
#:   * ``presentation``— projector: ~DOUBLES the fonts and, via ``min_pt``,
#:     CLAMPS every text element so nothing renders below 14 pt at 1080p.
#: ``print`` is kept as the original alias of ``publication``.
PROFILES = {
    "publication":  {"scale": 1.0, "lw": 1.4, "ms": 4, "min_pt": 9.0},
    "print":        {"scale": 1.0, "lw": 1.4, "ms": 4, "min_pt": 9.0},
    "thesis":       {"scale": 1.0, "lw": 1.4, "ms": 4, "min_pt": 9.0},
    "screen":       {"scale": 1.6, "lw": 2.0, "ms": 6},
    "presentation": {"scale": 2.0, "lw": 2.6, "ms": 8, "min_pt": 14.0},
}

#: Figure-SIZE presets, in inches — the missing half of a template (PROFILES
#: only carried fonts). ``cell`` is the size of ONE axes; a grid of
#: ``ncols x nrows`` axes is ``(cell_w * ncols, cell_h * nrows)`` (this is what
#: :func:`subplots` / :func:`figsize` use). ``widths`` gives fixed FULL-figure
#: sizes for a single-panel figure at that medium's ``1col`` / ``2col`` /
#: ``full`` width. Switch the preset to reflow a case from journal to thesis
#: to slides without touching its plotting code.
#:
#: ``text`` is the width a figure gets when it is placed NORMALLY, i.e. into the
#: running text column, and it is the token to reach for unless the figure is
#: explicitly marked full-width in the document. For ``thesis`` it is measured
#: rather than guessed: ``tufte-book`` at ``b5paper`` (the thesis template)
#: reports ``\textwidth = 292.096 pt = 4.042 in``, ``\marginparwidth = 1.443 in``
#: and ``\marginparsep = 0.289 in``, so a full-width float spans 5.774 in.
#: Authoring at ``full`` (5.5 in) and then placing the figure normally scales it
#: by 0.735, which turns the 9 pt house font into 6.6 pt on the page. That is a
#: silent defect: nothing errors, the labels just come out below the readable
#: minimum. Match the token to the placement.
#:
#: ``full`` is correspondingly the full-width float, ``\textwidth +
#: \marginparsep + \marginparwidth = 5.774 in``, and a figure authored at it
#: must carry ``:class: full-width`` in the document or it will be squeezed back
#: into the 4.04 in column.
SIZES = {
    "publication":  {"cell": (3.4, 2.6),
                     "widths": {"1col": (3.4, 2.6), "2col": (7.0, 3.4),
                                "text": (3.4, 2.6), "full": (7.0, 3.4)}},
    "print":        {"cell": (3.4, 2.6),
                     "widths": {"1col": (3.4, 2.6), "2col": (7.0, 3.4),
                                "text": (3.4, 2.6), "full": (7.0, 3.4)}},
    "thesis":       {"cell": (2.75, 2.4),
                     "widths": {"1col": (2.7, 2.2), "2col": (5.77, 3.4),
                                "text": (4.04, 3.0), "full": (5.77, 3.4)}},
    "screen":       {"cell": (5.0, 3.6),
                     "widths": {"1col": (6.0, 4.0), "2col": (11.0, 5.0),
                                "text": (6.0, 4.0), "full": (12.0, 6.0)}},
    "presentation": {"cell": (6.0, 4.2),
                     "widths": {"1col": (7.0, 4.5), "2col": (12.0, 6.0),
                                "text": (7.0, 4.5), "full": (13.3, 7.5)}},
}


@dataclass
class PlotConfig:
    """All styling knobs for :class:`MatplotlibPlotter`.

    Mutate ``CONFIG`` directly for a session-wide change, or wrap a block
    in :func:`apply_style` to get a scoped override that reverts on exit.
    """

    # --- Colormap ---
    cmap: str = "viridis"

    # --- 1D plot ---
    line_color: Optional[str] = None   # None => matplotlib's default prop_cycle
    line_linewidth: float = 1.5
    line_linestyle: str = "-"
    node_show: bool = True
    node_color: str = "black"
    node_markersize: float = 3.0
    node_marker: str = "o"

    # --- Mesh overlay (2D + 3D when show_mesh=True) ---
    mesh_edgecolor: str = "black"
    mesh_linewidth: float = 0.3
    mesh_alpha: float = 1.0

    # --- 3D rendering ---
    face_alpha: float = 0.9
    face_shade: bool = True
    viewpoint_elev: float = 30.0

    # --- Colorbar ---
    colorbar_show: bool = True
    colorbar_shrink: float = 0.8
    colorbar_pad: float = 0.05

    # --- Figure-wide rcParams pushed by apply_style() ---
    font_family: str = "serif"
    font_size: float = 9.0
    profile: str = "print"
    axes_grid_1d: bool = True
    axes_grid_2d3d: bool = False
    grid_alpha: float = 0.3
    figure_figsize: tuple = (7.0, 5.0)
    figure_dpi: float = 120.0


# Module-level singleton; users edit this directly for session-wide defaults.
CONFIG = PlotConfig()


def _as_rcparams(cfg: PlotConfig) -> dict:
    """Project a :class:`PlotConfig` onto matplotlib rcParams (the full
    Zoomy publication scheme: Okabe–Ito + marker rotation, serif/STIX,
    inward ticks, profile-scaled sizes)."""
    import matplotlib as mpl
    prof = PROFILES.get(cfg.profile, PROFILES["print"])
    s = prof["scale"]
    base = cfg.font_size
    # ``min_pt`` (presentation) is a HARD floor: every text element is
    # clamped up to it so nothing is illegible on a projector regardless of
    # the base ``font_size`` the case set.
    floor = prof.get("min_pt", 0.0)

    def _sz(pt):
        return round(max(pt * s, floor), 1)

    return {
        "image.cmap": cfg.cmap,
        "font.family": cfg.font_family,
        "mathtext.fontset": "stix",
        "font.size": _sz(base),
        # Titles are SETUP-only and small (CASE_STYLE §7): axes titles at body
        # size, the figure suptitle just one point up — never a headline.
        "axes.titlesize": _sz(base),
        "figure.titlesize": _sz(base + 1),
        "axes.labelsize": _sz(base),
        "xtick.labelsize": _sz(base - 1),
        "ytick.labelsize": _sz(base - 1),
        "legend.fontsize": _sz(base - 1),
        "lines.linewidth": prof["lw"],
        "lines.markersize": prof["ms"],
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.prop_cycle": (mpl.cycler(color=CYCLE)
                            + mpl.cycler(marker=MARKERS)),
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": cfg.figure_figsize,
        "figure.dpi": cfg.figure_dpi,
    }


def use(profile="publication"):
    """Select a template GLOBALLY (non-scoped variant of :func:`apply_style`).

    ``profile`` is a key of :data:`PROFILES` / :data:`SIZES`
    (``publication`` / ``thesis`` / ``screen`` / ``presentation``; ``print`` is
    an alias). Sets the fonts AND the default figure size for that medium so a
    bare ``plt.subplots()`` afterwards is already correctly sized.
    """
    import matplotlib as mpl
    CONFIG.profile = profile
    spec = SIZES.get(profile)
    if spec is not None:
        CONFIG.figure_figsize = spec["widths"]["full"]
    mpl.rcParams.update(_as_rcparams(CONFIG))


def figsize(preset=None, ncols=1, nrows=1, width=None):
    """Figure size (inches) for a ``ncols x nrows`` axes grid in ``preset``.

    ``preset`` defaults to the active :data:`CONFIG` ``profile``. Pass ``width``
    (``"1col"`` / ``"2col"`` / ``"full"``) to get a fixed single-panel width
    from :data:`SIZES` instead of the per-axes-cell grid size.
    """
    preset = preset or CONFIG.profile
    spec = SIZES.get(preset, SIZES["publication"])
    if width is not None:
        return spec["widths"][width]
    cw, ch = spec["cell"]
    return (cw * ncols, ch * nrows)


def subplots(nrows=1, ncols=1, *, preset=None, width=None, apply=True, **kwargs):
    """``plt.subplots`` with the Zoomy template pre-applied and sized.

    A case writes ``fig, axes = zp.subplots(1, 3, preset="thesis")`` and owns
    NO styling: :func:`use` activates the preset's fonts/cycle/ticks and the
    figure is sized from :data:`SIZES` (per-axes cell * grid, or the fixed
    ``width`` token). ``figsize`` in ``kwargs`` overrides the computed size;
    ``apply=False`` skips the global style switch (keep whatever is active).
    Returns ``(fig, axes)`` exactly like ``plt.subplots``.
    """
    import matplotlib.pyplot as plt
    preset = preset or CONFIG.profile
    if apply:
        use(preset)
    fs = kwargs.pop("figsize", None) or figsize(preset, ncols, nrows, width)
    return plt.subplots(nrows, ncols, figsize=fs, **kwargs)


def resolve_color(role_or_color):
    """Single point of role -> color resolution.

    Maps a semantic role name (a key of :data:`COLORS`, e.g. ``"water"``)
    to its hex value; passes any other spec (hex string, named color,
    ``None``) straight through. Both :func:`line` and the
    :mod:`zoomy_plotting.plot.panels` building blocks route through here so
    the ``COLORS`` lookup lives in exactly one place.
    """
    return COLORS.get(role_or_color, role_or_color)


class _Colors:
    """Attribute view over :data:`COLORS` — ``colors.experiment`` etc.

    Gives cases a typo-safe, discoverable handle on the semantic palette
    (``colors.experiment`` / ``colors.reference`` / ``colors.analytic`` and the
    coupling roles) so an experimental or reference curve looks identical in
    every figure. ``colors.cycle`` returns the Okabe–Ito data-series cycle;
    ``colors["anything"]`` falls through :func:`resolve_color`.
    """

    @property
    def cycle(self):
        return list(CYCLE)

    def __getattr__(self, name):
        try:
            return COLORS[name]
        except KeyError:
            raise AttributeError(
                f"no semantic color {name!r}; known: {sorted(COLORS)} (+ 'cycle')"
            )

    def __getitem__(self, name):
        return resolve_color(name)

    def __dir__(self):
        return sorted(COLORS) + ["cycle"]


#: Singleton semantic-color accessor (``zp.colors.experiment`` …).
colors = _Colors()


def line(role_or_color, ls="-", lw=None, marker=None):
    """Proxy legend handle (e.g. for marker-line meanings)."""
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    color = resolve_color(role_or_color)
    return Line2D([], [], color=color, ls=ls,
                  lw=lw or mpl.rcParams["lines.linewidth"], marker=marker)


def _collect_handles_labels(axes, extra=None, remove_in_axes=True):
    """Gather (handles, labels) across ``axes``, de-duplicated by label.

    Shared mechanics for :func:`figure_legend` and
    :func:`zoomy_plotting.plot.panels.row_legend`: pull each axis's legend
    entries, drop blank labels and duplicates, optionally remove any
    in-axes legends (so the collected one is the only legend drawn), then
    append ``extra`` ``[(label, handle), ...]`` proxies.
    """
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li and li not in labels:
                handles.append(hi)
                labels.append(li)
        if remove_in_axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    for label, handle in (extra or []):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    return handles, labels


#: Largest share of the figure height ``figure_legend`` will surrender to the
#: legend strip.  Beyond this the figure is mostly legend, so the caller is
#: warned rather than the axes being squeezed away.
_LEGEND_MAX_BOTTOM = 0.75


def label_panels(fig, axes=None, fmt="({}) ", start="a", upper=False):
    """Prefix each subplot's TITLE with a panel letter: ``(a)``, ``(b)``, ...

    A multi-panel figure has to be referable from the caption and the prose
    ("panel (b) shows ...").  Putting the letter in the title rather than
    floating it in the corner keeps it with the thing it names, survives
    tight_layout, and cannot collide with the data.

    Idempotent: an axes whose title already starts with the pattern is left
    alone, so calling this twice (or on a figure built by a helper that
    already labelled it) does not produce ``(a) (a) depth``.

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("depth"); ax2.set_title("velocity")
        style.label_panels(fig)        # -> "(a) depth", "(b) velocity"
    """
    import re

    axes = list(axes if axes is not None else fig.axes)
    axes = [ax for ax in axes if ax.get_visible()]
    if len(axes) < 2:
        return axes                      # a single panel needs no letter
    pat = re.compile(r"^\s*[\(\[]?[A-Za-z][\)\].]\s+")
    for i, ax in enumerate(axes):
        title = ax.get_title()
        if pat.match(title):
            continue                     # already labelled
        letter = chr(ord(start) + i)
        if upper:
            letter = letter.upper()
        ax.set_title(fmt.format(letter) + title)
    return axes


def _frame_legend(leg):
    """Apply the publication thin-gray legend frame (shared mechanics)."""
    leg.get_frame().set_edgecolor("#BBBBBB")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)
    return leg


def figure_legend(fig, extra=None, ncol=None, reserve=0.12, pad=0.02):
    """ONE legend row underneath the whole figure (publication convention:
    thin light frame separating it from the caption).  Collects entries
    from all axes (de-duplicated, in-axes legends removed) plus ``extra``
    [(label, handle), ...] proxies.

    The reserved strip is MEASURED, not assumed.  ``reserve`` is only a
    lower bound: after the legend is placed we render once and grow the
    bottom margin to whatever the legend plus the axes' own decorations
    actually occupy.  A fixed reserve silently fails as soon as the legend
    needs a second row, the font grows (``use("presentation")``), or an
    x-label carries a unit -- the legend then sits ON the x-label, which is
    the one collision this layout exists to prevent.

    ``pad`` is the clear gap left between the top of the legend and the
    bottom of the lowest x-label, in figure fractions.
    """
    handles, labels = _collect_handles_labels(fig.axes, extra)
    if not handles:
        return None
    ncol = ncol or min(len(handles), 5)
    fig.subplots_adjust(bottom=reserve + 0.06)
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), ncol=ncol,
                     frameon=True, fancybox=True, borderpad=0.6)
    _frame_legend(leg)

    # Measure the REAL clearance and grow the margin until the legend clears
    # the lowest x-label by ``pad``.  Iterated rather than solved in one shot:
    # moving the axes changes both the legend's wrapped height and the tight
    # bbox, so a single analytic estimate is systematically short (measured:
    # fine at thesis/publication sizes, ~0.011 of the figure INTO the x-label
    # at presentation size).  Two or three passes converge.
    try:
        inv = fig.transFigure.inverted()
        for _ in range(4):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            top = leg.get_window_extent(renderer).transformed(inv).y1
            lowest = None
            for ax in fig.axes:
                if not ax.get_visible():
                    continue
                tb = ax.get_tightbbox(renderer)
                if tb is None:
                    continue
                y0 = tb.transformed(inv).y0
                lowest = y0 if lowest is None else min(lowest, y0)
            if lowest is None:
                break
            deficit = pad - (lowest - top)
            if deficit <= 1e-4:
                break
            want = fig.subplotpars.bottom + deficit
            if want > _LEGEND_MAX_BOTTOM:
                # The legend needs more than this share of the figure -- at
                # that point the plot is mostly legend and growing the margin
                # further is not the right answer.  Say so instead of leaving
                # a silent overlap: the caller wants fewer entries or more
                # columns.
                import warnings
                warnings.warn(
                    f"figure_legend: {len(handles)} entries in {ncol} column(s) "
                    f"need {want:.0%} of the figure height; capped at "
                    f"{_LEGEND_MAX_BOTTOM:.0%}, so the legend may overlap the "
                    "x-label. Pass a larger ncol or use fewer entries.",
                    stacklevel=2)
                fig.subplots_adjust(bottom=_LEGEND_MAX_BOTTOM)
                break
            fig.subplots_adjust(bottom=want)
    except Exception:
        # A backend without a usable renderer keeps the historical fixed
        # reserve rather than failing the plot -- the legend is cosmetic,
        # the figure is not.
        pass
    return leg


@contextmanager
def apply_style(**overrides):
    """Temporarily patch :data:`CONFIG` and matplotlib rcParams.

    Any keyword matching a :class:`PlotConfig` field replaces the config
    value for the duration of the block; all changes revert on exit. Use
    without arguments to get the publication-ready defaults.
    """
    # Snapshot old CONFIG state so we can restore it.
    field_names = {f.name for f in _fields(PlotConfig)}
    unknown = set(overrides) - field_names
    if unknown:
        raise TypeError(
            f"apply_style() got unexpected kwargs: {sorted(unknown)}. "
            f"known PlotConfig fields: {sorted(field_names)}"
        )

    old_values = {name: getattr(CONFIG, name) for name in field_names}
    for name, value in overrides.items():
        setattr(CONFIG, name, value)

    # Lazy matplotlib import — only pay the cost when a user actually
    # enters the styling block (which implies they're about to plot).
    import matplotlib as mpl

    rc_patch = _as_rcparams(CONFIG)
    with mpl.rc_context(rc_patch):
        try:
            yield CONFIG
        finally:
            for name, value in old_values.items():
                setattr(CONFIG, name, value)


def reset_config() -> None:
    """Restore every :data:`CONFIG` attribute to its dataclass default.

    Handy in test fixtures (``conftest.py`` autouse) to keep session state
    from leaking between tests.
    """
    default = PlotConfig()
    for f in _fields(PlotConfig):
        setattr(CONFIG, f.name, getattr(default, f.name))
