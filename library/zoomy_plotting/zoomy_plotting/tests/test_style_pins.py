"""#5 test_style_pins — the style system pins (absorbs test_style + the
test_panels re-export test):

* template profiles (publication/thesis/screen/presentation + print alias),
* figsize tokens + grid scaling + ``use``/``subplots`` sizing,
* CYCLE / COLORS semantic palette identity,
* the "all styling from CONFIG" invariant (static AST + one behavioral pin),
* the load-bearing ``zoomy_core.postprocessing.style`` re-export path.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from zoomy_plotting import (
    CONFIG, CYCLE, COLORS, MatplotlibPlotter, SimulationStore, Zstruct,
    apply_style, colors, figsize, subplots, use,
)
from zoomy_plotting.plot.style import (
    PROFILES, SIZES, PlotConfig, _as_rcparams, reset_config,
)

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

_FONT_RC = ("font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize")


# ── profiles ─────────────────────────────────────────────────────────────────

def test_named_presets_resolve_with_fonts_and_sizes():
    for name in ("publication", "thesis", "screen"):
        assert name in PROFILES and name in SIZES
        assert "cell" in SIZES[name]
        for w in ("1col", "2col", "full"):
            assert w in SIZES[name]["widths"]
    assert "print" in PROFILES  # the original alias survives


def test_thesis_profile_matches_print_fonts():
    CONFIG.profile = "thesis"
    rc_thesis = {k: _as_rcparams(CONFIG)[k] for k in _FONT_RC}
    CONFIG.profile = "print"
    rc_print = {k: _as_rcparams(CONFIG)[k] for k in _FONT_RC}
    assert rc_thesis == rc_print


def test_presentation_profile_clamps_fonts_to_floor():
    CONFIG.profile = "presentation"
    rc = _as_rcparams(CONFIG)
    floor = PROFILES["presentation"]["min_pt"]
    for key in _FONT_RC:
        assert rc[key] >= floor, f"{key}={rc[key]} below {floor}pt floor"
    assert rc["figure.titlesize"] >= floor


def test_figure_titlesize_stays_small():
    CONFIG.profile = "publication"
    rc = _as_rcparams(CONFIG)
    assert rc["figure.titlesize"] == rc["font.size"] + 1  # never a headline


# ── figsize tokens ───────────────────────────────────────────────────────────

def test_figsize_grid_scales_with_ncols_nrows():
    cw, ch = SIZES["publication"]["cell"]
    assert figsize("publication", ncols=3, nrows=1) == (cw * 3, ch)
    assert figsize("publication", ncols=1, nrows=2) == (cw, ch * 2)
    # same layout reflows narrower in the thesis text column
    assert figsize("thesis", ncols=2)[0] < figsize("screen", ncols=2)[0]


def test_figsize_width_token_and_active_preset_default():
    assert figsize("thesis", width="2col") == SIZES["thesis"]["widths"]["2col"]
    use("thesis")
    try:
        cw, ch = SIZES["thesis"]["cell"]
        assert figsize(ncols=2) == (cw * 2, ch)
    finally:
        reset_config()


def test_use_and_subplots_apply_preset_and_size():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    use("thesis")
    try:
        assert tuple(mpl.rcParams["figure.figsize"]) \
            == SIZES["thesis"]["widths"]["full"]
        fig, axes = subplots(1, 3, preset="thesis")
        assert CONFIG.profile == "thesis"
        assert axes.shape == (3,)
        cw, ch = SIZES["thesis"]["cell"]
        assert tuple(fig.get_size_inches()) == (cw * 3, ch)
        plt.close(fig)
        # apply=False keeps the active preset; explicit figsize wins
        CONFIG.profile = "publication"
        fig, ax = subplots(preset="screen", apply=False, figsize=(4.0, 3.0))
        assert CONFIG.profile == "publication"
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)
        plt.close(fig)
    finally:
        reset_config()


# ── CYCLE / COLORS ───────────────────────────────────────────────────────────

def test_semantic_colors_and_cycle():
    assert colors.experiment == COLORS["experiment"]
    assert colors.reference == COLORS["reference"]
    assert colors.analytic == COLORS["analytic"]
    assert len({colors.experiment, colors.reference, colors.analytic}) == 3
    assert colors.cycle == list(CYCLE)
    assert CYCLE[0] == "#0072B2"                 # Okabe–Ito rotation
    assert COLORS["water"] == "#56B4E9"          # coupling roles untouched
    assert colors["#123456"] == "#123456"        # fall-through
    with pytest.raises(AttributeError):
        colors.nonexistent_role


def test_colors_dict_identity_preserved():
    import zoomy_plotting
    from zoomy_plotting.plot.style import COLORS as canon
    assert zoomy_plotting.COLORS is canon


# ── the "all styling from CONFIG" invariant ──────────────────────────────────

BLACKLISTED_KWARGS = {
    "color", "colors", "edgecolor", "edgecolors", "linewidth", "linewidths",
    "markersize", "cmap", "alpha",
}


def _is_hardcoded(node):
    if isinstance(node, ast.Constant):
        if node.value is None:
            return False
        if isinstance(node.value, str) and node.value == "none":
            return False  # matplotlib's no-draw flag, plumbing not styling
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_hardcoded(elt) for elt in node.elts)
    return False


def test_no_hardcoded_styles_in_plot_matplotlib():
    import zoomy_plotting.plot.matplotlib as mpl_plotters
    src = Path(inspect.getfile(mpl_plotters)).read_text()
    offenders = []

    class KwargVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg in BLACKLISTED_KWARGS and _is_hardcoded(kw.value):
                    offenders.append(
                        f"{kw.arg}={ast.unparse(kw.value)!r} "
                        f"at line {kw.value.lineno}")
            self.generic_visit(node)

    KwargVisitor().visit(ast.parse(src))
    assert not offenders, (
        "plot/matplotlib.py hardcodes style literals — every "
        "color/linewidth/alpha/cmap/markersize must come from CONFIG:\n  "
        + "\n  ".join(offenders))


def _tiny_store():
    n = 8
    verts = np.linspace(0.0, 1.0, n + 1)[:, None]
    cells = np.column_stack([np.arange(n), np.arange(1, n + 1)])
    return SimulationStore(
        dim=1, cell_type="line", vertices=verts, cells=cells,
        field=Zstruct({"h": 0}),
        _cell_reader=lambda t, i: np.linspace(1.0, 2.0, n),
    )


def test_config_reaches_artists():
    """Behavioral half of the invariant: an apply_style override must land
    on the produced artist (CONFIG is live, not decorative)."""
    import matplotlib.pyplot as plt
    plotter = MatplotlibPlotter(_tiny_store())
    fig, ax = plt.subplots()
    with apply_style(line_linewidth=7.5):
        out = plotter.plot_1d(ax, time_step=0, field="h", show_nodes=False)
        assert out["line"].get_linewidth() == 7.5
    plt.close(fig)


def test_apply_style_scoping_and_reset():
    CONFIG.cmap = "plasma"
    with apply_style(cmap="magma"):
        assert CONFIG.cmap == "magma"
    assert CONFIG.cmap == "plasma"
    with pytest.raises(TypeError, match="unexpected kwargs"):
        with apply_style(not_a_field="x"):
            pass
    reset_config()
    assert CONFIG.cmap == PlotConfig().cmap


# ── the zoomy_core re-export path (6 thesis deliverables ride on it) ─────────

def test_zoomy_core_style_reexport():
    zc_style = pytest.importorskip("zoomy_core.postprocessing.style")
    zc_panels = pytest.importorskip("zoomy_core.postprocessing.panels")
    from zoomy_plotting.plot import style as canon_style
    from zoomy_plotting.plot import panels as canon_panels
    assert zc_style.COLORS is canon_style.COLORS
    assert zc_style.figure_legend is canon_style.figure_legend
    assert zc_style.line is canon_style.line
    assert zc_panels.line_plot is canon_panels.line_plot
    assert zc_panels.profile_plot is canon_panels.profile_plot
    assert zc_panels.row_legend is canon_panels.row_legend
