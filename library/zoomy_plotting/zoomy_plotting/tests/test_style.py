"""Tests that enforce the 'all styling from CONFIG' design rule.

Two layers:
  1. Static AST check: scan plot/matplotlib.py for hardcoded literal
     values passed to a blacklisted set of matplotlib kwargs.
  2. Behavior check: mutating CONFIG / using apply_style() actually changes
     the artist properties produced by the plotters.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zoomy_plotting.plot.matplotlib as mpl_plotters
from zoomy_plotting import CONFIG, MatplotlibPlotter, apply_style
from zoomy_plotting.plot.style import PlotConfig, reset_config


BLACKLISTED_KWARGS = {
    "color", "colors", "edgecolor", "edgecolors",
    "linewidth", "linewidths",
    "markersize",
    "cmap",
    "alpha",
    # Note: 'facecolor' is set from computed arrays (not literals), so it's fine.
}


def _path_of_plot_matplotlib() -> Path:
    return Path(inspect.getfile(mpl_plotters))


def test_no_hardcoded_styles_in_plot_matplotlib():
    """No literal color/linewidth/alpha/cmap/markersize inside plot_* calls."""
    src = _path_of_plot_matplotlib().read_text()
    tree = ast.parse(src)

    offenders = []

    class KwargVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            for kw in node.keywords:
                if kw.arg in BLACKLISTED_KWARGS and _is_hardcoded(kw.value):
                    offenders.append(
                        f"{node.func.__class__.__name__} "
                        f"passes {kw.arg}={ast.unparse(kw.value)!r} "
                        f"at line {kw.value.lineno}"
                    )
            self.generic_visit(node)

    KwargVisitor().visit(tree)
    assert not offenders, (
        "plot/matplotlib.py contains hardcoded style literals — "
        "every color/linewidth/alpha/cmap/markersize must come from "
        "CONFIG or a kwarg. Offenders:\n  " + "\n  ".join(offenders)
    )


def _is_hardcoded(node):
    """True iff the AST node is a literal (str, number) or tuple/list of literals.

    Accepts ``None`` and names like ``"none"`` (the matplotlib no-draw keyword)
    only when they come from :data:`CONFIG`. We treat string *literals* as
    hardcoded; the only exception is the lowercase string ``"none"``, which is
    matplotlib's 'draw nothing' flag and is conventional plumbing, not a
    style choice.
    """
    if isinstance(node, ast.Constant):
        if node.value is None:
            return False
        if isinstance(node.value, str) and node.value == "none":
            return False
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_hardcoded(elt) for elt in node.elts)
    return False


def test_apply_style_restores_on_exit():
    """Context manager must restore CONFIG to its prior state."""
    CONFIG.cmap = "plasma"
    with apply_style(cmap="magma"):
        assert CONFIG.cmap == "magma"
    assert CONFIG.cmap == "plasma"


def test_apply_style_rejects_unknown_kwargs():
    import pytest

    with pytest.raises(TypeError, match="unexpected kwargs"):
        with apply_style(not_a_field="x"):
            pass


def test_mutating_config_affects_plot_2d(store_2d_quad):
    """Changing CONFIG.cmap must reach into the produced PolyCollection."""
    plotter = MatplotlibPlotter(store_2d_quad)
    fig, ax = plt.subplots()

    CONFIG.cmap = "viridis"
    out_viridis = plotter.plot_2d(ax, time_step=0, field="h")
    # Extract a face color from the PolyCollection.
    fc_viridis = out_viridis["mesh"].get_facecolor().copy()

    ax.clear()
    CONFIG.cmap = "plasma"
    out_plasma = plotter.plot_2d(ax, time_step=0, field="h")
    fc_plasma = out_plasma["mesh"].get_facecolor().copy()

    # If the cmap reached the artist, the face colors differ.
    assert not np.allclose(fc_viridis, fc_plasma), (
        "CONFIG.cmap change did not propagate to the PolyCollection's colors "
        "— plot_2d is probably hardcoding a colormap."
    )
    plt.close(fig)


def test_apply_style_overrides_affect_plot_1d(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    with apply_style(line_linewidth=7.5):
        out = plotter.plot_1d(ax, time_step=0, field="h", show_nodes=False)
        assert out["line"].get_linewidth() == 7.5
    plt.close(fig)


def test_reset_config_restores_defaults():
    CONFIG.cmap = "inferno"
    reset_config()
    assert CONFIG.cmap == PlotConfig().cmap


# ── REQ-98: selectable thesis / presentation templates ──────────────────────

_FONT_RC = ("font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize")


def test_presentation_profile_clamps_all_fonts_to_floor():
    """Every text element is >= 14 pt under the ``presentation`` template,
    even the ticks/legend that ``print`` shrinks below the base size."""
    from zoomy_plotting.plot.style import _as_rcparams, PROFILES
    try:
        CONFIG.profile = "presentation"
        rc = _as_rcparams(CONFIG)
        floor = PROFILES["presentation"]["min_pt"]
        for key in _FONT_RC:
            assert rc[key] >= floor, f"{key}={rc[key]} below {floor}pt floor"
    finally:
        reset_config()


def test_thesis_profile_matches_print():
    """``thesis`` is the final-print template — identical font sizes to the
    original ``print`` profile (non-breaking rename-by-alias)."""
    from zoomy_plotting.plot.style import _as_rcparams
    try:
        CONFIG.profile = "thesis"
        rc_thesis = {k: _as_rcparams(CONFIG)[k] for k in _FONT_RC}
        CONFIG.profile = "print"
        rc_print = {k: _as_rcparams(CONFIG)[k] for k in _FONT_RC}
        assert rc_thesis == rc_print
    finally:
        reset_config()
