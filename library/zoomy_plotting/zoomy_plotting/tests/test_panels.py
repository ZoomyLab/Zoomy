"""Smoke + behaviour tests for the composable panel building blocks.

These primitives draw into a caller-owned ``ax`` and never add their own
legend; a shared :func:`row_legend` (the missing counterpart of
:func:`figure_legend`) collects the row's labels into one centered legend.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from zoomy_plotting.plot.panels import line_plot, profile_plot, row_legend
from zoomy_plotting.plot.style import COLORS, apply_style


def test_mosaic_line_profile_row_legends(tmp_path):
    """2x2 mosaic: line_plot row 0, profile_plot row 1, a row_legend under
    EACH row; save a png. Asserts no error and the legends are real."""
    x = np.linspace(0.0, 1.0, 50)
    sigma = np.linspace(0.0, 1.0, 25)  # depth fraction

    with apply_style():
        fig, axd = plt.subplot_mosaic(
            [["tl", "tr"], ["bl", "br"]],
            figsize=(8.0, 6.0),
            gridspec_kw={"hspace": 0.55, "wspace": 0.3},
        )

        # Row 0 — line plots (role-colored + prop-cycle).
        line_plot(
            axd["tl"],
            [
                {"x": x, "y": np.sin(2 * np.pi * x), "label": "free surface",
                 "role": "water"},
                {"x": x, "y": np.cos(2 * np.pi * x), "label": "bed",
                 "role": "reference", "ls": "--"},
            ],
            xlabel="x", ylabel="z", title="surface vs bed",
        )
        line_plot(
            axd["tr"],
            [
                (x, x ** 2, "quadratic"),
                {"x": x, "y": x ** 0.5, "label": "sqrt", "marker": "o"},
            ],
            xlabel="x", ylabel="f(x)", logy=False, title="misc curves",
        )

        # Row 1 — vertical velocity profiles (u on x, sigma on y).
        profile_plot(
            axd["bl"],
            [
                {"values": sigma, "coord": sigma, "label": "linear",
                 "role": "resolved"},
                {"values": sigma ** 2, "coord": sigma, "label": "parabolic",
                 "role": "reduced", "ls": ":"},
            ],
            xlabel="u", ylabel=r"$\sigma$", title="profiles",
        )
        profile_plot(
            axd["br"],
            [(np.sqrt(sigma), sigma, "sqrt profile")],
            xlabel="u", ylabel=r"$\sigma$", title="profile",
        )

        # One legend centered under EACH row.
        leg_top = row_legend(fig, [axd["tl"], axd["tr"]])
        leg_bot = row_legend(fig, [axd["bl"], axd["br"]])

        out = tmp_path / "mosaic.png"
        fig.savefig(out)
        plt.close(fig)

    assert out.exists() and out.stat().st_size > 0
    # Two distinct row legends got created (not None, not the same object).
    assert leg_top is not None and leg_bot is not None
    assert leg_top is not leg_bot
    # Top row legend carries exactly the two row-0 labels.
    assert {t.get_text() for t in leg_top.get_texts()} == {"free surface", "bed",
                                                            "quadratic", "sqrt"}


def test_line_plot_role_resolves_to_palette_color():
    """A ``role`` must resolve to its COLORS hex on the drawn artist."""
    fig, ax = plt.subplots()
    (ln,) = line_plot(ax, [{"x": [0, 1], "y": [0, 1], "role": "water",
                            "label": "w"}])
    # matplotlib normalises hex to rgba; compare via to_rgba.
    from matplotlib.colors import to_rgba
    assert to_rgba(ln.get_color()) == to_rgba(COLORS["water"])
    plt.close(fig)


def test_panels_do_not_add_own_legend():
    """Building blocks attach labels but never create an in-axes legend."""
    fig, ax = plt.subplots()
    line_plot(ax, [{"x": [0, 1], "y": [0, 1], "label": "a"}])
    assert ax.get_legend() is None
    # ... yet the label is available for a shared legend to collect.
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["a"]
    plt.close(fig)


def test_row_legend_empty_row_returns_none():
    fig, ax = plt.subplots()
    line_plot(ax, [{"x": [0, 1], "y": [0, 1]}])  # no label
    assert row_legend(fig, [ax]) is None
    plt.close(fig)


def test_profile_plot_orientation_swaps_axes():
    """vertical -> (values, coord); horizontal -> (coord, values)."""
    vals = np.array([0.0, 0.5, 1.0])
    coord = np.array([0.0, 0.5, 1.0])
    skew = np.array([0.1, 0.2, 0.9])  # break symmetry so x/y differ

    fig, ax = plt.subplots()
    (lv,) = profile_plot(ax, [{"values": skew, "coord": coord}],
                         orientation="vertical")
    xdata_v, ydata_v = lv.get_xdata(), lv.get_ydata()
    plt.close(fig)

    fig, ax = plt.subplots()
    (lh,) = profile_plot(ax, [{"values": skew, "coord": coord}],
                         orientation="horizontal")
    xdata_h, ydata_h = lh.get_xdata(), lh.get_ydata()
    plt.close(fig)

    assert np.allclose(xdata_v, skew) and np.allclose(ydata_v, coord)
    assert np.allclose(xdata_h, coord) and np.allclose(ydata_h, skew)


def test_zoomy_core_style_reexport_still_works():
    """Regression: the historical zoomy_core.postprocessing.style path and
    the new panels shim both import without error."""
    zc_style = pytest.importorskip("zoomy_core.postprocessing.style")
    zc_panels = pytest.importorskip("zoomy_core.postprocessing.panels")
    # Existing names still resolve to the canonical objects.
    from zoomy_plotting.plot import style as canon_style
    assert zc_style.COLORS is canon_style.COLORS
    assert zc_style.figure_legend is canon_style.figure_legend
    assert zc_style.line is canon_style.line
    # New panels are re-exported through zoomy_core too.
    from zoomy_plotting.plot import panels as canon_panels
    assert zc_panels.line_plot is canon_panels.line_plot
    assert zc_panels.profile_plot is canon_panels.profile_plot
    assert zc_panels.row_legend is canon_panels.row_legend
