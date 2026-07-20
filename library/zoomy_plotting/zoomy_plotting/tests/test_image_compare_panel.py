"""#6 test_image_compare_panel — matplotlib image comparison (NEW; pins the
published-figure look).

Uses ``matplotlib.testing.compare.compare_images`` directly (Agg backend,
explicit tolerance, NO pytest plugins): ``plot_topo_field`` +
``row_legend`` / ``figure_legend`` under the print profile on the committed
2-D triangle fixture (``fixtures/series.h5``).

The reference PNG is committed; regenerate it (after an INTENDED look
change) with::

    python -c "from zoomy_plotting.tests import test_image_compare_panel as t; \\
               t.regenerate_reference()"
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import zoomy_plotting as zp
from zoomy_plotting.plot import style
from zoomy_plotting.plot.panels import line_plot, row_legend

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SERIES_H5 = os.path.join(FIXTURES, "series.h5")
REFERENCE = os.path.join(FIXTURES, "topo_panel_ref.png")

#: RMS tolerance for compare_images — lenient enough for freetype/matplotlib
#: point-release drift, tight enough to catch palette/layout regressions.
TOL = 10.0
DPI = 100


def draw_panel():
    """The pinned figure: topography+field plan view (left, with contours and
    dry-cell masking) and labeled water-level curves with a shared row legend
    + figure legend (right). Deterministic: committed fixture + print profile.
    """
    import matplotlib.pyplot as plt

    zp.use("print")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    with zp.read_hdf5(SERIES_H5) as store:
        from zoomy_core.postprocessing.mesh_plots import plot_topo_field
        out = plot_topo_field(ax0, store, topo="b", field="h", time_step=0,
                              hide_below=1e-9, contours=3, colorbar=True)
        assert "mappable" in out
        ax0.set_title("topo + h")

        centers = store.vertices[store.cells].mean(axis=1)
        order = np.argsort(centers[:, 0])
        x = centers[order, 0]
        b = np.asarray(store.get_cell(0, "b"))[order]
        h0 = np.asarray(store.get_cell(0, "h"))[order]
        h2 = np.asarray(store.get_cell(2, "h"))[order]
        line_plot(ax1, [
            {"x": x, "y": b + h0, "label": "surface t0", "role": "water"},
            {"x": x, "y": b + h2, "label": "surface t2", "role": "reference",
             "ls": "--"},
        ], xlabel="x", ylabel="b + h", title="water level")

    leg_row = row_legend(fig, [ax0, ax1])
    leg_fig = style.figure_legend(
        fig, extra=[("interface", style.line("interface"))])
    return fig, leg_row, leg_fig


def _render(fig, path):
    fig.savefig(path, dpi=DPI)


def regenerate_reference():
    import matplotlib
    matplotlib.use("Agg")
    fig, _, _ = draw_panel()
    _render(fig, REFERENCE)
    print("reference written ->", REFERENCE)


def test_topo_panel_matches_reference(tmp_path):
    from matplotlib.testing.compare import compare_images
    import matplotlib.pyplot as plt

    assert os.path.exists(REFERENCE), (
        f"missing committed reference {REFERENCE} — regenerate with "
        f"regenerate_reference() after an intended look change")

    fig, leg_row, leg_fig = draw_panel()
    # the legends are real, de-duplicated, and carry the row labels
    assert leg_row is not None and leg_fig is not None
    row_labels = {t.get_text() for t in leg_row.get_texts()}
    assert {"surface t0", "surface t2"} <= row_labels
    fig_labels = {t.get_text() for t in leg_fig.get_texts()}
    assert "interface" in fig_labels

    actual = str(tmp_path / "topo_panel.png")
    _render(fig, actual)
    plt.close(fig)

    err = compare_images(REFERENCE, actual, tol=TOL)
    assert err is None, err
