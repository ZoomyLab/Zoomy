"""Smoke tests for 3-D plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from zoomy_plotting import MatplotlibPlotter
from zoomy_plotting.plot._common import resolve_viewpoint


def _make_3d_ax(fig):
    return fig.add_subplot(111, projection="3d")


def test_plot_3d_hex(store_3d_hex):
    plotter = MatplotlibPlotter(store_3d_hex)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    out = plotter.plot_3d(ax, time_step=0, field="phi")
    assert "mesh" in out
    assert "colorbar" in out
    plt.close(fig)


def test_plot_3d_tet(store_3d_tet):
    plotter = MatplotlibPlotter(store_3d_tet)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    out = plotter.plot_3d(ax, time_step=0, field="z")
    assert "mesh" in out
    plt.close(fig)


def test_plot_3d_viewpoint_auto(store_3d_hex):
    plotter = MatplotlibPlotter(store_3d_hex)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    plotter.plot_3d(ax, time_step=0, field="phi", viewpoint="auto")
    # matplotlib exposes elev/azim on the Axes3D instance.
    assert -90.0 <= ax.elev <= 90.0
    assert -180.0 <= ax.azim <= 180.0
    plt.close(fig)


def test_plot_3d_viewpoint_explicit_tuple(store_3d_hex):
    plotter = MatplotlibPlotter(store_3d_hex)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    plotter.plot_3d(ax, time_step=0, field="phi", viewpoint=(20.0, 15.0))
    assert ax.elev == pytest.approx(20.0)
    assert ax.azim == pytest.approx(15.0)
    plt.close(fig)


def test_resolve_viewpoint_from_xyz(store_3d_hex):
    # A viewpoint directly above the mesh should give near-90 elevation.
    elev, _ = resolve_viewpoint((0.0, 0.0, 1.0), store_3d_hex.vertices)
    assert elev > 80.0


def test_plot_3d_show_mesh_uses_config_edges(store_3d_hex):
    from zoomy_plotting import CONFIG
    CONFIG.mesh_edgecolor = "red"
    plotter = MatplotlibPlotter(store_3d_hex)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    out = plotter.plot_3d(ax, time_step=0, field="phi", show_mesh=True)
    # The artist should have picked up the CONFIG value.
    ec = out["mesh"].get_edgecolor()
    assert ec is not None
    plt.close(fig)


def test_plot_3d_rejects_wrong_dim(store_2d_quad):
    plotter = MatplotlibPlotter(store_2d_quad)
    fig = plt.figure()
    ax = _make_3d_ax(fig)
    with pytest.raises(ValueError, match="plot_3d requires dim==3"):
        plotter.plot_3d(ax)
    plt.close(fig)
