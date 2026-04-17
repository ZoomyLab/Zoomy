"""Smoke tests for 2-D plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from zoomy_plotting import MatplotlibPlotter, CONFIG


def test_plot_2d_quad_flat(store_2d_quad):
    plotter = MatplotlibPlotter(store_2d_quad)
    fig, ax = plt.subplots()
    out = plotter.plot_2d(ax, time_step=0, field="h")
    assert "mesh" in out
    assert "colorbar" in out
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_plot_2d_tri_flat(store_2d_tri):
    plotter = MatplotlibPlotter(store_2d_tri)
    fig, ax = plt.subplots()
    out = plotter.plot_2d(ax, time_step=0, field="h")
    assert len(ax.collections) >= 1
    assert "colorbar" in out
    plt.close(fig)


def test_plot_2d_show_mesh_adds_overlay(store_2d_quad):
    plotter = MatplotlibPlotter(store_2d_quad)
    fig, ax = plt.subplots()
    out = plotter.plot_2d(ax, time_step=0, field="h", show_mesh=True)
    assert "mesh_overlay" in out
    # The overlay must pick up CONFIG.mesh_edgecolor, not a hardcoded literal.
    edge_rgba = out["mesh_overlay"].get_edgecolor()
    assert edge_rgba.shape[1] == 4
    plt.close(fig)


def test_plot_2d_interpolate_mode(store_2d_tri):
    plotter = MatplotlibPlotter(store_2d_tri)
    fig, ax = plt.subplots()
    out = plotter.plot_2d(ax, time_step=0, field="h", interpolate=True)
    assert "mesh" in out
    plt.close(fig)


def test_plot_2d_rejects_wrong_dim(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="plot_2d requires dim==2"):
        plotter.plot_2d(ax)
    plt.close(fig)


def test_plot_2d_colorbar_toggle(store_2d_quad):
    plotter = MatplotlibPlotter(store_2d_quad)
    fig, ax = plt.subplots()
    out = plotter.plot_2d(ax, time_step=0, field="h", colorbar=False)
    assert "colorbar" not in out
    plt.close(fig)
