"""Smoke tests for 1-D plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from zoomy_plotting import MatplotlibPlotter


def test_plot_1d_returns_line_artist(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    out = plotter.plot_1d(ax, time_step=0, field="h")
    assert "line" in out
    assert len(ax.lines) >= 1
    xmin, xmax = ax.get_xlim()
    assert xmax > xmin
    plt.close(fig)


def test_plot_1d_nodes_toggle(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    out_on = plotter.plot_1d(ax, time_step=0, field="h", show_nodes=True)
    assert "nodes" in out_on
    ax.clear()
    out_off = plotter.plot_1d(ax, time_step=0, field="h", show_nodes=False)
    assert "nodes" not in out_off
    plt.close(fig)


def test_plot_1d_accepts_int_field(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    plotter.plot_1d(ax, time_step=1, field=1)
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_1d_rejects_wrong_dim(store_2d_quad):
    plotter = MatplotlibPlotter(store_2d_quad)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="plot_1d requires dim==1"):
        plotter.plot_1d(ax)
    plt.close(fig)


def test_plot_dispatcher_picks_1d(store_1d):
    plotter = MatplotlibPlotter(store_1d)
    fig, ax = plt.subplots()
    out = plotter.plot(ax, time_step=0, field="h")
    assert "line" in out
    plt.close(fig)
