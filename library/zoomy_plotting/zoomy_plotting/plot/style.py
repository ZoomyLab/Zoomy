"""Styling configuration — the single source of truth for every visual
parameter used by the plotters.

Design rule enforced by tests:
    Nothing inside ``plot_1d / plot_2d / plot_3d`` or their helpers may
    hardcode a color, linewidth, alpha, markersize, or colormap. All such
    values come from :data:`CONFIG`. Plot functions may accept matching
    kwargs that, when non-``None``, override the config for that one call.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace, fields as _fields
from typing import Optional

import matplotlib as mpl


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
    font_size: float = 11.0
    axes_grid_1d: bool = True
    axes_grid_2d3d: bool = False
    grid_alpha: float = 0.3
    figure_figsize: tuple = (7.0, 5.0)
    figure_dpi: float = 120.0


# Module-level singleton; users edit this directly for session-wide defaults.
CONFIG = PlotConfig()


def _as_rcparams(cfg: PlotConfig) -> dict:
    """Project a :class:`PlotConfig` onto matplotlib rcParams."""
    return {
        "image.cmap": cfg.cmap,
        "font.family": cfg.font_family,
        "font.size": cfg.font_size,
        "figure.figsize": cfg.figure_figsize,
        "figure.dpi": cfg.figure_dpi,
    }


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
