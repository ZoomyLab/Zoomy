"""Zoomy plotting style — RE-EXPORT shim.

The canonical style lives in `zoomy_plotting.plot.style` (the published
plotting package, decision 2026-06-11/B): Okabe–Ito palette + semantic
colors, marker rotation, print/screen profiles, the below-figure
`figure_legend`.  This module preserves the historical import path
`zoomy_core.postprocessing.style`.
"""
from zoomy_plotting.plot.style import (  # noqa: F401
    CONFIG, PlotConfig, apply_style, use, line, resolve_color, figure_legend, label_panels,
    CYCLE, COLORS, MARKERS, MARKEVERY,
    CMAP_CONTINUOUS, CMAP_DIVERGING, CMAP_TOPO, PROFILES,
)

__all__ = [
    "CONFIG", "PlotConfig", "apply_style", "use", "line", "resolve_color",
    "figure_legend", "label_panels",
    "CYCLE", "COLORS", "MARKERS", "MARKEVERY",
    "CMAP_CONTINUOUS", "CMAP_DIVERGING", "CMAP_TOPO", "PROFILES",
]
