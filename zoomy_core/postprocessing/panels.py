"""Composable single-axes plotting building blocks — RE-EXPORT shim.

The canonical implementations live in `zoomy_plotting.plot.panels` (the
published plotting package). This module preserves a stable import path
`zoomy_core.postprocessing.panels` so thesis/tutorial drivers can drop
these blocks into `plt.subplots` / `plt.subplot_mosaic` layouts:

    from zoomy_core.postprocessing.panels import (
        line_plot, profile_plot, row_legend,
    )

Mirrors the `zoomy_core.postprocessing.style` shim.
"""
from zoomy_plotting.plot.panels import (  # noqa: F401
    line_plot, profile_plot, row_legend,
)

__all__ = ["line_plot", "profile_plot", "row_legend"]
