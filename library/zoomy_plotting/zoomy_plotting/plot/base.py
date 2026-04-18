"""Backend-agnostic plotter base class.

Holds a :class:`SimulationStore` and lazily precomputes expensive mesh
derivatives (boundary faces for 3-D meshes, fan triangulation for 2-D) so
repeated calls to the plot methods are cheap.
"""

from __future__ import annotations

from functools import cached_property
from typing import List, Tuple

import numpy as np

from ..mesh.faces import boundary_faces_3d, triangulate
from ..store import SimulationStore


class BasePlotter:
    """Shared state for any backend-specific plotter.

    Subclasses (e.g. :class:`MatplotlibPlotter`) implement ``plot_1d``,
    ``plot_2d``, ``plot_3d``, and a ``plot`` dispatcher that picks based
    on ``self.store.dim``.
    """

    def __init__(self, store: SimulationStore):
        self.store = store

    # --- cached mesh derivatives ----------------------------------------
    @cached_property
    def _triangulation(self):
        """For 2-D plotters: fan triangulation ``(i, j, k, parent_cells)``."""
        return triangulate(self.store.cells)

    @cached_property
    def _boundary_3d(self) -> Tuple[List[List[int]], List[int]]:
        """For 3-D plotters: ``(faces, parent_cells)`` of the boundary."""
        return boundary_faces_3d(self.store.cells, self.store.cell_type)

    # --- convenience ----------------------------------------------------
    def _cell_values(self, time_step: int, field) -> np.ndarray:
        """Fetch one cell-centered 1-D array at ``time_step`` for ``field``."""
        return np.asarray(self.store.get_cell(time_step, field))

    def _time_label(self, time_step: int) -> str:
        """Human-readable label for the current frame, preferring real time.

        If the store has a ``times`` array, format ``t = <value>`` with
        enough precision to distinguish adjacent snapshots. Otherwise fall
        back to the integer step number. Snippets / custom titles can
        reuse this so matplotlib and plotly titles stay consistent.
        """
        t_arr = self.store.times
        ts = int(time_step)
        if t_arr is None or len(t_arr) == 0:
            return f"step {ts}"
        ts = max(0, min(ts, len(t_arr) - 1))
        t = float(t_arr[ts])
        # Pick precision from the typical gap between snapshots so two
        # adjacent titles aren't identical. Fall back to 3 decimals.
        if len(t_arr) >= 2:
            gaps = np.diff(np.asarray(t_arr))
            gap = float(np.median(gaps)) if gaps.size else 0.0
            if gap > 0:
                import math
                decimals = max(2, min(6, int(math.ceil(-math.log10(gap))) + 1))
                return f"t = {t:.{decimals}f}"
        return f"t = {t:.3f}"

    def _field_label(self, field) -> str:
        """Human-readable label for plot titles / colorbars.

        ``store.field`` stores ``name -> index``; this reverses the map.
        """
        if isinstance(field, str):
            return field
        name = getattr(field, "name", None)
        if isinstance(name, str):
            return name
        # int: try reverse lookup on the Zstruct.
        if isinstance(field, (int, np.integer)):
            for k, v in self.store.field.items():
                if int(v) == int(field):
                    return str(k)
            return f"field[{int(field)}]"
        return str(field)
