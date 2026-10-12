"""Column structure of extruded meshes (cell index = ``iz * n_horizontal + i_horizontal``) for
vertical integrals in sigma coordinates.
"""

from __future__ import annotations

import numpy as np


class ColumnStructure:
    """Column-to-cell mapping and uniform layer weights of an extruded mesh, in sigma coordinate
    zeta in [0, 1].
    """

    def __init__(self, n_horizontal: int, n_layers: int):
        """Build the ``(n_horizontal, n_layers)`` column-cell table for ``n_horizontal`` cells and
        ``n_layers`` layers.
        """
        self.n_horizontal = n_horizontal
        self.n_layers = n_layers

        # Column cells: (n_horizontal, n_layers) global cell indices
        # column_cells[i_h, iz] = iz * n_horizontal + i_h
        self.column_cells = np.arange(
            n_horizontal * n_layers, dtype=int,
        ).reshape(n_layers, n_horizontal).T

        # Uniform layer weight in sigma coordinates: dzeta = 1 / n_layers
        self.d_zeta = 1.0 / n_layers

        # Layer midpoints in sigma coordinates (for function evaluation)
        self.zeta_midpoints = (np.arange(n_layers) + 0.5) / n_layers

    def integrate(self, Q: np.ndarray) -> np.ndarray:
        """Midpoint-rule vertical integral of ``Q`` over zeta in [0, 1].

        Args:
            Q: ``(n_vars, n_3d_cells)`` field on the extruded mesh.
        Returns:
            ``(n_vars, n_horizontal)`` integrals per column.
        """
        n_vars = Q.shape[0]
        result = np.zeros((n_vars, self.n_horizontal))
        for iz in range(self.n_layers):
            cells = self.column_cells[:, iz]
            result += Q[:, cells] * self.d_zeta
        return result

    def partial_integrate(self, Q: np.ndarray, from_top: bool = True) -> np.ndarray:
        """Running vertical integral per cell, from the surface down (``from_top``) or from the
        bottom up.

        Args:
            Q: ``(n_vars, n_3d_cells)`` field on the extruded mesh.
        Returns:
            ``(n_vars, n_3d_cells)`` running integral at each cell.
        """
        n_vars = Q.shape[0]
        result = np.zeros_like(Q)
        running = np.zeros((n_vars, self.n_horizontal))

        if from_top:
            layer_range = range(self.n_layers - 1, -1, -1)
        else:
            layer_range = range(self.n_layers)

        for iz in layer_range:
            cells = self.column_cells[:, iz]
            running += Q[:, cells] * self.d_zeta
            result[:, cells] = running

        return result

    def scatter_to_column(self, Q_horiz: np.ndarray) -> np.ndarray:
        """Broadcast ``(n_vars, n_horizontal)`` values to every layer, giving ``(n_vars,
        n_3d_cells)``.
        """
        n_vars = Q_horiz.shape[0]
        Q_3d = np.zeros((n_vars, self.n_horizontal * self.n_layers))
        for iz in range(self.n_layers):
            cells = self.column_cells[:, iz]
            Q_3d[:, cells] = Q_horiz
        return Q_3d

    def gather_layer(self, Q: np.ndarray, iz: int) -> np.ndarray:
        """Extract layer ``iz`` (0 = bottom) of ``Q`` as ``(n_vars, n_horizontal)``."""
        cells = self.column_cells[:, iz]
        return Q[:, cells]
