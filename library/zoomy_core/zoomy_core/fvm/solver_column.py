"""Column-integrating solver for depth-integrated equations on extruded meshes.

Extends `IMEXSolver` with numerical vertical integration. Operates on
the raw depth-integrated equations (from `sme()` or `vam()`) before
basis projection — depth integrals are evaluated numerically by summing
over vertical cells in each column.

Solver hierarchy:
    IMEXSolver (explicit flux + implicit source)
      -> ColumnIntegratingSolver
           setup_simulation — builds column structure from extruded mesh
           integrate_vertical — numerical depth integral
           depth_average — sigma-coordinate average
           partial_integrate — running integral from surface/bottom
"""

import numpy as np
import param

from zoomy_core.fvm.solver_imex_numpy import IMEXSolver
from zoomy_core.mesh.column_structure import ColumnStructure


class ColumnIntegratingSolver(IMEXSolver):
    """Solver for depth-integrated equations with numerical vertical integration.

    Operates on an extruded mesh in (x, zeta) space. The state vector
    contains u(x, zeta) at each 3D cell. Depth integrals that appear
    in the flux and source terms are evaluated numerically by summing
    over vertical cells in each column.

    Parameters
    ----------
    n_horizontal : int
        Number of horizontal inner cells.
    n_layers : int
        Number of vertical layers in the extrusion.
    """

    n_horizontal = param.Integer(default=0, doc="Number of horizontal inner cells")
    n_layers = param.Integer(default=1, bounds=(1, None),
                             doc="Number of vertical layers")

    def setup_simulation(self, mesh, model, **kwargs):
        """Build all operators + column structure from extruded mesh."""
        super().setup_simulation(mesh, model, **kwargs)
        if self.n_horizontal == 0:
            self.n_horizontal = mesh.n_inner_cells // self.n_layers
        self._columns = ColumnStructure(self.n_horizontal, self.n_layers)

    @property
    def columns(self):
        """Access the column structure (after setup_simulation)."""
        return self._columns

    def integrate_vertical(self, Q):
        """Numerical sigma-coordinate integral: integral_0^1 Q(zeta) dzeta.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)

        Returns
        -------
        ndarray, shape (n_vars, n_horizontal)
        """
        return self._columns.integrate(Q)

    def depth_average(self, Q, field_index):
        """Sigma-coordinate average of a single field.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)
        field_index : int

        Returns
        -------
        ndarray, shape (n_horizontal,)
        """
        return self._columns.integrate(Q[field_index:field_index + 1, :])[0]

    def partial_integrate(self, Q, from_top=True):
        """Running vertical integral per cell.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)
        from_top : bool
            If True, integral_zeta^1; if False, integral_0^zeta.

        Returns
        -------
        ndarray, shape (n_vars, n_3d_cells)
        """
        return self._columns.partial_integrate(Q, from_top=from_top)
