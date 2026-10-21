"""Module `zoomy_core.model.aux_boundary_conditions`."""

import param
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.boundary_conditions import BoundaryCondition


class Extrapolation(BoundaryCondition):
    """
    Extrapolation for Auxiliary Variables.
    Behavior: Sets Ghost Cell Qaux = Interior Cell Qaux.
    """

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        # Return Qaux directly (No swapping needed in the model)
        """Compute boundary condition."""
        return ZArray(Qaux)


class Lambda(BoundaryCondition):
    """
    Lambda BC for Auxiliary Variables.
    Allows arbitrary modification of Qaux via user-defined functions.
    """

    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Qaux)
        for k, func in self.prescribe_fields.items():
            Qout[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return Qout
