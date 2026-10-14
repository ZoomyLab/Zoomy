"""A 2:1 dam break in 1-D: shallow water (SME level 0) on [0, 10], open ends, second order.

Model -> SystemModel -> NumericalSystemModel; every backend takes the NSM."""
import numpy as np

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel


def dambreak(order=2, n=300):
    """Returns ``(mesh, nsm)``; the state is ``[b, h, q_0]``."""
    # A 1-D case is the dimension=2 model (x and the vertical) on a 1-D mesh.
    model = SME(
        level=0, dimension=2,
        closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BC.BoundaryConditions(
            [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
    )
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 2.0 if x[0] < 5.0 else 1.0, 0.0]))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n), nsm
