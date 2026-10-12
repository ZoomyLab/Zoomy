"""The C1 bump of the test matrix: shallow water (SME level 0) on [0, 10]^2,
walls, at rest with a Gaussian hump in the depth.

Model -> SystemModel -> NumericalSystemModel; every backend takes the NSM."""
import numpy as np

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel


def bump(order=2, nx=40):
    """Returns ``(mesh, nsm)``."""
    # 1. Model: SME level 0 = shallow water; dimension 3 = (x, y) + vertical.
    walls = [FromModel(tag=t, definition="wall") for t in ("left", "right", "bottom", "top")]
    model = SME(level=0, dimension=3,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BoundaryConditions(walls))

    # 2. SystemModel: the operator slots (flux, nonconservative_matrix, source, ...).
    sm = SystemModel.from_model(model)

    def ic(x):                                   # state (b, h, q_x, q_y)
        h = 0.5 + 0.05 * np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0)
        return np.array([0.0, h, 0.0, 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)

    # 3. NumericalSystemModel: reconstruction, Riemann solver (default: Rusanov).
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod",
                                              variables="conservative"))
    mesh = LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx)
    return mesh, nsm
