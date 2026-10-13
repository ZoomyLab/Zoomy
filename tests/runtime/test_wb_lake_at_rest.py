"""Lake at rest over topography — the well-balancing gate."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL, march


@pytest.mark.small
@pytest.mark.jax
def test_lake_at_rest_over_bump(overwrite):
    """Topography gate: mass conservation is BLIND to well-balancing."""
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    model = models.swe(dimension=2, bc="wall")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lake_at_rest_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter=Limiter.minmod),
        riemann=PositiveNonconservativeRusanov)
    print(describe(nsm))
    set_state_width(nsm)
    assert nsm.update_variables is None

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=100)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=1.0)
    elapsed = time.perf_counter() - t0

    eta, u = Q[0] + Q[1], Q[2] / Q[1]
    assert np.abs(eta - eta[0]).max() < 1e-12, "lake tilted — WB lost"
    assert np.abs(u).max() < 1e-12, "spurious currents over the bed"
    refs.check("wb_lake", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("wb_lake", elapsed, overwrite)
