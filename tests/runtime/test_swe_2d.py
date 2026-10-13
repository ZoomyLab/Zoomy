"""2-D Gaussian pulse — exercises both horizontal dimensions end to end."""
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
def test_swe_2d_pulse(overwrite):
    model = models.swe(dimension=3, bc="extrapolation")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=gaussian_pulse_2d)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)
    assert nsm.update_variables is None

    # REAL API: create_2d(domain=(x0, x1, y0, y1), nx, ny) — the proposal's
    # ``domain=((-1,1),(-1,1)), n_inner_cells=(32,32)`` does not exist
    # (zoomy_core/mesh/lsq_mesh.py:300).
    mesh = LSQMesh.create_2d(domain=(-1.0, 1.0, -1.0, 1.0), nx=32, ny=32)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=0.1)        # 2-D law
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and Q[1].min() > 0.0
    refs.check("swe_2d", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("swe_2d", elapsed, overwrite)
