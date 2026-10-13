"""Well-balancing over a LONG march — the drift history, not just an end state."""
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
from conftest import CFL


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_wb_drift_history(overwrite):
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    model = models.swe(dimension=2, bc="wall")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lake_at_rest_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod),
        riemann=PositiveNonconservativeRusanov)
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=200)
    t0 = time.perf_counter()
    Q, Qaux, t, drift, umax = march_with_history(nsm, mesh, t_end=50.0,
                                                 cfl=CFL)
    elapsed = time.perf_counter() - t0

    assert drift.max() < 1e-10, f"surface drift {drift.max():.2e} at t=50 s"
    refs.check("wb_long", overwrite, Q=Q, Qaux=Qaux, t=t, drift=drift,
               umax=umax)
    refs.check_time("wb_long", elapsed, overwrite)
