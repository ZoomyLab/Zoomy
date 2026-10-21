"""Small twin of the AHS26 multilayer regression: ML-SME, order 2, tiny mesh,
2 steps."""
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
def test_ahs26_small(overwrite):
    model = models.mlsme(n_layers=2, level=1, bc="periodic")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ahs26_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=AHS26_DOMAIN, n_inner_cells=20)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, n_steps=2)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    refs.check("ahs26_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("ahs26_small", elapsed, overwrite)
