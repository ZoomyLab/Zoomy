"""AHS26 multilayer (ML-SME) well-balancing march."""
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


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_ahs26_multilayer(overwrite):
    model = models.mlsme(n_layers=2, level=1, bc="periodic")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ahs26_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=AHS26_DOMAIN, n_inner_cells=200)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=AHS26_T_END)
    elapsed = time.perf_counter() - t0

    err = ahs26_l1_vs_reference(Q, mesh)
    print(f"AHS26 L1 departure from equilibrium: {err:.4e}")
    assert np.isfinite(Q).all()
    refs.check("ahs26", overwrite, Q=Q, Qaux=Qaux, err=np.array([err]))
    refs.check_time("ahs26", elapsed, overwrite)
