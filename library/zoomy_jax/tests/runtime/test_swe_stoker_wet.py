"""Stoker wet dam break — the small gate for the derived SWE march, plus its
order-2 twin of the ``swashes_stoker_o2`` regression."""
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
def test_stoker_wet(overwrite):
    model = models.swe(dimension=2, bc="swashes")            # Model
    sm = SystemModel.from_model(model)                       # SystemModel
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(             # NumericalSystemModel
        sm, reconstruction=ReconstructionSpec(order=1, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)
    assert nsm.update_variables is None                      # cap-free (cid=54)

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=100)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=1.0)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and Q[1].min() > 0.0
    assert np.abs(Q[2]).max() > 0.0, "momentum is zero — the cap bug is back"
    refs.check("stoker_wet", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("stoker_wet", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.jax
def test_stoker_wet_o2_small(overwrite):
    """Small twin of swashes_stoker_o2: same model, same reconstruction,
    20 cells, 2 steps.  Pins the pipeline in seconds.

    It does NOT measure the convergence order — a rate needs a resolution
    sweep, which is what the regression twin is for.  What it catches is any
    change in the machinery that produces those rates.
    """
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, n_steps=2)     # 2 steps, not t_end
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    refs.check("stoker_wet_o2_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("stoker_wet_o2_small", elapsed, overwrite)
