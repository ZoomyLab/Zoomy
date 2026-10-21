"""Ritter dry dam break — capless dry front.  NO floor is permitted anywhere."""
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
from conftest import CFL, march, wet_dry_o2


@pytest.mark.small
@pytest.mark.jax
def test_ritter_dry(overwrite):
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ritter_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)
    assert nsm.update_variables is None

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=100)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=1.0)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    assert Q[1].min() >= 0.0, "negative depth — and NO floor is permitted"
    refs.check("ritter_dry", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("ritter_dry", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.jax
def test_ritter_dry_o2_small(overwrite):
    """Small twin of swashes_ritter_o2: same model, order 2, 20 cells,
    2 steps.  Pins the pipeline that produces the recorded dry-front rate.

    ORDER-2 WET/DRY NUMERICS ARE EXPLICIT (``wet_dry_o2``).  As first landed
    this test passed NO solver kwargs, so it marched plain conservative
    LSQ-MUSCL with no η reconstruction, no front pre-detector and MOOD OFF —
    ``ReconstructionSpec`` carries only ``order`` / ``limiter`` to the jax
    backend, so ``positivity="mood"`` on the spec would ALSO have been a
    no-op.  Measured consequence at 20 cells: h = -7.200562e+04 on step 2.
    Nothing in the solver was broken — the test simply never asked for the
    positivity mechanism it asserts on.  Documented so it is not "simplified"
    back out.
    """
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ritter_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=20)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL, n_steps=2, **wet_dry_o2(nsm))
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    assert Q[1].min() >= 0.0, "negative depth — and NO floor is permitted"
    refs.check("ritter_dry_o2_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("ritter_dry_o2_small", elapsed, overwrite)
