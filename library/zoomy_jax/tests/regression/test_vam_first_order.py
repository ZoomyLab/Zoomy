"""VAM FIRST ORDER — the regression pair.

Second order is not pursued for now (user, 2026-07-22): the order-2 ladder is a
separate open question, and `h` converging at ~1.07 there is not this pair's
business.  These two lock in that the first-order scheme marches and keeps
producing the SAME numbers.

Both compare the FULL Q and Qaux against stored arrays, and both pass the order
knobs EXPLICITLY — ReconstructionSpec defaults to order=1 and
ChorinSplitVAMSolverJax.time_order to 1, so a call that omits them runs
first-order by accident rather than by intent.  That accident is what made
test_vam_second_order assert 2nd order against a 1st-order run for months.
"""
import time

import numpy as np
import pytest
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL

FIRST_ORDER = dict(reconstruction=ReconstructionSpec(order=1), time_order=1)


def _march(n_cells, n_steps):
    """Derived VAM(1,2), first order, `n_steps` steps on `n_cells`."""
    model = models.vam(level=1, dimension=2, bc="periodic")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    set_state_width(triple[0])
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n_cells)
    return chorin_march(triple, mesh, cfl=CFL, ic=smooth_vam_ic,
                        n_steps=n_steps, pressure_tol=VAM_PRESSURE_TOL,
                        **FIRST_ORDER)


@pytest.mark.small
@pytest.mark.jax
def test_vam_first_order_small(overwrite):
    """20 cells, 2 steps — the fast twin of the regression below."""
    t0 = time.perf_counter()
    Q, Qaux = _march(20, 2)
    elapsed = time.perf_counter() - t0

    assert np.all(np.isfinite(Q)), "first-order VAM went non-finite in 2 steps"
    refs.check("vam_first_order_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("vam_first_order_small", elapsed, overwrite)


@pytest.mark.regression
@pytest.mark.jax
def test_vam_first_order(overwrite):
    """100 cells, 20 steps — same scheme, same direction, resolved."""
    t0 = time.perf_counter()
    Q, Qaux = _march(100, 20)
    elapsed = time.perf_counter() - t0

    assert np.all(np.isfinite(Q)), "first-order VAM went non-finite in 20 steps"
    refs.check("vam_first_order", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("vam_first_order", elapsed, overwrite)
