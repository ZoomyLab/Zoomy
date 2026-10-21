"""VAM Chorin split-solver small gates: the bump runtime gate and the smooth
periodic twin of the Richardson order study."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL


@pytest.mark.small
@pytest.mark.jax
def test_vam_chorin_short(overwrite):
    """Split-solver runtime gate; also the cid=50 dry-stage regression."""
    model = models.vam(level=1, dimension=2, bc="bump")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    print(describe(triple[0]))                     # SM_pred
    set_state_width(triple[0])

    mesh = LSQMesh.create_1d(domain=ESC_DOMAIN, n_inner_cells=ESC_NCELLS)
    t0 = time.perf_counter()
    Q, Qaux = chorin_march(triple, mesh, cfl=CFL, ic=bump_ic, t_end=0.5,
                           h_scale=ESC_H_RES)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and Q[1].min() > 0.0
    refs.check("vam_chorin", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("vam_chorin", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.jax
def test_vam_smooth_small(overwrite):
    """Small twin of vam_order2: same smooth periodic VAM through the chorin
    split, 20 cells, 2 steps.  Pins the machinery the Richardson rate is
    measured on."""
    model = models.vam(level=1, dimension=2, bc="periodic")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    print(describe(triple[0]))
    set_state_width(triple[0])

    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    t0 = time.perf_counter()
    Q, Qaux = chorin_march(triple, mesh, cfl=CFL, ic=smooth_vam_ic,
                           n_steps=2, pressure_tol=VAM_PRESSURE_TOL)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    refs.check("vam_smooth_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("vam_smooth_small", elapsed, overwrite)
