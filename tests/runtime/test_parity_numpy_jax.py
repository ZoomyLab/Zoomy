"""The two solvers are each other's reference — full state, both fields."""
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
def test_numpy_jax_parity(overwrite):
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NumpySolver

    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter=Limiter.minmod))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)
    t0 = time.perf_counter()
    Qj, Aj = march(nsm, mesh, cfl=CFL, t_end=0.5)
    Qn, An = NumpySolver(time_end=0.5,
                         compute_dt=timestepping.adaptive(
                             CFL=CFL, dimension=int(mesh.dimension))
                         ).solve(mesh, nsm, write_output=False)
    elapsed = time.perf_counter() - t0
    n = mesh.n_inner_cells
    Qn, An = np.asarray(Qn)[:, :n], np.asarray(An)[:, :n]

    assert np.allclose(Qj, Qn, atol=1e-10), \
        f"jax vs numpy state: max|diff| {np.abs(Qj - Qn).max():.3e}"
    assert np.allclose(Aj, An, atol=1e-10), \
        f"jax vs numpy aux: max|diff| {np.abs(Aj - An).max():.3e}"
    refs.check("parity", overwrite, Q=Qj, Qaux=Aj)
    refs.check_time("parity", elapsed, overwrite)
