"""The JAX symbolic march lands on its end time like NumPy's (the same template, whose step count
zoomy_core's test_end_time_landing pins): ten steps of 0.005 sum to 0.05 - 6.9e-18, and the
leftover is merged into the last step (`landed`: a `lax.cond` on the step under jit)."""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.small, pytest.mark.jax]


def _run(solver_cls, t_end, dt):
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BC.BoundaryConditions([BC.FromModel(tag=t, definition="wall")
                                                   for t in ("left", "right")])))
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 0.5 + 0.05 * np.exp(-(x[0] - 5.0) ** 2), 0.0]))
    s = solver_cls(time_end=t_end, dt=dt, mood=False, cfl_check_every=0)
    Q0, A = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20),
                    NumericalSystemModel.from_system_model(sm))
    t, Q, _ = s.run(Q0, A)
    return float(t), np.asarray(Q)


def test_lands_like_numpy():
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    tj, Qj = _run(SymbolicHyperbolicSolverJax, 0.05, 0.005)
    tn, Qn = _run(SymbolicHyperbolicSolverNumpy, 0.05, 0.005)
    assert tj == tn == 0.05
    assert np.abs(Qj - Qn).max() <= 1e-14
