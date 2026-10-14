"""The march lands on its end time: a floating-point leftover below `LANDING_TOL` of the step is
merged into the last step (NumPy backend; the template is the same on every backend).

Ten steps of dt = 0.005 sum to 0.05 - 6.9e-18: before, an eleventh step of 6.9e-18 followed,
and a Chorin projection (1 / dt^2 in its pressure operator) failed there.
"""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver.discretization.time import LANDING_TOL
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.numpy_backend import runtime as R
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _steps(t_end, dt):
    """`(steps, final time)` of a 1-D SME(0) march at the constant `dt`."""
    sm = SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BC.BoundaryConditions([BC.FromModel(tag=t, definition="wall")
                                                   for t in ("left", "right")])))
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 0.5 + 0.05 * np.exp(-(x[0] - 5.0) ** 2), 0.0]))
    s = SymbolicHyperbolicSolverNumpy(time_end=t_end, dt=dt, mood=False, cfl_check_every=0)
    Q0, A = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20),
                    NumericalSystemModel.from_system_model(sm))
    count, loop = [0], s.module["while_loop"]

    def counted(c, b, init):
        def body(*a):
            count[0] += 1
            return b(*a)
        return loop(c, body, init)
    s._fn = R.build(s.source, "hyperbolic", {**s.module, "while_loop": counted})
    t, _, _ = s.run(Q0, A)
    return count[0], t


@pytest.mark.parametrize("t_end, dt, steps", [
    (0.05, 0.005, 10),                              # a 6.9e-18 sliver: merged
    (0.05 + 0.5 * LANDING_TOL * 0.005, 0.005, 10),  # leftover half the tolerance: merged
    (0.05 + 2 * LANDING_TOL * 0.005, 0.005, 11),    # twice the tolerance: a step of its own
    (0.05, 0.004, 13)])                             # a genuine half step
def test_landing(t_end, dt, steps):
    n, t = _steps(t_end, dt)
    assert n == steps
    assert t == t_end


def test_chorin_constant_dt_lands_on_end_time():
    """VAM(1) Chorin, 1-D, dt 0.005 to t 0.05: completes at t_end (before: the sliver step's
    pressure solve failed, 'relative residual 9.4e-03 > accept gate')."""
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.symsolver.numpy_backend import SymbolicChorinSolverNumpy
    from zoomy_core.symsolver.settings import LinearSolver
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    bcs = ([BC.Dirichlet("left", on=f, value=0.0) for f in ("q_0", "q_1", "r_0", "r_1")]
           + [BC.Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)])
    model = VAM(level=1, dimension=2, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    s = SymbolicChorinSolverNumpy(time_end=0.05, dt=0.005, mood=False, cfl_check_every=0,
                                  pressure_linear_solver=LinearSolver.gmres)
    Q, A = s.setup(LSQMesh.create_1d(domain=(0, 3), n_inner_cells=30),
                   predictor_nsm(r.SM_pred, r.SM_press, r.SM_corr), r.SM_press, r.SM_corr)
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q[names.index("h")] = 0.5 + 0.02 * np.exp(-(x - 1.5) ** 2 / 0.1)
    t, Q, _ = s.run(Q, s.initial_aux(Q, A))
    assert t == 0.05
    assert np.isfinite(Q).all()
