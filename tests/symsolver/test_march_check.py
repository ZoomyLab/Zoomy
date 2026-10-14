"""The march raises instead of ending silently: a step that does not advance the time (dt NaN,
<= 0, or below the resolution of time) stops the loop at the last state and `check.march`
raises `FloatingPointError`, as does a non-finite final state (ANALYSIS 49, item 1).
"""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, PositiveNonconservativeHLL
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.numpy_backend import runtime as R
from zoomy_core.symsolver.strategies.hyperbolic import _time_loop
from zoomy_core.symsolver.vocabulary import march_error
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small]


def _dambreak(n=20):
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 1.0 + (x[0] < 0.5), 0.0]))
    return (NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeHLL),
            LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n))


@pytest.mark.parametrize("dt", [None, 0.01])
def test_non_finite_state_raises(dt):
    """A NaN cell: the adaptive march stops at t = 0 (its CFL step is not positive), the
    constant-dt march reaches t_end and raises on the non-finite state; a finite overflow
    (q = 1e300) raises too."""
    nsm, mesh = _dambreak()
    s = SymbolicHyperbolicSolverNumpy(time_end=0.05, cfl=0.4, dt=dt)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    assert t == pytest.approx(0.05) and np.isfinite(Q).all()
    Q0 = np.array(Q0)
    Q0[1, 7] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite state" if dt else
                       r"stopped at t = 0 .*initial state"):
        s.run(Q0, A)
    Q0[1, 7], Q0[2, 7] = 1.0, 1e300
    with pytest.raises(FloatingPointError, match="zoomy march"):
        s.run(Q0, A)


def test_stalled_step_stops_the_loop():
    """A step rule whose dt drops to 0 after t = 0.25 (it would loop forever): the march stops
    at the last state, step 4, and says so; dt < 0 likewise."""
    nsm, mesh = _dambreak(4)
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0, cfl=0.4)
    s.setup(mesh, nsm)
    for bad, text in ((sp.S.Zero, "dt = 0,"), (sp.Float(-1), "dt = -1,")):
        def march(Q, Qaux, p, time, time_end, mesh):
            def step(Q, Qaux, time, cap):
                return Q + 1, Qaux, sp.Piecewise((sp.Float(0.1), time < 0.25), (bad, True))
            return _time_loop(Q, Qaux, time, time_end, None, step)
        fn = R.build(PyPrinter().procedure("march", sp.symbols("Q Qaux p time time_end mesh"),
                                           march), "march", s.module)
        with pytest.raises(FloatingPointError,
                           match=rf"stopped at t = 0\.3 .*step 4 has {text}.*finite"):
            fn(np.zeros((3, 4)), np.zeros((1, 4)), s.parameters, 0.0, 1.0, s.mesh, 0)


def test_march_error_text():
    """The message names the time, the step, dt and whether the state was finite."""
    assert march_error(1.0, 1.0, 0.1, 10, 3.0) is None
    assert "step 3 has dt = nan" in march_error(0.5, 1.0, np.nan, 3, 2.0)
    assert "step 2 produced it" in march_error(0.5, 1.0, 0.0, 3, np.nan)
    assert "reached t = 1" in march_error(1.0, 1.0, 0.1, 3, np.inf)
