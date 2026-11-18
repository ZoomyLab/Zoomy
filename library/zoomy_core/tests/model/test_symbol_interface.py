"""State fields, parameters, boundary and initial conditions by symbol, next to the index forms."""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel import SystemModel

pytestmark = [pytest.mark.model, pytest.mark.small]
CL = lambda: [Newtonian(), ManningFriction(), StressFree()]     # noqa: E731


def test_state_layout_matches_derivation():
    for kw in (dict(level=0, dimension=2), dict(level=1, dimension=3)):
        Q = SME.state(**kw)
        sm = SystemModel.from_model(SME(closures=CL(), **kw))
        assert list(Q) == list(sm.state)
        assert list(sm.fields) == list(sm.state)


def test_names_are_aliases_and_groups():
    Q = SME.state(level=0, dimension=3, names=("b", "h", "hu", "hv"))
    assert Q.hu is Q.q_x_0 and Q.hv is Q.q_y_0
    assert tuple(Q.momentum) == (Q.q_x_0, Q.q_y_0)
    assert Q.slot(Q.hu) == 2
    with pytest.raises(ValueError):
        SME.state(level=0, dimension=3, names=("b", "h"))
    with pytest.raises(AttributeError, match="fields: b, h, q_x_0, q_y_0"):
        Q.q_0


def _run(bcs, ic, t_end=0.2):
    sm = SystemModel.from_model(SME(level=0, dimension=2, parameters={"n": 0.03}, closures=CL(),
                                    boundary_conditions=bcs))
    sm.initial_conditions = ic
    s = SymbolicHyperbolicSolverNumpy(time_end=t_end, dt=0.01, mood=False)
    Q0, A0 = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=20),
                     NumericalSystemModel.from_system_model(sm))
    return s, Q0, A0, sm


def test_symbol_and_index_forms_agree():
    Q = SME.state(level=0, dimension=2)
    sym = [BC.Dirichlet("left", on=Q.q_0, value=0.5),
           BC.Dirichlet("right", on=Q.h, value=0.6 - Q.b)]
    old = [BC.Dirichlet("left", on="q_0", value=0.5),
           BC.Lambda("right", prescribe_fields={1: lambda t, X, dX, Qs, A, p, n: 0.6 - Qs[0]})]
    ic_sym = IC.UserFunction(lambda x: {Q.h: 0.5, Q.b: 0.01 * x[0]})
    ic_old = IC.UserFunction(lambda x: np.array([0.01 * x[0], 0.5, 0.0]))
    a = _run(sym, ic_sym)
    b = _run(old, ic_old)
    np.testing.assert_allclose(a[1], b[1])
    ra = a[0].run(a[1], a[2])
    rb = b[0].run(b[1], b[2])
    assert ra.time == rb.time
    np.testing.assert_allclose(ra.Q, rb.Q, rtol=1e-12, atol=1e-14)
    time, Qe, Qaux = ra                                 # unpacking still works
    assert Qe is ra.Q


def test_lambda_and_inflowoutflow_symbol_keys():
    Q = SME.state(level=0, dimension=2)
    lam = [BC.Dirichlet("left", on=Q.q_0, value=0.5),
           BC.Lambda("right", prescribe_fields={Q.h: lambda t, X, dX, Qs, A, p, n: 0.6 - Q.b})]
    io = [BC.InflowOutflow("left", prescribe_fields={Q.q_0: 0.5}),
          BC.Dirichlet("right", on=Q.h, value=0.6 - Q.b)]
    ic = IC.Constant({Q.h: 0.5})
    a, b = _run(lam, ic), _run(io, ic)
    np.testing.assert_allclose(a[0].run(a[1], a[2]).Q, b[0].run(b[1], b[2]).Q, rtol=1e-12)


def test_parameter_by_symbol():
    Q = SME.state(level=0, dimension=2)
    s, Q0, A0, sm = _run([BC.Dirichlet("left", on=Q.q_0, value=0.5), BC.Extrapolation("right")],
                         IC.Constant({Q.h: 0.5}))
    P = sm.parameters
    assert s.parameters[P.n] == pytest.approx(0.03)
    r1 = s.run(Q0, A0, parameters={P.n: 0.05}).Q
    v = np.array(s.parameters)
    v[[str(p) for p in P.get_list()].index("n")] = 0.05
    r2 = s.run(Q0, A0, parameters=v).Q                  # a vector still works
    np.testing.assert_allclose(r1, r2)
    with pytest.raises(KeyError, match="no parameter 'nn'"):
        s.run(Q0, A0, parameters={sp.Symbol("nn"): 1.0})


def test_unknown_field_symbol_fails():
    with pytest.raises(ValueError, match="not a state field"):
        _run([BC.Dirichlet("left", on=sp.Symbol("hu"), value=0.5), BC.Extrapolation("right")],
             IC.Constant({}))


def test_wall_groups_and_far_field_by_symbol():
    Q = SME.state(level=0, dimension=2)
    ic = IC.UserFunction(lambda x: {Q.h: 0.5 + 0.1 * np.exp(-(x[0] - 5) ** 2), Q.q_0: 0.1})
    pairs = [
        ([BC.Wall(tag="left", momentum_field_indices=[[Q.q_0]]), BC.Wall(tag="right", momentum_field_indices=[[Q.q_0]])],
         [BC.Wall(tag="left", momentum_field_indices=[[2]]), BC.Wall(tag="right", momentum_field_indices=[[2]])]),
        ([BC.CharacteristicFarField(tag=t, far_field={Q.h: 0.5}) for t in ("left", "right")],
         [BC.CharacteristicFarField(tag=t, far_field=[0.0, 0.5, 0.0]) for t in ("left", "right")]),
    ]
    for sym, old in pairs:
        a, b = _run(BC.BoundaryConditions(sym), ic), _run(BC.BoundaryConditions(old), ic)
        np.testing.assert_allclose(a[0].run(a[1], a[2]).Q, b[0].run(b[1], b[2]).Q, rtol=1e-12, atol=1e-14)
