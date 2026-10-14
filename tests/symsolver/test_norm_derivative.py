"""A friction law's |u| = (u.u)**(1/2) has the derivative 0 at u = 0, not 0/0 = NaN: the NSM
operation `norm_derivative_at_zero` (default, before the source linearization) keeps the value
and makes the point-implicit Jacobian and JAX's AD Newton finite at rest (ANALYSIS 49, item 4).
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, SourceTreatment
from zoomy_core.systemmodel.operations import _nonnegative, norm_derivative_at_zero
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.nsm, pytest.mark.small]


def _sm(level):
    sm = SystemModel.from_model(SME(level=level, dimension=3,
                                    closures=[Newtonian(), ManningFriction(), StressFree()]))
    sm.parameter_values.n = 0.5
    return sm


def _fractional(e):
    return [p for p in sp.sympify(e).atoms(sp.Pow) if not p.exp.is_integer]


def test_nonnegative_forms():
    x, y, h = sp.symbols("x y h")
    assert _nonnegative(x**2 / h**2 - 2 * x * y / h**2 + y**2 / h**2)
    assert _nonnegative(h**2 * x**2 + h**2 * y**2) and _nonnegative(3 * x**2)
    assert not _nonnegative(x**2 - y**2) and not _nonnegative(h * x**2)
    assert not _nonnegative(x**2 - 3 * x * y + y**2)


@pytest.mark.parametrize("level", [0, 1])
def test_source_value_kept_and_jacobian_finite_at_rest(level, monkeypatch):
    """SME(0) and SME(1) 2-D with Manning: no fractional power of a squared norm is left, the
    source equals that of the NSM without the operation on random states and at rest, and its
    symbolic Jacobian is finite at rest (without the operation: NaN)."""
    import zoomy_core.systemmodel.operations as O
    nsm = NumericalSystemModel.from_system_model(_sm(level))
    with monkeypatch.context() as m:
        m.setattr(O, "norm_derivative_at_zero", lambda: (lambda sm: None))
        ref = NumericalSystemModel.from_system_model(_sm(level))
    assert any(_nonnegative(p.base) for e in sp.flatten(ref.source) for p in _fractional(e))
    assert not any(_nonnegative(p.base) for e in sp.flatten(nsm.source)
                   for p in _fractional(e) if not isinstance(p.base, sp.Function))
    syms = [*nsm.state, *nsm.aux_state, *nsm.parameters.values()]
    assert [str(s) for s in syms] == [str(s) for s in
                                      (*ref.state, *ref.aux_state, *ref.parameters.values())]
    names = [str(s) for s in syms]
    mods = [{"conditional": lambda c, t, f: np.where(c, t, f)}, "numpy"]

    def f(model, expr):
        args = [*model.state, *model.aux_state, *model.parameters.values()]
        return sp.lambdify(args, list(sp.flatten(expr)), mods)
    f_new, f_ref = f(nsm, nsm.source), f(ref, ref.source)
    rng = np.random.default_rng(4)
    at_rest = [0.0 if n.startswith("q_") else 1.0 for n in names]
    with np.errstate(all="ignore"):
        for v in [*rng.uniform(0.1, 2.0, (20, len(syms))), at_rest]:
            assert np.array_equal(np.asarray(f_new(*v), float), np.asarray(f_ref(*v), float))
        J = f(nsm, nsm._compute_source_jacobian_wrt_variables())(*at_rest)
        J_ref = f(ref, ref._compute_source_jacobian_wrt_variables())(*at_rest)
    assert np.isfinite(np.asarray(J, float)).all()
    assert not np.isfinite(np.asarray(J_ref, float)).all()


def test_operation_is_idempotent_and_leaves_other_slots():
    sm = _sm(0)
    nsm = NumericalSystemModel.from_system_model(sm)
    src = list(sp.flatten(nsm.source))
    ev = nsm.eigenvalues
    norm_derivative_at_zero()(nsm)
    assert list(sp.flatten(nsm.source)) == src and nsm.eigenvalues == ev


def test_point_implicit_manning_at_rest_numpy():
    """SWE 2-D (SME(0)) with Manning n = 0.5, walls, the point-implicit (linearized) source,
    h = 1 with a flow patch q_x = 2 at x < 3 next to cells exactly at rest: finite (before: NaN
    after the first step in every cell at rest)."""
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    sm = SystemModel.from_model(SME(
        level=0, dimension=3, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right", "bottom", "top")])))
    sm.parameter_values.n = 0.5
    nsm = NumericalSystemModel.from_system_model(sm, source_treatment=SourceTreatment.linearized)
    s = SymbolicHyperbolicSolverNumpy(time_end=0.05, cfl=0.4)
    Q, A = s.setup(LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=8, ny=8), nsm)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = np.array(Q)
    Q[1], Q[2] = 1.0, np.where(xc < 3, 2.0, 0.0)
    with np.errstate(all="ignore"):
        t, Q, _ = s.run(Q, s.initial_aux(Q, A))
    assert t == pytest.approx(0.05) and np.isfinite(Q).all()
    assert (Q[2] == 0).any()                    # cells still exactly at rest
