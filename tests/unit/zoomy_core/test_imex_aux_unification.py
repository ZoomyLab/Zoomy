"""DerivativeAwareSolverMixin.update_qaux must COMPOSE with the canonical
Solver.update_qaux, not shadow it.

Regression guard for the IMEX aux-unification fix: before the fix the mixin
early-returned ``Qaux`` unchanged whenever the model carried no
``derivative_specs`` (every canonical ``aux_registry`` model — SME / VAM /
free-surface), silently freezing the auxiliary vector at its initial condition
for the whole IMEX / FSFIMEX / ColumnIntegrating run.  The fix makes the mixin
call ``super().update_qaux`` first (the local ``update_aux_variables`` leg +
the ``aux_registry`` spatial-derivative leg) and only then overlay the
``derivative_specs`` buffers (the Green-Naghdi ``(Q-Qold)/dt`` capability).

The real IMEX solvers' MRO is ``(…, DerivativeAwareSolverMixin,
HyperbolicSolver, Solver, …)`` so ``super()`` inside the mixin resolves to the
canonical ``Solver.update_qaux``.  These tests mirror that MRO with a tiny
canonical stub so they assert the composition WITHOUT a full mesh/runtime
build, and stay green regardless of which concrete model is used.
"""
import numpy as np
import pytest

from zoomy_core.fvm.solver_numpy import Solver
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin


class _CanonicalStub(Solver):
    """Stands in for the canonical Solver.update_qaux: marks that it ran and
    returns a recognisably transformed Qaux (so the test can prove the mixin
    delegated rather than early-returning the input)."""

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model,
                    parameters, time, dt):
        self._canonical_ran = True
        return np.asarray(Qaux, dtype=float) + 10.0   # canonical leg sentinel


class _MixedSolver(DerivativeAwareSolverMixin, _CanonicalStub):
    """Same MRO shape as IMEXSolver(DerivativeAwareSolverMixin, Hyperbolic…)."""


def _call(solver, model, Qaux, Q=None):
    Q = np.zeros((2, Qaux.shape[1])) if Q is None else Q
    return solver.update_qaux(
        Q, np.asarray(Qaux, dtype=float), Q, Qaux,
        mesh=None, model=model, parameters=None, time=0.0, dt=0.1)


def test_mixin_delegates_to_canonical_when_no_derivative_specs():
    """Canonical aux_registry models (no derivative_specs) must get the
    canonical legs — NOT an unchanged early-return."""
    solver = _MixedSolver()
    solver._canonical_ran = False

    class _CanonicalModel:        # no `derivative_specs` attribute at all
        pass

    Qaux = np.ones((1, 3))
    out = _call(solver, _CanonicalModel(), Qaux)

    assert solver._canonical_ran is True            # super() WAS called
    assert np.allclose(out, 11.0)                   # canonical transform applied
    assert not np.allclose(out, Qaux)               # NOT the old no-op return


def test_mixin_delegates_when_derivative_specs_empty():
    """An empty (falsy) derivative_specs still routes through the canonical
    legs rather than short-circuiting."""
    solver = _MixedSolver()
    solver._canonical_ran = False

    class _EmptySpecsModel:
        derivative_specs = []

    Qaux = np.ones((1, 3))
    out = _call(solver, _EmptySpecsModel(), Qaux)

    assert solver._canonical_ran is True
    assert np.allclose(out, 11.0)


def test_mixin_overlays_derivative_specs_on_top_of_canonical():
    """When derivative_specs ARE present, the canonical legs run first and the
    declared buffer rows are overlaid on top (Green-Naghdi path preserved)."""
    solver = _MixedSolver()
    solver._canonical_ran = False

    class _Spec:
        def __init__(self, key, field, axes):
            self.key, self.field, self.axes = key, field, axes

    class _SpecModel:
        # state fields h, q; aux row 0 holds d(h) (axes=() => identity copy)
        variables = {"h": 0, "q": 1}
        derivative_specs = [_Spec("dh", "h", ())]
        derivative_key_to_index = {"dh": 0}

    Q = np.array([[2.0, 3.0, 4.0],     # h row
                  [0.0, 0.0, 0.0]])    # q row
    Qaux = np.zeros((1, 3))
    out = _call(solver, _SpecModel(), Qaux, Q=Q)

    assert solver._canonical_ran is True
    # axes=() => _compute_derivative returns q_now (the h row) verbatim,
    # overwriting the canonical sentinel on the spec row.
    assert np.allclose(out[0], Q[0])


def test_real_imex_solver_mro_uses_canonical_super():
    """The concrete numpy IMEX solvers must keep the mixin as the entry point
    with Solver.update_qaux reachable via super() (i.e. compose, not shadow)."""
    from zoomy_core.fvm.solver_imex_numpy import IMEXSolver, FSFIMEXSolver
    from zoomy_core.fvm.solver_column import ColumnIntegratingSolver
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver

    for cls in (IMEXSolver, FSFIMEXSolver, ColumnIntegratingSolver):
        mro = cls.__mro__
        # mixin entry point, canonical Solver reachable AFTER it in the MRO
        assert DerivativeAwareSolverMixin in mro
        assert HyperbolicSolver in mro
        assert mro.index(DerivativeAwareSolverMixin) < mro.index(HyperbolicSolver)
        assert cls.update_qaux is DerivativeAwareSolverMixin.update_qaux


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
