"""Gate 1a-2 — declared ROLES + ADMISSIBILITY (general PDE core).

* every free-surface model class declares depth / bed / stationary /
  positive / momentum / gravity, and generic code reads them;
* role-gated NSM default operations and Riemann kernels give the SAME NSM as
  the old name gate for the free-surface families (in-process emulation of the
  legacy ``any(str(s) == "h")`` gate, compared by full golden render);
* the name-collision regressions are fixed: a tracer named ``h`` gets no KP
  reciprocal / dry CFL gate, a tracer named ``b`` keeps its Rusanov
  dissipation;
* the admissibility list is emitted ([depth] / declared / empty);
* split (Chorin) stages inherit the roles; diffusion is classified from
  ``diffusion_matrix`` alone (no parameter named ``nu``).
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.models import (
    SWE, SME, VAM, MLSME, MLVAM, MLSWE, KESME, ElderSME, MalpassetSWE)
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.diffusion import classify_diffusion
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

pytestmark = [pytest.mark.template_gate]


def _cl():
    return [Newtonian(), StressFree()]


# name -> (factory, expected momentum per direction)
FREE_SURFACE = {
    "SWE1": (lambda: SWE(dimension=1), [["hu"]]),
    "SWE2": (lambda: SWE(dimension=2), [["hu"], ["hv"]]),
    "MalpassetSWE": (lambda: MalpassetSWE(), [["hu"], ["hv"]]),
    "SME0": (lambda: SME(level=0, dimension=2, closures=_cl()), [["q_0"]]),
    "SME1_2d": (lambda: SME(level=1, dimension=3, closures=_cl()),
                [["q_x_0", "q_x_1"], ["q_y_0", "q_y_1"]]),
    "ElderSME": (lambda: ElderSME(level=0), [["q_0"]]),
    "KESME": (lambda: KESME(level=0), [["q_0"]]),
    "VAM1": (lambda: VAM(level=1, dimension=2, closures=_cl()), [["q_0", "q_1"]]),
    "MLSWE": (lambda: MLSWE(n_layers=2, dimension=2,
                            closures=[Newtonian(), NavierSlip(), StressFree()]),
              [["q_1_0"]]),
    "MLSME": (lambda: MLSME(level=1, n_layers=2, dimension=2, closures=_cl()),
              [["q_1_0", "q_1_1", "q_2_0", "q_2_1"]]),
    "MLVAM0": (lambda: MLVAM(level=0, n_layers=2, dimension=2, closures=_cl()),
               [["q_1_0", "q_2_0"]]),
}

_SM_CACHE = {}


def _sm(name):
    if name not in _SM_CACHE:
        _SM_CACHE[name] = SystemModel.from_model(FREE_SURFACE[name][0]())
    return _SM_CACHE[name]


@pytest.mark.parametrize("name", list(FREE_SURFACE))
def test_free_surface_roles_declared(name):
    sm = _sm(name)
    r = sm.roles
    names = [str(s) for s in sm.state]
    assert r.declared and r.depth_declared and r.is_free_surface
    assert names[r.depth] == "h" and names[r.bed] == "b"
    assert [names[i] for i in r.stationary] == ["b"]
    assert "h" in [names[i] for i in r.positive]
    assert r.gravity == "g"
    assert [[names[i] for i in d] for d in r.momentum] == FREE_SURFACE[name][1]
    assert [str(e) for e in sm.admissible_constraints()] == ["h"]
    # the NSM (assumption-free twin) carries the same roles
    nsm = NumericalSystemModel.from_system_model(sm)
    assert nsm.roles.as_names() == r.as_names()
    assert [str(e) for e in nsm.admissible_constraints()] == ["h"]


# ── NSM equality: role gate vs the legacy NAME gate ─────────────────────

def _legacy_default_operations(self):
    """The pre-roles gate, verbatim: depth ops iff a state is NAMED ``h``."""
    from zoomy_core.systemmodel.operations import (
        desingularize_hinv, gate_eigenvalues_dry, guard_eigenvalue_powers,
        normalize_face_normal, regularize_depth_aux, regularize_depth_direct)
    ops = []
    if self.normalize_normal:
        ops.append(normalize_face_normal())
    if self._is_transport_system() and any(str(s) == "h" for s in self.state):
        ops.extend(self._depth_regularization_operations(
            desingularize_hinv, regularize_depth_aux, regularize_depth_direct))
        ops.extend(self._eigenvalue_guard_operations(
            guard_eigenvalue_powers, gate_eigenvalues_dry))
    return ops


NSM_EQUAL = {
    "SWE1": ("SWE1", {}),
    "SWE2": ("SWE2", {}),
    "SME0": ("SME0", {}),
    "SME1": (None, {}),
    "VAM1": ("VAM1", {}),
    "MLVAM0": ("MLVAM0", {}),
}


@pytest.mark.parametrize("case", list(NSM_EQUAL))
def test_nsm_equal_under_role_and_name_gate(case, monkeypatch):
    import goldenlib
    key, kw = NSM_EQUAL[case]
    if key is None:
        sm = SystemModel.from_model(SME(level=1, dimension=2, closures=_cl()))
    else:
        sm = _sm(key)
    new = NumericalSystemModel.from_system_model(sm, **kw)
    with monkeypatch.context() as mp:
        mp.setattr(NumericalSystemModel, "default_operations",
                   _legacy_default_operations)
        old = NumericalSystemModel.from_system_model(sm, **kw)
        old_txt = goldenlib.render_nsm(old, title=case, with_numerics=False)
        old_ops = [o.name for o in _legacy_default_operations(old)]
    assert [o.name for o in new.default_operations()] == old_ops
    assert goldenlib.render_nsm(new, title=case, with_numerics=False) == old_txt
    # Riemann kernels: the role-declared stationary rows are exactly the
    # legacy "row named b"; the depth / bed handles are the legacy h / b
    names = [str(s) for s in new.state]
    num = new.build_numerics()
    assert num._stationary_rows == (names.index("b"),)
    assert num._field_handles["h"].index == names.index("h")
    assert num._field_handles["b"].index == names.index("b")
    assert new.depth_scaled_state_indices == [
        i for i, s in enumerate(names) if s not in ("h", "b")]


# ── name-collision regressions ──────────────────────────────────────────

def _advection_sm(name, assumptions=None, with_eigenvalues=True):
    """Linear advection of ONE scalar field called ``name`` (no roles)."""
    t, x = sp.Symbol("t", real=True), sp.Symbol("x", real=True)
    q = sp.Symbol(name, **(assumptions or {"real": True}))
    a = sp.Symbol("a", real=True)
    n0 = sp.Symbol("n0", real=True)
    p = Zstruct(a=a)
    p._symbolic_name = "p"
    sm = SystemModel(
        time=t, space=[x], state=[q], aux_state=[], parameters=p,
        flux=sp.Matrix([[a * q]]), hydrostatic_pressure=sp.zeros(1, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, 1),
        source=sp.zeros(1, 1), mass_matrix=sp.eye(1),
        eigenvalues=sp.Matrix([[a * n0]]) if with_eigenvalues else None,
        parameter_values=Zstruct(a=1.0))
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]))
    return sm


def test_tracer_named_h_gets_no_free_surface_ops(monkeypatch):
    sm = _advection_sm("h", {"nonnegative": True})
    nsm = NumericalSystemModel.from_system_model(sm)
    assert nsm.roles.depth is None and not nsm.roles.depth_declared
    assert [o.name for o in nsm.default_operations()] == ["normalize_face_normal"]
    assert not any(str(s) == "hinv" for s in nsm.aux_state)
    assert nsm.eigenvalues_cfl is None
    rt = NumpyRuntimeModel.from_system_model(nsm)
    q = np.array([[1e-3, 0.5]])     # a tiny tracer value is NOT a dry cell
    lam = np.asarray(rt.eigenvalues(q, np.zeros((0, 2)), np.array([1.0]),
                                    np.array([[1.0, 1.0]])), float)
    assert np.allclose(np.abs(lam), 1.0)
    # the legacy name gate WOULD have gated it (the defect being fixed)
    with monkeypatch.context() as mp:
        mp.setattr(NumericalSystemModel, "default_operations",
                   _legacy_default_operations)
        old = NumericalSystemModel.from_system_model(sm)
    assert old.eigenvalues_cfl is not None
    # positive role seeded from the analytic sign assumption (before the NSM
    # strips it); the admissibility list stays empty (nothing declared)
    assert nsm.roles.positive == (0,)
    assert rt.admissible(q, np.zeros((0, 2)), np.array([1.0])).shape == (0, 2)


@pytest.mark.parametrize("riemann", ["NonconservativeRusanov", "Rusanov", "HLL"])
def test_tracer_named_b_keeps_dissipation(riemann):
    import zoomy_core.fvm.riemann_solvers as R
    sm = _advection_sm("b")
    nsm = NumericalSystemModel.from_system_model(sm, riemann=getattr(R, riemann))
    num = nsm.build_numerics()
    assert num._stationary_rows == ()
    rt = num.to_runtime_numpy()
    qL, qR = np.array([0.0]), np.array([1.0])
    e = np.zeros(0)
    p, n = np.array([1.0]), np.array([1.0])
    F = np.asarray(rt.numerical_flux(qL, qR, e, e, p, n), float).ravel()
    D = np.asarray(rt.numerical_fluctuations(qL, qR, e, e, p, n), float).ravel()
    Dp, Dm = D[0], D[1]
    # a = 1, q_L = 0: the dissipative (upwind) face moves NOTHING out of the
    # left cell and nothing into the right one.  Without dissipation on the
    # row (the legacy "row named b" exclusion) both cells change by 0.5.
    assert abs(F[0] + Dm) < 1e-14 and abs(-F[0] + Dp) < 1e-14


def test_free_surface_bed_stays_dissipation_free():
    nsm = NumericalSystemModel.from_system_model(_sm("SWE1"))
    num = nsm.build_numerics()
    Id = num.get_viscosity_identity_fluctuations()
    assert Id[0, 0] == 0 and Id[1, 1] == 1


def test_state_named_T_flattens_positionally():
    """A state called ``T`` must index ``Q[i]``, not pick the transpose
    attribute of an array (the jax BC-kernel crash)."""
    sm = _advection_sm("T")
    rt = NumpyRuntimeModel.from_system_model(sm)

    class ArrayLike:                  # a non-numpy array (as jax) with .T
        def __init__(self, a):
            self.a = np.asarray(a)
            self.shape = self.a.shape

        def __getitem__(self, i):
            return self.a[i]

        @property
        def T(self):
            return self.a.T
    q = ArrayLike([0.7])
    out = rt.boundary_conditions(0, 0.0, np.zeros(3), 0.1, q, np.zeros(0),
                                 np.array([1.0]), np.array([1.0, 0, 0]))
    assert np.allclose(np.asarray(out, float).ravel(), [0.7])


# ── admissibility ───────────────────────────────────────────────────────

def _euler_sm():
    t, x = sp.Symbol("t", real=True), sp.Symbol("x", real=True)
    rho, m, E = sp.symbols("rho m E", real=True)
    gam = sp.Symbol("gamma", positive=True)
    p = Zstruct(gamma=gam)
    p._symbolic_name = "p"
    pr = (gam - 1) * (E - m ** 2 / (2 * rho))
    sm = SystemModel(
        time=t, space=[x], state=[rho, m, E], aux_state=[], parameters=p,
        flux=sp.Matrix([[m], [m ** 2 / rho + pr], [(E + pr) * m / rho]]),
        hydrostatic_pressure=sp.zeros(3, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(3, 3, 1),
        source=sp.zeros(3, 1), mass_matrix=sp.eye(3),
        parameter_values=Zstruct(gamma=1.4))
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]))
    sm.declare_admissible([rho, pr])
    return sm


def test_admissibility_emitted_and_lowered():
    # free surface: [depth]
    nsm = NumericalSystemModel.from_system_model(_sm("SWE2"))
    rt = NumpyRuntimeModel.from_system_model(nsm)
    Q = np.array([[0.0, 0.1], [1.0, -0.2], [0.3, 0.0], [0.0, 0.0]])
    Qa = np.ones((len(nsm.aux_state), 2))
    p = np.array(list(nsm.parameter_values.values()))
    assert np.array_equal(rt.admissible(Q, Qa, p), Q[1:2])
    # Euler: the model DECLARES [rho, p(Q)]
    enm = NumericalSystemModel.from_system_model(_euler_sm())
    assert enm.roles.depth is None
    ert = NumpyRuntimeModel.from_system_model(enm)
    Q = np.array([[1.0, 0.5], [0.2, 0.1], [2.5, 0.001]])
    got = ert.admissible(Q, np.zeros((0, 2)), np.array([1.4]))
    pr = 0.4 * (Q[2] - Q[1] ** 2 / (2 * Q[0]))
    assert np.allclose(got, np.stack([Q[0], pr]))
    assert got[1, 1] < 0          # inadmissible state detected


def test_split_stages_inherit_roles():
    split = VAM(level=1, dimension=2, closures=_cl()).chorin_split(0.01)
    for stage in (split.SM_pred, split.SM_press, split.SM_corr):
        names = [str(s) for s in stage.state]
        r = stage.roles
        assert names[r.depth] == "h" and names[r.bed] == "b"
        assert r.gravity == "g"
    # only the transport stage gets the free-surface default ops
    assert any(str(s) == "hinv" for s in split.SM_pred.aux_state)


def test_diffusion_classified_from_the_tensor_not_a_nu_parameter():
    t, x, y = (sp.Symbol(s, real=True) for s in "txy")
    u, c = sp.symbols("u c", real=True)
    kappa = sp.Symbol("kappa", positive=True)
    p = Zstruct(kappa=kappa)
    p._symbolic_name = "p"

    def sm_with(A):
        return SystemModel(
            time=t, space=[x, y], state=[u, c], aux_state=[], parameters=p,
            flux=sp.zeros(2, 2), hydrostatic_pressure=sp.zeros(2, 2),
            nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 2),
            source=sp.zeros(2, 1), mass_matrix=sp.eye(2),
            diffusion_matrix=A, parameter_values=Zstruct(kappa=0.3))
    A = sp.MutableDenseNDimArray.zeros(2, 2, 2, 2)
    for i in range(2):
        for d in range(2):
            A[i, i, d, d] = kappa
    s = classify_diffusion(sm_with(A))
    assert s.kind == "isotropic" and not s.state_dependent
    assert s.coefficients == {0: kappa, 1: kappa}
    A2 = sp.MutableDenseNDimArray(A.tolist())
    A2[0, 1, 0, 0] = kappa * u
    s2 = classify_diffusion(sm_with(A2))
    assert s2.kind == "dense" and s2.state_dependent
    assert classify_diffusion(
        sm_with(sp.MutableDenseNDimArray.zeros(2, 2, 2, 2))).kind == "none"
