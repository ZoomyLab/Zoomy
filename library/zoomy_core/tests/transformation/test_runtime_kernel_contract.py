"""E8 — the runtime kernel contract (spec §1b/§2 rank 8; kernel-values
category of the v3 gate).

Merged from: test_userfunctions_contract.py + test_inf_guard_req168.py + the
runtime-aux test of tests/systemmodel/test_operations.py + the t/dt/x binding
tests of test_time_position_operators_req185.py + the VALUE core of
test_riemann_hll_hllc.py + test_solver_honors_nsm_riemann.py.

No golden covers any of this: REQUIRED_KERNELS is a red-test, the opaque
kernels / HLL-HLLC fluxes are numeric VALUES, t/dt/x binding and per-step aux
application are runtime-only (X01 is rest-state and time-independent).
"""
import numpy as np
import pytest
import sympy as sp
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.model.kernel_functions import REQUIRED_KERNELS
from zoomy_core.fvm import userfunctions as uf
from zoomy_core.fvm.riemann_solvers import HLL, HLLC
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, DT_SYMBOL
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small, pytest.mark.gate]

# strong refs so the 1-slot ID caches can never see a recycled id
_KEEP = []


def _keep(arrs):
    _KEEP.append(arrs)
    return arrs


# ── 1. the REQUIRED_KERNELS red-test ────────────────────────────────────────

def test_required_kernels_covered():
    """numpy's UserFunctions table supplies EVERY backend-supplied kernel core
    declares — a new opaque kernel goes RED here instead of surfacing as a
    silent lambdify NameError; ``conditional`` is printer-lowered and must NOT
    be demanded of any backend table."""
    table = uf.numpy_module()
    missing = REQUIRED_KERNELS - set(table)
    assert not missing, f"numpy UserFunctions missing kernels: {sorted(missing)}"
    assert "conditional" not in REQUIRED_KERNELS


# ── 2. opaque kernel numerics (values + inf-tolerance + id-cache) ───────────

def _swe_jacobian(u, c):
    """SWE quasilinear matrix [[0, 1], [c^2-u^2, 2u]] — known spectrum u-+c."""
    return [0.0, 1.0, c * c - u * u, 2.0 * u]


def test_opaque_kernel_numerics():
    """The opaque kernels return the WHOLE packed result (``eigensystem`` →
    ``[λ, R, L]``, ``eigenvalues`` → ``λ``, ``solve`` → ``A⁻¹b``); component
    access is ``pick(K(*a), idx)``.  eigenvalues scalar + batched; eigensystem
    reconstructs A = R |Lambda| L with L R = I; solve == linalg.solve;
    inf-TOLERANT (poisoned members report +inf eigenvalues / identity
    eigenbasis, finite members untouched — REQ-168 addenda); ``pick`` reads
    a component of the packed stack."""
    n = 2
    # scalar values — eigenvalues returns the whole spectrum vector
    a = _keep([np.array(x) for x in [0.0, 1.0, 19.62, 0.0]])
    ev = sorted(np.abs(uf.eigenvalues(*a)))
    assert np.allclose(ev, [np.sqrt(19.62), np.sqrt(19.62)])
    # batched: per-cell spectral radius (component axis is trailing)
    g_h = np.array([19.62, 9.81, 4.905])
    a = _keep([np.zeros(3), np.ones(3), g_h, np.zeros(3)])
    evb = np.abs(uf.eigenvalues(*a))
    got = np.maximum(evb[..., 0], evb[..., 1])
    assert np.allclose(got, np.sqrt(g_h))
    # pick reads a component of the packed spectrum (== trailing-axis index)
    assert np.allclose(uf.pick(uf.eigenvalues(*a), 1),
                       np.asarray(uf.eigenvalues(*a))[..., 1])
    # eigensystem reconstructs the matrix from ONE shared eigenbasis
    a = _keep([np.array(x) for x in _swe_jacobian(2.0, 3.0)])
    stack = np.asarray(uf.eigensystem(*a), dtype=float)
    assert stack.shape == (n + 2 * n * n,)
    lam = stack[:n]
    R = stack[n:n + n * n].reshape(n, n)
    L = stack[n + n * n:].reshape(n, n)
    A = np.array(_swe_jacobian(2.0, 3.0)).reshape(n, n)
    assert np.allclose(sorted(lam), [-1.0, 5.0])
    assert np.allclose(R @ np.diag(lam) @ L, A, atol=1e-10)
    assert np.allclose(L @ R, np.eye(n), atol=1e-12)
    # solve kernel == linalg.solve (whole vector; pick reads one component)
    args = [2.0, 1.0, 1.0, 3.0, 1.0, 2.0]
    sol = np.asarray(uf.solve(*args)).ravel()
    assert np.allclose(sol, np.linalg.solve([[2, 1], [1, 3]], [1, 2]))
    assert np.allclose(np.ravel(uf.pick(uf.solve(*args), 0)), sol[0])
    # inf-tolerance: fully poisoned -> +inf, never a LinAlgError
    a = _keep([np.array(x) for x in [np.inf, 1.0, 5.0, 4.0]])
    lams = np.asarray(uf.eigenvalues(*a)).ravel()
    assert all(np.isposinf(lams))
    # mixed batch masks PER MEMBER
    a = _keep([np.array([x, np.nan]) for x in _swe_jacobian(2.0, 3.0)])
    ev = np.asarray(uf.eigenvalues(*a))          # (batch=2, n=2)
    assert np.allclose(sorted(ev[0]), [-1.0, 5.0])   # finite member
    assert np.isposinf(ev[1]).all()                  # poisoned member
    # poisoned eigensystem returns the identity eigenbasis
    a = _keep([np.array(x) for x in [np.inf, 0.0, 0.0, np.inf]])
    out = np.asarray(uf.eigensystem(*a), dtype=float)
    lam, R, L = out[:n], out[n:n + n * n].reshape(n, n), out[n + n * n:].reshape(n, n)
    assert np.isposinf(lam).all()
    assert np.array_equal(R, np.eye(n)) and np.array_equal(L, np.eye(n))


# ── 3. HLL / HLLC flux VALUES + REQ-157 riemann honoured ────────────────────

class MiniSWE(Model):
    """Self-contained SWE [h, hu, hv (2-D)] — exercises the dimension-generic
    HLL/HLLC paths without any production model."""

    def __init__(self, dimension=2, **kwargs):
        var_names = ["h", "hu", "hv"][: dimension + 1]
        super().__init__(
            dimension=dimension,
            variables=var_names,
            parameters={"g": (9.81, "positive")},
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        u = hu / h
        g = self.parameters.g
        F = Matrix.zeros(self.n_variables, dim)
        F[0, :] = hu.T
        F[1:, :] = h * u * u.T + 0.5 * g * h**2 * Matrix.eye(dim)
        return ZArray(F)

    def eigenvalues(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        n = Matrix(self.normal[:dim])
        g = self.parameters.g
        un = (hu.T * n)[0] / h
        c = sqrt(g * h)
        return ZArray([un - c, un + c] + [un] * (dim - 1))


def _phys_flux_n(mrt, q, aux, p, n):
    F = np.asarray(mrt.flux(q, aux, p), dtype=float)
    P = np.asarray(mrt.hydrostatic_pressure(q, aux, p), dtype=float)
    return (F + P) @ n


def _num_flux(scheme_rt, qL, qR, aux, p, n):
    """``numerical_flux`` is the flux ALONE (n rows) — the trailing
    face-wavespeed row of REQ-212 was removed by the v6 solver design.  The
    face speed is still pinned here, read from its surviving home: the last
    row of the fused ``numerical_face`` kernel ([flux | D+ | D- | lambda])."""
    flux = np.asarray(
        scheme_rt.numerical_flux(qL, qR, aux, aux, p, n), dtype=float
    ).reshape(-1)
    face = np.asarray(
        scheme_rt.numerical_face(qL, qR, aux, aux, p, n), dtype=float
    ).reshape(-1)
    assert face.shape[0] == 3 * flux.shape[0] + 1
    return flux, face[-1]


def _state(dim, h, u, v=0.0):
    if dim == 1:
        return np.array([h, h * u])
    return np.array([h, h * u, h * v])


def test_riemann_flux_values():
    """HLL/HLLC numeric VALUES (no golden covers them): consistency
    F(q,q) == (F+P)@n, supersonic upwinding both signs, rotational
    antisymmetry, subsonic finiteness, HLLC contact preserved vs HLL smear —
    and the numpy solvers honour an EXPLICIT nsm.riemann while a defaulted NSM
    keeps each solver's own default (REQ-157)."""
    for dim in (1, 2):
        m = MiniSWE(dimension=dim)
        mrt = NumpyRuntimeModel(m)
        schemes = {"hll": HLL(m).to_runtime_numpy(),
                   "hllc": HLLC(m).to_runtime_numpy()}
        p = np.array(list(m.parameter_values.values()), dtype=float)
        aux = np.zeros(m.n_aux_variables)
        n = np.array([1.0] + [0.0] * (dim - 1))

        # consistency + wavespeed row
        q = _state(dim, h=2.0, u=0.7, v=-0.3)
        expect = _phys_flux_n(mrt, q, aux, p, n)
        got, lam = _num_flux(schemes["hll"], q, q, aux, p, n)
        np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)
        lam_local = float(np.asarray(
            schemes["hll"].local_max_abs_eigenvalue(q, aux, p, n), dtype=float))
        np.testing.assert_allclose(lam, lam_local, rtol=1e-12)
        got, _ = _num_flux(schemes["hllc"], q, q, aux, p, n)
        np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)

        # supersonic upwinding, both signs; subsonic finite; antisymmetry
        qLs = _state(dim, h=1.0, u=20.0, v=1.0)
        qRs = _state(dim, h=0.4, u=15.0, v=-2.0)
        qLm = _state(dim, h=1.0, u=-20.0, v=1.0)
        qRm = _state(dim, h=0.4, u=-15.0, v=-2.0)
        qL = _state(dim, h=2.0, u=0.4, v=0.1)
        qR = _state(dim, h=1.0, u=0.8, v=-0.2)
        for name, rt in schemes.items():
            got, _ = _num_flux(rt, qLs, qRs, aux, p, n)
            np.testing.assert_allclose(
                got, _phys_flux_n(mrt, qLs, aux, p, n), rtol=1e-7, atol=1e-7,
                err_msg=f"{name} supersonic-right")
            got, _ = _num_flux(rt, qLm, qRm, aux, p, n)
            np.testing.assert_allclose(
                got, _phys_flux_n(mrt, qRm, aux, p, n), rtol=1e-7, atol=1e-7,
                err_msg=f"{name} supersonic-left")
            got, lam = _num_flux(rt, qL, qR, aux, p, n)
            assert np.all(np.isfinite(got)) and np.isfinite(lam)
            fwd, lam_f = _num_flux(rt, qL, qR, aux, p, n)
            rev, lam_r = _num_flux(rt, qR, qL, aux, p, -n)
            np.testing.assert_allclose(fwd, -rev, rtol=1e-7, atol=1e-7,
                                       err_msg=f"{name} antisymmetry")
            np.testing.assert_allclose(lam_f, lam_r, rtol=1e-12)

    # HLLC contact: pure shear — mass + normal-momentum flux exact, HLL smears
    m = MiniSWE(dimension=2)
    mrt = NumpyRuntimeModel(m)
    hll = HLL(m).to_runtime_numpy()
    hllc = HLLC(m).to_runtime_numpy()
    p = np.array(list(m.parameter_values.values()), dtype=float)
    aux = np.zeros(m.n_aux_variables)
    n = np.array([1.0, 0.0])
    h, un = 1.5, 0.3
    qL = np.array([h, h * un, h * 1.2])
    qR = np.array([h, h * un, h * -1.2])
    fL = _phys_flux_n(mrt, qL, aux, p, n)
    got_hllc, _ = _num_flux(hllc, qL, qR, aux, p, n)
    got_hll, _ = _num_flux(hll, qL, qR, aux, p, n)
    np.testing.assert_allclose(got_hllc[:2], fL[:2], rtol=1e-7, atol=1e-7)
    assert abs(got_hll[2] - got_hllc[2]) > 1e-3

    # REQ-157: explicit nsm.riemann honoured by BOTH numpy solvers; a
    # defaulted NSM keeps each solver's own default.
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm import solver_numpy as SN
    from zoomy_core.fvm.riemann_solvers import (
        NonconservativeRusanov, PositiveNonconservativeHLL,
        PositiveNonconservativeRusanov)
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.numerics.numerical_system_model import (
        NumericalSystemModel, ReconstructionSpec)

    def _numerics(solver_cls, nsm):
        solver = solver_cls(
            time_end=1.0, compute_dt=timestepping.adaptive(CFL=0.45))
        solver.nsm = nsm
        return solver._build_numerics(solver._get_symbolic_model(nsm))

    def _nsm(riemann=None):
        return NumericalSystemModel.from_model(
            SWE(dimension=2, parameters={"g": 9.81}),
            riemann=riemann,
            reconstruction=ReconstructionSpec(order=1))

    nsm = _nsm(riemann=PositiveNonconservativeHLL)
    assert nsm.riemann_explicit is True
    assert isinstance(_numerics(SN.HyperbolicSolver, nsm),
                      PositiveNonconservativeHLL)
    assert isinstance(_numerics(SN.FreeSurfaceFlowSolver, nsm),
                      PositiveNonconservativeHLL)
    nsm = _nsm()
    assert nsm.riemann_explicit is False
    assert type(_numerics(SN.HyperbolicSolver, nsm)) is NonconservativeRusanov
    assert type(_numerics(SN.FreeSurfaceFlowSolver, nsm)) \
        is PositiveNonconservativeRusanov


# ── 4. t / dt / x binding + per-step aux application ────────────────────────

class RainTracer(StructuredDerivativeModel):
    """d_t c + a d_x c = r_o,  r_o = rain rate active only while t < T_rain
    (a plain tracer so the KP-hinv desingularisation is not auto-attached)."""

    dimension = 1
    variables = ["c"]
    parameters = {
        "a":         (1.0, "real"),
        "rain_rate": (0.5, "nonnegative"),
        "T_rain":    (0.05, "positive"),
    }

    def __init__(self, **kw):
        self._coupling_bcs = kw.pop("boundary_conditions", None)
        super().__init__(aux_variables=["r_o"], **kw)

    def _build_function_groups(self):
        return {}

    def flux(self):
        F = sp.Matrix.zeros(1, 1)
        F[0, 0] = self.parameters.a * self.Q.c
        return ZArray(F)

    def eigenvalues(self):
        return ZArray([self.parameters.a])

    def update_aux_variables(self):
        t = sp.Symbol("t", real=True)
        p = self.parameters
        return ZArray([sp.Piecewise((p.rain_rate, t < p.T_rain),
                                    (sp.S.Zero, True))])

    def source(self):
        S = sp.Matrix.zeros(1, 1)
        S[0, 0] = self.aux_variables.r_o
        return ZArray(S)


def test_runtime_binds_t_dt_x():
    """REQ-185 runtime binding (the solver golden has no time-dependent
    source): the rain gate turns OFF after T_rain (t reaches update_aux), a
    dt-dependent source matches the hand evaluation (dt bound), position is a
    length-3 vector even for 1-D — and the numpy FVM applies
    ``update_aux_variables`` EVERY step (hinv == 1/h after a short dam break;
    masked at rest, so invisible to X01)."""
    # t reaches update_aux
    sm = SystemModel.from_model(RainTracer())
    rt = NumpyRuntimeModel.from_system_model(sm)
    p = rt.parameters
    Q = np.array([1.0])
    Qaux = np.zeros(rt.n_aux_variables)
    x = np.zeros(3)
    r_before = float(np.ravel(rt.update_aux_variables(Q, Qaux, p, 0.02, x))[0])
    r_after = float(np.ravel(rt.update_aux_variables(Q, Qaux, p, 0.20, x))[0])
    assert r_before == pytest.approx(0.5)
    assert r_after == pytest.approx(0.0)

    # dt bound in source
    sm = SystemModel.from_model(RainTracer())
    c = sm.state[0]
    sm.source = ZArray(sp.Matrix([[sm.source[0, 0] + DT_SYMBOL * c]]))
    rt = NumpyRuntimeModel.from_system_model(sm)
    assert rt.source_needs_dt is True
    p = rt.parameters
    cval, dt = 2.0, 0.3
    Q = np.array([[cval]])
    Qaux = np.zeros((rt.n_aux_variables, 1))
    x = np.zeros((3, 1))
    val = float(np.asarray(rt.source(Q, Qaux, p, 0.0, dt, x)).ravel()[0])
    assert val == pytest.approx(dt * cval), f"dt not bound: {val} != {dt*cval}"

    # length-3 position regardless of dimension
    sm = SystemModel.from_model(RainTracer())
    x_sym, y_sym, z_sym = (sm.position.x, sm.position.y, sm.position.z)
    sm.source = ZArray(sp.Matrix([[x_sym + 10 * y_sym + 100 * z_sym]]))
    rt = NumpyRuntimeModel.from_system_model(sm)
    p = rt.parameters
    ncell = 4
    Q = np.ones((1, ncell))
    Qaux = np.zeros((rt.n_aux_variables, ncell))
    pos = np.array([[0.1, 0.2, 0.3, 0.4],
                    [1.0, 2.0, 3.0, 4.0],
                    [0.5, 0.5, 0.5, 0.5]])
    out = np.asarray(rt.source(Q, Qaux, p, 0.0, 0.0, pos)).ravel()
    np.testing.assert_allclose(out, pos[0] + 10 * pos[1] + 100 * pos[2])

    # update_aux applied each step: hinv == 1/h after a short dam break
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    from zoomy_core.model.boundary_conditions import Extrapolation
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.initial_conditions as IC
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel import register_aux, regularize_pow

    smx = SystemModel.from_model(SME(
        level=1, parameters={"nu": 1e-3, "lambda_s": 0.0},
        closures=[Newtonian(), NavierSlip(), StressFree()],
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")]))
    hsym = next(s for s in smx.state if str(s) == "h")
    kp = sp.sqrt(2) * sp.Max(hsym, 0) / sp.sqrt(
        hsym ** 4 + sp.Max(hsym, sp.Float(1e-3)) ** 4)
    smx.apply(register_aux("hinv", kp, positive=True)) \
       .apply(regularize_pow("h", "hinv"))

    def _ic(xv):
        hval = 1.5 if float(xv[0]) < 5.0 else 0.75
        return np.array([0.0, hval] + [0.0] * (smx.n_equations - 2))

    smx.initial_conditions = IC.UserFunction(function=_ic)
    smx.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)
    nsm = NumericalSystemModel.from_system_model(
        smx, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=0.02,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Qs, Qauxs = solver.solve(mesh, nsm, write_output=False)
    Qs = np.asarray(Qs, float)
    Qauxs = np.asarray(Qauxs, float)
    hinv_row = [str(s) for s in smx.aux_state].index("hinv")
    hcells = Qs[1, :50]
    hinv_cells = Qauxs[hinv_row, :50]
    assert np.any(hinv_cells != 0.0), "hinv aux never populated (still zero)"
    assert np.allclose(hinv_cells, 1.0 / hcells, atol=1e-6)
    assert np.all(np.isfinite(Qs))
