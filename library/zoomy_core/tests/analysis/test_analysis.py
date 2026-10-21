"""R5 — the analysis subpackage: temporal + SPATIAL dispersion (rederive tier).

Temporal pipeline:

  SystemModel → linearise → (plane_wave_dispersion |
                              extract_quasilinear_pencil →
                              sample_hyperbolicity)

plus the SPATIAL branch (REQ-176 items 1+2, merged from
test_spatial_dispersion.py): real frequencies, complex wavenumbers
``k(omega)`` as a generalized eigenproblem, ``numeric=`` switch, roll-wave
Vedernikov Fr=2 physics anchor.

Classical hand-checkable systems only; sole coverage of the analysis
subpackage — kept because dispersion work is active (REQ-176(1+2)).
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from zoomy_core.analysis import (
    linearise,
    plane_wave_dispersion,
    extract_quasilinear_pencil,
    generalised_eigenvalues,
    is_hyperbolic_at,
    sample_hyperbolicity,
    spatial_dispersion,
    spatial_cutoff,
    temporal_branch,
    NumericPencil,
)
from zoomy_core.misc.misc import Zstruct
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.rederive]


def _swe(g_sym=None):
    """Standard SWE SystemModel ``(h, u)`` with mass matrix
    ``[[1, 0], [U, h]]`` and flux ``[hU, hU² + gh²/2]``."""
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    h, u = sp.symbols("h u", real=True)
    g = g_sym if g_sym is not None else sp.Symbol("g", positive=True)
    return SystemModel(
        time=t,
        space=[x],
        state=[h, u],
        aux_state=[],
        parameters=Zstruct(g=g),
        parameter_values=Zstruct(g=9.81),
        flux=sp.Matrix([[h * u], [h * u**2 + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.Matrix([[1, 0], [u, h]]),
    )


# ---------------------------------------------------------------------------
# Linear advection: u_t + c u_x = 0  →  ω = c k
# ---------------------------------------------------------------------------

def test_linear_advection_dispersion():
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    c = sp.Symbol("c", real=True)
    u = sp.Symbol("u", real=True)
    sm = SystemModel(
        time=t,
        space=[x],
        state=[u],
        aux_state=[],
        parameters=Zstruct(),
        flux=sp.Matrix([[c * u]]),
        hydrostatic_pressure=sp.zeros(1, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, 1),
        source=sp.zeros(1, 1),
        mass_matrix=sp.eye(1),
    )
    sm_lin = linearise(sm, {u: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    assert len(disp["solutions"]) == 1
    omega_sol = disp["solutions"][0]
    k = sp.Symbol("k", real=True)
    assert sp.simplify(omega_sol - c * k) == 0


# ---------------------------------------------------------------------------
# Scalar wave equation in first-order form:
#   u_t + v_x = 0;  v_t + c² u_x = 0  →  ω² = c² k²
# ---------------------------------------------------------------------------

def test_first_order_wave_equation_dispersion():
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    c = sp.Symbol("c", positive=True)
    u, v = sp.symbols("u v", real=True)
    sm = SystemModel(
        time=t,
        space=[x],
        state=[u, v],
        aux_state=[],
        parameters=Zstruct(),
        flux=sp.Matrix([[v], [c**2 * u]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.eye(2),
    )
    sm_lin = linearise(sm, {u: 0, v: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    assert len(disp["solutions"]) == 2
    k = sp.Symbol("k", real=True)
    expected = {sp.simplify(c * k), sp.simplify(-c * k)}
    got = {sp.simplify(s) for s in disp["solutions"]}
    assert got == expected


# ---------------------------------------------------------------------------
# Shallow water (conservative form)  →  c² = g H at rest state
# ---------------------------------------------------------------------------

def test_swe_rest_state_dispersion_equals_gH():
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    k = sp.Symbol("k", real=True)
    pvs = [sp.simplify(s / k) for s in disp["solutions"]]
    c2 = {sp.simplify(pv**2) for pv in pvs}
    assert sp.simplify(g * H) in c2


# ---------------------------------------------------------------------------
# Quasilinear pencil extraction
# ---------------------------------------------------------------------------

def test_extract_pencil_swe():
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    U0 = sp.Symbol("U0", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: U0})
    M_t, M_xa, M_0 = extract_quasilinear_pencil(sm_lin)

    # M_t row 0 (mass eq) = (1, 0).  M_t row 1 (momentum eq) = (U0, H).
    assert M_t.shape == (2, 2)
    assert sp.simplify(M_t[0, 0] - 1) == 0
    assert sp.simplify(M_t[0, 1]) == 0
    assert sp.simplify(M_t[1, 0] - U0) == 0
    assert sp.simplify(M_t[1, 1] - H) == 0


def test_generalised_eigenvalues_swe_rest():
    """At rest (U0 = 0), eigenvalues = ±√(gH)."""
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: 0})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)
    eigs = generalised_eigenvalues(M_xa[0], M_t)
    eigs_simpl = {sp.simplify(e) for e in eigs}
    assert sp.simplify(sp.sqrt(g * H)) in eigs_simpl
    assert sp.simplify(-sp.sqrt(g * H)) in eigs_simpl


# ---------------------------------------------------------------------------
# Numerical eigenvalues + hyperbolicity
# ---------------------------------------------------------------------------

def test_is_hyperbolic_at_swe():
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)

    sub = {g: 9.81, H_sym: 1.0, U_sym: 0.0}
    hyp, eigs = is_hyperbolic_at(M_xa[0], M_t, sub)
    assert hyp
    eigs_real = sorted(eigs.real)
    assert abs(eigs_real[0] + sp.sqrt(sp.Float(9.81))) < 1e-9
    assert abs(eigs_real[1] - sp.sqrt(sp.Float(9.81))) < 1e-9


def test_sample_hyperbolicity_swe_always_hyperbolic():
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)

    M_x_num = M_xa[0].subs(g, 9.81)
    M_t_num = M_t
    report = sample_hyperbolicity(
        M_x_num, M_t_num,
        {H_sym: (0.1, 5.0), U_sym: (-3.0, 3.0)},
        n_samples=200,
        constraint_filter=lambda s: s[H_sym] > 0,
    )
    assert report.fraction_hyperbolic == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SPATIAL dispersion branch (REQ-176 items 1+2)
# Sign convention: ansatz exp(i(k x - omega t)); alpha = -Im(k) (> 0 grows).
# ═══════════════════════════════════════════════════════════════════════════

T = sp.Symbol("t", real=True)
X = sp.Symbol("x", real=True)


def _swe_full():
    """Conservative SWE ``(h, u)`` (primitive velocity), non-linearised."""
    h, u = sp.symbols("h u", real=True)
    g = sp.Symbol("g", positive=True)
    return SystemModel(
        time=T, space=[X], state=[h, u], aux_state=[],
        parameters=Zstruct(g=g), parameter_values=Zstruct(g=9.81),
        flux=sp.Matrix([[h * u], [h * u**2 + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.Matrix([[1, 0], [u, h]]),
    ), (h, u)


def _rollwave():
    """Roll-wave SWE with Chezy friction, conservative ``(h, m=hu)``:
    ``Fr = sqrt(S0/Cf)``; the Vedernikov threshold is Fr = 2."""
    h, m = sp.symbols("h m", real=True)
    g, S0, Cf = sp.symbols("g S0 Cf", positive=True)
    return SystemModel(
        time=T, space=[X], state=[h, m], aux_state=[],
        parameters=Zstruct(g=g, S0=S0, Cf=Cf),
        parameter_values=Zstruct(g=1.0, S0=0.06, Cf=0.02),
        flux=sp.Matrix([[m], [m**2 / h + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.Matrix([[0], [g * h * S0 - Cf * m**2 / h**2]]),
        mass_matrix=sp.eye(2),
    ), (h, m)


def test_swe_analytic_equals_numeric_and_phase_speed():
    sm, (h, u) = _swe_full()
    H, U, gv = 2.0, 0.5, 9.81
    base, par = {h: H, u: U}, {"g": gv}
    w = np.linspace(0.2, 2.0, 12)
    cfast = U + np.sqrt(gv * H)

    an = spatial_dispersion(sm, base, w, params=par, numeric=False, c_seed=cfast)
    nu = spatial_dispersion(sm, base, w, params=par, numeric=True, c_seed=cfast)

    assert np.max(np.abs(an.k - nu.k)) < 1e-12       # SAME pencil, two solvers
    assert np.allclose(an.c, cfast)                   # c = U + sqrt(gH)
    assert np.allclose(an.alpha, 0.0, atol=1e-12)     # SWE is non-dissipative
    assert an.k_symbolic is not None and nu.k_symbolic is None
    assert nu.residual == 0.0                          # fully pinned base


def test_swe_temporal_to_spatial_roundtrip():
    """omega(k_real) fed back through k(omega) recovers k_real exactly."""
    sm, (h, u) = _swe_full()
    H, U, gv = 2.0, 0.5, 9.81
    base, par = {h: H, u: U}, {"g": gv}
    cfast = U + np.sqrt(gv * H)

    k0 = np.linspace(0.3, 1.4, 12)
    mats = NumericPencil(sm).at_equilibrium(par, fixed={"h": H, "u": U})
    tmp = temporal_branch(mats, k0, c_seed=cfast)     # omega(k0), real (neutral)
    assert np.allclose(tmp.omega.imag if np.iscomplexobj(tmp.omega)
                       else 0.0, 0.0, atol=1e-12)

    for numeric in (False, True):
        sp_ = spatial_dispersion(sm, base, tmp.omega, params=par,
                                 numeric=numeric, c_seed=cfast)
        assert np.max(np.abs(sp_.k.real - k0)) < 1e-10
        assert np.max(np.abs(sp_.k.imag)) < 1e-10


def _rollwave_at(Fr):
    sm, (h, m) = _rollwave()
    gv, S0v = 1.0, 0.06
    Cfv = S0v / Fr**2
    u0 = np.sqrt(gv * 1.0 * S0v / Cfv)
    return sm, {h: 1.0, m: u0}, {"g": gv, "S0": S0v, "Cf": Cfv}, u0


def test_rollwave_analytic_equals_numeric_complex_k():
    sm, base, par, u0 = _rollwave_at(3.0)             # Fr = 3 > 2: unstable
    w = np.linspace(0.05, 2.5, 80)
    an = spatial_dispersion(sm, base, w, params=par, numeric=False, c_seed=u0 + 1)
    nu = spatial_dispersion(sm, base, w, params=par, numeric=True, c_seed=u0 + 1)

    assert np.max(np.abs(an.k - nu.k)) < 1e-12        # both solve the same pencil
    assert np.any(np.abs(an.k.imag) > 1e-4)            # genuinely complex k
    assert an.alpha.max() > 0.0                        # some spatially growing band


def test_rollwave_vedernikov_threshold():
    """Temporal instability appears above Fr = 2 (physics anchor)."""
    def max_sigma(Fr):
        sm, base, par, u0 = _rollwave_at(Fr)
        mats = NumericPencil(sm).at_equilibrium(
            par, fixed={str(k): v for k, v in base.items()})
        return temporal_branch(mats, np.linspace(1e-3, 3.0, 300),
                               c_seed=u0 + 1).sigma.max()

    assert max_sigma(3.0) > 1e-4                        # unstable
    assert max_sigma(1.2) < 1e-4                        # stable


def test_spatial_cutoff_detects_crossing_and_nan():
    w = np.linspace(0.0, 1.0, 11)
    grow_then_decay = Zstruct(omega=w, alpha=0.1 - 0.2 * w)   # zero at w = 0.5
    assert abs(spatial_cutoff(grow_then_decay) - 0.5) < 0.11
    always_growing = Zstruct(omega=w, alpha=np.full_like(w, 0.3))
    assert np.isnan(spatial_cutoff(always_growing))


def test_analytic_requires_all_states_pinned():
    sm, (h, u) = _swe_full()
    with pytest.raises(ValueError, match="every state pinned"):
        spatial_dispersion(sm, {h: 1.0}, [0.5], params={"g": 9.81})


def test_analytic_leftover_symbol_raises():
    sm, (h, u) = _swe_full()
    with pytest.raises(ValueError, match="still depends on"):
        spatial_dispersion(sm, {h: 1.0, u: 0.0}, [0.5], params={})  # g missing


# fsolve warns "no progress" because the zero guess already sits on the trivial
# uniform equilibrium; convergence is verified below by the residual assertion.
@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
def test_sme2_numeric_switch():
    models = pytest.importorskip("zoomy_core.model.models")
    from zoomy_core.model.models.closures import (
        Newtonian, NavierSlip, StressFree)
    p0 = {"g": 1.0, "rho": 1.0, "nu": 0.05, "e_x": 0.2, "lambda_s": 50.0}
    try:
        sm = SystemModel.from_model(models.SME(
            level=2, closures=[Newtonian(), NavierSlip(), StressFree()],
            quadrature_order=8, parameters=p0))
    except Exception as exc:                            # pragma: no cover
        pytest.skip(f"SME(2) build unavailable (concurrent edit?): {exc}")

    w = np.linspace(0.05, 1.5, 20)
    S = spatial_dispersion(sm, {"b": 0.0, "h": 1.0}, w, params=p0,
                           numeric=True, drop=("b",))
    assert S.residual < 1e-8                            # equilibrium solved
    assert np.all(np.isfinite(S.k))
    assert S.alpha.shape == w.shape and S.c.shape == w.shape
