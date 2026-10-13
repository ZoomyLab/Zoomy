"""R1 — the merged EXTERNAL-TRUTH anchors (rederive tier; spec §1c).

This suite is the MANDATORY validator for any golden re-bless: a golden
detects CHANGE, not WRONGNESS — these pin the derived systems term-by-term
against published references and first principles.

Merged from: test_sme_reference.py (Kowalski & Torrilhon 2019) +
test_mlswe_reference.py (Aguillon/Hoernschemeyer/Sainte-Marie eq (8)/(9)) +
test_vam_reference.py (Escalante et al. JCP 2024, incl. predictor/corrector +
REQ-80 flux routing) + the in-plane first-principles G-matrix cross-check
(test_inplane_viscous_couplings_req176c.py) + the Sigma3D conservative-sigma
identity and shallow recovery (test_sigma3d.py).

Shape, count and smoke checks are NEVER sufficient — pin every term.
"""
import warnings

import pytest
import sympy as sp

from zoomy_core.model.models import SME, MLSWE, VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.model, pytest.mark.rederive]


# ═══════════════════════════════════════════════════════════════════════════
# SME vs Kowalski & Torrilhon 2019 — term-by-term (incl. friction)
# ═══════════════════════════════════════════════════════════════════════════

def _kt_reference_rows(sm, N):
    """K&T general 1-D moment system in OUR variables: [b-row, mass, mean
    (4.7), moment_1..N (4.10)] aligned with sm.state = [b, h, q_0..q_N], each
    pre-multiplied by the (-1)^i row-sign that maps theirs onto ours.
    Appendix-B Legendre integrals computed HERE in sympy, never hand-copied."""
    t, x = sm.time, sm.space[0]
    Fn = lambda n: sp.Function(n, real=True)(t, x)
    h, b = Fn("h"), Fn("b")
    q = [Fn(f"q_{i}") for i in range(N + 1)]
    g = sm.parameters.g
    # their nu/lambda (kinematic) = our lambda_s/rho; interior C-term nu = ours
    lam_s = sm.parameters.lambda_s / sm.parameters.rho
    nu = sm.parameters.nu
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    zeta, s_ = sp.symbols("zeta s_", nonnegative=True)
    phi = [sp.legendre(j, 1 - 2 * zeta) for j in range(N + 1)]   # THEIR basis

    def I01(e):
        return sp.integrate(sp.expand(e), (zeta, 0, 1))

    A = lambda i, j, k: (2 * i + 1) * I01(phi[i] * phi[j] * phi[k])
    B = lambda i, j, k: (2 * i + 1) * I01(
        sp.diff(phi[i], zeta)
        * sp.integrate(phi[j].subs(zeta, s_), (s_, 0, zeta)) * phi[k])
    Cf = lambda i, j: I01(sp.diff(phi[i], zeta) * sp.diff(phi[j], zeta))

    um = q[0] / h
    alpha = {i: (-1) ** i * q[i] / h for i in range(1, N + 1)}
    D = {j: Dx(h * alpha[j]) for j in range(1, N + 1)}
    J = range(1, N + 1)

    rows = [Dt(b)]                                               # inert bed
    rows.append(Dt(h) + Dx(h * um))                              # mass
    e_x = sm.parameters.e_x
    rows.append(                                                 # mean (4.7)
        Dt(h * um)
        + Dx(h * (um**2 + sum(alpha[j]**2 / (2 * j + 1) for j in J))
             + g * h**2 / 2)
        + g * h * Dx(b)
        - g * h * e_x
        + lam_s * (um + sum(alpha[j] for j in J)))
    for i in J:                                                  # moments (4.10)
        row = (Dt(h * alpha[i])
               + Dx(h * (2 * um * alpha[i]
                         + sum(A(i, j, k) * alpha[j] * alpha[k]
                               for j in J for k in J)))
               - um * D[i]
               + sum(B(i, j, k) * D[j] * alpha[k] for j in J for k in J)
               + (2 * i + 1) * (lam_s * (um + sum(alpha[j] for j in J))
                                + (nu / h) * sum(Cf(i, j) * alpha[j]
                                                 for j in J)))
        rows.append((-1) ** i * row)        # their row -> our row sign
    return rows


@pytest.mark.parametrize("level", [1, 2, 3])
def test_sme_rows_match_kowalski_torrilhon(level):
    sm = SystemModel.from_model(SME(
        closures=[Newtonian(), NavierSlip(), StressFree()], level=level))
    assert [str(s) for s in sm.state] == (
        ["b", "h"] + [f"q_{i}" for i in range(level + 1)])
    rv = sm.reconstruct_residuals()
    refs = _kt_reference_rows(sm, level)
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        diff = sp.simplify(sp.expand(sp.sympify(mine).doit())
                           - sp.expand(ref.doit()))
        name = str(sm.state[i])
        assert diff == 0, (
            f"SME({level}) row {i} ({name}) differs from K&T: {diff}")


# ═══════════════════════════════════════════════════════════════════════════
# MLSWE vs Aguillon / Hoernschemeyer / Sainte-Marie eq (8) + upwinding (9)
# ═══════════════════════════════════════════════════════════════════════════

def _fractions(sm, N):
    ls = [getattr(sm.parameters, f"l_{j}") for j in range(1, N)]
    return ls + [1 - sum(ls)]


def _mlswe_reference_rows(sm, N, ustar_of):
    """Paper system (8) rows in OUR variables/order [b, h, q_1..q_N]; momentum
    rows pre-multiplied by l_alpha.  ``ustar_of(a, G)`` returns the interface
    velocity u*_{a+1/2} for the paper-orientation exchange flux G."""
    t, x = sm.time, sm.space[0]
    Fn = lambda n: sp.Function(n, real=True)(t, x)
    h, b = Fn("h"), Fn("b")
    q = [Fn(f"q_{a}_0") for a in range(1, N + 1)]   # layer alpha, mode 0
    g = sm.parameters.g
    l = _fractions(sm, N)
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    u = [q[a] / (l[a] * h) for a in range(N)]
    ubar = sum(q) / h

    G = {0: sp.S.Zero, N: sp.S.Zero}            # G_{1/2} and G_{N+1/2}
    for a in range(1, N):
        G[a] = sum(l[j] * Dx(h * (u[j] - ubar)) for j in range(a))

    rows = [Dt(b), Dt(h) + Dx(h * ubar)]
    for a in range(1, N + 1):
        transfer = (ustar_of(a, G[a]) * G[a]
                    - ustar_of(a - 1, G[a - 1]) * G[a - 1])
        rows.append(l[a - 1] * (
            Dt(h * u[a - 1]) + Dx(h * u[a - 1]**2 + g * h**2 / 2)
            + g * h * Dx(b) - transfer / l[a - 1]))
    return rows, u


def _assert_mlswe_rows_equal(sm, refs):
    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0}
    rv = sm.reconstruct_residuals()
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        mine = sp.sympify(mine).subs(inviscid)
        diff = sp.simplify(sp.expand(mine.doit()) - sp.expand(ref.doit()))
        assert diff == 0, (
            f"row {i} ({sm.state[i]}) differs from Hoernschemeyer eq (8): "
            f"{diff}")


@pytest.mark.parametrize("n_layers", [2, 3])
def test_mlswe_mean_interface_matches_hornschemeyer_eq8(n_layers):
    sm = SystemModel.from_model(MLSWE(
        closures=[Newtonian(), NavierSlip(), StressFree()],
        n_layers=n_layers, interface_velocity="mean"))
    assert [str(s) for s in sm.state] == (
        ["b", "h"] + [f"q_{a}_0" for a in range(1, n_layers + 1)])

    def ustar_mean(a, G):
        if a == 0 or a == n_layers:
            return sp.S.Zero                    # multiplies G = 0
        return (u[a - 1] + u[a]) / 2

    refs, u = _mlswe_reference_rows(sm, n_layers, lambda a, G: 0)
    refs, u = _mlswe_reference_rows(sm, n_layers, ustar_mean)
    _assert_mlswe_rows_equal(sm, refs)


@pytest.mark.parametrize("branch", ["below", "above"])
def test_mlswe_upwind_interface_matches_eq9_branches(branch):
    """Their (9) is a sign switch per interface; N=2 has ONE internal
    interface — pin BOTH Piecewise branches (G_ours >= 0 <-> G_theirs <= 0)."""
    n_layers = 2
    sm = SystemModel.from_model(MLSWE(
        closures=[Newtonian(), NavierSlip(), StressFree()],
        n_layers=n_layers, interface_velocity="upwind"))

    take_true = branch == "below"

    def pick_branch(expr):
        def _sel(pw):
            return pw.args[0][0] if take_true else pw.args[1][0]
        return expr.replace(lambda e: isinstance(e, sp.Piecewise), _sel)

    refs_holder = {}

    def ustar_up(a, G):
        if a == 0 or a == n_layers:
            return sp.S.Zero
        u = refs_holder["u"]
        return u[a - 1] if take_true else u[a]

    refs, u = _mlswe_reference_rows(sm, n_layers, lambda a, G: 0)
    refs_holder["u"] = u
    refs, u = _mlswe_reference_rows(sm, n_layers, ustar_up)

    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0}
    rv = sm.reconstruct_residuals()
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        mine = pick_branch(sp.sympify(mine).subs(inviscid))
        diff = sp.simplify(sp.expand(mine.doit()) - sp.expand(ref.doit()))
        assert diff == 0, (
            f"row {i} ({sm.state[i]}) [{branch}-branch] differs from "
            f"Hoernschemeyer eq (8)+(9): {diff}")


# ═══════════════════════════════════════════════════════════════════════════
# VAM vs Escalante et al. JCP 2024 — rows, constraints, projection, REQ-80
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def vam1():
    model = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1)
    sm = SystemModel.from_model(model)
    return model, sm


@pytest.fixture(scope="module")
def vam_reference(vam1):
    """Escalante eq (4)/(5) in OUR variables, plus mapping helpers."""
    _, sm = vam1
    t, x = sm.time, sm.space[0]

    def Fn(name):
        return sp.Function(name, real=True)(t, x)

    hf, bf = Fn("h"), Fn("b")
    q0, q1, r0, r1, P0, P1 = (Fn(n) for n in ("q_0", "q_1", "r_0", "r_1",
                                              "P_0", "P_1"))
    rho = sm.parameters.rho
    g = sm.parameters.g
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    u0, u1, w0, w1 = q0 / hf, -q1 / hf, r0 / hf, -r1 / hf
    p0, p1 = P0 / rho, -P1 / rho
    w2 = -(w0 + w1) + (u0 + u1) * Dx(bf)

    R = {
        "h":   Dt(hf) + Dx(hf * u0),
        "q_0": Dt(hf * u0) + Dx(hf * u0**2 + hf * u1**2 / 3 + hf * p0)
               + g * hf * Dx(bf + hf) + 2 * p1 * Dx(bf),
        "r_0": Dt(hf * w0) + Dx(hf * u0 * w0 + hf * u1 * w1 / 3) - 2 * p1,
        "q_1": Dt(hf * u1) + Dx(2 * hf * u0 * u1 + hf * p1) - u0 * Dx(hf * u1)
               - (3 * p0 - p1) * Dx(hf) - 6 * (p0 - p1) * Dx(bf),
        "r_1": Dt(hf * w1)
               + Dx(hf * u0 * w1 + u1 * (hf * w0 + sp.Rational(2, 5) * hf * w2))
               + (w2 / 5 - w0) * Dx(hf * u1) + 6 * (p0 - p1),
    }
    C = {
        "P_0": hf * Dx(u0) + Dx(hf * u1) / 3 + u1 * Dx(hf) / 3
               + 2 * (w0 - u0 * Dx(bf)),
        "P_1": hf * Dx(u0) + u1 * Dx(hf) + 2 * (u1 * Dx(bf) - w1),
    }
    sign = {"h": 1, "q_0": 1, "r_0": 1, "q_1": -1, "r_1": -1}
    # Escalante eq (4) is the FLAT inviscid column; VAM now carries the general
    # downslope body force −g·e_x (cid 171), so specialise to e_x=0 to compare.
    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0,
                sm.parameters.e_x: 0}
    return {"R": R, "C": C, "sign": sign, "inviscid": inviscid, "Fn": Fn}


def test_vam1_dynamic_rows_match_escalante_eq4(vam1, vam_reference):
    _, sm = vam1
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    for name, ref in vam_reference["R"].items():
        mine = sp.sympify(rv[names.index(name)]).subs(vam_reference["inviscid"])
        diff = sp.simplify(sp.expand(mine.doit())
                           - vam_reference["sign"][name] * sp.expand(ref.doit()))
        assert diff == 0, f"row {name} differs from Escalante eq (4): {diff}"


def test_vam1_constraints_match_escalante_eq5(vam1, vam_reference):
    _, sm = vam1
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    for name, ref in vam_reference["C"].items():
        mine = sp.expand(sp.sympify(rv[names.index(name)]).doit())
        diff = sp.simplify(mine + sp.expand(ref.doit()))   # ours = -theirs
        assert diff == 0, f"constraint {name} differs from eq (5): {diff}"


def test_vam1_advective_divergences_routed_as_flux(vam1):
    """REQ-80: the conservative parts of the k=1 rows sit in the FLUX slot,
    and the bed-slope vertical-momentum flux d_x(A d_x b) is a CONSERVATIVE
    flux carrying the frozen gradient-aux b_x — NOT an off-diagonal diffusion
    entry (which dropped it at runtime and blew up the Escalante bump)."""
    _, sm = vam1
    names = [str(s) for s in sm.state]
    i_q1 = names.index("q_1")
    i_r1 = names.index("r_1")
    Fq1 = sp.sympify(sm.flux[i_q1, 0])
    Fr1 = sp.sympify(sm.flux[i_r1, 0])
    by_name = {str(s): s for s in sm.state}
    q0, q1, r0, r1, P1, h = (by_name[n] for n in
                             ("q_0", "q_1", "r_0", "r_1", "P_1", "h"))
    b_x = sp.Symbol("b_x", real=True)
    rho = sm.parameters.rho
    bed = sp.Rational(2, 5) * b_x * q0 * q1 / h - sp.Rational(2, 5) * b_x * q1**2 / h
    assert sp.simplify(Fq1 - (2*q0*q1/h + h*P1/rho)) == 0, f"q_1 flux: {Fq1}"
    assert sp.simplify(
        Fr1 - (q0*r1/h + q1*r0/h - sp.Rational(2, 5)*q1*(r0 - r1)/h + bed)) == 0, \
        f"r_1 flux: {Fr1}"
    assert sm.diffusion_matrix is None or all(
        sm.diffusion_matrix[i_r1, j, 0, 0] == 0
        for j in range(len(names))), "bed slope leaked into diffusion_matrix"


# ═══════════════════════════════════════════════════════════════════════════
# In-plane viscous h-column — first-principles G-matrix cross-check (REQ-176c)
# ═══════════════════════════════════════════════════════════════════════════

def test_inplane_h_column_matches_G_matrix_first_principles():
    """Build G_kj = int_0^1 zeta phi_j' phi_k dzeta from the shifted-Legendre
    basis, form V[q_k<-h] = -4 nu (abar_k + (2k+1)(G abar)_k), and assert the
    extracted diffusion column satisfies A[q_k<-h] = V[q_k<-h] — tying the
    ViscousDiffusion extraction back to the first-principles derivation
    (the closed forms themselves are pinned by the m07/m14 goldens).  In 1-D the
    in-plane stress relative to the normal stress is tau_xx - tau_zz = 4 nu u_x
    (Gerbeau-Perthame's 4 mu), and h d_x u = d_x q - (q/h) d_x h gives the minus sign."""
    from zoomy_core.model.models.closures import NewtonianInPlane
    from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
    sm = SystemModel.from_model(SME(
        level=2, dimension=2,
        closures=[NewtonianInPlane(), Newtonian(), NavierSlip(), StressFree()]))
    st = [str(s) for s in sm.state]
    sym = {str(s): s for s in sm.state}
    A = sm.diffusion_matrix
    assert A is not None
    hi = st.index("h")
    hcol = {}
    for i, name in enumerate(st):
        if name.startswith("q"):
            hcol[name] = sp.simplify(sp.sympify(A[i, hi, 0, 0]))
    nu = sm.parameters.nu
    h = sym["h"]
    a = [sym[f"q_{k}"] / h for k in range(3)]
    zeta = sp.Symbol("zeta", real=True)
    N = 2
    leg = Legendre_shifted(level=N + 2)
    phi = [sp.expand(leg.eval(i, zeta)) for i in range(N + 1)]
    # concrete zeta-polys with NO Derivative coefficients -> direct integration
    G = sp.Matrix(N + 1, N + 1, lambda k, j: sp.integrate(
        zeta * sp.diff(phi[j], zeta) * phi[k], (zeta, 0, 1)))
    for k in range(N + 1):
        Ga_k = sum(G[k, j] * a[j] for j in range(N + 1))
        V = -4 * nu * (a[k] + (2 * k + 1) * Ga_k)
        assert hcol[f"q_{k}"] != 0, f"k={k}: h-column identically zero (the bug)"
        assert sp.simplify(hcol[f"q_{k}"] - V) == 0, (
            f"k={k}: A[q_{k}<-h]={hcol[f'q_{k}']} != V={sp.simplify(V)}")


# ═══════════════════════════════════════════════════════════════════════════
# Sigma3D — conservative-sigma identity + shallow recovery
# ═══════════════════════════════════════════════════════════════════════════

def _sigma_mapped_model():
    """3-D mass+momentum -> hydrostatic p -> sigma-map; returns
    (m, h, b, u~, w~)."""
    from zoomy_core import coords as C
    import zoomy_core.derivatives as d
    from zoomy_core.model.derivation import (
        Model as DModel, PDETransformation, Simplify)
    from zoomy_core.model.operations import KinematicBC
    from zoomy_core.model.operations import Integrate as IntegrateZ
    from zoomy_core.model.models.equations import Mass, Momentum, moment_scaling

    t, x, z = C.t, C.x, C.z
    zeta = sp.Symbol("zeta", real=True)
    values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}
    m = DModel(coords=(t, x, z), parameters=values)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    m.declare_state(h)
    m.add_equation("bottom", d.t(b))
    m.add_equation(Mass(m))
    mom = Momentum(m); m.add_equation(mom)
    moment_scaling(m, mom)
    uvel, w, p = mom.uvel, mom.w, mom.p
    m.add_equation("kbc_top", KinematicBC(w=w, u=uvel[0], interface=b + h))
    m.add_equation("kbc_bot", KinematicBC(w=w, u=uvel[0], interface=b))
    mz = m.momentum.z
    mz.apply({d.t(w): 0, d.z(w * w): 0, d.x(uvel[0] * w): 0})
    mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
    mz.apply({p.subs(z, b + h): 0})
    m.momentum.x.apply(mz.solve_for(p)); mz.remove(); m.momentum.x.apply(Simplify())
    m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
    pool = m.momentum.x.expr.atoms(sp.Function)
    ut = next(a.func for a in pool if str(a.func) == r"\tilde{u}")(t, x, zeta)
    wt = next(a.func for a in pool if str(a.func) == r"\tilde{w}")(t, x, zeta)
    return m, h, b, ut, wt, zeta, t, x


def test_sigma3d_conservative_sigma_identity():
    """sigma-mass(xh) == d_t h + d_x(hu) + d_zeta(h omega); sigma-mom(xh) ==
    d_t(hu) + d_x(hu^2 + gh^2/2) + d_zeta(hu omega) + gh d_x b - e_x gh -
    tau~_z/rho; omega(0) = omega(1) = 0 <=> the two kinematic BCs."""
    from zoomy_core.model.operations import Multiply
    m, h, b, ut, wt, zeta, t, x = _sigma_mapped_model()
    g, rho, e_x = m.parameters.g, m.parameters.rho, m.parameters.e_x
    taut = next(a.func for a in m.momentum.x.expr.atoms(sp.Function)
                if str(a.func) == r"\tilde{tau_xz}")(t, x, zeta)
    m.mass.apply(Multiply(h))
    mass_res = sp.expand(m.mass.expr.doit())
    mom_res = sp.expand((h * m.momentum.x.expr).doit())
    zc = b + h * zeta
    homega = wt - sp.Derivative(zc, t) - ut * sp.Derivative(zc, x)   # = h*omega
    cons_mass = sp.expand((sp.Derivative(h, t) + sp.Derivative(h * ut, x)
                           + sp.Derivative(homega, zeta)).doit())
    cons_mom = sp.expand((sp.Derivative(h * ut, t)
                          + sp.Derivative(h * ut**2 + g * h**2 / 2, x)
                          + sp.Derivative(ut * homega, zeta)
                          + g * h * sp.Derivative(b, x) - e_x * g * h
                          - sp.Derivative(taut, zeta) / rho).doit())
    assert sp.simplify(cons_mass - mass_res) == 0
    assert sp.simplify(cons_mom - mom_res) == 0
    assert sp.simplify(homega.subs(zeta, 0)
                       - (wt.subs(zeta, 0) - sp.Derivative(b, t)
                          - ut.subs(zeta, 0) * sp.Derivative(b, x))) == 0
    assert sp.simplify(homega.subs(zeta, 1)
                       - (wt.subs(zeta, 1) - sp.Derivative(b + h, t)
                          - ut.subs(zeta, 1) * sp.Derivative(b + h, x))) == 0


def test_sigma3d_shallow_recovery():
    """Sigma3D keeps the FULL deviatoric stress by default (tau~_xx survives
    as an honest aux); adding ShallowInPlane drops it and recovers the shallow
    momentum EXACTLY (full momentum with tau~_xx -> 0)."""
    from zoomy_core.model.models.sigma3d import Sigma3D
    from zoomy_core.model.models.closures import ShallowInPlane
    from zoomy_core.model.basemodel import clear_derivation_model_cache

    clear_derivation_model_cache()
    full = Sigma3D()                       # default closures -> full stress
    mom_full = sp.sympify(full.derivation.momentum.expr)
    txx = [a for a in mom_full.atoms(sp.Function) if "tau_xx" in str(a.func)]
    assert txx, "full-stress Sigma3D must retain the in-plane stress tau~_xx"
    # The full-stress Sigma3D is deliberately OPEN — tau~_xx is the honest
    # unclosed in-plane stress, which is the thing under test — so the build
    # takes the explicit opt-in rather than the guard's refusal.
    from zoomy_core.systemmodel.system_model import allow_unclosed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with allow_unclosed():
            sm_full = SystemModel.from_model(full)
    assert any("tau_xx" in str(s) for s in sm_full.aux_state)
    assert sm_full._unclosed_closures == ["\\tilde{tau_xx}"]

    clear_derivation_model_cache()
    shallow = Sigma3D(closures=[Newtonian(), NavierSlip(), StressFree(),
                                ShallowInPlane()])
    mom_shallow = sp.sympify(shallow.derivation.momentum.expr)
    assert not [a for a in mom_shallow.atoms(sp.Function)
                if "tau_xx" in str(a.func)], "ShallowInPlane must drop tau~_xx"
    drop = {a: 0 for a in mom_full.atoms(sp.Function) if "tau_xx" in str(a.func)}
    assert sp.simplify(sp.expand((mom_shallow - mom_full.subs(drop)).doit())) == 0
