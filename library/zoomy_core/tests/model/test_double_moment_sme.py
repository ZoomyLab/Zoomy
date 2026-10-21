"""Double-moment SME — core port of thesis/notebooks/derivation/double_moment_sme.py.

Pins REQ-141: the validated ``TensorSeparationOfVariables`` now lives in
``zoomy_core.model.derivation`` and the double-moment pipeline is driven with it
straight from core.  The three notebook verification cells are ported ROW-FOR-ROW:

* ``verify_full``  — Nu=1, Ny=1 closed rows vs the hand-derived double-moment rows;
* ``verify_ny0``   — the Ny=0 reduction vs the reference core ``SME(level=1)`` rows;
* ``verify_nu0``   — the Nu=0 reduction vs the width-only hand rows.

Also exercises REQ-142: the pipeline is wrapped in ``@derivation_cache`` and a
second identical invocation is asserted to be a cache HIT that does NOT re-run the
(minutes-long) symbolic body.

These are heavy symbolic derivations (Nu=1,Ny=1 ≈ a few minutes) — marked ``slow``.
"""
import time

import pytest
import sympy as sp

# R3 (approved spec §1c): sole TensorSeparationOfVariables + disk-persist
# coverage; every test below is individually @pytest.mark.rederive already.
pytestmark = [pytest.mark.model]

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model, PDETransformation, Simplify, ResolveIntegral, Basis,
    ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, test_index as _test_index,
    TensorSeparationOfVariables)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.derivation.derivation_cache import (
    derivation_cache, clear_derivation_cache)
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)
eta = sp.Symbol("eta", real=True)


# ── the derivation pipeline (ported verbatim, core TensorSeparationOfVariables) ─

def derive_dm(Nu=1, Ny=1):
    m = Model(coords=(t, x, y, z),
              parameters={"g": 9.81, "rho": 1.0, "nu": 0.0,
                          "lambda_s": 0.0, "lambda_w": 0.0, "W": 1.0})
    g, rho = m.parameters.g, m.parameters.rho
    nu, lam_s, lam_w, W = (m.parameters.nu, m.parameters.lambda_s,
                           m.parameters.lambda_w, m.parameters.W)
    u = sp.Function("u", real=True)(t, x, y, z)
    w = sp.Function("w", real=True)(t, x, y, z)
    p = sp.Function("p", real=True)(t, x, y, z)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    txz = sp.Function("tau_xz", real=True)(t, x, y, z)
    txy = sp.Function("tau_xy", real=True)(t, x, y, z)

    # 1 — balance laws (v=0) + hydrostatic elimination
    m.Q = [h, u, w, p]
    m.add_equation("bottom", d.t(b))
    m.add_equation("mass", d.x(u) + d.z(w))
    m.add_equation("momentum_x", d.t(u) + d.x(u*u) + d.z(u*w) + d.x(p)/rho
                                 - d.z(txz)/rho - d.y(txy)/rho)
    m.add_equation("momentum_z", d.t(w) + d.x(u*w) + d.z(w*w) + d.z(p)/rho + g)
    m.add_equation("kbc_top", KinematicBC(interface=b + h, w=w, u=u))
    m.add_equation("kbc_bot", KinematicBC(interface=b, w=w, u=u))
    mz = m.momentum_z
    mz.apply({d.t(w): 0, d.z(w*w): 0, d.x(u*w): 0})
    mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
    mz.apply({p.subs(z, b + h): 0})
    m.momentum_x.apply(mz.solve_for(p)); mz.remove(); m.momentum_x.apply(Simplify())

    # 2 — vertical σ-map + projection (test φ_k) + stress closures
    m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h*zeta))}))
    bz = Basis(symbol="phi", weight="c"); cz = bz.weight
    k = _test_index(); phi_k = bz.phi(k, zeta)
    leg_z = Legendre_shifted(level=Nu + 2)
    for nm in ("mass", "momentum_x"):
        eq = getattr(m, nm)
        eq.apply(Multiply(h)); eq.apply(Multiply(cz(zeta)*phi_k))
        eq.apply(ProductRule(variables=[zeta])); eq.apply(Integrate(zeta, bounds=(0, 1)))
        eq.apply(ResolveIntegral())
        eq.apply(m.kbc_bot); eq.apply(m.kbc_top); eq.apply({sp.Derivative(b, t): 0})
    tau_z, uu = m.functions.tau_xz, m.functions.u
    m.momentum_x.apply({tau_z.at(1): 0, tau_z.at(0): lam_s*uu.at(0)})
    thz, uhd = tau_z.expr.func, uu.expr.func
    m.momentum_x.apply({a: rho*nu/h*sp.Derivative(uhd(*a.args), a.args[-1])
                        for a in m.momentum_x.expr.atoms(sp.Function) if a.func == thz})
    thy = m.functions.tau_xy.expr.func
    m.momentum_x.apply({a: rho*nu*sp.Derivative(uhd(*a.args), a.args[2])
                        for a in m.momentum_x.expr.atoms(sp.Function) if a.func == thy})
    m.momentum_x.apply(Simplify())

    # 3 — lateral σ-map + projection (test ψ_l2) + wall Robin closure
    m.apply(PDETransformation({y: (eta, sp.Eq(y, W*eta))}))
    by = Basis(symbol="varpsi", weight="e"); cy = by.weight
    l = _test_index("l_2"); psi_l = by.phi(l, eta)
    leg_y = Legendre_shifted(level=Ny + 2)
    for nm in ("mass", "momentum_x"):
        eq = getattr(m, nm)
        eq.apply(Multiply(W)); eq.apply(Multiply(cy(eta)*psi_l))
        eq.apply(ProductRule(variables=[eta])); eq.apply(Integrate(eta, bounds=(0, 1)))
        eq.apply(ResolveIntegral())

    def wall_subs(expr):   # Subs(∂_η F, η, w) → ±λ_w W/(ρν)·F|_{η=w}
        subs = {}
        for a in expr.atoms(sp.Subs):
            inner = a.expr
            if (isinstance(inner, sp.Derivative) and a.point
                    and a.point[0] in (sp.S.Zero, sp.S.One)
                    and any(str(v) == "eta" for v in a.variables)):
                sgn = 1 if a.point[0] == 0 else -1
                subs[a] = sgn*lam_w*W/(rho*nu)*inner.expr.subs(a.variables[0], a.point[0])
        return subs
    ws = wall_subs(m.momentum_x.expr)
    if ws:
        m.momentum_x.apply(ws)

    # 4 — ONE tensor-product separation of variables (u and w) — CORE op
    qh = sp.Function(r"\hat{q}", real=True)
    wq = sp.Function(r"\hat{\omega}", real=True)
    m.apply(TensorSeparationOfVariables(u, qh, pos_z=3, basis_z=bz, order_z=Nu,
                                        pos_y=2, basis_y=by, order_y=Ny, scale_y=W))
    m.apply(TensorSeparationOfVariables(w, wq, pos_z=3, basis_z=bz, order_z=Nu + 1,
                                        pos_y=2, basis_y=by, order_y=Ny, scale_y=W))

    # 5 — resolution: lateral (outer) first, then vertical, basis PER ROW
    for nm in ("mass", "momentum_x"):
        eq = getattr(m, nm)
        eq.apply(ExpandSums()); eq.apply(PullConstants())
        eq.apply(ExtractBrackets(by, var=eta)); eq.apply(EvaluateSums())
    m.mass.apply(ResolveModes(index=l, modes=range(Ny + 1)))
    m.momentum_x.apply(ResolveModes(index=l, modes=range(Ny + 1)))
    lat_rows = ([f"mass_{ll}" for ll in range(Ny + 1)]
                + [f"momentum_x_{ll}" for ll in range(Ny + 1)])
    for nm in lat_rows:
        getattr(m, nm).apply(ResolveBasis(leg_y, var=eta))
    for nm in lat_rows:
        r = getattr(m, nm)
        r.apply(ExpandSums()); r.apply(PullConstants())
        r.apply(ExtractBrackets(bz, var=zeta)); r.apply(EvaluateSums())
        modes = range(Nu + 3) if nm.startswith("mass") else range(Nu + 1)
        getattr(m, nm).apply(ResolveModes(index=k, modes=modes))

    def _row_names():
        for ll in range(Ny + 1):
            for kk in range(Nu + 3):
                if hasattr(m, f"mass_{ll}_{kk}"):
                    yield f"mass_{ll}_{kk}"
            for kk in range(Nu + 1):
                if hasattr(m, f"momentum_x_{ll}_{kk}"):
                    yield f"momentum_x_{ll}_{kk}"
    for nm in _row_names():
        getattr(m, nm).apply(ResolveBasis(leg_z, var=zeta))
    leftover = {nm for nm in _row_names()
                if any(f.func.__name__ in ("Gram", "Weight", "phi", "varpsi")
                       for f in getattr(m, nm).expr.atoms(sp.Function))}
    assert not leftover, f"unresolved opaque atoms in rows: {leftover}"
    rows = {nm: getattr(m, nm).expr for nm in _row_names()}
    return m, rows


def close_system(rows, Nu, Ny):
    _allf = {f.func.__name__: f.func
             for e in rows.values() for f in e.doit().atoms(sp.Function)}
    wq, qh = _allf[r"\hat{\omega}"], _allf[r"\hat{q}"]

    m00 = sp.expand(rows["mass_0_0"].doit())
    h = next(f for f in m00.atoms(sp.Function) if f.func.__name__ == "h")
    dth = sp.Derivative(h, t)
    h_rhs = sp.solve(m00, dth)[0]

    nclose = {}
    for ll in range(Ny + 1):
        eqs = [sp.expand(rows[f"mass_{ll}_{kk}"].doit().subs(dth, h_rhs))
               for kk in range(1, Nu + 3)]
        unks = [wq(ll, ji, t, x) for ji in range(Nu + 2)]
        A, rhs = sp.linear_eq_to_matrix(eqs, unks)
        sol = A.solve_least_squares(rhs) if A.rows != A.cols else A.LUsolve(rhs)
        for u_, s_ in zip(unks, sol):
            nclose[u_] = sp.simplify(s_)

    qnew, mom_rows = {}, {}
    for kk in range(Nu + 1):
        for ll in range(Ny + 1):
            e = sp.expand(rows[f"momentum_x_{ll}_{kk}"].doit())
            mom_rows[(kk, ll)] = sp.expand(e.subs(nclose).subs(dth, h_rhs))
    cov = {}
    for kk in range(Nu + 1):
        for ll in range(Ny + 1):
            qf = sp.Function(f"q_{kk}{ll}", real=True)(t, x)
            cov[qh(ll, kk, t, x)] = qf / h
            qnew[(kk, ll)] = qf
    for key in mom_rows:
        mom_rows[key] = sp.expand(mom_rows[key].subs(cov).doit())
    h_rhs_q = sp.expand(h_rhs.subs(cov).doit())

    closed = {"h": h_rhs_q}
    for (kk, ll), e in mom_rows.items():
        closed[f"q_{kk}{ll}"] = e
    return closed, mom_rows, qnew, h


# ── REQ-142: cache the notebook pipeline (keyed on op sequence source + orders) ─

@derivation_cache
def derive_dm_cached(Nu, Ny):
    return derive_dm(Nu, Ny)


_close_calls = {"n": 0}


@derivation_cache(persist=True)
def close_dm_cached(Nu, Ny):
    """Fully-resolved closed rows — an ordinary-sympy dict that round-trips to
    disk (unlike the intermediate Model), so it can be cached ACROSS sessions."""
    _close_calls["n"] += 1
    _, rows = derive_dm(Nu, Ny)
    closed, *_ = close_system(rows, Nu, Ny)
    return closed


# ── verification helpers (ported from the notebook cells) ──────────────────────

def _harvest(closed):
    closed = {k: v.doit() for k, v in closed.items()}
    allsyms, allfuncs = {}, {}
    for e in closed.values():
        for s in e.free_symbols:
            allsyms[str(s)] = s
        for f in e.atoms(sp.Function):
            allfuncs[str(f.func)] = f
    return closed, allsyms, allfuncs


def verify_full(closed):
    closed, allsyms, allfuncs = _harvest(closed)
    tt, xx = allsyms["t"], allsyms["x"]
    h = allfuncs["h"]
    P = lambda n: allsyms.get(n, sp.S.Zero)
    g, rho, nu, lam_s, lam_w, W = [P(n) for n in
                                   ("g", "rho", "nu", "lambda_s", "lambda_w", "W")]
    Dx = lambda e: sp.Derivative(e, xx)
    dth = sp.Derivative(h, tt)
    q = {nm[2:]: f for nm, f in allfuncs.items() if nm.startswith("q_")}

    def solved(nm):
        dtq = sp.Derivative(allfuncs[nm], tt)
        return sp.solve(closed[nm], dtq)[0].subs(dth, -Dx(allfuncs["q_00"]))

    hand = {
      "q_00": -sp.expand(Dx(q["00"]**2/h + q["01"]**2/(3*h) + q["10"]**2/(3*h)
                            + q["11"]**2/(9*h) + g*h**2/2).doit())
              - g*h*Dx(allfuncs["b"]) - lam_s*(q["00"]-q["10"])/(rho*h)
              - 2*lam_w*q["00"]/(rho*W),
      "q_01": -sp.expand(Dx(2*q["00"]*q["01"]/h + 2*q["10"]*q["11"]/(3*h)).doit())
              - lam_s*(q["01"]-q["11"])/(rho*h) - 6*lam_w*q["01"]/(rho*W)
              - 12*nu*q["01"]/W**2}
    results = {}
    for nm, rhs_hand in hand.items():
        d_ = sp.simplify(sp.expand(solved(nm)) - sp.expand(rhs_hand))
        results[nm] = d_
    return results


def verify_ny0(closed):
    """Ny=0 reduction vs the reference CORE ``SME(level=1, dimension=2)`` rows."""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    from zoomy_core.systemmodel.system_model import SystemModel

    closed, allsyms, allfuncs = _harvest(closed)
    tt, xx = allsyms["t"], allsyms["x"]
    h = allfuncs["h"]
    lam_w = allsyms.get("lambda_w", sp.S.Zero)
    Dx = lambda e: sp.Derivative(e, xx)
    dth = sp.Derivative(h, tt)

    def solved(nm):
        dtq = sp.Derivative(allfuncs[nm], tt)
        s = sp.solve(closed[nm], dtq)[0].subs(dth, -Dx(allfuncs["q_00"]))
        return s.subs(lam_w, 0) if lam_w != 0 else s

    mref = SME(level=1, dimension=2,
               closures=[Newtonian(), NavierSlip(), StressFree()])
    smref = SystemModel.from_model(mref)
    Rref = {str(s): sp.sympify(r).doit()
            for s, r in zip(smref.state, smref.reconstruct_residuals())}

    results = {}
    for nm, ref_nm in (("q_00", "q_0"), ("q_10", "q_1")):
        ref = Rref[ref_nm]
        ren = {}
        for f in ref.atoms(sp.Function):
            tgt = {"q_0": "q_00", "q_1": "q_10"}.get(str(f.func), str(f.func))
            if tgt in allfuncs:
                ren[f] = allfuncs[tgt]
        for s in ref.free_symbols:
            if str(s) == "e_x":                 # no gravity tilt in the DM channel
                ren[s] = sp.S.Zero
            elif str(s) in allsyms:
                ren[s] = allsyms[str(s)]
        refr = sp.expand(ref.xreplace(ren).doit())
        dtq = sp.Derivative(allfuncs[nm], tt)
        refsol = sp.solve(refr, dtq)[0].subs(dth, -Dx(allfuncs["q_00"]))
        d_ = sp.simplify(sp.expand(solved(nm)) - sp.expand(refsol))
        results[(nm, ref_nm)] = d_
    return results


def verify_nu0(closed):
    closed, allsyms, allfuncs = _harvest(closed)
    tt, xx = allsyms["t"], allsyms["x"]
    h = allfuncs["h"]
    P = lambda n: allsyms.get(n, sp.S.Zero)
    g, rho, nu, lam_s, lam_w, W = [P(n) for n in
                                   ("g", "rho", "nu", "lambda_s", "lambda_w", "W")]
    Dx = lambda e: sp.Derivative(e, xx)
    dth = sp.Derivative(h, tt)
    q = {nm[2:]: f for nm, f in allfuncs.items() if nm.startswith("q_")}

    def solved(nm):
        dtq = sp.Derivative(allfuncs[nm], tt)
        return sp.solve(closed[nm], dtq)[0].subs(dth, -Dx(allfuncs["q_00"]))

    hand = {
      "q_00": -sp.expand(Dx(q["00"]**2/h + q["01"]**2/(3*h) + g*h**2/2).doit())
              - g*h*Dx(allfuncs["b"]) - lam_s*q["00"]/(rho*h)
              - 2*lam_w*q["00"]/(rho*W),
      "q_01": -sp.expand(Dx(2*q["00"]*q["01"]/h).doit())
              - lam_s*q["01"]/(rho*h) - 6*lam_w*q["01"]/(rho*W)
              - 12*nu*q["01"]/W**2}
    results = {}
    for nm, rhs_hand in hand.items():
        d_ = sp.simplify(sp.expand(solved(nm)) - sp.expand(rhs_hand))
        results[nm] = d_
    return results


# ── tests ──────────────────────────────────────────────────────────────────

@pytest.mark.rederive
def test_dm_full_matches_hand_rows():
    """(full) Nu=1, Ny=1 closed rows == hand-derived double-moment rows."""
    _, rows = derive_dm(Nu=1, Ny=1)
    closed, *_ = close_system(rows, Nu=1, Ny=1)
    res = verify_full(closed)
    assert all(v == 0 for v in res.values()), \
        {k: sp.sstr(v)[:160] for k, v in res.items() if v != 0}


@pytest.mark.rederive
def test_dm_ny0_reduces_to_reference_sme():
    """(SME) Ny=0 reduction == core SME(level=1) reconstruct rows, row-for-row."""
    _, rows = derive_dm(Nu=1, Ny=0)
    closed, *_ = close_system(rows, Nu=1, Ny=0)
    res = verify_ny0(closed)
    assert all(v == 0 for v in res.values()), \
        {k: sp.sstr(v)[:160] for k, v in res.items() if v != 0}


@pytest.mark.rederive
def test_dm_nu0_reduces_to_width_rows():
    """(width) Nu=0 reduction == width-only hand rows, row-for-row."""
    _, rows = derive_dm(Nu=0, Ny=1)
    closed, *_ = close_system(rows, Nu=0, Ny=1)
    res = verify_nu0(closed)
    assert all(v == 0 for v in res.values()), \
        {k: sp.sstr(v)[:160] for k, v in res.items() if v != 0}


@pytest.mark.rederive
def test_dm_pipeline_second_invocation_is_cache_hit(monkeypatch):
    """REQ-142: a 2nd identical pipeline call is a HIT — the minutes-long
    symbolic body does NOT re-run (asserted on the wrapped-body call count)."""
    # cache tests enable the cache explicitly; the ambient default is OFF
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "1")
    clear_derivation_cache(derive_dm_cached)
    derive_dm_cached.stats.hits = 0
    derive_dm_cached.stats.misses = 0

    t0 = time.time()
    r1 = derive_dm_cached(1, 1)
    cold = time.time() - t0
    assert derive_dm_cached.stats.misses == 1

    t0 = time.time()
    r2 = derive_dm_cached(1, 1)
    warm = time.time() - t0
    assert derive_dm_cached.stats.hits == 1, "2nd identical call must hit cache"
    assert r1 is r2, "in-memory hit returns the same object"
    # a genuine hit skips the whole build: warm is a tiny fraction of cold.
    assert warm < 0.05 * cold + 0.01, f"warm={warm:.3f}s not << cold={cold:.1f}s"


@pytest.mark.rederive
def test_dm_pipeline_persists_across_sessions(tmp_path, monkeypatch):
    """REQ-142: the closed rows persist to disk, so a FRESH session (in-memory
    store cleared) re-loads them from disk WITHOUT re-running the symbolic body."""
    from zoomy_core.model.derivation import derivation_cache as dc
    monkeypatch.setenv("ZOOMY_CACHE_DIR", str(tmp_path))
    # cache tests enable the cache explicitly; the ambient default is OFF
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "1")
    clear_derivation_cache(close_dm_cached)
    _close_calls["n"] = 0
    close_dm_cached.stats.hits = 0

    c1 = close_dm_cached(1, 1)                 # cold: runs body, writes disk
    assert _close_calls["n"] == 1
    assert any(tmp_path.iterdir()), "cold run must write a disk entry"

    # simulate a fresh process: wipe the in-memory store, keep the disk tier.
    dc._STORE.get(close_dm_cached._cache_id, {}).clear()

    c2 = close_dm_cached(1, 1)                 # warm across "sessions": disk hit
    assert _close_calls["n"] == 1, "disk hit must NOT re-run the symbolic body"
    assert close_dm_cached.stats.hits == 1
    # round-trip fidelity of the closed q_00 row.
    assert sp.srepr(sp.sympify(c1["q_00"])) == sp.srepr(sp.sympify(c2["q_00"]))
