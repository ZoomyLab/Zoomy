"""Gate 47: the viscous operator of SME / VAM (level >= 1) + Newtonian with stress=Stress.full against
an independent Galerkin projection of the full 3-D Newtonian stress divergence (ANALYSIS 45h).

Reference, built here from the modal profiles only (no derivation machinery): physical
derivatives under z = b + h zeta, d_e f = f_e - (b_e + zeta h_e)/h f_zeta, d_z f = f_zeta / h;
  tau_de = rho nu (d_d u_e + d_e u_d), tau_zz = -2 rho nu div u, s' = tau - tau_zz I,
  tau_dz = rho nu (d_z u_d + d_d w)                         (complete shear),
  p_v = tau_zz - rho nu (div u(eta) - div u(z))             (SME: Marche 2007 eq. (3.11)),
  w: SME the kinematic w(b) - int_b^z div u; VAM the state (r modes + the bed-kinematic top mode).
Momentum row d, mode k (nu-part, RHS sign):
  (2k+1)/rho [<phi_k, h (d_e s'_de - d_d (p_v - tau_zz))> + phi_k(1) tau_dz(eta) - phi_k(0) tau_dz(b)
              - <phi_k', tau_dz>],
  tau_dz(eta) = s'_de(eta) d_e eta (StressFree), tau_dz(b) = s'_de(b) d_e b (NavierSlip, lambda part
  dropped: no nu); in VAM the pressure unknown is -sigma_zz, so the p_v - tau_zz term is absent.
VAM w row, mode k: (2k+1)/rho <phi_k, h d_e (rho nu d_z u_e)> (the vertical row of Marche 2007
(3.5); the pressure unknown is -sigma_zz, so no d_z tau_zz).
SystemModel operator (nu-parts, aux = exact derivatives): S - div F - B grad Q + div(A grad Q).

    python gate47_inplane_projection.py [sme:1:2 sme:2:2 vam:1:2 sme:1:3 vam:1:3 ...]
"""
import sys
import time
import warnings

import sympy as sp

from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.models import SME, VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.models.equations import Stress
from zoomy_core.systemmodel import SystemModel

x, y, zeta = sp.symbols("x y zeta", real=True)
nu, rho = sp.Symbol("nu_r", positive=True), sp.Symbol("rho_r", positive=True)


def build(kind, level, dim):
    """The SystemModel and the unclassified-term count."""
    M = {"sme": SME, "vam": VAM}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = SystemModel.from_model(M(level=level, dimension=dim, stress=Stress.full,
                                      closures=[Newtonian(), NavierSlip(), StressFree()]))
    return sm, len(sm.unclassified_terms)


def fields(sm, X):
    """Generic smooth functions of the horizontal coordinates for every state."""
    return {str(s): sp.Function("F_" + str(s))(*X) for s in sm.state}


def sm_operator(sm, F, X):
    """``{state name: nu-part of the row operator}`` with aux = exact derivatives."""
    nus, rhos = sm.parameters.nu, sm.parameters.rho
    St = [str(s) for s in sm.state]

    def tof(e):
        e = sp.sympify(e)
        return e.subs({a: F[str(a)] for a in e.free_symbols if str(a) in F}).subs({nus: nu, rhos: rho})
    aux = {}
    for e in sm.aux_registry:
        if e.get("kind") != "derivative" or e["target_name"] not in F:
            continue                  # a compound target (ML-VAM's d_x(l_1 h)): no nu part
        der = F[e["target_name"]]
        for v, k in zip(X, e["multi_index"]):
            der = sp.diff(der, v, k)
        aux[e["aux_symbol"]] = der
    nd = len(X)
    A, B = sm.diffusion_matrix, sm.nonconservative_matrix
    out = {}
    for i, name in enumerate(St):
        nupart = lambda e: sum((t for t in sp.Add.make_args(sp.expand(sp.sympify(e))) if t.has(nus)),
                               sp.S.Zero)
        op = tof(nupart(sm.source[i, 0]).subs(aux))
        op -= sum(sp.diff(tof(nupart(sm.flux[i, a]).subs(aux)), X[a]) for a in range(nd))
        op -= sum(tof(nupart(B[i, j, a]).subs(aux)) * sp.diff(F[St[j]], X[a])
                  for j in range(len(St)) for a in range(nd))
        if A is not None:
            op += sum(sp.diff(tof(sp.sympify(A[i, j, a, b]).subs(aux)) * sp.diff(F[St[j]], X[b]), X[a])
                      for j in range(len(St)) for a in range(nd) for b in range(nd))
        out[name] = op
    return out


def reference(kind, level, F, X):
    """``{state name: nu-part}`` of the independent projection (module docstring)."""
    N, nd = level, len(X)
    L = Legendre_shifted(level=N + 2)
    phi = [L.eval(j, zeta) for j in range(N + 2)]
    h, b = F["h"], F["b"]
    qn = (lambda d, j: f"q_{j}") if nd == 1 else (lambda d, j: f"q_{'xy'[d]}_{j}")
    U = [sum(F[qn(d, j)] / h * phi[j] for j in range(N + 1)) for d in range(nd)]
    Dx = lambda f, e: sp.diff(f, X[e]) - (sp.diff(b, X[e]) + zeta * sp.diff(h, X[e])) / h * sp.diff(f, zeta)
    Dz = lambda f: sp.diff(f, zeta) / h
    div = lambda: sum(Dx(U[e], e) for e in range(nd))
    if kind == "sme":
        s_ = sp.Dummy("s")
        W = (sum(U[e].subs(zeta, 0) * sp.diff(b, X[e]) for e in range(nd))
             - sp.integrate((h * div()).subs(zeta, s_), (s_, 0, zeta)))
    else:
        r = [F[f"r_{j}"] for j in range(N + 1)]
        wtop = sp.Symbol("wtop")
        W0 = sum(r[j] / h * phi[j] for j in range(N + 1)) + wtop * phi[N + 1]
        bed = sum(U[e].subs(zeta, 0) * sp.diff(b, X[e]) for e in range(nd))
        W = W0.subs(wtop, sp.solve(W0.subs(zeta, 0) - bed, wtop)[0])
    D = div()
    tzz = -2 * rho * nu * D
    tau = [[rho * nu * (Dx(U[e], d) + Dx(U[d], e)) for e in range(nd)] for d in range(nd)]
    sp_ = [[tau[d][e] - (tzz if d == e else 0) for e in range(nd)] for d in range(nd)]
    tdz = [rho * nu * (Dz(U[d]) + Dx(W, d)) for d in range(nd)]
    eta = b + h
    integ = lambda f: sp.integrate(sp.expand(f), (zeta, 0, 1))
    out = {}
    for d in range(nd):
        top = sum(sp_[d][e].subs(zeta, 1) * sp.diff(eta, X[e]) for e in range(nd))
        bot = sum(sp_[d][e].subs(zeta, 0) * sp.diff(b, X[e]) for e in range(nd))
        vol = sum(Dx(sp_[d][e], e) for e in range(nd))
        if kind == "sme":                     # -(d_d p_v) + d_d tau_zz = d_d (rho nu (div u(eta) - div u))
            vol += Dx(rho * nu * (D.subs(zeta, 1) - D), d)
        for k in range(N + 1):
            pk = phi[k]
            val = (integ(pk * h * vol) + pk.subs(zeta, 1) * top - pk.subs(zeta, 0) * bot
                   - integ(sp.diff(pk, zeta) * tdz[d]))
            out[qn(d, k)] = (2 * k + 1) * val / rho
    if kind == "vam":
        for k in range(N + 1):
            vz = sum(Dx(rho * nu * Dz(U[e]), e) for e in range(nd))
            out[f"r_{k}"] = (2 * k + 1) * integ(phi[k] * h * vz) / rho
    return out


def check(kind, level, dim, flat=False):
    """``(unclassified, {row: simplified difference})``."""
    sm, ncl = build(kind, level, dim)
    X = (x,) if dim == 2 else (x, y)
    F = fields(sm, X)
    if flat:
        F["b"] = sp.S.Zero
    op, ref = sm_operator(sm, F, X), reference(kind, level, F, X)
    return ncl, {k: sp.simplify(sp.expand((op[k] - ref[k]).doit())) for k in ref}


def main(specs):
    for spec in specs:
        kind, lvl, dim = spec.split(":")
        t0 = time.time()
        ncl, diff = check(kind, int(lvl), int(dim))
        bad = {k: v for k, v in diff.items() if v != 0}
        print(f"gate47 {kind.upper()}({lvl}) dim {dim}: unclassified {ncl}; rows "
              f"{', '.join(diff)}: SM - reference "
              + ("0" if not bad else f"NONZERO in {list(bad)}") + f"  [{time.time() - t0:.0f} s]",
              flush=True)
        for k, v in bad.items():
            print(f"   {k}: {sp.count_ops(v)} ops: {str(v)[:600]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["sme:1:2", "sme:2:2", "vam:1:2", "sme:1:3", "vam:1:3"])
