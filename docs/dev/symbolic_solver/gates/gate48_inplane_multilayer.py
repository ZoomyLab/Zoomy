"""Gate 48: the viscous operator of ML-SME / ML-VAM (2 layers) + Newtonian with stress=Stress.full
against an independent per-layer Galerkin projection of the full 3-D Newtonian stress, and the
layer sum against the single-layer model (ANALYSIS 45h).

Per layer l (z = z_l-1 + h_l zeta, h_l = l_l h; physical derivatives as gate 47), RHS sign:
  (2k+1)/rho [<phi_k, h_l (d_e s'_de - d_d Pi)> + phi_k(1) tau_l(top) - phi_k(0) tau_l(bot)
              - <phi_k', tau_dz>],
  s' = tau - tau_zz I, tau_dz = rho nu (d_z u_d + d_d w) (complete);
  w: the layer's kinematic w from its bottom, w(z_l-1) = u(z_l-1).grad z_l-1 - sum_(m<l) div q_m0
     (ML-SME), the state with that bottom value closing the top mode (ML-VAM);
  Pi (ML-SME): the vertical balance d_z Pi = d_e (rho nu d_z u_e) integrated pointwise from the
     surface (Pi = 0) through the layers above (continuous at the interface);
  traction: surface s'.grad eta, bed s'.grad b (Navier-slip part has no nu), internal interface
     z_1: tau_l = s'_l.grad z_1 + T*, T* = tau* - s*.grad z_1, (.)* the mean of the two traces.
ML-VAM w rows: (2k+1)/rho <phi_k, h_l d_e (rho nu d_z u_e)>.
Layer sum: with identical layer states (the single-layer profile restricted to each layer) the sum
of the mode-0 rows equals the single-layer SME / VAM row; with different layer states the sum is
sum_l d_e(h_l <s'_l> - delta h_l <Pi_l>) - Pi(b) d_d b (interface terms cancel).

ML-VAM w rows, `vertical_traction` (spec suffix `ws`, ANALYSIS 45j): `one_sided` = the pointwise
projection above; `shared` adds (2k+1)/rho [-phi_k(1) (f* - f_l(z_1^-)).grad z_1 (lower layer),
+phi_k(0) (f* - f_l(z_1^+)).grad z_1 (upper)], f = rho nu d_z u, f* the mean of the two traces.
Checked: rows = reference, identical layer states give shared = one_sided, and the mode-0 layer
sum of shared is the telescoped d_e sum_l h_l <f_l> - f(eta).grad eta + f(b).grad b, while
one_sided keeps -(f(z_1^-) - f(z_1^+)).grad z_1.

    python gate48_inplane_multilayer.py [mlsme:0:2 mlsme:1:2 mlvam:0:2 mlvam:1:2 mlsme:0:3 ...
                                         mlvam:1:2:ws ...]
"""
import sys
import time
import warnings

import sympy as sp

import gate47_inplane_projection as G47
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.models import SME, VAM
from zoomy_core.model.models.ml_sme import MLSME
from zoomy_core.model.models.ml_vam import MLVAM, VerticalTraction
from zoomy_core.model.models.closures import (Newtonian, NavierSlip, StressFree, UpwindInterface,
                                              InterfaceQuantity)
from zoomy_core.model.models.equations import Stress
from zoomy_core.systemmodel import SystemModel

x, y, zeta = G47.x, G47.y, G47.zeta
nu, rho = G47.nu, G47.rho
NL = 2


def build(kind, level, dim, upwind=False):
    """The ML SystemModel and its unclassified-term count."""
    M = {"mlsme": MLSME, "mlvam": MLVAM}[kind]
    clos = [Newtonian(), NavierSlip(), StressFree()]
    if upwind:
        clos.append(UpwindInterface(quantity=InterfaceQuantity.stress))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the pointwise per-layer reference is the one-sided trace (the default is shared, 63)
        vt = dict(vertical_traction=VerticalTraction.one_sided) if kind == "mlvam" else {}
        sm = SystemModel.from_model(M(n_layers=NL, level=level, dimension=dim,
                                      stress=Stress.full, closures=clos, **vt))
    return sm, len(sm.unclassified_terms)


def names(kind, nd):
    """Row-name makers ``q(l, d, k)``, ``r(l, k)``."""
    q = (lambda l, d, k: f"q_{l}_{k}") if nd == 1 else (lambda l, d, k: f"q_{'xy'[d]}_{l}_{k}")
    return q, (lambda l, k: f"r_{l}_{k}")


def layer_fields(kind, N, F, X, l1):
    """Per layer: u profiles, w profile, h_l, base, physical derivatives (lists over layers)."""
    nd = len(X)
    qn, rn = names(kind, nd)
    L = Legendre_shifted(level=N + 2)
    phi = [L.eval(j, zeta) for j in range(N + 2)]
    h, b = F["h"], F["b"]
    ls = [l1, 1 - l1]
    out = []
    for li in range(NL):
        ell = li + 1
        hl = ls[li] * h
        z0 = b + sum(ls[:li]) * h
        U = [sum(F[qn(ell, d, k)] / hl * phi[k] for k in range(N + 1)) for d in range(nd)]
        Dx = (lambda hl_, z0_: lambda f, e: sp.diff(f, X[e]) - (sp.diff(z0_, X[e]) + zeta
                                                                 * sp.diff(hl_, X[e])) / hl_
              * sp.diff(f, zeta))(hl, z0)
        Dz = (lambda hl_: lambda f: sp.diff(f, zeta) / hl_)(hl)
        div = sum(Dx(U[e], e) for e in range(nd))
        W0 = (sum(U[e].subs(zeta, 0) * sp.diff(z0, X[e]) for e in range(nd))
              - sum(sp.diff(F[qn(m + 1, e, 0)], X[e]) for m in range(li) for e in range(nd)))
        if kind == "mlsme":
            s_ = sp.Dummy("s")
            W = W0 - sp.integrate((hl * div).subs(zeta, s_), (s_, 0, zeta))
        else:
            wt = sp.Symbol("wtop")
            Wr = sum(F[rn(ell, j)] / hl * phi[j] for j in range(N + 1)) + wt * phi[N + 1]
            W = Wr.subs(wt, sp.solve(Wr.subs(zeta, 0) - W0, wt)[0])
        out.append(dict(U=U, W=W, h=hl, z0=z0, Dx=Dx, Dz=Dz, D=div, phi=phi))
    return out


def reference(kind, N, F, X, l1):
    """``{row: nu-part}`` of the per-layer projection (module docstring), mean interface."""
    nd = len(X)
    qn, rn = names(kind, nd)
    Ls = layer_fields(kind, N, F, X, l1)
    integ = lambda f: sp.integrate(sp.expand(f), (zeta, 0, 1))
    for Lr in Ls:
        Dx, Dz, U, W = Lr["Dx"], Lr["Dz"], Lr["U"], Lr["W"]
        tzz = -2 * rho * nu * Lr["D"]
        tau = [[rho * nu * (Dx(U[e], d) + Dx(U[d], e)) for e in range(nd)] for d in range(nd)]
        Lr["s"] = [[tau[d][e] - (tzz if d == e else 0) for e in range(nd)] for d in range(nd)]
        Lr["tdz"] = [rho * nu * (Dz(U[d]) + Dx(W, d)) for d in range(nd)]
    # viscous pressure Pi (ML-SME): pointwise from the surface, continuous at z_1
    if kind == "mlsme":
        top, bot = Ls[1], Ls[0]
        top["Pi"] = -rho * nu * (top["D"].subs(zeta, 1) - top["D"])
        bot["Pi"] = top["Pi"].subs(zeta, 0) - rho * nu * (bot["D"].subs(zeta, 1) - bot["D"])
    else:
        for Lr in Ls:
            Lr["Pi"] = sp.S.Zero
    z1, eta, b = Ls[1]["z0"], F["b"] + F["h"], F["b"]
    grad = lambda f: [sp.diff(f, X[e]) for e in range(nd)]
    ss = [[(Ls[0]["s"][d][e].subs(zeta, 1) + Ls[1]["s"][d][e].subs(zeta, 0)) / 2
           for e in range(nd)] for d in range(nd)]
    ts = [(Ls[0]["tdz"][d].subs(zeta, 1) + Ls[1]["tdz"][d].subs(zeta, 0)) / 2 for d in range(nd)]
    Tstar = [ts[d] - sum(ss[d][e] * grad(z1)[e] for e in range(nd)) for d in range(nd)]
    out = {}
    for li, Lr in enumerate(Ls):
        ell = li + 1
        phi, Dx = Lr["phi"], Lr["Dx"]
        for d in range(nd):
            s_top = (grad(eta) if li == NL - 1 else grad(z1))
            s_bot = (grad(b) if li == 0 else grad(z1))
            ttop = sum(Lr["s"][d][e].subs(zeta, 1) * s_top[e] for e in range(nd)) + (
                Tstar[d] if li < NL - 1 else 0)
            tbot = sum(Lr["s"][d][e].subs(zeta, 0) * s_bot[e] for e in range(nd)) + (
                Tstar[d] if li > 0 else 0)
            vol = sum(Dx(Lr["s"][d][e], e) for e in range(nd)) - Dx(Lr["Pi"], d)
            for k in range(N + 1):
                pk = phi[k]
                val = (integ(pk * Lr["h"] * vol) + pk.subs(zeta, 1) * ttop
                       - pk.subs(zeta, 0) * tbot - integ(sp.diff(pk, zeta) * Lr["tdz"][d]))
                out[qn(ell, d, k)] = (2 * k + 1) * val / rho
        if kind == "mlvam":
            vz = sum(Dx(rho * nu * Lr["Dz"](Lr["U"][e]), e) for e in range(nd))
            for k in range(N + 1):
                out[rn(ell, k)] = (2 * k + 1) * integ(phi[k] * Lr["h"] * vz) / rho
    return out


def restrict(kind, N, Fs, X, l1):
    """Layer states that restrict the single-layer profile ``Fs`` (q_k, h, b, r_k) to each layer."""
    nd = len(X)
    qn, rn = names(kind, nd)
    L = Legendre_shifted(level=N + 2)
    phi = [L.eval(j, zeta) for j in range(N + 2)]
    h, b = Fs["h"], Fs["b"]
    qs = (lambda d, k: f"q_{k}") if nd == 1 else (lambda d, k: f"q_{'xy'[d]}_{k}")
    U = [sum(Fs[qs(d, k)] / h * phi[k] for k in range(N + 1)) for d in range(nd)]
    out = {"h": h, "b": b}
    ls, c0 = [l1, 1 - l1], [0, l1]
    if kind == "mlvam":
        s_ = sp.Dummy("s")
        div = sum(sp.diff(U[e], X[e]) - (sp.diff(b, X[e]) + zeta * sp.diff(h, X[e])) / h
                  * sp.diff(U[e], zeta) for e in range(nd))
        Wk = (sum(U[e].subs(zeta, 0) * sp.diff(b, X[e]) for e in range(nd))
              - sp.integrate((h * div).subs(zeta, s_), (s_, 0, zeta)))
    for li in range(NL):
        loc = lambda f: f.subs(zeta, c0[li] + ls[li] * zeta)
        for k in range(N + 1):
            for d in range(nd):
                out[qn(li + 1, d, k)] = sp.simplify((2 * k + 1) * ls[li] * h * sp.integrate(
                    sp.expand(loc(U[d]) * phi[k]), (zeta, 0, 1)))
            if kind == "mlvam":
                out[rn(li + 1, k)] = sp.simplify((2 * k + 1) * ls[li] * h * sp.integrate(
                    sp.expand(loc(Wk) * phi[k]), (zeta, 0, 1)))
    return out


def check(kind, level, dim, upwind=False):
    """``(unclassified, per-row differences, layer-sum difference)``."""
    sm, ncl = build(kind, level, dim, upwind)
    X = (x,) if dim == 2 else (x, y)
    nd = len(X)
    F = G47.fields(sm, X)
    l1 = sm.parameters.l_1
    op = G47.sm_operator(sm, F, X)
    rows = {}
    if not upwind:
        ref = reference(kind, level, F, X, l1)
        rows = {k: sp.simplify(sp.expand((op[k] - ref[k]).doit())) for k in ref}
    # layer sum with identical states = the single-layer model
    single = {"mlsme": SME, "mlvam": VAM}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm1 = SystemModel.from_model(single(level=level, dimension=dim, stress=Stress.full,
                                            closures=[Newtonian(), NavierSlip(), StressFree()]))
    Fs = G47.fields(sm1, X)
    if kind == "mlvam":                           # the single-layer w: the kinematic one
        rs = restrict(kind, level, Fs, X, sp.Rational(1, 1))
        Fs.update({f"r_{k}": rs[f"r_1_{k}"] for k in range(level + 1)})
    op1 = G47.sm_operator(sm1, Fs, X)
    lay = restrict(kind, level, Fs, X, l1)
    qn, _ = names(kind, nd)
    qs = (lambda d: "q_0") if nd == 1 else (lambda d: f"q_{'xy'[d]}_0")
    sums = {}
    for d in range(nd):
        tot = sum(op[qn(l + 1, d, 0)] for l in range(NL))
        tot = tot.subs({F[k]: v for k, v in lay.items()}).doit()
        sums[qs(d)] = sp.simplify(sp.expand(tot - op1[qs(d)]))
    return ncl, rows, sums


def check_vertical(level, dim):
    """`vertical_traction` one_sided / shared (module docstring): ``[(label, ok)]``."""
    X = (x,) if dim == 2 else (x, y)
    nd = len(X)
    sms = {}
    for vt in (VerticalTraction.one_sided, VerticalTraction.shared):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sms[vt] = SystemModel.from_model(MLVAM(
                n_layers=NL, level=level, dimension=dim, stress=Stress.full, vertical_traction=vt,
                closures=[Newtonian(), NavierSlip(), StressFree()]))
    sm = sms[VerticalTraction.shared]
    F = G47.fields(sm, X)
    l1 = sm.parameters.l_1
    ops = {vt: G47.sm_operator(s_, F, X) for vt, s_ in sms.items()}
    ref = reference("mlvam", level, F, X, l1)
    Ls = layer_fields("mlvam", level, F, X, l1)
    f = [[rho * nu * Ls[li]["Dz"](Ls[li]["U"][e]) for e in range(nd)] for li in range(NL)]
    fb, fa = [f[0][e].subs(zeta, 1) for e in range(nd)], [f[1][e].subs(zeta, 0) for e in range(nd)]
    gz = [sp.diff(Ls[1]["z0"], X[e]) for e in range(nd)]
    corr = {}
    for li, own, side, sgn in ((0, fb, 1, -1), (1, fa, 0, 1)):
        for k in range(level + 1):
            corr[f"r_{li + 1}_{k}"] = ((2 * k + 1) / rho * sgn * Ls[li]["phi"][k].subs(zeta, side)
                                       * sum(((fb[e] + fa[e]) / 2 - own[e]) * gz[e] for e in range(nd)))
    zero = lambda e: sp.simplify(sp.expand(sp.sympify(e).doit())) == 0
    out = [(f"{vt.value}: w rows = reference",
            all(zero(ops[vt][r] - ref[r] - (corr[r] if vt is VerticalTraction.shared else 0))
                for r in corr)) for vt in (VerticalTraction.one_sided, VerticalTraction.shared)]
    out.append(("horizontal rows: shared = one_sided",
                all(zero(ops[VerticalTraction.shared][r] - ops[VerticalTraction.one_sided][r])
                    for r in ops[VerticalTraction.shared] if r.startswith("q"))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm1 = SystemModel.from_model(VAM(level=level, dimension=dim, stress=Stress.full,
                                         closures=[Newtonian(), NavierSlip(), StressFree()]))
    Fs = G47.fields(sm1, X)
    rs = restrict("mlvam", level, Fs, X, sp.Rational(1, 1))
    Fs.update({f"r_{k}": rs[f"r_1_{k}"] for k in range(level + 1)})
    lay = {F[k]: v for k, v in restrict("mlvam", level, Fs, X, l1).items()}
    out.append(("identical layer states: shared = one_sided",
                all(zero((ops[VerticalTraction.shared][r] - ops[VerticalTraction.one_sided][r])
                         .subs(lay)) for r in corr)))
    integ = lambda g: sp.integrate(sp.expand(g), (zeta, 0, 1))
    eta, b = F["b"] + F["h"], F["b"]
    tele = (sum(sp.diff(sum(Ls[li]["h"] * integ(f[li][e]) for li in range(NL)), X[e])
                - f[1][e].subs(zeta, 1) * sp.diff(eta, X[e]) + f[0][e].subs(zeta, 0) * sp.diff(b, X[e])
                for e in range(nd)) / rho)
    jump = sum((fb[e] - fa[e]) * gz[e] for e in range(nd)) / rho
    s0 = {vt: ops[vt]["r_1_0"] + ops[vt]["r_2_0"] for vt in (VerticalTraction.one_sided, VerticalTraction.shared)}
    out.append(("mode-0 layer sum: shared telescopes", zero(s0[VerticalTraction.shared] - tele)))
    out.append(("mode-0 layer sum: one_sided = telescoped - [f].grad z_1 / rho",
                zero(s0[VerticalTraction.one_sided] - tele + jump)))
    return out


def main(specs):
    for spec in specs:
        kind, lvl, dim, *opt = spec.split(":")
        up = "up" in opt
        if "ws" in opt:
            t0 = time.time()
            res = check_vertical(int(lvl), int(dim))
            print(f"gate48 MLVAM({lvl}) dim {dim} vertical_traction: "
                  + "; ".join(f"{lab}: {'ok' if ok else 'FAIL'}" for lab, ok in res)
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
            continue
        t0 = time.time()
        ncl, rows, sums = check(kind, int(lvl), int(dim), up)
        bad = {k: v for k, v in rows.items() if v != 0}
        bads = {k: v for k, v in sums.items() if v != 0}
        print(f"gate48 {kind.upper()}({lvl}) dim {dim}{' upwind' if up else ''}: unclassified {ncl}; "
              + (f"rows {', '.join(rows)}: SM - per-layer reference "
                 + ("0" if not bad else f"NONZERO in {list(bad)}") + "; " if rows else "")
              + "layer sum (identical states) - single layer: "
              + ("0" if not bads else f"NONZERO in {list(bads)}") + f"  [{time.time() - t0:.0f} s]",
              flush=True)
        for k, v in {**bad, **bads}.items():
            print(f"   {k}: {sp.count_ops(v)} ops: {str(v)[:500]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["mlsme:0:2", "mlsme:1:2", "mlvam:0:2", "mlvam:1:2"])
