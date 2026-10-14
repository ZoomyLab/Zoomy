"""Gate 50: HLL's wave speeds without a closed-form spectrum (ANALYSIS 49) are bounds of the
signed extremes of the spectrum: s_L <= min Re lambda(A), s_R >= max Re lambda(A), from the
Gelfand bound of shifted matrices, s_R = min(G(A + a I) - a, a), s_L = max(a - G(a I - A), -a),
a = G(A) (`Numerics._gelfand_extremes`; for any real c, rho(A + c I) >= max Re lambda + c).

- random matrices (gate 49's sets: Gaussian, non-normal triangular, Jordan blocks, nilpotent,
  complex pairs, ill-conditioned V D V^-1; n = 2 .. 10; scaled over 12 decades): the formula
  with the NumPy and the JAX `gelfand_bound` kernel;
- the production expression `HLL.wave_speed_bounds(q, q, aux, aux, p, n)` (lambdified, NumPy
  kernel) of SME(0..3) 1-D / 2-D, VAM(1) (Chorin predictor), ML-SME (2 layers, level 0 / 1) at
  gate 49's random wet / thin / dry cell states and Audusse-thinned face states.
Pass: s_R - max Re lambda >= -1e-12 a and min Re lambda - s_L >= -1e-12 a (a 40-digit
eigensolve rechecks a state below that).  Printed: the overestimates (s_R - max Re lambda) / a,
(min Re lambda - s_L) / a (median / max) and the fan (s_R - s_L) / (2 a) (Rusanov: 1).

    python gate50_davis_bounds.py [--part random|models]
"""
import argparse
import sys

import numpy as np
import sympy as sp

import gate49_wave_speed_bound as G49
from zoomy_core.fvm.riemann_solvers import HLL
from zoomy_core.fvm.userfunctions import gelfand_bound as bound_np
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver import default_wave_speed
from zoomy_jax.fvm.userfunctions import gelfand_bound as bound_jax

J, D, TOL = G49.J, G49.D, 1e-12


def extremes(A, bound):
    """`(s_L, s_R, a)` of the batch `A` by the formula of `_gelfand_extremes` (L = R)."""
    n = A.shape[-1]
    a = np.asarray(bound(J, D, *G49.cols(A)), float)
    I = np.eye(n) * a[:, None, None]
    sR = np.minimum(np.asarray(bound(J, D, *G49.cols(A + I)), float) - a, a)
    sL = np.maximum(a - np.asarray(bound(J, D, *G49.cols(I - A)), float), -a)
    return sL, sR, a


def re_extremes_mp(A):
    import mpmath as mp
    mp.mp.dps = 40
    ev = mp.eig(mp.matrix(A.tolist()))[0]
    return float(min(mp.re(e) for e in ev)), float(max(mp.re(e) for e in ev))


def check(label, A, sLsRa):
    """Rigour and tightness of `(s_L, s_R, a)` against the spectra of the batch `A`."""
    ev = np.linalg.eigvals(A)
    lo, hi = ev.real.min(-1), ev.real.max(-1)
    sL, sR, a = sLsRa
    scale = np.maximum(a, 1e-300)
    bad = np.where((sR - hi < -TOL * scale) | (lo - sL < -TOL * scale))[0]
    for k in bad:                                      # the eigensolve's round-off?
        lo[k], hi[k] = re_extremes_mp(A[k])
    eR, eL = (sR - hi) / scale, (lo - sL) / scale
    ok = bool((eR >= -TOL).all() and (eL >= -TOL).all())
    big = a > 1e-8 * np.abs(A).sum(-1).max(-1)
    if not big.any():                                  # nilpotent: rho = 0
        big = np.ones_like(big)
    fan = ((sR - sL) / (2 * scale))[big]
    print(f"{label:38s} {len(A):6d} n={A.shape[-1]:2d}  s_R over: med {np.median(eR[big]):.2e} "
          f"max {eR[big].max():.2e}  s_L over: med {np.median(eL[big]):.2e} max "
          f"{eL[big].max():.2e}  min {min(eR.min(), eL.min()):.1e}  fan med {np.median(fan):.3f} "
          f"min {fan.min():.3f}  mp {len(bad)}  {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(label=label, m=len(A), ok=ok, over=max(eR[big].max(), eL[big].max()))


def random_part():
    rows, orig = [], G49.check

    def both(label, A, rho=None):                      # gate 49's sets, our check
        A = np.asarray(A, float)
        A = A[np.isfinite(A).all((-1, -2))]
        rows.append(check(label + " (NumPy)", A, extremes(A, bound_np)))
        rows.append(check(label + " (JAX)", A, extremes(A, bound_jax)))
        return rows[-1]
    G49.check = both
    try:
        G49.random_sets(m=2000)
    finally:
        G49.check = orig
    return rows


def production(nsm):
    """`(f(Q, aux, p, n) -> (s_L, s_R), ev)`: `HLL.wave_speed_bounds` of the NSM at one state on
    both sides, lambdified with the NumPy kernel; `ev` gate 49's speed matrix."""
    default_wave_speed(nsm)
    ev, names = G49.speed_matrix(nsm)
    num = HLL(nsm.sm)
    Q, A, P, N = (list(num.variables), list(num.aux_variables), list(num.parameters),
                  list(num.normal))
    sL, sR = num.wave_speed_bounds(Q, Q, A, A, P, N)
    mods = [{"conditional": lambda c, x, y: np.where(c, x, y), "gelfand_bound": bound_np},
            "numpy"]
    f = sp.lambdify([*Q, *A, *P, *N], [sL, sR], modules=mods)
    uav = nsm.operator("update_aux_variables")
    g = (None if uav is None
         else sp.lambdify([*Q, *A, *P], list(sp.flatten(uav.definition)), modules=mods))
    pv = [float(getattr(nsm.sm.parameter_values, str(p))) for p in P]

    def bounds(Qv, nrm):
        m = Qv.shape[1]
        aux = np.zeros((len(A), m))
        if g is not None:
            aux = np.array([np.broadcast_to(np.asarray(v, float), (m,))
                            for v in g(*Qv, *aux, *pv)])
        lo, hi = f(*Qv, *aux, *pv, *nrm)
        return np.broadcast_to(lo, (m,)), np.broadcast_to(hi, (m,))
    return bounds, ev, names


def model_part():
    rows = []
    for label, (nsm, ndim) in G49.models().items():
        bounds, ev, names = production(nsm)
        for kind, Qv in zip(("cells", "faces"), G49.random_states(names, 20000, ndim)):
            nrm = G49.normals(Qv.shape[1], ndim)
            A = ev(Qv, nrm)
            fin = np.isfinite(A).all((-1, -2))
            with np.errstate(all="ignore"):
                sL, sR = bounds(Qv, nrm)
            a = np.asarray(bound_np(J, D, *G49.cols(A)), float)
            rows.append(check(f"{label} {kind}", A[fin], (sL[fin], sR[fin], a[fin])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "random", "models"], default="all")
    args = ap.parse_args()
    rows = (random_part() if args.part != "models" else []) + (
        model_part() if args.part != "random" else [])
    ok = all(r["ok"] for r in rows)
    print(f"{len(rows)} sets, {sum(r['m'] for r in rows)} matrices: max overestimate / a "
          f"{max(r['over'] for r in rows):.3f}; {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
