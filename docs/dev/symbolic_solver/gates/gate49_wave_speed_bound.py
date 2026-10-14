"""Gate 49: the default wave speed of a model without a closed-form spectrum, `gelfand_bound`
(ANALYSIS 48), is an upper bound of the spectral radius of the matrix it is given.

Through the production path (`local_max_abs_eigenvalue` -> `gelfand_bound`, the NumPy kernel and
the JAX kernel) on
- random matrices: Gaussian, non-normal triangular, defective (Jordan blocks), nilpotent, complex
  pairs, ill-conditioned diagonalizable, scaled over 12 decades (exact spectra where constructed);
- the wave-speed matrices of SME(0..3) 1-D / 2-D, VAM(1) (Chorin predictor), ML-SME (2 layers,
  level 0 / 1) at random wet, thin and dry cell states and hydrostatically reconstructed
  (thinned) face states;
- the states of given runs (`--states run.npz ...`, `Q` of shape (nq, nc) or (nt, nq, nc) with
  `names` or `state`).
Pass: bound / max|eig| >= 1 - 1e-12 for the NumPy and the JAX kernel wherever
max|eig| > 1e-100 ||A||_inf (states below that are re-checked with a 40-digit eigensolve; below
1e-100 ||A||, i.e. dry states, the squares underflow and the bound may return 0); median and max
over rho > 1e-8 ||A||.  Also printed:
the ratios for 4, 5, 6, 8 squarings, of the plain `||A^m||_inf^(1/m)` (no Collatz-Wielandt step) and
of the former default (power iteration on A^2, 10 steps, x 1.06), which falls below 1.

    python gate49_wave_speed_bound.py [--part random|models] [--states run.npz ...]
"""
import argparse
import os
import sys

import numpy as np
import sympy as sp

import common as C
from zoomy_core.fvm.userfunctions import gelfand_bound as bound_np
from zoomy_core.model.kernel_functions import gelfand_bound
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver import default_wave_speed
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.fvm.userfunctions import gelfand_bound as bound_jax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "matrix"))
J, D = SystemModel.spectral_squarings, SystemModel.spectral_cw_offset
TOL = 1e-12
rng = np.random.default_rng(49)


def cols(A):
    """Row-major entry list of a batch `(m, n, n)`."""
    n = A.shape[-1]
    return [A[:, i, j] for i in range(n) for j in range(n)]


def power_x106(A, k=10):
    """The former default: `1.06 sqrt(||A^2 x||)`, k power steps on A^2 from x = 1."""
    x = np.ones(A.shape[:-1]) / np.sqrt(A.shape[-1])
    lam = np.zeros(A.shape[:-2])
    for _ in range(k):
        y = np.einsum("...ij,...j->...i", A, np.einsum("...ij,...j->...i", A, x))
        lam = np.linalg.norm(y, axis=-1)
        x = y / np.where(lam > 0, lam, 1.0)[..., None]
    return 1.06 * np.sqrt(lam)


def plain(A, j):
    """`||A^m||_inf^(1/m)`, `m = 2^j`, without the Collatz-Wielandt step (for comparison)."""
    C, r, e = A.copy(), np.ones(A.shape[:-2]), 1.0
    for _ in range(j):
        s = np.abs(C).sum(-1).max(-1)
        s = np.where(s > 0, s, 1.0)
        C, r, e = C / s[..., None, None], r * s ** e, e / 2
        C = C @ C
    return r * np.abs(C).sum(-1).max(-1) ** e


def rho_mp(A):
    """max|eig| of one float64 matrix with a 40-digit eigensolve."""
    import mpmath as mp
    mp.mp.dps = 40
    return float(max(abs(e) for e in mp.eig(mp.matrix(A.tolist()))[0]))


def check(label, A, rho=None):
    """Ratios bound / rho of the batch `A` (rho: exact, else numpy eigvals) -> a table row.
    States with rho < 1e-100 ||A|| (dry states, rho = 0) are counted apart: there A^2 underflows."""
    A = np.asarray(A, float)
    A = A[np.isfinite(A).all((-1, -2))]
    if rho is None:
        rho = np.abs(np.linalg.eigvals(A)).max(-1)
    b = bound_np(J, D, *cols(A))
    bj = np.asarray(bound_jax(J, D, *cols(A)))
    pos = rho > 1e-100 * np.abs(A).sum(-1).max(-1)
    r, rj = b[pos] / rho[pos], bj[pos] / rho[pos]
    low = np.where(np.minimum(r, rj) < 1 - TOL)[0]
    for k in low:                                    # the eigensolve's round-off?
        rm = rho_mp(A[pos][k])
        r[k], rj[k] = b[pos][k] / rm, bj[pos][k] / rm
    jax_dev = float((np.abs(r - rj) / r).max()) if r.size else 0.0
    # tightness over the states with rho > 1e-8 ||A|| (not dry round-off garbage)
    big = (rho > 1e-8 * np.abs(A).sum(-1).max(-1))[pos]
    rest = {f"G{j}": (bound_np(j, D, *cols(A))[pos] / rho[pos])[big] for j in (4, 5, 6, 8)}
    rest[f"plain{J}"] = (plain(A, J)[pos] / rho[pos])[big]
    rest["power"] = power_x106(A)[pos] / rho[pos]
    rb = r[big]
    if not rb.size:                                  # all nilpotent: rho = 0 = bound
        r = rj = rb = np.ones(1)
        rest = {k: np.ones(1) for k in rest}
    ok = min(r.min(), rj.min()) >= 1 - TOL
    row = dict(label=label, m=len(A), n=A.shape[-1], min=min(r.min(), rj.min()), med=np.median(rb),
               max=rb.max(), rest=rest, jax=jax_dev, under=int((~pos).sum()), mp=len(low), ok=ok)
    print(f"{label:34s} {len(A):6d} n={A.shape[-1]:2d}  G{J}: min {row['min']:.15f} "
          f"med {np.median(rb):.4f} max {rb.max():.4f}  | "
          + "  ".join(f"{k} {v.min():.3f}/{np.median(v):.3f}/{v.max():.3f}" for k, v in rest.items())
          + f"  | rho~0: {row['under']}  mp-rechecked {len(low)}  NumPy-JAX {jax_dev:.0e}  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return row


# ── random matrices ─────────────────────────────────────────────────────────
def permute(T):
    """An exact similarity: the same permutation of rows and columns of every matrix."""
    p = rng.permutation(T.shape[-1])
    return T[:, p][:, :, p]


def scale(m):
    return 10.0 ** rng.uniform(-6, 6, (m, 1, 1))


def random_sets(m=4000):
    rows = []
    for n in (2, 3, 5, 7, 10):
        rows.append(check(f"Gaussian", rng.normal(size=(m, n, n)) * scale(m)))
        for s in (1.0, 1e3):
            d = rng.uniform(-1, 1, (m, n))
            T = np.triu(rng.normal(0, s, (m, n, n)), 1) + d[:, :, None] * np.eye(n)
            sc = scale(m)
            rows.append(check(f"triangular, upper part {s:g}", permute(T * sc),
                              np.abs(d).max(-1) * sc[:, 0, 0]))
        lam = rng.uniform(-1, 1, (m, 2))
        T = np.zeros((m, n, n))
        k = (n + 1) // 2                            # Jordan blocks of size k and n - k
        for i in range(n):
            T[:, i, i] = lam[:, 0] if i < k else lam[:, 1]
            if i + 1 < n and i + 1 != k:
                T[:, i, i + 1] = 10.0 ** rng.uniform(0, 3, m)
        sc = scale(m)
        rows.append(check("defective (Jordan blocks)", permute(T * sc), np.abs(lam).max(-1) * sc[:, 0, 0]))
        rows.append(check("nilpotent", permute(np.triu(rng.normal(size=(m, n, n)), 1) * scale(m)),
                          np.zeros(m)))
        if n % 2 == 0:
            a, w = rng.uniform(-1, 1, (m, n // 2)), rng.uniform(-1, 1, (m, n // 2))
            T = np.zeros((m, n, n))
            for i in range(n // 2):
                T[:, 2 * i, 2 * i] = T[:, 2 * i + 1, 2 * i + 1] = a[:, i]
                T[:, 2 * i, 2 * i + 1], T[:, 2 * i + 1, 2 * i] = -w[:, i], w[:, i]
            Qo = np.linalg.qr(rng.normal(size=(m, n, n)))[0]
            rows.append(check("complex pairs (normal)", Qo @ T @ np.swapaxes(Qo, 1, 2)))
        V = rng.normal(size=(m, n, n)) + np.eye(n) * 1e-3
        D = rng.uniform(-1, 1, (m, n))[:, :, None] * np.eye(n)
        rows.append(check("diagonalizable, random V", V @ D @ np.linalg.inv(V)))
    return rows


# ── model wave-speed matrices ───────────────────────────────────────────────
def speed_matrix(nsm):
    """The flat matrix of the production `gelfand_bound` call, lambdified over (Q, Qaux, p, n),
    and an aux evaluator."""
    default_wave_speed(nsm)
    num = nsm.build_numerics()
    calls = sp.sympify(num.functions["local_max_abs_eigenvalue"].definition).atoms(gelfand_bound)
    assert len(calls) == 1, calls
    flat = list(next(iter(calls)).args[2:])
    k = int(round(len(flat) ** 0.5))
    Q, A, P, N = (list(num.variables), list(num.aux_variables), list(num.parameters),
                  list(num.normal))
    mods = [{"conditional": lambda c, a, b: np.where(c, a, b)}, "numpy"]
    f = sp.lambdify([*Q, *A, *P, *N], flat, modules=mods)
    uav = nsm.operator("update_aux_variables")
    g = (None if uav is None
         else sp.lambdify([*Q, *A, *P], list(sp.flatten(uav.definition)), modules=mods))
    pv = [float(getattr(nsm.sm.parameter_values, str(p))) for p in P]

    def ev(Qv, nrm):
        m = Qv.shape[1]
        aux = np.zeros((len(A), m))
        if g is not None:
            aux = np.array([np.broadcast_to(np.asarray(v, float), (m,))
                            for v in g(*Qv, *aux, *pv)])
        vals = f(*Qv, *aux, *pv, *nrm)
        return np.stack([np.broadcast_to(np.asarray(v, float), (m,)) for v in vals],
                        -1).reshape(m, k, k)
    return ev, [str(q) for q in Q]


def models():
    from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
    from zoomy_core.model.models.ml_sme import MLSME
    import cases
    out = {}
    for dim, tags in ((3, ("left", "right", "bottom", "top")), (2, ("left", "right"))):
        for level in (0, 1, 2, 3):
            sm = SystemModel.from_model(C.sme(level, dim, tags, "wall"))
            out[f"SME({level}) {dim - 1}-D"] = (NumericalSystemModel.from_system_model(sm), dim - 1)
    out["VAM(1) 2-D predictor"] = (cases.vam_wave(1, 30)[1][0], 2)
    for level in (0, 1):
        sm = SystemModel.from_model(MLSME(n_layers=2, level=level, dimension=3,
                                          closures=[Newtonian(), NavierSlip(), StressFree()]))
        out[f"ML-SME(2 layers, {level}) 2-D"] = (NumericalSystemModel.from_system_model(sm), 2)
    return out


def random_states(names, m, ndim):
    """Wet (h 1e-3..50), thin (1e-9..1e-3) and dry (h = 0) cells, |u| <= 8, moments up to 3|u|,
    and the same states thinned to h* in 1e-6..1 h with q* = q h*/h (Audusse faces)."""
    h = np.concatenate([10 ** rng.uniform(-3, 1.7, m - m // 5), 10 ** rng.uniform(-9, -3, m // 10),
                        np.zeros(m // 5 - m // 10)])
    Q = np.zeros((len(names), m))
    for k, nm in enumerate(names):
        Q[k] = (h if nm == "h" else rng.uniform(0, 2, m) if nm == "b" else
                h * rng.uniform(-1, 1, m) * (8.0 if nm.endswith("_0") else 3.0 * rng.uniform(-1, 1, m)))
    Qf = Q.copy()
    f = 10 ** rng.uniform(-6, 0, m)
    for k, nm in enumerate(names):
        if nm != "b":
            Qf[k] *= f
    return [Q, Qf]


def normals(m, ndim):
    if ndim == 1:
        return np.sign(rng.uniform(-1, 1, (1, m)))
    th = rng.uniform(0, 2 * np.pi, m)
    return np.stack([np.cos(th), np.sin(th)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*", default=[])
    ap.add_argument("--part", choices=["all", "random", "models"], default="all",
                    help="random matrices, or the models (and --states), or both")
    args = ap.parse_args()
    print(f"gelfand_bound, {J} squarings: bound / max|eig|", flush=True)
    rows = random_sets() if args.part != "models" else []
    built = {}
    for label, (nsm, ndim) in (models() if args.part != "random" else {}).items():
        ev, names = speed_matrix(nsm)
        built[tuple(names)] = (ev, ndim)
        for kind, Qv in zip(("cells", "faces"), random_states(names, 20000, ndim)):
            rows.append(check(f"{label} {kind}", ev(Qv, normals(Qv.shape[1], ndim))))
    for path in args.states:
        d = np.load(path, allow_pickle=True)
        names = tuple(str(s) for s in (d["names"] if "names" in d else d["state"]))
        if names not in built:
            print(f"{path}: no model with state {names}")
            continue
        ev, ndim = built[names]
        Q = np.asarray(d["Q"], float)
        Q = Q[None] if Q.ndim == 2 else Q
        A = np.concatenate([ev(q, normals(q.shape[1], ndim)) for q in Q])
        rows.append(check(os.path.basename(path), A))
    ok = all(r["ok"] for r in rows)
    print(f"{len(rows)} sets, {sum(r['m'] for r in rows)} matrices: min ratio "
          f"{min(r['min'] for r in rows):.15f}, max {max(r['max'] for r in rows):.4f}; "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
