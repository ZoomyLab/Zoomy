"""Spectral radius of the VAM(1) 2-D quasilinear matrix A_n at real states
(flow over the bump, 120x12, t = 0.1): exact (eig) vs power iteration on A^2
(k matvecs, x0 = ones) vs the core's Perron bound ||(|A|^k) 1||^(1/k)."""
import sys
import numpy as np, jax, jax.numpy as jnp, sympy as sp
sys.argv = ["x", "0.1", "120"]
import bench_vam2d_bump as B   # runs the 120x12 case to t = 0.1 (prints timing)

s = B.s
nsm = s.nsm
Q = np.asarray(B.Qe)
A_aux = np.asarray(s.initial_aux(B.Qe, jnp.zeros((len(nsm.aux_state), Q.shape[1]))))
p = np.asarray(s.parameters)
qm = nsm.quasilinear_matrix                       # (n, n, dim)
nv, dim = qm.shape[0], qm.shape[2]
args = list(nsm.state) + list(nsm.aux_state) + list(nsm.parameters.values())
f = sp.lambdify(args, sp.Array(qm), modules="numpy")
mats = []
for c in range(Q.shape[1]):
    M = np.asarray(f(*Q[:, c], *A_aux[:, c], *p), dtype=float)
    for d in range(dim):
        mats.append(M[:, :, d])
A = np.stack(mats)                                  # (cells*dim, n, n)
exact = np.abs(np.linalg.eigvals(A)).max(axis=1)
ok = exact > 1e-8
print(f"{ok.sum()} matrices {nv}x{nv}; exact rho in [{exact[ok].min():.3g}, {exact[ok].max():.3g}]")


def power_sq(A, k):
    x = np.ones(A.shape[:2])
    lam = None
    for _ in range(k):
        y = np.einsum("bij,bj->bi", A, np.einsum("bij,bj->bi", A, x))
        lam = np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1)
        x = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.sqrt(lam)


def perron(A, k):
    M = np.abs(A); v = np.ones(A.shape[:2]); best = np.full(A.shape[0], np.inf)
    for j in range(1, k + 1):
        v = np.einsum("bij,bj->bi", M, v)
        best = np.minimum(best, v.max(axis=1) ** (1 / j))
    return best


for name, fn, ks in (("power on A^2", power_sq, (5, 10, 20, 40)), ("Perron bound", perron, (1, 2, 4, 8))):
    for k in ks:
        r = fn(A, k)[ok] / exact[ok]
        print(f"  {name:13s} k={k:3d}: ratio est/exact min {r.min():.4f} median {np.median(r):.4f} max {r.max():.4f}")
