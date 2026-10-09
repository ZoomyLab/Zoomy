"""eigvals cost vs matrix content: the 8x8 normal-projected quasilinear
matrices of the VAM predictor at (a) cell states (what the CFL uses) and
(b) Audusse hydrostatically reconstructed face states (what the Riemann
solver uses), t = 10 s of the bump run."""
import time as wall, numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
s, Q0, A0 = g7.new_solver(1)
fr = np.load("../results/escalante_bump_symbolic_frames.npz")
rt, p = s.rt, s.parameters
names = [str(v) for v in s.nsm.state]; ib, ih = names.index("b"), names.index("h")
mom = [i for i, n in enumerate(names) if n.startswith(("q_", "r_"))]
Q = np.asarray(fr["Q"][20]); A = np.asarray(s.initial_aux(jnp.asarray(Q), A0))
nc = Q.shape[1]; L, R = np.arange(nc - 1), np.arange(1, nc)
def mats(Qs, As):
    ql = np.asarray(rt.quasilinear_matrix(jnp.asarray(Qs), jnp.asarray(As), p))   # (n, n, dim, N)
    return jnp.asarray(np.moveaxis(ql[:, :, 0, :], -1, 0))                       # normal (1,): x
def hr(QL_, QR_):                       # Audusse: h* = max(0, h + b - max(bL, bR)), q *= h*/h
    bs = np.maximum(QL_[ib], QR_[ib]); out = []
    for Qs in (QL_, QR_):
        Qh = Qs.copy(); hst = np.maximum(0.0, Qs[ih] + Qs[ib] - bs)
        fac = np.where(Qs[ih] > 0, hst / np.maximum(Qs[ih], 1e-300), 0.0)
        Qh[ih] = hst; Qh[mom] = Qs[mom] * fac; Qh[ib] = bs; out.append(Qh)
    return out
QLs, QRs = hr(Q[:, L], Q[:, R])
sets = {"cell states (CFL)": mats(np.concatenate([Q[:, L], Q[:, R]], 1), np.concatenate([A[:, L], A[:, R]], 1)),
        "HR face states (flux)": mats(np.concatenate([QLs, QRs], 1), np.concatenate([A[:, L], A[:, R]], 1))}
f = jax.jit(jnp.linalg.eigvals)
for k, M in sets.items():
    jax.block_until_ready(f(M)); t0 = wall.perf_counter()
    for _ in range(100): r = f(M)
    jax.block_until_ready(r)
    dry = int((np.abs(np.asarray(M)).sum((1, 2)) == 0).sum())
    print(f"{k:24s}: {M.shape[0]} matrices, {1e6 * (wall.perf_counter() - t0) / 100:.0f} us per eigvals batch; all-zero matrices {dry}; "
          f"zero entries {float((np.asarray(M) == 0).mean()):.2f}")
