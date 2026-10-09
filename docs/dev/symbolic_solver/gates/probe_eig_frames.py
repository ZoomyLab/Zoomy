"""CFL wave speed (eigensolve) cost on every saved frame of the bump run,
cell states vs Audusse face states, min over 3x50 calls."""
import time as wall, numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
import probe_eig_content as P
s, _, A0 = g7.new_solver(1)
rt, p = s.rt, s.parameters
fr = np.load("../results/escalante_bump_symbolic_frames.npz")
f = jax.jit(lambda Qs, As, n: rt.eigenvalues_cfl(Qs, As, p, n))
def tm(Qs, As, n):
    jax.block_until_ready(f(Qs, As, n)); best = 1e9
    for _ in range(3):
        t0 = wall.perf_counter()
        for _ in range(50): r = f(Qs, As, n)
        jax.block_until_ready(r); best = min(best, (wall.perf_counter() - t0) / 50)
    return best, r
rows = []
for k in range(0, len(fr["t"]), 4):
    Q = np.asarray(fr["Q"][k]); A = np.asarray(s.initial_aux(jnp.asarray(Q), A0))
    L, R = P.L, P.R
    cQ = jnp.asarray(np.concatenate([Q[:, L], Q[:, R]], 1)); cA = jnp.asarray(np.concatenate([A[:, L], A[:, R]], 1))
    QLs, QRs = P.hr(Q[:, L], Q[:, R]); hQ = jnp.asarray(np.concatenate([QLs, QRs], 1))
    n = jnp.ones((1, cQ.shape[1]))
    tc, rc = tm(cQ, cA, n); th, _ = tm(hQ, cA, n)
    finite = bool(np.isfinite(np.asarray(rc)).all())
    print(f"t = {float(fr['t'][k]):5.1f}: cell states {1e6 * tc:6.0f} us | Audusse face states {1e6 * th:6.0f} us | max|lambda| {float(np.nanmax(np.abs(np.asarray(rc)))):.3g} finite {finite}")
