"""Same kernel, different matrices; same matrices, different kernel.
VAM bump at t = 10 s.  Times are per call (no profiler), min over 5x200."""
import time as wall, numpy as np, jax, jax.numpy as jnp
import gate7_chorin as g7
from probe_eig_content import hr, Q, A, L, R   # cell states and Audusse face states
s, _, _ = g7.new_solver(1)
rt, p = s.rt, s.parameters
n = jnp.ones((1, 2 * len(L)))
cellQ = jnp.asarray(np.concatenate([Q[:, L], Q[:, R]], 1)); cellA = jnp.asarray(np.concatenate([A[:, L], A[:, R]], 1))
QLs, QRs = hr(Q[:, L], Q[:, R])
hrQ = jnp.asarray(np.concatenate([QLs, QRs], 1))
cases = {"CFL kernel on cell states": (rt.eigenvalues_cfl, cellQ),
         "CFL kernel on Audusse face states": (rt.eigenvalues_cfl, hrQ),
         "plain eig kernel on cell states": (rt.eigenvalues, cellQ),
         "plain eig kernel on Audusse face states": (rt.eigenvalues, hrQ)}
for k, (fn, Qs) in cases.items():
    f = jax.jit(lambda Qs: fn(Qs, cellA, p, n))
    jax.block_until_ready(f(Qs)); best = 1e9
    for _ in range(5):
        t0 = wall.perf_counter()
        for _ in range(200): r = f(Qs)
        jax.block_until_ready(r); best = min(best, (wall.perf_counter() - t0) / 200)
    print(f"{k:42s} {1e6 * best:6.0f} us for {Qs.shape[1]} matrices")
