"""The SAME algorithm hand-written directly in JAX (reference for length + values).

Uses the runtime's mesh ops / Root / LinSolve, like the template does.
"""
import jax.numpy as jnp
from jax import lax


def make_march(r, order=2, fixed_dt=None):
    def L(Q, g, dx, o):
        QL, QR = r["reconstruct"](r["ghost"](Q), o)
        return -r["div"](r["face_map"](r["rusanov"], QL, QR, g), dx)

    def march(Q, t, T, g, cf, nu, cfl, dx, hb):
        def body(s):
            t, Q, n, tok = s
            dt = (cfl * dx / jnp.max(r["cell_map"](r["wavespeed"], Q, g))
                  if fixed_dt is None else fixed_dt)
            dt = jnp.minimum(dt, T - t)
            Q1 = Q + dt * L(Q, g, dx, order)
            Qc = 0.5 * Q + 0.5 * (Q1 + dt * L(Q1, g, dx, order)) if order == 2 else Q1
            bad = r["cell_map"](r["troubled"], Qc, hb) > 0.5
            Qc = lax.cond(jnp.any(bad),
                          lambda: jnp.where(bad, Q + dt * L(Q, g, dx, 1), Qc),
                          lambda: Qc)
            h, hu = Qc[0:1], Qc[1:2]
            hu = r["root"](lambda x: x - hu + dt * cf * x * jnp.abs(x) / h**2, hu)
            hu = r["linsolve"](lambda x: x - dt * nu * r["lap"](x, dx), hu)
            return t + dt, jnp.concatenate([h, hu]), n + 1, tok + 1
        s = (jnp.asarray(t, float), Q, jnp.asarray(0), jnp.asarray(0))
        t, Q, n, _ = lax.while_loop(lambda s: s[0] < T, body, s)
        return Q, t, n
    return march
