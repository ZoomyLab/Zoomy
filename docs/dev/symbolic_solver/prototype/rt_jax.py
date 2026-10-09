"""JAX runtime: what a backend implementer writes (blocks + mesh ops).

mode: 'fwd' (lax.while_loop, jvp-able) | 'rev' (equinox checkpointed).
"""
import jax
import jax.numpy as jnp
from jax import lax


def runtime(kernels, mode="fwd", checkpoints=32, on_write=None, couple=None):
    m = dict(kernels)

    # ── mesh ops (1-D uniform, transmissive) ────────────────────────────────
    def ghost(Q):
        return jnp.concatenate([Q[:, :1], Q[:, :1], Q, Q[:, -1:], Q[:, -1:]], 1)

    def minmod(a, b):
        return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(abs(a), abs(b)), 0.0)

    def reconstruct(Qg, order):
        if order == 1:
            return Qg[:, 1:-2], Qg[:, 2:-1]
        s = minmod(Qg[:, 1:-1] - Qg[:, :-2], Qg[:, 2:] - Qg[:, 1:-1])
        return Qg[:, 1:-2] + 0.5 * s[:, :-1], Qg[:, 2:-1] - 0.5 * s[:, 1:]

    def lap(x, dx):
        xg = jnp.concatenate([x[:, :1], x, x[:, -1:]], 1)
        return (xg[:, :-2] - 2 * x + xg[:, 2:]) / dx**2

    m.update(
        ghost=ghost, reconstruct=reconstruct, lap=lap,
        face_map=lambda k, QL, QR, par: k(QL[0], QL[1], QR[0], QR[1], par),
        cell_map=lambda k, Q, par: k(Q[0], Q[1], par),
        div=lambda F, dx: (F[:, 1:] - F[:, :-1]) / dx,
        reduce_max=jnp.max, reduce_any=lambda x: jnp.any(x > 0.5),
        select=lambda b, A, B: jnp.where(b > 0.5, A, B),
        comp=lambda Q, i: Q[i:i + 1], assemble=lambda *a: jnp.concatenate(a),
        minimum=jnp.minimum, maximum=jnp.maximum, abs=jnp.abs)

    # ── control blocks ──────────────────────────────────────────────────────
    def while_loop(c, b, init):
        init = tuple(jnp.asarray(a, dtype=jnp.asarray(a).dtype) for a in init)
        cf, bf = (lambda s: c(*s)), (lambda s: tuple(b(*s)))
        if mode == "rev":
            import equinox.internal as eqxi
            return eqxi.while_loop(cf, bf, init, kind="checkpointed",
                                   checkpoints=checkpoints)
        return lax.while_loop(cf, bf, init)

    def cond(p, a, b):
        return lax.cond(p, a, b)

    def root(f, x0):                       # Newton, implicit-function AD
        def newton(f, x):
            def body(s):
                r, dr = jax.jvp(f, (s[0],), (jnp.ones_like(s[0]),))
                return s[0] - r / dr, s[1] + 1

            def go(s):
                return (s[1] < 50) & (jnp.max(abs(f(s[0])))
                                      > 1e-14 * (1 + jnp.max(abs(s[0]))))
            return lax.while_loop(go, body, (x, 0))[0]
        return lax.custom_root(f, x0, newton, lambda g, y: y / g(jnp.ones_like(y)))

    def linsolve(A, b):                    # CG, implicit-function AD
        def cg(A, b):
            def body(s):
                x, r, p, rr, i = s
                Ap = A(p)
                a = rr / jnp.vdot(p, Ap)
                x, r = x + a * p, r - a * Ap
                rr2 = jnp.vdot(r, r)
                return x, r, r + (rr2 / rr) * p, rr2, i + 1
            bb = jnp.vdot(b, b)
            s = (jnp.zeros_like(b), b, b, bb, 0)
            return lax.while_loop(lambda s: (s[3] > 1e-28 * bb) & (s[4] < 500),
                                  body, s)[0]
        return lax.custom_linear_solve(A, b, solve=cg, symmetric=True)

    def io_write(tok, t, Q):
        if on_write is not None:
            jax.debug.callback(on_write, t, Q)
        return (tok + 1,)

    def set_ghost(Qg, side, q):
        q = jnp.broadcast_to(q, (Qg.shape[0], 2))
        return Qg.at[:, -2:].set(q) if side == 1 else Qg.at[:, :2].set(q)

    m.update(while_loop=while_loop, cond=cond, root=root, linsolve=linsolve,
             io_write=io_write, set_ghost=set_ghost,
             edge=lambda Q, side: Q[:, -1:] if side == 1 else Q[:, :1])
    if couple is not None:
        m.update(couple_effects(couple))
    return m


class PreciceAdapter:
    """Host side of the couple_* effects: preCICE API calls only."""

    def __init__(self, name, config, mesh, write, read, x):
        import numpy as np
        import petsc4py  # noqa: F401  (must precede precice)
        import precice
        self.np, self.mesh, self.wn, self.rn = np, mesh, write, read
        self.p = precice.Participant(name, config, 0, 1)
        self.ids = self.p.set_mesh_vertices(mesh, np.array([[x, 0.0]]))

    def _write(self, q):
        for i, n in enumerate(self.wn):
            self.p.write_data(self.mesh, n, self.ids, self.np.asarray(q[i]))

    def init(self, q):
        if self.p.requires_initial_data():
            self._write(q)
        self.p.initialize()
        return self.np.bool_(self.p.is_coupling_ongoing())

    def max_dt(self):
        return self.np.float64(self.p.get_max_time_step_size())

    def read(self, dt):
        return self.np.stack([self.p.read_data(self.mesh, n, self.ids, float(dt))
                              for n in self.rn])

    def write(self, q):
        self._write(q)
        return self.np.int64(0)

    def advance(self, dt):
        self.p.advance(float(dt))
        return self.np.bool_(self.p.is_coupling_ongoing())


def couple_effects(a):
    """Each host call consumes the previous token and returns the next one:
    a strict data chain, so XLA cannot reorder the preCICE calls."""
    from jax.experimental import io_callback
    S, i64 = jax.ShapeDtypeStruct, jnp.int64
    T, F, B = S((), i64), S((), jnp.float64), S((), jnp.bool_)
    Qe = S((len(a.rn), 1), jnp.float64)

    def eff(host, out, n_out=1):
        def body(tok, *args):
            r = io_callback(lambda k, *x: (a.np.int64(k + 1), host(*x)),
                            (T, out), tok, *args)
            return r if n_out else (r[0],)
        return body
    return dict(couple_init=eff(a.init, B), couple_max_dt=eff(a.max_dt, F),
                couple_read=eff(a.read, Qe), couple_write=eff(a.write, T, 0),
                couple_advance=eff(a.advance, B))


def build(src, name, module):
    ns = dict(module)
    exec(src, ns)
    return ns[name]
