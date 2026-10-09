"""Gate 5: full mesh sensitivity.  J(X) = sum_c h_c^2 |V_c|(X) at T as a
function of ALL vertex coordinates X; the geometry (incl. LSQ matrices) is
computed from X inside the differentiated run.  Forward (jvp) along a smooth
interior-vertex displacement vs central FD; reverse (checkpointed, jax.grad
w.r.t. every vertex) vs forward."""
import numpy as np
import jax
import jax.numpy as jnp
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax


def objective(make, ad_mode, t_end, dt):
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, dt=dt, ad_mode=ad_mode,
                                    ad_checkpoints=16)
    Q0, A0 = s.setup(mesh, nsm)
    h, nc = s.nsm.sm.roles.depth, Q0.shape[1]

    def J(X):
        from zoomy_jax.symsolver.runtime import geometry_of
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, X=X), X=X)
        return jnp.sum(Q[h] ** 2 * geometry_of(s.topo, X).cell_volumes[:nc])
    return jax.jit(J), s.X


def bump_direction(X):
    """Smooth displacement of interior vertices, zero on the boundary."""
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    s = jnp.prod(jnp.sin(jnp.pi * (X - lo) / (hi - lo)), axis=0)
    d = jnp.stack([s * (1.0 + 0.5 * k) for k in range(X.shape[0])])
    return d * (hi - lo) / X.shape[1] ** (1 / X.shape[0])


for label, make, t_end, dt in (
        ("SWE 2-D bump O1, walls, background flow",
         lambda: C.bump_2d(1, nx=16, flow=(0.1, 0.05)), 0.3, 2e-2),
        ("SME(1) 1-D dam break O1", lambda: C.dambreak_1d(1, 1, nx=60), 0.05, 1e-3),
        ("SWE 2-D bump O2, walls, background flow",
         lambda: C.bump_2d(2, nx=16, flow=(0.1, 0.05)), 0.3, 2e-2),
        ("SME(1) 1-D dam break O2", lambda: C.dambreak_1d(1, 2, nx=60), 0.05, 1e-3)):
    Jf, X0 = objective(make, "forward", t_end, dt)
    Jr, _ = objective(make, "reverse", t_end, dt)
    v = bump_direction(X0)
    d = float(jax.jvp(Jf, (X0,), (v,))[1])
    fd = min(abs(float((Jf(X0 + r * v) - Jf(X0 - r * v)) / (2 * r)) - d)
             for r in (1e-3, 1e-4, 1e-5, 1e-6)) / abs(d)
    g = jax.grad(Jr)(X0)
    rev = float(jnp.vdot(g, v))
    print(f"{label}: dJ/dX.v = {d: .10e}  |FD-jvp| {fd:.1e}  |rev-jvp| "
          f"{abs(rev - d) / abs(d):.1e}  (gradient w.r.t. {g.size} vertex "
          f"coordinates, finite: {bool(jnp.isfinite(g).all())})")
