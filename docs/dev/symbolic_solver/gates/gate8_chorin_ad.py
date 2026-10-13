"""Gate 8: AD through the symbolic Chorin solver (JAX).

Escalante VAM(1) bump, 16 steps of 2^-7.  J = sum(h^2) + sum(q_0^2).
dJ/dg and dJ/dX.v (all vertex coordinates): forward jvp vs central FD,
reverse (checkpointed loop, custom_linear_solve transpose) vs forward.
"""
import jax
import jax.numpy as jnp
import numpy as np

import gate7_chorin as g7
from zoomy_jax.symsolver.solver import AdMode

N = 16


def solver(ad_mode):
    g7.N_STEPS = N
    return g7.new_solver(1, ad_mode=ad_mode)


def J(s, Q, Qaux, p, X):
    _, Qn, _ = s.run(Q, Qaux, parameters=p, X=X)
    return jnp.sum(Qn[1] ** 2) + jnp.sum(Qn[2] ** 2)


if __name__ == "__main__":
    sf, Q, Qaux = solver(AdMode.forward)
    sr, _, _ = solver(AdMode.reverse)
    p0, X0 = jnp.asarray(sf.parameters), sf.X
    names = list(sf.rt.parameter_names)
    ig = names.index("g")
    e = jnp.zeros_like(p0).at[ig].set(1.0)
    fwd = jax.jvp(lambda p: J(sf, Q, Qaux, p, X0), (p0,), (e,))[1]
    eps = 1e-5
    fd = (J(sf, Q, Qaux, p0 + eps * e, X0) - J(sf, Q, Qaux, p0 - eps * e, X0)) / (2 * eps)
    rev = jax.grad(lambda p: J(sr, Q, Qaux, p, X0))(p0)[ig]
    print(f"Chorin AD dJ/dg = {float(fwd): .10e}: |FD-jvp| {abs(float(fd - fwd)):.1e}, "
          f"|rev-jvp| {abs(float(rev - fwd)):.1e}")

    v = jnp.asarray(np.random.default_rng(0).standard_normal(X0.shape)) * 1e-3
    fwd = jax.jvp(lambda X: J(sf, Q, Qaux, p0, X), (X0,), (v,))[1]
    eps = 1e-4
    fd = (J(sf, Q, Qaux, p0, X0 + eps * v) - J(sf, Q, Qaux, p0, X0 - eps * v)) / (2 * eps)
    gX = jax.grad(lambda X: J(sr, Q, Qaux, p0, X))(X0)
    rev = jnp.sum(gX * v)
    print(f"Chorin mesh AD dJ/dX.v = {float(fwd): .10e}: |FD-jvp| {abs(float(fd - fwd)):.1e}, "
          f"|rev-jvp| {abs(float(rev - fwd)):.1e}  (gradient w.r.t. {X0.size} "
          f"vertex coordinates, finite: {bool(jnp.isfinite(gX).all())})")
    print("G8_DONE")
