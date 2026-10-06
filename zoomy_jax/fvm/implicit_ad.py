"""Differentiable implicit solves: Newton roots and the linear solves they use.

Every implicit stage of the jax solvers (IMEX source, state-dependent implicit
diffusion) finds a root of a residual ``R(x) = 0`` with a Newton loop.  The
loop itself is not differentiated.  :func:`newton_root` wraps it in
``jax.lax.custom_root``, so the derivative of the converged root follows from
the implicit function theorem,

    dx/dm = -(dR/dx)^{-1} dR/dm,

for every quantity ``m`` the residual closes over (the explicit predictor,
``dt``, the physical parameters, ...).  The tangent system takes the exact AD
linearization of ``R`` and is solved with the same linear solver as the Newton
updates, inside ``jax.lax.custom_linear_solve`` with a ``transpose_solve``, so
the root is differentiable in forward AND reverse mode.

The linear solvers (:func:`gmres`, :func:`cellwise_solve`) carry no AD rule of
their own and accept any ``matvec``, including a finite-difference Jacobian
action that jax cannot transpose.
"""

import jax
import jax.numpy as jnp


def gmres(matvec, b, *, tol, maxiter, restart=20):
    """Restarted GMRES: ``x`` with ``||b - matvec(x)||_2 <= tol ||b||_2`` or
    after ``maxiter`` restart cycles of ``restart`` Arnoldi steps each.
    ``x0 = 0``.  ``matvec`` maps arrays of ``b``'s shape to arrays of ``b``'s
    shape."""
    shape, dtype = b.shape, b.dtype
    bf = b.reshape(-1)
    n = bf.size
    m = int(min(restart, n))

    def A(v):
        return matvec(v.reshape(shape)).reshape(-1)

    def cycle(x, r, beta):
        V = jnp.zeros((m + 1, n), dtype).at[0].set(r / beta)
        H = jnp.zeros((m + 1, m), dtype)

        def arnoldi(j, VH):
            V, H = VH
            w = A(V[j])
            mask = jnp.arange(m + 1) <= j
            h = (V @ w) * mask                 # classical Gram-Schmidt,
            w = w - V.T @ h
            h2 = (V @ w) * mask                # applied twice (CGS2)
            w = w - V.T @ h2
            hn = jnp.linalg.norm(w)
            H = H.at[:, j].set((h + h2).at[j + 1].set(hn))
            V = V.at[j + 1].set(jnp.where(hn > 0, w / jnp.where(hn > 0, hn, 1.0),
                                          0.0))
            return V, H

        V, H = jax.lax.fori_loop(0, m, arnoldi, (V, H))
        e1 = jnp.zeros(m + 1, dtype).at[0].set(beta)
        y = jnp.linalg.lstsq(H, e1)[0]
        return x + V[:m].T @ y

    atol = tol * jnp.linalg.norm(bf)

    def cond(carry):
        _, _, beta, k = carry
        return jnp.logical_and(k < maxiter, beta > atol)

    def body(carry):
        x, r, beta, k = carry
        x = cycle(x, r, beta)
        r = bf - A(x)
        return x, r, jnp.linalg.norm(r), k + 1

    init = (jnp.zeros_like(bf), bf, jnp.linalg.norm(bf), 0)
    return jax.lax.while_loop(cond, body, init)[0].reshape(shape)


def cellwise_solve(matvec, b):
    """``x`` with ``matvec(x) = b`` for an operator that is BLOCK-DIAGONAL
    over the trailing (cell) axis of ``b`` (shape ``(n_vars, n_cells)``).

    The ``n_vars x n_vars`` block of every cell is read off ``n_vars``
    applications of ``matvec`` to the unit vector of one variable placed in
    every cell; each cell is then solved densely."""
    n = b.shape[0]
    units = jnp.eye(n, dtype=b.dtype)
    # cols[k, i, c] = J_c[i, k]
    cols = jax.vmap(lambda e: matvec(jnp.broadcast_to(e[:, None], b.shape)))(
        units)
    blocks = jnp.transpose(cols, (2, 1, 0))                # (n_cells, i, k)
    x = jax.vmap(jnp.linalg.solve)(blocks, b.T[:, :, None])[:, :, 0]
    return x.T


def newton_root(residual, x0, jvp, solve, *, tol, maxiter):
    """Root of ``residual`` by Newton, differentiable by the implicit function
    theorem.

    Parameters
    ----------
    residual : callable ``x -> R(x)``, same shape in and out.
    x0 : initial iterate.
    jvp : callable ``(x, v) -> J(x) v``, the Jacobian action of the NEWTON
        updates (exact AD, finite differences or an analytic form).  Not used
        for derivatives: the tangent system always takes the exact AD
        linearization of ``residual``.
    solve : callable ``(matvec, b) -> x`` (:func:`gmres` with its tolerances
        bound, or :func:`cellwise_solve`), used for the Newton updates and
        for the tangent (and transposed tangent) systems.
    tol, maxiter : the iteration stops when ``||R||_2 <= tol`` or after
        ``maxiter`` updates.
    """

    def newton(f, x_init):
        def cond(carry):
            _, r, i = carry
            return jnp.logical_and(i < maxiter, jnp.linalg.norm(r) > tol)

        def body(carry):
            x, r, i = carry
            x = x + solve(lambda v: jvp(x, v), -r)
            return x, f(x), i + 1

        return jax.lax.while_loop(cond, body, (x_init, f(x_init), 0))[0]

    def tangent_solve(g, y):
        return jax.lax.custom_linear_solve(g, y, solve=solve,
                                           transpose_solve=solve)

    return jax.lax.custom_root(residual, x0, newton, tangent_solve)
