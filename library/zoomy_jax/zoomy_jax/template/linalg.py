"""Differentiable implicit solves for the template's `Root` / `LinSolve`.

Moved (semantics kept, old file untouched) from
`zoomy_jax.fvm.implicit_ad` and extended:

* `newton_root` -- Newton inside `jax.lax.custom_root`: the loop is not
  differentiated, the root's derivative follows from the implicit function
  theorem with the exact AD linearisation of the residual, solved with the
  same linear solver (`custom_linear_solve` with a transpose solve, so it is
  forward AND reverse differentiable).  Returns the solver statistics
  `(rnorm, iters)` as non-differentiable aux.
* `gmres` -- restarted GMRES with an INJECTABLE inner product (SPMD:
  a `psum`-ed inner product makes every norm global), right
  preconditioning, returns `(x, rel_resid, iters)`.
* `cellwise_solve` (block-diagonal over the trailing cell axis),
  `dense_solve`, `banded_solve` (block-banded over the cell axis,
  read off by graph-coloured probing).
* `make_solver` -- the `LinSolveOpts.kind` dispatch, with the
  `constant` / `constant_per_row` null-space projections.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

FD_EPS = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)


def default_inner(a, b):
    return jnp.vdot(jnp.ravel(a), jnp.ravel(b))


def project(v, mode):
    if mode is None:
        return v
    if mode == "constant":
        return v - jnp.mean(v)
    if mode == "constant_per_row":
        return v - jnp.mean(v, axis=-1, keepdims=True)
    raise ValueError(f"unknown nullspace {mode!r}")


def gmres(matvec, b, *, tol, maxiter, restart=30, inner=default_inner,
          precond=None):
    """Right-preconditioned restarted GMRES: `x` with
    `||b - A x|| <= tol ||b||` (norms from `inner`) or after `maxiter`
    restart cycles.  `x0 = 0`.  Returns `(x, rel_resid, iters)`."""
    shape, dtype = b.shape, b.dtype
    bf = jnp.ravel(b)
    n = bf.size
    m = int(min(int(restart), n))

    def M(v):
        return v if precond is None else jnp.ravel(precond(v.reshape(shape)))

    def A(v):
        return jnp.ravel(matvec(v.reshape(shape)))

    def dot(a, c):
        return inner(a.reshape(shape), c.reshape(shape))

    def norm(a):
        return jnp.sqrt(dot(a, a))

    vdot = jax.vmap(dot, in_axes=(0, None))

    def cycle(y, r, beta):
        V = jnp.zeros((m + 1, n), dtype).at[0].set(r / beta)
        H = jnp.zeros((m + 1, m), dtype)

        def arnoldi(j, VH):
            V, H = VH
            w = A(M(V[j]))
            mask = jnp.arange(m + 1) <= j
            h = vdot(V, w) * mask                  # classical Gram-Schmidt,
            w = w - V.T @ h
            h2 = vdot(V, w) * mask                 # applied twice (CGS2)
            w = w - V.T @ h2
            hn = norm(w)
            H = H.at[:, j].set((h + h2).at[j + 1].set(hn))
            V = V.at[j + 1].set(jnp.where(hn > 0, w / jnp.where(hn > 0, hn, 1.0),
                                          0.0))
            return V, H

        V, H = jax.lax.fori_loop(0, m, arnoldi, (V, H))
        e1 = jnp.zeros(m + 1, dtype).at[0].set(beta)
        z = jnp.linalg.lstsq(H, e1)[0]
        return y + V[:m].T @ z

    bnorm = norm(bf)
    atol = tol * bnorm

    def cond(c):
        _, _, beta, k = c
        return jnp.logical_and(k < maxiter, beta > atol)

    def body(c):
        y, r, beta, k = c
        y = cycle(y, r, jnp.where(beta > 0, beta, 1.0))
        r = bf - A(M(y))
        return y, r, norm(r), k + 1

    y, r, beta, k = jax.lax.while_loop(
        cond, body, (jnp.zeros_like(bf), bf, bnorm, jnp.asarray(0)))
    x = M(y).reshape(shape)
    rel = beta / jnp.where(bnorm > 0, bnorm, 1.0)
    return x, rel, k * m


def _cells(b):
    return b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(1, -1)


def cellwise_solve(matvec, b):
    """`x` with `matvec(x) = b` for an operator BLOCK-DIAGONAL over the
    trailing (cell) axis: one probe per leading row reads every cell block."""
    B2 = _cells(b)
    nv = B2.shape[0]
    units = jnp.eye(nv, dtype=b.dtype)
    cols = jax.vmap(lambda e: _cells(matvec(jnp.broadcast_to(
        e[:, None], B2.shape).reshape(b.shape))))(units)   # (k, i, c)
    blocks = jnp.transpose(cols, (2, 1, 0))                # (c, i, k)
    x = jax.vmap(jnp.linalg.solve)(blocks, B2.T[:, :, None])[:, :, 0]
    return x.T.reshape(b.shape)


def dense_solve(matvec, b, nullspace=None):
    n = b.size
    A = jax.vmap(lambda e: jnp.ravel(matvec(e.reshape(b.shape))))(
        jnp.eye(n, dtype=b.dtype)).T
    if nullspace is None:
        return jnp.linalg.solve(A, jnp.ravel(b)).reshape(b.shape)
    return jnp.linalg.lstsq(A, jnp.ravel(b))[0].reshape(b.shape)


def banded_solve(matvec, b, bw):
    """Operator block-banded over the cell axis with half-bandwidth `bw`
    cells: read off with `nv*(2bw+1)` coloured probes, assembled in
    cell-major order and solved densely (correct for any size; a dedicated
    banded factorisation is a performance refinement)."""
    B2 = _cells(b)
    nv, nc = B2.shape
    stride = 2 * bw + 1
    rows, cols, src = [], [], []
    for v in range(nv):
        for color in range(stride):
            for i in range(nc):
                for c in range(max(0, i - bw), min(nc, i + bw + 1)):
                    if c % stride == color:
                        for r in range(nv):
                            rows.append(i * nv + r)
                            cols.append(c * nv + v)
                            src.append(((v * stride + color) * nv + r) * nc + i)
    rows, cols, src = (np.asarray(a) for a in (rows, cols, src))
    probes = []
    for v in range(nv):
        for color in range(stride):
            e = np.zeros((nv, nc))
            e[v, color::stride] = 1.0
            probes.append(e)
    E = jnp.asarray(np.stack(probes), dtype=b.dtype)
    resp = jax.vmap(lambda e: _cells(matvec(e.reshape(b.shape))))(E)
    n = nv * nc
    A = jnp.zeros((n, n), b.dtype).at[rows, cols].set(jnp.ravel(resp)[src])
    x = jnp.linalg.solve(A, B2.T.reshape(-1))
    return x.reshape(nc, nv).T.reshape(b.shape)


def make_solver(*, kind, tol, restart=30, maxiter=50, precond=None,
                nullspace=None, bandwidth=None, inner=default_inner,
                symmetric=False):
    """`solve(matvec, b) -> (x, (rel_resid, iters))` for one LinSolve kind.
    Carries no AD rule: wrap it in `custom_linear_solve`."""

    def solve(matvec, b):
        rhs = project(b, nullspace)

        def mv(x):
            return matvec(x)

        if kind == "krylov":
            x, rel, it = gmres(mv, rhs, tol=tol, maxiter=maxiter,
                               restart=restart, inner=inner, precond=precond)
        else:
            if kind == "dense":
                x = dense_solve(mv, rhs, nullspace)
            elif kind == "cellwise":
                x = cellwise_solve(mv, rhs)
            elif kind == "banded":
                x = banded_solve(mv, rhs, int(bandwidth))
            else:
                raise ValueError(f"unknown linear solve kind {kind!r}")
            r = rhs - mv(x)
            bn = jnp.sqrt(inner(rhs, rhs))
            rel = jnp.sqrt(inner(r, r)) / jnp.where(bn > 0, bn, 1.0)
            it = jnp.asarray(1)
        x = project(x, nullspace)
        return x, (rel, jnp.asarray(it).astype(rel.dtype))

    return solve


def fd_jvp(residual):
    def jv(x, v):
        vmax = jnp.max(jnp.abs(v))
        safe = jnp.where(vmax > 0, vmax, 1.0)
        h = FD_EPS * (1.0 + jnp.max(jnp.abs(x))) / safe
        d = (residual(x + h * v) - residual(x - h * v)) / (2.0 * h)
        return jnp.where(vmax > 0, d, jnp.zeros_like(d))
    return jv


def newton_root(residual, x0, jvp, solve, *, tol, maxiter,
                inner=default_inner):
    """Root of `residual` by Newton, differentiable by the implicit function
    theorem (`custom_root`).

    `jvp(x, v)`: Jacobian action of the NEWTON updates (AD, FD or
    analytic); derivatives always use the exact AD linearisation.
    `solve(matvec, b) -> (x, aux)` (see `make_solver`).
    Stops when `||R|| <= tol` or after `maxiter` updates (a NaN residual
    stops too).  Returns `(x, (rnorm, iters))`; the template guards on
    `rnorm <= tol` (D9)."""

    def norm(r):
        return jnp.sqrt(inner(r, r))

    def plain(mv, b):
        return solve(mv, b)[0]

    def newton(f, x_init):
        def cond(c):
            _, r, i = c
            return jnp.logical_and(i < maxiter, norm(r) > tol)

        def body(c):
            x, r, i = c
            x = x + plain(lambda v: jvp(x, v), -r)
            return x, f(x), i + 1

        x, r, i = jax.lax.while_loop(cond, body, (x_init, f(x_init),
                                                   jnp.asarray(0)))
        # aux as floats: custom_root's JVP gives integer aux an integer (not
        # float0) tangent; the caller casts the count back.
        return x, (norm(r), i.astype(r.dtype))

    def tangent_solve(g, y):
        return jax.lax.custom_linear_solve(g, y, solve=plain,
                                           transpose_solve=plain)

    return jax.lax.custom_root(residual, x0, newton, tangent_solve,
                               has_aux=True)


__all__ = ["gmres", "cellwise_solve", "dense_solve", "banded_solve",
           "make_solver", "newton_root", "fd_jvp", "project", "default_inner"]
