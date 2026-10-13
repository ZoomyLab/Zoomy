"""1D Dirichlet Poisson on interior cells: `L u = f` (second-difference, dx=1)."""

from __future__ import annotations

import jax.numpy as jnp


def laplacian_1d_dense(n: int, dtype=jnp.float64) -> jnp.ndarray:
    """Tridiagonal `n×n` interior Laplacian (Dirichlet ghosts zero)."""
    main = -2.0 * jnp.ones((n,), dtype=dtype)
    off = jnp.ones((n - 1,), dtype=dtype)
    return jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)


def solve_poisson_1d(f: jnp.ndarray, a: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return `u` with `A u = f`."""
    n = f.shape[0]
    if a is None:
        a = laplacian_1d_dense(n, dtype=f.dtype)
    return jnp.linalg.solve(a, f)
