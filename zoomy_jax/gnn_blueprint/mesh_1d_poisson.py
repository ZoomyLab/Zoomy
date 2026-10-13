"""1D structured Poisson interior: tridiagonal Laplacian and chain edges."""

from __future__ import annotations

import numpy as np


def laplacian_1d_interior_dense(n: int) -> np.ndarray:
    """Interior Dirichlet stencil, same sign as `poisson_1d.laplacian_1d_dense` (diag -2, off +1)."""
    if n < 1:
        return np.zeros((0, 0), dtype=np.float64)
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a[i, i] = -2.0
        if i > 0:
            a[i, i - 1] = 1.0
        if i < n - 1:
            a[i, i + 1] = 1.0
    return a


def edge_list_chain(n: int) -> np.ndarray:
    """Undirected neighbor edges along a line, shape (2, E); both directions."""
    if n < 2:
        return np.zeros((2, 0), dtype=np.int32)
    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
    e = np.asarray(edges, dtype=np.int32).T
    return e


def gaussian_bump_1d(
    n: int,
    *,
    amplitude: float = 0.12,
    sigma: float = 0.15,
    cx: float = 0.5,
) -> np.ndarray:
    """Gaussian bump on interior nodes `i = 0..n-1` in normalized coordinate `(i+0.5)/n`."""
    out = np.zeros(n, dtype=np.float64)
    denom = 2.0 * sigma**2
    for i in range(n):
        x = (i + 0.5) / max(n, 1)
        out[i] = amplitude * np.exp(-((x - cx) ** 2) / denom)
    return out
