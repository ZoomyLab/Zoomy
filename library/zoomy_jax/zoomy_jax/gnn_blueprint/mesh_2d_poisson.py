"""2D Poisson meshes: structured 5-point interior grid and helpers."""

from __future__ import annotations

import numpy as np


def interior_grid_shape(nx: int, ny: int) -> tuple[int, int, int]:
    """Interior unknowns excluding boundary layer (Dirichlet zero on boundary)."""
    if nx < 3 or ny < 3:
        raise ValueError("nx, ny must be >= 3")
    ni_x = nx - 2
    ni_y = ny - 2
    return ni_x, ni_y, ni_x * ni_y


def flat_interior_index(ix: int, iy: int, ni_x: int) -> int:
    return iy * ni_x + ix


def laplacian_2d_interior_dense(nx: int, ny: int) -> np.ndarray:
    """5-point negative Laplacian on interior; SPD for Dirichlet BC."""
    ni_x, ni_y, n = interior_grid_shape(nx, ny)
    a = np.zeros((n, n), dtype=np.float64)
    for iy in range(ni_y):
        for ix in range(ni_x):
            i = flat_interior_index(ix, iy, ni_x)
            a[i, i] = 4.0
            if ix > 0:
                a[i, flat_interior_index(ix - 1, iy, ni_x)] = -1.0
            if ix < ni_x - 1:
                a[i, flat_interior_index(ix + 1, iy, ni_x)] = -1.0
            if iy > 0:
                a[i, flat_interior_index(ix, iy - 1, ni_x)] = -1.0
            if iy < ni_y - 1:
                a[i, flat_interior_index(ix, iy + 1, ni_x)] = -1.0
    return a


def edge_list_from_grid(ni_x: int, ni_y: int) -> np.ndarray:
    """Undirected 4-neighbor edges as (2, E) with both directions."""
    edges = []
    for iy in range(ni_y):
        for ix in range(ni_x):
            i = flat_interior_index(ix, iy, ni_x)
            if ix < ni_x - 1:
                j = flat_interior_index(ix + 1, iy, ni_x)
                edges.extend([(i, j), (j, i)])
            if iy < ni_y - 1:
                j = flat_interior_index(ix, iy + 1, ni_x)
                edges.extend([(i, j), (j, i)])
    if not edges:
        return np.zeros((2, 0), dtype=np.int32)
    e = np.asarray(edges, dtype=np.int32).T
    return e


def gaussian_bump_interior(
    nx: int,
    ny: int,
    *,
    amplitude: float = 0.12,
    sigma: float = 0.15,
    cx: float = 0.5,
    cy: float = 0.5,
) -> np.ndarray:
    """Smooth Gaussian bump on interior nodes (iy-major), normalized cell coords in (0,1)²."""
    ni_x, ni_y, n = interior_grid_shape(nx, ny)
    out = np.zeros(n, dtype=np.float64)
    denom = 2.0 * sigma**2
    for iy in range(ni_y):
        for ix in range(ni_x):
            x = (ix + 0.5) / ni_x
            y = (iy + 0.5) / ni_y
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            i = flat_interior_index(ix, iy, ni_x)
            out[i] = amplitude * np.exp(-r2 / denom)
    return out


def edge_list_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric 0/1 adjacency → (2, E) both directions."""
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                edges.extend([(i, j), (j, i)])
    if not edges:
        return np.zeros((2, 0), dtype=np.int32)
    return np.asarray(edges, dtype=np.int32).T
