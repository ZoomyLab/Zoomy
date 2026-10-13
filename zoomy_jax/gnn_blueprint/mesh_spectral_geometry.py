"""Geometry helpers for spectral predictors on non-uniform 1D and 2D triangular (Delaunay) meshes."""

from __future__ import annotations

import numpy as np


def laplacian_1d_nonuniform_vertex(x_int: np.ndarray, x_left: float = 0.0, x_right: float = 1.0) -> np.ndarray:
    """Dense interior Poisson stencil `-u'' ≈ f` with Dirichlet `u=0` at `x_left, x_right`.

    Interior unknowns at sorted `x_int` (strictly inside `(x_left, x_right)`).
    """
    x_int = np.asarray(x_int, dtype=np.float64).reshape(-1)
    n = x_int.size
    if n == 0:
        return np.zeros((0, 0))
    x = np.concatenate([[x_left], x_int, [x_right]])
    L = np.zeros((n, n))
    for i in range(n):
        k = i + 1
        h_left = x[k] - x[k - 1]
        h_right = x[k + 1] - x[k]
        if h_left <= 0 or h_right <= 0:
            raise ValueError("non-increasing coordinates")
        s = h_left + h_right
        if i > 0:
            L[i, i - 1] = 2.0 / (h_left * s)
        L[i, i] = -2.0 / (h_left * h_right)
        if i < n - 1:
            L[i, i + 1] = 2.0 / (h_right * s)
    return L


def morton_z_order_perm(xy: np.ndarray, grid_bits: int = 10) -> np.ndarray:
    """Z-order (Morton) sort of points in `[0,1]^2` for a 1D line relaxation ordering."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (n, 2)")
    s = (1 << grid_bits) - 1
    ix = np.clip((xy[:, 0] * s).astype(np.int64), 0, s)
    iy = np.clip((xy[:, 1] * s).astype(np.int64), 0, s)
    key = np.zeros(xy.shape[0], dtype=np.int64)
    for b in range(grid_bits):
        key |= ((ix >> b) & 1) << (2 * b)
        key |= ((iy >> b) & 1) << (2 * b + 1)
    return np.argsort(key, kind="mergesort")


def delaunay_centroid_adjacency(
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random Delaunay in `[0,1]^2`: triangle centroids and **binary adjacency** (shared edge).

    Returns `(centroids (n_tri,2), adj (n_tri,n_tri), points (n_points,2), simplices)`.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n_points, 2))
    from scipy.spatial import Delaunay

    tri = Delaunay(pts)
    S = tri.simplices
    centroids = pts[S].mean(axis=1)
    n_c = centroids.shape[0]
    adj = np.zeros((n_c, n_c), dtype=np.float64)
    edges: dict[tuple[int, int], int] = {}
    for ti, simplex in enumerate(S):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, v = int(simplex[a]), int(simplex[b])
            e = (min(u, v), max(u, v))
            if e in edges:
                tj = edges[e]
                adj[ti, tj] = 1.0
                adj[tj, ti] = 1.0
            else:
                edges[e] = ti
    return centroids, adj, pts, S


def delaunay_triangle_centroids_laplacian(
    n_points: int,
    seed: int,
    ridge: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`A = L_graph + ridge * I` on centroid adjacency (SPD if ridge>0)."""
    centroids, adj, pts, S = delaunay_centroid_adjacency(n_points, seed)
    n_c = centroids.shape[0]
    deg = adj.sum(axis=1)
    l_g = -adj + np.diag(deg)
    a_dense = l_g + ridge * np.eye(n_c)
    return centroids, a_dense, pts, S
