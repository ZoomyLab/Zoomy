"""Strict structured 2x2 cell coarsening for 2D Poisson interior grids.

Restriction R (full weighting on macro-cells) and piecewise-constant prolongation P
satisfy R @ P = I. Coarse operators use Galerkin triple products A_c = R @ A_f @ P.

The finest interior must be even (ni_x, ni_y) so the first 2x2 block partition exists.
Coarsening stops at 2x2 or when the current level is odd (e.g. 6x6 -> 3x3 coarsest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zoomy_jax.gnn_blueprint.mesh_2d_poisson import (
    edge_list_from_grid,
    flat_interior_index,
    laplacian_2d_interior_dense,
)


def _require_even_interior(ni_x: int, ni_y: int) -> None:
    if ni_x < 2 or ni_y < 2:
        raise ValueError("interior dimensions must be >= 2")
    if ni_x % 2 != 0 or ni_y % 2 != 0:
        raise ValueError(
            f"strict 2x2 coarsening requires even ni_x, ni_y; got {ni_x=} {ni_y=}"
        )


def restriction_prolongation_2x2(ni_x: int, ni_y: int) -> tuple[np.ndarray, np.ndarray]:
    """R (n_c, n_f), P (n_f, n_c) for interior ordering iy-major, ix-minor."""
    _require_even_interior(ni_x, ni_y)
    n_f = ni_x * ni_y
    ncx, ncy = ni_x // 2, ni_y // 2
    n_c = ncx * ncy
    r = np.zeros((n_c, n_f), dtype=np.float64)
    p = np.zeros((n_f, n_c), dtype=np.float64)
    w = 0.25
    for iy in range(ni_y):
        for ix in range(ni_x):
            j = flat_interior_index(ix, iy, ni_x)
            cx, cy = ix // 2, iy // 2
            k = flat_interior_index(cx, cy, ncx)
            r[k, j] = w
            p[j, k] = 1.0
    return r, p


def build_poisson_hierarchy(nx: int, ny: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """Returns (A_list, R_list, P_list, edges_list, shapes).

    `A_list[0]` is the 5-point Laplacian on the finest interior; coarser `A_list[ell]`
    are Galerkin triple products. `R_list[ell]` maps level ell → ell+1 (coarser);
    `P_list[ell]` maps ell+1 → ell. The coarsest level has no trailing R/P.
    """
    ni_x, ni_y, _ = _shape_interior(nx, ny)
    _require_even_interior(ni_x, ni_y)

    shapes: list[tuple[int, int]] = []
    a_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    cur_nx, cur_ny = ni_x, ni_y
    a_list.append(laplacian_2d_interior_dense(cur_nx + 2, cur_ny + 2))
    shapes.append((cur_nx, cur_ny))
    edges_list.append(edge_list_from_grid(cur_nx, cur_ny))

    while True:
        if cur_nx == 2 and cur_ny == 2:
            break
        if cur_nx < 2 or cur_ny < 2:
            break
        if cur_nx % 2 != 0 or cur_ny % 2 != 0:
            break
        nx_next, ny_next = cur_nx // 2, cur_ny // 2
        if nx_next < 2 or ny_next < 2:
            break
        r_mat, p_mat = restriction_prolongation_2x2(cur_nx, cur_ny)
        a_f = a_list[-1]
        a_c = r_mat @ a_f @ p_mat
        cur_nx, cur_ny = nx_next, ny_next
        r_list.append(r_mat)
        p_list.append(p_mat)
        a_list.append(a_c)
        shapes.append((cur_nx, cur_ny))
        edges_list.append(edge_list_from_grid(cur_nx, cur_ny))

    return a_list, r_list, p_list, edges_list, shapes


def _shape_interior(nx: int, ny: int) -> tuple[int, int, int]:
    if nx < 3 or ny < 3:
        raise ValueError("nx, ny must be >= 3")
    ni_x = nx - 2
    ni_y = ny - 2
    return ni_x, ni_y, ni_x * ni_y


@dataclass(frozen=True)
class StructuredHierarchy:
    """JAX-friendly hierarchy metadata (arrays supplied separately)."""

    nx: int
    ny: int
    n_levels: int
    shapes: tuple[tuple[int, int], ...]


def restrict_field_to_coarser_levels(v: np.ndarray, r_list: list[np.ndarray]) -> list[np.ndarray]:
    """Full-weighting restriction of a scalar per-node field to each multigrid level."""
    levels: list[np.ndarray] = [np.asarray(v, dtype=np.float64)]
    for r in r_list:
        levels.append(r @ levels[-1])
    return levels


def hierarchy_metadata(nx: int, ny: int) -> StructuredHierarchy:
    *_, shapes = build_poisson_hierarchy(nx, ny)
    return StructuredHierarchy(
        nx=nx,
        ny=ny,
        n_levels=len(shapes),
        shapes=tuple(shapes),
    )
