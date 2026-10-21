"""Strict 1D pairwise coarsening for Poisson interior: R, P with R @ P = I, Galerkin A_c = R A P."""

from __future__ import annotations

import numpy as np

from zoomy_jax.gnn_blueprint.mesh_1d_poisson import edge_list_chain, laplacian_1d_interior_dense


def restriction_prolongation_1d_pair(n_f: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-weighting restriction and piecewise-constant prolongation.

    Fine length `n_f` must be even and >= 2. Coarse length `n_c = n_f // 2`.
    Shapes: `R` is `(n_c, n_f)`, `P` is `(n_f, n_c)`.
    """
    if n_f < 2 or n_f % 2 != 0:
        raise ValueError(f"1D pairwise coarsening requires even n_f >= 2; got {n_f=}")
    n_c = n_f // 2
    r = np.zeros((n_c, n_f), dtype=np.float64)
    p = np.zeros((n_f, n_c), dtype=np.float64)
    for i in range(n_c):
        j0, j1 = 2 * i, 2 * i + 1
        r[i, j0] = 0.5
        r[i, j1] = 0.5
        p[j0, i] = 1.0
        p[j1, i] = 1.0
    return r, p


def build_poisson_hierarchy_1d(
    n_interior: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Returns `(A_list, R_list, P_list, edges_list, shapes)`.

    `shapes[k]` is the number of interior unknowns at level `k`. Halving continues
    while the current length is even and at least 2, down to a single coarse unknown.
    """
    if n_interior < 1:
        raise ValueError("n_interior must be >= 1")

    shapes: list[int] = []
    a_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    cur_n = n_interior
    a_list.append(laplacian_1d_interior_dense(cur_n))
    shapes.append(cur_n)
    edges_list.append(edge_list_chain(cur_n))

    while cur_n >= 2 and cur_n % 2 == 0:
        n_c = cur_n // 2
        r_mat, p_mat = restriction_prolongation_1d_pair(cur_n)
        a_f = a_list[-1]
        a_c = r_mat @ a_f @ p_mat
        r_list.append(r_mat)
        p_list.append(p_mat)
        a_list.append(a_c)
        shapes.append(n_c)
        edges_list.append(edge_list_chain(n_c))
        cur_n = n_c

    return a_list, r_list, p_list, edges_list, shapes


def build_poisson_hierarchy_1d_vector(
    n_interior: int,
    n_components: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Same graph as scalar 1D, but unknowns are `d` per cell (ordering: cell-major).

    Fine operator $A = L \\otimes I_d$ with $L$ the tridiagonal 1D interior
    Laplacian (`np.kron(L, np.eye(d))`). Restriction / prolongation
    $R = R_{1\\mathrm{d}} \\otimes I_d$, $P = P_{1\\mathrm{d}} \\otimes I_d$.
    GMRES and Jacobi operate on the stacked vector in
    $\\mathbb{R}^{n_\\ell d}$ at each level $\\ell$.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if n_components == 1:
        return build_poisson_hierarchy_1d(n_interior)

    idc = np.eye(n_components, dtype=np.float64)
    a_list_s, r_list_s, p_list_s, edges_list, shapes = build_poisson_hierarchy_1d(n_interior)
    a_list = [np.kron(a, idc) for a in a_list_s]
    r_list = [np.kron(r, idc) for r in r_list_s]
    p_list = [np.kron(p, idc) for p in p_list_s]
    return a_list, r_list, p_list, edges_list, shapes


if __name__ == "__main__":
    r8, p8 = restriction_prolongation_1d_pair(8)
    assert r8.shape == (4, 8) and p8.shape == (8, 4)
    assert np.allclose(r8 @ p8, np.eye(4))
    al, rl, pl, el, sh = build_poisson_hierarchy_1d(64)
    assert len(al) == len(sh) == len(el)
    assert len(rl) == len(pl) == len(al) - 1
    assert sh[0] == 64 and sh[-1] == 1
    _, _, _, _, sh8 = build_poisson_hierarchy_1d(8)
    av, rv, pv, _, shv = build_poisson_hierarchy_1d_vector(8, 3)
    assert shv == sh8 and av[0].shape == (24, 24)
    assert np.allclose(rv[0] @ pv[0], np.eye(rv[0].shape[0]), atol=1e-10)
    print("mg_structured_hierarchy_1d checks OK", sh, "vector(8,3) A0", av[0].shape)
