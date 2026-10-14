"""Assembly of a cell operator by colouring and the host AMG solve, shared by the numpy and
JAX runtimes (pure numpy / scipy / pyamg).

References:
    Griewank, Walther 2008, https://doi.org/10.1137/1.9780898717761
    Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
"""
import numpy as np


def cell_pattern(n, nc, nbrs, n_rows=None):
    """Sparsity of a cell operator on `k = n / nc` cell-major unknowns: each couples to those of
    the cells in its LSQ stencil.

    Returns:
        `rows, cols` in CSR order and a colour per column such that columns sharing a row
        differ.
    """
    k, n_rows = n // nc, nc if n_rows is None else n_rows
    nb = np.asarray(nbrs)[:n_rows]
    cells = np.concatenate([np.arange(n_rows)[:, None], nb], axis=1)  # self + stencil
    c_of = np.repeat(np.arange(n_rows, dtype=np.int64), cells.shape[1])
    d_of = cells.reshape(-1).astype(np.int64)
    keep = (d_of >= 0) & (d_of < nc)
    code = np.unique(c_of[keep] * nc + d_of[keep])          # sorted (c, d) pairs
    c, d = code // nc, code % nc
    # entry (c, r, d, q) of row c k + r: CSR order without a sort
    ptr = np.searchsorted(c, np.arange(n_rows + 1))
    m = np.diff(ptr)[c]
    base = (ptr[c] * k * k + (np.arange(len(c)) - ptr[c]) * k)[:, None, None]
    r, q = np.arange(k)[None, :, None], np.arange(k)[None, None, :]
    pos = (base + r * (m * k)[:, None, None] + q).reshape(-1)
    shape, rows, cols = (len(c), k, k), np.empty(len(pos), np.int32), np.empty(len(pos), np.int32)
    rows[pos] = np.broadcast_to(c[:, None, None] * k + r, shape).reshape(-1)
    cols[pos] = np.broadcast_to(d[:, None, None] * k + q, shape).reshape(-1)
    unknown = np.arange(n)
    return rows, cols, (cell_colors(c, d, nc)[unknown // k] * k
                        + unknown % k).astype(np.int32)


def operator_neighbors(topo, mesh_np):
    """Per cell the cells an operator of the derivative aux couples it to, the sparsity
    `cell_pattern` assembles: the samples of its LSQ fit, i.e. the topology's stencil
    (`topo.lsq_neighbors`, which wraps a periodic seam; the mesh's `lsq_neighbors` ends there)
    and the cell behind each boundary-face sample's ghost (`boundary_face_cells`: the inner
    cell, or the partner across a periodic seam).  With the mesh's stencil alone the assembled
    matrix lacked the cross-seam entries: the AMG solve of a periodic pressure system converged
    to the wrong matrix (true relative residual 0.78 on a periodic VAM(3) channel).  Rows padded
    with -1.
    """
    nb = np.asarray(topo.lsq_neighbors)
    bdy = np.asarray(topo.lsq_boundary_face_neighbors)
    if not bdy.size:
        return nb
    n_bf = int(mesh_np.n_boundary_faces)
    behind = np.asarray(mesh_np.boundary_face_cells, dtype=int)[:n_bf]
    ghost_cells = np.where(bdy >= 0, behind[np.clip(bdy, 0, max(n_bf - 1, 0))], -1)
    return np.concatenate([nb, ghost_cells.astype(nb.dtype)], axis=1)


def cell_colors(c, d, nc, seed=0):
    """Distance-2 colouring of the stencil pairs `(c, d)` by speculative parallel greedy (a clash
    with a heavier random-weight neighbour retries).
    """
    import scipy.sparse as sps
    P = sps.csr_matrix((np.ones(len(c), np.int8), (c, d)), shape=(int(c.max()) + 1, nc))
    C = (P.T @ P).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    w = np.random.default_rng(seed).permutation(nc)
    colors, A = np.full(nc, -1, np.int64), np.arange(nc)
    while len(A):
        S = C[A]                                       # rows of the uncoloured
        deg = np.diff(S.indptr)
        row, nb, has = np.repeat(np.arange(len(A)), deg), S.indices, deg > 0
        used, nbc = np.zeros(len(A), np.uint64), colors[S.indices]
        if has.any():
            used[has] = np.bitwise_or.reduceat(np.where(nbc >= 0, np.left_shift(
                np.uint64(1), np.maximum(nbc, 0).astype(np.uint64)), np.uint64(0)),
                S.indptr[:-1][has])
        free = ~used
        low = free & (~free + np.uint64(1))           # lowest free bit
        cand = np.full(nc, -1, np.int64)
        cand[A] = np.log2(low.astype(np.float64)).round().astype(np.int64)
        if (low == 0).any() or (cand[A] >= 63).any():
            raise RuntimeError("colouring needs more than 63 colours")
        lose = (cand[nb] == cand[A][row]) & (w[nb] > w[A][row])
        lost = np.bincount(row[lose], minlength=len(A)) > 0
        colors[A[~lost]] = cand[A[~lost]]
        A = A[lost]
    return colors


def host_amg_solve(vals, rhs, rows, cols, n, k, ls, pyamg, cache, transpose=False):
    """GMRES preconditioned by a smoothed-aggregation AMG hierarchy of the cell-block-scaled
    matrix; the hierarchy in `cache` is lagged: rebuilt every `ls['amg_rebuild']` solves, or once
    the iterations exceed `ls['amg_growth']` times those after the last rebuild.

    Args:
        vals, rows, cols: the `(n, n)` matrix in COO form, `k x k` blocks per cell
        ls: `tol`, `restart`, `maxiter`, `amg_rebuild`, `amg_growth`
        cache: a dict kept between the solves of one operator
    Returns:
        the solution, `rhs`'s dtype

    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
        Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
    """
    import scipy.sparse as sps
    from scipy.sparse.linalg import gmres as sp_gmres
    rebuild, growth = int(ls.get("amg_rebuild", 1)), float(ls.get("amg_growth", 2.0))
    A = sps.csr_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
    A = (A.T if transpose else A).tobsr(blocksize=(k, k))
    # left-scale by the inverse cell blocks: A ~ M (x) L with a k x k coupling M
    # (indefinite for VAM) becomes ~ I (x) L, AMG-friendly
    A.sort_indices()
    row = np.repeat(np.arange(n // k), np.diff(A.indptr))
    Dinv = sps.block_diag(np.linalg.inv(A.data[A.indices == row]), format="bsr")
    As, bs = (Dinv @ A).tobsr(blocksize=(k, k)), Dinv @ np.asarray(rhs)
    c = cache
    if not c:
        c.update(ml=None, age=0, it0=None, it=0)
    if (c["ml"] is None or c["age"] >= rebuild
            or (c["it0"] is not None and c["it"] > growth * max(c["it0"], 1))):
        c.update(ml=pyamg.smoothed_aggregation_solver(As), age=0, it0=None)
    its = [0]
    x, _ = sp_gmres(As, bs, M=c["ml"].aspreconditioner(), rtol=ls["tol"],
                    atol=0.0, restart=ls["restart"], maxiter=ls["maxiter"],
                    callback=lambda _: its.__setitem__(0, its[0] + 1),
                    callback_type="pr_norm")
    c["age"], c["it"] = c["age"] + 1, its[0]
    if c["it0"] is None:
        c["it0"] = its[0]
    return np.asarray(x, dtype=np.asarray(rhs).dtype)
