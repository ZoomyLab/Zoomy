"""Mesh geometry for the solver template (`prepare_mesh`): static topology and LSQ stencils built
once in numpy (`build_topology`), then gather-only geometry and LSQ operators from the node
coordinates (`compute_geometry`).  It is written against `X.__array_namespace__()`, so numpy
and jax.numpy run the same code and `jax.grad` w.r.t. `X` goes through; the module never
imports jax.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sps

from zoomy_core.mesh.base_mesh import _local_faces_for_type
from zoomy_core.mesh.lsq_reconstruction import (
    build_monomial_indices,
    expand_neighbors,
    find_derivative_indices,
    get_polynomial_degree,
    get_required_monomials_count,
    scale_lsq_derivative,
)

_SUPPORTED = {"line": 1, "triangle": 2, "quad": 2, "tetra": 3, "hexahedron": 3}

#: Floor of the TPFA |n·e| obliquity factor (DiffusionOperatorV2 convention).
_TPFA_MIN_COS = 0.1
#: Degenerate-distance threshold of the TPFA faces (DiffusionOperatorV2).
_TPFA_MIN_DIST = 1e-30

#: LSQ stencil selection (:func:`_lsq_stencils`).  A fit is UNISOLVENT when
#: every coefficient c_r it reads is determined, and well: with V its
#: monomial matrix (offsets scaled by the stencil radius), e_r lies in the
#: well-conditioned row space of V (singular values below ``_LSQ_COND``
#: times the largest count as zero, :func:`_rows_ok`; residual <=
#: ``_LSQ_RESID``) and its relative singular value 1 / (|V| |row r of
#: pinv(V)|) is >= ``_LSQ_COND``, |V|**2 = |V^T V|_F (sigma_max <= |V| <=
#: M**(1/4) sigma_max).  Exact degeneracies (a 3x3x3 ring: x**3 = x) sit at
#: 1e-16; the compact stencils of gate 39's quads, jittered quads, triangles
#: and hexahedra (degrees 1-3) at >= 4e-3; an extension just past rank (as
#: many samples as coefficients, jittered quads) down to 1e-6.
_LSQ_COND = 1e-3
_LSQ_RESID = 1e-8
#: Extension candidates whose distances agree to this (relative) join the
#: stencil together: equidistant cells extend it symmetrically.
_LSQ_TIE = 1e-6
#: Face-neighbour layers an extension may add; a fit that is still not
#: unisolvent keeps its unextended stencil (nothing in reach determines it).
_LSQ_MAX_LAYERS = 2
#: Cells per batch of the extension (bounds the candidate arrays).
_LSQ_CHUNK = 20000
#: Relative cut-off of the closure fits' pseudo-inverses
#: (:func:`_lsq_closure_matrices`): their null directions sit at round-off
#: (~1e-16), a unisolvent fit's singular values at >= ``_LSQ_COND``.
_LSQ_PINV_RTOL = 1e-10


# ═════════════════════════════════════════════════════════════════════════
# Topology (numpy, static)
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class MeshTopology:
    """Static integer topology plus frozen geometric decisions (normal signs, ghost ranges, LSQ
    stencils and plan, TPFA masks, boundary routing) of a mesh.
    """

    dim: int
    mesh_type: str
    n_cells: int
    n_inner_cells: int
    n_faces: int
    n_vertices: int
    n_boundary_faces: int

    cell_vertices: np.ndarray          # (n_vert_per_cell, n_inner)
    cell_faces: np.ndarray             # (n_faces_per_cell, n_inner)
    face_cells: np.ndarray             # (2, n_faces)
    face_vertices: np.ndarray          # (n_vert_per_face, n_faces), owner order
    face_sign: np.ndarray              # (n_faces,) float ±1 orientation
    face_first_cell: np.ndarray        # (n_faces,) inner cell defining the face

    boundary_faces: np.ndarray         # (n_bf,) face ids (mesh boundary order)
    boundary_owner: np.ndarray         # (n_bf,) inner cell (= face_cells[0])
    boundary_ghost: np.ndarray         # (n_bf,) ghost cell index
    interior_faces: np.ndarray         # faces with both cells inner
    interior_A: np.ndarray
    interior_B: np.ndarray
    face_other_inner: np.ndarray       # (n_faces,) cell 1 if inner else cell 0

    inradius_fallback: np.ndarray      # (n_inner,) bool (degenerate cells)

    # LSQ (degree, monomials, stencils, assembly plan)
    lsq_degree: int
    lsq_monomial_multi_index: tuple
    lsq_scale_factors: np.ndarray      # (n_mon,)
    lsq_grad_cols: np.ndarray          # (dim,) monomial column of d/dx_d
    lsq_neighbors: np.ndarray          # (n_cells, max_nbr)
    lsq_boundary_face_neighbors: np.ndarray  # (n_cells, max_bdy), -1 padded
    lsq_plan: dict = field(repr=False, default_factory=dict)
    # periodic seams: the translation of every stencil sample to the image
    # nearest the cell, (n_cells, max_nbr, dim); None: no periodic BC
    lsq_shift: Optional[np.ndarray] = field(repr=False, default=None)
    # boundary closure (ANALYSIS 40): per closure space ``(owner, space)``
    # the class of each row (-1: no closure, the ghost-sample fit); the
    # classes' kinds per boundary face and the closure cells are in
    # ``lsq_plan["closure"]``
    lsq_closure_class: dict = field(repr=False, default_factory=dict)

    # TPFA (two-point flux) masks
    tpfa_faces: np.ndarray = field(default=None)    # valid interior faces
    tpfa_A: np.ndarray = field(default=None)
    tpfa_B: np.ndarray = field(default=None)
    tpfa_boundary_valid: np.ndarray = field(default=None)  # (n_bf,) bool

    # Boundary routing (by tag NAME) and the BC input cell per boundary face
    bc_tags: Optional[list] = None
    bc_branch: Optional[np.ndarray] = None          # (n_bf,) branch index
    bc_input_cells: Optional[np.ndarray] = None     # (n_bf,) owner/partner
    bc_periodic: Optional[np.ndarray] = None        # (n_bf,) bool

    @property
    def ghost_range(self):
        """Ghost cells are `range(n_inner_cells, n_cells)`."""
        return range(self.n_inner_cells, self.n_cells)

    def branch_groups(self):
        """`[(branch, boundary-face positions), ...]` of `bc_branch`."""
        if self.bc_branch is None:
            return []
        b = np.asarray(self.bc_branch)
        return [(int(k), np.flatnonzero(b == k)) for k in np.unique(b)]


def _first_visitor(cell_faces):
    """Per face, the first inner cell and local face that define it, in the order of the BaseMesh
    geometry loops.
    """
    nfpc, n_inner = cell_faces.shape
    flat = cell_faces.T.ravel()                      # order ic*nfpc + lf
    faces, first = np.unique(flat, return_index=True)
    ic = first // nfpc
    lf = first % nfpc
    return faces, ic, lf


class _FitCheck:
    """Batched unisolvency test of the LSQ fits as `_lsq_plan` assembles them; at a wall a ghost
    informs only the pure monomials of its axis, every other row comes from the interior samples
    alone.
    """

    def __init__(self, dim, mon, centers, ghost_offsets, ghost_axis,
                 translations=None):
        """Store the monomials, cell centres, ghost offsets and axes, and optional periodic
        translations.
        """
        self.dim = dim
        self.powers = np.asarray(mon, dtype=int).reshape(len(mon), dim)
        self.centers = np.asarray(centers, dtype=float)
        self.ghost_offsets = np.asarray(ghost_offsets, dtype=float).reshape(-1, dim)
        self.ghost_axis = np.asarray(ghost_axis, dtype=int)
        self.translations = translations
        self.axis_rows = np.zeros((dim, len(mon)), dtype=bool)
        for k, mi in enumerate(mon):
            nz = [a for a in range(dim) if mi[a] != 0]
            if len(nz) == 1:
                self.axis_rows[nz[0], k] = True

    def offsets(self, cells, S):
        """Offsets `(n, w, dim)` of the samples `S` (-1: none, offset 0) from `cells`, to the
        nearest periodic image.
        """
        D = self.centers[np.maximum(S, 0)] - self.centers[cells][:, None, :]
        if self.translations is not None:
            D = D + _image_shifts(D, self.translations)
        return np.where((S >= 0)[..., None], D, 0.0)

    def _mono(self, D):
        """Monomial columns `(M, ...)` of the offsets `D` `(..., dim)`."""
        pw = [[np.ones(D.shape[:-1])] for _ in range(self.dim)]
        for a in range(self.dim):
            for _ in range(int(self.powers[:, a].max(initial=0))):
                pw[a].append(pw[a][-1] * D[..., a])
        cols = []
        for mi in self.powers:
            c = pw[0][mi[0]]
            for a in range(1, self.dim):
                c = c * pw[a][mi[a]]
            cols.append(c)
        return np.stack(cols)

    def __call__(self, cells, S, B):
        """Per cell, whether the fit from inner samples `S` and ghosts of boundary faces `B`
        (both -1 padded) is unisolvent.
        """
        n = len(cells)
        ok = np.ones(n, dtype=bool)
        if n == 0:
            return ok
        D = self.offsets(cells, S).transpose(1, 0, 2)          # (w, n, dim)
        r2 = sum(D[..., a] ** 2 for a in range(self.dim)).max(axis=0, initial=0.0)
        gh = np.flatnonzero((B >= 0).any(axis=1))
        G = np.where((B[gh] >= 0)[..., None],
                     self.ghost_offsets[np.maximum(B[gh], 0)], 0.0)
        r2[gh] = np.maximum(r2[gh], (G * G).sum(axis=-1).max(axis=1, initial=0.0))
        live = r2 > 0
        r = np.sqrt(np.where(live, r2, 1.0))
        Ci = self._mono(D / r[:, None])                       # (M, w, n)
        Cg = np.zeros((Ci.shape[0], B.shape[1], n))
        Cg[:, :, gh] = self._mono(G.transpose(1, 0, 2) / r[gh, None])
        if self.dim >= 2:                  # ghosts decoupled (REQ-174b)
            cpl = gh[live[gh]]
            plain = np.flatnonzero(live & ~np.isin(np.arange(n), gh))
            if cpl.size:
                ok[cpl] = self._coupled(Ci[:, :, cpl].transpose(2, 1, 0),
                                        Cg[:, :, cpl].transpose(2, 1, 0), B[cpl])
            if plain.size:
                ok[plain] = _full_rank_ok(Ci[:, :, plain])
        else:                              # a ghost is a sample like any other
            plain = np.flatnonzero(live)
            ok[plain] = _full_rank_ok(np.concatenate(
                [Ci[:, :, plain], Cg[:, :, plain]], axis=1))
        return ok

    def _coupled(self, Vi, Vg, B):
        """Unisolvency of the decoupled fits of cells with ghost samples, one test per ghost axis
        plus the interior-only fit.
        """
        n, w, M = Vi.shape
        b = B.shape[1]
        valid = B >= 0
        axis = np.where(valid, self.ghost_axis[np.maximum(B, 0)], -1)
        j = np.arange(b)
        later = ((axis[:, :, None] == axis[:, None, :])
                 & (j[None, None, :] > j[None, :, None])
                 & valid[:, None, :]).any(axis=2)
        own = self.axis_rows[np.maximum(axis, 0)] & (valid & ~later)[..., None]
        rows = np.concatenate([~own.any(axis=1)[:, None], own], axis=1)
        extra = np.concatenate([np.zeros((n, 1, M)), Vg], axis=1)    # (n, b+1, M)
        blocks = np.concatenate(
            [np.broadcast_to(Vi[:, None], (n, b + 1, w, M)), extra[:, :, None]],
            axis=2).reshape(n * (b + 1), w + 1, M)
        ok = _rows_ok(blocks, rows.reshape(n * (b + 1), M))
        return ok.reshape(n, b + 1).all(axis=1)


def _normal_derivative_columns(xp, X, nrm, powers):
    """`n . grad` of the monomial columns at offsets `X` along `nrm`, shape `(..., M)`."""
    cols = []
    for pw in powers:
        col = None
        for a, p in enumerate(pw):
            if p == 0:
                continue
            red = tuple(q - (b == a) for b, q in enumerate(pw))
            t = p * nrm[..., a] * _monomial_columns(xp, X, [red])[..., 0]
            col = t if col is None else col + t
        cols.append(col if col is not None else xp.zeros_like(X[..., 0]))
    return xp.stack(cols, axis=-1)


class _ClosureCheck:
    """Unisolvency including the boundary closure: the base fit plus, per closure class, the
    constrained fit (interior samples, ghost faces, Dirichlet value and Neumann normal-derivative
    rows) must have full column rank.
    """

    def __init__(self, base, kinds, foot, normal):
        """Store the base check, the closure `kinds` `(n_class, n_bf)`, foot offsets and outward
        normals.
        """
        self.base = base
        self.kinds = np.asarray(kinds)
        self.foot = np.asarray(foot, dtype=float)
        self.normal = np.asarray(normal, dtype=float)

    def offsets(self, cells, S):
        """Sample offsets of the base check."""
        return self.base.offsets(cells, S)

    def __call__(self, cells, S, B):
        """Per cell, whether the base fit and every constrained closure fit are unisolvent."""
        from zoomy_core.numerics.boundary_closure import GHOST
        ok = self.base(cells, S, B)
        for k in range(self.kinds.shape[0]):
            kb = np.where(B >= 0, self.kinds[k][np.maximum(B, 0)], -1)
            act = np.flatnonzero(ok & (kb > GHOST).any(axis=1))
            if act.size:
                ok[act] = self._constrained_ok(cells[act], S[act], B[act], kb[act])
        return ok

    def _constrained_ok(self, cells, S, B, kb):
        """Full-column-rank test of the constrained fits of cells `cells` for the face kinds
        `kb`.
        """
        from zoomy_core.numerics.boundary_closure import DIRICHLET, GHOST, NEUMANN
        base = self.base
        D = base.offsets(cells, S)                                # (n, w, dim)
        Bi = np.maximum(B, 0)
        F = np.where((B >= 0)[..., None], self.foot[Bi], 0.0)     # (n, b, dim)
        Nf = self.normal[Bi]
        r2 = np.maximum((D * D).sum(-1).max(axis=1, initial=0.0),
                        (4 * F * F).sum(-1).max(axis=1, initial=0.0))
        r = np.sqrt(np.where(r2 > 0, r2, 1.0))[:, None, None]
        Vi = base._mono(D / r)                                   # (M, n, w)
        Vg = np.where((kb == GHOST)[None], base._mono(2 * F / r), 0.0)
        Vd = np.where((kb == DIRICHLET)[None], base._mono(F / r), 0.0)
        Vn = np.where((kb == NEUMANN)[None],
                      np.moveaxis(_normal_derivative_columns(
                          np, F / r, Nf, [tuple(p) for p in base.powers]), -1, 0),
                      0.0)
        C = np.concatenate([Vi, Vg, Vd, Vn], axis=2)               # (M, n, rows)
        return _full_rank_ok(C.transpose(0, 2, 1))


def _full_rank_ok(C):
    """Whether every coefficient of the fits with monomial columns `C` `(M, R, n)` is determined
    (full column rank, conditioning cut `_LSQ_COND`), via an unrolled Cholesky.
    """
    M = C.shape[0]
    G = [[None] * M for _ in range(M)]
    for i in range(M):
        for j in range(i + 1):
            G[i][j] = (C[i] * C[j]).sum(axis=0)
    L = [[None] * M for _ in range(M)]
    for j in range(M):
        d = G[j][j] - sum((L[j][k] ** 2 for k in range(j)), np.zeros_like(G[j][j]))
        L[j][j] = np.sqrt(np.where(d > 0, d, np.nan))
        for i in range(j + 1, M):
            L[i][j] = (G[i][j] - sum((L[i][k] * L[j][k] for k in range(j)),
                                     np.zeros_like(d))) / L[j][j]
    X = [[None] * M for _ in range(M)]                        # X = L^-1
    for j in range(M):
        X[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, M):
            X[i][j] = -sum((L[i][k] * X[k][j] for k in range(j, i)),
                           np.zeros_like(d)) / L[i][i]
    norm = np.sqrt(sum((G[i][j] ** 2 * (1 if i == j else 2)
                        for i in range(M) for j in range(i + 1)), np.zeros_like(d)))
    ok = np.ones(C.shape[2], dtype=bool)
    for r in range(M):
        ok &= norm * sum((X[k][r] ** 2 for k in range(r, M)), np.zeros_like(d)) \
            <= _LSQ_COND ** -2
    return ok


def _rows_ok(V, rows):
    """Whether the coefficients `rows` of the possibly rank-deficient fits `V` `(n, R, M)` are
    determined; ill-conditioned directions count as null, as in `_lsq_matrices`.
    """
    G = np.einsum("nri,nrj->nij", V, V)
    lam, Q = np.linalg.eigh(G)
    live = lam > _LSQ_COND ** 2 * lam[:, -1:]
    M = V.shape[2]
    Qn = Q * ~live[:, None, :]                       # null-space eigenvectors
    resid = np.sqrt((Qn * Qn).sum(axis=2))           # |null part of e_r|
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(live, 1.0 / lam, 0.0)
    amp2 = np.sqrt((G * G).sum(axis=(1, 2)))[:, None] * np.einsum("nri,ni->nr", Q * Q, w)
    good = (resid <= _LSQ_RESID) & (amp2 <= _LSQ_COND ** -2)
    return (good | ~rows[:, :M]).all(axis=1)


def _compact(S):
    """Move the -1 entries of each row of `S` to the end (order kept) and drop trailing all -1
    columns.
    """
    S = np.take_along_axis(S, np.argsort(S < 0, axis=1, kind="stable"), axis=1)
    return S[:, :int((S >= 0).sum(axis=1).max(initial=0))]


def _pad(S, w):
    """Pad `S` with -1 columns to width `w`."""
    return np.concatenate([S, -np.ones((S.shape[0], w - S.shape[1]), dtype=S.dtype)],
                          axis=1) if S.shape[1] < w else S


def _next_layer(check, cells, S, nl, n_inner):
    """Face neighbours of `cells` and their stencils `S` not yet in `S`, nearest first, with
    distance-class numbers (equidistant cells share one).
    """
    n = len(cells)
    src = np.concatenate([cells[:, None], S], axis=1)
    nb = nl[np.maximum(src, 0)]
    nb = np.where((src >= 0)[..., None] & (nb >= 0) & (nb < n_inner), nb, -1)
    nb = nb.reshape(n, -1)
    row = np.arange(n)[:, None]
    have = np.concatenate([cells[:, None], S], axis=1)
    taken = np.isin(row * n_inner + nb, (row * n_inner + have)[have >= 0])
    nb = np.sort(np.where(taken | (nb < 0), -1, nb), axis=1)
    nb[:, 1:] = np.where(nb[:, 1:] == nb[:, :-1], -1, nb[:, 1:])
    d = np.where(nb >= 0, np.linalg.norm(check.offsets(cells, nb), axis=2), np.inf)
    o = np.argsort(d, axis=1, kind="stable")
    nb, d = np.take_along_axis(nb, o, axis=1), np.take_along_axis(d, o, axis=1)
    w = int((nb >= 0).sum(axis=1).max(initial=0))
    nb, d = nb[:, :w], d[:, :w]
    new = np.zeros(nb.shape, dtype=bool)
    new[:, 1:] = d[:, 1:] > d[:, :-1] * (1 + _LSQ_TIE)
    return nb, np.where(nb >= 0, np.cumsum(new, axis=1), w)


def _extend(check, cells, S, B, nl, n_inner):
    """Smallest extension of the stencils `S` that makes the fits unisolvent, one distance class
    at a time over at most `_LSQ_MAX_LAYERS` layers; returns the new rows and which cells
    succeeded.
    """
    out, done = [None] * len(cells), np.zeros(len(cells), dtype=bool)
    act = np.arange(len(cells))
    cur = S
    for _ in range(_LSQ_MAX_LAYERS):
        if not act.size:
            break
        cand, grp = _next_layer(check, cells[act], cur, nl, n_inner)
        if not cand.shape[1]:
            break
        left = np.ones(act.size, dtype=bool)
        for g in range(cand.shape[1]):
            sel = np.flatnonzero(left & (grp == g).any(axis=1))
            if not sel.size:
                continue
            trial = np.concatenate([cur[sel], np.where(grp[sel] <= g, cand[sel], -1)],
                                   axis=1)
            good = check(cells[act[sel]], trial, B[act[sel]])
            for i, t in zip(sel[good], _compact(trial[good])):
                out[act[i]] = t
            done[act[sel[good]]] = True
            left[sel[good]] = False
        act, cur = act[left], np.concatenate([cur[left], cand[left]], axis=1)
    return out, done


def _unisolvent_stencils(check, cells, S, B, nl, n_inner, fallback=None):
    """Stencils made unisolvent: kept where they already are, else replaced by `fallback` if that
    is, else extended.
    """
    ok = check(cells, S, B)
    if fallback is not None:
        bad = np.flatnonzero(~ok)
        w = max(S.shape[1], fallback.shape[1])
        S, fallback = _pad(S, w), _pad(fallback, w)
        S[bad] = fallback[bad]
        ok[bad] = check(cells[bad], S[bad], B[bad])
    bad = np.flatnonzero(~ok)
    rows = {}
    for k in range(0, bad.size, _LSQ_CHUNK):
        part = bad[k:k + _LSQ_CHUNK]
        grown, done = _extend(check, cells[part], _compact(S[part]), B[part],
                              nl, n_inner)
        rows.update({int(part[i]): grown[i] for i in np.flatnonzero(done)})
    if rows:
        S = _pad(S, max(len(t) for t in rows.values()))
        for i, t in rows.items():
            S[i] = -1
            S[i, :len(t)] = t
    return _compact(S)


def _face_stencils(nl, n_inner):
    """Inner face neighbours of each inner cell, -1 padded, in `cell_neighbors` order."""
    f = np.asarray(nl[:n_inner])
    return _compact(np.where((f >= 0) & (f < n_inner)
                             & (f != np.arange(n_inner)[:, None]), f, -1))


def _ghost_stencils(n_cells, nl, n_inner, required):
    """Ghost rows: the ghost's inner face neighbours, grown by whole layers to `required` samples."""
    out = {}
    for i_c in range(n_inner, n_cells):
        cur = [m for m in nl[i_c] if 0 <= m < n_inner]
        while len(cur) < required:
            grown = list(set(expand_neighbors(nl, cur, n_valid=n_inner)) - {i_c})
            if len(grown) <= len(cur):          # nothing more to reach
                break
            cur = grown
        out[i_c] = cur
    return out


def _boundary_slots(owner, n_cells, boundary_face_centers, centers):
    """Per cell its sampled boundary faces in boundary-face order (-1 padded) and each one's ghost
    axis.
    """
    valid = np.flatnonzero(owner >= 0)
    counts = np.bincount(owner[valid], minlength=n_cells)
    max_bdy = int(counts.max()) if valid.size else 0
    bdy = -np.ones((n_cells, max_bdy), dtype=int)
    n_axis = -np.ones((n_cells, max_bdy), dtype=int)
    if valid.size:
        o = valid[np.argsort(owner[valid], kind="stable")]
        first = np.concatenate([[0], np.cumsum(counts)])[owner[o]]
        slot = np.arange(len(o)) - first
        bdy[owner[o], slot] = o
        d = 2 * (np.asarray(boundary_face_centers)[o] - np.asarray(centers)[owner[o]])
        n_axis[owner[o], slot] = np.argmax(np.abs(d), axis=1)
    return bdy, n_axis


def _lsq_stencils(n_cells, dim, neighbors_list, centers, lsq_degree,
                  n_inner_cells, boundary_owner, cell_vertices,
                  boundary_face_centers, ghost_offsets, sampled=None,
                  translations=None, closure_kinds=None, normals=None):
    """LSQ stencil selection (neighbours, boundary-face slots, ghost axes) without the
    pseudo-inverse: the vertex ring (face neighbours in 1D and for degree-1 wall cells) kept if
    its fit is unisolvent, else extended by nearest face neighbours.  Periodic seams get no ghost
    sample, and boundary-closure kinds add the constrained-fit check.
    """
    owner = np.asarray(boundary_owner, dtype=int)
    if sampled is not None:
        owner = np.where(np.asarray(sampled, dtype=bool), owner, -1)
    nl = np.asarray(neighbors_list, dtype=int)
    mon = build_monomial_indices(lsq_degree, dim)
    degree = get_polynomial_degree(mon)
    required = get_required_monomials_count(degree, dim)
    bdy, n_axis = _boundary_slots(owner, n_cells, boundary_face_centers, centers)
    n = n_inner_cells
    ghost_axis = np.argmax(np.abs(np.asarray(boundary_face_centers)
                                  - np.asarray(centers)[np.maximum(owner, 0)]),
                           axis=1) if owner.size else np.zeros(0, dtype=int)
    check = _FitCheck(dim, mon, centers, ghost_offsets, ghost_axis, translations)
    if closure_kinds is not None and len(closure_kinds):
        check = _ClosureCheck(check, closure_kinds,
                              0.5 * np.asarray(ghost_offsets, dtype=float), normals)
    cells = np.arange(n)
    face = _face_stencils(nl, n)
    ring_mode = cell_vertices is not None and dim >= 2
    if ring_mode:
        cv = np.asarray(cell_vertices)[:, :n]
        k = cv.shape[0]
        C = sps.csr_matrix((np.ones(k * n, dtype=np.int32),
                            (np.repeat(np.arange(n), k), cv.T.reshape(-1))),
                           shape=(n, int(cv.max()) + 1))
        rows, cols = (C @ C.T).nonzero()
        off = rows != cols
        rows, cols = rows[off], cols[off]
        o = np.lexsort((cols, rows))                 # each ring sorted
        rows, cols = rows[o], cols[o]
        lens = np.bincount(rows, minlength=n)
        start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        ring = -np.ones((n, int(lens.max(initial=0))), dtype=int)
        ring[rows, np.arange(len(rows)) - start[rows]] = cols
        # A linear fit has no curvature term: on a boundary cell the vertex
        # ring (more cells inside than ghosts outside) turns curvature along
        # the wall into a wall-normal gradient.  Face neighbours + ghosts are
        # mirror-symmetric: the boundary stencil of a gradient.
        if degree == 1 and (bdy[:n] >= 0).any():
            w = max(ring.shape[1], face.shape[1])
            base = _pad(ring.copy(), w)
            wall = (bdy[:n] >= 0).any(axis=1)
            base[wall] = _pad(face, w)[wall]
            S = _unisolvent_stencils(check, cells, base, bdy[:n], nl, n,
                                     fallback=ring)
        else:
            S = _unisolvent_stencils(check, cells, ring, bdy[:n], nl, n)
        special = _ghost_stencils(n_cells, nl, n, required)
        max_nbr = max([S.shape[1]] + [len(v) for v in special.values()])
        neighbors = np.repeat(np.arange(n_cells)[:, None], max_nbr, axis=1)
        neighbors[:n, :S.shape[1]] = np.where(S >= 0, S, cells[:, None])
        for i_c, cur in special.items():
            neighbors[i_c, :min(len(cur), max_nbr)] = cur[:max_nbr]
    else:
        S = _unisolvent_stencils(check, cells, face, bdy[:n], nl, n)
        # the width of the INNER stencils: a ghost row (its inner cell's
        # neighbours, the inner cell itself included) is cut to it, never
        # widens the inner stencils (ANALYSIS 23); inner rows are padded
        # with the cell itself (a zero row of the fit)
        max_nbr = S.shape[1]
        neighbors = np.repeat(np.arange(n_cells)[:, None], max_nbr, axis=1)
        neighbors[:n] = np.where(S >= 0, S, cells[:, None])
        for i_c, cur in _ghost_stencils(n_cells, nl, n, required).items():
            while len(cur) < max_nbr:       # (legacy) ghost rows padded by growth
                new = [m for m in set(expand_neighbors(nl, cur, n_valid=n)) - {i_c}
                       if m not in cur and m < n]
                if not new:
                    break
                cur = cur + new
            cur = cur[:max_nbr]
            neighbors[i_c, :len(cur)] = cur
        if dim < 2:
            n_axis[:] = -1
    pure_axis_rows = {}
    for kk, mi in enumerate(mon):
        nz = [a for a in range(dim) if mi[a] != 0]
        if len(nz) == 1:
            pure_axis_rows.setdefault(nz[0], []).append(kk)
    return mon, neighbors, bdy, n_axis, pure_axis_rows


def _lsq_plan(topo_n_cells, dim, mon, neighbors, bdy, n_axis, pure_axis_rows,
              closure_kinds=None):
    """Static assembly plan of the batched pseudo-inverses: plain, interior-only (`bint`) and
    per-ghost pure-normal (`bone`) fits, zero rows for degenerate cells, and the closure cells
    and kinds.
    """
    n_cells, max_nbr = neighbors.shape
    max_bdy = bdy.shape[1]
    M = len(mon)
    n_bdy_here = (bdy >= 0).sum(axis=1)
    degenerate = ((neighbors == np.arange(n_cells)[:, None]).all(axis=1)
                  & (n_bdy_here == 0))
    coupled = (dim >= 2) & (n_bdy_here > 0) & ~degenerate
    plain = np.flatnonzero(~coupled & ~degenerate)
    bint = np.flatnonzero(coupled)
    pairs = []                     # (position in bint, ghost slot j, rows)
    for pos, c in enumerate(bint):
        for j in range(n_bdy_here[c]):
            rows = pure_axis_rows.get(int(n_axis[c, j]))
            if rows:
                pairs.append((pos, j, tuple(rows)))
    n_plain, n_bint, n_bone = len(plain), len(bint), len(pairs)
    zero_block = n_plain + n_bint + n_bone
    row_src = np.empty((n_cells, M), dtype=int)
    row_src[degenerate] = zero_block * M + np.arange(M)[None, :]
    row_src[plain] = (np.arange(n_plain)[:, None] * M + np.arange(M)[None, :])
    row_src[bint] = ((n_plain + np.arange(n_bint))[:, None] * M
                     + np.arange(M)[None, :])
    for p, (pos, j, rows) in enumerate(pairs):
        c = bint[pos]
        for r in rows:
            row_src[c, r] = (n_plain + n_bint + p) * M + r
    bdy_mask = bdy >= 0
    closure = None
    if closure_kinds is not None and len(closure_kinds):
        from zoomy_core.numerics.boundary_closure import GHOST
        K = np.asarray(closure_kinds)
        slot = np.where(bdy_mask[None], K[:, np.maximum(bdy, 0)], -1)
        active = (slot > GHOST).any(axis=2)                 # (n_class, n_cells)
        ccells = np.flatnonzero(active.any(axis=0))
        closure = {"cells": ccells, "kinds": slot[:, ccells].astype(np.int8),
                   "active": active[:, ccells]}
    return {
        "closure": closure,
        "plain": plain, "bint": bint,
        "bone_pos": np.array([p[0] for p in pairs], dtype=int),
        "bone_row": np.array([max_nbr + p[1] for p in pairs], dtype=int),
        "row_src": row_src,
        "bdy_idx": np.maximum(bdy, 0),
        "bdy_mask": bdy_mask.astype(float),
        "max_nbr": max_nbr, "n_mon": M,
        "powers": [tuple(int(p) for p in mi) for mi in mon],
    }


def _periodic_pairs(mesh, bcs, centers):
    """Per `Periodic` BC, the boundary-face positions of its faces and of their partners, matched
    by sorting both sides along the seam.
    """
    from zoomy_core.model.boundary_conditions import Periodic
    if bcs is None:
        return []
    owner = np.asarray(mesh.face_cells[0, mesh.boundary_face_face_indices])
    name_to_index = {v: i for i, v in
                     enumerate(mesh.boundary_conditions_sorted_names)}
    index_to_tag = {i: v for i, v in
                    enumerate(mesh.boundary_conditions_sorted_physical_tags)}
    ghosts = np.asarray(mesh.boundary_face_ghosts)
    ptags = np.asarray(mesh.boundary_face_physical_tags)
    pairs = []
    for bc in bcs.boundary_conditions_list:
        if type(bc) is not Periodic:
            continue
        from_tag = index_to_tag[name_to_index[bc.periodic_to_physical_tag]]
        to_tag = index_to_tag[name_to_index[bc.tag]]
        m_from = ptags == from_tag
        m_to = ptags == to_tag
        fc = centers[:, owner[m_from]]
        tc = centers[:, ghosts[m_to]]
        sig = [fc[d, :].max() - fc[d, :].min() for d in range(mesh.dimension)]
        order_sig = np.lexsort([sig])
        f_order = np.lexsort([fc[d, :] for d in order_sig])
        t_order = np.lexsort([tc[d, :] for d in order_sig])
        idx = np.arange(m_to.shape[0])
        pairs.append((idx[m_to][t_order], idx[m_from][f_order]))
    return pairs


def _periodic_partners(mesh, bcs, centers):
    """`boundary_face_cells` after `BaseMesh.resolve_periodic_bcs` without mutating the mesh,
    plus the periodic mask.
    """
    owner = np.asarray(mesh.face_cells[0, mesh.boundary_face_face_indices])
    out = owner.copy()
    periodic = np.zeros(owner.size, dtype=bool)
    for to_bf, from_bf in _periodic_pairs(mesh, bcs, centers):
        out[to_bf] = owner[from_bf]
        periodic[to_bf] = True
    return out, periodic


def periodic_vertices(mesh, bcs, centers=None):
    """Identify the vertices of periodic seams; returns `(canon, translations)` or None without
    `Periodic` BCs.

    Args:
        centers: cell centres `(3, n_cells)`, default computed from the mesh.
    Returns:
        `canon` `(n_vertices,)`, the smallest vertex id of each class of vertices coinciding
            modulo the periods, and `translations` `(k, dim)`, one period vector per seam.
    """
    if centers is None:
        centers = cell_face_geometry(mesh)["cell_centers"]
    pairs = _periodic_pairs(mesh, bcs, centers)
    if not pairs:
        return None
    dim = int(mesh.dimension)
    X = np.asarray(mesh.vertex_coordinates, dtype=float)[:dim].T
    topo, g0, _ = _base_topology(mesh)
    fv = topo["face_vertices"]                       # (nvf, n_faces)
    bfaces = topo["boundary_faces"]
    fcen = g0["face_centers"][:, :dim]
    parent = np.arange(len(X))

    def root(v):
        """Union-find root of vertex `v` with path halving."""
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    shifts = []
    for to_bf, from_bf in pairs:
        ft, ff = bfaces[to_bf], bfaces[from_bf]
        t = fcen[ff] - fcen[ft]                      # per face pair
        shifts.append(t.mean(axis=0))
        for a, b, d in zip(fv[:, ft].T, fv[:, ff].T, t):
            for v in a:                              # its image on the partner face
                w = b[int(np.argmin(np.linalg.norm(X[b] - (X[v] + d), axis=1)))]
                rv, rw = root(int(v)), root(int(w))
                parent[max(rv, rw)] = min(rv, rw)
    canon = np.array([root(v) for v in range(len(X))])
    T = []
    for t in shifts:                                 # one per seam (t and -t)
        if not any(np.allclose(t, u) or np.allclose(t, -u) for u in T):
            T.append(t)
    return canon, np.array(T)


def _image_shifts(D, translations):
    """Translation (a combination of +-`translations`, or none) taking each offset in `D` to its
    shortest image.
    """
    from itertools import product
    T = np.asarray(translations)
    S = np.array([np.asarray(c) @ T for c in product((-1, 0, 1), repeat=len(T))])
    norms = np.linalg.norm(D[..., None, :] + S, axis=-1)
    return S[np.argmin(norms, axis=-1)]


def _base_topology(mesh):
    """Cell and face topology of `mesh` without stencils, with the frozen orientation decisions
    and the reference geometry; returns `(topo_partial, g0, inradius_fallback)`.
    """
    mtype = str(mesh.type)
    if mtype not in _SUPPORTED:
        raise NotImplementedError(
            f"geometry_xp: cell type {mtype!r} is not supported "
            f"({', '.join(_SUPPORTED)})")
    dim = int(mesh.dimension)
    n_inner = int(mesh.n_inner_cells)
    n_cells = int(mesh.n_cells)
    n_faces = int(mesh.n_faces)
    n_bf = int(mesh.n_boundary_faces)
    cv = np.asarray(mesh.cell_vertices, dtype=int)[:, :n_inner]
    cf = np.asarray(mesh.cell_faces, dtype=int)[:, :n_inner]
    fcells = np.asarray(mesh.face_cells, dtype=int)
    X0 = np.asarray(mesh.vertex_coordinates, dtype=float)[:dim]
    local = _local_faces_for_type(mtype)
    nvf = len(local[0])

    faces, ic, lf = _first_visitor(cf)
    if faces.size != n_faces or not np.array_equal(faces, np.arange(n_faces)):
        raise ValueError("geometry_xp: every face must belong to an inner cell")
    fverts = cv[np.asarray(local)[lf].T, ic]               # (nvf, n_faces)

    bfaces = np.asarray(mesh.boundary_face_face_indices, dtype=int)[:n_bf]
    bowner = fcells[0, bfaces]
    bghost = fcells[1, bfaces]
    inner_mask = (fcells[0] < n_inner) & (fcells[1] < n_inner)
    interior = np.flatnonzero(inner_mask)
    other = np.where(fcells[1] < n_inner, fcells[1], fcells[0])

    # ── orientation decisions (frozen from the initial coordinates) ──
    if dim == 1:
        # 1-D normal = sign(c[cell1] - c[cell0]) — needs the centres, which
        # need no normal in 1-D: centres of inner cells + reflected ghosts.
        xc = np.zeros(n_cells)
        xc[:n_inner] = (X0[0, cv[0]] + X0[0, cv[1]]) / 2
        fcx = X0[0, fverts[0]]
        xc[bghost] = 2 * fcx[bfaces] - xc[bowner]
        sign = np.sign(xc[fcells[1]] - xc[fcells[0]])
    else:
        if dim == 2:
            a, b = fverts[0], fverts[1]
            e = X0[:, b] - X0[:, a]
            raw = np.stack([-e[1], e[0]])
        else:
            raw = _face_area_vector(np, X0, fverts)
        fc = X0[:, fverts[0]]
        for k in range(1, nvf):
            fc = fc + X0[:, fverts[k]]
        fc = fc / nvf
        cc = np.zeros((dim, n_inner))
        for k in range(cv.shape[0]):
            cc = cc + X0[:, cv[k]]
        cc = cc / cv.shape[0]
        own = fcells[0] == ic
        dots = np.einsum("df,df->f", raw, fc - cc[:, ic])
        sign = np.where(own & (dots < 0), -1.0, 1.0)

    topo_partial = dict(
        dim=dim, mesh_type=mtype, n_cells=n_cells, n_inner_cells=n_inner,
        n_faces=n_faces, n_vertices=int(mesh.n_vertices),
        n_boundary_faces=n_bf, cell_vertices=cv, cell_faces=cf,
        face_cells=fcells, face_vertices=fverts,
        face_sign=np.asarray(sign, dtype=float), face_first_cell=ic,
        boundary_faces=bfaces, boundary_owner=bowner, boundary_ghost=bghost,
        interior_faces=interior, interior_A=fcells[0, interior],
        interior_B=fcells[1, interior], face_other_inner=other,
    )
    # Reference geometry with numpy (for the remaining frozen decisions).
    pre = MeshTopology(
        **topo_partial, inradius_fallback=np.zeros(n_inner, dtype=bool),
        lsq_degree=0, lsq_monomial_multi_index=(),
        lsq_scale_factors=np.zeros(0), lsq_grad_cols=np.zeros(0, dtype=int),
        lsq_neighbors=np.zeros((n_cells, 0), dtype=int),
        lsq_boundary_face_neighbors=np.zeros((n_cells, 0), dtype=int))
    g0 = _cell_face_geometry(np, pre, X0)

    # inradius degeneracy (the reference's fallback branch)
    if dim >= 2:
        fcd = g0["face_centers"][:, :dim]
        nrm = g0["face_normals"][:dim]
        dist = np.abs(np.einsum("lcd,dlc->lc",
                                fcd[cf] - g0["cell_centers"][:dim, :n_inner].T[None],
                                nrm[:, cf]))
        fallback = dist.min(axis=0) <= 0
    else:
        fallback = np.zeros(n_inner, dtype=bool)
    if fallback.any():
        g0 = _cell_face_geometry(
            np, dataclasses.replace(pre, inradius_fallback=fallback), X0)
    return topo_partial, g0, fallback


def cell_face_geometry(mesh):
    """FVMMesh geometry of `mesh` (cell centres with ghost mirrors, volumes, inradii, face
    normals, volumes, centres), vectorised in numpy.
    """
    return _base_topology(mesh)[1]


def _closure_classes(closure, sampled, n_bf):
    """Deduplicate the closure kind rows into classes; faces without a ghost sample count as ghost
    faces and rows without Dirichlet or Neumann faces have no class.
    """
    from zoomy_core.numerics.boundary_closure import GHOST
    keys, rows = [], []
    for key, K in closure.items():
        K = np.where(np.asarray(sampled, dtype=bool)[None, :],
                     np.asarray(K).reshape(-1, n_bf), GHOST)
        keys.append((key, K.shape[0]))
        rows.append(K)
    if not rows:
        return np.zeros((0, n_bf), dtype=np.int8), {}
    stack = np.concatenate(rows, axis=0)
    live = (stack != GHOST).any(axis=1)
    classes, inv = (np.unique(stack[live], axis=0, return_inverse=True)
                    if live.any() else (np.zeros((0, n_bf), dtype=np.int8),
                                        np.zeros(0, dtype=int)))
    cls = -np.ones(stack.shape[0], dtype=int)
    cls[live] = np.asarray(inv).reshape(-1)
    out, k = {}, 0
    for key, n in keys:
        out[key] = tuple(int(c) for c in cls[k:k + n])
        k += n
    return classes.astype(np.int8), out


def build_topology(mesh, *, nsm=None, lsq_degree=None, bcs=None,
                   partners=None, periodic_ids=None, systems=(), closure=None):
    """Build the static topology and LSQ stencils of `mesh` once in numpy, freezing the geometric
    decisions (normal orientation, ghost axes, boundary routing by tag name, periodic partners).

    Args:
        nsm: supplies the LSQ degree and the boundary conditions; `lsq_degree` and `bcs`
            override them.
        partners: input cell of each boundary face for a mesh that cannot pair its own periodic
            faces (a distributed piece).
        periodic_ids: seam vertex identification of such a piece.
        closure: `{key: (n_rows, n_bf) kinds}` of the row spaces closed at the boundary
            (degree >= 2); default from `nsm` and `systems`.
    Returns:
        `MeshTopology`.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    if lsq_degree is None:
        lsq_degree = nsm.resolved_lsq_degree() if nsm is not None else 1
    if bcs is None and nsm is not None:
        bcs = getattr(nsm, "_bc_source", None)
    topo_partial, g0, fallback = _base_topology(mesh)
    t = topo_partial
    dim, n_inner, n_cells = t["dim"], t["n_inner_cells"], t["n_cells"]
    n_bf, cf, fcells = t["n_boundary_faces"], t["cell_faces"], t["face_cells"]
    bfaces, bowner, bghost = t["boundary_faces"], t["boundary_owner"], t["boundary_ghost"]
    interior = t["interior_faces"]

    # LSQ stencil selection (reference order) + batched-pinv plan
    centers0 = g0["cell_centers"]
    if hasattr(mesh, "_cell_centers") and getattr(mesh, "_cell_centers", None) is not None:
        centers0 = np.asarray(mesh._cell_centers, dtype=float)
    bfc0 = g0["face_centers"][bfaces, :dim]

    # Boundary routing by tag NAME + periodic partners
    from zoomy_core.model.boundary_conditions import boundary_face_branches
    tags = (list(bcs.list_sorted_function_names) if bcs is not None else None)
    branch = (np.asarray(boundary_face_branches(mesh, bcs), dtype=int)
              if n_bf else np.zeros(0, dtype=int))
    if partners is None:
        inputs, periodic = _periodic_partners(mesh, bcs, centers0)
        if periodic.any() and periodic_ids is None:
            periodic_ids = periodic_vertices(mesh, bcs, centers0)
    else:
        inputs = np.asarray(partners)[:n_bf].copy()
        periodic = inputs != bowner

    # A periodic seam is no boundary for the stencils: the partner is the
    # neighbour across it, the seam's vertices are identified, no ghost
    # sample.  Boundary faces per cell by OWNER (face_cells[0]) otherwise.
    nl = np.array(mesh.cell_neighbors, dtype=int)
    cv_ring = np.asarray(mesh.cell_vertices)
    if periodic.any():
        for k in np.flatnonzero(periodic):           # the face's slot
            nl[bowner[k], np.argmax(cf[:, bowner[k]] == bfaces[k])] = inputs[k]
        if periodic_ids is not None:
            cv_ring = np.asarray(periodic_ids[0])[cv_ring]
    if periodic.any() and periodic_ids is None:
        raise ValueError("periodic faces without the seams' vertex "
                         "identification (periodic_ids)")
    # ghost samples at the mirror image of the owner centre (_lsq_matrices)
    bn = g0["face_normals"][:dim, bfaces].T
    bn = bn / np.linalg.norm(bn, axis=1)[:, None]
    ghost_off = 2 * np.sum((bfc0 - centers0[:dim, bowner].T) * bn, axis=1)[:, None] * bn
    if int(lsq_degree) < 2:
        # a linear fit has no closure: its gradient is one constant, so a
        # constraint pins it to the WALL value (first order at the cell
        # centre), while the D5 ghost as a sample is exact for linear data
        # already and centred (Neumann on quads: second order; measured,
        # ANALYSIS 40)
        closure = {}
    elif closure is None and nsm is not None and n_bf:
        from zoomy_core.numerics.boundary_closure import closure_kinds
        closure = closure_kinds(nsm, systems, branch, bn.T)
    classes, closure_class = _closure_classes(closure or {}, ~periodic, n_bf)
    mon, nbr, bdy, n_axis, pure_rows = _lsq_stencils(
        n_cells, dim, nl, centers0[:dim].T, lsq_degree, n_inner, bowner,
        cv_ring, bfc0, ghost_off, sampled=~periodic,
        translations=periodic_ids[1] if periodic.any() else None,
        closure_kinds=classes, normals=bn)
    plan = _lsq_plan(n_cells, dim, mon, nbr, bdy, n_axis, pure_rows,
                     closure_kinds=classes)
    shift = None
    if periodic.any():
        D = centers0[:dim, nbr].transpose(1, 2, 0) - centers0[:dim].T[:, None, :]
        shift = _image_shifts(D, periodic_ids[1])
    grad_cols = np.asarray(find_derivative_indices(
        mon, [tuple(1 if k == d else 0 for k in range(dim))
              for d in range(dim)]), dtype=int)

    # TPFA masks
    if interior.size:
        dxi = centers0[:dim, fcells[1, interior]] - centers0[:dim, fcells[0, interior]]
        ok = np.sqrt((dxi ** 2).sum(axis=0)) >= _TPFA_MIN_DIST
    else:
        ok = np.zeros(0, dtype=bool)
    tf = interior[ok]
    dxb = centers0[:dim, bghost] - centers0[:dim, bowner]
    bvalid = np.sqrt((dxb ** 2).sum(axis=0)) >= _TPFA_MIN_DIST

    return MeshTopology(
        **topo_partial, inradius_fallback=fallback,
        lsq_degree=int(lsq_degree), lsq_monomial_multi_index=tuple(mon),
        lsq_scale_factors=np.asarray(scale_lsq_derivative(mon), dtype=float),
        lsq_grad_cols=grad_cols, lsq_neighbors=nbr,
        lsq_boundary_face_neighbors=bdy, lsq_plan=plan, lsq_shift=shift,
        lsq_closure_class=closure_class,
        tpfa_faces=tf, tpfa_A=fcells[0, tf], tpfa_B=fcells[1, tf],
        tpfa_boundary_valid=bvalid,
        bc_tags=tags, bc_branch=branch, bc_input_cells=inputs,
        bc_periodic=periodic)


# ═════════════════════════════════════════════════════════════════════════
# Geometry (xp, gather-only, differentiable in X)
# ═════════════════════════════════════════════════════════════════════════

def _xp_of(X):
    """Array namespace of `X` (numpy if it has none)."""
    ns = getattr(X, "__array_namespace__", None)
    if ns is None:
        return np
    return ns()


def _pad3_rows(xp, a, dim):
    """Pad `(dim, n)` to `(3, n)` with zero rows (FVMMesh convention)."""
    if dim == 3:
        return a
    z = xp.zeros((3 - dim,) + tuple(a.shape[1:]), dtype=a.dtype)
    return xp.concatenate([a, z], axis=0)


def _norm_rows(xp, v):
    """Euclidean norm over the leading axis, summing squares in row order."""
    acc = v[0] * v[0]
    for k in range(1, v.shape[0]):
        acc = acc + v[k] * v[k]
    return xp.sqrt(acc)


def _heron(xp, X, i, j, k):
    """Triangle area by Heron's formula over the vertices `i, j, k` in the operation order of the
    reference `_area_triangle_heron_formula`.
    """
    a = _norm_rows(xp, X[:, j] - X[:, i])
    b = _norm_rows(xp, X[:, k] - X[:, j])
    c = _norm_rows(xp, X[:, i] - X[:, k])
    s = (0.0 + a + b + c) / 2
    return xp.sqrt(s * (s - a) * (s - b) * (s - c))


def _cross(a, b):
    """Cross product of 3-vectors given as rows."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _face_area_vector(xp, X, fv):
    """Area vector (norm = area) of 3D faces: triangle 1/2 (x1-x0) x (x2-x0), quad 1/2 (x2-x0) x
    (x3-x1) (exact for planar quads).
    """
    if fv.shape[0] == 3:
        a, b = X[:, fv[1]] - X[:, fv[0]], X[:, fv[2]] - X[:, fv[0]]
    else:
        a, b = X[:, fv[2]] - X[:, fv[0]], X[:, fv[3]] - X[:, fv[1]]
    return xp.stack(_cross(a, b)) / 2


def _cell_volume_3d(xp, X, cv, local, cc):
    """Volume of convex 3D cells as tetrahedra from each fanned face triangle to the cell centre
    `cc`.
    """
    vol = None
    for lf in local:
        for k in range(1, len(lf) - 1):
            a = X[:, cv[lf[0]]] - cc
            b = X[:, cv[lf[k]]] - cc
            c = X[:, cv[lf[k + 1]]] - cc
            bc = _cross(b, c)
            t = xp.abs(a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]) / 6
            vol = t if vol is None else vol + t
    return vol


def _cell_face_geometry(xp, topo, X):
    """Cell and face geometry without LSQ (the FVMMesh fields), written against the array namespace
    `xp`.
    """
    dim = topo.dim
    n_inner, n_cells = topo.n_inner_cells, topo.n_cells
    cv = topo.cell_vertices
    fv = topo.face_vertices
    bfaces, bowner, bghost = (topo.boundary_faces, topo.boundary_owner,
                              topo.boundary_ghost)

    # face centres (owner-local vertex order)
    fc = X[:, fv[0]]
    for k in range(1, fv.shape[0]):
        fc = fc + X[:, fv[k]]
    fc = fc / fv.shape[0]                                    # (dim, n_faces)

    # inner cell centres
    cc_in = X[:, cv[0]]
    for k in range(1, cv.shape[0]):
        cc_in = cc_in + X[:, cv[k]]
    cc_in = cc_in / cv.shape[0]                              # (dim, n_inner)

    sign = xp.asarray(topo.face_sign, dtype=X.dtype)
    if dim == 1:
        normals = sign[None, :]
        fvol = xp.ones((topo.n_faces,), dtype=X.dtype)
        vol_in = _norm_rows(xp, X[:, cv[1]] - X[:, cv[0]])
        ghost_c = 2 * fc[:, bfaces] - cc_in[:, bowner]
    elif dim == 3:
        av = _face_area_vector(xp, X, fv)
        fvol = _norm_rows(xp, av)
        normals = av / fvol[None, :] * sign[None, :]
        vol_in = _cell_volume_3d(xp, X, cv, _local_faces_for_type(topo.mesh_type),
                                 cc_in)
    else:
        e = X[:, fv[1]] - X[:, fv[0]]
        length = _norm_rows(xp, e)
        normals = xp.stack([-e[1] / length, e[0] / length]) * sign[None, :]
        fvol = length
        if topo.mesh_type == "triangle":
            vol_in = _heron(xp, X, cv[0], cv[1], cv[2])
        else:   # quad = two Heron triangles (0,1,2) + (2,3,0)
            vol_in = (_heron(xp, X, cv[0], cv[1], cv[2])
                      + _heron(xp, X, cv[2], cv[3], cv[0]))
    if dim >= 2:
        fn = normals[:, bfaces]
        fn = fn / _norm_rows(xp, fn)[None, :]
        inner_c = cc_in[:, bowner]
        off = fc[:, bfaces] - inner_c
        proj = off[0] * fn[0]
        for k in range(1, dim):
            proj = proj + off[k] * fn[k]
        ghost_c = inner_c + 2 * proj[None, :] * fn

    # assemble (inner | ghosts ordered by ghost index) by one gather
    n_ghost = n_cells - n_inner
    order = np.empty(n_ghost, dtype=int)
    order[bghost - n_inner] = np.arange(bghost.size)
    cc = xp.concatenate([cc_in, ghost_c[:, order]], axis=1)  # (dim, n_cells)
    ghost_inner = np.empty(n_ghost, dtype=int)
    ghost_inner[bghost - n_inner] = bowner
    vol = xp.concatenate([vol_in, vol_in[ghost_inner]])

    # inradius
    if dim == 1:
        ir_in = vol_in / 2.0
    else:
        cf = topo.cell_faces
        dist = None
        for l in range(cf.shape[0]):
            f = cf[l]
            d = (fc[0, f] - cc_in[0]) * normals[0, f]
            for k in range(1, dim):
                d = d + (fc[k, f] - cc_in[k]) * normals[k, f]
            d = xp.abs(d)
            dist = d if dist is None else xp.minimum(dist, d)
        ir_in = dist
        if np.any(topo.inradius_fallback):
            # reference fallback: min over (face centre − cell centre)
            alt = None
            for l in range(cf.shape[0]):
                f = cf[l]
                for k in range(dim):
                    v = fc[k, f] - cc_in[k]
                    alt = v if alt is None else xp.minimum(alt, v)
            ir_in = xp.where(xp.asarray(topo.inradius_fallback), alt, ir_in)
    ir = xp.concatenate([ir_in, ir_in[ghost_inner]])

    return {
        "cell_centers": _pad3_rows(xp, cc, dim),             # (3, n_cells)
        "cell_volumes": vol,                                  # (n_cells,)
        "cell_inradius": ir,                                  # (n_cells,)
        "face_normals": _pad3_rows(xp, normals, dim),        # (3, n_faces)
        "face_volumes": fvol,                                 # (n_faces,)
        "face_centers": xp.transpose(_pad3_rows(xp, fc, dim)),  # (n_faces, 3)
    }


def _monomial_columns(xp, dX, powers):
    """Vandermonde columns `prod_a dX_a**p_a` without `x**0` nodes, whose derivative would be
    NaN at a zero offset.
    """
    cols = []
    for pw in powers:
        col = None
        for a, p in enumerate(pw):
            if p == 0:
                continue
            x = dX[..., a]
            # p >= 2 reproduces the reference ``cell_diffs ** powers``
            # rounding: numpy evaluates it with the scalar-exponent fast path
            # when the exponent vector has length 1 (1-D) and with the
            # elementwise ``pow`` loop otherwise (the two differ by an ulp).
            if p == 1:
                t = x
            elif dX.shape[-1] == 1:
                t = x ** p
            else:
                t = xp.power(x, xp.full(x.shape, p))
            col = t if col is None else col * t
        cols.append(col if col is not None else xp.ones_like(dX[..., 0]))
    return xp.stack(cols, axis=-1)


def _pinv_cut(xp, V):
    """`pinv(V)` with singular values below `_LSQ_COND` times the largest cut, evaluated as
    `pinv(V P V)` so the pinv derivative rule sees a constant-rank matrix.
    """
    P = xp.linalg.pinv(V, rtol=_LSQ_COND)
    return xp.linalg.pinv(V @ P @ V, rtol=_LSQ_COND)


def _lsq_matrices(xp, topo, cc, fc, nrm):
    """Batched LSQ derivative operators `(n_cells, R, n_mon)`, the `LSQMesh.lsq_gradQ` contract
    (`A_glob[i] = pinv(V_i).T`), with each ghost sample at the mirror image of the cell centre
    across its boundary face.
    """
    plan = topo.lsq_plan
    dim = topo.dim
    nbr = topo.lsq_neighbors
    bdy = plan["bdy_idx"]
    bmask = xp.asarray(plan["bdy_mask"], dtype=cc.dtype)
    ccT = xp.transpose(cc[:dim])                         # (n_cells, dim)
    bfc = fc[topo.boundary_faces, :dim]                  # (n_bf, dim)
    bn = xp.transpose(nrm[:dim, topo.boundary_faces])    # (n_bf, dim)
    bn = bn / xp.sqrt(xp.sum(bn * bn, axis=1))[:, None]
    dXc = ccT[nbr] - ccT[:, None, :]
    if topo.lsq_shift is not None:                       # periodic images
        dXc = dXc + xp.asarray(topo.lsq_shift, dtype=dXc.dtype)
    parts = [dXc]
    if bdy.shape[1]:
        r = bfc[bdy] - ccT[:, None, :]                   # cell -> face centre
        nb = bn[bdy]
        dXb = (2 * xp.sum(r * nb, axis=-1)[..., None] * nb) * bmask[..., None]
        parts.append(dXb)
    dX = xp.concatenate(parts, axis=1)                   # (n_cells, R, dim)
    V = _monomial_columns(xp, dX, plan["powers"])        # (n_cells, R, M)
    max_nbr = plan["max_nbr"]
    R = V.shape[1]
    rowmask = np.ones((topo.n_cells, R))
    if bdy.shape[1]:
        rowmask[:, max_nbr:] = plan["bdy_mask"]
    V = V * xp.asarray(rowmask, dtype=V.dtype)[..., None]
    M = plan["n_mon"]
    blocks = []
    if plan["plain"].size:
        blocks.append(xp.linalg.pinv(V[plan["plain"]]))       # (n, M, R)
    if plan["bint"].size:
        ghost_rows = np.zeros(R)
        ghost_rows[max_nbr:] = 1.0
        keep = xp.asarray(1.0 - ghost_rows, dtype=V.dtype)
        # The decoupled fits are rank deficient by design: pinv them with the
        # null space :func:`_rows_ok` accepted them with -- offsets scaled by
        # the stencil radius (as :class:`_FitCheck`), cut at ``_LSQ_COND``.
        # Without the cut a null direction lifted by jitter or by moving
        # nodes is inverted, and its 1/sigma leaks into the kept rows.
        # ``_pinv_cut`` differentiates the cut map exactly (to sigma**2).
        r2 = xp.max(xp.sum(dX[plan["bint"]] ** 2, axis=-1), axis=1)
        r = xp.sqrt(xp.where(r2 > 0, r2, 1.0))
        scale = r[:, None] ** xp.asarray(np.asarray(plan["powers"]).sum(axis=1),
                                         dtype=V.dtype)[None, :]   # (n, M)
        Vb = V[plan["bint"]] / scale[:, None, :]
        V_int = Vb * keep[None, :, None]
        blocks.append(_pinv_cut(xp, V_int) / scale[:, :, None])
        if plan["bone_pos"].size:
            onehot = np.zeros((plan["bone_pos"].size, R))
            onehot[np.arange(onehot.shape[0]), plan["bone_row"]] = 1.0
            oh = xp.asarray(onehot, dtype=V.dtype)
            V_one = (V_int[plan["bone_pos"]]
                     + Vb[plan["bone_pos"]] * oh[:, :, None])
            blocks.append(_pinv_cut(xp, V_one)
                          / scale[plan["bone_pos"]][:, :, None])
    blocks.append(xp.zeros((1, M, R), dtype=V.dtype))
    S = xp.concatenate(blocks, axis=0).reshape(-1, R)
    A = S[plan["row_src"]]                               # (n_cells, M, R)
    return xp.transpose(A, (0, 2, 1))                    # (n_cells, R, M)


def _lsq_closure_matrices(xp, topo, cc, fc, nrm, vol, A_base):
    """Constrained LSQ fits of the boundary closure, `(n_class, n_cc, R, M)` tables in the
    `lsq_gradQ` layout: minimise the residual over interior and ghost samples subject to one
    Dirichlet value or Neumann normal-derivative constraint per face, in null-space form.

    References:
        Lawson, Hanson 1995, https://doi.org/10.1137/1.9781611971217
    """
    plan = topo.lsq_plan
    cl = plan.get("closure")
    M, w = plan["n_mon"], plan["max_nbr"]
    b = plan["bdy_idx"].shape[1]
    if cl is None or not cl["cells"].size:
        return xp.zeros((0, 0, w + b, M), dtype=cc.dtype)
    from zoomy_core.numerics.boundary_closure import DIRICHLET, GHOST, NEUMANN
    dim = topo.dim
    cells, K = cl["cells"], cl["kinds"]                  # K (n_class, n_cc, b)
    n_class, n_cc = K.shape[:2]
    ccT = xp.transpose(cc[:dim])
    c0 = ccT[cells][:, None, :]
    dXc = ccT[topo.lsq_neighbors[cells]] - c0            # (n_cc, w, dim)
    if topo.lsq_shift is not None:
        dXc = dXc + xp.asarray(topo.lsq_shift[cells], dtype=dXc.dtype)
    bdy, bmask = plan["bdy_idx"][cells], plan["bdy_mask"][cells] > 0
    bn = xp.transpose(nrm[:dim, topo.boundary_faces])
    bn = bn / xp.sqrt(xp.sum(bn * bn, axis=1))[:, None]
    nb = bn[bdy]                                         # (n_cc, b, dim)
    dn = xp.sum((fc[topo.boundary_faces, :dim][bdy] - c0) * nb, axis=-1)
    dn = xp.where(xp.asarray(bmask), dn, 1.0)            # padded slots
    s = vol[cells] ** (1.0 / dim)                        # (n_cc,)
    foot = dn[..., None] * nb / s[:, None, None]
    powers = plan["powers"]
    Vc = _monomial_columns(xp, dXc / s[:, None, None], powers)   # (n_cc, w, M)
    Vg = _monomial_columns(xp, 2 * foot, powers)                 # (n_cc, b, M)
    Vd = _monomial_columns(xp, foot, powers)
    Vn = _normal_derivative_columns(xp, foot, nb, powers)
    dt = Vc.dtype
    isG = xp.asarray((K == GHOST)[..., None], dtype=dt)          # (k, n_cc, b, 1)
    isD = xp.asarray((K == DIRICHLET)[..., None], dtype=dt)
    isN = xp.asarray((K == NEUMANN)[..., None], dtype=dt)
    V = xp.concatenate([xp.broadcast_to(Vc[None], (n_class,) + Vc.shape),
                        isG * Vg[None]], axis=2)                 # (k, n_cc, w+b, M)
    C = isD * Vd[None] + isN * Vn[None]                          # (k, n_cc, b, M)
    N = n_class * n_cc
    V = xp.reshape(V, (N, w + b, M))
    C = xp.reshape(C, (N, b, M))
    I = xp.eye(M, dtype=dt)
    Cp = xp.linalg.pinv(C, rtol=_LSQ_PINV_RTOL)                  # (N, M, b)
    Qc = xp.matmul(Cp, C)                                        # onto row(C)
    # (V P)^+ as the first block of pinv([V P; C^+ C]): the projector rows
    # lift the null directions of V P (exact zeros where the constraints
    # fix every coefficient) to 1, so the relative cut-off sees O(1)
    X = xp.concatenate([xp.matmul(V, I - Qc), Qc], axis=1)
    Ab = xp.linalg.pinv(X, rtol=_LSQ_PINV_RTOL)[:, :, :w + b]   # (N, M, w+b)
    Bb = xp.matmul(I - xp.matmul(Ab, V), Cp)                     # (N, M, b)
    fac = (isD[..., 0] * 0.5
           + isN[..., 0] * (s[:, None] / (2.0 * dn))[None])      # (k, n_cc, b)
    fac = xp.reshape(fac, (N, 1, b))
    Tb = xp.where(xp.reshape(xp.asarray(K == GHOST), (N, 1, b)),
                  Ab[:, :, w:], Bb * fac)
    T = xp.concatenate([Ab[:, :, :w], Tb], axis=2)               # (N, M, R)
    deg = np.asarray([sum(p) for p in powers], dtype=float)
    unscale = s[:, None] ** (-deg)[None, :]                      # (n_cc, M)
    T = xp.reshape(T, (n_class, n_cc, M, w + b)) * unscale[None, :, :, None]
    T = xp.transpose(T, (0, 1, 3, 2))                            # (k, n_cc, R, M)
    return xp.where(xp.asarray(cl["active"])[..., None, None], T,
                    A_base[cells][None])


def compute_geometry(topo: MeshTopology, X, *, lsq=True):
    """All template geometry from the node coordinates `X` `(dim, n_vertices)`, numpy or
    jax.numpy as `X` decides; gather-only and differentiable in `X`.

    Args:
        lsq: also compute `lsq_gradQ` and the closure tables `lsq_closure`.
    Returns:
        dict with the FVMMesh/LSQMesh fields (cell centres with ghosts, volumes, inradii, face
            normals, volumes, centres), boundary distances, interior face vectors, TPFA
            coefficients and, with `lsq`, the LSQ operators.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    xp = _xp_of(X)
    dim = topo.dim
    X = X[:dim]
    g = _cell_face_geometry(xp, topo, X)
    cc = g["cell_centers"]
    fcT = g["face_centers"]                       # (n_faces, 3)
    fc = xp.transpose(fcT)                        # (3, n_faces)
    nrm = g["face_normals"]
    vol = g["cell_volumes"]
    ir = g["cell_inradius"]
    fcells = topo.face_cells

    g["face_inradius"] = xp.minimum(ir[fcells[0]], ir[topo.face_other_inner])
    g["ghost_centers"] = cc[:, topo.boundary_ghost]
    bf = topo.boundary_faces
    rI = fc[:dim, bf] - cc[:dim, topo.boundary_owner]
    g["r_I"] = rI
    g["d_face"] = _norm_rows(xp, rI)
    # NORMAL distance owner centre → boundary face, (x_f − x_c)·n: the
    # ``distance`` the ghost-state kernels receive.  The LSQ virtual point
    # sits at x_c + 2(x_f − x_c), whose NORMAL offset is 2·d_normal, so a
    # prescribed normal gradient g has the exact ghost Q + 2·d_normal·g even
    # on skewed boundary cells (= d_face on orthogonal ones).
    nb = nrm[:dim, bf]
    dn = rI[0] * nb[0]
    for k in range(1, dim):
        dn = dn + rI[k] * nb[k]
    g["d_normal"] = dn
    it = topo.interior_faces
    g["r_A"] = fc[:dim, it] - cc[:dim, topo.interior_A]
    g["r_B"] = fc[:dim, it] - cc[:dim, topo.interior_B]

    # TPFA (DiffusionOperatorV2 geometry, nu-free)
    def _tpfa(faces, a, b, valid=None):
        """Distance and floored obliquity factor `max(|n.e|, _TPFA_MIN_COS)` between cells `a`
        and `b` across `faces`.

        References:
            Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
        """
        dx = cc[:dim, b] - cc[:dim, a]
        dist = _norm_rows(xp, dx)
        if valid is not None:      # degenerate faces: finite, masked below
            dist = xp.where(valid, dist, 1.0)
        ne = nrm[0, faces] * (dx[0] / dist)
        for k in range(1, dim):
            ne = ne + nrm[k, faces] * (dx[k] / dist)
        cosf = xp.maximum(xp.abs(ne), _TPFA_MIN_COS)
        return dist, cosf
    tf = topo.tpfa_faces
    if tf.size:
        dist, cosf = _tpfa(tf, topo.tpfa_A, topo.tpfa_B)
        gf = g["face_volumes"][tf] / dist / cosf
        g["tpfa_gA"] = gf / vol[topo.tpfa_A]
        g["tpfa_gB"] = gf / vol[topo.tpfa_B]
    else:
        g["tpfa_gA"] = xp.zeros((0,), dtype=X.dtype)
        g["tpfa_gB"] = xp.zeros((0,), dtype=X.dtype)
    if bf.size:
        valid = xp.asarray(topo.tpfa_boundary_valid)
        dist_b, cos_b = _tpfa(bf, topo.boundary_owner, topo.boundary_ghost,
                              valid)
        g["tpfa_bnd_g"] = xp.where(
            valid, g["face_volumes"][bf] / cos_b / vol[topo.boundary_owner], 0.0)
        g["tpfa_bnd_dist"] = dist_b
    else:
        g["tpfa_bnd_g"] = xp.zeros((0,), dtype=X.dtype)
        g["tpfa_bnd_dist"] = xp.zeros((0,), dtype=X.dtype)

    if lsq:
        g["lsq_gradQ"] = _lsq_matrices(xp, topo, cc, fcT, nrm)
        g["lsq_closure"] = _lsq_closure_matrices(xp, topo, cc, fcT, nrm, vol,
                                                 g["lsq_gradQ"])
    return g


def lsq_closure_derivatives(topo: MeshTopology, table, U, U_ghost, classes):
    """Boundary-closure LSQ derivatives for the rows of `U` that have a closure class, at the
    closure cells.

    Args:
        U_ghost: ghost values at the boundary-face virtual points.
        table: class tables `geom["lsq_closure"]`.
        classes: closure class per row of `U` (-1: none).
    Returns:
        `(rows, D)` with `D` `(len(rows), n_cc, n_mon)` scaled derivatives, or
        `([], None)` without closure.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    xp = _xp_of(U)
    cl = topo.lsq_plan.get("closure")
    rows = [v for v, k in enumerate(classes) if k >= 0]
    if cl is None or not rows or not cl["cells"].size:
        return [], None
    cells = cl["cells"]
    r = np.asarray(rows)
    nbr = topo.lsq_neighbors[cells]
    bdy = topo.lsq_boundary_face_neighbors[cells]
    Ur = U[r]
    uc = Ur[:, cells][:, :, None]
    parts = [Ur[:, nbr] - uc]
    if bdy.shape[1]:
        ub = U_ghost[r][:, np.maximum(bdy, 0)]
        parts.append(xp.where(xp.asarray(bdy >= 0)[None], ub - uc, 0.0))
    delta = xp.concatenate(parts, axis=2)                # (n_r, n_cc, R)
    T = table[np.asarray([classes[v] for v in rows])]    # (n_r, n_cc, R, M)
    coeffs = xp.einsum("vcr,vcrm->vcm", delta, T)
    return rows, xp.asarray(topo.lsq_scale_factors, dtype=coeffs.dtype) * coeffs


def lsq_derivatives(topo: MeshTopology, geom, U, U_ghost, classes=None):
    """Batched LSQ derivatives of the fields `U` `(n_v, n_cells)` using ghost values at the
    boundary-face virtual points, rows with a closure class taking the constrained fit.

    Args:
        U_ghost: `(n_v, n_bf)` ghost values; `None`: the cell value (extrapolation).
        classes: closure class per row of `U` (-1: none).
    Returns:
        `(n_v, n_cells, n_mon)` derivatives scaled by `lsq_scale_factors`.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    xp = _xp_of(U)
    A = geom["lsq_gradQ"]
    n = U.shape[1]
    nbr = topo.lsq_neighbors[:n]
    bdy = topo.lsq_boundary_face_neighbors[:n]
    U_self = U
    delta_c = U[:, nbr] - U_self[:, :, None]
    parts = [delta_c]
    if bdy.shape[1] and U_ghost is None:     # no samples: the cell value, a zero difference
        parts.append(xp.zeros(tuple(delta_c.shape[:2]) + (bdy.shape[1],), dtype=delta_c.dtype))
    elif bdy.shape[1]:
        ub = U_ghost[:, np.maximum(bdy, 0)]
        mask = xp.asarray(bdy >= 0)
        parts.append(xp.where(mask[None], ub - U_self[:, :, None], 0.0))
    delta = xp.concatenate(parts, axis=2)                # (n_v, n, R)
    coeffs = xp.matmul(xp.transpose(A[:n], (0, 2, 1))[None],
                       delta[..., None])[..., 0]          # (n_v, n, M)
    D = xp.asarray(topo.lsq_scale_factors, dtype=coeffs.dtype) * coeffs
    if classes is None:
        return D
    rows, Dc = lsq_closure_derivatives(topo, geom["lsq_closure"], U, U_ghost,
                                       classes)
    if not rows:
        return D
    cells = topo.lsq_plan["closure"]["cells"]
    if xp is np:
        D = D.copy()
        D[np.ix_(rows, cells)] = Dc
        return D
    return D.at[np.asarray(rows)[:, None], cells[None, :]].set(Dc)


def lsq_gradient(topo: MeshTopology, geom, U, U_ghost, classes=None):
    """First derivatives `(n_v, dim, n_cells)` from `lsq_derivatives`.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    xp = _xp_of(U)
    d = lsq_derivatives(topo, geom, U, U_ghost, classes)
    return xp.transpose(d[:, :, topo.lsq_grad_cols], (0, 2, 1))


__all__ = ["MeshTopology", "build_topology", "compute_geometry", "periodic_vertices",
           "lsq_derivatives", "lsq_gradient", "lsq_closure_derivatives"]
