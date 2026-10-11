"""Single-source mesh geometry for the solver template (``prepare_mesh``).

Two halves, split by what may change:

* :func:`build_topology` — STATIC integer topology, built ONCE with numpy from
  a :class:`~zoomy_core.mesh.base_mesh.BaseMesh` (or FVMMesh / LSQMesh):
  owner-oriented face vertices, orientation signs, ghost ranges, the
  interior / boundary face split, the LSQ stencil selection (incl. the
  REQ-174b ghost-decoupling decisions), TPFA valid masks and the boundary
  routing (branch per face by tag NAME, periodic partner cells).  Every
  geometric DECISION (which way a normal points, which axis a ghost aligns
  with, whether a face is degenerate) is taken here, from the initial
  coordinates, and frozen.
* :func:`compute_geometry` — GATHER-ONLY, vectorised geometry of the node
  coordinates ``X`` (``(dim, n_vertices)``), written once against the array
  namespace ``xp = X.__array_namespace__()``: numpy and ``jax.numpy`` run the
  SAME code, and ``jax.grad`` w.r.t. ``X`` goes straight through (no
  in-place writes, no data-dependent branches, no ``0**0`` power nodes).

``compute_geometry`` reproduces :class:`FVMMesh` / :class:`LSQMesh`
(``cell_centers`` incl. ghosts, ``cell_volumes``, ``cell_inradius``,
``face_normals``, ``face_volumes``, ``face_centers``, ``lsq_gradQ``) to
round-off.  Supported cell types: ``line`` (1-D), ``triangle`` and ``quad``
(2-D) — the horizontal meshes the template targets.

This module never imports jax.
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
    build_vertex_to_cells,
    expand_neighbors,
    find_derivative_indices,
    get_polynomial_degree,
    get_required_monomials_count,
    scale_lsq_derivative,
    vertex_one_ring,
)

_SUPPORTED = {"line": 1, "triangle": 2, "quad": 2}

#: Floor of the TPFA |n·e| obliquity factor (DiffusionOperatorV2 convention).
_TPFA_MIN_COS = 0.1
#: Degenerate-distance threshold of the TPFA faces (DiffusionOperatorV2).
_TPFA_MIN_DIST = 1e-30


# ═════════════════════════════════════════════════════════════════════════
# Topology (numpy, static)
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class MeshTopology:
    """Static integer topology + frozen geometric decisions (numpy)."""

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
        """Ghost cells are ``range(n_inner_cells, n_cells)``."""
        return range(self.n_inner_cells, self.n_cells)

    def branch_groups(self):
        """``[(branch, boundary-face positions), …]`` of :attr:`bc_branch`."""
        if self.bc_branch is None:
            return []
        b = np.asarray(self.bc_branch)
        return [(int(k), np.flatnonzero(b == k)) for k in np.unique(b)]


def _first_visitor(cell_faces):
    """Per face: the FIRST (inner cell, local face) defining it, in the
    (cell, local face) order every BaseMesh geometry loop runs."""
    nfpc, n_inner = cell_faces.shape
    flat = cell_faces.T.ravel()                      # order ic*nfpc + lf
    faces, first = np.unique(flat, return_index=True)
    ic = first // nfpc
    lf = first % nfpc
    return faces, ic, lf


def _lsq_stencils(n_cells, dim, neighbors_list, centers, lsq_degree,
                  n_inner_cells, boundary_owner, cell_vertices,
                  boundary_face_centers):
    """The STENCIL SELECTION of
    :func:`zoomy_core.mesh.lsq_reconstruction.least_squares_reconstruction_local`
    (same neighbour order, same padding, same REQ-174b axis decisions) --
    without the per-cell pseudo-inverse, which :func:`compute_geometry`
    batches.  ``centers`` is ``(n_cells, dim)``; ``boundary_owner`` the inner
    cell of every boundary face.  >= 2-D: the sorted vertex ring of every
    inner cell from one sparse product; per-cell loops only for ghosts and
    cells whose ring is too small (and in 1-D)."""
    owner = np.asarray(boundary_owner, dtype=int)
    if cell_vertices is None or dim < 2:
        cell_bf = [[] for _ in range(n_cells)]
        for i_bf, c in enumerate(owner):
            cell_bf[int(c)].append(i_bf)
        return _lsq_stencils_loop(n_cells, dim, neighbors_list, centers,
                                  lsq_degree, n_inner_cells, cell_bf,
                                  cell_vertices, boundary_face_centers)
    mon = build_monomial_indices(lsq_degree, dim)
    required = get_required_monomials_count(get_polynomial_degree(mon), dim)
    cv = np.asarray(cell_vertices)[:, :n_inner_cells]
    k, n = cv.shape
    C = sps.csr_matrix((np.ones(k * n, dtype=np.int32),
                        (np.repeat(np.arange(n), k), cv.T.reshape(-1))),
                       shape=(n, int(cv.max()) + 1))
    rows, cols = (C @ C.T).nonzero()
    off = rows != cols
    rows, cols = rows[off], cols[off]
    o = np.lexsort((cols, rows))                     # each ring sorted
    rows, cols = rows[o], cols[o]
    lens = np.bincount(rows, minlength=n)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    special = {}                                     # cell -> stencil list
    # Extend a ring only when its samples cannot fix the fit: the fit is of
    # differences to the cell (no constant), and every boundary face adds a
    # ghost sample.  (Counting neither extended every wall and corner cell
    # one-sidedly into the domain: wall-normal derivatives of fields constant
    # along the wall.)
    n_bf = np.bincount(owner, minlength=n)[:n] if owner.size else 0
    for i_c in np.flatnonzero(lens + n_bf < required - 1):   # degenerate cells
        cur = cols[start[i_c]:start[i_c] + lens[i_c]].tolist()
        fill = [m for m in neighbors_list[i_c]
                if m < n_inner_cells and m != i_c and m not in cur]
        cur = cur + fill
        while len(cur) < required:
            ext = expand_neighbors(neighbors_list, cur, n_valid=n_inner_cells)
            new = [m for m in ext if m != i_c and m not in cur]
            if not new:
                break
            cur.extend(new)
        special[int(i_c)] = cur
    # A linear fit has no curvature term: on a boundary cell the vertex ring
    # (more cells inside than ghosts outside) turns curvature along the wall
    # into a wall-normal gradient.  Face neighbours + ghosts are mirror-
    # symmetric: the boundary stencil of a gradient.
    if get_polynomial_degree(mon) == 1 and owner.size:
        for i_c in np.flatnonzero(n_bf > 0):
            face = [int(m) for m in neighbors_list[i_c]
                    if 0 <= m < n_inner_cells and m != i_c]
            if int(i_c) not in special and len(face) + n_bf[i_c] >= required - 1:
                special[int(i_c)] = face
    for i_c in range(n_inner_cells, n_cells):        # ghosts: face neighbours
        cur = [m for m in neighbors_list[i_c] if m < n_inner_cells]
        while len(cur) < required:
            new_nb = expand_neighbors(neighbors_list, cur, n_valid=n_inner_cells)
            cur = list(set(new_nb) - {i_c})
            if len(cur) == 0:
                break
        special[i_c] = cur
    max_nbr = max([int(lens.max()) if n else 0]
                  + [len(v) for v in special.values()])
    neighbors = np.repeat(np.arange(n_cells)[:, None], max_nbr, axis=1)
    neighbors[rows, np.arange(len(rows)) - start[rows]] = cols
    for i_c, cur in special.items():
        neighbors[i_c] = np.arange(max_nbr) * 0 + i_c
        neighbors[i_c, :min(len(cur), max_nbr)] = cur[:max_nbr]
    counts = np.bincount(owner, minlength=n_cells)
    max_bdy = int(counts.max()) if owner.size else 0
    bdy = -np.ones((n_cells, max_bdy), dtype=int)
    n_axis = -np.ones((n_cells, max_bdy), dtype=int)
    if owner.size:
        o = np.argsort(owner, kind="stable")         # in boundary-face order
        first = np.concatenate([[0], np.cumsum(counts)])[owner[o]]
        slot = np.arange(len(o)) - first
        bdy[owner[o], slot] = o
        d = 2 * (np.asarray(boundary_face_centers)[o] - np.asarray(centers)[owner[o]])
        n_axis[owner[o], slot] = np.argmax(np.abs(d), axis=1)
    pure_axis_rows = {}
    for kk, mi in enumerate(mon):
        nz = [a for a in range(dim) if mi[a] != 0]
        if len(nz) == 1:
            pure_axis_rows.setdefault(nz[0], []).append(kk)
    return mon, neighbors, bdy, n_axis, pure_axis_rows


def _lsq_stencils_loop(n_cells, dim, neighbors_list, centers, lsq_degree,
                       n_inner_cells, cell_boundary_faces, cell_vertices,
                       boundary_face_centers):
    """The STENCIL SELECTION of
    :func:`zoomy_core.mesh.lsq_reconstruction.least_squares_reconstruction_local`
    (same neighbour order, same padding, same REQ-174b axis decisions) —
    without the per-cell pseudo-inverse, which :func:`compute_geometry`
    batches.  ``centers`` is ``(n_cells, dim)``."""
    has_bdy = any(len(b) > 0 for b in cell_boundary_faces)
    max_bdy = max(len(b) for b in cell_boundary_faces) if has_bdy else 0
    mon = build_monomial_indices(lsq_degree, dim)
    degree = get_polynomial_degree(mon)
    required = get_required_monomials_count(degree, dim)
    use_vring = cell_vertices is not None and dim >= 2
    if use_vring:
        cell_vertices = np.asarray(cell_vertices)
        n_vring = min(n_inner_cells, cell_vertices.shape[1])
        v2c = build_vertex_to_cells(cell_vertices, n_vring)

    nbrs_all = []
    for i_c in range(n_cells):
        if use_vring and i_c < n_vring:
            cur = vertex_one_ring(i_c, cell_vertices, v2c, n_inner_cells)
            if len(cur) < required:
                fill = [n for n in neighbors_list[i_c]
                        if n < n_inner_cells and n != i_c and n not in cur]
                cur = cur + fill
                while len(cur) < required:
                    ext = expand_neighbors(neighbors_list, cur,
                                           n_valid=n_inner_cells)
                    new = [n for n in ext if n != i_c and n not in cur]
                    if not new:
                        break
                    cur.extend(new)
        else:
            cur = [n for n in neighbors_list[i_c] if n < n_inner_cells]
            while len(cur) < required:
                new_nb = expand_neighbors(neighbors_list, cur,
                                          n_valid=n_inner_cells)
                cur = list(set(new_nb) - {i_c})
                if len(cur) == 0:
                    break
        nbrs_all.append(cur)

    # the width of the INNER stencils: a ghost row (its inner cell's
    # neighbours, the inner cell itself included) is cut to it, never widens
    # the interior stencils one-sidedly
    max_nbr = max(len(n) for n in nbrs_all[:n_inner_cells])
    neighbors = np.empty((n_cells, max_nbr), dtype=int)
    bdy = -np.ones((n_cells, max_bdy), dtype=int)
    pure_axis_rows = {}
    for k, mi in enumerate(mon):
        nz = [a for a in range(dim) if mi[a] != 0]
        if len(nz) == 1:
            pure_axis_rows.setdefault(nz[0], []).append(k)
    n_axis = -np.ones((n_cells, max_bdy), dtype=int)

    for i_c in range(n_cells):
        cur = list(nbrs_all[i_c])
        n_nbr = len(cur)
        if not use_vring:
            while n_nbr < max_nbr:
                ext = expand_neighbors(neighbors_list, cur,
                                       n_valid=n_inner_cells)
                ext = list(set(ext) - {i_c})
                new = [n for n in ext if n not in cur and n < n_inner_cells]
                cur.extend(new)
                n_nbr = len(cur)
                if n_nbr >= max_nbr:
                    break
        if len(cur) >= max_nbr:
            trimmed = cur[:max_nbr]
        else:
            trimmed = cur + [i_c] * (max_nbr - len(cur))
        neighbors[i_c] = trimmed
        bf = list(cell_boundary_faces[i_c])
        bdy[i_c, :len(bf)] = bf
        if dim >= 2:
            for j, f in enumerate(bf):
                d = 2 * (boundary_face_centers[f] - centers[i_c])
                n_axis[i_c, j] = int(np.argmax(np.abs(d)))
    return mon, neighbors, bdy, n_axis, pure_axis_rows


def _lsq_plan(topo_n_cells, dim, mon, neighbors, bdy, n_axis, pure_axis_rows):
    """Static assembly plan of the batched pseudo-inverses.

    Groups (each ONE batched ``pinv``):

    * ``plain``  — cells fitted from the full stencil (1-D, or no boundary
      face): ``A = pinv(V)``;
    * ``bint``   — ≥2-D boundary cells: the interior-only fit
      ``A = pinv(V_int)`` (every ghost row zeroed, REQ-174b) …
    * ``bone``   — … whose PURE-normal rows along each ghost's axis are taken
      from ``pinv(V_int + that ghost row)``;
    * degenerate cells (no off-centre sample at all) get the exact zero the
      reference ``pinv(0)`` returns, WITHOUT a pinv (finite gradients).

    ``row_src`` maps every ``(cell, monomial row)`` to its row in the stacked
    ``[plain | bint | bone | zero]`` result, so assembly is one gather."""
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
    return {
        "plain": plain, "bint": bint,
        "bone_pos": np.array([p[0] for p in pairs], dtype=int),
        "bone_row": np.array([max_nbr + p[1] for p in pairs], dtype=int),
        "row_src": row_src,
        "bdy_idx": np.maximum(bdy, 0),
        "bdy_mask": bdy_mask.astype(float),
        "max_nbr": max_nbr, "n_mon": M,
        "powers": [tuple(int(p) for p in mi) for mi in mon],
    }


def _periodic_partners(mesh, bcs, centers):
    """``boundary_face_cells`` after :meth:`BaseMesh.resolve_periodic_bcs`
    WITHOUT mutating the mesh: the partner inner cell for every face of a
    ``Periodic`` tag, the owner elsewhere.  ``centers`` is ``(3, n_cells)``."""
    from zoomy_core.model.boundary_conditions import Periodic
    owner = np.asarray(mesh.face_cells[0, mesh.boundary_face_face_indices])
    out = owner.copy()
    periodic = np.zeros(owner.size, dtype=bool)
    if bcs is None:
        return out, periodic
    name_to_index = {v: i for i, v in
                     enumerate(mesh.boundary_conditions_sorted_names)}
    index_to_tag = {i: v for i, v in
                    enumerate(mesh.boundary_conditions_sorted_physical_tags)}
    ghosts = np.asarray(mesh.boundary_face_ghosts)
    ptags = np.asarray(mesh.boundary_face_physical_tags)
    for bc in bcs.boundary_conditions_list:
        if type(bc) is not Periodic:
            continue
        from_tag = index_to_tag[name_to_index[bc.periodic_to_physical_tag]]
        to_tag = index_to_tag[name_to_index[bc.tag]]
        m_from = ptags == from_tag
        m_to = ptags == to_tag
        from_cells = owner[m_from]
        to_cells = ghosts[m_to]
        fc = centers[:, from_cells]
        tc = centers[:, to_cells]
        sig = [fc[d, :].max() - fc[d, :].min() for d in range(mesh.dimension)]
        order_sig = np.lexsort([sig])
        f_order = np.lexsort([fc[d, :] for d in order_sig])
        t_order = np.lexsort([tc[d, :] for d in order_sig])
        idx = np.arange(m_to.shape[0])
        to_sorted = idx[m_to][t_order]
        from_sorted = idx[m_from][f_order]
        out[to_sorted] = owner[from_sorted]
        periodic[to_sorted] = True
    return out, periodic


def _base_topology(mesh):
    """Cell/face topology of ``mesh`` (no stencils) and its geometry from the
    current vertex coordinates (numpy): ``(topo_partial, g0)``."""
    mtype = str(mesh.type)
    if mtype not in _SUPPORTED:
        raise NotImplementedError(
            f"geometry_xp: cell type {mtype!r} is not supported (line, "
            "triangle, quad — the horizontal meshes of the solver template)")
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
        a, b = fverts[0], fverts[1]
        e = X0[:, b] - X0[:, a]
        raw = np.stack([-e[1], e[0]])
        fc = (X0[:, a] + X0[:, b]) / 2
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
    """The FVMMesh geometry of ``mesh`` (numpy, vectorised): cell centres
    (incl. ghost mirrors), volumes, inradius; face normals, volumes, centres."""
    return _base_topology(mesh)[1]


def build_topology(mesh, *, nsm=None, lsq_degree=None, bcs=None):
    """Build the STATIC topology of ``mesh`` (numpy, once).

    ``nsm`` (optional) supplies the LSQ degree (``resolved_lsq_degree``) and
    the ``BoundaryConditions`` container (routing by tag name, periodic
    partners); ``lsq_degree`` / ``bcs`` override them."""
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
    # boundary faces per cell by OWNER (face_cells[0]) — a periodic remap of
    # ``boundary_face_cells`` must not move a boundary sample to the partner.
    mon, nbr, bdy, n_axis, pure_rows = _lsq_stencils(
        n_cells, dim, np.asarray(mesh.cell_neighbors), centers0[:dim].T,
        lsq_degree, n_inner, bowner, np.asarray(mesh.cell_vertices), bfc0)
    plan = _lsq_plan(n_cells, dim, mon, nbr, bdy, n_axis, pure_rows)
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

    # Boundary routing by tag NAME + periodic partners
    from zoomy_core.model.boundary_conditions import boundary_face_branches
    tags = (list(bcs.list_sorted_function_names) if bcs is not None else None)
    branch = (np.asarray(boundary_face_branches(mesh, bcs), dtype=int)
              if n_bf else np.zeros(0, dtype=int))
    inputs, periodic = _periodic_partners(mesh, bcs, centers0)

    return MeshTopology(
        **topo_partial, inradius_fallback=fallback,
        lsq_degree=int(lsq_degree), lsq_monomial_multi_index=tuple(mon),
        lsq_scale_factors=np.asarray(scale_lsq_derivative(mon), dtype=float),
        lsq_grad_cols=grad_cols, lsq_neighbors=nbr,
        lsq_boundary_face_neighbors=bdy, lsq_plan=plan,
        tpfa_faces=tf, tpfa_A=fcells[0, tf], tpfa_B=fcells[1, tf],
        tpfa_boundary_valid=bvalid,
        bc_tags=tags, bc_branch=branch, bc_input_cells=inputs,
        bc_periodic=periodic)


# ═════════════════════════════════════════════════════════════════════════
# Geometry (xp, gather-only, differentiable in X)
# ═════════════════════════════════════════════════════════════════════════

def _xp_of(X):
    ns = getattr(X, "__array_namespace__", None)
    if ns is None:
        return np
    return ns()


def _pad3_rows(xp, a, dim):
    """``(dim, n)`` → ``(3, n)`` with zero rows (FVMMesh convention)."""
    if dim == 3:
        return a
    z = xp.zeros((3 - dim,) + tuple(a.shape[1:]), dtype=a.dtype)
    return xp.concatenate([a, z], axis=0)


def _norm_rows(xp, v):
    """Euclidean norm over the leading axis (sum of squares in row order)."""
    acc = v[0] * v[0]
    for k in range(1, v.shape[0]):
        acc = acc + v[k] * v[k]
    return xp.sqrt(acc)


def _heron(xp, X, i, j, k):
    """Triangle area by Heron's formula over edges (i,j), (j,k), (k,i) — the
    reference ``_area_triangle_heron_formula`` operation order."""
    a = _norm_rows(xp, X[:, j] - X[:, i])
    b = _norm_rows(xp, X[:, k] - X[:, j])
    c = _norm_rows(xp, X[:, i] - X[:, k])
    s = (0.0 + a + b + c) / 2
    return xp.sqrt(s * (s - a) * (s - b) * (s - c))


def _cell_face_geometry(xp, topo, X):
    """Cell / face geometry (no LSQ): the FVMMesh fields."""
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
            d = xp.abs((fc[0, f] - cc_in[0]) * normals[0, f]
                       + (fc[1, f] - cc_in[1]) * normals[1, f])
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
    """Vandermonde columns ``prod_a dX_a**p_a`` WITHOUT ``x**0`` nodes (their
    derivative is ``0·x**-1`` = NaN at a zero offset)."""
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


def _lsq_matrices(xp, topo, cc, fc, nrm):
    """Batched LSQ derivative operators ``(n_cells, R, n_mon)`` — the
    ``LSQMesh.lsq_gradQ`` contract (``A_glob[i] = pinv(V_i).T``).  A boundary
    face's ghost sample sits at the mirror image of the cell centre (normal
    offset ``2 (x_f - x_c).n``), where the BC kernels' ghost value belongs."""
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
        Vb = V[plan["bint"]]
        V_int = Vb * keep[None, :, None]
        blocks.append(xp.linalg.pinv(V_int))
        if plan["bone_pos"].size:
            onehot = np.zeros((plan["bone_pos"].size, R))
            onehot[np.arange(onehot.shape[0]), plan["bone_row"]] = 1.0
            oh = xp.asarray(onehot, dtype=V.dtype)
            V_one = (V_int[plan["bone_pos"]]
                     + Vb[plan["bone_pos"]] * oh[:, :, None])
            blocks.append(xp.linalg.pinv(V_one))
    blocks.append(xp.zeros((1, M, R), dtype=V.dtype))
    S = xp.concatenate(blocks, axis=0).reshape(-1, R)
    A = S[plan["row_src"]]                               # (n_cells, M, R)
    return xp.transpose(A, (0, 2, 1))                    # (n_cells, R, M)


def compute_geometry(topo: MeshTopology, X, *, lsq=True):
    """All template geometry from the node coordinates ``X`` ``(dim,
    n_vertices)`` — numpy or jax.numpy, decided by ``X`` itself.

    Returns a dict (a pytree) with the FVMMesh/LSQMesh layouts:

    ``cell_centers (3, n_cells)`` incl. ghost (mirror) centres,
    ``cell_volumes (n_cells,)``, ``cell_inradius (n_cells,)``,
    ``face_normals (3, n_faces)`` (cell0 → cell1), ``face_volumes``,
    ``face_centers (n_faces, 3)``, ``face_inradius (n_faces,)`` (min of the
    two cells; boundary: the owner's), ``ghost_centers (3, n_bf)``,
    ``d_face (n_bf,)`` (owner centre → boundary face centre), ``d_normal
    (n_bf,)`` (its normal component — the ghost kernels' ``distance``), the
    face-minus-cell vectors ``r_A``/``r_B`` (interior faces) and ``r_I``
    (boundary faces), TPFA coefficients ``tpfa_gA``/``tpfa_gB`` (valid
    interior faces), ``tpfa_bnd_g``/``tpfa_bnd_dist`` (boundary faces) and,
    with ``lsq=True``, ``lsq_gradQ (n_cells, R, n_mon)``."""
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
    return g


def lsq_derivatives(topo: MeshTopology, geom, U, U_ghost):
    """LSQ derivative estimates of the fields ``U`` ``(n_v, n_cells)`` with the
    D5 GHOST values ``U_ghost`` ``(n_v, n_bf)`` at the boundary-face virtual
    points: ``(n_v, n_cells, n_mon)`` scaled to derivatives (the
    ``LSQMesh.compute_derivatives`` contract with ghost semantics, batched
    over fields and cells).  ``U`` must cover every cell the stencil reads
    (inner cells; ghost rows of ``lsq_gradQ`` are not needed for inner
    gradients — pass ``U`` with ``n_inner_cells`` columns and only inner rows
    are returned when ``U`` is that short)."""
    xp = _xp_of(U)
    A = geom["lsq_gradQ"]
    n = U.shape[1]
    nbr = topo.lsq_neighbors[:n]
    bdy = topo.lsq_boundary_face_neighbors[:n]
    U_self = U
    delta_c = U[:, nbr] - U_self[:, :, None]
    parts = [delta_c]
    if bdy.shape[1]:
        ub = U_ghost[:, np.maximum(bdy, 0)]
        mask = xp.asarray(bdy >= 0)
        parts.append(xp.where(mask[None], ub - U_self[:, :, None], 0.0))
    delta = xp.concatenate(parts, axis=2)                # (n_v, n, R)
    coeffs = xp.matmul(xp.transpose(A[:n], (0, 2, 1))[None],
                       delta[..., None])[..., 0]          # (n_v, n, M)
    return xp.asarray(topo.lsq_scale_factors, dtype=coeffs.dtype) * coeffs


def lsq_gradient(topo: MeshTopology, geom, U, U_ghost):
    """First derivatives ``(n_v, dim, n_cells)`` (see :func:`lsq_derivatives`)."""
    xp = _xp_of(U)
    d = lsq_derivatives(topo, geom, U, U_ghost)
    return xp.transpose(d[:, :, topo.lsq_grad_cols], (0, 2, 1))


__all__ = ["MeshTopology", "build_topology", "compute_geometry",
           "lsq_derivatives", "lsq_gradient"]
