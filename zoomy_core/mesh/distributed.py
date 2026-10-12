"""Distributed meshes: each MPI rank holds its owned cells plus two halo layers and never the whole
mesh, with cells numbered along a Hilbert (2D) or Morton (3D) space-filling curve.
Backend-neutral (numpy, h5py, mpi4py); a backend adds the halo exchange and global reductions.
"""
from __future__ import annotations

import os

import numpy as np

from zoomy_core.mesh.base_mesh import BaseMesh, _DIM_OF_TYPE, _local_faces_for_type

HALO = "halo"            # boundary tag of the faces the partition cuts
FORMAT_VERSION = 3


# ── global numbering ─────────────────────────────────────────────────────────
def _hilbert_2d(x, y, order):
    """Hilbert index of integer points in ``[0, 2**order)^2``."""
    x, y = x.astype(np.int64), y.astype(np.int64)
    d = np.zeros_like(x)
    n = np.int64(1) << order
    s = n >> 1
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx
        x, y = np.where(flip, n - 1 - x, x), np.where(flip, n - 1 - y, y)
        x, y = np.where(~ry, y, x), np.where(~ry, x, y)
        s >>= 1
    return d


def _morton(q, bits):
    """Morton index of integer points ``q`` of shape ``(dim, n)`` in ``[0, 2**bits)``."""
    d = np.zeros(q.shape[1], dtype=np.int64)
    for b in range(bits):
        for k in range(q.shape[0]):
            d |= ((q[k] >> b) & 1) << (b * q.shape[0] + k)
    return d


def curve_keys(centers, lo, hi):
    """Position along the space-filling curve of centres ``(dim, n)`` in the box ``lo``..``hi``
    (Hilbert in 2D, Morton in 3D, x in 1D).
    """
    c = np.asarray(centers, dtype=float)
    dim = c.shape[0]
    if dim == 1:
        return c[0]
    bits = 16 if dim == 2 else 20
    q = np.floor((c - lo) / np.where(hi > lo, hi - lo, 1.0)
                 * ((1 << bits) - 1)).astype(np.int64)
    return _hilbert_2d(q[0], q[1], bits) if dim == 2 else _morton(q, bits)


def curve_order(centers):
    """Cell order along the space-filling curve through the centres ``(dim, n)``, ties by index."""
    c = np.asarray(centers, dtype=float)
    key = curve_keys(c, c.min(axis=1, keepdims=True), c.max(axis=1, keepdims=True))
    return np.argsort(key, kind="stable")


def cell_face_tags(mesh):
    """``(faces_per_cell, n_inner_cells)`` physical tag of each boundary face at its inner cell's
    local face, -1 elsewhere.
    """
    nc, n_bf = int(mesh.n_inner_cells), int(mesh.n_boundary_faces)
    cf = np.asarray(mesh.cell_faces)[:, :nc]
    tags = -np.ones(cf.shape, dtype=np.int64)
    if n_bf:
        f = np.asarray(mesh.boundary_face_face_indices)[:n_bf]
        fc = np.asarray(mesh.face_cells)[:, f]
        c = np.where(fc[0] < nc, fc[0], fc[1])          # the inner side
        lf = np.argmax(cf[:, c] == f[None], axis=0)
        tags[lf, c] = np.asarray(mesh.boundary_face_physical_tags)[:n_bf]
    return tags


def partitioned(mesh):
    """Arrays of ``mesh`` in the global numbering (cells along the curve, vertices by first
    appearance), built from the cells' own faces so a periodic remap of ``boundary_face_cells``
    does not enter.
    """
    nc = int(mesh.n_inner_cells)
    dim = int(mesh.dimension)
    X = np.asarray(mesh.vertex_coordinates, dtype=float)[:dim]
    cv = np.asarray(mesh.cell_vertices)[:, :nc]
    order = curve_order(X[:, cv].mean(axis=1))
    cv = cv[:, order]
    used, first = np.unique(cv.T.reshape(-1), return_index=True)
    vorder = used[np.argsort(first, kind="stable")]     # new -> old vertex
    new_id = np.full(X.shape[1], -1, dtype=np.int64)
    new_id[vorder] = np.arange(len(vorder))
    names = dict(zip((int(t) for t in np.asarray(
        mesh.boundary_conditions_sorted_physical_tags)),
        mesh.boundary_conditions_sorted_names))
    return dict(type=str(mesh.type), dim=dim, X=X[:, vorder], cv=new_id[cv],
                tags=cell_face_tags(mesh)[:, order], names=names,
                order=order, vorder=vorder)


def _mesh_group(f, mesh_type, dim, n_cells, n_vertices, k, nfpc, names,
                cell_data, x_dtype=float, i_dtype=np.int64):
    """Create the (unfilled) ``/mesh`` group of a mesh file."""
    g = f.create_group("mesh")
    g.attrs["format_version"] = FORMAT_VERSION
    g.create_dataset("dimension", data=dim)
    # fixed-length string: parallel HDF5 cannot write variable-length ones
    g.create_dataset("type", data=np.bytes_(mesh_type))
    g.create_dataset("n_cells", data=n_cells)
    g.create_dataset("n_inner_cells", data=n_cells)
    g.create_dataset("boundary_tags", data=np.array(list(names), dtype=int))
    g.create_dataset("boundary_names", data=np.array(
        [str(v) for v in names.values()], dtype="S"))
    g.create_dataset("vertex_coordinates", shape=(dim, n_vertices), dtype=x_dtype)
    g.create_dataset("cell_vertices", shape=(k, n_cells), dtype=i_dtype)
    g.create_dataset("cell_face_tags", shape=(nfpc, n_cells), dtype=np.int64)
    g.create_dataset("cell_order", shape=(n_cells,), dtype=np.int64)
    for name, dt in cell_data.items():
        g.create_dataset(f"cell_data/{name}", shape=(n_cells,), dtype=dt)
    return g


def write_mesh(mesh, path, cell_data=None):
    """Write ``mesh`` as a partition-ready mesh file (serial); use ``write_mesh_parallel`` for large
    meshes.

    Args:
        cell_data: ``{name: (n_cells,)}`` per-cell values in the mesh's cell order.
    """
    import h5py
    p = partitioned(mesh)
    cell_data = {k: np.asarray(v) for k, v in (cell_data or {}).items()}
    with h5py.File(path, "w") as f:
        g = _mesh_group(f, p["type"], p["dim"], p["cv"].shape[1], p["X"].shape[1],
                        p["cv"].shape[0], p["tags"].shape[0], p["names"],
                        {k: v.dtype for k, v in cell_data.items()})
        g["vertex_coordinates"][...] = p["X"]
        g["cell_vertices"][...] = p["cv"]
        g["cell_face_tags"][...] = p["tags"]
        g["cell_order"][...] = p["order"]
        for k, v in cell_data.items():
            g[f"cell_data/{k}"][...] = v[p["order"]]


def read_mesh(path):
    """Read a mesh file as one serial BaseMesh."""
    src = FileSource(path)
    ids = np.arange(src.n_cells)
    return BaseMesh.from_arrays(src.vertices(np.arange(src.n_vertices)),
                                src.cells(ids), src.type,
                                cell_face_tags=src.face_tags(ids),
                                tag_names=src.names)


# ── sources: the global arrays, read in pieces ──────────────────────────────
class ArraySource:
    """In-memory mesh (identical on every rank) renumbered on the fly; ``cell_ids`` / ``vertex_ids``
    map the global numbering back to the mesh.
    """
    is_file = False

    def __init__(self, mesh):
        """Renumber ``mesh`` into the global numbering."""
        p = partitioned(mesh)
        self.type, self.dim, self.names = p["type"], p["dim"], p["names"]
        self._X, self._cv, self._tags = p["X"], p["cv"], p["tags"]
        self.cell_ids, self.vertex_ids = p["order"], p["vorder"]
        self.n_cells, self.n_vertices = self._cv.shape[1], self._X.shape[1]
        self.mesh = mesh

    def cells(self, ids):
        """Vertex ids of the cells ``ids``, ``(verts_per_cell, n)``."""
        return self._cv[:, ids]

    def face_tags(self, ids):
        """Face tags of the cells ``ids``, ``(faces_per_cell, n)``."""
        return self._tags[:, ids]

    def vertices(self, ids):
        """Coordinates of the vertices ``ids``, ``(dim, n)``."""
        return self._X[:, ids]

    def periodic_partners(self, bcs):
        """Periodic faces of ``bcs`` in the global numbering as ``(cell, slot, partner)``: face
        ``slot`` of ``cell`` takes ``partner``'s state.
        """
        from zoomy_core.mesh.geometry_xp import _periodic_partners
        m = self.mesh
        n_bf = int(m.n_boundary_faces)
        f = np.asarray(m.boundary_face_face_indices)[:n_bf]
        owner = np.asarray(m.face_cells)[0][f]
        partner, periodic = _periodic_partners(m, bcs, m.cell_centers_computed())
        slot = np.argmax(np.asarray(m.cell_faces)[:, owner] == f[None], axis=0)
        gid = np.empty(self.n_cells, dtype=np.int64)
        gid[self.cell_ids] = np.arange(self.n_cells)
        return (gid[owner[periodic]], slot[periodic].astype(np.int64),
                gid[np.asarray(partner)[periodic]])

    def periodic_vertices(self, bcs):
        """Seam vertex identification of ``bcs`` in the global vertex numbering, ``(canon,
        translations)``.
        """
        from zoomy_core.mesh.geometry_xp import periodic_vertices
        m = self.mesh
        canon, T = periodic_vertices(m, bcs, m.cell_centers_computed())
        gvid = np.empty(self.n_vertices, dtype=np.int64)
        gvid[self.vertex_ids] = np.arange(self.n_vertices)
        return gvid[canon[self.vertex_ids]], T


class FileSource:
    """Mesh file written by ``write_mesh``; each read touches only the rows asked for."""
    is_file = True

    def __init__(self, path):
        """Open the mesh file at ``path``; raises ValueError if it is not partition-ready."""
        import h5py
        self.path = os.fspath(path)
        self._f = h5py.File(self.path, "r")
        g = self._f["mesh"]
        if int(g.attrs.get("format_version", 0)) < FORMAT_VERSION:
            raise ValueError(f"{self.path}: not a partition-ready mesh file "
                             "(write it with zoomy_core.mesh.distributed.write_mesh)")
        t = g["type"][()]
        self.type = t.decode() if isinstance(t, bytes) else str(t)
        self.dim = _DIM_OF_TYPE[self.type]
        self.names = dict(zip(g["boundary_tags"][()].tolist(),
                              [n.decode() for n in g["boundary_names"][()]]))
        self.n_cells = int(g["cell_vertices"].shape[1])
        self.n_vertices = int(g["vertex_coordinates"].shape[1])
        self.cell_ids = self.vertex_ids = None
        self._g = g

    def _read(self, name, ids):
        """Columns ``ids`` (any order) of dataset ``name``, read as slices of consecutive ids."""
        d = self._g[name]
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty(d.shape[:-1] + (ids.size,), dtype=d.dtype)
        if ids.size == 0:
            return out
        o = np.argsort(ids, kind="stable")
        s = ids[o]
        breaks = np.flatnonzero(np.diff(s) != 1) + 1
        for a, b in zip(np.concatenate([[0], breaks]),
                        np.concatenate([breaks, [s.size]])):
            out[..., o[a:b]] = d[..., s[a]:s[b - 1] + 1]
        return out

    def cell_data(self, name, ids):
        """Per-cell values ``name`` of the cells ``ids``."""
        return self._read(f"cell_data/{name}", ids)

    def cells(self, ids):
        """Vertex ids of the cells ``ids``, ``(verts_per_cell, n)``."""
        return self._read("cell_vertices", ids)

    def face_tags(self, ids):
        """Face tags of the cells ``ids``, ``(faces_per_cell, n)``."""
        return self._read("cell_face_tags", ids)

    def vertices(self, ids):
        """Coordinates of the vertices ``ids``, ``(dim, n)``."""
        return self._read("vertex_coordinates", ids)

    def periodic_partners(self, bcs):
        """Not supported from a file; pass the mesh object instead."""
        raise NotImplementedError(
            "periodic BCs in a distributed run from a mesh file: pass the "
            "mesh object instead (its periodic faces are paired in memory)")

    periodic_vertices = periodic_partners


def source(mesh):
    """FileSource for a path, ArraySource for a mesh."""
    if isinstance(mesh, (str, os.PathLike)):
        return FileSource(mesh)
    return ArraySource(mesh)


# ── communication ────────────────────────────────────────────────────────────
def _alltoallv(comm, parts, dtype=np.int64):
    """Send ``parts[q]`` (1D) to rank q and return what each rank sent here."""
    from mpi4py import MPI
    dtype = np.dtype(dtype)
    parts = [np.ascontiguousarray(p, dtype=dtype).reshape(-1) for p in parts]
    counts = np.array([len(p) for p in parts], dtype=np.int64)
    rcounts = np.array(comm.alltoall(counts.tolist()), dtype=np.int64)
    send = np.concatenate(parts) if parts else np.zeros(0, dtype)
    recv = np.empty(int(rcounts.sum()), dtype=dtype)
    sd = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rd = np.concatenate([[0], np.cumsum(rcounts)[:-1]])
    t = MPI._typedict[dtype.char]
    comm.Alltoallv([send, (counts, sd), t], [recv, (rcounts, rd), t])
    return np.split(recv, np.cumsum(rcounts)[:-1])


def _route(comm, dest, *arrays):
    """Move the columns of each array to rank ``dest`` and return what arrived, concatenated in
    source-rank order.
    """
    P = comm.Get_size()
    o = np.argsort(dest, kind="stable")
    cut = np.searchsorted(dest[o], np.arange(1, P))
    out = []
    for A in arrays:
        A = np.asarray(A)
        rows = A.shape[:-1]
        flat = int(np.prod(rows, dtype=int))
        cols = A[..., o].reshape(flat, A.shape[-1]).T         # (n, rows)
        got = _alltoallv(comm, list(np.split(cols, cut)), A.dtype)
        m = np.concatenate([g.reshape(-1, flat) for g in got]).T
        out.append(m.reshape(rows + (m.shape[-1],)))
    return out


def _ask(comm, ids, owner, answer, dtype):
    """Evaluate ``answer(ids_q)`` on the rank ``owner`` names for each id and return the answers
    aligned with ``ids``.
    """
    P = comm.Get_size()
    dest = owner(ids)
    o = np.argsort(dest, kind="stable")
    cut = np.searchsorted(dest[o], np.arange(1, P))
    asked = _alltoallv(comm, list(np.split(ids[o], cut)))
    replies = [np.asarray(answer(q), dtype=dtype) for q in asked]
    rows = replies[0].shape[:-1] if replies else ()
    flat = int(np.prod(rows, dtype=int))
    back = _alltoallv(comm, [r.reshape(flat, r.shape[-1]).T.reshape(-1)
                             for r in replies], dtype)
    got = np.concatenate([b.reshape(-1, flat) for b in back]).T.reshape(rows + (-1,))
    out = np.empty(rows + (ids.size,), dtype=dtype)
    out[..., o] = got
    return out


def _sample_sort(comm, k1, k2, *payload):
    """Globally sort by the distinct pairs ``(k1, k2)`` so each rank ends with a contiguous sorted
    range; returns ``(k1, k2, *payload)`` of that range.
    """
    P = comm.Get_size()
    o = np.lexsort((k2, k1))
    k1, k2 = k1[o], k2[o]
    payload = [np.asarray(a)[..., o] for a in payload]
    n = k1.size
    s = (np.arange(1, P) * n // P) if n else np.zeros(0, dtype=int)
    samples = comm.allgather((k1[s], k2[s]))
    s1 = np.concatenate([a for a, _ in samples])
    s2 = np.concatenate([b for _, b in samples])
    so = np.lexsort((s2, s1))
    pick = so[(np.arange(1, P) * so.size // P)] if so.size else np.zeros(0, dtype=int)
    dest = np.zeros(n, dtype=np.int64)
    for a, b in zip(s1[pick], s2[pick]):             # splitters, ascending
        dest += (k1 > a) | ((k1 == a) & (k2 >= b))
    got = _route(comm, dest, k1, k2, *payload)
    o = np.lexsort((got[1], got[0]))
    return [g[..., o] for g in got]


def write_mesh_parallel(comm, path, mesh_type, X, v0, cv, c0, *, faces=None,
                        face_tags=None, tag_names=None, default_tag=0,
                        cell_data=None):
    """Write the mesh file of ``write_mesh`` from pieces held by the ranks, no rank holding the
    whole mesh.

    Args:
        X: this rank's vertices ``(dim, n)``, ids ``v0, v0 + 1, ...`` in contiguous rank-ordered
            blocks.
        cv: this rank's cells ``(verts_per_cell, n)`` with global vertex ids, ids ``c0, c0 + 1,
            ...``.
        faces: boundary elements ``(verts_per_face, n)`` with ``face_tags`` (any rank); untagged
            boundary faces get ``default_tag``.
        cell_data: ``{name: (n,)}`` of this rank's cells.
    """
    from mpi4py import MPI
    import h5py
    P, rank = comm.Get_size(), comm.Get_rank()
    X, cv = np.asarray(X, dtype=float), np.asarray(cv, dtype=np.int64)
    dim, k = X.shape[0], cv.shape[0]
    cell_data = {n: np.asarray(v) for n, v in (cell_data or {}).items()}
    vstart = np.array(comm.allgather(int(v0)))
    vend = np.array(comm.allgather(int(v0) + X.shape[1]))

    def owner(v):                                    # vertex block of a rank
        """Rank holding the vertex block of vertex ``v``."""
        return np.searchsorted(vend, v, side="right")

    # cells along the curve: centroids from the vertices' owners
    used = np.unique(cv)
    xyz = _ask(comm, used, owner, lambda q: X[:, q - v0], float)
    cent = xyz[:, np.searchsorted(used, cv)].mean(axis=1)
    lo, hi = np.empty(dim), np.empty(dim)
    comm.Allreduce(cent.min(axis=1) if cent.size else np.full(dim, np.inf), lo, op=MPI.MIN)
    comm.Allreduce(cent.max(axis=1) if cent.size else np.full(dim, -np.inf), hi, op=MPI.MAX)
    key = curve_keys(cent, lo[:, None], hi[:, None])
    ids = np.arange(c0, c0 + cv.shape[1], dtype=np.int64)
    names_cd = list(cell_data)
    key, ids, cv, *cds = _sample_sort(comm, key, ids, cv,
                                      *[cell_data[n] for n in names_cd])
    n_mine = ids.size
    off = comm.exscan(n_mine) or 0
    n_cells = comm.allreduce(n_mine)

    # vertices by first appearance: min (cell position * k + slot) per vertex
    pos = (off + np.arange(n_mine))[None, :] * k + np.arange(k)[:, None]
    v_got, p_got = _route(comm, owner(cv.reshape(-1)), cv.reshape(-1), pos.reshape(-1))
    o = np.lexsort((p_got, v_got))
    v_got, p_got = v_got[o], p_got[o]
    first = np.concatenate([[True], v_got[1:] != v_got[:-1]]) if v_got.size else v_got.astype(bool)
    v_used, p_min = v_got[first], p_got[first]
    p_min, v_used, xv = _sample_sort(comm, p_min, v_used, X[:, v_used - v0])
    nv_mine = v_used.size
    voff = comm.exscan(nv_mine) or 0
    n_vertices = comm.allreduce(nv_mine)
    new_of = np.full(X.shape[1], -1, dtype=np.int64)  # old -> new, my block
    v_back, n_back = _route(comm, owner(v_used), v_used,
                            voff + np.arange(nv_mine, dtype=np.int64))
    new_of[v_back - v0] = n_back
    uv = np.unique(cv)
    cv_new = _ask(comm, uv, owner, lambda q: new_of[q - v0], np.int64)[
        np.searchsorted(uv, cv)]

    # boundary faces and their tags: faces hashed to their smallest vertex
    local = _local_faces_for_type(mesh_type)
    w = len(local[0])
    if any(len(lf) != w for lf in local):
        raise NotImplementedError(f"{mesh_type}: faces of different sizes")
    fkeys = np.sort(np.stack([cv[list(lf)] for lf in local], axis=1), axis=0)  # (w, nf, n)
    fkeys = fkeys.reshape(w, -1)
    fpos = (off + np.arange(n_mine))[None, :] * len(local) + np.arange(len(local))[:, None]
    fk, fp = _route(comm, owner(fkeys[0]), fkeys, fpos.reshape(-1))
    faces = np.zeros((w, 0), np.int64) if faces is None else np.sort(np.asarray(faces, np.int64), axis=0)
    ftags = np.zeros(0, np.int64) if face_tags is None else np.asarray(face_tags, np.int64)
    bk, bt = _route(comm, owner(faces[0]), faces, ftags)
    void = np.dtype((np.void, 8 * w))
    kv = np.ascontiguousarray(fk.T).view(void).ravel()
    uk, inv, cnt = np.unique(kv, return_inverse=True, return_counts=True)
    bv = np.ascontiguousarray(bk.T).view(void).ravel()
    tag_of = np.full(uk.size, default_tag, dtype=np.int64)
    hit = np.searchsorted(uk, bv)
    ok = (hit < uk.size) & (uk[np.minimum(hit, uk.size - 1)] == bv)
    tag_of[hit[ok]] = bt[ok]
    t = np.where(cnt[inv.ravel()] == 1, tag_of[inv.ravel()], -1)
    starts = np.array(comm.allgather(off))
    fp_back, t_back = _route(comm, np.searchsorted(starts, fp // len(local), side="right") - 1,
                             fp, t)
    cft = np.full((len(local), n_mine), -1, dtype=np.int64)
    cft[fp_back % len(local), fp_back // len(local) - off] = t_back
    present = sorted(set().union(*comm.allgather(set(np.unique(cft[cft >= 0]).tolist()))))
    names = {tg: (tag_names or {}).get(tg, f"tag_{tg}") for tg in present}

    blocks = dict(vertex_coordinates=(xv, voff), cell_vertices=(cv_new, off),
                  cell_face_tags=(cft, off), cell_order=(ids, off),
                  **{f"cell_data/{n}": (c, off) for n, c in zip(names_cd, cds)})
    meta = (mesh_type, dim, n_cells, n_vertices, k, len(local), names,
            {n: cell_data[n].dtype for n in names_cd})
    if h5py.get_config().mpi:                        # every rank its block
        with h5py.File(path, "w", driver="mpio", comm=comm) as f:
            g = _mesh_group(f, *meta)
            for name, (a, o0) in blocks.items():
                if a.shape[-1]:
                    g[name][..., o0:o0 + a.shape[-1]] = a
        return
    got = comm.gather(blocks, root=0)                # else: rank 0 writes
    if rank == 0:
        with h5py.File(path, "w") as f:
            g = _mesh_group(f, *meta)
            for name in blocks:
                g[name][...] = np.concatenate([b[name][0] for b in got], axis=-1)


class _Directory:
    """Vertex-to-cells directory, vertex block ``v * P // n_v`` held by one rank each."""

    def __init__(self, comm, own, own_cv, n_vertices):
        """Distribute the (vertex, cell) pairs of the owned cells to the directory ranks."""
        self.comm, self.n_v, self.P = comm, n_vertices, comm.Get_size()
        v = own_cv.reshape(-1)
        c = np.broadcast_to(own[None, :], own_cv.shape).reshape(-1)
        dest = v * self.P // self.n_v
        o = np.argsort(dest, kind="stable")
        cut = np.searchsorted(dest[o], np.arange(1, self.P))
        got_v = _alltoallv(comm, np.split(v[o], cut))
        got_c = _alltoallv(comm, np.split(c[o], cut))
        v, c = np.concatenate(got_v), np.concatenate(got_c)
        o = np.lexsort((c, v))
        self.v, self.c = v[o], c[o]

    def cells_around(self, vertices):
        """Union of the cells around ``vertices``, asked of their directory holders."""
        vertices = np.unique(vertices)
        dest = vertices * self.P // self.n_v
        cut = np.searchsorted(dest, np.arange(1, self.P))   # sorted already
        asked = _alltoallv(self.comm, np.split(vertices, cut))
        answers = []
        for q in asked:
            lo = np.searchsorted(self.v, q, side="left")
            n = np.searchsorted(self.v, q, side="right") - lo
            start = np.cumsum(n) - n
            idx = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(start, n)
            answers.append(np.unique(self.c[idx]))
        return np.unique(np.concatenate(_alltoallv(self.comm, answers)))


# ── the distributed mesh ────────────────────────────────────────────────────
class Partition:
    """One rank's piece: local cells (owned, then halo layer 1, then 2) and the exchange lists;
    ``cells`` / ``vertices`` map local to the source's numbering.
    """

    def __init__(self, comm, src, offsets, gids, n_own, n_ring, vids, send,
                 recv, periodic=None, periodic_ids=None):
        """Store the rank's cells, vertices, exchange lists and periodic data, and clone ``comm``
        for host-side collectives.
        """
        self.comm, self.source, self.offsets = comm, src, offsets
        # periodic faces held here: (local cell, face slot, local partner);
        # the seams' vertex identification of the piece (local vertices)
        self.periodic, self.periodic_ids = periodic, periodic_ids
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        self.gids, self.n_own, self.n_ring = gids, n_own, n_ring
        self.cells = gids if src.cell_ids is None else src.cell_ids[gids]
        self.vertices = vids if src.vertex_ids is None else src.vertex_ids[vids]
        self.n_global, self.n_global_vertices = src.n_cells, src.n_vertices
        self.send, self.recv = send, recv           # {peer: local indices}
        self._hcomm = comm.Clone()                   # host-side collectives

    def gather(self, X, root=0):
        """Owned columns of ``X`` ``(rows, local cells)`` from every rank in the source's cell
        order, on ``root`` (None elsewhere).
        """
        X = np.asarray(X)
        got = self._hcomm.gather((self.cells[:self.n_own], X[..., :self.n_own]),
                                 root=root)
        if got is None:
            return None
        out = np.empty(X.shape[:-1] + (self.n_global,), dtype=X.dtype)
        for ids, x in got:
            out[..., ids] = x
        return out

    def scatter(self, X_global):
        """Local columns (owned plus halo) of a global ``(rows, cells)`` array."""
        return np.asarray(X_global)[..., self.cells]

    def sum_to_global(self, G_local, ids, n_global):
        """Sum local contributions per global index over all copies on all ranks, i.e. the
        derivative with respect to a global quantity every rank holds copies of.

        Args:
            ids: global index of each local column (``self.vertices``, ``self.cells``), or None
                for replicated parameters.
        """
        from mpi4py import MPI
        G = np.asarray(G_local)
        if ids is None:
            return self._hcomm.allreduce(G, op=MPI.SUM)
        out = np.zeros(G.shape[:-1] + (n_global,), dtype=G.dtype)
        np.add.at(out, (..., np.asarray(ids)), G)
        return self._hcomm.allreduce(out, op=MPI.SUM)

    def resolve_periodic(self, local_mesh):
        """Point every periodic face whose partner cell this piece holds at the partner's halo copy."""
        if self.periodic is None or not len(self.periodic[0]):
            return
        cells, slots, partners = self.periodic
        n_bf = int(local_mesh.n_boundary_faces)
        f = np.asarray(local_mesh.boundary_face_face_indices)[:n_bf]
        owner = np.asarray(local_mesh.face_cells)[0][f]
        slot = np.argmax(np.asarray(local_mesh.cell_faces)[:, owner] == f[None], axis=0)
        nf = int(np.asarray(local_mesh.cell_faces).shape[0])
        keys = cells * nf + slots
        o = np.argsort(keys)
        at = np.searchsorted(keys[o], owner * nf + slot)
        at = np.minimum(at, len(o) - 1)
        hit = keys[o][at] == owner * nf + slot
        local_mesh.boundary_face_cells[:n_bf][hit] = partners[o][at][hit]

    def check(self, local_mesh):
        """Raise if a cut face touches an owned or first-layer cell, i.e. the halo is too thin."""
        names = list(local_mesh.boundary_conditions_sorted_names)
        if HALO not in names:
            return
        n_bf = int(local_mesh.n_boundary_faces)
        nums = np.asarray(local_mesh.boundary_face_function_numbers)[:n_bf]
        cut = nums == names.index(HALO)
        f = np.asarray(local_mesh.boundary_face_face_indices)[:n_bf][cut]
        owner = np.asarray(local_mesh.face_cells)[0, f]
        if (owner < self.n_ring).any():
            raise RuntimeError(f"rank {self.rank}: a cut face touches an owned "
                               "or first-layer cell; the halo is too thin")

    # ── output ─────────────────────────────────────────────────────────────
    def parallel_io(self):
        """Whether each rank can write its own block (mesh-file source and h5py with MPI)."""
        if not self.source.is_file:
            return False
        try:
            import h5py
            return bool(h5py.get_config().mpi)
        except ImportError:
            return False

    def write_mesh(self, path, names=None):
        """Write the output file's ``/mesh`` (each rank its share of the input blocks) and the field
        names.
        """
        import h5py
        src, P, r = self.source, self.size, self.rank
        with h5py.File(path, "w", driver="mpio", comm=self._hcomm) as f:
            g = f.create_group("mesh")
            g.attrs["format_version"] = FORMAT_VERSION
            for k in ("dimension", "type", "n_cells", "n_inner_cells",
                      "boundary_tags", "boundary_names"):
                v = src._g[k][()]
                g.create_dataset(k, data=np.bytes_(v) if isinstance(v, (bytes, str)) else v)
            for k in ("vertex_coordinates", "cell_vertices", "cell_face_tags",
                      "cell_order"):
                a = src._g[k]
                d = g.create_dataset(k, shape=a.shape, dtype=a.dtype)
                n = a.shape[-1]
                lo, hi = r * n // P, (r + 1) * n // P
                if hi > lo:
                    d[..., lo:hi] = a[..., lo:hi]
            fg = f.create_group("fields")
            if names is not None:
                fg.attrs["names"] = np.array([str(n) for n in names], dtype="S")

    def write_fields(self, path, i, time, Q, Qaux):
        """Write snapshot ``i``, each rank its owned columns of ``Q`` and ``Qaux``."""
        import h5py
        a, b = int(self.offsets[self.rank]), int(self.offsets[self.rank + 1])
        with h5py.File(path, "a", driver="mpio", comm=self._hcomm) as f:
            g = f["fields"].create_group(f"iteration_{int(i)}")
            t = g.create_dataset("time", shape=(), dtype=float)
            if self.rank == 0:
                t[()] = float(time)
            for k, X in (("Q", Q), ("Qaux", Qaux)):
                X = np.asarray(X)
                d = g.create_dataset(k, shape=(X.shape[0], self.n_global),
                                     dtype=X.dtype)
                d[:, a:b] = X[:, :self.n_own]


def distribute(src, comm, bcs=None):
    """Build this rank's local BaseMesh (owned cells, two halo layers, cut faces tagged ``halo``)
    and its ``Partition``.  Periodic partners of owned cells join layer 1 and those of layer 1
    join layer 2.

    Args:
        src: ArraySource, FileSource, mesh or path.
        bcs: boundary conditions; their ``Periodic`` entries are paired across ranks.
    Returns:
        ``(local_mesh, partition)``.
    """
    from zoomy_core.model.boundary_conditions import Periodic
    if not isinstance(src, (ArraySource, FileSource)):
        src = source(src)
    pairs = (src.periodic_partners(bcs) if bcs is not None and any(
        type(b) is Periodic for b in bcs.boundary_conditions_list) else None)
    rank, P, N = comm.Get_rank(), comm.Get_size(), src.n_cells
    offsets = np.arange(P + 1, dtype=np.int64) * N // P
    own = np.arange(offsets[rank], offsets[rank + 1])
    cv_own = src.cells(own)
    # vertex rings across periodic seams: the seams' vertices identified
    ids = src.periodic_vertices(bcs) if pairs is not None else None
    canon = (lambda v: v) if ids is None else (lambda v: ids[0][v])   # noqa: E731
    directory = _Directory(comm, own, canon(cv_own), src.n_vertices)
    around = lambda v: directory.cells_around(canon(v))               # noqa: E731

    def partners(cells):           # periodic partners of ``cells``
        """Periodic partners of ``cells``."""
        if pairs is None:
            return np.zeros(0, dtype=np.int64)
        return pairs[2][np.isin(pairs[0], cells)]
    # layer 1: the vertex ring, and the periodic partners (an owned seam face
    # reads its partner's reconstructed trace: the partner's whole stencil)
    l1 = np.setdiff1d(np.union1d(around(cv_own), partners(own)), own)
    cv_l1 = src.cells(l1)
    new_v = np.setdiff1d(np.unique(cv_l1), np.unique(cv_own))
    near = np.concatenate([own, l1])
    # layer 2: layer 1's vertex ring and the partners' values its ghosts
    # read; a layer-1 cell on the physical boundary (a periodic partner) has
    # a one-sided stencil that reaches a ring further (1-D: two cells in)
    edge = src.cells(l1[(src.face_tags(l1) >= 0).any(axis=0)])
    far = around(src.cells(around(edge)))
    l2 = np.setdiff1d(np.union1d(np.union1d(around(new_v), far),
                                 partners(near)), near)
    gids = np.concatenate([own, l1, l2])
    cv = np.concatenate([cv_own, cv_l1, src.cells(l2)], axis=1)

    # exchange lists: ask each owner for its cells, in local order
    halo = gids[len(own):]
    owner = np.searchsorted(offsets, halo, side="right") - 1
    asks = [halo[owner == q] for q in range(P)]
    recv = {q: len(own) + np.flatnonzero(owner == q)
            for q in range(P) if (owner == q).any()}
    asked = _alltoallv(comm, asks)
    send = {q: a - offsets[rank] for q, a in enumerate(asked) if len(a)}

    vids, lcv = np.unique(cv, return_inverse=True)
    names = dict(src.names)
    if HALO in names.values():
        raise ValueError(f"the mesh already uses the boundary tag {HALO!r}")
    halo_tag = max(list(names) + [-1]) + 1
    names[halo_tag] = HALO
    local = BaseMesh.from_arrays(src.vertices(vids), lcv.reshape(cv.shape),
                                 src.type, cell_face_tags=src.face_tags(gids),
                                 tag_names=names, default_tag=halo_tag)
    periodic = None
    if pairs is not None:          # the pairs this piece holds, local indices
        o = np.argsort(gids)
        loc = lambda g: o[np.searchsorted(gids[o], g)]            # noqa: E731
        held = np.isin(pairs[0], gids) & np.isin(pairs[2], gids)
        periodic = (loc(pairs[0][held]), pairs[1][held], loc(pairs[2][held]))
    periodic_ids = None
    if ids is not None:            # local vertices, classes renumbered
        periodic_ids = (np.unique(ids[0][vids], return_inverse=True)[1], ids[1])
    part = Partition(comm, src, offsets, gids, len(own), len(own) + len(l1),
                     vids, send, recv, periodic, periodic_ids)
    part.check(local)
    return local, part
