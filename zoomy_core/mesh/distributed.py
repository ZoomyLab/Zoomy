"""Distributed meshes: every MPI rank holds the cells it owns plus two layers
of halo cells (copies of the neighbouring ranks' cells), and no rank ever
holds the whole mesh.  Backend-neutral (numpy, h5py, mpi4py); a backend adds
the halo exchange and the global reductions of its compiled loop.

Global numbering: cells sorted along a space-filling curve (Hilbert in 2-D,
Morton in 3-D), vertices in order of first appearance.  Rank ``r`` owns the
contiguous block of cells ``offsets[r]:offsets[r + 1]`` -- the partition is
the curve, so a rank reads only its own slice of a mesh file.

Mesh file (``write_mesh``; the ``/mesh`` group of the solver output plus
``cell_face_tags``, so an output file is again an input)::

    /mesh/dimension, type, n_cells, n_inner_cells, format_version = 3
    /mesh/vertex_coordinates  (dim, n_vertices)
    /mesh/cell_vertices       (verts_per_cell, n_cells)
    /mesh/cell_face_tags      (faces_per_cell, n_cells)  physical tag, -1 inside
    /mesh/boundary_tags, boundary_names
    /mesh/cell_order          (n_cells,) cell index in the source mesh

Halo discovery (``distribute``): each vertex has a directory rank holding the
cells around it; a rank asks the directory for the cells around its owned
cells' vertices (layer 1), then around layer 1's new vertices (layer 2).
Every rank then reads exactly its local cells, vertices and tags.
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
    """Morton index of integer points ``q`` ``(dim, n)`` in ``[0, 2**bits)``."""
    d = np.zeros(q.shape[1], dtype=np.int64)
    for b in range(bits):
        for k in range(q.shape[0]):
            d |= ((q[k] >> b) & 1) << (b * q.shape[0] + k)
    return d


def curve_order(centers):
    """Cells sorted along a space-filling curve through their centres
    ``(dim, n)``: Hilbert (2-D), Morton (3-D), x (1-D); stable."""
    c = np.asarray(centers, dtype=float)
    dim = c.shape[0]
    if dim == 1:
        return np.argsort(c[0], kind="stable")
    bits = 16 if dim == 2 else 20
    lo, hi = c.min(axis=1, keepdims=True), c.max(axis=1, keepdims=True)
    q = np.floor((c - lo) / np.where(hi > lo, hi - lo, 1.0)
                 * ((1 << bits) - 1)).astype(np.int64)
    key = _hilbert_2d(q[0], q[1], bits) if dim == 2 else _morton(q, bits)
    return np.argsort(key, kind="stable")


def cell_face_tags(mesh):
    """``(faces_per_cell, n_inner_cells)``: the physical tag of every
    boundary face at its inner cell's local face; -1 elsewhere."""
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
    """The arrays of ``mesh`` in the global numbering: cells along the curve,
    vertices by first appearance.  Periodic meshes are refused."""
    nc, n_bf = int(mesh.n_inner_cells), int(mesh.n_boundary_faces)
    if n_bf and (np.asarray(mesh.boundary_face_cells)[:n_bf]
                 != np.asarray(mesh.face_cells)[0][np.asarray(
                     mesh.boundary_face_face_indices)[:n_bf]]).any():
        raise NotImplementedError("distributed run on a periodic mesh")
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


def write_mesh(mesh, path):
    """Write ``mesh`` as a partition-ready mesh file (serial, once)."""
    import h5py
    p = partitioned(mesh)
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.attrs["format_version"] = FORMAT_VERSION
        g.create_dataset("dimension", data=p["dim"])
        g.create_dataset("type", data=np.bytes_(p["type"]))  # fixed length:
        # parallel HDF5 cannot write variable-length strings
        g.create_dataset("n_cells", data=p["cv"].shape[1])
        g.create_dataset("n_inner_cells", data=p["cv"].shape[1])
        g.create_dataset("vertex_coordinates", data=p["X"])
        g.create_dataset("cell_vertices", data=p["cv"])
        g.create_dataset("cell_face_tags", data=p["tags"])
        g.create_dataset("cell_order", data=p["order"])
        g.create_dataset("boundary_tags", data=np.array(list(p["names"]), dtype=int))
        g.create_dataset("boundary_names", data=np.array(
            [str(v) for v in p["names"].values()], dtype="S"))


def read_mesh(path):
    """A mesh file as one (serial) BaseMesh."""
    src = FileSource(path)
    ids = np.arange(src.n_cells)
    return BaseMesh.from_arrays(src.vertices(np.arange(src.n_vertices)),
                                src.cells(ids), src.type,
                                cell_face_tags=src.face_tags(ids),
                                tag_names=src.names)


# ── sources: the global arrays, read in pieces ──────────────────────────────
class ArraySource:
    """An in-memory mesh (the same on every rank), renumbered on the fly.
    ``cell_ids`` / ``vertex_ids`` map the global numbering back to the mesh."""
    is_file = False

    def __init__(self, mesh):
        p = partitioned(mesh)
        self.type, self.dim, self.names = p["type"], p["dim"], p["names"]
        self._X, self._cv, self._tags = p["X"], p["cv"], p["tags"]
        self.cell_ids, self.vertex_ids = p["order"], p["vorder"]
        self.n_cells, self.n_vertices = self._cv.shape[1], self._X.shape[1]
        self.mesh = mesh

    def cells(self, ids):
        return self._cv[:, ids]

    def face_tags(self, ids):
        return self._tags[:, ids]

    def vertices(self, ids):
        return self._X[:, ids]


class FileSource:
    """A mesh file (``write_mesh``); each read touches only the rows asked
    for.  ``cell_ids`` / ``vertex_ids``: the file's own numbering."""
    is_file = True

    def __init__(self, path):
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
        """Columns ``ids`` (any order), read as slices of consecutive ids."""
        d = self._g[name]
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((d.shape[0], ids.size), dtype=d.dtype)
        if ids.size == 0:
            return out
        o = np.argsort(ids, kind="stable")
        s = ids[o]
        breaks = np.flatnonzero(np.diff(s) != 1) + 1
        for a, b in zip(np.concatenate([[0], breaks]),
                        np.concatenate([breaks, [s.size]])):
            out[:, o[a:b]] = d[:, s[a]:s[b - 1] + 1]
        return out

    def cells(self, ids):
        return self._read("cell_vertices", ids)

    def face_tags(self, ids):
        return self._read("cell_face_tags", ids)

    def vertices(self, ids):
        return self._read("vertex_coordinates", ids)


def source(mesh):
    """A path -> FileSource; a mesh -> ArraySource."""
    if isinstance(mesh, (str, os.PathLike)):
        return FileSource(mesh)
    return ArraySource(mesh)


# ── communication ────────────────────────────────────────────────────────────
def _alltoallv(comm, parts):
    """Send ``parts[q]`` (int64) to rank q; returns what each rank sent here."""
    from mpi4py import MPI
    counts = np.array([len(p) for p in parts], dtype=np.int64)
    rcounts = np.array(comm.alltoall(counts.tolist()), dtype=np.int64)
    send = (np.concatenate(parts) if len(parts) else np.zeros(0)).astype(np.int64)
    recv = np.empty(int(rcounts.sum()), dtype=np.int64)
    sd = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rd = np.concatenate([[0], np.cumsum(rcounts)[:-1]])
    comm.Alltoallv([send, (counts, sd), MPI.INT64_T],
                   [recv, (rcounts, rd), MPI.INT64_T])
    return np.split(recv, np.cumsum(rcounts)[:-1])


class _Directory:
    """Vertex -> the cells around it, the vertex block ``v * P // n_v`` held
    by one rank each."""

    def __init__(self, comm, own, own_cv, n_vertices):
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
        """Union of the cells around ``vertices`` (asked of their holders)."""
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
    """One rank's piece: local cells (owned, then halo layer 1, then 2) and
    the exchange lists.  ``cells`` / ``vertices``: local -> the source's
    numbering (the mesh's own ids, or the file's)."""

    def __init__(self, comm, src, offsets, gids, n_own, n_ring, vids, send,
                 recv):
        self.comm, self.source, self.offsets = comm, src, offsets
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        self.gids, self.n_own, self.n_ring = gids, n_own, n_ring
        self.cells = gids if src.cell_ids is None else src.cell_ids[gids]
        self.vertices = vids if src.vertex_ids is None else src.vertex_ids[vids]
        self.n_global, self.n_global_vertices = src.n_cells, src.n_vertices
        self.send, self.recv = send, recv           # {peer: local indices}
        self._hcomm = comm.Clone()                   # host-side collectives

    def gather(self, X, root=0):
        """Owned columns of ``X`` ``(rows, local cells)`` from every rank, in
        the source's cell order, on ``root`` (None elsewhere)."""
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
        """The local columns (owned + halo) of a global ``(rows, cells)``."""
        return np.asarray(X_global)[..., self.cells]

    def sum_to_global(self, G_local, ids, n_global):
        """Sum local contributions per global index over all copies on all
        ranks: the derivative w.r.t. a global quantity that each rank holds
        copies of (vertex coordinates: ``ids=self.vertices``; initial cell
        states: ``ids=self.cells``; replicated parameters: ``ids=None``)."""
        from mpi4py import MPI
        G = np.asarray(G_local)
        if ids is None:
            return self._hcomm.allreduce(G, op=MPI.SUM)
        out = np.zeros(G.shape[:-1] + (n_global,), dtype=G.dtype)
        np.add.at(out, (..., np.asarray(ids)), G)
        return self._hcomm.allreduce(out, op=MPI.SUM)

    def check(self, local_mesh):
        """Every cell whose stencil an owned face reads (owned + layer 1)
        must be away from the cut faces: those belong to layer 2 only."""
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
        """Each rank writes its own block (a mesh file, h5py with MPI)."""
        if not self.source.is_file:
            return False
        try:
            import h5py
            return bool(h5py.get_config().mpi)
        except ImportError:
            return False

    def write_mesh(self, path, names=None):
        """The output file's ``/mesh`` (blocks of the input mesh file, each
        rank its share) and the field names."""
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
        """Snapshot ``i``: each rank writes its owned columns."""
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


def distribute(src, comm):
    """This rank's local mesh (a BaseMesh: owned cells, then 2 halo layers;
    cut faces tagged ``halo``) and its :class:`Partition`.  ``src``: an
    :class:`ArraySource` / :class:`FileSource` (or a mesh / path)."""
    if not isinstance(src, (ArraySource, FileSource)):
        src = source(src)
    if src.dim < 2:
        raise NotImplementedError(
            "distributed 1-D runs: the 1-D LSQ stencil is sized by the widest "
            "cell of the whole mesh, not locally")
    rank, P, N = comm.Get_rank(), comm.Get_size(), src.n_cells
    offsets = np.arange(P + 1, dtype=np.int64) * N // P
    own = np.arange(offsets[rank], offsets[rank + 1])
    cv_own = src.cells(own)
    directory = _Directory(comm, own, cv_own, src.n_vertices)
    l1 = np.setdiff1d(directory.cells_around(cv_own), own)
    cv_l1 = src.cells(l1)
    new_v = np.setdiff1d(np.unique(cv_l1), np.unique(cv_own))
    l2 = np.setdiff1d(directory.cells_around(new_v), np.concatenate([own, l1]))
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
    part = Partition(comm, src, offsets, gids, len(own), len(own) + len(l1),
                     vids, send, recv)
    part.check(local)
    return local, part
