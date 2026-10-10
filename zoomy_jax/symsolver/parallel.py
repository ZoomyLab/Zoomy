"""Distributed runs: one MPI rank per device, each on its piece of the mesh.

A rank's local mesh holds the cells it owns, then two layers of halo cells:
copies of the neighbouring ranks' cells, enough for every stencil of an owned
cell (the vertex-ring LSQ gradient and limiter of the cells across its faces).
Faces cut by the partition are boundary faces tagged ``halo`` (the runtime
copies the inner state there; nothing an owned cell reads depends on them).

The template's ``halo`` block refreshes the copies after each stencil update;
the reductions (the CFL step, MOOD's any) count every face and cell exactly
once and combine the ranks.  Everything else is the serial runtime on the
local mesh.  The exchanges are mpi4jax collectives inside the compiled loop,
differentiable in both directions (``alltoall`` is linear, its transpose is
``alltoall``; the minimum's tangent is a sum).

    from mpi4py import MPI
    solver = SymbolicHyperbolicSolverJax(time_end=0.5, comm=MPI.COMM_WORLD)
    Q, Qaux = solver.setup(global_mesh, model)        # the local piece
    time, Q, Qaux = solver.run(Q, Qaux)
    Q_all = solver.partition.gather(Q)                # global order, on root
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sps

from zoomy_core.mesh.base_mesh import BaseMesh, _local_faces_for_type

HALO = "halo"            # boundary tag of the faces the partition cuts
LAYERS = 2               # halo depth: the order-2 stencil of a halo face


def _mpi4jax():
    try:
        import mpi4jax
    except ImportError as e:
        raise ImportError(
            "distributed runs need mpi4jax (and mpi4py): pip install mpi4jax "
            "-- see docs/dev/symbolic_solver/PARALLEL.md") from e
    _register_alltoall_ad()
    return mpi4jax


_AD_DONE = []


def _register_alltoall_ad():
    """mpi4jax's alltoall has no AD rules; it is linear and its transpose is
    alltoall again (what rank r sent to s comes back from s to r)."""
    if _AD_DONE:
        return
    from jax.interpreters import ad
    from mpi4jax._src.collective_ops.alltoall import mpi_alltoall_p as p

    def jvp(primals, tangents, comm):
        (x,), (t,) = primals, tangents
        return p.bind(x, comm=comm), p.bind(ad.instantiate_zeros(t), comm=comm)

    def transpose(ct, x, comm):
        return [p.bind(ad.instantiate_zeros(ct), comm=comm)]

    ad.primitive_jvps[p] = jvp
    ad.primitive_transposes[p] = transpose
    _AD_DONE.append(True)


# ── partition ───────────────────────────────────────────────────────────────
def bisection(centers, n_parts):
    """Recursive coordinate bisection of the cell centres ``(dim, n)`` into
    ``n_parts`` pieces of (nearly) equal size; deterministic."""
    centers = np.asarray(centers)
    part = np.zeros(centers.shape[1], dtype=np.int64)

    def split(idx, parts, first):
        if parts == 1:
            part[idx] = first
            return
        left = parts // 2
        x = centers[:, idx]
        axis = int(np.argmax(np.ptp(x, axis=1)))
        idx = idx[np.argsort(x[axis], kind="stable")]
        k = len(idx) * left // parts
        split(idx[:k], left, first)
        split(idx[k:], parts - left, first + left)

    split(np.arange(centers.shape[1]), int(n_parts), 0)
    return part


def _vertex_adjacency(cell_vertices, n_vertices):
    """Cells sharing a vertex (the LSQ stencil's ring), as a sparse matrix."""
    cv = np.asarray(cell_vertices)
    k, n = cv.shape
    C = sps.csr_matrix((np.ones(k * n, dtype=np.int8),
                        (np.repeat(np.arange(n), k), cv.T.reshape(-1))),
                       shape=(n, n_vertices))
    return (C @ C.T).tocsr()


def _halo_of(A, own):
    """Global ids of the ``LAYERS`` halo layers around the mask ``own``,
    layer by layer, each sorted."""
    have, layers = own.copy(), []
    for _ in range(LAYERS):
        ring = (A @ have.astype(np.int32)) > 0
        new = ring & ~have
        layers.append(np.flatnonzero(new))
        have |= new
    return layers


class Partition:
    """One rank's view: its local cells (owned first, then the halo) and the
    exchange lists.  ``cells`` / ``vertices``: local -> global ids."""

    def __init__(self, comm, part, cells, n_own, n_ring, vertices, send, recv,
                 n_global):
        self.comm, self.part = comm, part
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        self.cells, self.n_own, self.vertices = cells, n_own, vertices
        self.n_ring = n_ring                         # owned + first layer
        self.n_global = n_global
        self.send, self.recv = send, recv           # {peer: local indices}
        self._jcomm = comm.Clone()                   # the compiled exchanges
        self._hcomm = comm.Clone()                   # host-side gathers

    # ── host side ──────────────────────────────────────────────────────────
    def gather(self, X, root=0):
        """Owned columns of ``X`` ``(rows, local cells)`` from every rank, in
        global cell order, on ``root`` (None elsewhere)."""
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

    def check_stencils(self, global_mesh, local_mesh):
        """The LSQ stencils (cells and boundary samples) of the owned cells
        and the first halo layer must be the global mesh's: their gradients
        and limiters feed every owned face."""
        n, N = self.n_ring, int(global_mesh.n_inner_cells)
        nc, cells = int(local_mesh.n_inner_cells), self.cells

        def rows(nbrs, ids, size, me):
            x = np.asarray(nbrs)[:n]
            g = np.where((x < size) & (x != me[:, None]), ids[np.minimum(x, size - 1)], -1)
            return np.sort(g, axis=1)
        g = rows(np.asarray(global_mesh.lsq_neighbors)[cells[:n]], np.arange(N),
                 N, cells[:n])
        l = rows(local_mesh.lsq_neighbors, cells, nc, np.arange(n))
        w = max(g.shape[1], l.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (w - a.shape[1], 0)), constant_values=-1)
        nb = lambda m, c: (np.asarray(m.lsq_boundary_face_neighbors)[c] >= 0).sum(axis=1)
        bad = np.flatnonzero((pad(g) != pad(l)).any(axis=1)
                             | (nb(global_mesh, cells[:n]) != nb(local_mesh, np.arange(n))))
        if len(bad):
            raise RuntimeError(
                f"rank {self.rank}: {len(bad)} owned/first-layer cells have a "
                f"different LSQ stencil than on the global mesh (e.g. global "
                f"cell {cells[bad[0]]}); the halo is too thin for this stencil")

    # ── blocks of the compiled loop ────────────────────────────────────────
    def blocks(self, mesh_np):
        """``halo`` and the global reductions for the runtime's name table."""
        mpi4jax = _mpi4jax()
        from mpi4py import MPI
        comm, size = self._jcomm, self.size
        nc, nf = int(mesh_np.n_inner_cells), int(mesh_np.n_faces)
        if nc == nf:
            raise ValueError("local mesh with as many faces as cells: the "
                             "reductions cannot tell them apart")
        m = max(1, comm.allreduce(max([len(v) for v in self.send.values()]
                                      + [0]), op=MPI.MAX))
        send = np.zeros((size, m), dtype=np.int64)          # pad: cell 0
        recv = np.full((size, m), nc, dtype=np.int64)       # pad: dropped
        for s, idx in self.send.items():
            send[s, :len(idx)] = idx
        for s, idx in self.recv.items():
            recv[s, :len(idx)] = idx
        send, recv = jnp.asarray(send), jnp.asarray(recv.reshape(-1))

        def halo(X, mesh):
            if size == 1:
                return X
            buf = jnp.moveaxis(X[:, send], 1, 0)            # (size, rows, m)
            got = mpi4jax.alltoall(buf, comm=comm)          # got[s]: from s
            got = jnp.moveaxis(got, 0, 1).reshape(X.shape[0], -1)
            return X.at[:, recv].set(got, mode="drop")

        # each face and cell counted on exactly one rank: a cell where it is
        # owned, a face where the lower-numbered of its inner cells is owned
        cells_g = np.append(self.cells, np.iinfo(np.int64).max)
        fc = np.asarray(mesh_np.face_cells)[:2]
        side = np.where(fc < nc, fc, nc)
        g = cells_g[side]
        first = np.where(g[0] <= g[1], side[0], side[1])
        own_face = jnp.asarray(first < self.n_own)
        own_cell = jnp.asarray(np.arange(nc) < self.n_own)

        def owned(x, empty):
            x = jnp.asarray(x)
            if x.ndim == 0:
                return x
            if x.shape[-1] == nf:
                return jnp.where(own_face, x, empty)
            if x.shape[-1] == nc:
                return jnp.where(own_cell, x, empty)
            raise ValueError(f"reduce over {x.shape}: neither {nc} cells nor "
                             f"{nf} faces")

        def total(x):
            """Sum over the ranks, by alltoall: its transpose sums the
            cotangents of all ranks (mpi4jax transposes allreduce as the
            identity, right only for a cotangent equal on every rank)."""
            every = jnp.broadcast_to(x, (size,) + jnp.shape(x))
            return jnp.sum(mpi4jax.alltoall(every, comm=comm), axis=0)

        def chooser(op, local, empty):
            @jax.custom_jvp
            def reduce(x):
                return mpi4jax.allreduce(local(owned(x, empty)), op, comm=comm)

            @reduce.defjvp
            def _(primals, tangents):          # tangent at the attaining entries
                (x,), (t,) = primals, tangents
                v = reduce(x)
                hit = owned(x, empty) == v
                cnt = mpi4jax.allreduce(jnp.sum(hit.astype(t.dtype)), MPI.SUM,
                                        comm=comm)
                return v, total(jnp.sum(jnp.where(hit, t, 0.0))) / cnt
            return reduce

        def reduce_any(x):
            hit = jnp.any(owned(jax.lax.stop_gradient(x), False))
            return mpi4jax.allreduce(hit.astype(jnp.int32), MPI.MAX,
                                     comm=comm) > 0

        return dict(halo=halo, reduce_any=reduce_any,
                    reduce_min=chooser(MPI.MIN, jnp.min, jnp.inf),
                    reduce_max=chooser(MPI.MAX, jnp.max, -jnp.inf))


def split(mesh, comm, part=None):
    """This rank's local mesh (a BaseMesh: owned cells, then the halo
    layers; cut faces tagged ``halo``) and its :class:`Partition`.

    ``mesh``: the global mesh, the same on every rank.  ``part``: owner rank
    of every cell (default: coordinate bisection)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    nc = int(mesh.n_inner_cells)
    if int(mesh.dimension) < 2:
        raise NotImplementedError(
            "distributed 1-D runs: the 1-D LSQ stencil is sized by the widest "
            "cell of the whole mesh, not locally")
    if (np.asarray(mesh.boundary_face_cells)[:int(mesh.n_boundary_faces)]
            != np.asarray(mesh.face_cells)[0][np.asarray(
                mesh.boundary_face_face_indices)]).any():
        raise NotImplementedError("distributed run on a periodic mesh")
    cv = np.asarray(mesh.cell_vertices)[:, :nc]
    X = np.asarray(mesh.vertex_coordinates)
    dim = int(mesh.dimension)
    if part is None:
        part = bisection(_cell_centers(X[:dim], cv), size)
    part = np.asarray(part)
    if part.max() >= size:
        raise ValueError(f"partition into {part.max() + 1} pieces on {size} ranks")
    A = _vertex_adjacency(cv, X.shape[1])
    halos = [_halo_of(A, part == s) for s in range(size)]

    own = np.flatnonzero(part == rank)
    cells = np.concatenate([own, *halos[rank]])
    where = np.full(nc, -1, dtype=np.int64)
    where[cells] = np.arange(len(cells))
    halo_cells = np.concatenate(halos[rank])
    recv = {s: where[c] for s in range(size) if s != rank
            for c in [halo_cells[part[halo_cells] == s]] if len(c)}
    send = {}
    for s in range(size):
        if s == rank:
            continue
        theirs = np.concatenate(halos[s])
        c = theirs[part[theirs] == rank]               # sorted by global id
        if len(c):
            send[s] = where[c]

    local = _local_mesh(mesh, cells, cv, X[:dim])
    vertices = local.pop("vertices")
    lmesh = BaseMesh.from_arrays(**local)
    return lmesh, Partition(comm, part, cells, len(own),
                            len(own) + len(halos[rank][0]), vertices, send,
                            recv, nc)


def _cell_centers(X, cv):
    return X[:, cv].mean(axis=1)


def _local_mesh(mesh, cells, cv, X):
    """``from_arrays`` input of the local mesh: renumbered vertices, the
    global boundary faces among the local cells with their tags, every other
    boundary face (a cut) tagged ``halo``."""
    vertices, inv = np.unique(cv[:, cells], return_inverse=True)
    lcv = inv.reshape(cv[:, cells].shape)
    to_local = {int(v): i for i, v in enumerate(vertices)}
    n_bf = int(mesh.n_boundary_faces)
    bf_cells = np.asarray(mesh.boundary_face_cells)[:n_bf]
    bf_face = np.asarray(mesh.boundary_face_face_indices)[:n_bf]
    bf_tags = np.asarray(mesh.boundary_face_physical_tags)[:n_bf]
    cell_faces = np.asarray(mesh.cell_faces)
    faces_of = _local_faces_for_type(mesh.type)
    mine = np.zeros(int(mesh.n_inner_cells), dtype=bool)
    mine[cells] = True
    face_tags = {}
    for i in np.flatnonzero(mine[bf_cells]):
        c, f = bf_cells[i], bf_face[i]
        lf = int(np.flatnonzero(cell_faces[:, c] == f)[0])
        key = tuple(sorted(to_local[int(v)] for v in cv[list(faces_of[lf]), c]))
        face_tags[key] = int(bf_tags[i])
    sorted_tags = [int(t) for t in np.asarray(
        mesh.boundary_conditions_sorted_physical_tags)]
    names = dict(zip(sorted_tags, mesh.boundary_conditions_sorted_names))
    if HALO in names.values():
        raise ValueError(f"the mesh already uses the boundary tag {HALO!r}")
    halo_tag = max(sorted_tags + [-1]) + 1
    names[halo_tag] = HALO
    return dict(vertex_coordinates=X[:, vertices], cell_vertices=lcv,
                mesh_type=mesh.type, face_tags=face_tags, tag_names=names,
                default_tag=halo_tag, vertices=vertices)
