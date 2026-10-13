"""Distributed runs on JAX: the `halo` exchange and the global reductions of the runtime's name
table, one MPI rank per device.  Used by `zoomy_jax.symsolver.runtime` when the solver gets
`comm=`; the exchanges are mpi4jax collectives, differentiable in both directions.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from zoomy_core.mesh.distributed import HALO  # noqa: F401  (re-exported)


def _mpi4jax():
    """Import mpi4jax and register the AD rules of its `alltoall`; a clear error when it is
    missing.
    """
    try:
        import mpi4jax
    except ImportError as e:
        raise ImportError(
            "distributed runs need mpi4jax (and mpi4py): pip install mpi4jax "
            "-- see docs/dev/symbolic_solver/PARALLEL.md") from e
    _register_alltoall_ad()
    return mpi4jax


def bind_device(comm):
    """Bind this rank to one GPU of its node (round robin), before JAX starts if possible.  No-op on
    the CPU, with one GPU, or when `CUDA_VISIBLE_DEVICES` is set.
    """
    import os
    from mpi4py import MPI
    from jax._src import xla_bridge
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local, n_local = node.Get_rank(), node.Get_size()
    node.Free()
    if not xla_bridge.backends_are_initialized():
        try:
            n_gpu = len(os.listdir("/proc/driver/nvidia/gpus"))
        except OSError:
            return                                   # no NVIDIA driver
        if n_gpu > 1 and "cpu" not in os.environ.get("JAX_PLATFORMS", ""):
            import warnings
            warnings.warn("GPUs bound inside the run: libraries using the CUDA "
                          "runtime directly (AmgX, mpi4jax's copies) still see "
                          "GPU 0 -- launch with 'python -m zoomy_jax.gpurank'")
            # JAX's config (JAX reads it, not the environment, from here on)
            # and the environment (every other CUDA user in the process --
            # AmgX would pick device 0)
            gpu = str(local % n_gpu)
            jax.config.update("jax_cuda_visible_devices", gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        n = n_gpu
    else:
        gpus = [d for d in jax.local_devices() if d.platform == "gpu"]
        n = len(gpus)
        if n > 1:
            jax.config.update("jax_default_device", gpus[local % n])
    if n > 1 and n_local > n and local == 0:
        import warnings
        warnings.warn(f"{n_local} ranks on a node with {n} GPUs: ranks share GPUs")


_AD_DONE = []


def _register_alltoall_ad():
    """Register the JVP and transpose of mpi4jax's `alltoall` (linear; its transpose is
    `alltoall`), once.
    """
    if _AD_DONE:
        return
    from jax.interpreters import ad
    from mpi4jax._src.collective_ops.alltoall import mpi_alltoall_p as p

    def jvp(primals, tangents, comm):
        """JVP of `alltoall`: `alltoall` of the tangent."""
        (x,), (t,) = primals, tangents
        return p.bind(x, comm=comm), p.bind(ad.instantiate_zeros(t), comm=comm)

    def transpose(ct, x, comm):
        """Transpose of `alltoall`: `alltoall` of the cotangent."""
        return [p.bind(ad.instantiate_zeros(ct), comm=comm)]

    ad.primitive_jvps[p] = jvp
    ad.primitive_transposes[p] = transpose
    _AD_DONE.append(True)


# ── partition ───────────────────────────────────────────────────────────────
_LOCAL = [False]


class no_halo:
    """Context manager inside which `halo` is the identity, giving the rank's own block of an
    operator (block-Jacobi preconditioner).
    """

    def __enter__(self):
        """Switch the halo exchange off."""
        self._was, _LOCAL[0] = _LOCAL[0], True

    def __exit__(self, *exc):
        """Restore the previous halo state."""
        _LOCAL[0] = self._was


def krylov(part, nc):
    """The owned-cell mask `(nc,)` and the global sum (allreduce) a distributed Krylov solve
    needs.

    Returns:
        `(mask, gsum)`; inside `custom_linear_solve` nothing is differentiated through `gsum`.
    """
    mpi4jax = _mpi4jax()
    from mpi4py import MPI
    comm = part.comm.Clone()
    return (jnp.asarray(np.arange(nc) < part.n_own),
            lambda x: mpi4jax.allreduce(x, MPI.SUM, comm=comm))


def blocks(part, mesh_np):
    """The `halo` exchange and the global reductions of the runtime's name table for this rank's
    piece, each face and cell counted once.

    Args:
        part: `zoomy_core.mesh.distributed.Partition` of this rank
        mesh_np: the rank's local mesh
    Returns:
        dict with `halo`, `reduce_any`, `reduce_min`, `reduce_max`
    """
    mpi4jax = _mpi4jax()
    from mpi4py import MPI
    comm, size = part.comm.Clone(), part.size
    nc, nf = int(mesh_np.n_inner_cells), int(mesh_np.n_faces)
    if nc == nf:
        raise ValueError("local mesh with as many faces as cells: the "
                         "reductions cannot tell them apart")
    m = max(1, comm.allreduce(max([len(v) for v in part.send.values()]
                                  + [0]), op=MPI.MAX))
    send = np.zeros((size, m), dtype=np.int64)          # pad: cell 0
    recv = np.full((size, m), nc, dtype=np.int64)       # pad: dropped
    for s, idx in part.send.items():
        send[s, :len(idx)] = idx
    for s, idx in part.recv.items():
        recv[s, :len(idx)] = idx
    send, recv = jnp.asarray(send), jnp.asarray(recv.reshape(-1))

    def halo(X, mesh):
        """Refresh the halo copies in `X` from their owners by `alltoall`."""
        if size == 1 or jnp.ndim(X) == 0 or _LOCAL[0]:
            return X
        buf = jnp.moveaxis(X[:, send], 1, 0)            # (size, rows, m)
        got = mpi4jax.alltoall(buf, comm=comm)          # got[s]: from s
        got = jnp.moveaxis(got, 0, 1).reshape(X.shape[0], -1)
        return X.at[:, recv].set(got, mode="drop")

    # each face and cell counted on exactly one rank: a cell where it is
    # owned, a face where the lower-numbered of its inner cells is owned
    cells_g = np.append(part.gids, np.iinfo(np.int64).max)
    fc = np.asarray(mesh_np.face_cells)[:2]
    side = np.where(fc < nc, fc, nc)
    g = cells_g[side]
    first = np.where(g[0] <= g[1], side[0], side[1])
    own_face = jnp.asarray(first < part.n_own)
    own_cell = jnp.asarray(np.arange(nc) < part.n_own)

    def owned(x, empty):
        """Entries of `x` this rank owns (faces or cells); the others are replaced by `empty`."""
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
        """Sum over the ranks by `alltoall`, whose transpose sums the cotangents of all ranks
        (mpi4jax transposes allreduce as the identity).
        """
        every = jnp.broadcast_to(x, (size,) + jnp.shape(x))
        return jnp.sum(mpi4jax.alltoall(every, comm=comm), axis=0)

    def chooser(op, local, empty):
        """A differentiable global min / max: `local` over the owned entries, then an allreduce
        with `op`.
        """
        @jax.custom_jvp
        def reduce(x):
            """Global extremum of the owned entries of `x`."""
            return mpi4jax.allreduce(local(owned(x, empty)), op, comm=comm)

        @reduce.defjvp
        def _(primals, tangents):          # tangent at the attaining entries
            """Tangent at the attaining entries, averaged over ties."""
            (x,), (t,) = primals, tangents
            v = reduce(x)
            hit = owned(x, empty) == v
            cnt = mpi4jax.allreduce(jnp.sum(hit.astype(t.dtype)), MPI.SUM,
                                    comm=comm)
            return v, total(jnp.sum(jnp.where(hit, t, 0.0))) / cnt
        return reduce

    def reduce_any(x):
        """Global any over the owned entries (no gradient)."""
        hit = jnp.any(owned(jax.lax.stop_gradient(x), False))
        return mpi4jax.allreduce(hit.astype(jnp.int32), MPI.MAX,
                                 comm=comm) > 0

    return dict(halo=halo, reduce_any=reduce_any,
                reduce_min=chooser(MPI.MIN, jnp.min, jnp.inf),
                reduce_max=chooser(MPI.MAX, jnp.max, -jnp.inf))
