"""Halo exchange alone: N x N quads split over the ranks, 4 rows, 1000
exchanges inside one compiled loop.
    mpiexec -n 4 python bench_halo.py N   (MPI4JAX_USE_CUDA_MPI=0/1)"""
import os
import sys
import time as clock
import jax
import jax.numpy as jnp
from jax import lax
from mpi4py import MPI
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_jax.symsolver.parallel import bind_device, split

comm = MPI.COMM_WORLD
bind_device(comm)
N, reps = int(sys.argv[1]), 1000
mesh, part = split(BaseMesh.create_2d((0, 10, 0, 10), N, N), comm)
halo = part.blocks(mesh)["halo"]
X = jnp.ones((4, int(mesh.n_inner_cells)))
loop = jax.jit(lambda X: lax.fori_loop(0, reps, lambda i, X: halo(X, None) * 1.0000001, X))
loop(X).block_until_ready()
comm.Barrier()
t0 = clock.perf_counter()
loop(X).block_until_ready()
comm.Barrier()
dt = (clock.perf_counter() - t0) / reps
n_halo = sum(len(v) for v in part.recv.values())
if comm.Get_rank() == 0:
    print(f"N={N} ranks={comm.Get_size()} dev={X.devices().pop().platform} halo cells/rank0 "
          f"{n_halo} ({n_halo * 4 * 8 / 1024:.0f} KiB)  CUDA-aware={os.environ.get('MPI4JAX_USE_CUDA_MPI', '0')}  "
          f"{1e6 * dt:.0f} us/exchange", flush=True)
