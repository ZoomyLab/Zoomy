"""Large meshes: SWE 2-D bump O2 + MOOD on N x N quads from a mesh file,
constant dt.  Rank 0 writes the file once (serial, timed); every rank then
reads only its own cells.
    mpiexec -n R python bench_large.py N STEPS DIR"""
import os
import sys
import time as clock
from mpi4py import MPI
import common as C
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.distributed import write_mesh
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
N, steps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
path = os.path.join(out, f"bump_{N}.h5")
t_write = 0.0
if comm.Get_rank() == 0 and not os.path.exists(path):
    t0 = clock.perf_counter()
    write_mesh(BaseMesh.create_2d((0, 10, 0, 10), N, N), path)
    t_write = clock.perf_counter() - t0
comm.Barrier()
_, nsm = C.bump_2d(2, nx=4, flow=(0.1, 0.05))      # the model only
dt = 0.2 * 10.0 / N
s = SymbolicHyperbolicSolverJax(time_end=dt, dt=dt, cfl_check_every=0,
                                comm=comm if comm.Get_size() > 1 else None)
comm.Barrier()
t0 = clock.perf_counter()
Q, A = s.setup(path, nsm)
comm.Barrier()
t_setup = clock.perf_counter() - t0
if comm.Get_rank() == 0:
    print(f"  setup {t_setup:.1f}s", flush=True)
t0 = clock.perf_counter()
s.run(Q, A)[1].block_until_ready()
comm.Barrier()
t_compile = clock.perf_counter() - t0
if comm.Get_rank() == 0:
    print(f"  compile + 1 step {t_compile:.1f}s", flush=True)
s.time_end = steps * dt
t0 = clock.perf_counter()
s.run(Q, A)[1].block_until_ready()
comm.Barrier()
t_run = clock.perf_counter() - t0
own = s.partition.n_own if s.partition else N * N
if comm.Get_rank() == 0:
    print(f"N={N} cells={N*N/1e6:.2f}M ranks={comm.Get_size()} owned/rank0={own} "
          f"local/rank0={Q.shape[1]}  write file {t_write:.1f}s  setup {t_setup:.1f}s  "
          f"compile {t_compile:.1f}s  {1e3 * t_run / steps:.2f} ms/step", flush=True)
