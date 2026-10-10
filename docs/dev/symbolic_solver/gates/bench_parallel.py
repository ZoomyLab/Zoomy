"""Distributed scaling: SWE 2-D bump O2 + MOOD, N x N quads, constant dt.
    mpiexec -n R bash -c 'CUDA_VISIBLE_DEVICES=$PMI_RANK python bench_parallel.py N STEPS'
Prints setup (mesh build + split), compile (first run) and the march time."""
import sys
import time as clock
from mpi4py import MPI
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
import jax

comm = MPI.COMM_WORLD
N, steps = int(sys.argv[1]), int(sys.argv[2])
dt = 0.2 * 10.0 / N                       # CFL ~0.45 at |u| + c ~ 2.3
t0 = clock.perf_counter()
mesh, nsm = C.bump_2d(2, nx=N, flow=(0.1, 0.05))
t_mesh = clock.perf_counter() - t0
s = SymbolicHyperbolicSolverJax(time_end=steps * dt, dt=dt, cfl_check_every=0,
                                comm=comm if comm.Get_size() > 1 else None)
Q, A = s.setup(mesh, nsm)
t_setup = clock.perf_counter() - t0
t1 = clock.perf_counter()
s.time_end = dt                            # compile (time_end is an input)
s.run(Q, A)[1].block_until_ready()
t_compile = clock.perf_counter() - t1
s.time_end = steps * dt
comm.Barrier()
t2 = clock.perf_counter()
_, Q2, _ = s.run(Q, A)
Q2.block_until_ready()
comm.Barrier()
t_run = clock.perf_counter() - t2
if comm.Get_rank() == 0:
    own = s.partition.n_own if s.partition else N * N
    print(f"N={N} cells={N*N} ranks={comm.Get_size()} dev={jax.devices()[0].platform} "
          f"owned/rank0={own} local/rank0={Q.shape[1]}  mesh {t_mesh:.1f}s  "
          f"setup {t_setup:.1f}s  compile {t_compile:.1f}s  "
          f"{1e3 * t_run / steps:.2f} ms/step", flush=True)
