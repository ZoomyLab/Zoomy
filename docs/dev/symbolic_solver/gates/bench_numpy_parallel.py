"""Distributed NumPy backend (ANALYSIS 50): setup and march time against the number of ranks.

    mpiexec -n R python bench_numpy_parallel.py swe N STEPS              # SWE bump O2 + MOOD, N x N
    mpiexec -n R python bench_numpy_parallel.py vam NX STEPS SOLVER METHOD
        # VAM(1) Chorin channel NX x NX/10 (gate 21); SOLVER gmres | pyamg, METHOD petsc | krylov
    mpiexec -n R python bench_numpy_parallel.py visc N STEPS METHOD      # implicit diffusion (gate 31)

Constant dt; prints one line on rank 0.
"""
import os
import sys
import time as clock

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "matrix"))
import common as C  # noqa: E402
from zoomy_core.symsolver.numpy_backend import (SymbolicChorinSolverNumpy,  # noqa: E402
                                                SymbolicHyperbolicSolverNumpy)
from zoomy_core.symsolver.numpy_backend.parallel import MpiLinearSolver  # noqa: E402
from zoomy_core.symsolver.settings import Imex, LinearSolver  # noqa: E402

comm = MPI.COMM_WORLD
kind, N, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = comm if comm.Get_size() > 1 else None
t0 = clock.perf_counter()
if kind == "swe":
    dt = 0.2 * 10.0 / N
    mesh, nsm = C.bump_2d(2, nx=N, flow=(0.1, 0.05))
    s = SymbolicHyperbolicSolverNumpy(time_end=steps * dt, dt=dt, cfl_check_every=0, comm=c)
    Q, A = s.setup(mesh, nsm)
elif kind == "vam":
    from cases import vam_q0
    from gate21_parallel_chorin import B
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    pred, press, corr = B.split()
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=N, ny=N // 10)
    dt = 0.1 * 3.0 / N
    s = SymbolicChorinSolverNumpy(time_end=steps * dt, dt=dt, cfl_check_every=0, comm=c,
                                  pressure_linear_solver=LinearSolver(sys.argv[4]),
                                  mpi_linear_solver=MpiLinearSolver(sys.argv[5]))
    Q, A = s.setup(mesh, predictor_nsm(pred, press, corr), press, corr)
    Q = vam_q0([str(v) for v in s.nsm.state], np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]])
    A = s.initial_aux(Q, A)
else:
    from cases import IMPLICIT
    from gate31_parallel_implicit_diffusion import viscous
    mesh, nsm = viscous(3, nx=N)
    dt = 2e-3
    s = SymbolicHyperbolicSolverNumpy(time_end=steps * dt, dt=dt, cfl_check_every=0, comm=c,
                                      imex=Imex.lie, mpi_linear_solver=MpiLinearSolver(
                                          sys.argv[4]), **IMPLICIT)
    Q, A = s.setup(mesh, nsm)
t_setup = clock.perf_counter() - t0
comm.Barrier()
t1 = clock.perf_counter()
t, Q, _ = s.run(Q, A)
comm.Barrier()
t_run = clock.perf_counter() - t1
n = s.partition.n_global if s.partition else int(s.mesh_np.n_inner_cells)
own = s.partition.n_own if s.partition else n
if comm.Get_rank() == 0:
    print(f"{kind} cells={n} ranks={comm.Get_size()} owned/rank0={own} local/rank0={Q.shape[1]} "
          f"{' '.join(sys.argv[4:])}  setup {t_setup:.1f}s  {1e3 * t_run / steps:.1f} ms/step",
          flush=True)
