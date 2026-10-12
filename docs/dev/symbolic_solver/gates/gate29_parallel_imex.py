"""Gate 29: the IMEX march (implicit-Euler source by cell-local Newton) on
MPI ranks vs serial: gate 6's stiff cases (Manning n = 0.05, nu = 0.01).
    mpiexec -n R python gate29_parallel_imex.py"""
import time as clock
import numpy as np
from mpi4py import MPI
from gate6_imex import CASES
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
for label, make, t_end, cfl in CASES:
    out = []
    for c in (None, comm):
        mesh, nsm = make()
        s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, imex="lie", comm=c)
        Q, A = s.setup(mesh, nsm)
        t0 = clock.perf_counter()
        t, Q, A = s.run(Q, A)
        Q.block_until_ready()
        out.append((s.partition.gather(Q) if s.partition else np.asarray(Q),
                    float(t), clock.perf_counter() - t0))
    if comm.Get_rank() == 0:
        (Qs, ts, _), (Qp, tp, wp) = out
        d = np.abs(Qp - Qs).max() / np.abs(Qs).max()
        print(f"IMEX {label}: ranks {comm.Get_size()}  t={tp:.3g}/{ts:.3g}  "
              f"max rel |par-ser| {d:.1e}  finite {bool(np.isfinite(Qp).all())}  "
              f"[run {wp:.1f}s]", flush=True)
