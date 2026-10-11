"""Gate 3d-d: distributed (MPI ranks, mpi4jax) = serial on 3-D hexahedra
(Euler SystemModel, gate16 style): the run is gathered to rank 0 and compared
with the serial run there.
    mpiexec -n 4 python gate3d_d_parallel.py"""
import time as clock

import numpy as np
from mpi4py import MPI

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import TAGS, box, conserved, nsm_of, sod_system
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.euler import compressible_euler
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def pulse(order, n=12):
    def ic(x):
        r2 = sum((x[d] - 0.4) ** 2 for d in range(3))
        b = np.exp(-r2 / 0.02)
        return conserved(1.0 + 0.5 * b, np.zeros(3), 1.0 + 2.0 * b)
    sm = compressible_euler(3, 1.4, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in TAGS[3]]), initial_conditions=IC.UserFunction(function=ic))
    return (LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n),
            nsm_of(sm, order))


CASES = (("Euler 3-D wall box 12^3, off-centre pulse O1", lambda: pulse(1), 0.3),
         ("Euler 3-D wall box 12^3, off-centre pulse O2 + MOOD", lambda: pulse(2), 0.3),
         ("Euler 3-D Sod along y, 60 x 4 x 4, O2 + MOOD",
          lambda: (box(3, 60, axis=1, thin=4), nsm_of(sod_system(3, 1), 2)), 0.2))
for label, make, t_end in CASES:
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.45, comm=comm)
    Q, A = s.setup(mesh, nsm)
    t0 = clock.perf_counter()
    t, Q, A = s.run(Q, A)
    Q.block_until_ready()
    wall = clock.perf_counter() - t0
    Qg = s.partition.gather(Q) if s.partition else np.asarray(Q)
    if rank == 0:
        mesh, nsm = make()
        r = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.45)
        Qs, As = r.setup(mesh, nsm)
        _, Qs, _ = r.run(Qs, As)
        Qs = np.asarray(Qs)
        err = np.abs(Qg - Qs).max() / max(np.abs(Qs).max(), 1e-300)
        own = s.partition.n_own if s.partition else Qs.shape[1]
        print(f"{label}: ranks {comm.Get_size()}  t={float(t):.4g}  "
              f"max rel |par-ser| {err:.1e}  owned/rank0 {own} of {Qs.shape[1]}  "
              f"run {wall:.2f}s", flush=True)
