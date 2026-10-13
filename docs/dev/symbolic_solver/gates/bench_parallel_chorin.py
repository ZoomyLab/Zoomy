"""Chorin on several GPUs: VAM(1) bump channel (gate 21's case), nx x nx/10
cells, constant dt = 0.15 dx, STEPS steps; pressure by AmgX (1 rank: AmgX
FGMRES + AMG; R ranks: global FGMRES, per-rank AmgX V-cycle).
    mpiexec -n R python -m zoomy_jax.gpurank bench_parallel_chorin.py NX STEPS [SOLVER]"""
import sys
import time as wall
import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
import common  # noqa: F401
from zoomy_core.mesh import LSQMesh
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from gate7_chorin import pred_nsm
from gate21_parallel_chorin import B
from zoomy_core.symsolver.settings import LinearSolver

comm = MPI.COMM_WORLD
NX, STEPS = int(sys.argv[1]), int(sys.argv[2])
SOLVER = LinearSolver(sys.argv[3] if len(sys.argv) > 3 else "amgx")
dt = 0.15 * 3.0 / NX
pred, press, corr = B.split()
t0 = wall.perf_counter()
s = SymbolicChorinSolverJax(time_end=dt, dt=dt, cfl=0.9, cfl_check_every=0,
                            pressure_linear_solver=SOLVER,
                            comm=comm if comm.Get_size() > 1 else None)
mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
names = [str(v) for v in s.nsm.state]
xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
q = np.zeros(Q.shape)
b = B.BUMP(xc)
q[names.index("b")] = b
q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
Q = jnp.asarray(q)
A = s.initial_aux(Q, A)
comm.Barrier()
t1 = wall.perf_counter()
if comm.Get_rank() == 0:
    print(f"  setup {t1 - t0:.0f} s", flush=True)
s.run(Q, A)[1].block_until_ready()             # compile + 1 step
comm.Barrier()
t2 = wall.perf_counter()
if comm.Get_rank() == 0:
    print(f"  compile + 1 step {t2 - t1:.0f} s", flush=True)
s.time_end = STEPS * dt
_, Qe, _ = s.run(Q, A)
Qe.block_until_ready()
comm.Barrier()
t3 = wall.perf_counter()
if comm.Get_rank() == 0:
    print(f"VAM(1) Chorin {NX}x{NX // 10} = {NX * NX // 10} cells, {comm.Get_size()} GPU(s), "
          f"{SOLVER}: setup {t1 - t0:.0f} s, compile {t2 - t1:.0f} s, {STEPS} steps "
          f"{1e3 * (t3 - t2) / STEPS:.1f} ms/step, finite {bool(np.isfinite(np.asarray(Qe)).all())}",
          flush=True)
