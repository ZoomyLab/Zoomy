"""The 1-D dam break on AMReX (apptainer), compared with NumPy."""
from pathlib import Path

import numpy as np

from dambreak import dambreak
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex

mesh, nsm = dambreak(order=2, n=300)
Q0 = nsm.sm.initial_conditions.apply(
    np.asarray(mesh.cell_centers)[:, :mesh.n_inner_cells], np.zeros((3, mesh.n_inner_cells)))

solver = SymbolicHyperbolicSolverAmrex(time_end=0.5, cfl=0.9)     # ranks=4: MPI, gpu=True: CUDA
time, Q = solver.run(mesh, nsm, Q0, Path("work_amrex").resolve())  # prints, builds (cached), runs

reference = SymbolicHyperbolicSolverNumpy(time_end=0.5, cfl=0.9)
Qn, Qauxn = reference.setup(mesh, nsm)
_, Qn, _ = reference.run(Qn, Qauxn)
h = Q[1]
print(f"t = {time:.3f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
print(f"max |h_amrex - h_numpy| = {np.abs(h - Qn[1]).max():.2e}")
