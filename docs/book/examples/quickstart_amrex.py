"""Quick start: the same NSM on AMReX (apptainer; ranks=N for MPI, gpu=True for CUDA)."""
from pathlib import Path

import numpy as np

from bump import bump
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex

mesh, nsm = bump(order=2, nx=40)
Q0 = nsm.sm.initial_conditions.apply(
    np.asarray(mesh.cell_centers)[:, :mesh.n_inner_cells], np.zeros((4, mesh.n_inner_cells)))

solver = SymbolicHyperbolicSolverAmrex(time_end=2.5, cfl=0.9, mood=False)
time, Q = solver.run(mesh, nsm, Q0, Path("work_amrex").resolve())         # prints, builds (cached), runs

h = Q[1]
print(f"t = {time:.3f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
ref = np.load("quickstart_jax.npy")
print(f"max |h_amrex - h_jax| = {np.abs(h - ref[1]).max():.2e}")
