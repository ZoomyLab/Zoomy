"""The 1-D dam break on AMReX with one refinement level on the waves (apptainer)."""
from pathlib import Path

import numpy as np

from dambreak import dambreak
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex

mesh, nsm = dambreak(order=2, n=128)
Q0 = nsm.sm.initial_conditions.apply(
    np.asarray(mesh.cell_centers)[:, :mesh.n_inner_cells], np.zeros((3, mesh.n_inner_cells)))

# level 1 (ratio 2) where |grad h| exceeds 0.1, regridded every 4 steps
solver = SymbolicHyperbolicSolverAmrex(
    time_end=0.5, cfl=0.9, amr=dict(max_level=1, ratio=2, regrid_int=4, tag=("h", "gradient", 0.1)))
time, Q = solver.run(mesh, nsm, Q0, Path("work_amrex_amr").resolve())    # Q: level 0, averaged down

h = Q[1]
cells = [int(ln.split()[-1]) for ln in solver.log.splitlines() if "AMR: level" in ln]
print(f"t = {time:.3f}, h in [{h.min():.5f}, {h.max():.5f}], mass = {h.sum() * 10.0 / h.size:.6f}")
print(f"cells per level at the end: {cells}")
