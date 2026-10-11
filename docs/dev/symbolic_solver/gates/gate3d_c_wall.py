"""Gate 3d-c: closed box with walls (Euler SystemModel, JAX symbolic solver,
O2 + MOOD, HLL): a centred Gaussian pressure / density pulse in [0,1]^3,
t = 0.6 (the pulse reflects off all six walls).
Total mass and energy conserved to round-off; the solution symmetric under
the box's mirror planes x -> 1 - x (y, z) and the axis permutations.

    python gate3d_c_wall.py [N] [ORDER]        (default 16, 2)
"""
import sys
import time

import numpy as np

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import TAGS, cell_centers, cell_volumes, conserved, nsm_of
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.euler import compressible_euler
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DIM = 3


def ic(x):
    r2 = sum((x[d] - 0.5) ** 2 for d in range(DIM))
    bump = np.exp(-r2 / 0.02)
    return conserved(1.0 + 0.5 * bump, np.zeros(DIM), 1.0 + 2.0 * bump)


sm = compressible_euler(DIM, 1.4, boundary_conditions=BC.BoundaryConditions(
    [BC.Wall(tag=t) for t in TAGS[DIM]]), initial_conditions=IC.UserFunction(function=ic))
mesh = LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=N, ny=N, nz=N)
s = SymbolicHyperbolicSolverJax(time_end=0.6, cfl=0.45)
t0 = time.time()
Q0, A = s.setup(mesh, nsm_of(sm, ORDER))
t, Q, _ = s.run(Q0, A)
Q0, Q = np.asarray(Q0), np.asarray(Q)
vol = cell_volumes(s.mesh_np)
tot0, tot = Q0 @ vol, Q @ vol
print(f"walls, {N}^3, O{ORDER}, t = {float(t):.3f}, mood {s.mood_active}  ({time.time() - t0:.1f} s)")
print(f"mass   {tot0[0]:.15f} -> {tot[0]:.15f}  rel {abs(tot[0] - tot0[0]) / tot0[0]:.1e}")
print(f"energy {tot0[-1]:.15f} -> {tot[-1]:.15f}  rel {abs(tot[-1] - tot0[-1]) / tot0[-1]:.1e}")
print(f"momentum {tot[1:-1]}  (pulse symmetric: 0)")
# symmetry: map every cell to its image under a mirror / permutation
x = cell_centers(s.mesh_np)
ijk = np.rint(x * N - 0.5).astype(int)                 # cell index per axis
lin = lambda a: (a[0] * N + a[1]) * N + a[2]           # noqa: E731
pos = np.empty(N ** 3, dtype=int)
pos[lin(ijk)] = np.arange(Q.shape[1])
rho = Q[0]
for name, img, sign_rows in (
        ("x->1-x", (N - 1 - ijk[0], ijk[1], ijk[2]), (1,)),
        ("y->1-y", (ijk[0], N - 1 - ijk[1], ijk[2]), (2,)),
        ("z->1-z", (ijk[0], ijk[1], N - 1 - ijk[2]), (3,)),
        ("x<->y", (ijk[1], ijk[0], ijk[2]), ()),
        ("x<->z", (ijk[2], ijk[1], ijk[0]), ())):
    other = pos[lin(np.asarray(img))]
    print(f"symmetry {name}: max|rho - rho(image)| {np.abs(rho - rho[other]).max():.1e}  "
          f"max|E - E(image)| {np.abs(Q[-1] - Q[-1][other]).max():.1e}")
