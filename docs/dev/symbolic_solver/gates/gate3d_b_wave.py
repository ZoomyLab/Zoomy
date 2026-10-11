"""Gate 3d-b: smooth density wave advected diagonally (Euler SystemModel, JAX
symbolic solver), periodic box [0,1]^dim, constant u and p:
    rho = 1 + 0.2 sin(2 pi (sum_d x_d - t sum_d u_d)),  exact.
t = 0.1 (O1 over a full period at <= 24 cells per wavelength is
pre-asymptotic: the wave is gone).  L1(rho) and its observed order.

    python gate3d_b_wave.py DIM ORDER LIMITER [N ...]
    (defaults: 3 2 none, N = 8 16 24 in 3-D, 16 32 64 in 2-D)
"""
import sys
import time

import numpy as np

from euler_common import (box, cell_centers, cell_volumes, nsm_of, primitive,
                          wave_exact, wave_system)
from zoomy_core.mesh import LSQMesh
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LIMITER = sys.argv[3] if len(sys.argv) > 3 else "none"
NS = [int(a) for a in sys.argv[4:]] or ([8, 16, 24] if DIM == 3 else [16, 32, 64])
VEL = {1: (1.0,), 2: (1.0, 0.5), 3: (1.0, 0.5, 0.25)}[DIM]
T_END = 0.1


def mesh_of(n):
    if DIM == 1:
        return LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
    if DIM == 2:
        return LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n)
    return LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n)


print(f"wave {DIM}-D, O{ORDER}, limiter {LIMITER}, HLL, u = {VEL}, t = {T_END:.4f}")
errs, t0 = [], time.time()
for n in NS:
    s = SymbolicHyperbolicSolverJax(time_end=T_END, cfl=0.45)
    tic = time.time()
    Q, Qaux = s.setup(mesh_of(n), nsm_of(wave_system(DIM, VEL), ORDER,
                                         limiter=LIMITER))
    t, Q, _ = s.run(Q, Qaux)
    Q = np.asarray(Q)
    x, vol = cell_centers(s.mesh_np), cell_volumes(s.mesh_np)
    rho, u, p = primitive(Q)
    e = float(np.sum(np.abs(rho - wave_exact(x, float(t), VEL)) * vol) / vol.sum())
    errs.append(e)
    print(f" n {n:3d}  cells {Q.shape[1]:6d}  L1(rho) {e:.4e}  max|u-u0| "
          f"{np.abs(u - np.asarray(VEL)[:, None]).max():.1e}  max|p-1| "
          f"{np.abs(p - 1).max():.1e}  ({time.time() - tic:.1f} s)", flush=True)
e = np.asarray(errs)
eoc = np.log(e[:-1] / e[1:]) / np.log(np.asarray(NS[1:]) / np.asarray(NS[:-1]))
print(f"EOC: {' '.join(f'{v:.2f}' for v in eoc)}   wall {time.time() - t0:.1f} s")
