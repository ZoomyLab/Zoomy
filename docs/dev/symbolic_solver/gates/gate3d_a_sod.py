"""Gate 3d-a: Sod shock tube (Euler as a SystemModel) on the JAX symbolic
solver, 1-D and along x, y, z of a thin 3-D box (n x 2 x 2 hexahedra,
Extrapolation everywhere), vs the exact Riemann solution (Toro ch. 4).
Reports L1(rho) and its observed order, the difference between the three
directions and the difference to the 1-D run at the same dt (the CFL step
carries 1/dim: the 1-D run uses CFL/3).

    python gate3d_a_sod.py DIM ORDER [N ...]   (DIM 2 or 3; N: 50 100 200)
"""
import sys
import time

import numpy as np

from euler_common import (SOD, box, cell_centers, cell_volumes, exact_riemann,
                          nsm_of, primitive, sod_system)
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NS = [int(a) for a in sys.argv[3:]] or [50, 100, 200]
AXES = tuple(range(DIM))
T_END = 0.2


CFL = 0.45


def run(dim, axis, n):
    mesh = box(dim, n, axis)
    s = SymbolicHyperbolicSolverJax(time_end=T_END, cfl=CFL * dim / DIM)
    Q, Qaux = s.setup(mesh, nsm_of(sod_system(dim, axis), ORDER))
    t, Q, _ = s.run(Q, Qaux)
    Q = np.asarray(Q)
    x = cell_centers(s.mesh_np)[axis]
    vol = cell_volumes(s.mesh_np)
    rho, u, p = primitive(Q)
    ex = exact_riemann(SOD[0], SOD[1], (x - 0.5) / float(t))
    err = float(np.sum(np.abs(rho - ex[0]) * vol) / vol.sum())
    # the profile along the axis (cells sorted by x_axis; columns averaged)
    o = np.argsort(x, kind="stable")
    k = Q.shape[1] // n
    prof = np.stack([rho, u[axis], p])[:, o].reshape(3, n, k)
    spread = float(np.abs(prof - prof[:, :, :1]).max())     # across the column
    perp = float(np.abs(np.delete(u, axis, 0)).max()) if dim > 1 else 0.0
    return err, prof.mean(axis=2), spread, perp, float(t)


print(f"Sod, O{ORDER}, HLL, minmod, primitive reconstruction, t = {T_END}, "
      f"CFL {CFL} ({DIM}-D) / {CFL / DIM:.3f} (1-D)")
print(" n    | 1-D L1     | " + " ".join(f"{'xyz'[a]} L1      " for a in AXES)
      + " | max|x-other| | max|nD-1D| | col spread | u_perp")
errs = {k: [] for k in ("1d",) + AXES}
t0 = time.time()
for n in NS:
    e1, p1, _, _, _ = run(1, 0, n)
    errs["1d"].append(e1)
    res = [run(DIM, a, n) for a in AXES]
    for a in AXES:
        errs[a].append(res[a][0])
    dxy = max(np.abs(res[0][1] - res[a][1]).max() for a in AXES[1:])
    d1 = max(np.abs(r[1] - p1).max() for r in res)
    spread = max(r[2] for r in res)
    perp = max(r[3] for r in res)
    print(f" {n:4d} | {e1:.4e} | " + " ".join(f"{r[0]:.4e}" for r in res)
          + f" | {dxy:.1e}      | {d1:.1e}    | {spread:.1e}    | {perp:.1e}", flush=True)
for k, lab in (("1d", "1-D"),) + tuple((a, "xyz"[a]) for a in AXES):
    e = np.asarray(errs[k])
    eoc = np.log(e[:-1] / e[1:]) / np.log(np.asarray(NS[1:]) / np.asarray(NS[:-1]))
    print(f"EOC {lab}: {' '.join(f'{v:.2f}' for v in eoc)}")
print(f"wall {time.time() - t0:.1f} s")
