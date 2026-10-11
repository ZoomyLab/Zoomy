"""Gate 3d-f: foam (OpenFOAM 13, blockMesh hexahedra) vs JAX on the 3-D Euler
cases.  O1: same stencils -> round-off.  O2: foam's own stencils (fvc::grad,
cellLimited) -> both converge to the exact solution, foam - JAX shrinks with
the mesh.  ``mpi``: foam on 2 and 4 ranks (decomposePar) = foam serial.

    python gate3d_f_foam.py sod ORDER [N ...]     Sod along x, y, z (n x 2 x 2)
    python gate3d_f_foam.py wall ORDER [N]        wall box, pulse: mass, energy
    python gate3d_f_foam.py mpi ORDER             wall box 12^3: 2/4 ranks
    python gate3d_f_foam.py conv ORDER [N ...]    wall box, acoustic pulse (amp
                                                  0.01, smooth): L1|foam - jax| vs N
"""
import os
import sys
import tempfile
import time

import numpy as np

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import (SOD, TAGS, box, cell_centers, cell_volumes, conserved,
                          exact_riemann, nsm_of, primitive, sod_system)
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.euler import compressible_euler
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

WHAT = sys.argv[1] if len(sys.argv) > 1 else "sod"
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ARGS = [int(a) for a in sys.argv[3:]]
WORK = tempfile.mkdtemp(prefix="gate3d_foam_")
CFL = 0.45


def pulse(n, amp=1.0, width=0.02):
    """Off-centre Gaussian pulse; ``amp`` = 1: strong (a shock forms),
    small: an acoustic (smooth) pulse."""
    def ic(x):
        r2 = sum((x[d] - 0.4) ** 2 for d in range(3))
        b = amp * np.exp(-r2 / width)
        return conserved(1.0 + 0.5 * b, np.zeros(3), 1.0 + 2.0 * b)
    sm = compressible_euler(3, 1.4, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in TAGS[3]]), initial_conditions=IC.UserFunction(function=ic))
    return LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n), sm


def both(mesh, sm, t_end, nprocs=(1,)):
    """(JAX run, {nprocs: foam run}): (t, Q), Q in mesh order; plus the mesh
    the JAX solver used (centres, volumes)."""
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=CFL)
    Q0, A = s.setup(mesh, nsm_of(sm, ORDER))
    tj, Qj, _ = s.run(Q0, A)
    foam = {}
    for k in nprocs:
        f = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=CFL, **(
            {"nprocs": k} if k > 1 else {}))
        tf, Qf = f.run(s.mesh_np, nsm_of(sm, ORDER), np.asarray(Q0),
                       os.path.join(WORK, f"run{k}"))
        foam[k] = (float(tf), np.asarray(Qf))
    return (float(tj), np.asarray(Qj)), foam, s.mesh_np, np.asarray(Q0)


t0 = time.time()
if WHAT == "sod":
    ns = ARGS or [50, 100]
    print(f"Sod 3-D, O{ORDER}, HLL, minmod: foam vs JAX, t = 0.2")
    for n in ns:
        for axis in (0, 1, 2):
            (tj, Qj), foam, m, _ = both(box(3, n, axis), sod_system(3, axis), 0.2)
            tf, Qf = foam[1]
            x, vol = cell_centers(m)[axis], cell_volumes(m)
            ex = exact_riemann(SOD[0], SOD[1], (x - 0.5) / tj)
            l1 = lambda r: float(np.sum(np.abs(r - ex[0]) * vol) / vol.sum())   # noqa: E731
            dl1 = float(np.sum(np.abs(Qf[0] - Qj[0]) * vol) / vol.sum())
            print(f" n {n:4d} axis {'xyz'[axis]}: t jax/foam {tj:.6f}/{tf:.6f}  "
                  f"L1(rho) jax {l1(Qj[0]):.4e} foam {l1(Qf[0]):.4e}  "
                  f"L1|foam-jax| {dl1:.2e}  max|foam-jax| "
                  f"{np.abs(Qf - Qj).max():.2e}  ({time.time() - t0:.0f} s)", flush=True)
elif WHAT == "wall":
    n = ARGS[0] if ARGS else 12
    (tj, Qj), foam, m, Q0 = both(*pulse(n), 0.3)
    tf, Qf = foam[1]
    vol = cell_volumes(m)
    for lab, Q in (("jax", Qj), ("foam", Qf)):
        d = (Q - Q0) @ vol / (Q0 @ vol)
        print(f"wall {n}^3 O{ORDER} {lab}: t {tj if lab == 'jax' else tf:.4f}  "
              f"rel mass {d[0]:.1e}  rel energy {d[-1]:.1e}")
    print(f"max|foam-jax| / max|jax| {np.abs(Qf - Qj).max() / np.abs(Qj).max():.2e}  "
          f"L1(rho) {float(np.abs(Qf[0] - Qj[0]) @ vol / vol.sum()):.2e}  "
          f"({time.time() - t0:.0f} s)")
elif WHAT == "conv":
    prev = None
    for n in ARGS or [8, 16, 24]:
        (tj, Qj), foam, m, Q0 = both(*pulse(n, 0.01, 0.04), 0.3)
        vol = cell_volumes(m)
        d = float(np.abs(foam[1][1][0] - Qj[0]) @ vol / vol.sum())
        rate = "" if prev is None else f"  order {np.log(prev[1] / d) / np.log(n / prev[0]):.2f}"
        print(f"wall {n}^3 O{ORDER}: L1(rho) |foam-jax| {d:.3e}{rate}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        prev = (n, d)
elif WHAT == "mpi":
    (tj, Qj), foam, m, Q0 = both(*pulse(12), 0.3, nprocs=(1, 2, 4))
    ref = foam[1][1]
    for k in (2, 4):
        print(f"foam wall 12^3 O{ORDER}: {k} ranks vs serial max rel "
              f"{np.abs(foam[k][1] - ref).max() / np.abs(ref).max():.1e}  "
              f"(t {foam[k][0]:.4f} / {foam[1][0]:.4f})")
    print(f"({time.time() - t0:.0f} s)")
