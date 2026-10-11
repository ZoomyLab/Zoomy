"""Gate 23: foam on OpenFOAM's own stencils and processor patches.

    python gate23_foam_parallel.py CASE [NPROCS ...]
    CASE: bump1 bump2 tri2 dry chorin1 chorin2

Serial foam vs the JAX template: first order to round-off, second order to
the discretization (L1: OpenFOAM's gradients, limiters, second derivatives).
Foam on N ranks (decomposePar, scotch) vs serial foam: round-off, Chorin to
the pressure tolerance; closed domains keep the mass."""
import os
import sys
import time as wall

import common
import numpy as np

from gate9_foam import FAILED, WORK, check
from zoomy_foam.symsolver import SymbolicChorinSolverFoam, SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

# serial foam vs JAX (L1 rel; bump1: max rel), N ranks vs serial (max rel)
BOUND = dict(bump1=(1e-15, 1e-15), bump2=(4e-3, 2e-15), tri2=(6e-3, 4e-15),
             dry=(7e-2, 1e-14), chorin1=(1e-2, 1e-11), chorin2=(1e-2, 1e-11))
CLOSED = ("bump1", "bump2", "tri2", "dry")


def hyperbolic(case):
    kw = dict(cfl=0.9, mood=True)
    if case in ("bump1", "bump2"):
        mesh, nsm = common.bump_2d(int(case[-1]), nx=40, flow=(0.3, 0.1))
        t_end = 0.5
    elif case == "tri2":                    # unstructured: gmshToFoam
        from gate16_parallel import tri_bump
        mesh, nsm = tri_bump(2)
        t_end = 0.5
    else:
        mesh, nsm = common.dry_dambreak_2d(nx=24)
        t_end, kw = 0.02, dict(cfl=0.45, mood=True)
    sj = SymbolicHyperbolicSolverJax(time_end=t_end, **kw)
    Q0, A0 = sj.setup(mesh, nsm)
    _, Qj, _ = sj.run(Q0, A0)

    def foam(n):
        s = SymbolicHyperbolicSolverFoam(time_end=t_end, nprocs=n, **kw)
        t0 = wall.perf_counter()
        t, Q = s.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/{case}_np{n}")
        return t, Q, wall.perf_counter() - t0
    return np.asarray(Qj), foam


def chorin(case):
    import gate7_chorin as g7
    to = int(case[-1])
    s, Q, Qaux = g7.new_solver(to)
    _, Qj, _ = s.run(Q, Qaux)

    def foam(n):
        sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                      time_order=to, nprocs=n,
                                      pressure_preconditioner=os.environ.get("PREC", "gamg"))
        t0 = wall.perf_counter()
        t, Qf = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q),
                       f"{WORK}/{case}_np{n}")
        return t, Qf, wall.perf_counter() - t0
    return np.asarray(Qj), foam


if __name__ == "__main__":
    case, nps = sys.argv[1], [int(a) for a in sys.argv[2:]]
    Qj, foam = (chorin if case.startswith("chorin") else hyperbolic)(case)
    t, Qs, w = foam(1)
    d = np.abs(Qs - Qj)
    mx, l1 = d.max() / np.abs(Qj).max(), d.sum() / np.abs(Qj).sum()
    print(f"{case}: serial foam vs JAX: max rel {mx:.1e}, L1 rel {l1:.1e}, "
          f"mass foam-jax {abs(Qs[1].sum() - Qj[1].sum()):.1e}, t {t:.6g} [{w:.0f}s]", flush=True)
    check(f"{case} serial vs JAX {'max' if case == 'bump1' else 'L1'} rel",
          mx if case == "bump1" else l1, BOUND[case][0])
    if case in CLOSED:
        check(f"{case} mass foam-jax", abs(Qs[1].sum() - Qj[1].sum()), 1e-12)
    for n in nps:
        t, Qp, w = foam(n)
        r = np.abs(Qp - Qs).max() / np.abs(Qs).max()
        print(f"{case}: {n} ranks vs serial foam: max rel {r:.1e}, "
              f"mass {abs(Qp[1].sum() - Qs[1].sum()):.1e}, t {t:.6g} [{w:.0f}s]", flush=True)
        check(f"{case} {n} ranks vs serial max rel", r, BOUND[case][1])
        if case in CLOSED:
            check(f"{case} {n} ranks mass", abs(Qp[1].sum() - Qs[1].sum()), 1e-12)
    print("G23_DONE" if not FAILED else f"G23_FAILED {FAILED}")
