"""Gate 11: IMEX template on foam vs on JAX (gate 6 cases: stiff friction
and viscosity, implicit-Euler source by cell-local Newton), and foam on N
ranks vs serial foam.

JAX takes the exact Jacobian (jvp), foam one-sided differences: where the
stencils coincide (1-D, 2-D first order) they agree to the Newton tolerance.
2-D second order: OpenFOAM's gradients and limiters, wall shocks -> L1, and
the L1 difference shrinks with the mesh.  N ranks: the Newton stop test is
global (one iteration count on every rank), but its norm sums in another
order: a step whose ||R|| sits at the tolerance may take one update more or
less, so parallel = serial to the Newton tolerance (1-D: 3.5e-12 at tol 1e-8,
1.3e-13 at tol 1e-12), else round-off.
    python gate11_foam_imex.py [CASE ...]   CASE: 1d 2d1 2d par
"""
import sys

import common as C
import numpy as np

from gate6_imex import CASES, stiff
from gate9_foam import FAILED, WORK, check
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from zoomy_core.symsolver.settings import Imex

ONE_D, TWO_D = CASES
# foam vs JAX: max rel (1-D, 2-D O1: Newton tolerance), L1 rel (2-D O2)
BOUND = {"1d": 2e-9, "2d1": 1e-12, "2d": 2e-3}


def run(make, t_end, cfl, tag, nprocs=(1,)):
    mesh, nsm = make()
    sj = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, imex=Imex.lie)
    Q0, A0 = sj.setup(mesh, nsm)
    tj, Qj, _ = sj.run(Q0, A0)
    out = {}
    for n in nprocs:
        sf = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=cfl, imex=Imex.lie, nprocs=n)
        out[n] = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/imex_{tag}_np{n}")
    return float(tj), np.asarray(Qj), out


def compare(label, tj, Qj, tf, Qf):
    d = np.abs(Qf - Qj)
    mx, l1 = d.max() / np.abs(Qj).max(), d.sum() / np.abs(Qj).sum()
    print(f"IMEX {label}: t foam {tf:.6g} / jax {tj:.6g}, max rel {mx:.1e}, L1 rel {l1:.1e}, "
          f"mass foam-jax {abs(Qf[1].sum() - Qj[1].sum()):.1e}", flush=True)
    return mx, l1


def case(w):
    if w == "1d":
        name, make, t_end, cfl = ONE_D
        tj, Qj, out = run(make, t_end, cfl, "1d")
        mx, _ = compare(name, tj, Qj, *out[1])
        check("IMEX 1-D max rel", mx, BOUND["1d"])
    elif w == "2d1":                           # the 2-D case at first order
        tj, Qj, out = run(stiff(lambda: C.bump_2d(1, nx=20, flow=(0.1, 0.05))), 0.3, 0.9, "2d1")
        mx, _ = compare("SWE 2-D bump O1, flow", tj, Qj, *out[1])
        check("IMEX 2-D O1 max rel", mx, BOUND["2d1"])
        check("IMEX 2-D O1 mass", abs(out[1][1][1].sum() - Qj[1].sum()), 1e-12)
    elif w == "2d":
        l1s = []
        for nx in (20, 40):
            tj, Qj, out = run(stiff(lambda: C.bump_2d(2, nx=nx, flow=(0.1, 0.05))), 0.3, 0.9,
                              f"2d_{nx}")
            _, l1 = compare(f"SWE 2-D bump O2, flow, nx {nx}", tj, Qj, *out[1])
            l1s.append(l1)
            check(f"IMEX 2-D O2 nx {nx} mass", abs(out[1][1][1].sum() - Qj[1].sum()), 1e-12)
        check("IMEX 2-D O2 nx 20 L1 rel", l1s[0], BOUND["2d"])
        check("IMEX 2-D O2 L1 nx 20 / nx 40", l1s[0] / l1s[1], 1.5, at_least=True)
    elif w == "par":
        for label, make, t_end, cfl in CASES:
            _, _, out = run(make, t_end, cfl, "par_" + ("1d" if "1-D" in label else "2d"), (1, 2, 4))
            Qs = out[1][1]
            for n in (2, 4):
                Qp = out[n][1]
                r = np.abs(Qp - Qs).max() / np.abs(Qs).max()
                print(f"IMEX {label}: {n} ranks vs serial foam max rel {r:.1e}, "
                      f"mass {abs(Qp[1].sum() - Qs[1].sum()):.1e}, t {out[n][0]:.6g}", flush=True)
                check(f"IMEX {label} {n} ranks max rel", r, 1e-11)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1d", "2d1", "2d", "par"]:
        case(w)
    print("G11_DONE" if not FAILED else f"G11_FAILED {FAILED}")
