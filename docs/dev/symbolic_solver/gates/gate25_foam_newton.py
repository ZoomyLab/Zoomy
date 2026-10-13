"""Gate 25: foam Chorin pressure by Newton vs the linear solve (gate 13 case),
serial and on N ranks; the constraint is affine, so they agree to the
solver tolerance.   python gate25_foam_newton.py [N ...]"""
import sys
import numpy as np
import gate7_chorin as g7
from gate9_foam import FAILED, WORK, check
from zoomy_foam.symsolver import SymbolicChorinSolverFoam
from zoomy_core.symsolver.settings import PressureSolve

s, Q, _ = g7.new_solver(1, pressure_solve=PressureSolve.newton)
out = {}
for solve in PressureSolve:
    for n in [1] + [int(a) for a in sys.argv[1:]]:
        sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                      pressure_solve=solve, nprocs=n)
        out[solve, n] = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q),
                               f"{WORK}/newton_{solve}_np{n}")[1]
ref = out["linear", 1]
for (solve, n), Qf in out.items():
    r = np.abs(Qf - ref).max() / np.abs(ref).max()
    print(f"foam {solve} {n} ranks vs foam linear serial: max rel {r:.1e}")
    check(f"{solve} {n} ranks vs linear serial", r, 6e-12)
print("G25_DONE" if not FAILED else f"G25_FAILED {FAILED}")
