"""Gate 13: Chorin pressure by Newton (pressure_solve=PressureSolve.newton) vs the linear
solve, gate 7 case.  The VAM constraint is affine, so Newton must land on the
linear answer (to the solver tolerance), on JAX and on foam; foam vs JAX
(Newton) agree to the discretization (L1, as gate 12)."""
import numpy as np
import gate7_chorin as g7
from gate9_foam import FAILED, WORK, check
from zoomy_foam.symsolver import SymbolicChorinSolverFoam
from zoomy_core.symsolver.settings import PressureSolve

if __name__ == "__main__":
    s, Q, A = g7.new_solver(1)
    _, Ql, _ = s.run(Q, A)
    sn, Qn0, An0 = g7.new_solver(1, pressure_solve=PressureSolve.newton)
    _, Qn, _ = sn.run(Qn0, An0)
    Ql, Qn = np.asarray(Ql), np.asarray(Qn)
    d = np.abs(Qn - Ql).max()
    print(f"JAX  newton vs linear: max diff {d:.2e}")
    check("JAX newton vs linear max", d, 2e-12)
    out = {}
    for solve in PressureSolve:
        sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                      pressure_solve=solve)
        out[solve] = sf.run(sn.mesh_np, (sn.nsm, sn.press, sn.corr), np.asarray(Qn0),
                            f"{WORK}/chorin_{solve}")[1]
    d = np.abs(out["newton"] - out["linear"]).max()
    print(f"foam newton vs linear: max diff {d:.2e}")
    check("foam newton vs linear max", d, 2e-12)
    l1 = np.abs(out["newton"] - Qn).sum() / np.abs(Qn).sum()
    print(f"foam newton vs JAX newton: max diff {np.abs(out['newton'] - Qn).max():.2e}, L1 rel {l1:.1e}")
    check("foam vs JAX newton L1 rel", l1, 1e-10)   # same derivative fit (1.9e-13)
    print("G13_DONE" if not FAILED else f"G13_FAILED {FAILED}")
