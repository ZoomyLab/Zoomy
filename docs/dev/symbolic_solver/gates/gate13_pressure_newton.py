"""Gate 13: Chorin pressure by Newton (pressure_solve="newton") vs the linear
solve, gate 7 case.  The VAM constraint is affine, so Newton must land on the
linear answer (to the solver tolerance); JAX and foam."""
import numpy as np
import gate7_chorin as g7
from gate9_foam import WORK
from zoomy_foam.symsolver import SymbolicChorinSolverFoam

if __name__ == "__main__":
    s, Q, A = g7.new_solver(1)
    _, Ql, _ = s.run(Q, A)
    sn, Qn0, An0 = g7.new_solver(1, pressure_solve="newton")
    _, Qn, _ = sn.run(Qn0, An0)
    Ql, Qn = np.asarray(Ql), np.asarray(Qn)
    print(f"JAX  newton vs linear: max diff {np.abs(Qn - Ql).max():.2e}")
    sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                  pressure_solve="newton")
    _, Qf = sf.run(sn.mesh_np, (sn.nsm, sn.press, sn.corr), np.asarray(Qn0),
                   f"{WORK}/chorin_newton")
    print(f"foam newton vs JAX newton: max diff {np.abs(Qf - Qn).max():.2e}")
    print("G13_DONE")
