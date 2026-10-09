"""Gate 12: the Chorin template on foam vs on JAX (gate 7 case: Escalante
VAM(1) bump, 60 cells, dt = 2^-7 to t = 0.5).  The pressure GMRES differs
between backends (jax.scipy vs rt.H), so agreement to the elliptic
tolerance is the expectation."""
import numpy as np

import gate7_chorin as g7
from gate9_foam import WORK
from zoomy_foam.symsolver import SymbolicChorinSolverFoam

if __name__ == "__main__":
    for to in (1, 2):
        s, Q, Qaux = g7.new_solver(to)
        tj, Qj, _ = s.run(Q, Qaux)
        Qj = np.asarray(Qj)
        sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                      time_order=to)
        tf, Qf = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q),
                        f"{WORK}/chorin{to}")
        d = np.abs(Qf - Qj).max()
        names = [str(v) for v in s.nsm.state]
        per = {n: f"{np.abs(Qf[i] - Qj[i]).max():.1e}" for i, n in enumerate(names)}
        print(f"Chorin VAM(1) bump, time_order {to}: t foam {tf:.6g} / jax {float(tj):.6g}, "
              f"max|foam-jax| {d:.2e} (rel {d / np.abs(Qj).max():.1e})")
        print("   per row:", per)
        if to == 1:
            open(f"{WORK}/printed_foam_chorin.H", "w").write(sf.source)
    print("G12_DONE")
