"""Gate 11: IMEX template on foam vs on JAX (gate 6 cases: stiff friction
and viscosity, implicit-Euler source by cell-local Newton).  JAX takes the
exact Jacobian (jvp), foam one-sided differences: agreement to the Newton
tolerance is the expectation."""
import numpy as np

from gate6_imex import CASES
from gate9_foam import WORK
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

if __name__ == "__main__":
    for k, (name, make, t_end, cfl) in enumerate(CASES):
        mesh, nsm = make()
        sj = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, imex=True)
        Q0, A0 = sj.setup(mesh, nsm)
        tj, Qj, _ = sj.run(Q0, A0)
        Qj = np.asarray(Qj)
        sf = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=cfl, imex=True)
        tf, Qf = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/imex{k}")
        d = np.abs(Qf - Qj).max()
        print(f"IMEX {name}: t foam {tf:.6g} / jax {float(tj):.6g}, max|foam-jax| "
              f"{d:.2e} (rel {d / np.abs(Qj).max():.1e})")
    print("G11_DONE")
