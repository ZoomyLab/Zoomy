"""Gate 12: the Chorin template on foam vs on JAX (gate 7 case: Escalante
VAM(1) bump, 60 cells, dt = 2^-7 to t = 0.5).  foam takes OpenFOAM's
derivatives (compact second derivatives, boundary face values), JAX its LSQ
fits: they agree to the discretization (L1), and the gap shrinks with the
mesh (time_order 1 at 120 cells, dt / 2)."""
import numpy as np

import gate7_chorin as g7
from gate9_foam import FAILED, WORK, check
from zoomy_core.mesh import LSQMesh
from zoomy_foam.symsolver import SymbolicChorinSolverFoam


def compare(to, k=1):
    g7.mesh = lambda: LSQMesh.create_1d(domain=g7.cases.ESC_DOMAIN,
                                        n_inner_cells=g7.cases.ESC_NCELLS * k)
    g7.DT, g7.N_STEPS = 2.0 ** -7 / k, 64 * k
    s, Q, Qaux = g7.new_solver(to)
    tj, Qj, _ = s.run(Q, Qaux)
    Qj = np.asarray(Qj)
    sf = SymbolicChorinSolverFoam(time_end=g7.DT * g7.N_STEPS, dt=g7.DT, cfl=0.9,
                                  time_order=to)
    tf, Qf = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q),
                    f"{WORK}/chorin{to}_x{k}")
    d = np.abs(Qf - Qj)
    l1 = d.sum() / np.abs(Qj).sum()
    names = [str(v) for v in s.nsm.state]
    per = {n: f"{d[i].sum() / max(np.abs(Qj[i]).sum(), 1e-300):.1e}" for i, n in enumerate(names)}
    print(f"Chorin VAM(1) bump, time_order {to}, {Qj.shape[1]} cells: t foam {tf:.6g} / "
          f"jax {float(tj):.6g}, max|foam-jax| {d.max():.2e}, L1 rel {l1:.1e}")
    print("   L1 rel per row:", per)
    return sf, l1


if __name__ == "__main__":
    l1 = {}
    for to in (1, 2):
        sf, l1[to] = compare(to)
        check(f"time_order {to} L1 rel", l1[to], 1e-2)
        if to == 1:
            open(f"{WORK}/printed_foam_chorin.H", "w").write(sf.source)
    _, l1x2 = compare(1, 2)
    check("time_order 1 L1 rel, 60 / 120 cells", l1[1] / l1x2, 1.3, at_least=True)
    print("G12_DONE" if not FAILED else f"G12_FAILED {FAILED}")
