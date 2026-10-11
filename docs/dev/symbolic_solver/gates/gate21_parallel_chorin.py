"""Gate 21: the distributed Chorin solver vs serial: VAM(1) flow over a bump
in a 2-D channel (bench_vam2d_bump.py), short run.  The pressure solves are
iterative (to the elliptic tolerance), so the runs agree to that tolerance,
not to round-off: distributed = FGMRES with global inner products, block-
Jacobi AMG per rank; serial = GMRES / global AMG.
Setup builds the VAM numerics in sympy (~2 min), so the serial reference
is its own run:
    python gate21_parallel_chorin.py ref SOLVER DIR          # serial -> DIR/ref_SOLVER.npy
    mpiexec -n 3 python gate21_parallel_chorin.py par SOLVER DIR   (gmres, pyamg, amgx)"""
import sys
import time as wall
import numpy as np
import jax.numpy as jnp
from mpi4py import MPI
import common  # noqa: F401  (x64)
from zoomy_core.mesh import LSQMesh
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
from zoomy_core.model.initial_conditions import Constant
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
from gate7_chorin import pred_nsm


class B:
    """The bench_vam2d_bump case (inflow left, P = 0 right, open sides)."""
    H_RES, H_DRY, Q_IN = 0.34, 0.015, 0.11197
    BUMP = staticmethod(lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2)))

    @staticmethod
    def split():
        bcs = ([Dirichlet("left", on=f, value=v) for f, v in
                (("q_x_0", B.Q_IN), ("q_x_1", 0.0), ("r_0", 0.0), ("r_1", 0.0))]
               + [Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
               + [Extrapolation(tag=t) for t in ("bottom", "top")])
        model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                    closures=[Newtonian(), StressFree()])
        sm = SystemModel.from_model(model)
        sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
        sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
        r = model.chorin_split(_DT_SYMBOL, system_model=sm)
        return r.SM_pred, r.SM_press, r.SM_corr

comm = MPI.COMM_WORLD
NX, T = 60, 0.05


def run(comm, solver):
    pred, press, corr = B.split()
    import os
    s = SymbolicChorinSolverJax(time_end=T, cfl=0.9, pressure_linear_solver=solver,
                                comm=comm, pressure_coarse=os.environ.get("COARSE", "1") == "1")
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
    Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    q = np.zeros(Q.shape)
    b = B.BUMP(xc)
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
    Q = jnp.asarray(q)
    A = s.initial_aux(Q, A)
    t0 = wall.perf_counter()
    t, Q, A = s.run(Q, A)
    Q.block_until_ready()
    Qg = s.partition.gather(Q) if s.partition else np.asarray(Q)
    return Qg, names, wall.perf_counter() - t0, float(t)


if __name__ == "__main__":
    mode, solver, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ref = f"{out}/ref_{solver}.npy"
    if mode == "ref":
        Qs, names, wall_s, t = run(None, solver)
        np.save(ref, Qs)
        print(f"Chorin VAM(1) bump, {solver}: serial reference t={t:.3g} [{wall_s:.1f}s]", flush=True)
    else:
        Qp, names, wall_p, t = run(comm, solver)
        if comm.Get_rank() == 0:
            Qs = np.load(ref)
            scale = np.abs(Qs).max(axis=1)
            d = np.abs(Qp - Qs).max(axis=1) / np.where(scale > 0, scale, 1)
            worst = names[int(np.argmax(d))]
            print(f"Chorin VAM(1) bump, {solver}: ranks {comm.Get_size()}  t={t:.3g}  "
                  f"max rel |par-ser| per row {d.max():.1e} ({worst}), P rows "
                  f"{max(d[i] for i, n in enumerate(names) if n.startswith('P')):.1e}  "
                  f"finite {bool(np.isfinite(Qp).all())}  [run {wall_p:.1f}s]", flush=True)
