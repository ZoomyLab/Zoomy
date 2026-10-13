"""Gate 24: the 2-D Chorin case of gate 21 (VAM(1) channel over a bump, 60 x 6,
t = 0.05) on foam: serial vs JAX, N ranks vs serial: the pressure tolerance
(foam's derivative aux = zoomy_core's LSQ fit on its stencil, as JAX).  The sympy split takes ~2 min, so the JAX
reference is its own run:
    python gate24_foam_chorin2d.py ref               -> WORK/c2d_ref.npy
    python gate24_foam_chorin2d.py foam N [N ...]    (N = 1: serial vs ref)
Rows that are zero up to round-off (q_y: the flow is uniform in y) are
compared in absolute terms."""
import os
import sys
import time as wall
import numpy as np
import jax.numpy as jnp
import common  # noqa: F401  (x64)
from zoomy_core.mesh import LSQMesh
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from zoomy_foam.symsolver import SymbolicChorinSolverFoam
from gate7_chorin import pred_nsm
from gate9_foam import FAILED, WORK, check
from gate21_parallel_chorin import B, NX, T
from zoomy_core.symsolver.settings import LinearSolver

BOUND_JAX = 1e-9     # same LSQ derivative fit as JAX: the pressure tolerance (4.6e-11)


def setup():
    pred, press, corr = B.split()
    s = SymbolicChorinSolverJax(time_end=T, cfl=0.9, pressure_linear_solver=LinearSolver.gmres)
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
    Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    q = np.zeros(Q.shape)
    b = B.BUMP(xc)
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
    return s, jnp.asarray(q), A, names


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    t0 = wall.perf_counter()
    s, Q, A, names = setup()
    print(f"setup {wall.perf_counter() - t0:.0f}s", flush=True)
    if sys.argv[1] == "ref":
        A = s.initial_aux(Q, A)
        t, Qj, _ = s.run(Q, A)
        np.save(f"{WORK}/c2d_ref.npy", np.asarray(Qj))
        print(f"JAX serial t={float(t):.4g}", flush=True)
        sys.exit()
    ref = np.load(f"{WORK}/c2d_ref.npy")
    for n in map(int, sys.argv[2:]):
        sf = SymbolicChorinSolverFoam(time_end=T, cfl=0.9, nprocs=n,
                                      linear_solver=LinearSolver(
                                          os.environ.get("LINSOLVE", "hypre")))
        t1 = wall.perf_counter()
        t, Qf = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q), f"{WORK}/c2d_np{n}")
        w = wall.perf_counter() - t1
        if n == 1:
            np.save(f"{WORK}/c2d_foam1.npy", Qf)
        other = ref if n == 1 else np.load(f"{WORK}/c2d_foam1.npy")
        scale = np.abs(other).max(axis=1)
        live = scale > 1e-8 * scale.max()
        d = np.abs(Qf - other)
        l1 = d[live].sum() / np.abs(other[live]).sum()
        r = (d.max(axis=1) / np.where(live, scale, 1))[live]
        print(f"Chorin 2-D VAM(1) bump: {n} ranks vs {'JAX' if n == 1 else 'serial foam'}: "
              f"L1 rel {l1:.1e}, max rel per row {r.max():.1e} "
              f"({np.array(names)[live][np.argmax(r)]}), zero rows max abs "
              f"{d[~live].max() if (~live).any() else 0:.1e}, finite {bool(np.isfinite(Qf).all())}, "
              f"t {t:.4g} [{w:.0f}s]", flush=True)
        if n == 1:
            check("2-D Chorin serial vs JAX L1 rel", l1, BOUND_JAX)
        else:
            check(f"2-D Chorin {n} ranks vs serial max rel", r.max(), 2e-10)
    print("G24_DONE" if not FAILED else f"G24_FAILED {FAILED}")
