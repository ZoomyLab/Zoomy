"""Gate 14: Chorin VAM(1) bump, pressure solve by matrix-free GMRES vs AmgX
(JAX-AMG, optional, GPU).  Same template; only the linear solver differs.

    JAX_PLATFORMS=cuda,cpu python gate14_amgx.py [N ...]
"""
import os
import sys
import time as wall

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa: F401
import escalante_bump as EB
import gate7_chorin as g7
import jax
import jax.numpy as jnp
import numpy as np
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

T = float(os.environ.get("T_END", 0.5))
for n in [int(a) for a in sys.argv[1:]] or [60, 2000]:
    out = {}
    for ls in (LinearSolver.gmres, LinearSolver.amgx):
        pred, press, corr = EB.split(_DT_SYMBOL)
        kw = {}
        if ls == "amgx" and os.environ.get("AMGX_CFG"):
            kw["pressure_amgx"] = eval(os.environ["AMGX_CFG"])
        s = SymbolicChorinSolverJax(time_end=T, cfl=0.9, pressure_linear_solver=ls, **kw)
        mesh = LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=n)
        Q, A = s.setup(mesh, g7.pred_nsm(pred, press, corr), press, corr)
        xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
        Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
        A = s.initial_aux(Q, A)
        t0 = wall.perf_counter()
        jax.block_until_ready(s.run(Q, A))
        first = wall.perf_counter() - t0
        t0 = wall.perf_counter()
        t, Qe, _ = s.run(Q, A)
        jax.block_until_ready(Qe)
        out[ls] = (np.asarray(Qe), wall.perf_counter() - t0, first)
    d = np.abs(out["gmres"][0] - out["amgx"][0]).max()
    print(f"N={n:5d} ({2 * n} pressure unknowns), t={T}: gmres {out['gmres'][1]:.2f}s "
          f"(first {out['gmres'][2]:.1f}s) | amgx {out['amgx'][1]:.2f}s (first {out['amgx'][2]:.1f}s) "
          f"| max|gmres-amgx| {d:.2e} (rel {d / np.abs(out['gmres'][0]).max():.1e}) "
          f"on {jax.devices()[0].platform}", flush=True)
print("G14_DONE")
