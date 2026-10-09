"""Wall time of one solver.run() vs the number of steps (Chorin bump, constant
dt): the intercept is the per-call overhead."""
import os, sys, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa: F401
import escalante_bump as EB, gate7_chorin as g7
import jax, jax.numpy as jnp, numpy as np
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
for n in (1, 64, 256, 1280):
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=n * EB.DT, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, cfl_check_every=0)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
    jax.block_until_ready(s.run(Q, A))
    t = min((lambda t0: (jax.block_until_ready(s.run(Q, A)), wall.perf_counter() - t0)[1])(wall.perf_counter())
            for _ in range(3))
    print(f"{n:5d} steps: {t:.3f} s  ({1e3 * t / n:.3f} ms/step)", flush=True)
