"""Chorin VAM(1) bump, constant dt: CFL check never / every step / every
10th, wave speeds from cells (eigensolve) or faces (the flux's).  One
process, alternating runs, min of REPS.

    python bench_cfl_every.py [T_END]
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
from zoomy_core.symsolver.settings import CflWaveSpeed
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

T_END = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
REPS = int(os.environ.get("REPS", 3))
n = int(round(T_END / EB.DT))
runs = {}
for ws in (CflWaveSpeed.cells, CflWaveSpeed.faces):
    for every in (0, 1, 10):
        pred, press, corr = EB.split(_DT_SYMBOL)
        s = SymbolicChorinSolverJax(time_end=T_END, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                    pressure_maxit=200, cfl_check_every=every,
                                    cfl_wave_speed=ws)
        Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                       g7.pred_nsm(pred, press, corr), press, corr)
        xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
        Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
        A = s.initial_aux(Q, A)
        f = (lambda s=s, Q=Q, A=A: s.run(Q, A))
        jax.block_until_ready(f())
        runs[(ws, every)] = f
best = {k: 1e9 for k in runs}
for _ in range(REPS):
    for k, f in runs.items():
        t0 = wall.perf_counter()
        jax.block_until_ready(f())
        best[k] = min(best[k], wall.perf_counter() - t0)
print(f"Chorin VAM(1) bump, {n} steps, dt constant, min of {REPS}  (load {os.getloadavg()[0]:.1f})")
for (ws, every), t in best.items():
    print(f"  wave speed {ws:5s}  check {('never', 'every step', '', '', '', '', '', '', '', '', 'every 10th')[every]:10s}"
          f"  {1e3 * t / n:6.3f} ms/step")
print("CFL_EVERY_DONE")
