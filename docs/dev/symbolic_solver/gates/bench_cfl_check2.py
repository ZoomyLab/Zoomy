"""Check every step: with the warning callback vs the same wave-speed
computation without any callback."""
import os, sys, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa
import jax, jax.numpy as jnp, numpy as np, sympy as sp
import escalante_bump as EB, gate7_chorin as g7
import zoomy_jax.symsolver.runtime as RT
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
n = int(round(EB.T_END / EB.DT))
orig = RT.runtime
def runtime_nocb(*a, **k):
    m = orig(*a, **k)
    m["check_cfl"] = lambda tok, dt_cfl, dt: (tok + (dt > 2.0 * dt_cfl).astype(tok.dtype),)
    return m
runs = {}
for label in ("callback", "no callback"):
    RT.runtime = orig if label == "callback" else runtime_nocb
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=EB.T_END, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, cfl_check_every=1)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
    runs[label] = (lambda s=s, Q=Q, A=A: s.run(Q, A))
for f in runs.values():
    jax.block_until_ready(f())
best = {k: 1e9 for k in runs}
for _ in range(3):
    for k, f in runs.items():
        t0 = wall.perf_counter(); jax.block_until_ready(f())
        best[k] = min(best[k], wall.perf_counter() - t0)
for k, v in best.items():
    print(f"check every step, {k:12s} {v:6.2f} s = {1e3 * v / n:.3f} ms/step")
