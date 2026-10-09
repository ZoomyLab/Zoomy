"""Check every step: wave speed by eigensolve (as is) vs a stand-in wave
speed without eigensolve (max|Q| per face) -- same loop structure."""
import os, sys, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa
import jax, jax.numpy as jnp, numpy as np
import escalante_bump as EB, gate7_chorin as g7
import zoomy_jax.symsolver.runtime as RT
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
n = int(round(EB.T_END / EB.DT))
orig = RT.runtime
def runtime_noeig(*a, **k):
    m = orig(*a, **k)
    fw = m["face_wavespeed"]
    m["face_wavespeed"] = lambda kern, Q, Qaux, p, mesh: 0.0 * fw.__wrapped__ if False else (
        jnp.max(jnp.abs(Q)) + 0.0 * mesh.face_volumes)
    return m
runs = {}
for label, every, rtf in (("never", 0, orig), ("every step, eigensolve", 1, orig),
                          ("every step, no eigensolve", 1, runtime_noeig)):
    RT.runtime = rtf
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=EB.T_END, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, cfl_check_every=every)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
    runs[label] = (lambda s=s, Q=Q, A=A: s.run(Q, A))
RT.runtime = orig
for f in runs.values():
    jax.block_until_ready(f())
best = {k: 1e9 for k in runs}
for _ in range(3):
    for k, f in runs.items():
        t0 = wall.perf_counter(); jax.block_until_ready(f())
        best[k] = min(best[k], wall.perf_counter() - t0)
for k, v in best.items():
    print(f"check {k:28s} {1e3 * v / n:.3f} ms/step")
