"""Cost of the constant-dt CFL check: Chorin bump (5120 steps), symbolic
solver with cfl_check_every = 0 / 1 / 10, old solver (no check) as reference.
One process, runs alternated, minimum of 3."""
import os, sys, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa
import jax, jax.numpy as jnp, numpy as np, sympy as sp
import escalante_bump as EB, gate7_chorin as g7
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
n = int(round(EB.T_END / EB.DT))
pred, press, corr = EB.split(sp.Symbol("dt", positive=True))
o = ChorinSplitVAMSolverJax(pred, press, corr, pressure_tol=1e-9, pressure_maxit=200)
Q0 = np.array(o.setup_simulation(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS)))
xc = np.asarray(o._rt_mesh.cell_centers)[0, :o.nc]
Q0[:, :o.nc] = EB.initial_state(xc, Q0.shape[0])
runs = {"old (no check)": lambda: o.run_jit_steps(jnp.asarray(EB.DT), n, jnp.asarray(Q0))}
for every in (0, 1, 10):
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=EB.T_END, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, cfl_check_every=every)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
    runs[f"symbolic, check every {every}"] = (lambda s=s, Q=Q, A=A: s.run(Q, A))
for f in runs.values():
    jax.block_until_ready(f())
best = {k: 1e9 for k in runs}
for _ in range(3):
    for k, f in runs.items():
        t0 = wall.perf_counter(); jax.block_until_ready(f())
        best[k] = min(best[k], wall.perf_counter() - t0)
for k, v in best.items():
    print(f"{k:28s} {v:6.2f} s  = {1e3 * v / n:.3f} ms/step  (load {os.getloadavg()[0]:.1f})")
