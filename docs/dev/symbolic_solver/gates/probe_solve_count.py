"""How many host AMG solves per Chorin step?"""
import sys, os
sys.path.insert(0, "../cases")
import common
import jax, jax.numpy as jnp, numpy as np
import escalante_bump as EB, gate7_chorin as g7
import zoomy_jax.symsolver.runtime as RT
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
calls = [0]
import pyamg
orig = pyamg.smoothed_aggregation_solver
def counted(A, **k):
    calls[0] += 1
    return orig(A, **k)
class P: smoothed_aggregation_solver = staticmethod(counted)
RT._pyamg = lambda: P()
for n in (1, 2, 5):
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=n * EB.DT, dt=EB.DT, cfl=0.9, pressure_linear_solver="pyamg")
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS), g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
    calls[0] = 0
    jax.block_until_ready(s.run(Q, A)); jax.effects_barrier()
    print(n, "steps ->", calls[0], "AMG setups")
