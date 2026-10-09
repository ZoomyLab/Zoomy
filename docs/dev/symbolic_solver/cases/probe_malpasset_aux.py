import sys, numpy as np
sys.argv = ["x", "2"]
import malpasset as MP
import jax.numpy as jnp
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
import zoomy_core.fvm.timestepping as ts
mesh, b = MP.build()
s = SymbolicHyperbolicSolverJax(time_end=2.0, cfl=0.4)
Q, A = s.setup(mesh, b["nsm"])
print("topo monomials", s.topo.lsq_monomial_multi_index, "| mesh monomials", np.asarray(s.mesh_np.lsq_monomial_multi_index).tolist())
print("lsq_gradQ traced vs mesh max|diff|:", float(np.abs(np.asarray(s.mesh.lsq_gradQ)[:Q.shape[1]] - np.asarray(s.mesh_np.lsq_gradQ)[:Q.shape[1]]).max()))
mesh2, b2 = MP.build()
o = HyperbolicSolver(time_end=2.0, compute_dt=ts.constant(0.01), march_log_interval=0, **b2["fs_kwargs"])
Qo, Ao = o.setup_simulation(mesh2, b2["nsm"])
names = [str(v) for v in s.nsm.aux_state]
d = np.abs(np.asarray(A) - np.asarray(Ao)[:, :Q.shape[1]])
print("initial aux max|new-old| per row:", {n: f"{d[k].max():.1e}" for k, n in enumerate(names)})
