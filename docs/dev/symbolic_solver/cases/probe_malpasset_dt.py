import sys, numpy as np
sys.argv = ["x", "2"]
import malpasset as MP
import jax.numpy as jnp
import zoomy_core.fvm.timestepping as ts
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
mesh, b = MP.build()
s = SymbolicHyperbolicSolverJax(time_end=2.0, cfl=0.4)
Q, A = s.setup(mesh, b["nsm"])
m, g = s.module, s.mesh
lam = m["face_wavespeed"](s.rt.eigenvalues_cfl if s.rt.eigenvalues_cfl is not None else s.rt.eigenvalues, Q, A, s.parameters, g)
r = m["face_inradius"](g)
dt_new = float(jnp.min(0.4 * 2 * r / (2 * lam)))
mesh2, b2 = MP.build()
o = HyperbolicSolver(time_end=2.0, compute_dt=ts.adaptive(CFL=0.4), march_log_interval=0, **b2["fs_kwargs"])
Qo, Ao = o.setup_simulation(mesh2, b2["nsm"])
dt_old = float(o.compute_timestep(Qo, Ao))
print("dt new", dt_new, "dt old", dt_old, "ratio", dt_old / dt_new)
print("max|Q0 new-old|", float(np.abs(np.asarray(Q) - np.asarray(Qo)[:, :Q.shape[1]]).max()))
ci_new = np.asarray(g.cell_inradius)[:Q.shape[1]]; ci_old = np.asarray(s.mesh_np.cell_inradius)[:Q.shape[1]]
print("cell inradius new/old: min", ci_new.min(), ci_old.min(), "max rel diff", float(np.abs(ci_new/ci_old - 1).max()))
print("old face inradius attr:", [a for a in dir(o) if "inradius" in a])
rf_old = np.asarray(getattr(o, "_rt_inradius_face"))
print("face inradius old vs new: shapes", rf_old.shape, np.asarray(r).shape, "max rel", float(np.abs(rf_old[:len(r)]/np.asarray(r) - 1).max()) if rf_old.shape == np.asarray(r).shape else "shape differs")
