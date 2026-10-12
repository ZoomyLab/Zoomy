"""Dry dam break, ONE step from the initial state: template advance vs the
old solver's step: dt, candidate, MOOD mask, final."""
import numpy as np
import jax.numpy as jnp
import sympy as sp
import common as C
import gate1_vs_solver_jax as G
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.discretization.residual import explicit_step, mood_step
from zoomy_core.symsolver.discretization.time import cfl_dt
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

make = lambda: C.dry_dambreak_2d(ties="old")     # noqa: E731
mesh, nsm = make()
o = HyperbolicSolver(time_end=0.02, compute_dt=C.adaptive(0.45, mesh), march_log_interval=0,
                     **G.eta_kw(make))
Qo, Ao = o.setup_simulation(mesh, nsm)
dto = float(o.compute_timestep(Qo, Ao))
Qo1 = np.asarray(o.step(dto, 0.0, Qo, Ao))

mesh, nsm = make()
s = SymbolicHyperbolicSolverJax(time_end=0.02, cfl=0.45)
Q, A = s.setup(mesh, nsm)
nc = Q.shape[1]
print("initial Q equal:", np.abs(np.asarray(Q) - np.asarray(Qo)[:, :nc]).max(),
      "aux equal:", np.abs(np.asarray(A) - np.asarray(Ao)[:, :nc]).max())
Qs, As, ps, ts, ms = sp.symbols("Q Qaux p time mesh")
dim = int(s.mesh_np.dimension)
src = PyPrinter().procedure("one", (Qs, As, ps, ts, ms), lambda Q, Qaux, p, time, mesh: (
    cfl_dt(nsm, Q, Qaux, p, mesh, 0.45, dim),
    explicit_step(nsm, Q, Qaux, p, time, sp.Float(dto), mesh, 2),
    mood_step(nsm, Q, Qaux, p, time, sp.Float(dto), mesh, 2)))
one = R.build(src, "one", s.module)
dtn, cand, step, _ = one(Q, A, s.parameters, 0.0, s.mesh, jnp.int64(0))
print(f"dt old {dto:.15e} new {float(dtn):.15e}")
cand, step = np.asarray(cand), np.asarray(step)
names = [str(v) for v in nsm.state]
h = names.index("h")
print("candidate h_min", cand[h].min(), "bad cells", int((cand[h] < 0).sum()))
d = np.abs(step - Qo1[:, :nc])
print("mood step max|new-old| per row:", " ".join(f"{n}={v:.2e}" for n, v in zip(names, d.max(axis=1))),
      "cells", int((d.max(axis=0) > 1e-10).sum()))
d2 = np.abs(cand - Qo1[:, :nc])
print("candidate vs old step:", d2.max(), "cells", int((d2.max(axis=0) > 1e-10).sum()))
print("old step h_min", Qo1[h, :nc].min(), "neg cells", int((Qo1[h, :nc] < 0).sum()),
      "old _mood_active", getattr(o, "_mood_active", None))
bad = np.flatnonzero(cand[h] < 0)
print("cand at bad cells", cand[h, bad][:5], "old at bad cells", Qo1[h, bad][:5])
