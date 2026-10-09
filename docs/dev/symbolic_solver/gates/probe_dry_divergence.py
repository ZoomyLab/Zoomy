"""Dry dam break: old solver step by step; the template run to each old
time t_n.  First n where they differ."""
import numpy as np
import common as C
import gate1_vs_solver_jax as G
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

make = lambda: C.dry_dambreak_2d(ties="old")     # noqa: E731
mesh, nsm = make()
o = HyperbolicSolver(time_end=0.02, compute_dt=C.adaptive(0.45, mesh), march_log_interval=0,
                     **G.eta_kw(make))
Q, Qaux = o.setup_simulation(mesh, nsm)
t, states = 0.0, []
while t < 0.02:
    dt = min(float(o.compute_timestep(Q, Qaux)), 0.02 - t)
    Qn = o.step(dt, t, Q, Qaux)
    Q, Qaux = o.post_step(t + dt, dt, Qn, Q, Qaux)
    t += dt
    states.append((t, dt, np.asarray(Q)))
for n in (1, 2, 3, 5, 8, 12, 22):
    if n > len(states):
        break
    tn, dtn, Qo = states[n - 1]
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=tn, cfl=0.45)
    Q0, A0 = s.setup(mesh, nsm)
    tt, Qs, _ = s.run(Q0, A0)
    Qs = np.asarray(Qs)
    d = np.abs(Qs - Qo[:, :Qs.shape[1]])
    print(f"n={n:2d} t={tn:.6f} dt={dtn:.3e}: max|new-old| {d.max():.2e} cells>1e-10: {(d.max(axis=0) > 1e-10).sum()}", flush=True)
