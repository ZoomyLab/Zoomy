import sys, numpy as np
sys.argv = ["x", "2"]
import malpasset as MP
import zoomy_core.fvm.timestepping as ts
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
DT = 0.01
names = None
for n in (1, 2, 5, 20):
    mesh, b = MP.build()
    s = SymbolicHyperbolicSolverJax(time_end=n * DT, dt=DT, cfl=0.4)
    Q, A = s.setup(mesh, b["nsm"])
    _, Qn, _ = s.run(Q, A); Qn = np.asarray(Qn)
    names = [str(v) for v in s.nsm.state]
    mesh2, b2 = MP.build()
    o = HyperbolicSolver(time_end=n * DT, compute_dt=ts.constant(DT), march_log_interval=0, **b2["fs_kwargs"])
    Qo, Ao = o.setup_simulation(mesh2, b2["nsm"])
    t = 0.0
    for k in range(n):
        Qs = o.step(DT, t, Qo, Ao); Qo, Ao = o.post_step(t + DT, DT, Qs, Qo, Ao); t += DT
    Qo = np.asarray(Qo)[:, :Qn.shape[1]]
    d = np.abs(Qn - Qo)
    i = np.unravel_index(np.argmax(d), d.shape)
    print(f"{n:3d} steps: max|new-old| {d.max():.2e} at row {names[i[0]]} cell {i[1]} (h={Qo[1, i[1]]:.3g}); per row",
          {nm: f"{d[k].max():.1e}" for k, nm in enumerate(names)}, flush=True)
