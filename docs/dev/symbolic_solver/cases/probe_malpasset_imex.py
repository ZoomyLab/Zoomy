import sys, numpy as np
sys.argv = ["x", "2"]
import malpasset as MP
import zoomy_core.fvm.timestepping as ts
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
DT = 0.01
for n in (5, 20, 50):
    mesh, b = MP.build()
    s = SymbolicHyperbolicSolverJax(time_end=n * DT, dt=DT, cfl=0.4, imex=True)
    Q, A = s.setup(mesh, b["nsm"]); _, Qn, _ = s.run(Q, A); Qn = np.asarray(Qn)
    mesh2, b2 = MP.build()
    o = IMEXSourceSolverJax(time_end=n * DT, compute_dt=ts.constant(DT), march_log_interval=0, **b2["fs_kwargs"])
    object.__setattr__(o, "source_mode", "local")
    Qo, Ao = o.setup_simulation(mesh2, b2["nsm"]); t = 0.0
    for k in range(n):
        Qs = o.step(DT, t, Qo, Ao); Qo, Ao = o.post_step(t + DT, DT, Qs, Qo, Ao); t += DT
    Qo = np.asarray(Qo)[:, :Qn.shape[1]]
    d = np.abs(Qn - Qo)
    names = [str(v) for v in s.nsm.state]
    print(f"IMEX {n:3d} steps: max|new-old| per row", {nm: f"{d[k].max():.1e}" for k, nm in enumerate(names)},
          f"| clamp-saturated cells new {int((np.abs(Qn[3]) >= 30*np.maximum(Qn[1]-1e-3,0)-1e-12).sum() - (Qn[1] <= 1e-3).sum())}", flush=True)
