import sys, numpy as np
sys.argv = ["x", "2"]
import malpasset as MP
import zoomy_core.fvm.timestepping as ts
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
DT = 0.01
mesh, b = MP.build()
nsm = b["nsm"]
print("update_variables:", nsm.call.update_variables is not None, "| source:", nsm.call.source is not None,
      "| source_explicit:", nsm.call.source_explicit is not None)
for n in (3, 4, 5, 8, 12):
    mesh, b = MP.build()
    s = SymbolicHyperbolicSolverJax(time_end=n * DT, dt=DT, cfl=0.4, reconstruction_variables="eta")
    Q, A = s.setup(mesh, b["nsm"]); _, Qn, _ = s.run(Q, A); Qn = np.asarray(Qn)
    mesh2, b2 = MP.build()
    o = HyperbolicSolver(time_end=n * DT, compute_dt=ts.constant(DT), march_log_interval=0, **b2["fs_kwargs"])
    Qo, Ao = o.setup_simulation(mesh2, b2["nsm"]); t = 0.0
    for k in range(n):
        Qs = o.step(DT, t, Qo, Ao); Qo, Ao = o.post_step(t + DT, DT, Qs, Qo, Ao); t += DT
    Qo = np.asarray(Qo)[:, :Qn.shape[1]]
    d = np.abs(Qn - Qo).max(0)
    c = int(np.argmax(d))
    rel = d / (np.abs(Qo).max(0) + 1e-300)
    print(f"{n:2d} steps: worst cell {c}: h new/old {Qn[1,c]:.4g}/{Qo[1,c]:.4g}, q_x_1 {Qn[3,c]:.4g}/{Qo[3,c]:.4g}, "
          f"q_y_1 {Qn[5,c]:.4g}/{Qo[5,c]:.4g}; cells with |diff|>1e-8: {(d>1e-8).sum()}, min h there "
          f"{Qo[1, d>1e-8].min() if (d>1e-8).any() else 0:.3g}; max |q_1|/h new {np.max(np.abs(Qn[3])/np.maximum(Qn[1],1e-3)):.3g} old {np.max(np.abs(Qo[3])/np.maximum(Qo[1],1e-3)):.3g}", flush=True)
