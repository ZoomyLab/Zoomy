"""Probe per-step MOOD firing / positivity on the reference solver2 path."""
from common import *  # noqa
import numpy as np
import jax.numpy as jnp
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.solver2 import blocks as B

for ic, n, tend in [("ritter", 100, 1.0), ("ritter", 200, 2.0), ("ritter", 400, 2.0)]:
    s, S, nsm, mesh = make_solver(MarchSolver, ic, 2, n_cells=n, t_end=tend,
                                  recon="conservative")
    nc = mesh.n_inner_cells
    p = s._rt_parameters
    dx = 10.0 / n
    m0 = float(np.asarray(S.Q)[1, :nc].sum() * dx)
    fired = 0
    log = []
    while float(S.time) < tend:
        lo, hi, dt = s.step_head(S, p, tend - S.time, None)
        dt = B.assert_dt_admissible(dt, lo, hi, s.MeshRT.inradius_f,
                                    time=S.time, iteration=S.iteration)
        Qc, tr = s.stage_loop(S.Q, S.Qaux, p, S.time, dt, False)
        Q1, tr1 = s.stage_loop(S.Q, S.Qaux, p, S.time, dt, True)
        S2, troubled = s.hyperbolic_step(S, dt, p)
        if bool(np.asarray(troubled).any()):
            fired += 1
            log.append((int(S.iteration), float(np.asarray(Qc)[1, :nc].min()),
                        float(np.asarray(Q1)[1, :nc].min()),
                        int(np.asarray(tr1).sum()),
                        float(np.asarray(S2.Q)[1, :nc].min())))
        S = S2
    h = np.asarray(S.Q)[1, :nc]
    print(ic, n, tend, "steps", int(S.iteration), "mood fired", fired,
          "hmin", h.min(), "mass drift", h.sum() * dx - m0)
    for r in log[:6]:
        print("   it", r[0], "hmin(o2 cand)", r[1], "hmin(o1 redo)", r[2],
              "o1 troubled", r[3], "accepted hmin", r[4])
