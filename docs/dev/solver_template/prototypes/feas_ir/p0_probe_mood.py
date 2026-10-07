"""Probe: does MOOD fire on the 1-D cases; Emitted vs MarchSolver bitwise."""
from common import *  # noqa
import numpy as np
from zoomy_jax.fvm.solver2.march import MarchSolver
from zoomy_jax.fvm.march_emitted import EmittedMarchSolver

for ic, order, recon in [("stoker", 1, None), ("ritter", 2, "eta"),
                         ("ritter", 2, "conservative"), ("stoker", 2, None)]:
    try:
        ref, S, nsm, mesh = make_solver(MarchSolver, ic, order, recon=recon)
        Sr, st = ref.march(S)
        em, S2, _, _ = make_solver(EmittedMarchSolver, ic, order, recon=recon)
        Se, ste = em.march(S2)
        nc = mesh.n_inner_cells
        d = float(np.abs(np.asarray(Sr.Q) - np.asarray(Se.Q)).max())
        print(ic, order, recon, "steps", st["steps"], ste["steps"],
              "troubled_steps", st["troubled_steps"], "maxdiff", d,
              "bitwise", bool((np.asarray(Sr.Q) == np.asarray(Se.Q)).all()),
              "hmin", float(np.asarray(Sr.Q)[1, :nc].min()),
              "mass", float(np.asarray(Sr.Q)[1, :nc].sum() * 0.1))
    except Exception as e:  # noqa
        print(ic, order, recon, "ERR", type(e).__name__, str(e)[:300])
