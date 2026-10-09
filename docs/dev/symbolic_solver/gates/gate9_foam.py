"""Gate 9: the hyperbolic template on foam (C++) vs on JAX.

Same template, same NSM, same mesh; only the backend differs.
SME(level) 1-D dam break, extrapolation ends.
"""
import sys
import time as wall

import common  # noqa: F401  (x64)
import numpy as np

from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

WORK = "/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/foam_cases"


def case(level, order, t_end=0.1, nx=200):
    if level == "2d":                       # SWE 2-D bump, walls, background flow
        mesh, nsm = common.bump_2d(order, nx=40, flow=(0.3, 0.1))
        t_end = 0.5
    else:
        mesh, nsm = common.dambreak_1d(level, order, nx=nx)
    sj = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.9, mood=False)
    Q0, A0 = sj.setup(mesh, nsm)
    t0 = wall.perf_counter()
    tj, Qj, _ = sj.run(Q0, A0)
    tj, Qj = float(tj), np.asarray(Qj)
    dt_j = wall.perf_counter() - t0
    sf = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=0.9)
    t0 = wall.perf_counter()
    tf, Qf = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/sme{level}_o{order}")
    dt_f = wall.perf_counter() - t0
    d = np.abs(Qf - Qj).max()
    label = "SWE 2-D bump, walls" if level == "2d" else f"SME({level}) 1-D dam break"
    print(f"{label} O{order}: t foam {tf:.6g} / jax {tj:.6g}, "
          f"max|foam-jax| = {d:.2e} (rel {d / np.abs(Qj).max():.1e}), "
          f"mass foam-jax {abs(Qf[1].sum() - Qj[1].sum()):.1e}, h_min {Qf[1].min():.3f}  "
          f"[foam {dt_f:.0f}s incl. build / jax {dt_j:.0f}s]")
    return sf


if __name__ == "__main__":
    which = sys.argv[1:] or ["0:1"]
    for w in which:
        level, order = w.split(":")
        level, order = (level if level == "2d" else int(level)), int(order)
        sf = case(level, order)
        if w == which[0]:
            open(f"{WORK}/printed_foam.H", "w").write(sf.source)
    print("G9_DONE")
