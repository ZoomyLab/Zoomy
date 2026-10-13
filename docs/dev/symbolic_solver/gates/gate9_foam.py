"""Gate 9: the hyperbolic template on foam (C++) vs on JAX.

Same template, same NSM, same mesh; only the backend differs.  First order
and 1-D: round-off.  2-D second order: foam takes OpenFOAM's gradients and
limiters (face neighbours, boundary face values), JAX its vertex-ring LSQ, so
they agree to the discretization: the smooth bump converges (max), the bump
with wall shocks and the dry dam break are compared in L1.  Closed domains:
mass to round-off.
    python gate9_foam.py [CASE ...]   CASE: level:order | 2d:o | smooth:2 | dry:2 | eta:2
"""
import os
import sys
import tempfile
import time as wall

import common  # noqa: F401  (x64)
import numpy as np

from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from zoomy_core.numerics import ReconstructedVariables

WORK = os.environ.get("ZOOMY_GATE_WORK", os.path.join(tempfile.gettempdir(), "zoomy_gates"))
FAILED = []


def check(label, value, bound, at_least=False):
    ok = value >= bound if at_least else value <= bound
    print(f"   check {label}: {value:.1e} {'>=' if at_least else '<='} {bound:.2g} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        FAILED.append(label)


def run(mesh, nsm, t_end, cfl=0.9, mood=False, tag="case"):
    sj = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, mood=mood)
    Q0, A0 = sj.setup(mesh, nsm)
    t0 = wall.perf_counter()
    tj, Qj, _ = sj.run(Q0, A0)
    tj, Qj = float(tj), np.asarray(Qj)
    dt_j = wall.perf_counter() - t0
    sf = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=cfl, mood=mood)
    t0 = wall.perf_counter()
    tf, Qf = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/{tag}")
    return sf, tf, tj, Qf, Qj, wall.perf_counter() - t0, dt_j


def report(label, sf, tf, tj, Qf, Qj, dt_f, dt_j):
    d = np.abs(Qf - Qj)
    print(f"{label}: t foam {tf:.6g} / jax {tj:.6g}, max|foam-jax| = {d.max():.2e} "
          f"(rel {d.max() / np.abs(Qj).max():.1e}), L1 rel {d.sum() / np.abs(Qj).sum():.1e}, "
          f"mass foam-jax {abs(Qf[1].sum() - Qj[1].sum()):.1e}, h_min {Qf[1].min():.3f}  "
          f"[foam {dt_f:.0f}s incl. build / jax {dt_j:.0f}s]")
    return d


def case(level, order):
    if level == "smooth":                   # SWE 2-D bump at rest, walls: smooth
        ds = []
        for nx in (40, 80):
            mesh, nsm = common.bump_2d(2, nx=nx)
            r = run(mesh, nsm, 2.5, mood=True, tag=f"smooth_{nx}")
            ds.append(report(f"SWE 2-D bump at rest O2+MOOD, nx {nx}", *r))
            check(f"smooth nx {nx} max", ds[-1].max(), {40: 2e-3, 80: 4e-4}[nx])
            check(f"smooth nx {nx} mass", abs(r[3][1].sum() - r[4][1].sum()), 1e-12)
        check("smooth max nx 40 / nx 80", ds[0].max() / ds[1].max(), 3.0, at_least=True)
        return r[0]
    if level == "2d":                       # background flow into the walls: wall shocks
        mesh, nsm = common.bump_2d(order, nx=40, flow=(0.3, 0.1))
        r = run(mesh, nsm, 0.5, mood=order >= 2, tag=f"bump2d_o{order}")
        d = report(f"SWE 2-D bump, walls O{order}", *r)
        Qj = r[4]
        if order == 1:
            check("bump O1 max rel", d.max() / np.abs(Qj).max(), 1e-14)
        else:
            check("bump O2 L1 rel", d.sum() / np.abs(Qj).sum(), BOUND["2d"])
        check("bump mass", abs(r[3][1].sum() - Qj[1].sum()), 1e-12)
        return r[0]
    if level == "dry":                      # model's W (eta, u) + MOOD
        mesh, nsm = common.dry_dambreak_2d(nx=24)
        r = run(mesh, nsm, 0.02, cfl=0.45, mood=True, tag="dry")
        d = report("radial dry dam break eta+MOOD O2", *r)
        check("dry L1 rel", d.sum() / np.abs(r[4]).sum(), BOUND["dry"])
        check("dry mass", abs(r[3][1].sum() - r[4][1].sum()), 1e-12)
        return r[0]
    if level == "eta":                      # SME(1) 1-D, the model's W
        mesh, nsm = common.dambreak_1d(1, order, nx=200, variables=ReconstructedVariables.model)
        label, level = "SME(1) 1-D dam break, model W", 1
    else:
        mesh, nsm = common.dambreak_1d(level, order, nx=200)
        label = f"SME({level}) 1-D dam break, conservative W"
    r = run(mesh, nsm, 0.1, tag=f"sme{level}_o{order}_{nsm.reconstruction.variables}")
    d = report(f"{label} O{order}", *r)
    # O2: round-off, amplified where the limiter switches at the shock (the
    # derivative aux by the LSQ fit: 2e-13 at t = 0.04, 1e-10 at t = 0.1)
    check(f"{label} O{order} max rel", d.max() / np.abs(r[4]).max(), 1e-14 if order == 1 else 2e-10)
    return r[0]


BOUND = {"2d": 4e-3, "dry": 7e-2}

if __name__ == "__main__":
    which = sys.argv[1:] or ["0:1"]
    for w in which:
        level, order = w.split(":")
        level, order = (level if level in ("2d", "dry", "eta", "smooth") else int(level)), int(order)
        sf = case(level, order)
        if w == which[0]:
            os.makedirs(WORK, exist_ok=True)
            open(f"{WORK}/printed_foam.H", "w").write(sf.source)
    print("G9_DONE" if not FAILED else f"G9_FAILED {FAILED}")
