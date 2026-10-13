"""Gate 27b: the template options gate 27 does not exercise, AMReX vs JAX.

    vam2  VAM(1) Chorin bump channel, time_order 2 (outer SSP-RK2)
    vamn  VAM(1) Chorin bump channel, pressure by Newton (pressure_solve=PressureSolve.newton)
    imex  SWE bump O1, Manning n = 0.05, IMEX: implicit source by per-cell Newton

    python gate34_amrex_pressure_imex.py ref  OUT [CASE..]
    python gate34_amrex_pressure_imex.py build OUT CASE CONFIG
    python gate34_amrex_pressure_imex.py run  OUT CASE CONFIG      # CONFIG: cpu1, cpu2, gpu1
    python gate34_amrex_pressure_imex.py table OUT
"""
import sys
import time as wall
from pathlib import Path

import numpy as np

import common as C
from zoomy_core.symsolver.settings import Imex, LinearSolver, PressureSolve

CASES = {"vam2": ("VAM(1) Chorin, time_order 2", 0.05),
         "vamn": ("VAM(1) Chorin, Newton pressure", 0.05),
         "imex": ("SWE bump O1 + Manning, IMEX", 0.5)}
CONFIGS = {"cpu1": (1, False), "cpu2": (2, False), "gpu1": (1, True)}


def options(case):
    return {"vam2": dict(time_order=2), "vamn": dict(pressure_solve=PressureSolve.newton),
            "imex": dict(imex=Imex.lie)}[case]


def setup(case):
    """(mesh, systems, Q0, Qaux0, JAX solver)."""
    if case.startswith("vam"):
        import jax.numpy as jnp
        from zoomy_core.mesh import LSQMesh
        from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
        from gate21_parallel_chorin import B, NX
        from gate7_chorin import pred_nsm
        pred, press, corr = B.split()
        s = SymbolicChorinSolverJax(time_end=CASES[case][1], cfl=0.9,
                                    pressure_linear_solver=LinearSolver.gmres, **options(case))
        mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
        Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
        names = [str(v) for v in s.nsm.state]
        xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
        q = np.zeros(Q.shape)
        b = B.BUMP(xc)
        q[names.index("b")] = b
        q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
        return s.mesh_np, (s.nsm, s.press, s.corr), jnp.asarray(q), A, s
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    mesh, nsm = C.bump_2d(1, nx=40, flow=(0.3, 0.1))
    nsm.parameter_values.n = 0.05                       # Manning: a stiff-ish source
    s = SymbolicHyperbolicSolverJax(time_end=CASES[case][1], cfl=0.9, **options(case))
    Q, A = s.setup(mesh, nsm)
    return s.mesh_np, s.nsm, Q, A, s


def ref(out, case):
    mesh, systems, Q0, A0, s = setup(case)
    if case.startswith("vam"):
        A0 = s.initial_aux(Q0, A0)
    t0 = wall.perf_counter()
    t, Q, _ = s.run(Q0, A0)
    np.savez(out / f"ref_{case}.npz", t=float(t), Q=np.asarray(Q), Q0=np.asarray(Q0))
    print(f"{case}: JAX t={float(t):.4g} [{wall.perf_counter() - t0:.1f}s]", flush=True)


def run(out, case, config, build_only=False):
    from zoomy_amrex.symsolver import (SymbolicChorinSolverAmrex,
                                       SymbolicHyperbolicSolverAmrex)
    ranks, gpu = CONFIGS[config]
    mesh, systems, Q0, _, _ = setup(case)
    kw = dict(time_end=CASES[case][1], cfl=0.9, ranks=ranks, gpu=gpu,
              build_only=build_only, **options(case))
    s = (SymbolicChorinSolverAmrex(max_grid_size=8, **kw) if case.startswith("vam")
         else SymbolicHyperbolicSolverAmrex(**kw))
    res = s.run(mesh, systems, np.asarray(Q0), out / f"work_{case}_{config}")
    if build_only:
        print(f"{case} {config}: built {res}", flush=True)
        return
    t, Q = res
    np.savez(out / f"{case}_{config}.npz", t=t, Q=Q, wall=s.wall)
    print(f"{case} {config}: t={t:.4g} march {s.wall:.2f}s", flush=True)


def table(out):
    rel = lambda a, b: np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)  # noqa: E731
    print("| case | t | CPU 1 vs JAX | MPI 2 vs CPU 1 | GPU 1 vs JAX | GPU vs CPU 1 |")
    print("|---|---|---|---|---|---|")
    for case, (label, _) in CASES.items():
        if not (out / f"ref_{case}.npz").exists():
            continue
        r = np.load(out / f"ref_{case}.npz")
        g = {c: np.load(out / f"{case}_{c}.npz")["Q"] for c in CONFIGS
             if (out / f"{case}_{c}.npz").exists()}
        f = lambda c, b: f"{rel(g[c], b):.1e}" if c in g and b is not None else "-"  # noqa: E731
        base = g.get("cpu1")
        print(f"| {label} | {float(r['t']):.4g} | {f('cpu1', r['Q'])} | {f('cpu2', base)} | "
              f"{f('gpu1', r['Q'])} | {f('gpu1', base)} |")


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "ref":
        for case in (sys.argv[3:] or CASES):
            ref(out, case)
    elif mode in ("run", "build"):
        run(out, sys.argv[3], sys.argv[4], build_only=mode == "build")
    else:
        table(out)
