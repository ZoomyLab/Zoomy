"""Gate amrex: the solver templates on AMReX (rt_amrex.H) vs on JAX.

Same template, same NSM, same mesh; only the backend differs.  Cases (2-D,
Cartesian): SWE bump O1, SWE bump O2 + MOOD, radial dry dam break (eta +
MOOD), VAM(1) Chorin bump channel (gate 21's case).  Each on CPU 1 rank, CPU
MPI 2 and 4 ranks, GPU 1 and 4 ranks.

    python gate27_amrex.py ref  OUT              # JAX references -> OUT/ref_*.npz
    python gate27_amrex.py build OUT CASE CONFIG # compile only (long: nohup)
    python gate27_amrex.py run  OUT CASE CONFIG  # one AMReX run -> OUT/<case>_<config>.npz
    python gate27_amrex.py table OUT             # the table

CASE: o1, o2, dry, vam.  CONFIG: cpu1, cpu2, cpu4, gpu1, gpu4.
"""
import sys
import time as wall
from pathlib import Path

import numpy as np

import common as C
from zoomy_core.symsolver.settings import LinearSolver

CASES = {   # name: (label, t_end, cfl)
    "o1": ("SWE bump O1", 0.5, 0.9),
    "o2": ("SWE bump O2 + MOOD", 0.5, 0.9),
    "dry": ("radial dry dam break eta + MOOD", 0.02, 0.9),
    "vam": ("VAM(1) Chorin bump channel", 0.05, 0.9),
}
CONFIGS = {"cpu1": (1, False), "cpu2": (2, False), "cpu4": (4, False),
           "gpu1": (1, True), "gpu4": (4, True)}


def setup(case):
    """(mesh, nsm or (pred, press, corr), Q0, Qaux0, jax solver)."""
    if case == "vam":
        import jax.numpy as jnp
        from zoomy_core.mesh import LSQMesh
        from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
        from gate21_parallel_chorin import B, NX
        from gate7_chorin import pred_nsm
        pred, press, corr = B.split()
        s = SymbolicChorinSolverJax(time_end=CASES[case][1], cfl=0.9,
                                    pressure_linear_solver=LinearSolver.gmres)
        mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
        Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
        names = [str(v) for v in s.nsm.state]
        xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
        q = np.zeros(Q.shape)
        b = B.BUMP(xc)
        q[names.index("b")] = b
        q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
        Q = jnp.asarray(q)
        return s.mesh_np, (s.nsm, s.press, s.corr), Q, A, s
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    mesh, nsm = {"o1": lambda: C.bump_2d(1, nx=40, flow=(0.1, 0.05)),
                 "o2": lambda: C.bump_2d(2, nx=40, flow=(0.1, 0.05)),
                 "dry": lambda: C.dry_dambreak_2d(nx=48)}[case]()
    s = SymbolicHyperbolicSolverJax(time_end=CASES[case][1], cfl=CASES[case][2])
    Q, A = s.setup(mesh, nsm)
    return s.mesh_np, s.nsm, Q, A, s


def ref(out, case):
    mesh, systems, Q0, A0, s = setup(case)
    if case == "vam":
        A0 = s.initial_aux(Q0, A0)
    t0 = wall.perf_counter()
    t, Q, _ = s.run(Q0, A0)
    Q = np.asarray(Q)
    np.savez(out / f"ref_{case}.npz", t=float(t), Q=Q, Q0=np.asarray(Q0),
             wall=wall.perf_counter() - t0)
    print(f"{case}: JAX t={float(t):.4g} [{wall.perf_counter() - t0:.1f}s]", flush=True)


def run(out, case, config, build_only=False):
    from zoomy_amrex.symsolver import (SymbolicChorinSolverAmrex,
                                       SymbolicHyperbolicSolverAmrex)
    ranks, gpu = CONFIGS[config]
    mesh, systems, Q0, A0, _ = setup(case)
    kw = dict(time_end=CASES[case][1], cfl=CASES[case][2], ranks=ranks, gpu=gpu,
              build_only=build_only)
    if case == "vam":
        s = SymbolicChorinSolverAmrex(max_grid_size=8, **kw)
    else:
        s = SymbolicHyperbolicSolverAmrex(**kw)
    t0 = wall.perf_counter()
    res = s.run(mesh, systems, np.asarray(Q0), out / f"work_{case}_{config}")
    if build_only:
        print(f"{case} {config}: built {res}", flush=True)
        return
    t, Q = res
    np.savez(out / f"{case}_{config}.npz", t=t, Q=Q, wall=s.wall,
             total=wall.perf_counter() - t0)
    print(f"{case} {config}: t={t:.4g} march {s.wall:.2f}s", flush=True)


def table(out):
    rel = lambda a, b: np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)  # noqa: E731
    print("| case | t | CPU 1 vs JAX | mass (rel) | MPI 2 / 4 vs CPU 1 | GPU 1 / 4 vs JAX | GPU vs CPU 1 |")
    print("|---|---|---|---|---|---|---|")
    for case, (label, *_r) in CASES.items():
        f = out / f"ref_{case}.npz"
        if not f.exists():
            continue
        r = np.load(f)
        get = {c: np.load(out / f"{case}_{c}.npz") for c in CONFIGS
               if (out / f"{case}_{c}.npz").exists()}
        cell = lambda c, base: f"{rel(get[c]['Q'], base):.1e}" if c in get else "-"  # noqa: E731
        s = get.get("cpu1")
        # closed domains: the change of total depth; VAM (inflow): AMReX - JAX
        m0 = r["Q"][1].sum() if case == "vam" else r["Q0"][1].sum()
        mass = f"{abs(s['Q'][1].sum() - m0) / abs(m0):.1e}" if s is not None else "-"
        tt = f"{r['t']:.4g}" + ("" if s is None or abs(float(s['t']) - float(r['t'])) < 1e-12
                                  else f" (amrex {float(s['t']):.4g})")
        base = s["Q"] if s is not None else None
        print(f"| {label} | {tt} | {cell('cpu1', r['Q'])} | {mass} | "
              f"{cell('cpu2', base) if base is not None else '-'} / "
              f"{cell('cpu4', base) if base is not None else '-'} | "
              f"{cell('gpu1', r['Q'])} / {cell('gpu4', r['Q'])} | "
              f"{cell('gpu1', base) if base is not None else '-'} |")
    for case in CASES:
        for c in CONFIGS:
            f = out / f"{case}_{c}.npz"
            if f.exists():
                print(f"  {case} {c}: march {float(np.load(f)['wall']):.3f}s")


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "ref":
        for case in (sys.argv[3:] or CASES):
            ref(out, case)
    elif mode in ("run", "build"):     # build: in the background (nohup)
        run(out, sys.argv[3], sys.argv[4], build_only=mode == "build")
    else:
        table(out)
