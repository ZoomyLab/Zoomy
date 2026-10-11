"""Gate 3d-amrex: the Euler SystemModel on AMReX (AMREX_SPACEDIM = 3, and 1)
vs JAX, same template, same NSM, same hexahedral (line) mesh.

    sod{x,y,z}{1,2}  Sod tube n x 2 x 2 along x / y / z, O1 / O2 (HLL, minmod,
                     primitive W, MOOD), t = 0.2, CFL 0.45
    wave2            smooth diagonal wave, periodic 8^3, O2 (no limiter), t = 0.1
    wall2            pulse in a wall box 10^3, O2 + MOOD, t = 0.3: mass, energy
    sod1d2           Sod 1-D (100 cells), O2: the AMREX_SPACEDIM = 1 build
    sme1d2           SME(1) 1-D dam break O2 (gate 16's), the 1-D build

    python gate33_amrex_3d.py ref  OUT [CASE ..]
    python gate33_amrex_3d.py build OUT CASE CONFIG
    python gate33_amrex_3d.py run  OUT CASE CONFIG      # CONFIG: cpu1, cpu2, gpu1
    python gate33_amrex_3d.py table OUT
"""
import sys
import time as wall
from pathlib import Path

import numpy as np

import common as C
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import (TAGS, box, conserved, nsm_of, sod_system, wave_exact,
                          wave_system)
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.euler import compressible_euler

CASES = {f"sod{a}{o}": (f"Sod 3-D along {a}, O{o}", 0.2, 0.45)
         for o in (1, 2) for a in "xyz"}
CASES.update({"wave1": ("wave 3-D periodic 8^3, O1", 0.1, 0.45),
              "wave2": ("wave 3-D periodic 8^3, O2", 0.1, 0.45),
              "wave2n16": ("wave 3-D periodic 16^3, O2", 0.1, 0.45),
              "wall2": ("wall box 10^3 pulse, O2 + MOOD", 0.3, 0.45),
              "sod1d2": ("Sod 1-D 100, O2", 0.2, 0.45),
              "sme1d2": ("SME(1) 1-D dam break O2", 0.1, 0.9)})
CONFIGS = {"cpu1": (1, False), "cpu2": (2, False), "gpu1": (1, True)}


def system(case):
    """(mesh, nsm)."""
    if case.startswith("sod") and not case.startswith("sod1d"):
        axis, order = "xyz".index(case[3]), int(case[4])
        return box(3, 50, axis), nsm_of(sod_system(3, axis), order)
    if case.startswith("wave"):
        n = 16 if case.endswith("n16") else 8
        return (LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n),
                nsm_of(wave_system(3, (1.0, 0.5, 0.25)), int(case[4]), limiter="none"))
    if case == "wall2":
        def ic(x):
            bump = np.exp(-sum((x[d] - 0.4) ** 2 for d in range(3)) / 0.02)
            return conserved(1.0 + 0.5 * bump, np.zeros(3), 1.0 + 2.0 * bump)
        sm = compressible_euler(3, 1.4, boundary_conditions=BC.BoundaryConditions(
            [BC.Wall(tag=t) for t in TAGS[3]]), initial_conditions=IC.UserFunction(function=ic))
        return LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=10, ny=10, nz=10), nsm_of(sm, 2)
    if case == "sod1d2":
        return box(1, 100), nsm_of(sod_system(1), 2)
    return C.dambreak_1d(1, 2, nx=200)


def setup(case):
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    mesh, nsm = system(case)
    s = SymbolicHyperbolicSolverJax(time_end=CASES[case][1], cfl=CASES[case][2])
    Q, A = s.setup(mesh, nsm)
    return s.mesh_np, s.nsm, Q, A, s


def ref(out, case):
    mesh, nsm, Q0, A0, s = setup(case)
    t0 = wall.perf_counter()
    t, Q, _ = s.run(Q0, A0)
    vol = np.asarray(mesh.cell_volumes)[:Q0.shape[1]]
    x = np.asarray(mesh.cell_centers)[:, :Q0.shape[1]]
    np.savez(out / f"ref_{case}.npz", t=float(t), Q=np.asarray(Q), Q0=np.asarray(Q0),
             vol=vol, x=x)
    print(f"{case}: JAX t={float(t):.4g} [{wall.perf_counter() - t0:.1f}s]", flush=True)


def run(out, case, config, build_only=False):
    from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
    ranks, gpu = CONFIGS[config]
    mesh, nsm, Q0, _, _ = setup(case)
    s = SymbolicHyperbolicSolverAmrex(time_end=CASES[case][1], cfl=CASES[case][2],
                                      ranks=ranks, gpu=gpu, build_only=build_only,
                                      max_grid_size=8 if ranks > 1 else None)
    res = s.run(mesh, nsm, np.asarray(Q0), out / f"work_{case}_{config}")
    if build_only:
        print(f"{case} {config}: built {res}", flush=True)
        return
    t, Q = res
    np.savez(out / f"{case}_{config}.npz", t=t, Q=Q, wall=s.wall)
    print(f"{case} {config}: t={t:.4g} march {s.wall:.2f}s", flush=True)


def table(out):
    rel = lambda a, b: np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)  # noqa: E731
    print("| case | t | CPU 1 vs JAX | mass, energy drift (CPU 1) | MPI 2 vs CPU 1 | GPU 1 vs JAX |")
    print("|---|---|---|---|---|---|")
    for case, (label, *_r) in CASES.items():
        if not (out / f"ref_{case}.npz").exists():
            continue
        r = np.load(out / f"ref_{case}.npz")
        g = {c: np.load(out / f"{case}_{c}.npz")["Q"] for c in CONFIGS
             if (out / f"{case}_{c}.npz").exists()}
        f = lambda c, b: f"{rel(g[c], b):.1e}" if c in g and b is not None else "-"  # noqa: E731
        drift = "-"
        if "cpu1" in g and (case == "wall2" or case.startswith("wave")):
            v = r["vol"]
            m = lambda Q, i: (Q[i] * v).sum()                               # noqa: E731
            drift = (f"{abs(m(g['cpu1'], 0) / m(r['Q0'], 0) - 1):.1e}, "
                     f"{abs(m(g['cpu1'], -1) / m(r['Q0'], -1) - 1):.1e}")
        print(f"| {label} | {float(r['t']):.4g} | {f('cpu1', r['Q'])} | {drift} | "
              f"{f('cpu2', g.get('cpu1'))} | {f('gpu1', r['Q'])} |")
    err = {}
    for case in ("wave2", "wave2n16"):
        f = out / f"{case}_cpu1.npz"
        if f.exists() and (out / f"ref_{case}.npz").exists():
            r, Q = np.load(out / f"ref_{case}.npz"), np.load(f)["Q"]
            ex = wave_exact(r["x"], float(r["t"]), (1.0, 0.5, 0.25))
            err[case] = [float(np.sum(np.abs(q[0] - ex) * r["vol"]) / r["vol"].sum())
                         for q in (Q, r["Q"])]
    if len(err) == 2:
        eoc = [np.log(err["wave2"][k] / err["wave2n16"][k]) / np.log(2) for k in (0, 1)]
        print(f"wave O2 L1(rho) 8^3 / 16^3: AMReX {err['wave2'][0]:.4e} / "
              f"{err['wave2n16'][0]:.4e} EOC {eoc[0]:.2f}; JAX EOC {eoc[1]:.2f}")


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
