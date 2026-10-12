"""Gate AMR-amrex: the hyperbolic template on an AMReX AMR hierarchy (one dt
for every level), scratch.

    cover_o2   SWE bump O2, the fine level covers the domain  == fine single level
    cover_dry  radial dry dam break (eta, MOOD), covering     == fine single level
    wave{n}    Euler smooth wave 2-D periodic, O2, static patch [0.25, 0.75]^2,
               base n x n (n = 16, 32, 64): L1 vs exact (order), vs fine single
    box / boxnr  Euler pulse in a wall box, O2 + MOOD, static patch: mass and
               energy with reflux / without
    lake       SWE lake at rest over a bump, (h, eta) variables, the C/F
               interface across the bump: |u|
    dry        radial dry dam break, regrid every 2 steps on the h difference:
               mass, min h

    python gate37_amrex_amr.py ref   OUT CASE        # single-level fine (AMReX + JAX)
    python gate37_amrex_amr.py run   OUT CASE CONFIG # CONFIG: cpu1, cpu2, gpu1
    python gate37_amrex_amr.py build OUT CASE CONFIG
    python gate37_amrex_amr.py table OUT
"""
import sys
from pathlib import Path

import numpy as np

import common as C
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import TAGS, conserved, nsm_of, wave_exact, wave_system
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.euler import compressible_euler
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_amrex.symsolver import cartesian

CONFIGS = {"cpu1": (1, False), "cpu2": (2, False), "gpu1": (1, True)}
VEL = (1.0, 0.5)


def system(case, n):
    """(mesh at base n, nsm, t_end, cfl, amr settings)."""
    if case == "cover_o2":
        _, nsm = C.bump_2d(2, nx=n, flow=(0.3, 0.1))
        return (LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=n, ny=n), nsm, 0.5, 0.9,
                dict(max_level=1, region=((0, 0), (10, 10))))
    if case in ("cover_dry", "dry"):
        mesh, nsm = C.dry_dambreak_2d(nx=n)
        amr = (dict(max_level=1, region=((0, 0), (1, 1))) if case == "cover_dry" else
               dict(max_level=1, tag=("h", "gradient", 0.05), regrid_int=2))
        return mesh, nsm, 0.02 if case == "cover_dry" else 0.05, 0.45, amr
    if case.startswith("wave"):
        return (LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n),
                nsm_of(wave_system(2, VEL), 2, limiter="none"), 0.2, 0.45,
                dict(max_level=1, region=((0.25, 0.25), (0.75, 0.75)), n_error_buf=0))
    if case.startswith("box"):
        def ic(x):
            bump = np.exp(-sum((x[d] - 0.4) ** 2 for d in range(2)) / 0.02)
            return conserved(1.0 + 0.5 * bump, np.zeros(2), 1.0 + 2.0 * bump)
        sm = compressible_euler(2, 1.4, boundary_conditions=BC.BoundaryConditions(
            [BC.Wall(tag=t) for t in TAGS[2]]), initial_conditions=IC.UserFunction(function=ic))
        return (LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n), nsm_of(sm, 2), 0.3, 0.45,
                dict(max_level=1, region=((0.3, 0.3), (0.7, 0.7)), reflux=case == "box",
                     n_error_buf=0))
    if case == "lake":
        sm = SystemModel.from_model(C.sme(0, 3, ("left", "right", "bottom", "top"), "wall"))
        names = [str(s) for s in sm.state]

        def ic(x):
            q = np.zeros(len(names))
            b = 0.4 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
            q[names.index("b")], q[names.index("h")] = b, 1.0 - b
            return q
        sm.initial_conditions = IC.UserFunction(function=ic)
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod"))
        return (LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n), nsm, 0.1, 0.45,
                dict(max_level=1, region=((0.5, 0.5), (0.9, 0.9)), n_error_buf=0))
    raise KeyError(case)


def base_n(case):
    return {"cover_o2": 16, "cover_dry": 24, "box": 24, "boxnr": 24, "lake": 24,
            "dry": 32}.get(case) or int(case[4:])


def initial(mesh, nsm):
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    s = SymbolicHyperbolicSolverJax(time_end=1.0)
    Q, _ = s.setup(mesh, nsm)
    return s.mesh_np, np.asarray(Q)


def fine_mesh(case, n):
    return system(case, 2 * n)[0]


def ref(out, case):
    """Single level on the fine grid: AMReX and JAX (CFL from the cells)."""
    from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    n = base_n(case)
    mesh, nsm, t_end, cfl, _ = system(case, 2 * n)
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, cfl_wave_speed="cells")
    Q0, A0 = s.setup(mesh, nsm)
    t, Qj, _ = s.run(Q0, A0)
    a = SymbolicHyperbolicSolverAmrex(time_end=t_end, cfl=cfl, cfl_wave_speed="cells")
    ta, Qa = a.run(s.mesh_np, s.nsm, np.asarray(Q0), out / f"work_ref_{case}")
    np.savez(out / f"ref_{case}.npz", t=float(t), Q=np.asarray(Qj), Qa=Qa, ta=ta)
    print(f"ref {case}: JAX t={float(t):.4g} AMReX t={ta:.4g} "
          f"AMReX vs JAX {rel(Qa, np.asarray(Qj)):.1e}", flush=True)


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def run(out, case, config, build_only=False):
    from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
    ranks, gpu = CONFIGS[config]
    n = base_n(case)
    mesh, nsm, t_end, cfl, amr = system(case, n)
    m0, Q0 = initial(mesh, nsm)
    m1, Q1 = initial(fine_mesh(case, n), nsm)
    s = SymbolicHyperbolicSolverAmrex(time_end=t_end, cfl=cfl, cfl_wave_speed="cells",
                                      ranks=ranks, gpu=gpu, amr=amr, build_only=build_only,
                                      max_grid_size=16 if ranks > 1 else 32)
    res = s.run(m0, nsm, Q0, out / f"work_{case}_{config}", fine=[(m1, Q1)])
    if build_only:
        print(f"{case} {config}: built {res}", flush=True)
        return
    t, Q = res
    np.savez(out / f"{case}_{config}.npz", t=t, Q=Q, Q0=Q0, I0=s.initial, L1=s.levels[1],
             wall=s.wall, steps=s.steps)
    print(f"{case} {config}: t={t:.4g} steps {s.steps} march {s.wall:.2f}s "
          f"fine cells {int(np.isfinite(s.levels[1][0]).sum())}", flush=True)


def composite(case, n, Qc, Qf, fn, ref=None):
    """sum over the composite grid of fn(Q, x) * vol (fine where it has cells);
    ``ref`` (fine grid): sum of |Q - ref| instead (ref averaged on coarse cells)."""
    m0, m1 = system(case, n)[0], fine_mesh(case, n)
    g0, g1 = cartesian(m0), cartesian(m1)
    if ref is not None:
        avg = np.zeros((Qc.shape[0],) + tuple(g0["n"][::-1]))
        np.add.at(avg, (slice(None), g1["ij"][1] // 2, g1["ij"][0] // 2), ref / 4)
        rc = avg[:, g0["ij"][1], g0["ij"][0]]
    x0 = np.asarray(m0.cell_centers)[:2, :g0["nc"]]
    x1 = np.asarray(m1.cell_centers)[:2, :g1["nc"]]
    v0, v1 = np.prod(g0["d"]), np.prod(g1["d"])
    has = np.isfinite(Qf[0])
    cov = np.zeros(g0["n"][::-1], bool)
    cov[g1["ij"][1][has] // 2, g1["ij"][0][has] // 2] = True
    unc = ~cov[g0["ij"][1], g0["ij"][0]]
    if ref is not None:
        return (np.abs(Qc[:, unc] - rc[:, unc]).sum() * v0
                + np.abs(Qf[:, has] - ref[:, has]).sum() * v1)
    return (fn(Qc[:, unc], x0[:, unc]).sum() * v0 + fn(Qf[:, has], x1[:, has]).sum() * v1)


def table(out):
    g = lambda f: np.load(f) if f.exists() else None                     # noqa: E731
    print("| case | measure (CPU 1) | MPI 2 vs CPU 1 | GPU 1 vs CPU 1 |")
    print("|---|---|---|---|")
    for case in ("cover_o2", "cover_dry", "wave16", "wave32", "wave64", "box", "boxnr",
                 "lake", "dry"):
        r = {c: g(out / f"{case}_{c}.npz") for c in CONFIGS}
        if r["cpu1"] is None:
            continue
        c1 = r["cpu1"]
        par = " | ".join(f"{rel(r[c]['Q'], c1['Q']):.1e}" if r[c] is not None else "-"
                         for c in ("cpu2", "gpu1"))
        n = base_n(case)
        if case.startswith("cover"):
            f = g(out / f"ref_{case}.npz")
            m = (f"fine level vs fine single AMReX {rel(c1['L1'], f['Qa']):.1e}, "
                 f"vs JAX {rel(c1['L1'], f['Q']):.1e}") if f is not None else "-"
        elif case.startswith("wave"):
            t = float(c1["t"])
            err = composite(case, n, c1["Q"], c1["L1"],
                            lambda Q, x: np.abs(Q[0] - wave_exact(x, t, VEL)))
            m = f"L1(rho - exact) {err:.4e}"
            f = g(out / f"ref_{case}.npz")
            if f is not None:
                m1 = fine_mesh(case, n)
                vf = 1.0 / f["Qa"].shape[1]
                xf = np.asarray(m1.cell_centers)[:2, :f["Qa"].shape[1]]
                ef = np.sum(np.abs(f["Qa"][0] - wave_exact(xf, t, VEL))) * vf
                m += (f", fine single {ef:.4e}; L1(AMR - fine single) "
                      f"{composite(case, n, c1['Q'], c1['L1'], None, ref=f['Qa']):.3e}")
        elif case.startswith("box"):
            m = ", ".join(f"{nm} drift {abs(c1['Q'][i].sum() / c1['I0'][i].sum() - 1):.1e}"
                          for nm, i in (("mass", 0), ("energy", -1)))
        elif case == "lake":
            u = [np.nanmax(np.abs(Q[2:4]) / Q[1]) for Q in (c1["Q"], c1["L1"])]
            m = f"max |u| coarse {u[0]:.1e}, fine {u[1]:.1e}"
        else:
            m = (f"mass drift {abs(c1['Q'][1].sum() / c1['I0'][1].sum() - 1):.1e}, "
                 f"min h {min(np.nanmin(c1['Q'][1]), np.nanmin(c1['L1'][1])):.1e}, "
                 f"fine cells {int(np.isfinite(c1['L1'][0]).sum())}")
        print(f"| {case} | {m} | {par} |")
    errs = {}
    for case in ("wave16", "wave32", "wave64"):
        f = out / f"{case}_cpu1.npz"
        if f.exists():
            c1 = np.load(f)
            t = float(c1["t"])
            errs[int(case[4:])] = composite(case, int(case[4:]), c1["Q"], c1["L1"],
                                            lambda Q, x: np.abs(Q[0] - wave_exact(x, t, VEL)))
    ks = sorted(errs)
    for a, b in zip(ks, ks[1:]):
        print(f"wave EOC {a}->{b}: {np.log(errs[a] / errs[b]) / np.log(b / a):.2f}")


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "ref":
        ref(out, sys.argv[3])
    elif mode in ("run", "build"):
        run(out, sys.argv[3], sys.argv[4], build_only=mode == "build")
    else:
        table(out)
