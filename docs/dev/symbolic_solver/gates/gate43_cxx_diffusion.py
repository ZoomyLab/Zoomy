"""Gate 43: the diffusion blocks of the C++ runtimes (foam rt.H, AMReX
rt_amrex.H) against JAX.

Gate 41's heat equation u_t = div(nu grad u) on [0, 1]^d (dir0, neu, dirL =
position-dependent Dirichlet, per = periodic, steady = a linear field kept
exactly), explicit (diffusion_matrix_explicit, Euler) and implicit
(diffusion_matrix, IMEX: Crank-Nicolson by Newton, gate 41's tolerances) on
1-D and 2-D quads (AMReX: Cartesian only; foam also gate 41's triangles),
nx = 16.  The same Q0 into every backend; per run max rel |C++ - JAX|:
  explicit  <= 1e-12   round-off (the same two-point fluxes and ghosts)
  implicit  <= 1e-9    the solver tolerance: Newton ||R||_2 <= 1e-10 (JAX:
                       the exact jvp, C++: a directional difference), GMRES
                       1e-13, 26 steps
  steady    max|u - L| <= 1e-12 on the C++ backend itself
L_inf against the exact solution is printed for information (gate 41 has the
orders).  ``par``: 2 ranks = serial on the same backend, nx = 32 (dir0, dirL,
per; explicit <= 1e-12, implicit <= 1e-9).

    python gate43_cxx_diffusion.py [foam|amrex ...] [serial|par]
GATE43_STORE=<file.json>: also store the numbers (and the 1-D dirL profiles),
merged into the file; gate43_figure.py plots them.
"""
import json
import os
import sys
import time as clock

import numpy as np

import common  # noqa: F401  (x64)
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

from gate41_diffusion_bc import NU, T_END, TIGHT, exact, heat, mesh_of  # noqa: F401
from zoomy_core.symsolver.settings import Imex

WORK = os.environ.get("ZOOMY_GATE_WORK", "/tmp/zoomy_gates")
BOUND = {"expl": 1e-12, "impl": 1e-9}
KINDS = {"foam": ("1d", "quad", "tri"), "amrex": ("1d", "quad")}
FAILED = []
STORE = {}


def setup(kind, case, path, nx):
    """(mesh with its LSQ stencils, nsm, Q0 in mesh order, dt)."""
    dim = 1 if kind == "1d" else 2
    nsm = heat(dim, case, path)
    mesh = ensure_lsq_mesh(mesh_of(kind, nx), nsm)
    xc = np.asarray(mesh.cell_centers)[:, :int(mesh.n_inner_cells)]
    Q0 = nsm.sm.initial_conditions.apply(xc, np.zeros((1, xc.shape[1])))
    h = 1.0 / nx
    rho = 4 * dim / h ** 2 if kind != "tri" else 24 / h ** 2      # gate 41
    dt = 0.4 / (rho * NU) if path == "expl" else h / 32
    return mesh, nsm, np.asarray(Q0, dtype=float), dt


def jax_run(kind, case, path, nx):
    mesh, nsm, Q0, dt = setup(kind, case, path, nx)
    s = SymbolicHyperbolicSolverJax(time_end=T_END, dt=dt, cfl_check_every=0,
                                    imex=Imex.lie if path == "impl" else False, **TIGHT)
    _, A = s.setup(mesh, nsm)
    import jax.numpy as jnp
    t, Q, _ = s.run(jnp.asarray(Q0), A)
    return float(t), np.asarray(Q)[0]


def cxx_run(backend, kind, case, path, nx, ranks=1):
    mesh, nsm, Q0, dt = setup(kind, case, path, nx)
    kw = dict(time_end=T_END, dt=dt, cfl_check_every=0, imex=Imex.lie if path == "impl" else False, **TIGHT)
    if backend == "foam":
        from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
        s = SymbolicHyperbolicSolverFoam(nprocs=ranks, **kw)
    else:
        from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
        s = SymbolicHyperbolicSolverAmrex(ranks=ranks, max_grid_size=8, **kw)
    work = f"{WORK}/gate43_{backend}_{kind}_{case}_{path}_n{nx}_r{ranks}"
    t, Q = s.run(mesh, nsm, Q0, work)
    return float(t), np.asarray(Q)[0]


def check(label, value, bound):
    ok = value <= bound
    if not ok:
        FAILED.append(label)
    print(f"gate43 {label}: {value:.2e} (<= {bound:.0e})  {'PASS' if ok else 'FAIL'}", flush=True)


def serial(backend, nx=16):
    for kind in KINDS[backend]:
        dim = 1 if kind == "1d" else 2
        for path in ("expl", "impl"):
            for case in ("dir0", "neu", "dirL", "per", "steady"):
                w0 = clock.perf_counter()
                tj, uj = jax_run(kind, case, path, nx)
                tc, uc = cxx_run(backend, kind, case, path, nx)
                xc = np.asarray(setup(kind, case, path, nx)[0].cell_centers)[:, :len(uc)]
                err = np.abs(uc - exact(case, dim, xc, tc)).max()
                d = np.abs(uc - uj).max() / np.abs(uj).max()
                lab = f"{backend} {kind:4s} {path} {case:6s} n{nx}"
                print(f"gate43 {lab}: t {tc:.6f} / JAX {tj:.6f}  L_inf vs exact {err:.2e}"
                      f"  [{clock.perf_counter() - w0:.0f}s]", flush=True)
                check(f"{lab} max rel |{backend} - JAX|", d, BOUND[path])
                STORE[f"{backend} {kind} {path} {case}"] = dict(rel=float(d), err=float(err),
                                                                 bound=BOUND[path])
                if kind == "1d" and case == "dirL":
                    STORE[f"profile {backend} {path}"] = dict(
                        x=xc[0].tolist(), cxx=uc.tolist(), jax=uj.tolist(),
                        exact=exact(case, dim, xc, tc).tolist())
                if case == "steady":
                    check(f"{lab} max|u - L|", err, 1e-12)


def parallel(backend, nx=32):
    for kind in KINDS[backend]:
        for path in ("expl", "impl"):
            for case in ("dir0", "dirL", "per"):
                _, us = cxx_run(backend, kind, case, path, nx)
                _, up = cxx_run(backend, kind, case, path, nx, ranks=2)
                d = np.abs(up - us).max() / np.abs(us).max()
                check(f"{backend} {kind:4s} {path} {case:4s} n{nx} 2 ranks vs serial", d,
                      BOUND[path])
                STORE[f"par {backend} {kind} {path} {case}"] = dict(rel=float(d), bound=BOUND[path])


if __name__ == "__main__":
    args = sys.argv[1:]
    backends = [a for a in args if a in KINDS] or list(KINDS)
    mode = "par" if "par" in args else "serial"
    for b in backends:
        (parallel if mode == "par" else serial)(b)
    print(f"gate43 {' '.join(backends)} {mode}: {len(FAILED)} FAIL", flush=True)
    if os.environ.get("GATE43_STORE"):
        f = os.environ["GATE43_STORE"]
        old = json.load(open(f)) if os.path.exists(f) else {}
        json.dump({**old, **STORE}, open(f, "w"), indent=1)
    sys.exit(1 if FAILED else 0)
