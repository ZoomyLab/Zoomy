"""Gate 30: periodic BCs on foam (OpenFOAM cyclic patches) vs JAX.

A smooth SWE wave on the periodic unit interval / square (zoomy Periodic
pairs -> cyclic patches in blockMesh; decomposePar makes processorCyclic
patches where a seam crosses ranks), t = 0.1, no MOOD.
- First order (1-D, 2-D): foam = JAX to round-off.
- Second order: foam's cyclic face takes the partner's reconstructed trace,
  its derivative stencils wrap the seam.  Foam is periodic to round-off:
  shifting the domain by half a period (the seam into the middle of the
  wave) gives the same cells.  JAX (zoomy_jax 89ed42a: the seam takes the
  partner's face trace) is not yet (1-D nx 400: 8.3e-6, 2-D nx 80: 5.9e-5,
  triangles 8.5e-4), and foam - JAX is that difference: it converges (L1
  >= 1.5 / 0.8 on triangles, max >= 0.8 in 1-D, 2.0 in 2-D).
- N ranks vs serial foam: round-off.
- 3-D: the Euler density wave of gate 3d-b (periodic box, blockMesh
  hexahedra, all three pairs cyclic), O1 round-off, O2 converging (no
  limiter), 2/4 ranks = serial.
    python gate35_foam_periodic.py [CASE ...]
    CASE: 1d:1 1d:2 2d:1 2d:2 par 3d:1 3d:2 par3d tri:1 tri:2
"""
import sys
import time as wall

import common  # noqa: F401  (x64)
import numpy as np

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, Periodic
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

from gate9_foam import FAILED, WORK, check

T_END = 0.1


def wave(dim, order, nx, shift=0.0):
    """SWE, h = 1 + 0.1 sin(2 pi x) [sin(2 pi y)], uniform (0.25, 0.1) flow,
    on [shift, 1 + shift]^dim."""
    pairs = [("left", "right")] + ([("bottom", "top")] if dim == 2 else [])
    bcs = [Periodic(tag=a, periodic_to_physical_tag=b) for p in pairs for a, b in (p, p[::-1])]
    sm = SystemModel.from_model(SME(level=0, dimension=dim + 1,
                                    closures=[Newtonian(), ManningFriction(), StressFree()],
                                    boundary_conditions=BoundaryConditions(bcs)))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        h = 1.0 + 0.1 * np.sin(2 * np.pi * x[0]) * (np.sin(2 * np.pi * x[1]) if dim == 2 else 1.0)
        q[names.index("h")] = h
        q[names.index("q_x_0" if dim == 2 else "q_0")] = 0.25 * h
        if dim == 2:
            q[names.index("q_y_0")] = 0.1 * h
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    a, b = shift, 1.0 + shift
    mesh = (LSQMesh.create_1d(domain=(a, b), n_inner_cells=nx) if dim == 1 else
            LSQMesh.create_2d(domain=(a, b, a, b), nx=nx, ny=nx))
    return mesh, nsm


def tri_square(nsm, nx, shift=0.0):
    """[shift, 1 + shift]^2 cut into triangles along alternating diagonals
    (gate 16's tri_bump grid): gmshToFoam, cyclic pairs from sorted seams."""
    from zoomy_core.mesh import ensure_lsq_mesh
    from zoomy_core.mesh.base_mesh import BaseMesh
    xs = np.linspace(shift, 1.0 + shift, nx + 1)
    vid = lambda i, k: i * (nx + 1) + k            # noqa: E731
    X = np.array([[x, y] for x in xs for y in xs]).T
    tris, tags = [], {}
    for i in range(nx):
        for k in range(nx):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            tris += ([(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)])
    for m in range(nx):
        tags[tuple(sorted((vid(0, m), vid(0, m + 1))))] = 0
        tags[tuple(sorted((vid(nx, m), vid(nx, m + 1))))] = 1
        tags[tuple(sorted((vid(m, 0), vid(m + 1, 0))))] = 2
        tags[tuple(sorted((vid(m, nx), vid(m + 1, nx))))] = 3
    base = BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=tags,
                                tag_names={0: "left", 1: "right", 2: "bottom", 3: "top"})
    return ensure_lsq_mesh(base, nsm)


def run(dim, order, nx, nprocs=(1,), shift=0.0, decomposition="scotch", tri=False):
    mesh, nsm = wave(dim, order, nx, shift)
    if tri:
        mesh = tri_square(nsm, nx, shift)
    sj = SymbolicHyperbolicSolverJax(time_end=T_END, cfl=0.9, mood=False)
    Q0, A0 = sj.setup(mesh, nsm)
    _, Qj, _ = sj.run(Q0, A0)
    out = {}
    for n in nprocs:
        sf = SymbolicHyperbolicSolverFoam(time_end=T_END, cfl=0.9, mood=False, nprocs=n,
                                          decomposition=decomposition)
        t0 = wall.perf_counter()
        t, Qf = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), f"{WORK}/per{dim}d{'tri' if tri else ''}_o{order}_{nx}_np{n}_s{shift}_{decomposition}")
        out[n] = (t, Qf, wall.perf_counter() - t0)
    xc = np.round(np.asarray(sj.mesh_np.cell_centers)[:dim, :Qj.shape[1]], 9) % 1.0   # position in the period
    return np.asarray(Qj), out, np.lexsort(xc[::-1])


def stats(Qa, Qb, mass=1):
    d = np.abs(Qa - Qb)
    return (d.max() / np.abs(Qb).max(), d.sum() / np.abs(Qb).sum(),
            abs(Qa[mass].sum() - Qb[mass].sum()))


def euler3d(order, n, nprocs=(1,), decomposition="scotch"):
    """Gate 3d-b's wave: JAX and foam runs (Q[0] = rho)."""
    from euler_common import nsm_of, wave_system
    vel, t_end = (1.0, 0.5, 0.25), 0.1
    mesh = LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n)
    sj = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.45)
    Q0, A0 = sj.setup(mesh, nsm_of(wave_system(3, vel), order, limiter="none"))
    _, Qj, _ = sj.run(Q0, A0)
    out = {}
    for k in nprocs:
        sf = SymbolicHyperbolicSolverFoam(time_end=t_end, cfl=0.45, nprocs=k,
                                          decomposition=decomposition)
        t0 = wall.perf_counter()
        t, Qf = sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0),
                       f"{WORK}/per3d_o{order}_{n}_np{k}_{decomposition}")
        out[k] = (t, Qf, wall.perf_counter() - t0)
    return np.asarray(Qj), out


def case3d(w):
    if w == "par3d":
        for method in ("scotch", "simple"):
            _, out = euler3d(2, 12, (1, 2, 4), method)
            for n in (2, 4):
                mx, _, dm = stats(out[n][1], out[1][1], mass=0)
                print(f"periodic 3-D Euler O2 n 12: {n} ranks ({method}) vs serial foam: "
                      f"max rel {mx:.1e}, mass {dm:.1e} [{out[n][2]:.0f}s]", flush=True)
                check(f"3-D {n} ranks {method} max rel", mx, 1e-14)
                check(f"3-D {n} ranks {method} mass", dm, 1e-11)
        return
    order = int(w[-1])
    ns = (12,) if order == 1 else (8, 16)
    errs = []
    for n in ns:
        Qj, out = euler3d(order, n)
        t, Qf, wt = out[1]
        mx, l1, dm = stats(Qf, Qj, mass=0)
        errs.append(l1)
        print(f"periodic 3-D Euler O{order} n {n}: foam vs JAX max rel {mx:.1e}, L1 rel "
              f"{l1:.1e}, mass {dm:.1e}, t {t:.6g} [{wt:.0f}s]", flush=True)
        check(f"3-D O{order} n {n} mass", dm, 1e-11)
        if order == 1:
            check("3-D O1 max rel", mx, 1e-14)
    if order == 2:
        check("3-D O2 L1 order n 8->16", np.log2(errs[0] / errs[1]), 1.5, at_least=True)


def case_tri(w):
    """Triangles (gmshToFoam): O1 round-off, O2 converging, 2/4 ranks."""
    if w == "tri:1":
        Qj, out, _ = run(2, 1, 24, (1, 2, 4), tri=True)
        mx, l1, dm = stats(out[1][1], Qj)
        print(f"periodic triangles O1 nx 24: foam vs JAX max rel {mx:.1e}, mass {dm:.1e}", flush=True)
        check("tri O1 max rel", mx, 1e-14)
        for n in (2, 4):
            r = stats(out[n][1], out[1][1])[0]
            print(f"periodic triangles O1: {n} ranks vs serial foam max rel {r:.1e}", flush=True)
            check(f"tri O1 {n} ranks max rel", r, 1e-14)
        return
    errs = []
    for nx in (16, 32, 64):
        Qj, out, xo = run(2, 2, nx, (1, 4) if nx == 32 else (1,), tri=True)
        mx, l1, dm = stats(out[1][1], Qj)
        errs.append(l1)
        print(f"periodic triangles O2 nx {nx}: foam vs JAX max rel {mx:.1e}, L1 rel {l1:.1e}, "
              f"mass {dm:.1e}", flush=True)
        check(f"tri O2 nx {nx} mass", dm, 1e-11)
        if nx == 32:
            r = stats(out[4][1], out[1][1])[0]
            print(f"periodic triangles O2 nx 32: 4 ranks vs serial foam max rel {r:.1e}", flush=True)
            check("tri O2 4 ranks max rel", r, 1e-14)
    Qj2, out2, xo2 = run(2, 2, nx, shift=0.5, tri=True)
    r = np.abs(out[1][1][:, xo] - out2[1][1][:, xo2]).max() / np.abs(out[1][1]).max()
    rj = np.abs(Qj[:, xo] - Qj2[:, xo2]).max() / np.abs(Qj).max()
    print(f"periodic triangles O2 nx {nx}: shifted by half a period vs unshifted: "
          f"foam max rel {r:.1e}, JAX {rj:.1e}", flush=True)
    check("tri O2 foam shift invariance", r, 1e-13)
    # triangles: both gradients first order (ANALYSIS 28), JAX's seam too
    for a, b in zip(errs, errs[1:]):
        check("tri O2 L1 order", np.log2(a / b), 0.8, at_least=True)


def case(w):
    if w.startswith("tri"):
        return case_tri(w)
    if w.startswith("3d") or w == "par3d":
        return case3d(w)
    if w == "par":    # scotch keeps the seams on a rank; simple (x slabs) cuts
        for dim, order, nx in ((1, 2, 200), (2, 2, 40)):   # x: processorCyclic
            for method in ("scotch", "simple"):
                _, out, _ = run(dim, order, nx, (1, 2, 4), decomposition=method)
                Qs = out[1][1]
                for n in (2, 4):
                    case = f"{WORK}/per{dim}d_o{order}_{nx}_np{n}_s0.0_{method}"
                    pc = sum(open(f"{case}/processor{k}/constant/polyMesh/boundary").read()
                             .count("processorCyclic;") for k in range(n))
                    mx, _, dm = stats(out[n][1], Qs)
                    print(f"periodic {dim}-D O{order} nx {nx}: {n} ranks ({method}, "
                          f"{pc} processorCyclic patches) vs serial foam: max rel "
                          f"{mx:.1e}, mass {dm:.1e} [{out[n][2]:.0f}s]", flush=True)
                    check(f"{dim}-D O{order} {n} ranks {method} max rel", mx, 1e-14)
                    check(f"{dim}-D O{order} {n} ranks {method} mass", dm, 1e-11)
                    if method == "simple":
                        check(f"{dim}-D {n} ranks simple: seam across ranks", pc, 2, at_least=True)
        return
    dim, order = int(w[0]), int(w[-1])
    if order == 1:
        nx = 200 if dim == 1 else 40
        Qj, out, _ = run(dim, 1, nx)
        t, Qf, wt = out[1]
        mx, l1, dm = stats(Qf, Qj)
        print(f"periodic {dim}-D O1 nx {nx}: foam vs JAX max rel {mx:.1e}, L1 rel {l1:.1e}, "
              f"mass {dm:.1e}, t {t:.6g} [{wt:.0f}s]", flush=True)
        check(f"{dim}-D O1 max rel", mx, 1e-14)
        check(f"{dim}-D O1 mass", dm, 1e-11)
        return
    nxs = (100, 200, 400) if dim == 1 else (20, 40, 80)
    errs = []
    for nx in nxs:
        Qj, out, xo = run(dim, 2, nx)
        t, Qf, wt = out[1]
        mx, l1, dm = stats(Qf, Qj)
        errs.append((mx, l1))
        print(f"periodic {dim}-D O2 nx {nx}: foam vs JAX max rel {mx:.1e}, L1 rel {l1:.1e}, "
              f"mass {dm:.1e}, t {t:.6g} [{wt:.0f}s]", flush=True)
        check(f"{dim}-D O2 nx {nx} mass", dm, 1e-11)
    # the same wave on the domain shifted by half a period: the seam moves to
    # the middle of the wave.  A periodic scheme gives the same cells.
    Qj2, out2, xo2 = run(dim, 2, nx, shift=0.5)
    for lbl, A, B in (("foam", Qf, out2[1][1]), ("JAX", Qj, Qj2)):
        r = np.abs(A[:, xo] - B[:, xo2]).max() / np.abs(A).max()
        print(f"periodic {dim}-D O2 nx {nx}: {lbl} shifted by half a period vs unshifted: "
              f"max rel {r:.1e}", flush=True)
        if lbl == "foam":
            check(f"{dim}-D O2 foam shift invariance", r, 1e-13)
    for (a, b), na, nb in zip(zip(errs, errs[1:]), nxs, nxs[1:]):
        check(f"{dim}-D O2 L1 order nx {na}->{nb}", np.log2(a[1] / b[1]), 1.5, at_least=True)
        check(f"{dim}-D O2 max order nx {na}->{nb}", np.log2(a[0] / b[0]),
              0.8 if dim == 1 else 2.0, at_least=True)     # 1-D: the seam

if __name__ == "__main__":
    for w in sys.argv[1:] or ["1d:1", "1d:2", "2d:1", "2d:2", "par"]:
        case(w)
    print("G30_DONE" if not FAILED else f"G30_FAILED {FAILED}")
