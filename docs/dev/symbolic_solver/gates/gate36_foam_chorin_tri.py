"""Gate 31: the Chorin template (VAM(1) pressure solve) on unstructured
triangles: gate 24's channel over a bump (60 x 6 quads, each cut into two
triangles along alternating diagonals, t = 0.05) through gmshToFoam.  Serial
foam vs JAX: both take the LSQ fit of zoomy_core's mesh for the derivative
aux (foam: OpenFOAM's cell-point-cell stencil), so they agree to the pressure
tolerance; N ranks vs serial foam likewise.  (Before: OpenFOAM's compact
second derivatives, inconsistent on triangles: L1 0.30 / 0.24 at NX 30 / 60.)  The sympy split takes ~2 min, so the
JAX reference is its own run:
    python gate36_foam_chorin_tri.py ref               -> WORK/c2dtri{NXT}_ref.npy
    python gate36_foam_chorin_tri.py foam N [N ...]    (N = 1: serial vs ref)
Rows zero up to round-off (q_y: uniform in y) are compared in absolute terms."""
import os
import sys
import time as wall

import common  # noqa: F401  (x64)
import jax.numpy as jnp
import numpy as np

from gate7_chorin import pred_nsm
from gate9_foam import FAILED, WORK, check
from gate21_parallel_chorin import B, NX, T
from zoomy_foam.symsolver import SymbolicChorinSolverFoam
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from zoomy_core.symsolver.settings import LinearSolver

# serial vs JAX (L1 rel): same derivative fit (vertex ring + ghosts, REQ-174b
# boundary rows), O1 predictor -> the pressure tolerance (NX 60: 2.8e-6);
# N ranks vs serial (max rel): ELLIPTIC_RTOL 1e-10 per solve, 11 solves
BOUND_JAX, BOUND_PAR = 1e-5, 1e-9
NXT = int(os.environ.get("NXT", NX))     # refinement: NXT x NXT / 10 quads


def channel_triangles(nsm, nx=NX, ny=NX // 10, domain=(-1.5, 1.5, 0.0, 0.3)):
    from zoomy_core.mesh import ensure_lsq_mesh
    from zoomy_core.mesh.base_mesh import BaseMesh
    xs = np.linspace(domain[0], domain[1], nx + 1)
    ys = np.linspace(domain[2], domain[3], ny + 1)
    vid = lambda i, k: i * (ny + 1) + k            # noqa: E731
    X = np.array([[x, y] for x in xs for y in ys]).T
    tris, tags = [], {}
    for i in range(nx):
        for k in range(ny):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            tris += ([(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)])
    for k in range(ny):
        tags[tuple(sorted((vid(0, k), vid(0, k + 1))))] = 0
        tags[tuple(sorted((vid(nx, k), vid(nx, k + 1))))] = 1
    for i in range(nx):
        tags[tuple(sorted((vid(i, 0), vid(i + 1, 0))))] = 2
        tags[tuple(sorted((vid(i, ny), vid(i + 1, ny))))] = 3
    base = BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=tags,
                                tag_names={0: "left", 1: "right", 2: "bottom", 3: "top"})
    return ensure_lsq_mesh(base, nsm)


def setup():
    pred, press, corr = B.split()
    s = SymbolicChorinSolverJax(time_end=T, cfl=0.9, pressure_linear_solver=LinearSolver.gmres)
    nsm = pred_nsm(pred, press, corr)
    Q, A = s.setup(channel_triangles(nsm, NXT, NXT // 10), nsm, press, corr)
    names = [str(v) for v in s.nsm.state]
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    q = np.zeros(Q.shape)
    b = B.BUMP(xc)
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
    return s, jnp.asarray(q), A, names


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    t0 = wall.perf_counter()
    s, Q, A, names = setup()
    print(f"setup {wall.perf_counter() - t0:.0f}s, {Q.shape[1]} triangles", flush=True)
    if sys.argv[1] == "ref":
        A = s.initial_aux(Q, A)
        t1 = wall.perf_counter()
        t, Qj, _ = s.run(Q, A)
        np.save(f"{WORK}/c2dtri{NXT}_ref.npy", np.asarray(Qj))
        print(f"JAX serial t={float(t):.4g} [{wall.perf_counter() - t1:.0f}s], finite "
              f"{bool(np.isfinite(np.asarray(Qj)).all())}", flush=True)
        sys.exit()
    ref = np.load(f"{WORK}/c2dtri{NXT}_ref.npy")
    for n in map(int, sys.argv[2:]):
        sf = SymbolicChorinSolverFoam(time_end=T, cfl=0.9, nprocs=n,
                                      linear_solver=LinearSolver(
                                          os.environ.get("LINSOLVE", "hypre")))
        t1 = wall.perf_counter()
        t, Qf = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q), f"{WORK}/c2dtri{NXT}_np{n}")
        w = wall.perf_counter() - t1
        if n == 1:
            np.save(f"{WORK}/c2dtri{NXT}_foam1.npy", Qf)
        other = ref if n == 1 else np.load(f"{WORK}/c2dtri{NXT}_foam1.npy")
        scale = np.abs(other).max(axis=1)
        live = scale > 1e-8 * scale.max()
        d = np.abs(Qf - other)
        l1 = d[live].sum() / np.abs(other[live]).sum()
        r = (d.max(axis=1) / np.where(live, scale, 1))[live]
        log = (sf_log := f"{WORK}/c2dtri{NXT}_np{n}/log.symFoam") and open(sf_log).read()
        its = [ln for ln in log.splitlines() if "linsolve" in ln]
        print(f"Chorin 2-D VAM(1) bump, triangles: {n} ranks vs "
              f"{'JAX' if n == 1 else 'serial foam'}: L1 rel {l1:.1e}, max rel per row "
              f"{r.max():.1e} ({np.array(names)[live][np.argmax(r)]}), zero rows max abs "
              f"{d[~live].max() if (~live).any() else 0:.1e}, finite "
              f"{bool(np.isfinite(Qf).all())}, t {t:.4g} [{w:.0f}s] {its[-1] if its else ''}",
              flush=True)
        if n == 1:
            check("2-D Chorin triangles serial vs JAX L1 rel", l1, BOUND_JAX)
        else:
            check(f"2-D Chorin triangles {n} ranks vs serial max rel", r.max(), BOUND_PAR)
    print("G31_DONE" if not FAILED else f"G31_FAILED {FAILED}")
