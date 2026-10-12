"""Gate 39: polynomial reproduction of the LSQ derivative fit.

A degree-p LSQ fit must reproduce EVERY derivative (multi-index of order 1..p)
of a polynomial of degree <= p exactly (round-off) in every cell whose fit
sees exact data.  Seeded random polynomials, p = 1, 2, 3, on: 1-D, 2-D quads,
2-D jittered quads, 2-D triangles (gate16 `tri_bump`), 3-D hexahedra.

Cell groups
  (a) interior   cells without a boundary face           -- MUST pass
  (b) boundary   cells, ghost = exact polynomial at the
                 MIRROR point m = c + 2((f-c).n)n          -- MUST pass
  (c) boundary   cells, the BC data as the solvers build it -- the D5 ghost
                 (foot point x_f = c + ((f-c).n)n, d_n = (f-c).n):
        Dirichlet     2 g(x_f) - u_c
        Neumann       u_c + 2 d_n dg/dn(x_f)
        zero-gradient u_c  (walls normal to axis 0, polynomial independent of
                            x_0; the other walls get the exact mirror value)
                 through the BOUNDARY CLOSURE (ANALYSIS 40; fits of degree
                 >= 2 -- degree 1 keeps the ghost sample): the topology's
                 closure classes (Dirichlet / Neumann rows on every face;
                 zero-gradient: Neumann on the axis-0 walls, the exact mirror
                 sample elsewhere) and their constrained fit
                 (geometry_xp.lsq_derivatives with ``classes`` -- the
                 functions the JAX runtime calls) -- MUST pass
The ghost read as a plain SAMPLE (the pre-closure fit) is inexact
(Taylor expansion of p(m) about c, n-direction, d=d_n):
  p(m)       = p + 2d p_n + 2d^2 p_nn + (4/3) d^3 p_nnn
  Dirichlet  = p + 2d p_n +   d^2 p_nn + (1/3) d^3 p_nnn   exact for p <= 1 only
  Neumann    = p + 2d p_n + 2d^2 p_nn +       d^3 p_nnn   exact for p <= 2 only
  zero-grad  exact for every p (polynomial constant along the wall normal)
its error is shown as "sample-fit" (sensitivity: the gate sees the old
defect where the table says so).  Any miss > TOL is FAIL.  The "ghost"
column is max|ghost - p(m)| / scale, the error of the ghost SAMPLE itself.
Group (c) is evaluated on the boundary cells where (b) holds ("excl=m" =
cells dropped because their stencil already fails with exact ghosts, those
are counted in (b)).  "bad=k/N" = cells above TOL in the group.

Run:  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      PYTHONPATH=<gates>:library/zoomy_core:library/zoomy_jax \\
      python gate39_lsq_polynomials.py          (exit 0 iff no unexpected FAIL)
"""
import itertools
import math
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.geometry_xp import build_topology, compute_geometry, lsq_derivatives
from zoomy_core.numerics.boundary_closure import DIRICHLET, GHOST, NEUMANN
from zoomy_jax.mesh.mesh import convert_mesh_to_jax, lsq_gradient_per_field
from gate16_parallel import tri_bump

TOL = 1e-9
SEED = 20260712
DEGREES = (1, 2, 3)
# BC -> smallest degree at which the ghost sample is no longer exact (None: never)
INEXACT_FROM = {"dirichlet": 2, "neumann": 3, "zerograd": None}


class Poly:
    """Random polynomial of total degree <= p in xi = (x - x0)/hw (so that the
    derivatives stay O(1)); `flat0` drops every monomial depending on x_0."""

    def __init__(self, d, p, seed, lo, hi, flat0=False):
        self.d, self.lo, self.hi = d, np.asarray(lo, float), np.asarray(hi, float)
        self.x0, self.hw = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        self.mis = [a for a in itertools.product(range(p + 1), repeat=d)
                    if sum(a) <= p and not (flat0 and a[0] > 0)]
        self.c = np.random.default_rng(seed).uniform(-1.0, 1.0, len(self.mis))

    def __call__(self, X, beta=None):
        """∂^beta P at the points X, shape (d, n)."""
        beta = beta or (0,) * self.d
        xi = (np.asarray(X, float)[:self.d] - self.x0[:, None]) / self.hw[:, None]
        out = np.zeros(xi.shape[1])
        for a, c in zip(self.mis, self.c):
            if any(ai < bi for ai, bi in zip(a, beta)):
                continue
            t = np.full(xi.shape[1], c)
            for i in range(self.d):
                t = t * (math.factorial(a[i]) // math.factorial(a[i] - beta[i])
                         / self.hw[i] ** beta[i]) * xi[i] ** (a[i] - beta[i])
            out += t
        return out

    def normal_derivative(self, X, n):
        return sum(n[i] * self(X, tuple(int(i == j) for j in range(self.d)))
                   for i in range(self.d))


# ── meshes ───────────────────────────────────────────────────────────────────
def jittered_quads(n, seed=7, amp=0.25):
    """n x n quads on the unit square, interior vertices jittered by amp*h
    (the boundary stays straight and axis-aligned)."""
    h = 1.0 / n
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array([[x, y] for x in xs for y in xs]).T                  # (2, nv)
    rng = np.random.default_rng(seed)
    inner = np.array([0 < i < n and 0 < k < n for i in range(n + 1) for k in range(n + 1)])
    X[:, inner] += rng.uniform(-amp * h, amp * h, (2, int(inner.sum())))
    vid = lambda i, k: i * (n + 1) + k
    cells = np.array([[vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)]
                      for i in range(n) for k in range(n)]).T
    return LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_arrays(X, cells, "quad")))


def mesh_cases():
    unit1, unit2, unit3 = (0, 1), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1)
    tri = tri_bump(1, nx=10)[0]                                       # (0,10)^2
    return [
        ("1d-line-16", 1, (0.0,), (1.0,), LSQMesh.create_1d(unit1, 16)),
        ("2d-quad-16x16", 2, (0.0,) * 2, (1.0,) * 2, LSQMesh.create_2d(unit2, 16, 16)),
        ("2d-jitter-12x12", 2, (0.0,) * 2, (1.0,) * 2, jittered_quads(12)),
        ("2d-tri-10x10x2", 2, (0.0,) * 2, (10.0,) * 2, tri),
        ("3d-hex-6^3", 3, (0.0,) * 3, (1.0,) * 3, LSQMesh.create_3d(unit3, 6, 6, 6)),
    ]


# ── one (mesh, p) ────────────────────────────────────────────────────────────
def geometry(mesh, d):
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:d]                            # all slots
    bfc = np.asarray(mesh.boundary_face_cells)
    fid = np.asarray(mesh.boundary_face_face_indices)
    fcen = np.asarray(mesh.face_centers)[fid, :d].T
    n = np.asarray(mesh.face_normals)[:d, fid]
    n = n / np.linalg.norm(n, axis=0)
    c = cc[:, bfc]
    dn = ((fcen - c) * n).sum(0)
    return dict(nc=nc, cc=cc, bfc=bfc, n=n, dn=dn, foot=c + dn * n, mir=c + 2 * dn * n)


def run_derivs(mj, mis, U, UB):
    """lsq_gradient_per_field for every multi-index and every row of (U, UB)."""
    out = {}
    for mi in mis:
        f = jax.jit(jax.vmap(lambda u, ub: lsq_gradient_per_field(mj, u, u_bf=ub, multi_index=mi)))
        out[mi] = np.asarray(f(jnp.asarray(U), jnp.asarray(UB)))      # (B, nc)
    return out


def cell_err(out, row, exact, scale_all):
    """Per inner cell: max over multi-indices of |fit - exact| / max|exact_mi|."""
    worst = 0.0
    for mi, g in out.items():
        e = exact[mi]
        den = np.abs(e).max()
        den = den if den > 1e-6 * scale_all else scale_all             # identically-zero ∂
        worst = np.maximum(worst, np.abs(g[row] - e) / den)
    return worst


def status(err):
    return "PASS" if err < TOL else "FAIL"


def run_closure(mesh, d, p, kinds, U, UG, mis):
    """The derivatives of every row of ``U`` (ghosts ``UG``) through the
    boundary closure with per-row face ``kinds`` ``(B, n_bf)``:
    ``{mi: (B, nc)}``."""
    topo = build_topology(mesh, lsq_degree=p, closure={("gate39", "state"): kinds})
    X = jnp.asarray(np.asarray(mesh.vertex_coordinates)[:d])
    g = compute_geometry(topo, X)
    nc = int(mesh.n_inner_cells)
    # degree 1 has no closure (the ghost sample is exact for linear data)
    classes = topo.lsq_closure_class.get(("gate39", "state"), (-1,) * len(kinds))
    assert (p >= 2) == any(c >= 0 for c in classes), (p, classes)
    D = np.asarray(lsq_derivatives(topo, g, jnp.asarray(U)[:, :nc], jnp.asarray(UG), classes))
    cols = [tuple(int(k) for k in mi) for mi in topo.lsq_monomial_multi_index]
    return {mi: D[:, :, cols.index(mi)] for mi in mis}


def main():
    t0 = time.time()
    fails, n_pass, n_fail, n_xfail, n_xpass = [], 0, 0, 0, 0
    for name, d, lo, hi, mesh in mesh_cases():
        for p in DEGREES:
            mesh._build_lsq_stencil(p)
            mj = convert_mesh_to_jax(mesh)
            mis = [tuple(int(k) for k in mi) for mi in mj.lsq_monomial_multi_index]
            assert max(sum(mi) for mi in mis) == p and len(mis) == math.comb(p + d, d) - 1, (name, p, mis)
            g = geometry(mesh, d)
            nc, cc, bfc, n = g["nc"], g["cc"], g["bfc"], g["n"]
            on_bnd = np.zeros(nc, bool)
            on_bnd[bfc[bfc < nc]] = True
            interior, boundary = np.flatnonzero(~on_bnd), np.flatnonzero(on_bnd)
            assert len(interior) and len(boundary), name

            P = Poly(d, p, SEED + 10 * p + d, lo, hi)
            u = P(cc)                                                 # all cell slots
            uc = u[bfc]
            pm = P(g["mir"])                                          # exact ghost sample
            ghost = {"exact": pm,
                     "dirichlet": 2 * P(g["foot"]) - uc,
                     "neumann": uc + 2 * g["dn"] * P.normal_derivative(g["foot"], n)}
            exact = {mi: P(cc[:, :nc], mi) for mi in mis}
            scale_all = max(np.abs(e).max() for e in exact.values())
            U = [u, u, u]
            UB = [ghost["exact"], ghost["dirichlet"], ghost["neumann"]]

            have_zg = d >= 2
            if have_zg:                                               # x_0-independent poly
                P0 = Poly(d, p, SEED + 10 * p + d + 1, lo, hi, flat0=True)
                wall0 = np.abs(n[0]) > 0.5                            # walls normal to x_0
                zg = np.where(wall0, P0(cc)[bfc], P0(g["mir"]))
                exact0 = {mi: P0(cc[:, :nc], mi) for mi in mis}
                scale0 = max(np.abs(e).max() for e in exact0.values())
                U.append(P0(cc))
                UB.append(zg)
            out = run_derivs(mj, mis, np.array(U), np.array(UB))
            nbf = len(bfc)
            kinds = [np.full(nbf, GHOST), np.full(nbf, DIRICHLET), np.full(nbf, NEUMANN)]
            if have_zg:
                kinds.append(np.where(wall0, NEUMANN, GHOST))
            closed = run_closure(mesh, d, p, np.array(kinds, dtype=np.int8),
                                 np.array(U), np.array(UB), mis)

            def ghost_err(gv, pm_):
                return np.abs(gv - pm_).max() / max(np.abs(pm_).max(), 1e-300)

            ce0 = cell_err(out, 0, exact, scale_all)
            good_b = boundary[ce0[boundary] <= TOL]                   # (b) holds here
            excl = len(boundary) - len(good_b)
            lines = [("interior", "-", interior, ce0, 0.0, None, 0),
                     ("boundary", "exact-mirror", boundary, ce0, 0.0, None, 0)]
            for row, bc in ((1, "dirichlet"), (2, "neumann")):
                lines.append(("boundary", bc, good_b, cell_err(closed, row, exact, scale_all),
                              ghost_err(ghost[bc], pm),
                              cell_err(out, row, exact, scale_all)[good_b].max(), excl))
            if have_zg:
                lines.append(("boundary", "zerograd", good_b, cell_err(closed, 3, exact0, scale0),
                              ghost_err(zg, P0(g["mir"])),
                              cell_err(out, 3, exact0, scale0)[good_b].max(), excl))
            for grp, bc, cells, ce, gerr, sample, ex in lines:
                err = ce[cells].max()
                st = status(err)
                n_pass += st == "PASS"
                n_fail += st == "FAIL"
                if sample is not None:                       # sensitivity
                    inex = INEXACT_FROM[bc] is not None and p >= INEXACT_FROM[bc]
                    n_xfail += bool(inex and sample > TOL)   # the old defect, seen
                    n_xpass += bool(inex and sample <= TOL)  # ... not seen
                if st == "FAIL":
                    fails.append(f"{name} p={p} {grp} {bc}: {err:.2e}")
                bad = int((ce[cells] > TOL).sum())
                tail = f" excl={ex}" if ex else ""
                tail += f"  sample-fit {sample:8.1e}" if sample is not None else ""
                print(f"{name:16s} p={p} {grp:8s} {bc:12s} maxrel={err:9.2e} bad={bad}/{len(cells)} "
                      f"ghost={gerr:8.1e} {st}{tail}", flush=True)
            if not have_zg:
                print(f"{name:16s} p={p} boundary zerograd     SKIP (1-D: only constants are flat along the normal)", flush=True)
    for f in fails:
        print("UNEXPECTED FAIL:", f)
    ok = n_fail == 0
    print(f"SUMMARY gate39: PASS={n_pass} FAIL={n_fail} (tol {TOL:g}); the ghost as a "
          f"plain sample is inexact in {n_xfail} cases (sensitivity; not seen: {n_xpass}) "
          f"runtime {time.time() - t0:.1f}s -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
