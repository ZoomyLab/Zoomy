"""Gate 57: free-slip walls not aligned with an axis (ANALYSIS 57).

A `Wall` reflects the normal momentum and keeps the tangential: on an
oblique face its ghost couples q_x and q_y, so neither row is Dirichlet or
Neumann by itself.  In the face's normal frame the normal component is
Dirichlet (n.q = 0) and the tangential components Neumann (d(t.q)/dn = 0);
the boundary closure fits the momentum group jointly there
(`boundary_closure.face_closure`, `geometry_xp._vector_tables`).

Part `orders` (as gate 44, through the JAX runtime's `derivative_aux`): SME(0),
`Wall` on every side, h = 1, a momentum field that satisfies the wall
conditions exactly and is generic at the walls (even parts in q_n, odd
cubic parts in q_t, which a ghost sample cannot reproduce):
  rbox-quad   the box (0, 10) x (0, 4) rotated by 30 deg, quads nx x nx/2
  rbox-tri    the same box, quads cut along alternating diagonals
  annulus     1 <= r <= 2, 0 <= phi <= pi/2 on polar quads (curved walls
              r = 1, 2; straight walls phi = 0, pi/2 meet them at the
              corners); the field vanishes to second order in q_r and its
              q_phi has no first and second r-derivative at r = 1, 2, so
              the polygonal wall (the foot of a polar cell is its face's
              midpoint, normal e_r) carries the BC to O(h^4)
nx = 24, 48, 96 (annulus 48, 96, 192); fit degree p = 2, 3; max error of the
derivatives of order k of q_x, q_y over the boundary and interior cells,
with the closure and with the ghost sample (no class).  PASS iff the
closure's boundary order of every k >= p + 1 - k - 0.2 and its interior
order the same (annulus: p = 2; at p = 3 its interior degree-3 fits do not
converge on polar quads, walls or not: information).

Part `rotation`: the same O2 run (SME(0), no limiter, constant dt, LSQ
degree 2 so the reconstruction gradient takes the closure, numerical
eigenvalues as the wave speed: the default bound on |A_n| depends on the
momentum's basis) of a hump in a walled box on the axis-aligned mesh and on
the mesh rotated by 30 deg: h and the rotated momentum equal to round-off
(max rel <= 1e-11).

Arguments: `orders`, `rotation` (default both); `JAX` / `numpy` backend for
`rotation` via GATE57_BACKEND (default jax).
"""
import functools
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

#: meshes per kind (the annulus needs one level more: its field is steep at the walls, its
#: first refinements pre-asymptotic); GATE57_NX overrides
NX = {"rbox-quad": (24, 48, 96), "rbox-tri": (24, 48, 96), "annulus": (48, 96, 192)}
if os.environ.get("GATE57_NX"):
    NX = {k: tuple(int(v) for v in os.environ["GATE57_NX"].split(",")) for k in NX}
#: fit degrees per kind whose orders are gated; the annulus at p = 3 is information only: its
#: interior degree-3 fits (10 samples, 9 monomials, two cells off the walls) do not converge
#: on polar quads, with or without walls (ANALYSIS 57)
GATED = {"rbox-quad": (2, 3), "rbox-tri": (2, 3), "annulus": (2,)}
TAGS = ("left", "right", "bottom", "top")
SLACK = 0.2
THETA = np.pi / 6
L, W = 10.0, 4.0


# ── meshes ───────────────────────────────────────────────────────────────────
def _grid(nx, ny, place):
    """Vertices `place(i / nx, k / ny)` of an (nx + 1) x (ny + 1) grid, quads and the
    boundary tags (left i = 0, right i = nx, bottom k = 0, top k = ny)."""
    vid = lambda i, k: i * (ny + 1) + k                                  # noqa: E731
    X = np.array([place(i / nx, k / ny) for i in range(nx + 1) for k in range(ny + 1)]).T
    quads = [(vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1))
             for i in range(nx) for k in range(ny)]
    tags = {}
    for k in range(ny):
        tags[tuple(sorted((vid(0, k), vid(0, k + 1))))] = 0
        tags[tuple(sorted((vid(nx, k), vid(nx, k + 1))))] = 1
    for i in range(nx):
        tags[tuple(sorted((vid(i, 0), vid(i + 1, 0))))] = 2
        tags[tuple(sorted((vid(i, ny), vid(i + 1, ny))))] = 3
    return X, quads, tags


def _base(X, quads, tags, tri):
    """BaseMesh of the quads (or of the triangles cut along alternating diagonals)."""
    names = dict(enumerate(TAGS))
    if not tri:
        return BaseMesh.from_arrays(X, np.array(quads).T, "quad", face_tags=tags,
                                    tag_names=names)
    tris = []
    for m, (a, b, c, d) in enumerate(quads):
        tris += [(a, b, c), (a, c, d)] if m % 2 else [(a, b, d), (b, c, d)]
    return BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=tags,
                                tag_names=names)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def box_mesh(nx, theta=THETA, tri=False, origin=(1.0, 2.0)):
    """The box (0, L) x (0, W) rotated by `theta` about the origin, then moved to `origin`."""
    Rm = rotation(theta)
    X, q, t = _grid(nx, max(nx // 2, 1),
                    lambda a, b: Rm @ np.array([a * L, b * W]) + np.asarray(origin))
    return _base(X, q, t, tri)


def annulus_mesh(nx):
    """Polar quads of 1 <= r <= 2, 0 <= phi <= pi/2: nx/2 radial x nx angular cells."""
    X, q, t = _grid(max(nx // 2, 1), nx,
                    lambda a, b: (1 + a) * np.array([np.cos(b * np.pi / 2),
                                                     np.sin(b * np.pi / 2)]))
    return _base(X, q, t, False)


# ── fields (sympy in the domain's own coordinates) ─────────────────────────
def box_field(x, y, m=sp, theta=THETA, origin=(1.0, 2.0)):
    """(q_x, q_y): q_xi = S(xi) A(eta), q_eta = T(eta) B(xi) in the box frame, S(0) = S(L) =
    T(0) = T(W) = 0 (normal momentum), A' = B' = 0 at the walls (tangential)."""
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = x - origin[0], y - origin[1]
    xi, eta = c * xr + s * yr, -s * xr + c * yr
    S = 4 * (xi / L) * (1 - xi / L) * (1 + 2 * xi / L)
    A = m.cos(m.pi * eta / W) + 3 * (eta / W) ** 2 * (1 - eta / W) ** 2
    T = 4 * (eta / W) * (1 - eta / W) * (1 + 2 * eta / W)
    B = 0.8 * m.cos(2 * m.pi * xi / L) + 3 * (xi / L) ** 2 * (1 - xi / L) ** 2 + 0.5
    q_xi, q_eta = S * A, T * B
    return c * q_xi - s * q_eta, s * q_xi + c * q_eta


def annulus_field(x, y, m=sp):
    """(q_x, q_y) of q_r = 4 (r-1)^2 (2-r)^2 cos(2 phi) (1 + r/4), q_phi = sin(2 phi) (1 + 2 (r-1)^3
    (2-r)^3): q_r, d q_r/dr vanish at r = 1, 2, d q_r/d phi at phi = 0, pi/2; q_phi vanishes at
    phi = 0, pi/2, its first and second r-derivatives at r = 1, 2."""
    r = m.sqrt(x ** 2 + y ** 2)
    c2, s2 = (x ** 2 - y ** 2) / r ** 2, 2 * x * y / r ** 2          # cos, sin of 2 phi
    qr = 4 * (r - 1) ** 2 * (2 - r) ** 2 * c2 * (1 + r / 4)
    qp = s2 * (1 + 2 * (r - 1) ** 3 * (2 - r) ** 3)
    return qr * x / r - qp * y / r, qr * y / r + qp * x / r


MESHES_OBLIQUE = ("rbox-quad", "rbox-tri", "annulus")
MESHES = {"rbox-quad": (lambda nx: box_mesh(nx), box_field),
          "rbox-tri": (lambda nx: box_mesh(nx, tri=True), box_field),
          "annulus": (annulus_mesh, annulus_field)}


# ── part orders ──────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def system():
    """SME(0) with a Wall on every side (derived once)."""
    bcs = BC.BoundaryConditions([BC.Wall(tag=t) for t in TAGS])
    return SystemModel.from_model(SME(level=0, dimension=3, closures=[Newtonian(), StressFree()],
                                      boundary_conditions=bcs))


def nsm_for(p, order=1):
    from zoomy_core.numerics import Limiter
    nsm = NumericalSystemModel.from_system_model(
        system(), reconstruction=ReconstructionSpec(order=order, limiter=Limiter.none))
    # the derivative a source of order p would declare -> LSQ degree p
    nsm.source_derivative_specs = [SimpleNamespace(axes=("x",) * p)]
    return nsm


def errors(kind, p, nx):
    make, fld = MESHES[kind]
    nsm = nsm_for(p)
    s = SymbolicHyperbolicSolverJax(time_end=0.1, dt=1e-3, cfl_check_every=0)
    s.setup(ensure_lsq_mesh(make(nx), nsm), nsm)
    m, g, mnp, topo = s.module, s.mesh, s.mesh_np, s.topo
    nc = int(mnp.n_inner_cells)
    cc = np.asarray(g.cell_centers)[:2, :nc]
    names = [str(v) for v in s.nsm.state]
    rows = [names.index("q_x_0"), names.index("q_y_0")]
    x, y = sp.symbols("x y", real=True)
    exact_q = fld(x, y)
    Q = np.zeros((len(names), nc))
    Q[names.index("h")] = 1.0
    for r, e in zip(rows, exact_q):
        Q[r] = sp.lambdify((x, y), e, "numpy")(cc[0], cc[1])
    Qj = jnp.asarray(Q)
    Qaux = jnp.zeros((len(s.nsm.aux_state), nc))
    Qg = m["boundary_ghosts"](s.rt.boundary_ghost_state, Qj, Qaux, s.parameters, 0.0, g)
    mis = [tuple(int(k) for k in mi) for mi in topo.lsq_monomial_multi_index]
    cls = topo.lsq_closure_class[("nsm", "state")]
    if kind in MESHES_OBLIQUE:
        assert all(isinstance(cls[r], tuple) for r in rows), cls  # the momentum: a vector class
    bnd = np.zeros(nc, bool)
    bfc = np.asarray(mnp.boundary_face_cells)
    bnd[bfc[bfc < nc]] = True
    out = {}
    for method in ("closure", "ghost"):
        E = {k: np.zeros(nc) for k in range(1, p + 1)}
        for r, e in zip(rows, exact_q):
            spec = [dict(kind="derivative", target_kind="state", state_index=r, multi_index=mi,
                         row=i, closure=cls[r] if method == "closure" else -1)
                    for i, mi in enumerate(mis)]
            A = np.asarray(m["derivative_aux"](spec, Qj, jnp.zeros((len(mis), nc)), g, Qg,
                                               tuple(rows)))
            for i, mi in enumerate(mis):
                ex = np.broadcast_to(sp.lambdify((x, y), sp.diff(e, x, mi[0], y, mi[1]),
                                                 "numpy")(cc[0], cc[1]), (nc,))
                k = sum(mi)
                E[k] = np.maximum(E[k], np.abs(A[i] - ex))
        for k in range(1, p + 1):
            out[(method, k, "boundary")] = float(E[k][bnd].max())
            out[(method, k, "interior")] = float(E[k][~bnd].max())
    return out


def part_orders(kinds):
    ok_all = True
    for kind in kinds:
        for p in (2, 3):
            E = [errors(kind, p, nx) for nx in NX[kind]]
            for method in ("closure", "ghost"):
                parts, ok = [], True
                for where in ("boundary", "interior"):
                    for k in range(1, p + 1):
                        e = [Ei[(method, k, where)] for Ei in E]
                        o = np.log2(e[-2] / e[-1])
                        good = o >= p + 1 - k - SLACK
                        if method == "closure":
                            ok &= good
                        parts.append(f"{where[0]}d{k} {e[0]:.1e}>{e[-1]:.1e} o"
                                     + "/".join(f"{np.log2(a / b):.1f}"
                                                for a, b in zip(e[:-1], e[1:]))
                                     + ("" if good or method != "closure" else "!"))
                gated = method == "closure" and p in GATED[kind]
                print(f"gate57 {kind:9s} p={p} {method:7s} " + "  ".join(parts)
                      + (f"  {'PASS' if ok else 'FAIL'}" if gated else
                         "  (information)" if method == "closure" else ""), flush=True)
                ok_all &= ok or not gated
    return ok_all


# ── part rotation ────────────────────────────────────────────────────────────
def run_box(theta, backend, nx=16, steps=40, dt=2e-3):
    """O2 (no limiter) SME(0) run of a hump in the walled box rotated by `theta`; returns
    (h, q in the box frame) per cell in the cells' order."""
    from zoomy_core.systemmodel.system_model import SpectralRadius
    nsm = nsm_for(2, order=2)
    # numerical eigenvalues: the default bound (Gelfand + Collatz-Wielandt on |A_n|) depends
    # on the basis of the momentum, so it is not rotation invariant
    nsm.spectral_radius_method = SpectralRadius.eig
    mesh = ensure_lsq_mesh(box_mesh(nx, theta=theta, origin=(0.0, 0.0)), nsm)
    Rm = rotation(theta)
    cc = np.asarray(mesh.cell_centers)[:2, :int(mesh.n_inner_cells)]
    xi = Rm.T @ cc
    h = 1.0 + 0.2 * np.exp(-((xi[0] - 3.0) ** 2 + (xi[1] - 1.5) ** 2) / 0.8)
    u = np.stack([0.15 * np.ones_like(h), 0.05 * np.ones_like(h)])     # box frame
    q = Rm @ (h * u)
    names = [str(v) for v in nsm.state]
    Q0 = np.zeros((len(names), cc.shape[1]))
    Q0[names.index("h")] = h
    Q0[names.index("q_x_0")], Q0[names.index("q_y_0")] = q
    if backend == "jax":
        S, conv = SymbolicHyperbolicSolverJax, jnp.asarray
    else:
        from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy as S
        conv = np.asarray
    s = S(time_end=steps * dt, dt=dt, cfl_check_every=0)
    _, A = s.setup(mesh, nsm)
    Q = conv(Q0)
    _, Q, _ = s.run(Q, s.initial_aux(Q, A))
    Q = np.asarray(Q)
    hq = Q[names.index("h")]
    qb = Rm.T @ np.stack([Q[names.index("q_x_0")], Q[names.index("q_y_0")]])
    return hq, qb


def part_rotation(backend):
    h0, q0 = run_box(0.0, backend)
    h1, q1 = run_box(THETA, backend)
    eh = np.abs(h1 - h0).max() / np.abs(h0).max()
    eq = np.abs(q1 - q0).max() / np.abs(q0).max()
    ok = max(eh, eq) <= 1e-11
    print(f"gate57 rotation {backend}: max rel h {eh:.1e}, q {eq:.1e}  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    t0 = time.time()
    args = sys.argv[1:] or ["orders", "rotation"]
    ok = True
    if "orders" in args:
        kinds = [a for a in args if a in MESHES] or list(MESHES)
        ok &= part_orders(kinds)
    if "rotation" in args:
        ok &= part_rotation(os.environ.get("GATE57_BACKEND", "jax"))
    print(f"SUMMARY gate57: {'OK' if ok else 'FAILED'} runtime {time.time() - t0:.0f}s",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
