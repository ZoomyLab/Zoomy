"""Gate 1a — the boundary closure of the LSQ derivative fits (ANALYSIS 40).

* A row whose ghost is Dirichlet-type (``2 v - Q``) or Neumann-type
  (``Q + 2 d_n g``) closes by a constraint at the foot point (fits of
  degree >= 2; degree 1 keeps the ghost sample): the fit
  reproduces every derivative of a polynomial of the fit's degree in every
  boundary cell -- where the same ghost read as a fit SAMPLE does not
  (Dirichlet from degree 2, Neumann from degree 3).
* The kinds are read off the real D5 ghost kernel: a wall reflects q_n
  (Dirichlet) and keeps q_t, h (Neumann) on an axis-aligned face, couples the
  momentum on an oblique one, where the face's normal frame classifies it
  (normal Dirichlet, tangential Neumann: `face_closure`); reconstruction
  variables (h, eta, u) map through the model's W(Q).
* A vector group closed in the normal frames (one joint fit of its
  components) reproduces polynomial vector fields on oblique walls and at
  their corners (ANALYSIS 57).
* Rotating the mesh rotates every derivative (the decoupled ghost fit in the
  wall frame, the vector closure): the same stencils, the derivative tensors
  rotated to round-off.
* The tables are differentiable in the vertex coordinates (finite).
"""
from __future__ import annotations

import numpy as np
import pytest

from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.geometry_xp import (
    build_topology, compute_geometry, lsq_derivatives)
from zoomy_core.numerics.boundary_closure import (
    DIRICHLET, GHOST, NEUMANN, RowClosure, face_closure, face_kinds)

from test_gate_1a_core_lsq_unisolvent import MESHES, _poly

pytestmark = [pytest.mark.template_gate]


def _bc_data(topo, g, f, dim, nc):
    cc = np.asarray(g["cell_centers"])[:dim, :nc]
    owner = topo.boundary_owner
    nrm = np.asarray(g["face_normals"])[:dim, topo.boundary_faces]
    nrm = nrm / np.linalg.norm(nrm, axis=0)
    dn = np.asarray(g["d_normal"])
    foot = cc[:, owner] + dn * nrm
    u = f(cc)
    grad_n = sum(nrm[a] * f(foot, tuple(int(a == b) for b in range(dim)))
                 for a in range(dim))
    return cc, u, {"dirichlet": 2 * f(foot) - u[owner],
                   "neumann": u[owner] + 2 * dn * grad_n}


@pytest.mark.parametrize("kind,p", [("line", 2), ("line", 3), ("quad", 2),
                                    ("quad", 3), ("jitter", 2), ("jitter", 3),
                                    ("tri", 2), ("tri", 3), ("hex", 2)])
def test_closure_reproduces_polynomials(kind, p):
    m = MESHES[kind]()
    dim, nc, nbf = int(m.dimension), int(m.n_inner_cells), int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf, [NEUMANN] * nbf], dtype=np.int8)
    topo = build_topology(m, lsq_degree=p, closure={("t", "state"): kinds})
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float))
    f = _poly(dim, p, 200 + 10 * p + dim)
    cc, u, ghost = _bc_data(topo, g, f, dim, nc)
    U = np.stack([u, u])
    G = np.stack([ghost["dirichlet"], ghost["neumann"]])
    closed = np.asarray(lsq_derivatives(topo, g, U, G, topo.lsq_closure_class[("t", "state")]))
    sample = np.asarray(lsq_derivatives(topo, g, U, G))
    worst_sample = 0.0
    for k, mi in enumerate(topo.lsq_monomial_multi_index):
        ref = f(cc, tuple(int(a) for a in mi))
        s = max(np.abs(ref).max(), 1.0)
        for r in range(2):
            err = np.abs(closed[r, :, k] - ref).max() / s
            assert err < 1e-9, (kind, p, mi, r, err)
        worst_sample = max(worst_sample, np.abs(sample[0, :, k] - ref).max() / s)
    assert worst_sample > 1e-6, worst_sample   # the Dirichlet ghost as a sample


def test_degree_one_keeps_the_ghost_sample():
    """A linear fit has no closure (its constraint would pin the gradient
    to the wall value); the D5 ghost sample is exact for linear data."""
    m = MESHES["quad"]()
    nbf = int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf], dtype=np.int8)
    topo = build_topology(m, lsq_degree=1, closure={("t", "state"): kinds})
    assert topo.lsq_closure_class == {} and topo.lsq_plan["closure"] is None


def test_kinds_from_the_ghost_kernel():
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    bcs = [BC.Wall(tag="left"), BC.Dirichlet("right", on="h", value=2.0),
           BC.Flux("bottom", on="h", gradient=0.3), BC.Extrapolation(tag="top")]
    sm = SystemModel.from_model(SME(level=0, dimension=3,
                                    closures=[Newtonian(), StressFree()],
                                    boundary_conditions=BC.BoundaryConditions(bcs)))
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2))
    # branches by sorted tag: bottom, left, right, top; faces: one of each,
    # an oblique left wall, and a partition cut (branch -1)
    branch = np.array([0, 1, 2, 3, 1, -1])
    normals = np.array([[0, -1, 1, 0, -0.6, 1], [-1, 0, 0, 1, 0.8, 0]], float)
    D, N, G = DIRICHLET, NEUMANN, GHOST
    state = face_kinds(nsm, "state", branch, normals)            # b, h, q_x, q_y
    assert state.tolist() == [[N, N, N, N, N, G],
                              [N, N, D, N, N, G],
                              [N, D, N, N, G, G],
                              [N, N, N, N, G, G]]
    recon = face_kinds(nsm, "reconstruction", branch, normals)   # h, eta, u_x, u_y
    assert recon.tolist() == [[N, N, D, N, N, G],
                              [N, N, G, N, N, G],
                              [G, D, G, N, G, G],
                              [G, N, G, N, G, G]]


def _rotated(m, theta):
    """Vertex coordinates of `m` rotated by `theta` about the origin."""
    X = np.asarray(m.vertex_coordinates, float)[:2]
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]]) @ X


@pytest.mark.parametrize("kind,p", [("quad", 2), ("quad", 3), ("jitter", 2), ("jitter", 3),
                                    ("tri", 2), ("tri", 3)])
def test_vector_closure_reproduces_polynomials(kind, p):
    """Momentum-like pair on walls at 35 deg: the normal component Dirichlet, the tangential
    Neumann (position-dependent data from the polynomial), one joint fit per cell, corners
    (two frames) included."""
    m = MESHES[kind]()
    nc, nbf = int(m.n_inner_cells), int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf, [NEUMANN] * nbf], dtype=np.int8)
    rc = RowClosure(kinds, ((0, 1),), np.ones((1, nbf), dtype=bool))
    topo = build_topology(m, lsq_degree=p, closure={("t", "state"): rc})
    g = compute_geometry(topo, _rotated(m, np.radians(35.0)))
    fx, fy = _poly(2, p, 300 + p), _poly(2, p, 400 + p)
    cc = np.asarray(g["cell_centers"])[:2, :nc]
    own = topo.boundary_owner
    n = np.asarray(g["face_normals"])[:2, topo.boundary_faces]
    n = n / np.linalg.norm(n, axis=0)
    t = np.array([-n[1], n[0]])
    dn = np.asarray(g["d_normal"])
    foot = cc[:, own] + dn * n
    dfn = [sum(n[a] * f(foot, tuple(int(a == b) for b in range(2))) for a in range(2))
           for f in (fx, fy)]
    u = np.stack([fx(cc), fy(cc)])
    un, ut = n[0] * u[0, own] + n[1] * u[1, own], t[0] * u[0, own] + t[1] * u[1, own]
    gn = 2 * (n[0] * fx(foot) + n[1] * fy(foot)) - un            # Dirichlet n.q
    gt = ut + 2 * dn * (t[0] * dfn[0] + t[1] * dfn[1])             # Neumann t.q
    G = np.stack([n[0] * gn + t[0] * gt, n[1] * gn + t[1] * gt])
    cls = topo.lsq_closure_class[("t", "state")]
    assert cls == (((0, 0), (1, 1)), ((0, 2), (1, 3))), cls
    closed = np.asarray(lsq_derivatives(topo, g, u, G, cls))
    sample = np.asarray(lsq_derivatives(topo, g, u, G))
    worst_sample = 0.0
    for k, mi in enumerate(topo.lsq_monomial_multi_index):
        for r, f in enumerate((fx, fy)):
            ref = f(cc, tuple(int(a) for a in mi))
            s = max(np.abs(ref).max(), 1.0)
            assert np.abs(closed[r, :, k] - ref).max() / s < 1e-9, (kind, p, mi, r)
            worst_sample = max(worst_sample, np.abs(sample[r, :, k] - ref).max() / s)
    assert worst_sample > 1e-3, worst_sample


def _wall_kinds(m, theta):
    """A Wall's kinds on the box rotated by `theta`: per row b, h (Neumann), q_x, q_y; the
    axis-aligned box per row, a rotated one in the normal frames."""
    nbf = int(m.n_boundary_faces)
    n = np.asarray(m.face_normals)[:2, np.asarray(m.boundary_face_face_indices)]
    kinds = np.full((4, nbf), NEUMANN, dtype=np.int8)
    if theta == 0.0:
        kinds[2] = np.where(np.abs(n[0]) > 0.5, DIRICHLET, NEUMANN)
        kinds[3] = np.where(np.abs(n[1]) > 0.5, DIRICHLET, NEUMANN)
        return RowClosure(kinds, ((2, 3),), np.zeros((1, nbf), dtype=bool))
    kinds[2] = DIRICHLET
    return RowClosure(kinds, ((2, 3),), np.ones((1, nbf), dtype=bool))


@pytest.mark.parametrize("kind,p", [("quad", 1), ("quad", 2), ("quad", 3), ("tri", 1),
                                    ("tri", 2), ("tri", 3), ("jitter", 2)])
def test_rotated_mesh_rotates_the_derivatives(kind, p):
    """The same mesh rotated by 30 deg: the same stencils (the decoupled ghost fit in the
    wall frame), every derivative of a smooth field (b, h scalars; q a vector, its ghost the
    wall's reflection) the rotated one to round-off."""
    from itertools import product
    m = MESHES[kind]()
    nc = int(m.n_inner_cells)
    th = np.radians(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = []
    for theta in (0.0, th):
        X = _rotated(m, theta)
        mm = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_arrays(
            X, np.asarray(m.cell_vertices), m.type)))
        topo = build_topology(mm, lsq_degree=p, closure={("t", "state"): _wall_kinds(mm, theta)}
                              if p >= 2 else {})
        g = compute_geometry(topo, X)
        cc = np.asarray(g["cell_centers"])[:2, :nc]
        xi = R.T @ cc if theta else cc                       # the box frame
        sc = 1.0 + 0.3 * np.sin(2.1 * xi[0] + 0.4) * np.cos(1.7 * xi[1] - 0.2)
        qb = np.stack([np.cos(1.3 * xi[0]) * np.sin(2.2 * xi[1] + 0.3),
                       np.sin(0.9 * xi[0] - 0.5) * np.cos(1.1 * xi[1])])
        q = R @ qb if theta else qb
        U = np.stack([0.2 * sc, sc, q[0], q[1]])
        n = np.asarray(g["face_normals"])[:2, topo.boundary_faces]
        n = n / np.linalg.norm(n, axis=0)
        qo = q[:, topo.boundary_owner]
        qg = qo - 2 * (n * qo).sum(axis=0) * n                   # the wall's reflection
        G = np.concatenate([U[:2, topo.boundary_owner], qg])
        cls = topo.lsq_closure_class.get(("t", "state"))
        D = np.asarray(lsq_derivatives(topo, g, U, G, cls))
        out.append((topo.lsq_neighbors.copy(), D, topo.lsq_monomial_multi_index))
    (S0, D0, mis), (S1, D1, _) = out
    # the same samples (ties of equal distance may come in another order)
    assert np.array_equal(np.sort(S0, axis=1), np.sort(S1, axis=1))
    idx = {tuple(int(v) for v in mi): i for i, mi in enumerate(mis)}

    def tensor(D, r, k):
        """The k-th derivative tensor of row r from the scaled monomial derivatives."""
        T = np.zeros((2,) * k + (nc,))
        for ax in product(range(2), repeat=k):
            T[ax] = D[r, :, idx[tuple(ax.count(a) for a in range(2))]]
        return T
    for k in range(1, p + 1):
        Rk = [R] * k
        for r in (0, 1):                                          # scalars
            T0, T1 = tensor(D0, r, k), tensor(D1, r, k)
            T0r = T0
            for a in range(k):
                T0r = np.moveaxis(np.tensordot(Rk[a], T0r, axes=([1], [a])), 0, a)
            assert np.abs(T1 - T0r).max() < 1e-10 * max(1.0, np.abs(T0).max()), (r, k)
        V0 = np.stack([tensor(D0, 2, k), tensor(D0, 3, k)])     # the vector: k + 1 indices
        V1 = np.stack([tensor(D1, 2, k), tensor(D1, 3, k)])
        for a in range(k + 1):
            V0 = np.moveaxis(np.tensordot(R, V0, axes=([1], [a])), 0, a)
        assert np.abs(V1 - V0).max() < 1e-10 * max(1.0, np.abs(V0).max()), k


def test_kinds_in_the_normal_frame():
    """`face_closure`: a Wall on an oblique face is Dirichlet in n.q and Neumann in t.q, on an
    aligned face per row as `face_kinds`."""
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    bcs = [BC.Wall(tag="left"), BC.Dirichlet("right", on="h", value=2.0),
           BC.Flux("bottom", on="h", gradient=0.3), BC.Extrapolation(tag="top")]
    sm = SystemModel.from_model(SME(level=0, dimension=3,
                                    closures=[Newtonian(), StressFree()],
                                    boundary_conditions=BC.BoundaryConditions(bcs)))
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2))
    branch = np.array([0, 1, 2, 3, 1, -1, 1])
    normals = np.array([[0, -1, 1, 0, -0.6, 1, 0.8], [-1, 0, 0, 1, 0.8, 0, 0.6]], float)
    D, N, G = DIRICHLET, NEUMANN, GHOST
    for space in ("state", "reconstruction"):
        rc = face_closure(nsm, space, branch, normals)
        cart = face_kinds(nsm, space, branch, normals)
        assert rc.groups == ((2, 3),)
        assert rc.rotated.tolist() == [[False, False, False, False, True, False, True]]
        keep = [0, 1, 2, 3, 5]
        assert rc.kinds[:, keep].tolist() == cart[:, keep].tolist()
        assert rc.kinds[2:, [4, 6]].tolist() == [[D, D], [N, N]]      # normal, tangential
        assert (cart[2:, [4, 6]] == G).all()


def test_closure_tables_differentiable():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    m = MESHES["tri"]()
    nbf = int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf, [NEUMANN] * nbf], dtype=np.int8)
    topo = build_topology(m, lsq_degree=2, closure={("t", "state"): kinds})
    X0 = jnp.asarray(np.asarray(m.vertex_coordinates, float)[:2])

    def J(X):
        return jnp.sum(compute_geometry(topo, X)["lsq_closure"] ** 2)
    gX = np.asarray(jax.grad(J)(X0))
    assert np.isfinite(gX).all() and np.abs(gX).max() > 0


def test_vector_closure_tables_differentiable():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    m = MESHES["tri"]()
    nbf = int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf, [NEUMANN] * nbf], dtype=np.int8)
    rc = RowClosure(kinds, ((0, 1),), np.ones((1, nbf), dtype=bool))
    topo = build_topology(m, lsq_degree=2, closure={("t", "state"): rc})
    X0 = jnp.asarray(_rotated(m, 0.4))

    def J(X):
        g = compute_geometry(topo, X)
        return jnp.sum(g["lsq_closure"] ** 2) + jnp.sum(g["lsq_gradQ"] ** 2)
    gX = np.asarray(jax.grad(J)(X0))
    assert np.isfinite(gX).all() and np.abs(gX).max() > 0


def test_wall_ghost_informs_the_normal_axis():
    """The decoupled degree-1 fit of a wall cell: the mirror ghost lies on the face normal, so
    it informs the derivative along the normal's axis, also on triangles stretched along the
    wall (there x_f - x_c points mostly along the wall).  A field of x alone, exact ghosts:
    the spurious wall-normal derivative stays small (0.16 with the ghost on the tangential
    axis, ANALYSIS 57)."""
    nx, ny, L = 25, 2, (0.0, 25.0, 0.0, 1.0)
    xs, ys = np.linspace(L[0], L[1], nx + 1), np.linspace(L[2], L[3], ny + 1)
    X = np.array([[x, y] for x in xs for y in ys]).T
    v = lambda i, k: i * (ny + 1) + k                                    # noqa: E731
    tris = []
    for i in range(nx):
        for k in range(ny):
            a, b, c, d = v(i, k), v(i + 1, k), v(i + 1, k + 1), v(i, k + 1)
            tris += [(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)]
    m = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_arrays(X, np.array(tris).T,
                                                                 "triangle")))
    topo = build_topology(m, lsq_degree=1, closure={})
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float)[:2])
    nc = int(m.n_inner_cells)
    cc = np.asarray(g["cell_centers"])[:2, :nc]
    f = lambda x: np.sin(0.4 * x) + 0.2 * x                              # noqa: E731
    own = topo.boundary_owner
    n = np.asarray(g["face_normals"])[:2, topo.boundary_faces]
    n = n / np.linalg.norm(n, axis=0)
    mirror = cc[:, own] + 2 * np.asarray(g["d_normal"]) * n
    D = np.asarray(lsq_derivatives(topo, g, f(cc[0])[None], f(mirror[0])[None]))[0]
    wall = np.zeros(nc, bool)
    wall[own] = (np.abs(n[1]) > 0.5)                                   # the y = 0, 1 walls
    dy = D[:, topo.lsq_grad_cols[1]]
    assert np.abs(dy[wall]).max() < 5e-2, np.abs(dy[wall]).max()


def test_vector_closure_reproduces_polynomials_3d():
    """A 3-component group on a rotated hexahedral box: n.q Dirichlet, the tangential plane
    Neumann (the rows of I - n n^T), one joint fit per cell, edges and corners included."""
    m = MESHES["hex"]()
    p = 2
    nc, nbf = int(m.n_inner_cells), int(m.n_boundary_faces)
    kinds = np.array([[DIRICHLET] * nbf, [NEUMANN] * nbf, [NEUMANN] * nbf], dtype=np.int8)
    rc = RowClosure(kinds, ((0, 1, 2),), np.ones((1, nbf), dtype=bool))
    topo = build_topology(m, lsq_degree=p, closure={("t", "state"): rc})
    a, b = np.radians(25.0), np.radians(40.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    g = compute_geometry(topo, Rx @ Rz @ np.asarray(m.vertex_coordinates, float)[:3])
    fs = [_poly(3, p, 500 + k) for k in range(3)]
    cc = np.asarray(g["cell_centers"])[:3, :nc]
    own = topo.boundary_owner
    n = np.asarray(g["face_normals"])[:3, topo.boundary_faces]
    n = n / np.linalg.norm(n, axis=0)
    dn = np.asarray(g["d_normal"])
    foot = cc[:, own] + dn * n
    u = np.stack([f(cc) for f in fs])
    uf = np.stack([f(foot) for f in fs])
    du = np.stack([sum(n[c] * f(foot, tuple(int(c == e) for e in range(3))) for c in range(3))
                   for f in fs])                                   # d/dn of each component
    uo = u[:, own]
    un, ufn = (n * uo).sum(0), (n * uf).sum(0)
    tan = lambda v: v - (n * v).sum(0) * n                          # noqa: E731
    G = n * (2 * ufn - un) + tan(uo) + 2 * dn * tan(du)            # Dirichlet n.q, Neumann t.q
    closed = np.asarray(lsq_derivatives(topo, g, u, G, topo.lsq_closure_class[("t", "state")]))
    for k, mi in enumerate(topo.lsq_monomial_multi_index):
        for r, f in enumerate(fs):
            ref = f(cc, tuple(int(v) for v in mi))
            s = max(np.abs(ref).max(), 1.0)
            assert np.abs(closed[r, :, k] - ref).max() / s < 1e-9, (mi, r)
