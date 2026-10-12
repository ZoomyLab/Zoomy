"""Gate 1a — the boundary closure of the LSQ derivative fits (ANALYSIS 40).

* A row whose ghost is Dirichlet-type (``2 v - Q``) or Neumann-type
  (``Q + 2 d_n g``) closes by a constraint at the foot point (fits of
  degree >= 2; degree 1 keeps the ghost sample): the fit
  reproduces every derivative of a polynomial of the fit's degree in every
  boundary cell -- where the same ghost read as a fit SAMPLE does not
  (Dirichlet from degree 2, Neumann from degree 3).
* The kinds are read off the real D5 ghost kernel: a wall reflects q_n
  (Dirichlet) and keeps q_t, h (Neumann) on an axis-aligned face, couples the
  momentum on an oblique one (ghost sample); reconstruction variables
  (h, eta, u) map through the model's W(Q).
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
    DIRICHLET, GHOST, NEUMANN, face_kinds)

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
