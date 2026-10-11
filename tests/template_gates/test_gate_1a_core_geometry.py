"""Gate 1a-2 — single-source geometry (``zoomy_core.mesh.geometry_xp``).

``build_topology`` (numpy, static) + ``compute_geometry`` (xp, gather-only)
must reproduce FVMMesh / LSQMesh — cell centres incl. ghosts, volumes,
inradii, face normals / areas / centres, the LSQ stencil and ``lsq_gradQ``
(incl. the REQ-174b ghost decoupling) — on create_1d, create_2d and a gmsh
triangle mesh, at LSQ degrees 1 and 2, to round-off (relative 1e-14), plus
the derived boundary / TPFA / routing tables.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.fvm.reconstruction import DiffusionOperatorV2
from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.geometry_xp import (
    build_topology, compute_geometry, lsq_derivatives)

from test_gate_1a_core_bc import write_tri_msh

pytestmark = [pytest.mark.template_gate]

FIELDS = ("cell_centers", "cell_volumes", "cell_inradius", "face_normals",
          "face_volumes", "face_centers", "lsq_gradQ")


@pytest.fixture(scope="module")
def gmsh_path(tmp_path_factory):
    return write_tri_msh(str(tmp_path_factory.mktemp("msh") / "tri.msh"))


def _mesh(kind, gmsh_path, degree):
    if kind == "1d":
        m = LSQMesh.create_1d((0.0, 2.0), 17)
    elif kind == "2d":
        m = LSQMesh.create_2d((0.0, 1.0, -0.5, 0.2), 7, 4)
    else:
        m = LSQMesh.from_msh(gmsh_path)
    if degree > 1:
        m._build_lsq_stencil(degree)
    return m


def _rel(a, r):
    a, r = np.asarray(a, float), np.asarray(r, float)
    assert a.shape == r.shape, (a.shape, r.shape)
    return np.abs(a - r).max() / max(np.abs(r).max(), 1e-300)


@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("kind", ("1d", "2d", "gmsh"))
def test_geometry_matches_fvm_lsq_mesh(kind, degree, gmsh_path):
    m = _mesh(kind, gmsh_path, degree)
    topo = build_topology(m, lsq_degree=degree)
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float))
    for k in FIELDS:
        assert _rel(g[k], getattr(m, k)) <= 1e-14, k
    assert np.array_equal(topo.lsq_neighbors, m.lsq_neighbors)
    assert np.array_equal(topo.lsq_boundary_face_neighbors,
                          m.lsq_boundary_face_neighbors)
    assert list(topo.lsq_monomial_multi_index) == list(m.lsq_monomial_multi_index)
    assert np.array_equal(topo.lsq_scale_factors, m.lsq_scale_factors)


@pytest.mark.parametrize("kind", ("1d", "2d", "gmsh"))
def test_derived_tables(kind, gmsh_path):
    m = _mesh(kind, gmsh_path, 1)
    topo = build_topology(m)
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float))
    dim = int(m.dimension)
    nc = m.n_inner_cells
    fc, cc = np.asarray(m.face_centers), np.asarray(m.cell_centers)
    bf = np.asarray(m.boundary_face_face_indices)
    own = m.face_cells[0, bf]
    # d_face / ghost centres / face-minus-cell vectors
    d_ref = np.linalg.norm(fc[bf, :dim] - cc[:dim, own].T, axis=1)
    assert _rel(g["d_face"], d_ref) <= 1e-14
    assert _rel(g["ghost_centers"], cc[:, m.face_cells[1, bf]]) <= 1e-14
    it = topo.interior_faces
    assert _rel(g["r_A"], fc[it, :dim].T - cc[:dim, m.face_cells[0, it]]) <= 1e-14
    assert _rel(g["r_I"], fc[bf, :dim].T - cc[:dim, own]) <= 1e-14
    # face inradius: the local-CFL size (min of the two inner cells)
    ir = np.asarray(m.cell_inradius)
    inner = (m.face_cells[0] < nc) & (m.face_cells[1] < nc)
    ref = np.where(inner, np.minimum(ir[m.face_cells[0]],
                                     ir[np.where(inner, m.face_cells[1], 0)]),
                   ir[m.face_cells[0]])
    assert _rel(g["face_inradius"], ref) <= 1e-14
    # TPFA geometry == DiffusionOperatorV2 (nu-free part)
    op = DiffusionOperatorV2(m, dim, nu=1.0)
    assert np.array_equal(op._ia, topo.tpfa_A) and np.array_equal(op._ib, topo.tpfa_B)
    assert _rel(g["tpfa_gA"], op._gA) <= 1e-14
    assert _rel(g["tpfa_gB"], op._gB) <= 1e-14
    assert _rel(g["tpfa_bnd_g"], op._bf_g) <= 1e-14
    # normal distance <= distance, equal on the orthogonal meshes
    assert np.all(np.asarray(g["d_normal"]) <= np.asarray(g["d_face"]) + 1e-15)
    if kind != "gmsh":
        assert _rel(g["d_normal"], g["d_face"]) <= 1e-14


@pytest.mark.parametrize("kind", ("1d", "2d", "gmsh"))
def test_lsq_derivatives_match_mesh(kind, gmsh_path):
    m = _mesh(kind, gmsh_path, 2)
    topo = build_topology(m, lsq_degree=2)
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float))
    rng = np.random.default_rng(1)
    u = rng.normal(size=m.n_cells)
    ug = rng.normal(size=m.n_boundary_faces)
    # LSQMesh takes FACE values and lifts them to 2·face − cell; we take the
    # ghost directly (D5)
    ref = m.compute_derivatives(u, u_boundary_face=(ug + u[m.boundary_face_cells]) / 2)
    got = np.asarray(lsq_derivatives(topo, g, u[None, :], ug[None, :]))[0]
    assert _rel(got, ref) <= 1e-13


def test_periodic_partners_and_routing():
    """Periodic BC input cells == ``resolve_periodic_bcs`` (computed without
    mutating the mesh); branches routed by tag NAME."""
    m = LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), 5, 4)
    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
        BC.Wall(tag="bottom", momentum_field_indices=[[2, 3]]),
        BC.Extrapolation(tag="top")])
    before = np.array(m.boundary_face_cells, copy=True)
    topo = build_topology(m, bcs=bcs)
    assert np.array_equal(np.asarray(m.boundary_face_cells), before)
    m2 = copy.deepcopy(m)
    m2.resolve_periodic_bcs(bcs)
    assert np.array_equal(topo.bc_input_cells, np.asarray(m2.boundary_face_cells))
    assert topo.bc_periodic.sum() == 2 * 4
    assert np.array_equal(topo.bc_branch, BC.boundary_face_branches(m, bcs))
    names = list(m.boundary_conditions_sorted_names)
    tags = [names[k] for k in m.boundary_face_function_numbers]
    srt = list(bcs.list_sorted_function_names)
    assert all(srt[b] == t for b, t in zip(topo.bc_branch, tags))


def test_geometry_3d_matches_cell_loops():
    """Hexahedra (sheared: planar faces, non-orthogonal cells) vs the
    per-cell loops of BaseMesh."""
    from zoomy_core.mesh.base_mesh import BaseMesh
    m = BaseMesh.create_3d((0.0, 1.0, 0.0, 2.0, 0.0, 0.5), 3, 4, 2)
    X = np.asarray(m.vertex_coordinates, float)
    X = np.array([X[0] + 0.3 * X[1], X[1] - 0.2 * X[2], X[2] + 0.1 * X[0]])
    m.vertex_coordinates = X
    g = compute_geometry(build_topology(m, lsq_degree=1), X, lsq=False)
    nc = m.n_inner_cells
    assert _rel(g["cell_volumes"][:nc], m.cell_volumes_computed()[:nc]) < 1e-14
    assert _rel(g["cell_centers"][:, :nc], m.cell_centers_computed()[:, :nc]) < 1e-14
    assert _rel(g["face_normals"], m.face_normals_computed()) < 1e-14
    assert _rel(g["face_volumes"], m.face_volumes_computed()) < 1e-14
    assert _rel(g["face_centers"], m.face_centers_computed()) < 1e-14
    assert _rel(g["cell_inradius"][:nc], m.cell_inradius_computed()[:nc]) < 1e-14
    assert abs(g["cell_volumes"][:nc].sum() - (1 - 0.3 * 0.02)) < 1e-14   # det of the shear


def test_topology_rejects_unsupported_type():
    from zoomy_core.mesh.base_mesh import BaseMesh
    m = BaseMesh.create_3d((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), 2, 2, 2)
    m.type = "wface"
    with pytest.raises(NotImplementedError):
        build_topology(m)
