"""#3 test_mesh_gmsh_roundtrip — BaseMesh <-> gmsh .msh (absorbs
test_mesh_to_gmsh): per-side boundary tags survive the round-trip, and a
STRIPPED mesh.h5 (vertex/cell arrays only) still reconstructs a boundary.
"""
from __future__ import annotations

import h5py
import meshio
import pytest

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_prepost import mesh_to_gmsh

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]


def test_2d_quad_per_side_tags(tmp_path):
    mesh = BaseMesh.create_2d(domain=(0.0, 10.0, 0.0, 10.0), nx=6, ny=4)
    out = tmp_path / "mesh.msh"
    mesh_to_gmsh(mesh, out, boundary_group=None)  # preserve left/right/bottom/top

    mio = meshio.read(str(out))
    names = set(mio.field_data.keys())
    assert {"left", "right", "bottom", "top"} <= names
    assert "domain" in names

    rt = BaseMesh.from_msh(str(out))
    assert rt.dimension == 2 and rt.type == "quad"
    assert rt.n_inner_cells == mesh.n_inner_cells
    assert rt.n_boundary_faces == mesh.n_boundary_faces
    assert {"left", "right", "bottom", "top"} <= \
        set(rt.boundary_conditions_sorted_names)


def test_1d_line_roundtrip(tmp_path):
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=8)
    out = tmp_path / "mesh1d.msh"
    mesh_to_gmsh(mesh, out, boundary_group=None)

    mio = meshio.read(str(out))
    types = {c.type for c in mio.cells}
    assert "line" in types and "vertex" in types  # 1-D volume + 0-D boundary

    rt = BaseMesh.from_msh(str(out))
    assert rt.dimension == 1 and rt.type == "line"
    assert rt.n_inner_cells == mesh.n_inner_cells


def test_stripped_h5_reconstructs_boundary(tmp_path):
    """A mesh.h5 with ONLY vertex/cell arrays still yields a valid .msh with
    a reconstructed single 'default' boundary group."""
    mesh = BaseMesh.create_2d(domain=(0.0, 2.0, 0.0, 1.0), nx=4, ny=3)
    h5 = tmp_path / "stripped.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("dimension", data=mesh.dimension)
        g.create_dataset("type", data=mesh.type)
        g.create_dataset("n_cells", data=mesh.n_cells)
        g.create_dataset("n_inner_cells", data=mesh.n_inner_cells)
        g.create_dataset("vertex_coordinates", data=mesh.vertex_coordinates)
        g.create_dataset("cell_vertices", data=mesh.cell_vertices)

    out = tmp_path / "from_stripped.msh"
    mesh_to_gmsh(str(h5), out)  # path input, single default group

    rt = BaseMesh.from_msh(str(out))
    assert rt.dimension == 2 and rt.type == "quad"
    assert rt.n_inner_cells == mesh.n_inner_cells
    assert rt.n_boundary_faces == 2 * (4 + 3)  # perimeter of a 4x3 quad grid
    assert list(rt.boundary_conditions_sorted_names) == ["default"]


def test_full_h5_path_preserves_sides(tmp_path):
    """Path to a FULL BaseMesh h5 keeps the per-side names when asked."""
    mesh = BaseMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=3, ny=3)
    h5 = tmp_path / "full.h5"
    mesh.write_to_hdf5(str(h5))

    out = tmp_path / "full.msh"
    mesh_to_gmsh(str(h5), out, boundary_group=None)

    rt = BaseMesh.from_msh(str(out))
    assert {"left", "right", "bottom", "top"} <= \
        set(rt.boundary_conditions_sorted_names)
