"""Round-trip tests for ``mesh_to_gmsh``: BaseMesh -> gmsh .msh -> BaseMesh.

Asserts dimension / n_inner_cells / type survive, the named physical groups
land in ``meshio.field_data``, and ``BaseMesh.from_msh`` re-reads the boundary
tags.
"""
import meshio
import numpy as np
import pytest

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_prepost import mesh_to_gmsh


def _read_back(path):
    return BaseMesh.from_msh(str(path))


def test_2d_quad_per_side_tags(tmp_path):
    mesh = BaseMesh.create_2d(domain=(0.0, 10.0, 0.0, 10.0), nx=6, ny=4)
    out = tmp_path / "mesh.msh"
    mesh_to_gmsh(mesh, out, boundary_group=None)  # preserve left/right/bottom/top

    mio = meshio.read(str(out))
    assert mio.field_data  # non-empty named groups
    names = set(mio.field_data.keys())
    assert {"left", "right", "bottom", "top"} <= names
    assert "domain" in names

    rt = _read_back(out)
    assert rt.dimension == 2
    assert rt.type == "quad"
    assert rt.n_inner_cells == mesh.n_inner_cells
    assert rt.n_boundary_faces == mesh.n_boundary_faces
    got = set(rt.boundary_conditions_sorted_names)
    assert {"left", "right", "bottom", "top"} <= got


def test_2d_single_default_group(tmp_path):
    mesh = BaseMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=5, ny=5)
    out = tmp_path / "mesh.msh"
    mesh_to_gmsh(mesh, out)  # default -> single "default" group

    mio = meshio.read(str(out))
    assert "default" in mio.field_data
    assert "domain" in mio.field_data
    # only one boundary group name (besides the volume "domain")
    bnd_groups = [n for n, d in mio.field_data.items() if d[1] == 1]
    assert bnd_groups == ["default"]

    rt = _read_back(out)
    assert rt.dimension == 2 and rt.type == "quad"
    assert list(rt.boundary_conditions_sorted_names) == ["default"]
    assert rt.n_boundary_faces == mesh.n_boundary_faces


def test_1d_line(tmp_path):
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=8)
    out = tmp_path / "mesh1d.msh"
    mesh_to_gmsh(mesh, out, boundary_group=None)

    mio = meshio.read(str(out))
    assert mio.field_data
    types = {c.type for c in mio.cells}
    assert "line" in types      # 1-D volume cells
    assert "vertex" in types    # 0-D boundary elements

    rt = _read_back(out)
    assert rt.dimension == 1
    assert rt.type == "line"
    assert rt.n_inner_cells == mesh.n_inner_cells


def test_from_stripped_h5_reconstructs_boundary(tmp_path):
    """A mesh.h5 with only vertex/cell arrays (no boundary datasets) still
    yields a valid .msh with a reconstructed default boundary group."""
    import h5py

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

    rt = _read_back(out)
    assert rt.dimension == 2 and rt.type == "quad"
    assert rt.n_inner_cells == mesh.n_inner_cells
    # perimeter of a 4x3 quad mesh = 2*(4+3) = 14 boundary faces
    assert rt.n_boundary_faces == 2 * (4 + 3)
    assert list(rt.boundary_conditions_sorted_names) == ["default"]


def test_full_h5_path_input_preserves_sides(tmp_path):
    """Path to a full BaseMesh h5 preserves per-side names when requested."""
    mesh = BaseMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=3, ny=3)
    h5 = tmp_path / "full.h5"
    mesh.write_to_hdf5(str(h5))

    out = tmp_path / "full.msh"
    mesh_to_gmsh(str(h5), out, boundary_group=None)

    rt = _read_back(out)
    assert {"left", "right", "bottom", "top"} <= set(rt.boundary_conditions_sorted_names)
