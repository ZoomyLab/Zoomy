"""Convert between Zoomy HDF5 meshes and Gmsh .msh files.

The main challenge is preserving boundary physical groups during round-trips.
Gmsh stores boundary elements (lines in 2D, triangles/quads in 3D) with
physical tags; BaseMesh stores boundary_face_physical_tags + sorted names.
This module bridges the two representations.
"""

from __future__ import annotations

import numpy as np
import meshio

from zoomy_core.mesh.base_mesh import (
    BaseMesh,
    _build_face_topology,
    _local_faces_for_type,
)


# ── face-element type for each cell type ─────────────────────────────────────
# Maps (zoomy cell type) -> meshio element type for its faces.
_BOUNDARY_ELEM_TYPE = {
    "line":        "vertex",      # 1D cells have point (vertex) faces
    "triangle":    "line",        # 2D tri -> line boundary elements
    "quad":        "line",        # 2D quad -> line boundary elements
    "tetra":       "triangle",    # 3D tet -> triangle boundary faces
    "hexahedron":  "quad",        # 3D hex -> quad boundary faces
}


def _face_vertex_count(mesh_type: str) -> int:
    """Number of vertices per face for a given cell type."""
    local = _local_faces_for_type(mesh_type)
    return len(local[0])


def h5_to_msh(h5_path: str, msh_path: str, file_format: str = "gmsh22") -> None:
    """Convert a Zoomy HDF5 mesh to Gmsh .msh with physical groups.

    Parameters
    ----------
    h5_path : str
        Path to the input HDF5 file written by ``BaseMesh.write_to_hdf5``.
    msh_path : str
        Path for the output ``.msh`` file.
    file_format : str
        meshio file format string (default ``"gmsh22"``).
    """
    mesh = BaseMesh.from_hdf5(h5_path)

    dim = mesh.dimension
    mesh_type = mesh.type
    n_inner = mesh.n_inner_cells

    # ── 1. Points ────────────────────────────────────────────────────────
    # vertex_coordinates is (dim, n_vertices); meshio wants (n_vertices, 3)
    n_verts = mesh.n_vertices
    points = np.zeros((n_verts, 3), dtype=float)
    points[:, :dim] = mesh.vertex_coordinates[:dim, :].T

    # ── 2. Primary (volume) cells ────────────────────────────────────────
    # cell_vertices is (verts_per_cell, n_inner_cells)
    meshio_cell_type = {
        "line": "line",
        "triangle": "triangle",
        "quad": "quad",
        "tetra": "tetra",
        "hexahedron": "hexahedron",
    }[mesh_type]
    primary_cells = mesh.cell_vertices[:, :n_inner].T  # (n_inner, vpc)

    # ── 3. Boundary face vertices ────────────────────────────────────────
    # Rebuild face_list from cell_vertices to get vertex indices per face.
    _, _, face_list = _build_face_topology(mesh.cell_vertices, mesh_type)

    bnd_elem_type = _BOUNDARY_ELEM_TYPE[mesh_type]
    n_bf = mesh.n_boundary_faces

    # Group boundary faces by physical tag so each tag becomes a cell block.
    # tag -> list of face vertex arrays
    tag_to_face_verts: dict[int, list[np.ndarray]] = {}
    for i_bf in range(n_bf):
        fidx = mesh.boundary_face_face_indices[i_bf]
        ptag = int(mesh.boundary_face_physical_tags[i_bf])
        # face_list[fidx] is a sorted tuple of vertex indices
        fverts = np.array(face_list[fidx], dtype=int)
        tag_to_face_verts.setdefault(ptag, []).append(fverts)

    # ── 4. Build meshio cells + cell_data + field_data ───────────────────
    cells = [meshio.CellBlock(meshio_cell_type, primary_cells)]
    phys_tags_data = [np.zeros(n_inner, dtype=int)]  # volume cells: tag 0

    # Build field_data: name -> (physical_tag, topological_dim)
    # Gmsh field_data maps name -> [physical_tag, topological_dim].
    field_data: dict[str, list[int]] = {}
    bnd_topo_dim = dim - 1  # boundary elements are one dimension lower

    sorted_tags = mesh.boundary_conditions_sorted_physical_tags
    sorted_names = mesh.boundary_conditions_sorted_names

    for i_tag, (ptag, name) in enumerate(zip(sorted_tags, sorted_names)):
        ptag = int(ptag)
        field_data[name] = [ptag, bnd_topo_dim]

        face_verts_list = tag_to_face_verts.get(ptag, [])
        if not face_verts_list:
            continue

        bnd_data = np.array(face_verts_list, dtype=int)
        cells.append(meshio.CellBlock(bnd_elem_type, bnd_data))
        phys_tags_data.append(np.full(len(face_verts_list), ptag, dtype=int))

    cell_data = {"gmsh:physical": phys_tags_data}

    meshio_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        field_data=field_data,
        cell_data=cell_data,
    )
    meshio.write(msh_path, meshio_mesh, file_format=file_format)


def msh_to_h5(msh_path: str, h5_path: str) -> None:
    """Convert a Gmsh .msh file to Zoomy HDF5 format.

    Parameters
    ----------
    msh_path : str
        Path to the input ``.msh`` file.
    h5_path : str
        Path for the output HDF5 file.
    """
    mesh = BaseMesh.from_msh(msh_path)
    mesh.write_to_hdf5(h5_path)
