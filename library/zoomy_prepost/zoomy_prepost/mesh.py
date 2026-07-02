"""zoomy ``BaseMesh`` -> gmsh ``.msh`` converter.

numpy / jax read the zoomy HDF5 (``mesh.h5``); dmplex (PETSc DMPlex) and
firedrake read a gmsh ``.msh`` whose *named* physical groups are the boundary
BC tags.  A shared case therefore needs BOTH formats emitted from the one
in-memory mesh.  This module is that single writer — no per-backend mesh
export.

Output (gmsh 2.2 ASCII, the safest interchange):

* one volume physical group ``"domain"`` (all inner cells), and
* the boundary faces as ``dim-1`` elements (line / vertex / quad / triangle)
  grouped into named physical group(s).  With ``boundary_group=None`` the
  mesh's per-side names (``left/right/bottom/top`` from
  ``BaseMesh.create_2d``) are preserved so a per-side BC model routes
  identically on every backend; with ``boundary_group="default"`` (the
  default) every boundary face lands in a single catch-all group, matching
  the ``FromModel(tag="default", definition="wall")`` pattern.

``BaseMesh.from_msh`` reads the named groups back via ``meshio.field_data``;
firedrake matches them to the model's BC tag names.

Face topology and face ordering are taken from ``zoomy_core`` (no local
re-derivation): ``_local_faces_for_type`` (the exact ordering used to build
``cell_faces``) recovers a boundary face's vertices from ``(cell, face)``, and
``_build_face_topology`` reconstructs boundaries from a stripped ``mesh.h5``
(``misc.io.write_mesh_to_hdf5`` stores no boundary arrays).
"""
from __future__ import annotations

import os

import numpy as np

# meshio boundary element type by vertex count (dim-1 elements).
_BND_TYPE_BY_NVERT = {1: "vertex", 2: "line", 3: "triangle", 4: "quad"}


def _points_3d(vertex_coordinates: np.ndarray) -> np.ndarray:
    """(dim, n_vert) zoomy coords -> (n_vert, 3) gmsh points (z-padded)."""
    vc = np.asarray(vertex_coordinates, dtype=float)
    n_vert = vc.shape[1]
    pts = np.zeros((n_vert, 3), dtype=float)
    rows = min(vc.shape[0], 3)
    pts[:, :rows] = vc[:rows, :].T
    return pts


def _from_basemesh(mesh):
    """Extract (dim, ztype, points, vol_conn, boundary_faces) from a BaseMesh.

    ``boundary_faces`` is a list of ``(vertex_id_tuple, side_name)`` where
    ``side_name`` is the mesh's per-side name for that face, or ``None`` when
    the mesh carries no named tags.
    """
    from zoomy_core.mesh.base_mesh import _local_faces_for_type, _build_face_topology

    dim = int(mesh.dimension)
    ztype = str(mesh.type)
    n_inner = int(mesh.n_inner_cells)
    cell_vertices = np.asarray(mesh.cell_vertices)[:, :n_inner]
    vol_conn = cell_vertices.T.astype(int)
    points = _points_3d(mesh.vertex_coordinates)

    n_bf = int(getattr(mesh, "n_boundary_faces", 0) or 0)
    bff = np.asarray(getattr(mesh, "boundary_face_face_indices", np.empty(0)))
    cell_faces = np.asarray(getattr(mesh, "cell_faces", np.empty((0, 0))))

    boundary_faces: list = []

    if n_bf > 0 and bff.size == n_bf and cell_faces.size > 0:
        # Full topology: recover each boundary face's vertices from its inner
        # cell + the local face index, using the SAME ordering that built
        # cell_faces.  Tags -> per-side names via the sorted-tag map.
        local_faces = _local_faces_for_type(ztype)
        b_cells = np.asarray(mesh.boundary_face_cells)
        b_tags = np.asarray(mesh.boundary_face_physical_tags)
        sorted_tags = np.asarray(mesh.boundary_conditions_sorted_physical_tags)
        sorted_names = list(mesh.boundary_conditions_sorted_names)
        tag_to_name = {int(t): str(n) for t, n in zip(sorted_tags, sorted_names)}
        for i_bf in range(n_bf):
            ic = int(b_cells[i_bf])
            fidx = int(bff[i_bf])
            matches = np.where(cell_faces[:, ic] == fidx)[0]
            if matches.size == 0:
                continue
            lf = int(matches[0])
            verts = tuple(int(cell_vertices[i, ic]) for i in local_faces[lf])
            side = tag_to_name.get(int(b_tags[i_bf])) if sorted_names else None
            boundary_faces.append((verts, side))
    else:
        # No boundary arrays (stripped mesh.h5): a boundary face is one owned
        # by exactly one cell.  Reconstruct via the core topology builder.
        cell_faces_r, face_cells_r, face_list = _build_face_topology(cell_vertices, ztype)
        for fidx, fkey in enumerate(face_list):
            if face_cells_r[1, fidx] == -1:
                boundary_faces.append((tuple(int(v) for v in fkey), None))

    return dim, ztype, points, vol_conn, boundary_faces


def _load_mesh(path):
    """Load a BaseMesh from a ``mesh.h5``.

    Prefers ``BaseMesh.from_hdf5`` (full topology incl. boundary arrays, as
    written by ``BaseMesh.write_to_hdf5``).  A stripped file written by
    ``misc.io.write_mesh_to_hdf5`` lacks those datasets; fall back to a
    minimal BaseMesh carrying only the vertex/cell arrays so boundaries are
    reconstructed downstream.
    """
    from zoomy_core.mesh.base_mesh import BaseMesh

    try:
        return BaseMesh.from_hdf5(str(path))
    except (KeyError, ValueError):
        pass

    import h5py

    with h5py.File(str(path), "r") as f:
        g = f["mesh"]
        mtype = g["type"][()]
        mtype = mtype.decode() if isinstance(mtype, bytes) else str(mtype)
        cell_vertices = np.asarray(g["cell_vertices"][()])
        kwargs = dict(
            dimension=int(g["dimension"][()]),
            type=mtype,
            vertex_coordinates=np.asarray(g["vertex_coordinates"][()]),
            cell_vertices=cell_vertices,
            n_inner_cells=int(g["n_inner_cells"][()]) if "n_inner_cells" in g
            else cell_vertices.shape[1],
            n_boundary_faces=0,
        )
    return BaseMesh(**kwargs)


def mesh_to_gmsh(mesh, out_msh, *, boundary_group="default"):
    """Write a zoomy ``BaseMesh`` (or a ``mesh.h5`` path) to a gmsh ``.msh``.

    Parameters
    ----------
    mesh : BaseMesh | str | os.PathLike
        An in-memory ``BaseMesh`` or a path to a ``mesh.h5``.
    out_msh : str | os.PathLike
        Destination ``.msh`` (gmsh 2.2 ASCII, overwritten).
    boundary_group : str | None, keyword-only
        ``"default"`` (the default): all boundary faces go into ONE physical
        group with this name — the catch-all ``default`` -> wall target.
        ``None``: preserve the mesh's per-side named tags (e.g.
        ``left/right/bottom/top``); if the mesh has none, a single
        ``"default"`` group is emitted.

    Returns
    -------
    str
        ``out_msh``.
    """
    import meshio

    from zoomy_core.mesh.mesh_util import convert_mesh_type_to_meshio_mesh_type

    if isinstance(mesh, (str, os.PathLike)):
        mesh = _load_mesh(mesh)

    dim, ztype, points, vol_conn, boundary_faces = _from_basemesh(mesh)

    # Resolve each boundary face's group name.
    def _name(side):
        if boundary_group is not None:
            return boundary_group
        return side if side is not None else "default"

    named_faces = [(verts, _name(side)) for verts, side in boundary_faces]

    # Physical-group ids: domain=1, then boundary groups by first appearance.
    domain_name = "domain"
    field_data = {domain_name: np.array([1, dim], dtype=int)}
    group_id: dict[str, int] = {}
    next_id = 2
    for _verts, name in named_faces:
        if name not in group_id:
            group_id[name] = next_id
            field_data[name] = np.array([next_id, max(dim - 1, 0)], dtype=int)
            next_id += 1

    # Volume block.
    vol_type = convert_mesh_type_to_meshio_mesh_type(ztype)
    cells = [(vol_type, vol_conn)]
    phys = [np.full(vol_conn.shape[0], 1, dtype=int)]

    # Boundary blocks, bucketed by element type (vertex/line/triangle/quad).
    buckets: dict[str, dict] = {}
    for verts, name in named_faces:
        etype = _BND_TYPE_BY_NVERT.get(len(verts))
        if etype is None:
            raise ValueError(f"unsupported boundary face with {len(verts)} vertices")
        b = buckets.setdefault(etype, {"conn": [], "ids": []})
        b["conn"].append(list(verts))
        b["ids"].append(group_id[name])
    for etype, b in buckets.items():
        cells.append((etype, np.asarray(b["conn"], dtype=int)))
        phys.append(np.asarray(b["ids"], dtype=int))

    cell_data = {"gmsh:physical": phys, "gmsh:geometrical": [p.copy() for p in phys]}
    out = meshio.Mesh(points, cells, cell_data=cell_data, field_data=field_data)

    out_msh = str(out_msh)
    os.makedirs(os.path.dirname(os.path.abspath(out_msh)) or ".", exist_ok=True)
    meshio.write(out_msh, out, file_format="gmsh22", binary=False)
    return out_msh
