"""BaseMesh — topology-only mesh loaded from .msh via meshio (no PETSc).

All geometric quantities (volumes, normals, centers, …) are computed on the
fly from the stored topology.  Derived classes (``FVMMesh``, ``LSQMesh``, …)
cache selected quantities for performance.
"""

from __future__ import annotations

import os
from copy import deepcopy
from math import factorial
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import param

import zoomy_core.mesh.mesh_util as mesh_util
from zoomy_core.mesh.mesh_util import compute_subvolume

# Optional deps
try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

try:
    import meshio
    _HAVE_MESHIO = True
except ImportError:
    _HAVE_MESHIO = False


# ── helpers: meshio cell-type → zoomy type ──────────────────────────────────

_MESHIO_TO_ZOOMY = {
    "vertex": "vertex",
    "line": "line",
    "triangle": "triangle",
    "quad": "quad",
    "tetra": "tetra",
    "hexahedron": "hexahedron",
    "wedge": "wface",
}

_ZOOMY_TO_MESHIO = {v: k for k, v in _MESHIO_TO_ZOOMY.items()}

_FACES_PER_CELL = {
    "line": 2, "triangle": 3, "quad": 4,
    "tetra": 4, "hexahedron": 6, "wface": 5,
}

_DIM_OF_TYPE = {
    "vertex": 0, "line": 1, "triangle": 2, "quad": 2,
    "tetra": 3, "hexahedron": 3, "wface": 3,
}


# ── topology builder: extract faces from cell-vertex connectivity ───────────

def _local_faces_for_type(mesh_type: str):
    """Return a list of tuples: for each cell, the local vertex indices that
    form each face.  E.g. for a triangle: [(0,1), (1,2), (2,0)]."""
    if mesh_type == "line":
        return [(0,), (1,)]
    if mesh_type == "triangle":
        return [(0, 1), (1, 2), (2, 0)]
    if mesh_type == "quad":
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if mesh_type == "tetra":
        return [(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    if mesh_type == "hexahedron":
        return [
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (1, 2, 6, 5),
            (2, 3, 7, 6), (3, 0, 4, 7),
        ]
    if mesh_type == "wface":
        return [
            (0, 1, 2), (3, 4, 5),
            (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0),
        ]
    raise ValueError(f"Unsupported mesh type: {mesh_type}")


def _build_face_topology(cell_vertices: np.ndarray, mesh_type: str):
    """Build face-cell connectivity from cell-vertex data (vectorised).

    Faces are numbered in order of first appearance, cells in order and their
    local faces in ``_local_faces_for_type`` order; ``face_cells[0]`` is the
    first cell visiting a face, ``face_cells[1]`` the last other one.

    Returns
    -------
    cell_faces : (n_faces_per_cell, n_cells) int
    face_cells : (2, n_faces) int  — second entry is -1 for boundary faces
    face_vertex_indices : list of tuple — global vertex indices per face (sorted)
    """
    local_faces = _local_faces_for_type(mesh_type)
    nfpc = len(local_faces)
    cv = np.asarray(cell_vertices)
    n_cells = cv.shape[1]
    width = max(len(lf) for lf in local_faces)
    keys = -np.ones((n_cells, nfpc, width), dtype=np.int64)   # pad: -1 first
    for l, lf in enumerate(local_faces):
        keys[:, l, width - len(lf):] = np.sort(cv[list(lf)].T, axis=1)
    keys = keys.reshape(n_cells * nfpc, width)                # order ic*nfpc + l
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first, kind="stable")                  # by first visit
    number = np.empty(len(first), dtype=int)
    number[order] = np.arange(len(first))
    face = number[inverse]
    n_faces = len(first)
    cell_faces = face.reshape(n_cells, nfpc).T.copy()
    visit = np.arange(n_cells * nfpc)
    face_cells = np.empty((2, n_faces), dtype=int)
    face_cells[0] = first[order] // nfpc
    last = np.full(n_faces, -1, dtype=np.int64)
    np.maximum.at(last, face, visit)
    face_cells[1] = np.where(last != first[order], last // nfpc, -1)
    rows = keys[first[order]]
    if all(len(lf) == width for lf in local_faces):
        face_list = list(map(tuple, rows.tolist()))
    else:
        face_list = [tuple(v for v in r if v >= 0) for r in rows.tolist()]
    return cell_faces, face_cells, face_list


def _build_cell_neighbors(face_cells: np.ndarray, cell_faces: np.ndarray,
                          n_cells: int, n_inner_cells: int):
    """Build cell_neighbors from face_cells (vectorised).

    Boundary cells get sentinel (n_cells+1) instead of ghost indices,
    so the LSQ stencil grows inward from interior data only.
    Ghost cells are NOT included as neighbors.
    """
    sentinel = n_cells + 1
    n_faces_per_cell = cell_faces.shape[0]
    cell_neighbors = sentinel * np.ones((n_cells, n_faces_per_cell), dtype=int)
    ic = np.arange(n_inner_cells)
    f = cell_faces[:, :n_inner_cells]
    c0, c1 = face_cells[0][f], face_cells[1][f]
    neighbor = np.where(c0 == ic[None, :], c1, c0)
    neighbor = np.where(neighbor >= n_inner_cells, sentinel, neighbor)
    cell_neighbors[:n_inner_cells] = neighbor.T
    return cell_neighbors


def _identify_boundary_faces(face_cells: np.ndarray, n_inner_cells: int,
                             mesh_type: str, cell_vertices: np.ndarray,
                             vertex_coordinates: np.ndarray,
                             meshio_mesh):
    """Identify boundary faces and assign ghost cells.

    Returns boundary arrays and updated face_cells with ghost cell indices.
    """
    boundary_mask = face_cells[1, :] == -1
    boundary_face_indices = np.where(boundary_mask)[0]
    n_boundary_faces = len(boundary_face_indices)

    # Ghost cells: one per boundary face, starting at n_inner_cells
    ghost_start = n_inner_cells
    n_ghost = n_boundary_faces
    n_cells = n_inner_cells + n_ghost

    boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
    boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
    boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
    boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
    boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

    # Build physical tag lookup from meshio
    boundary_dict = {name: data[0] for name, data in meshio_mesh.field_data.items()}
    # Reverse: tag_id → name
    tag_to_name = {v: k for k, v in boundary_dict.items()}

    # Build a vertex-set → physical tag map from meshio boundary cells
    dim = _DIM_OF_TYPE[mesh_type]
    boundary_element_types = {
        1: "line",    # 1D boundaries are vertices (point), but gmsh uses "vertex"
        2: "line",    # 2D boundaries are edges (lines)
        3: "triangle",  # 3D boundaries are faces (triangles or quads)
    }

    # Collect boundary elements and their physical tags from meshio
    face_verts_to_tag: dict[tuple, int] = {}
    for cb, cell_data_block in zip(meshio_mesh.cells, meshio_mesh.cell_data.get("gmsh:physical", [])):
        cell_type = cb.type
        # Only boundary elements (lower-dimensional)
        if cell_type in _MESHIO_TO_ZOOMY and _DIM_OF_TYPE.get(_MESHIO_TO_ZOOMY.get(cell_type, ""), 0) >= dim:
            continue
        for i_elem, elem in enumerate(cb.data):
            sorted_verts = tuple(sorted(elem))
            tag = cell_data_block[i_elem]
            face_verts_to_tag[sorted_verts] = tag

    # Assign ghost cells and tags to boundary faces
    new_face_cells = face_cells.copy()
    for i_bf, fidx in enumerate(boundary_face_indices):
        inner_cell = face_cells[0, fidx]
        ghost_idx = ghost_start + i_bf

        boundary_face_cells[i_bf] = inner_cell
        boundary_face_ghosts[i_bf] = ghost_idx
        boundary_face_face_indices[i_bf] = fidx
        new_face_cells[1, fidx] = ghost_idx

        # Look up physical tag from face vertices
        # Reconstruct face vertices from cell_faces
        # We need the actual face vertex set for this face
        # Get it from the face topology we built
        pass  # will be filled below

    return (n_cells, n_ghost, new_face_cells,
            boundary_face_cells, boundary_face_ghosts,
            boundary_face_face_indices, boundary_face_physical_tags,
            boundary_face_function_numbers,
            face_verts_to_tag, boundary_face_indices, boundary_dict)


# ── BaseMesh ────────────────────────────────────────────────────────────────

class BaseMesh(param.Parameterized):
    """Topology-only mesh.  Geometry computed on the fly."""

    # Metadata
    dimension = param.Integer(default=2)
    type = param.String(default="triangle")
    n_cells = param.Integer(default=0)
    n_inner_cells = param.Integer(default=0)
    n_faces = param.Integer(default=0)
    n_vertices = param.Integer(default=0)
    n_boundary_faces = param.Integer(default=0)
    n_faces_per_cell = param.Integer(default=0)

    # Topology
    vertex_coordinates = param.Array(default=np.empty((0, 0)))
    cell_vertices = param.Array(default=np.empty((0, 0), dtype=int))
    cell_faces = param.Array(default=np.empty((0, 0), dtype=int))
    face_cells = param.Array(default=np.empty((0, 0), dtype=int))
    cell_neighbors = param.Array(default=np.empty((0, 0), dtype=int))

    # Boundary
    boundary_face_cells = param.Array(default=np.empty(0, dtype=int))
    boundary_face_ghosts = param.Array(default=np.empty(0, dtype=int))
    boundary_face_function_numbers = param.Array(default=np.empty(0, dtype=int))
    boundary_face_physical_tags = param.Array(default=np.empty(0, dtype=int))
    boundary_face_face_indices = param.Array(default=np.empty(0, dtype=int))
    boundary_conditions_sorted_physical_tags = param.Array(default=np.empty(0, dtype=int))
    boundary_conditions_sorted_names = param.List(default=[])

    # Optional
    z_ordering = param.Array(default=np.array([-1]))

    # Mesh-carried field data (gmsh $NodeData / $ElementData), keyed by field
    # name.  ``node_data[field]`` has shape ``(n_vertices, ...)`` and is aligned
    # to ``vertex_coordinates`` columns; ``cell_data[field]`` has shape
    # ``(n_inner_cells, ...)``.  Populated by ``from_msh``; consumed by
    # ``zoomy_core.mesh.mesh_ic.initial_conditions_from_mesh``.
    node_data = param.Dict(default={})
    cell_data = param.Dict(default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Guarantee: face_cells[0] is always the inner cell at boundary faces.
        # Some factory methods (e.g. create_1d) may produce the ghost at
        # face_cells[0]; this normalizes the ordering so downstream code
        # can assume face_cells[0] = inner, face_cells[1] = ghost.
        if self.n_boundary_faces > 0 and self.face_cells.size > 0:
            self._normalize_boundary_face_order()

    def _normalize_boundary_face_order(self):
        """Ensure face_cells[0] is the inner cell at every boundary face.

        If face_cells[0] >= n_inner_cells (ghost on wrong side), swap both
        the cell indices and negate the face normal direction (stored as a
        flag for the 1D case where normals are computed from convention).
        """
        fc = self.face_cells
        nc = self.n_inner_cells
        for i_bf in range(self.n_boundary_faces):
            fidx = self.boundary_face_face_indices[i_bf]
            if fc[0, fidx] >= nc:
                fc[0, fidx], fc[1, fidx] = fc[1, fidx], fc[0, fidx]

    # ── on-the-fly geometry ─────────────────────────────────────────────

    def cell_centers_computed(self) -> np.ndarray:
        """Compute cell centers from vertex coordinates. Shape (3, n_cells)."""
        dim = self.dimension
        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        verts_per_cell = self.cell_vertices.shape[0]

        centers = np.zeros((3, n_cells), dtype=float)
        for ic in range(n_inner):
            vert_ids = self.cell_vertices[:, ic]
            coords = self.vertex_coordinates[:dim, vert_ids]  # (dim, verts_per_cell)
            centers[:dim, ic] = coords.mean(axis=1)

        # Ghost cell centers: reflect inner cell across boundary face
        all_face_centers = self.face_centers_computed()
        all_face_normals = self.face_normals_computed() if dim > 1 else None

        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            fidx = self.boundary_face_face_indices[i_bf]
            fc = all_face_centers[fidx, :dim]
            inner_center = centers[:dim, inner]

            if dim == 1:
                centers[:dim, ghost] = 2 * fc - inner_center
            else:
                fn = all_face_normals[:dim, fidx]
                fn = fn / np.linalg.norm(fn)
                offset = fc - inner_center
                centers[:dim, ghost] = inner_center + 2 * np.dot(offset, fn) * fn

        return centers

    def cell_volumes_computed(self) -> np.ndarray:
        """Compute cell volumes. Shape (n_cells,)."""
        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        dim = self.dimension
        volumes = np.ones(n_cells, dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(n_inner):
            vert_ids = list(self.cell_vertices[:, ic])
            volumes[ic] = mesh_util.volume(coords_3d, vert_ids, self.type)

        # Ghost cell volumes = same as their inner cell
        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            volumes[ghost] = volumes[inner]

        return volumes

    def cell_inradius_computed(self) -> np.ndarray:
        """Compute cell inradius (min distance from center to face plane).

        Uses the same projection-based approach as PETSc's
        ``computeCellGeometryFVM`` for consistency.
        """
        from zoomy_core.mesh.lsq_reconstruction import compute_inradius_generic as _compute_inradius_generic

        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        dim = self.dimension
        inradii = np.zeros(n_cells, dtype=float)

        if dim == 1:
            # 1D: inradius = half cell length
            vols = self.cell_volumes_computed()
            inradii[:] = vols / 2.0
        else:
            centers = self.cell_centers_computed()
            face_ctrs = self.face_centers_computed()
            face_norms = self.face_normals_computed()

            for ic in range(n_inner):
                fc_list = []
                fn_list = []
                for lf_idx in range(self.n_faces_per_cell):
                    fidx = self.cell_faces[lf_idx, ic]
                    fc_list.append(face_ctrs[fidx, :dim])
                    fn_list.append(face_norms[:dim, fidx])
                inradii[ic] = _compute_inradius_generic(
                    centers[:dim, ic], fc_list, fn_list
                )

        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            inradii[ghost] = inradii[inner]

        return inradii

    def face_normals_computed(self) -> np.ndarray:
        """Compute face normals. Shape (3, n_faces).

        Normal always points from ``face_cells[0]`` toward ``face_cells[1]``.
        Works for any dimension.
        """
        dim = self.dimension
        n_faces = self.n_faces
        normals = np.zeros((3, n_faces), dtype=float)

        if dim == 1:
            # In 1D a "face normal" is just a sign: +1 or -1.
            # Direction = from face_cells[0] toward face_cells[1].
            # 1D ghost-cell centers don't need face normals
            # (cell_centers_computed line 309–310 just reflects in 1D),
            # so this call is recursion-safe.
            centers = self.cell_centers_computed()
            for f in range(n_faces):
                c0, c1 = self.face_cells[0, f], self.face_cells[1, f]
                normals[0, f] = np.sign(centers[0, c1] - centers[0, c0])
            return normals
        # 2D+: orientation only needs inner cell centers, computed
        # inline below (line 407). Calling cell_centers_computed() here
        # would recurse: it needs face normals to place ghost cells.

        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(self.n_inner_cells):
            vert_ids = list(self.cell_vertices[:, ic])
            cell_normals = mesh_util.face_normals(coords_3d, vert_ids, self.type)
            cell_center = coords_3d[vert_ids].mean(axis=0)

            for lf in range(self.n_faces_per_cell):
                fidx = self.cell_faces[lf, ic]
                if np.linalg.norm(normals[:, fidx]) > 0:
                    continue  # already set by another cell

                n = cell_normals[lf]
                # Orient: normal should point from face_cells[0] to face_cells[1]
                c0 = self.face_cells[0, fidx]
                if c0 == ic:
                    face_verts = self._face_vertex_ids(fidx, ic, lf)
                    face_center = coords_3d[face_verts].mean(axis=0)
                    if np.dot(n, face_center - cell_center) < 0:
                        n = -n
                normals[:, fidx] = n

        return normals

    def face_volumes_computed(self) -> np.ndarray:
        """Compute face areas/lengths. Shape (n_faces,).

        For 1D meshes, face 'volumes' are 1.0 (point has no area).
        """
        dim = self.dimension
        n_faces = self.n_faces

        if dim == 1:
            return np.ones(n_faces, dtype=float)

        fvols = np.zeros(n_faces, dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(self.n_inner_cells):
            vert_ids = list(self.cell_vertices[:, ic])
            face_areas_arr = mesh_util.face_areas(coords_3d, vert_ids, self.type)
            for lf in range(self.n_faces_per_cell):
                fidx = self.cell_faces[lf, ic]
                if fvols[fidx] == 0:
                    fvols[fidx] = face_areas_arr[lf]

        return fvols

    def face_centers_computed(self) -> np.ndarray:
        """Compute face centers. Shape (n_faces, 3)."""
        dim = self.dimension
        n_faces = self.n_faces
        fcenters = np.zeros((n_faces, 3), dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T
        local_faces = _local_faces_for_type(self.type)
        done = np.zeros(n_faces, dtype=bool)

        for ic in range(self.n_inner_cells):
            verts = self.cell_vertices[:, ic]
            for lf_idx, lf in enumerate(local_faces):
                fidx = self.cell_faces[lf_idx, ic]
                if done[fidx]:
                    continue
                face_vert_ids = verts[list(lf)]
                fcenters[fidx] = coords_3d[face_vert_ids].mean(axis=0)
                done[fidx] = True

        return fcenters

    def face_subvolumes_computed(self) -> np.ndarray:
        """Compute face subvolumes (vectorised): per face and side, the
        simplex (2-D) / pyramid (3-D) between the face and the cell centre.
        Shape (n_faces, 2); a boundary face carries its inner side twice."""
        dim = self.dimension
        n_faces = self.n_faces
        subvols = np.zeros((n_faces, 2), dtype=float)
        fcells = np.asarray(self.face_cells)

        if dim == 1:
            # 1D: subvolume = half cell length on each side
            vols = self.cell_volumes_computed()
            return np.stack([vols[fcells[0]] / 2, vols[fcells[1]] / 2], axis=1)

        coords = np.asarray(self.vertex_coordinates, dtype=float)[:dim]
        centers = self.cell_centers_computed()
        nc = int(self.n_inner_cells)
        cv = np.asarray(self.cell_vertices)[:, :nc]
        cf = np.asarray(self.cell_faces)[:, :nc]
        cc = centers[:dim, :nc]
        ic = np.arange(nc)

        def cross(a, b):
            return np.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                             a[0] * b[1] - a[1] * b[0]])

        def tet(v0, v1, v2):
            return np.abs((cross(v1 - v0, v2 - v0) * (cc - v0)).sum(axis=0)) / 6.0

        for lf_idx, lf in enumerate(_local_faces_for_type(self.type)):
            V = [coords[:, cv[k]] for k in lf]                # (dim, nc) each
            if dim == 2:
                a, b = V[1] - V[0], cc - V[0]
                sv = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
            elif len(lf) == 3:
                sv = tet(V[0], V[1], V[2])
            elif len(lf) == 4:                                  # two tetrahedra
                sv = tet(V[0], V[1], V[2]) + tet(V[0], V[2], V[3])
            else:
                raise ValueError(f"face with {len(lf)} vertices")
            f = cf[lf_idx]
            side = np.where(fcells[0, f] == ic, 0, 1)
            subvols[f, side] = sv

        # For boundary ghost cells, copy inner cell subvolume
        n_bf = int(self.n_boundary_faces)
        f = np.asarray(self.boundary_face_face_indices)[:n_bf]
        inner = np.asarray(self.boundary_face_cells)[:n_bf]
        first = inner == fcells[0, f]
        subvols[f[first], 1] = subvols[f[first], 0]
        subvols[f[~first], 0] = subvols[f[~first], 1]
        return subvols

    # ── internal helpers ────────────────────────────────────────────────

    def _face_center_single(self, fidx: int) -> np.ndarray:
        """Compute center of a single face."""
        dim = self.dimension
        local_faces = _local_faces_for_type(self.type)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        # Find a cell that has this face
        for ic in range(self.n_inner_cells):
            for lf_idx, lf in enumerate(local_faces):
                if self.cell_faces[lf_idx, ic] == fidx:
                    verts = self.cell_vertices[:, ic]
                    face_vert_ids = verts[list(lf)]
                    return coords_3d[face_vert_ids].mean(axis=0)[:dim]
        raise ValueError(f"Face {fidx} not found in any cell")

    def _face_normal_single(self, fidx: int) -> np.ndarray:
        """Compute normal of a single face (unit, dim-dimensional)."""
        dim = self.dimension
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T
        local_faces = _local_faces_for_type(self.type)

        for ic in range(self.n_inner_cells):
            for lf_idx, lf in enumerate(local_faces):
                if self.cell_faces[lf_idx, ic] == fidx:
                    verts = self.cell_vertices[:, ic]
                    vert_ids = list(verts)
                    all_normals = mesh_util.face_normals(coords_3d, vert_ids, self.type)
                    n = all_normals[lf_idx, :dim]
                    n /= np.linalg.norm(n)
                    return n
        raise ValueError(f"Face {fidx} not found")

    def _face_vertex_ids(self, fidx: int, ic: int, lf_idx: int) -> list:
        """Get global vertex IDs for a face."""
        local_faces = _local_faces_for_type(self.type)
        verts = self.cell_vertices[:, ic]
        return list(verts[list(local_faces[lf_idx])])

    # ── gradient computation (on-the-fly LSQ) ───────────────────────────

    def compute_derivatives(self, u: np.ndarray, degree: int = 1,
                            derivatives_multi_index=None, *,
                            u_boundary_face) -> np.ndarray:
        """Compute derivatives using on-the-fly LSQ reconstruction."""
        from zoomy_core.mesh.lsq_reconstruction import (
            least_squares_reconstruction_local,
            scale_lsq_derivative,
            find_derivative_indices,
            compute_derivatives as _compute_derivs,
        )
        centers = self.cell_centers_computed()       # (dim_pad, n_cells)
        face_centers = self.face_centers_computed()  # (n_faces, dim_pad)
        dim = self.dimension

        bdy_face_centers = face_centers[
            self.boundary_face_face_indices, :dim
        ]
        cell_boundary_faces = [[] for _ in range(self.n_cells)]
        for i_bf, inner_cell in enumerate(self.boundary_face_cells):
            cell_boundary_faces[int(inner_cell)].append(i_bf)

        (lsq_gradQ,
         lsq_neighbors,
         lsq_boundary_face_neighbors,
         lsq_monomial_multi_index) = (
            least_squares_reconstruction_local(
                self.n_cells, dim, self.cell_neighbors,
                centers[:dim, :].T, degree,
                n_inner_cells=self.n_inner_cells,
                boundary_face_centers=bdy_face_centers,
                cell_boundary_faces=cell_boundary_faces,
                cell_vertices=self.cell_vertices,
            )
        )
        lsq_scale_factors = scale_lsq_derivative(lsq_monomial_multi_index)

        # Build a lightweight namespace that looks like a mesh to _compute_derivs
        class _Stencil:
            pass
        s = _Stencil()
        s.lsq_gradQ = lsq_gradQ
        s.lsq_neighbors = lsq_neighbors
        s.lsq_boundary_face_neighbors = lsq_boundary_face_neighbors
        s.lsq_monomial_multi_index = lsq_monomial_multi_index
        s.lsq_scale_factors = lsq_scale_factors
        s.boundary_face_cells = self.boundary_face_cells
        return _compute_derivs(u, s, derivatives_multi_index,
                               u_boundary_face=u_boundary_face)

    # ── I/O ─────────────────────────────────────────────────────────────

    @classmethod
    def from_msh(cls, filepath: str) -> "BaseMesh":
        """Load a .msh file via meshio and build topology (no PETSc)."""
        if not _HAVE_MESHIO:
            raise RuntimeError("BaseMesh.from_msh() requires meshio")

        mio = meshio.read(filepath)
        boundary_dict = {name: data[0] for name, data in mio.field_data.items()}
        tag_to_name = {v: k for k, v in boundary_dict.items()}

        # Find the primary (highest-dim) cell block
        dim = 0
        primary_block = None
        primary_type = None
        primary_block_index = -1
        for i_block, cb in enumerate(mio.cells):
            mtype = _MESHIO_TO_ZOOMY.get(cb.type)
            if mtype and _DIM_OF_TYPE.get(mtype, 0) > dim:
                dim = _DIM_OF_TYPE[mtype]
                primary_block = cb
                primary_type = mtype
                primary_block_index = i_block

        if primary_block is None:
            raise ValueError("No supported cell type found in mesh")

        # gmsh $NodeData / $ElementData carried in the mesh (aligned to the
        # vertex / primary-cell ordering).  Kept so a backend-agnostic reader
        # can lift them into an initial condition (see mesh_ic).
        node_field_data = {k: np.asarray(v) for k, v in mio.point_data.items()}
        cell_field_data = {
            k: np.asarray(v[primary_block_index])
            for k, v in mio.cell_data.items()
            if k != "gmsh:physical" and primary_block_index < len(v)
        }

        # face_verts → physical tag from meshio boundary elements
        face_verts_to_tag: dict[tuple, int] = {}
        phys_data = mio.cell_data.get("gmsh:physical", [])
        for i_block, cb in enumerate(mio.cells):
            mtype = _MESHIO_TO_ZOOMY.get(cb.type)
            if mtype is None or _DIM_OF_TYPE.get(mtype, 0) >= dim:
                continue
            if i_block < len(phys_data):
                tags = phys_data[i_block]
                for i_elem, elem in enumerate(cb.data):
                    key = tuple(sorted(elem))
                    face_verts_to_tag[key] = tags[i_elem]

        return cls.from_arrays(
            mio.points[:, :dim].T, primary_block.data.T, primary_type,
            face_tags=face_verts_to_tag, tag_names=tag_to_name,
            node_data=node_field_data, cell_data=cell_field_data)

    @classmethod
    def from_arrays(cls, vertex_coordinates, cell_vertices, mesh_type, *,
                    face_tags=None, cell_face_tags=None, tag_names=None,
                    default_tag=0, node_data=None,
                    cell_data=None) -> "BaseMesh":
        """A mesh from raw arrays: ``vertex_coordinates`` ``(dim, n_vertices)``,
        ``cell_vertices`` ``(verts_per_cell, n_cells)`` of one cell type.
        Boundary tags: ``face_tags`` maps a boundary face (its sorted vertex
        tuple) to a physical tag, or ``cell_face_tags``
        ``(faces_per_cell, n_cells)`` gives the tag of each cell's local face
        (< 0: none); ``default_tag`` where a boundary face has none;
        ``tag_names`` names the tags (default ``tag_<t>``)."""
        face_verts_to_tag = face_tags or {}
        tag_to_name = tag_names or {}
        dim = _DIM_OF_TYPE[mesh_type]
        vertex_coordinates = np.asarray(vertex_coordinates)[:dim]
        cell_vertices_arr = np.asarray(cell_vertices)
        n_faces_per_cell = _FACES_PER_CELL[mesh_type]
        n_inner_cells = cell_vertices_arr.shape[1]
        n_vertices = vertex_coordinates.shape[1]

        # Build face topology
        cell_faces_arr, face_cells_arr, face_list = _build_face_topology(
            cell_vertices_arr, mesh_type
        )
        n_faces = len(face_list)

        # Identify boundary faces (face_cells[1] == -1)
        boundary_mask = face_cells_arr[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_ghost = n_boundary_faces
        n_cells = n_inner_cells + n_ghost

        # Assign ghost cells
        boundary_face_cells_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags_arr = np.zeros(n_boundary_faces, dtype=int)
        boundary_face_function_numbers_arr = np.zeros(n_boundary_faces, dtype=int)

        # Assign ghosts and tags
        updated_face_cells = face_cells_arr.copy()
        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            boundary_face_cells_arr[i_bf] = face_cells_arr[0, fidx]
            boundary_face_ghosts_arr[i_bf] = ghost_idx
            boundary_face_face_indices_arr[i_bf] = fidx
            updated_face_cells[1, fidx] = ghost_idx

            if cell_face_tags is None:          # look up the physical tag
                tag = face_verts_to_tag.get(face_list[fidx], default_tag)
                boundary_face_physical_tags_arr[i_bf] = tag

        if cell_face_tags is not None and n_boundary_faces:
            c = face_cells_arr[0, boundary_face_indices]
            lf = np.argmax(cell_faces_arr[:, c] == boundary_face_indices[None], axis=0)
            tags = np.asarray(cell_face_tags)[lf, c]
            boundary_face_physical_tags_arr[:] = np.where(tags >= 0, tags, default_tag)

        # Build sorted tags and names
        unique_tags = sorted(set(boundary_face_physical_tags_arr))
        tag_to_func_num = {t: i for i, t in enumerate(unique_tags)}
        for i_bf in range(n_boundary_faces):
            t = boundary_face_physical_tags_arr[i_bf]
            boundary_face_function_numbers_arr[i_bf] = tag_to_func_num[t]

        sorted_tags = np.array(unique_tags, dtype=int)
        sorted_names = [tag_to_name.get(t, f"tag_{t}") for t in unique_tags]

        # Build cell neighbors
        cell_neighbors_arr = _build_cell_neighbors(
            updated_face_cells, cell_faces_arr, n_cells, n_inner_cells
        )
        # Ghost cell neighbors: copy from their inner cell's neighbors
        for i_bf in range(n_boundary_faces):
            inner = boundary_face_cells_arr[i_bf]
            ghost = boundary_face_ghosts_arr[i_bf]
            cell_neighbors_arr[ghost, :] = cell_neighbors_arr[inner, :]

        return cls(
            dimension=dim,
            type=mesh_type,
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=n_faces_per_cell,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices_arr,
            cell_faces=cell_faces_arr,
            face_cells=updated_face_cells,
            cell_neighbors=cell_neighbors_arr,
            boundary_face_cells=boundary_face_cells_arr,
            boundary_face_ghosts=boundary_face_ghosts_arr,
            boundary_face_function_numbers=boundary_face_function_numbers_arr,
            boundary_face_physical_tags=boundary_face_physical_tags_arr,
            boundary_face_face_indices=boundary_face_face_indices_arr,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
            node_data=node_data or {},
            cell_data=cell_data or {},
        )

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "BaseMesh":
        """Build a uniform 1D interval mesh."""
        xL, xR = domain
        n_cells = n_inner_cells + 2
        n_vertices = n_inner_cells + 1
        n_faces = n_inner_cells + 1
        n_boundary_faces = 2
        n_faces_per_cell = 2
        dx = (xR - xL) / n_inner_cells

        vertex_coordinates = np.linspace(xL, xR, n_vertices, dtype=float).reshape(1, -1)

        cell_vertices = np.zeros((2, n_inner_cells), dtype=int)
        cell_vertices[0, :] = np.arange(n_vertices - 1)
        cell_vertices[1, :] = np.arange(1, n_vertices)

        cell_faces = np.zeros((2, n_inner_cells), dtype=int)
        cell_faces[0, :] = np.arange(n_faces - 1)
        cell_faces[1, :] = np.arange(1, n_faces)

        face_cells = np.empty((2, n_faces), dtype=int)
        face_cells[0, 1:-1] = np.arange(n_inner_cells - 1)
        face_cells[1, 1:-1] = np.arange(1, n_inner_cells)
        face_cells[0, 0] = n_inner_cells      # ghost left
        face_cells[1, 0] = 0
        face_cells[0, -1] = n_inner_cells - 1
        face_cells[1, -1] = n_inner_cells + 1  # ghost right

        sentinel = n_cells + 1
        cell_neighbors = sentinel * np.ones((n_cells, 2), dtype=int)
        for ic in range(n_inner_cells):
            cell_neighbors[ic, :] = [ic - 1, ic + 1]
        # Boundary cells: sentinel instead of ghost
        cell_neighbors[0, 0] = sentinel
        cell_neighbors[n_inner_cells - 1, 1] = sentinel
        # Ghost cell neighbors (kept for now, not used by LSQ)
        cell_neighbors[n_inner_cells, :] = [1, 0]
        cell_neighbors[n_inner_cells + 1, :] = [n_inner_cells - 1, n_inner_cells - 2]

        boundary_face_cells = np.array([0, n_inner_cells - 1], dtype=int)
        boundary_face_ghosts = np.array([n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.array([0, 1], dtype=int)
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)

        return cls(
            dimension=1,
            type="line",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=n_faces_per_cell,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=np.array([0, 1], dtype=int),
            boundary_conditions_sorted_names=["left", "right"],
        )

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "BaseMesh":
        """Build a uniform 2D quad mesh.

        Parameters
        ----------
        domain : (x_min, x_max, y_min, y_max)
        nx, ny : number of inner cells in x and y directions
        """
        x_min, x_max, y_min, y_max = domain
        n_inner_cells = nx * ny

        # ── vertices on a regular grid ──
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")  # (nx+1, ny+1)
        n_vertices = (nx + 1) * (ny + 1)
        # vertex_coordinates: (3, n_vertices), row-major over (ix, iy)
        vertex_coordinates = np.zeros((3, n_vertices), dtype=float)
        vertex_coordinates[0] = gx.ravel()
        vertex_coordinates[1] = gy.ravel()

        def vid(ix, iy):
            return ix * (ny + 1) + iy

        # ── cell-vertex connectivity: (4, n_inner_cells) ──
        ix, iy = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny),
                                                  indexing="ij"))  # ic = ix*ny + iy
        cell_vertices = np.stack([vid(ix, iy), vid(ix + 1, iy),
                                  vid(ix + 1, iy + 1), vid(ix, iy + 1)])

        # ── build face topology via _build_face_topology ──
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_vertices, "quad"
        )
        n_faces = len(face_list)

        # ── identify boundary faces and assign ghost cells ──
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells = n_inner_cells + n_boundary_faces

        face_cells = face_cells_raw.copy()
        boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
        boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

        # Classify boundary faces by position of face center
        tag_map = {"left": 0, "right": 1, "bottom": 2, "top": 3}
        tol = 1e-12 * max(x_max - x_min, y_max - y_min)

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = face_cells_raw[0, fidx]

            boundary_face_cells[i_bf] = inner_cell
            boundary_face_ghosts[i_bf] = ghost_idx
            boundary_face_face_indices[i_bf] = fidx
            face_cells[1, fidx] = ghost_idx

            # Compute face center from vertices
            fverts = face_list[fidx]
            fc = vertex_coordinates[:2, list(fverts)].mean(axis=1)

            if abs(fc[0] - x_min) < tol:
                tag = tag_map["left"]
            elif abs(fc[0] - x_max) < tol:
                tag = tag_map["right"]
            elif abs(fc[1] - y_min) < tol:
                tag = tag_map["bottom"]
            elif abs(fc[1] - y_max) < tol:
                tag = tag_map["top"]
            else:
                tag = 0  # fallback

            boundary_face_physical_tags[i_bf] = tag
            boundary_face_function_numbers[i_bf] = tag

        # ── cell neighbors ──
        cell_neighbors = _build_cell_neighbors(
            face_cells, cell_faces, n_cells, n_inner_cells
        )

        # Ghost cell neighbors: each ghost mirrors its inner cell's stencil
        for i_bf in range(n_boundary_faces):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = boundary_face_cells[i_bf]
            # Ghost cell's neighbors = inner cell's neighbors
            cell_neighbors[ghost_idx, :] = cell_neighbors[inner_cell, :]

        sorted_tags = np.array([0, 1, 2, 3], dtype=int)
        sorted_names = ["left", "right", "bottom", "top"]

        return cls(
            dimension=2,
            type="quad",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=4,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "BaseMesh":
        """Build a uniform 3D hexahedral mesh.

        Parameters
        ----------
        domain : (x_min, x_max, y_min, y_max, z_min, z_max)
        nx, ny, nz : number of inner cells in each direction
        """
        x_min, x_max, y_min, y_max, z_min, z_max = domain
        n_inner_cells = nx * ny * nz

        # ── vertices on a regular grid ──
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)
        zs = np.linspace(z_min, z_max, nz + 1)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        n_vertices = (nx + 1) * (ny + 1) * (nz + 1)
        vertex_coordinates = np.zeros((3, n_vertices), dtype=float)
        vertex_coordinates[0] = gx.ravel()
        vertex_coordinates[1] = gy.ravel()
        vertex_coordinates[2] = gz.ravel()

        def vid(ix, iy, iz):
            return ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz

        # ── cell-vertex connectivity: (8, n_inner_cells) ──
        # Hexahedron vertex ordering (VTK convention):
        #   bottom face: 0,1,2,3   top face: 4,5,6,7
        #   0=(ix,iy,iz), 1=(ix+1,iy,iz), 2=(ix+1,iy+1,iz), 3=(ix,iy+1,iz)
        #   4=(ix,iy,iz+1), 5=(ix+1,iy,iz+1), 6=(ix+1,iy+1,iz+1), 7=(ix,iy+1,iz+1)
        cell_vertices = np.empty((8, n_inner_cells), dtype=int)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    ic = ix * ny * nz + iy * nz + iz
                    cell_vertices[0, ic] = vid(ix, iy, iz)
                    cell_vertices[1, ic] = vid(ix + 1, iy, iz)
                    cell_vertices[2, ic] = vid(ix + 1, iy + 1, iz)
                    cell_vertices[3, ic] = vid(ix, iy + 1, iz)
                    cell_vertices[4, ic] = vid(ix, iy, iz + 1)
                    cell_vertices[5, ic] = vid(ix + 1, iy, iz + 1)
                    cell_vertices[6, ic] = vid(ix + 1, iy + 1, iz + 1)
                    cell_vertices[7, ic] = vid(ix, iy + 1, iz + 1)

        # ── build face topology ──
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_vertices, "hexahedron"
        )
        n_faces = len(face_list)

        # ── boundary faces + ghost cells ──
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells = n_inner_cells + n_boundary_faces

        face_cells = face_cells_raw.copy()
        boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
        boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

        tag_map = {"left": 0, "right": 1, "front": 2, "back": 3,
                   "bottom": 4, "top": 5}
        tol = 1e-12 * max(x_max - x_min, y_max - y_min, z_max - z_min)

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = face_cells_raw[0, fidx]

            boundary_face_cells[i_bf] = inner_cell
            boundary_face_ghosts[i_bf] = ghost_idx
            boundary_face_face_indices[i_bf] = fidx
            face_cells[1, fidx] = ghost_idx

            fverts = face_list[fidx]
            fc = vertex_coordinates[:, list(fverts)].mean(axis=1)

            if abs(fc[0] - x_min) < tol:
                tag = tag_map["left"]
            elif abs(fc[0] - x_max) < tol:
                tag = tag_map["right"]
            elif abs(fc[1] - y_min) < tol:
                tag = tag_map["front"]
            elif abs(fc[1] - y_max) < tol:
                tag = tag_map["back"]
            elif abs(fc[2] - z_min) < tol:
                tag = tag_map["bottom"]
            elif abs(fc[2] - z_max) < tol:
                tag = tag_map["top"]
            else:
                tag = 0

            boundary_face_physical_tags[i_bf] = tag
            boundary_face_function_numbers[i_bf] = tag

        # ── cell neighbors ──
        cell_neighbors = _build_cell_neighbors(
            face_cells, cell_faces, n_cells, n_inner_cells
        )
        for i_bf in range(n_boundary_faces):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = boundary_face_cells[i_bf]
            cell_neighbors[ghost_idx, :] = cell_neighbors[inner_cell, :]

        sorted_tags = np.array(list(tag_map.values()), dtype=int)
        sorted_names = list(tag_map.keys())

        return cls(
            dimension=3,
            type="hexahedron",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=6,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    @classmethod
    def extrude_2d(cls, mesh2d, n_layers: int, height: float = 1.0) -> "BaseMesh":
        """Extrude a 2D quad mesh into 3D hexahedra.

        Parameters
        ----------
        mesh2d : BaseMesh
            A 2D mesh (type "quad").
        n_layers : int
            Number of layers in the z-direction.
        height : float
            Total extrusion height.

        Returns
        -------
        BaseMesh
            A 3D hexahedral mesh.
        """
        if mesh2d.dimension != 2:
            raise ValueError(f"extrude_2d requires a 2D mesh, got dimension={mesh2d.dimension}")

        nz = n_layers
        n2d_verts = mesh2d.n_vertices
        n2d_cells = mesh2d.n_inner_cells
        coords_2d = mesh2d.vertex_coordinates  # (3, n2d_verts) or (2, n2d_verts)
        cell_verts_2d = mesh2d.cell_vertices    # (verts_per_cell, n2d_cells)
        vpce_2d = cell_verts_2d.shape[0]        # 4 for quads

        # 3D vertices: duplicate 2D vertices at each z-level
        n3d_verts = n2d_verts * (nz + 1)
        z_levels = np.linspace(0, height, nz + 1)
        coords_3d = np.zeros((3, n3d_verts))
        for iz in range(nz + 1):
            s = iz * n2d_verts
            e = s + n2d_verts
            coords_3d[0, s:e] = coords_2d[0, :] if coords_2d.shape[0] > 0 else 0
            coords_3d[1, s:e] = coords_2d[1, :] if coords_2d.shape[0] > 1 else 0
            coords_3d[2, s:e] = z_levels[iz]

        # 3D cells: each 2D quad → nz hexahedra stacked in z
        n3d_inner = n2d_cells * nz
        cell_verts_3d = np.empty((8, n3d_inner), dtype=int)
        for iz in range(nz):
            off_bot = iz * n2d_verts
            off_top = (iz + 1) * n2d_verts
            for ic in range(n2d_cells):
                idx = iz * n2d_cells + ic
                v = cell_verts_2d[:, ic]
                cell_verts_3d[0, idx] = v[0] + off_bot
                cell_verts_3d[1, idx] = v[1] + off_bot
                cell_verts_3d[2, idx] = v[2] + off_bot
                cell_verts_3d[3, idx] = v[3] + off_bot
                cell_verts_3d[4, idx] = v[0] + off_top
                cell_verts_3d[5, idx] = v[1] + off_top
                cell_verts_3d[6, idx] = v[2] + off_top
                cell_verts_3d[7, idx] = v[3] + off_top

        # Build face topology
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_verts_3d, "hexahedron"
        )
        n_faces = len(face_list)

        # Identify boundary faces and build ghost cells
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells = n3d_inner + n_boundary_faces

        face_cells = face_cells_raw.copy()
        boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
        boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

        # Tag map: lateral tags from 2D mesh + bottom + top
        lateral_tags = list(mesh2d.boundary_conditions_sorted_names)
        tag_names = lateral_tags + ["bottom", "top"]
        tag_bottom = len(lateral_tags)
        tag_top = len(lateral_tags) + 1
        sorted_tags = list(range(len(tag_names)))
        tol = 1e-12 * height

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n3d_inner + i_bf
            inner_cell = face_cells_raw[0, fidx]
            face_cells[1, fidx] = ghost_idx
            boundary_face_cells[i_bf] = inner_cell
            boundary_face_ghosts[i_bf] = ghost_idx
            boundary_face_face_indices[i_bf] = fidx

            # Classify by z-coordinate of face center
            fv = face_list[fidx]
            face_center_z = np.mean(coords_3d[2, fv])
            if face_center_z < tol:
                tag = tag_bottom
            elif face_center_z > height - tol:
                tag = tag_top
            else:
                # Lateral face — match to 2D boundary tag by x,y position
                face_center_xy = np.mean(coords_3d[:2, fv], axis=1)
                tag = 0  # default to first lateral tag
                # Try matching 2D boundary face centers
                for i2d in range(mesh2d.n_boundary_faces):
                    bf_face_idx = mesh2d.boundary_face_face_indices[i2d]
                    ptag = mesh2d.boundary_face_physical_tags[i2d]
                    tag = ptag
                    break  # simplified: use first match
            boundary_face_physical_tags[i_bf] = tag
            boundary_face_function_numbers[i_bf] = tag

        # Cell neighbors
        cell_neighbors = _build_cell_neighbors(cell_faces, face_cells, n_cells)

        return cls(
            dimension=3,
            type="hex",
            n_cells=n_cells,
            n_inner_cells=n3d_inner,
            n_faces=n_faces,
            n_vertices=n3d_verts,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=6,
            vertex_coordinates=coords_3d,
            cell_vertices=cell_verts_3d,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=np.array(sorted_tags, dtype=int),
            boundary_conditions_sorted_names=tag_names,
        )

    def write_to_hdf5(self, filepath: str):
        """Serialize topology-only fields to HDF5."""
        if not _HAVE_H5PY:
            raise RuntimeError("write_to_hdf5 requires h5py")
        with h5py.File(filepath, "w") as f:
            g = f.create_group("mesh")
            g.attrs["format_version"] = 2

            g.create_dataset("dimension", data=self.dimension)
            g.create_dataset("type", data=self.type)
            g.create_dataset("n_cells", data=self.n_cells)
            g.create_dataset("n_inner_cells", data=self.n_inner_cells)
            g.create_dataset("n_faces", data=self.n_faces)
            g.create_dataset("n_vertices", data=self.n_vertices)
            g.create_dataset("n_boundary_faces", data=self.n_boundary_faces)
            g.create_dataset("n_faces_per_cell", data=self.n_faces_per_cell)

            g.create_dataset("vertex_coordinates", data=self.vertex_coordinates)
            g.create_dataset("cell_vertices", data=self.cell_vertices)
            g.create_dataset("cell_faces", data=self.cell_faces)
            g.create_dataset("face_cells", data=self.face_cells)
            g.create_dataset("cell_neighbors", data=self.cell_neighbors)

            g.create_dataset("boundary_face_cells", data=self.boundary_face_cells)
            g.create_dataset("boundary_face_ghosts", data=self.boundary_face_ghosts)
            g.create_dataset("boundary_face_function_numbers", data=self.boundary_face_function_numbers)
            g.create_dataset("boundary_face_physical_tags", data=self.boundary_face_physical_tags)
            g.create_dataset("boundary_face_face_indices", data=self.boundary_face_face_indices)
            g.create_dataset("boundary_conditions_sorted_physical_tags",
                             data=self.boundary_conditions_sorted_physical_tags)
            g.create_dataset("boundary_conditions_sorted_names",
                             data=np.array(self.boundary_conditions_sorted_names, dtype="S"))
            g.create_dataset("z_ordering", data=self.z_ordering)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "BaseMesh":
        """Load a BaseMesh from HDF5."""
        if not _HAVE_H5PY:
            raise RuntimeError("from_hdf5 requires h5py")
        with h5py.File(filepath, "r") as f:
            g = f["mesh"]
            return cls(
                dimension=int(g["dimension"][()]),
                type=g["type"][()].decode("utf-8") if isinstance(g["type"][()], bytes) else str(g["type"][()]),
                n_cells=int(g["n_cells"][()]),
                n_inner_cells=int(g["n_inner_cells"][()]),
                n_faces=int(g["n_faces"][()]),
                n_vertices=int(g["n_vertices"][()]),
                n_boundary_faces=int(g["n_boundary_faces"][()]),
                n_faces_per_cell=int(g["n_faces_per_cell"][()]),
                vertex_coordinates=g["vertex_coordinates"][()],
                cell_vertices=g["cell_vertices"][()],
                cell_faces=g["cell_faces"][()],
                face_cells=g["face_cells"][()],
                cell_neighbors=g["cell_neighbors"][()],
                boundary_face_cells=g["boundary_face_cells"][()],
                boundary_face_ghosts=g["boundary_face_ghosts"][()],
                boundary_face_function_numbers=g["boundary_face_function_numbers"][()],
                boundary_face_physical_tags=g["boundary_face_physical_tags"][()],
                boundary_face_face_indices=g["boundary_face_face_indices"][()],
                boundary_conditions_sorted_physical_tags=g["boundary_conditions_sorted_physical_tags"][()],
                boundary_conditions_sorted_names=list(
                    np.array(g["boundary_conditions_sorted_names"][()], dtype="str")
                ),
                z_ordering=g["z_ordering"][()] if "z_ordering" in g else np.array([-1]),
            )

    @classmethod
    def extrude_2d(cls, mesh_2d: "BaseMesh", n_layers: int,
                   height: float | list | np.ndarray) -> "BaseMesh":
        """Extrude a 2D mesh into 3D by stacking layers in the z-direction.

        Parameters
        ----------
        mesh_2d : BaseMesh
            A 2D mesh (type ``"quad"`` or ``"triangle"``).
        n_layers : int
            Number of layers in the z-direction.
        height : float or array-like
            Total extrusion height (uniform layers), or a list/array of
            ``n_layers`` individual layer heights.

        Returns
        -------
        BaseMesh
            A 3D mesh of type ``"hexahedron"`` (from quads) or
            ``"wface"`` (from triangles).
        """
        if mesh_2d.dimension != 2:
            raise ValueError(
                f"extrude_2d requires a 2D mesh, got dimension={mesh_2d.dimension}"
            )
        if mesh_2d.type not in ("quad", "triangle"):
            raise ValueError(
                f"extrude_2d supports 'quad' or 'triangle' meshes, got '{mesh_2d.type}'"
            )

        from zoomy_core.mesh.mesh_util import get_extruded_mesh_type

        mesh_type_3d = get_extruded_mesh_type(mesh_2d.type)
        n_faces_per_cell_3d = _FACES_PER_CELL[mesh_type_3d]
        vpn_2d = mesh_2d.cell_vertices.shape[0]  # vertices per 2D cell

        # -- z-levels ----------------------------------------------------------
        if np.ndim(height) == 0:
            dz = float(height) / n_layers
            z_levels = np.linspace(0.0, float(height), n_layers + 1)
        else:
            layer_heights = np.asarray(height, dtype=float)
            if len(layer_heights) != n_layers:
                raise ValueError(
                    f"len(height)={len(layer_heights)} but n_layers={n_layers}"
                )
            z_levels = np.concatenate([[0.0], np.cumsum(layer_heights)])

        # -- 3D vertex coordinates (3, N_v3d) ----------------------------------
        N_v = mesh_2d.n_vertices
        N_v3d = (n_layers + 1) * N_v
        vertex_coordinates = np.zeros((3, N_v3d), dtype=float)

        for k in range(n_layers + 1):
            offset = k * N_v
            vertex_coordinates[0, offset:offset + N_v] = mesh_2d.vertex_coordinates[0, :]
            vertex_coordinates[1, offset:offset + N_v] = mesh_2d.vertex_coordinates[1, :]
            vertex_coordinates[2, offset:offset + N_v] = z_levels[k]

        # -- 3D cell-vertex connectivity ---------------------------------------
        N_c2d = mesh_2d.n_inner_cells
        n_inner_cells_3d = n_layers * N_c2d

        if mesh_2d.type == "quad":
            # Hexahedron: 8 vertices per cell
            # Bottom: v0..v3 at layer k, Top: v4..v7 at layer k+1
            cell_vertices = np.empty((8, n_inner_cells_3d), dtype=int)
            for k in range(n_layers):
                for ic2d in range(N_c2d):
                    ic3d = k * N_c2d + ic2d
                    v2d = mesh_2d.cell_vertices[:, ic2d]  # (4,)
                    cell_vertices[0, ic3d] = v2d[0] + k * N_v
                    cell_vertices[1, ic3d] = v2d[1] + k * N_v
                    cell_vertices[2, ic3d] = v2d[2] + k * N_v
                    cell_vertices[3, ic3d] = v2d[3] + k * N_v
                    cell_vertices[4, ic3d] = v2d[0] + (k + 1) * N_v
                    cell_vertices[5, ic3d] = v2d[1] + (k + 1) * N_v
                    cell_vertices[6, ic3d] = v2d[2] + (k + 1) * N_v
                    cell_vertices[7, ic3d] = v2d[3] + (k + 1) * N_v
        else:
            # Wedge (wface): 6 vertices per cell
            # Bottom: v0..v2 at layer k, Top: v3..v5 at layer k+1
            cell_vertices = np.empty((6, n_inner_cells_3d), dtype=int)
            for k in range(n_layers):
                for ic2d in range(N_c2d):
                    ic3d = k * N_c2d + ic2d
                    v2d = mesh_2d.cell_vertices[:, ic2d]  # (3,)
                    cell_vertices[0, ic3d] = v2d[0] + k * N_v
                    cell_vertices[1, ic3d] = v2d[1] + k * N_v
                    cell_vertices[2, ic3d] = v2d[2] + k * N_v
                    cell_vertices[3, ic3d] = v2d[0] + (k + 1) * N_v
                    cell_vertices[4, ic3d] = v2d[1] + (k + 1) * N_v
                    cell_vertices[5, ic3d] = v2d[2] + (k + 1) * N_v

        # -- face topology via _build_face_topology ----------------------------
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_vertices, mesh_type_3d
        )
        n_faces = len(face_list)

        # -- boundary faces + ghost cells --------------------------------------
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells_3d = n_inner_cells_3d + n_boundary_faces

        face_cells = face_cells_raw.copy()
        bf_cells = np.empty(n_boundary_faces, dtype=int)
        bf_ghosts = np.empty(n_boundary_faces, dtype=int)
        bf_face_indices = np.empty(n_boundary_faces, dtype=int)
        bf_physical_tags = np.empty(n_boundary_faces, dtype=int)
        bf_function_numbers = np.empty(n_boundary_faces, dtype=int)

        # Build a tag scheme that extends the 2D boundary tags:
        # - Keep all existing 2D lateral tags (renaming "bottom"->"front",
        #   "top"->"back" when they came from a structured 2D mesh)
        # - Add new "bottom" (z=0) and "top" (z=height) tags
        existing_tags_2d = list(mesh_2d.boundary_conditions_sorted_physical_tags)
        existing_names_2d = list(mesh_2d.boundary_conditions_sorted_names)
        # Rename 2D y-direction boundaries to match 3D convention
        _rename = {"bottom": "front", "top": "back"}
        lateral_names = [_rename.get(n, n) for n in existing_names_2d]
        n_existing = len(existing_tags_2d)

        # New tag values for bottom/top (after existing ones)
        tag_bottom = n_existing
        tag_top = n_existing + 1
        z_min = z_levels[0]
        z_max = z_levels[-1]
        total_span = z_max - z_min if z_max > z_min else 1.0
        tol = 1e-12 * total_span

        # Build lookup: face index -> 2D boundary physical tag for lateral
        # A lateral boundary face has all its vertex z-coords NOT all the same
        # or equivalently its face center z is between z_min and z_max.
        # For lateral faces, project the face vertices to 2D and look up the
        # 2D boundary tag.
        #
        # Strategy: for each 2D boundary face, record the sorted 2D vertex pair.
        # Then for each 3D boundary face, check if it's top/bottom or lateral.
        # For lateral, find which 2D boundary edge it corresponds to.

        # Build 2D boundary edge -> physical tag map
        bf_2d_edge_to_tag = {}
        local_faces_2d = _local_faces_for_type(mesh_2d.type)
        for i_bf in range(mesh_2d.n_boundary_faces):
            fidx_2d = mesh_2d.boundary_face_face_indices[i_bf]
            ptag = mesh_2d.boundary_face_physical_tags[i_bf]
            # Find the 2D face vertex key (the sorted tuple of 2D vertex IDs)
            # by scanning the inner cell that owns this face
            inner_cell = mesh_2d.boundary_face_cells[i_bf]
            verts_2d = mesh_2d.cell_vertices[:, inner_cell]
            for lf_idx, lf in enumerate(local_faces_2d):
                if mesh_2d.cell_faces[lf_idx, inner_cell] == fidx_2d:
                    edge_verts = tuple(sorted(verts_2d[list(lf)]))
                    bf_2d_edge_to_tag[edge_verts] = ptag
                    break

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells_3d + i_bf
            inner_cell = face_cells_raw[0, fidx]

            bf_cells[i_bf] = inner_cell
            bf_ghosts[i_bf] = ghost_idx
            bf_face_indices[i_bf] = fidx
            face_cells[1, fidx] = ghost_idx

            # Classify: compute face center z-coordinate
            fverts = face_list[fidx]
            fc_z = vertex_coordinates[2, list(fverts)].mean()

            if abs(fc_z - z_min) < tol:
                bf_physical_tags[i_bf] = tag_bottom
            elif abs(fc_z - z_max) < tol:
                bf_physical_tags[i_bf] = tag_top
            else:
                # Lateral face: project vertices to 2D and match
                fverts_list = list(fverts)
                verts_2d_mod = sorted(set(v % N_v for v in fverts_list))
                edge_key = tuple(verts_2d_mod)
                tag = bf_2d_edge_to_tag.get(edge_key, 0)
                bf_physical_tags[i_bf] = tag

        # Build sorted tags and names
        all_tags = list(existing_tags_2d) + [tag_bottom, tag_top]
        all_names = lateral_names + ["bottom", "top"]
        sorted_tags = np.array(all_tags, dtype=int)
        sorted_names = all_names

        # Assign function numbers: tag value -> index in sorted_tags
        tag_to_func = {t: i for i, t in enumerate(sorted_tags)}
        for i_bf in range(n_boundary_faces):
            bf_function_numbers[i_bf] = tag_to_func.get(
                bf_physical_tags[i_bf], 0
            )

        # -- cell neighbors ----------------------------------------------------
        cell_neighbors = _build_cell_neighbors(
            face_cells, cell_faces, n_cells_3d, n_inner_cells_3d
        )
        for i_bf in range(n_boundary_faces):
            ghost_idx = n_inner_cells_3d + i_bf
            inner_cell = bf_cells[i_bf]
            cell_neighbors[ghost_idx, :] = cell_neighbors[inner_cell, :]

        return cls(
            dimension=3,
            type=mesh_type_3d,
            n_cells=n_cells_3d,
            n_inner_cells=n_inner_cells_3d,
            n_faces=n_faces,
            n_vertices=N_v3d,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=n_faces_per_cell_3d,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=bf_cells,
            boundary_face_ghosts=bf_ghosts,
            boundary_face_function_numbers=bf_function_numbers,
            boundary_face_physical_tags=bf_physical_tags,
            boundary_face_face_indices=bf_face_indices,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    def resolve_periodic_bcs(self, bcs):
        """Patch boundary_face_cells for periodic boundary conditions.

        Idempotent and stateless: the remap source is derived from
        ``face_cells[0]`` at the boundary faces — the PHYSICAL this-side
        cells, which this method never modifies — rather than from
        ``boundary_face_cells`` itself.  Deriving from the current
        (possibly already-patched) array swaps the remap back to the
        physical cells on every second call, silently alternating the
        boundary between periodic and open (mass pumps in/out).  A
        stashed-on-first-call pristine copy is NOT enough either: per-
        solve mesh copies (ensure_lsq_mesh) share the underlying numpy
        array with the original mesh while each copy starts without the
        stash, re-introducing the alternation across solver restarts.
        """
        from zoomy_core.model.boundary_conditions import Periodic

        dict_physical_name_to_index = {
            v: i for i, v in enumerate(self.boundary_conditions_sorted_names)
        }
        dict_function_index_to_physical_tag = {
            i: v for i, v in enumerate(self.boundary_conditions_sorted_physical_tags)
        }

        pristine_bf_cells = np.asarray(
            self.face_cells[0, self.boundary_face_face_indices])
        boundary_face_cells_copy = deepcopy(pristine_bf_cells)

        for bc in bcs.boundary_conditions_list:
            if type(bc) == Periodic:
                from_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.periodic_to_physical_tag]
                ]
                to_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.tag]
                ]

                mask_face_from = self.boundary_face_physical_tags == from_physical_tag
                mask_face_to = self.boundary_face_physical_tags == to_physical_tag

                from_cells = pristine_bf_cells[mask_face_from]
                to_cells = self.boundary_face_ghosts[mask_face_to]

                centers = self.cell_centers_computed()
                from_coords = centers[:, from_cells]
                to_coords = centers[:, to_cells]

                significance_per_dimension = [
                    from_coords[d, :].max() - from_coords[d, :].min()
                    for d in range(self.dimension)
                ]
                sort_order_significance = np.lexsort([significance_per_dimension])

                from_cells_sort_order = np.lexsort(
                    [from_coords[d, :] for d in sort_order_significance]
                )
                to_cells_sort_order = np.lexsort(
                    [to_coords[d, :] for d in sort_order_significance]
                )

                indices = np.arange(mask_face_to.shape[0])
                indices_to = indices[mask_face_to]
                indices_to_sort = indices_to[to_cells_sort_order]
                indices_from = indices[mask_face_from]
                indices_from_sort = indices_from[from_cells_sort_order]

                self.boundary_face_cells[indices_to_sort] = boundary_face_cells_copy[
                    indices_from_sort
                ]
