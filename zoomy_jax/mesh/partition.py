"""Mesh partitioning for MPI-parallel FVM solves.

Provides graph-based partitioning (via pymetis when available) and
local sub-mesh extraction with ghost-cell bookkeeping.  All routines
operate on the NumPy-level mesh classes (`BaseMesh` / `LSQMesh`)
so that they run *before* the JAX conversion step.  If `pymetis` is
not installed, a simple contiguous splitter is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from zoomy_core.mesh.base_mesh import BaseMesh

try:
    import pymetis

    _HAVE_PYMETIS = True
except ImportError:
    _HAVE_PYMETIS = False


# ---------------------------------------------------------------------------
# Data container returned by partition_mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionInfo:
    """Partition bookkeeping for a single MPI rank.

    Attributes
    ----------
    rank : int
        The MPI rank this partition belongs to.
    owned_cells : np.ndarray[int]
        Global cell indices owned by this rank (inner cells only).
    ghost_cells : np.ndarray[int]
        Global cell indices needed from other ranks for face-stencil
        computations.  These appear *after* owned cells in the local
        numbering.
    local_faces : np.ndarray[int]
        Global face indices where at least one adjacent cell is owned.
    send_map : dict[int, np.ndarray[int]]
        `{neighbor_rank: local cell indices to send}`.  These are
        owned cells that are ghosts on a neighbouring rank.
    recv_map : dict[int, np.ndarray[int]]
        `{neighbor_rank: local indices where received data goes}`.
        Local indices point into the ghost region of the local array
        (i.e. offset by `len(owned_cells)`).
    global_to_local : np.ndarray[int]
        Mapping from global cell index to local cell index.  Only
        entries for `owned_cells` and `ghost_cells` are meaningful;
        all others are set to -1.
    """

    rank: int
    owned_cells: np.ndarray
    ghost_cells: np.ndarray
    local_faces: np.ndarray
    send_map: Dict[int, np.ndarray] = field(default_factory=dict)
    recv_map: Dict[int, np.ndarray] = field(default_factory=dict)
    global_to_local: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


# ---------------------------------------------------------------------------
# Graph building helpers
# ---------------------------------------------------------------------------


def _build_cell_adjacency(face_cells: np.ndarray, n_inner_cells: int):
    """Return per-cell adjacency lists from the face connectivity.

    Parameters
    ----------
    face_cells : ndarray, shape `(2, n_faces)`
        `face_cells[0, f]` and `face_cells[1, f]` are the two cells
        sharing face *f*.
    n_inner_cells : int
        Number of inner (non-ghost) cells.  Only inner-to-inner edges
        are included in the graph.

    Returns
    -------
    adjacency : list[list[int]]
        `adjacency[i]` is the sorted list of inner-cell neighbours of
        cell *i* (length `n_inner_cells`).
    """
    adjacency: List[List[int]] = [[] for _ in range(n_inner_cells)]
    n_faces = face_cells.shape[1]
    for f in range(n_faces):
        a, b = int(face_cells[0, f]), int(face_cells[1, f])
        if a < n_inner_cells and b < n_inner_cells and a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)
    # Deduplicate and sort
    for i in range(n_inner_cells):
        adjacency[i] = sorted(set(adjacency[i]))
    return adjacency


def _adjacency_to_csr(adjacency: List[List[int]]):
    """Convert adjacency lists to CSR (xadj, adjncy) for pymetis."""
    xadj = [0]
    adjncy: List[int] = []
    for nbrs in adjacency:
        adjncy.extend(nbrs)
        xadj.append(len(adjncy))
    return xadj, adjncy


# ---------------------------------------------------------------------------
# Partitioning back-ends
# ---------------------------------------------------------------------------


def _partition_pymetis(adjacency: List[List[int]], n_parts: int) -> np.ndarray:
    """Graph-partition using pymetis.  Returns a cell->part assignment."""
    xadj, adjncy = _adjacency_to_csr(adjacency)
    _cuts, membership = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy)
    return np.asarray(membership, dtype=int)


def _partition_contiguous(n_inner_cells: int, n_parts: int) -> np.ndarray:
    """Trivial contiguous block partition (fallback)."""
    base = n_inner_cells // n_parts
    remainder = n_inner_cells % n_parts
    membership = np.empty(n_inner_cells, dtype=int)
    offset = 0
    for p in range(n_parts):
        size = base + (1 if p < remainder else 0)
        membership[offset: offset + size] = p
        offset += size
    return membership


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def partition_mesh(mesh, n_parts: int) -> List[PartitionInfo]:
    """Partition *mesh* into *n_parts* sub-domains.

    Parameters
    ----------
    mesh : BaseMesh (or FVMMesh / LSQMesh)
        The full (global) Zoomy mesh.  Only topology arrays
        (`face_cells`, `n_inner_cells`, `n_cells`, `n_faces`)
        are used.
    n_parts : int
        Number of partitions (typically `MPI comm size`).

    Returns
    -------
    list[PartitionInfo]
        One entry per rank.
    """
    if n_parts <= 0:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    if n_parts == 1:
        return _partition_single(mesh)

    n_inner = mesh.n_inner_cells
    face_cells = mesh.face_cells  # shape (2, n_faces)

    adjacency = _build_cell_adjacency(face_cells, n_inner)

    if _HAVE_PYMETIS and n_parts > 1:
        membership = _partition_pymetis(adjacency, n_parts)
    else:
        membership = _partition_contiguous(n_inner, n_parts)

    return _build_partition_infos(mesh, membership, n_parts)


# ---------------------------------------------------------------------------
# Partition construction helpers
# ---------------------------------------------------------------------------


def _partition_single(mesh) -> List[PartitionInfo]:
    """Fast path: the whole mesh is one partition (serial mode)."""
    n_inner = mesh.n_inner_cells
    n_cells = mesh.n_cells
    owned = np.arange(n_inner, dtype=int)
    ghosts = np.arange(n_inner, n_cells, dtype=int)
    all_faces = np.arange(mesh.n_faces, dtype=int)
    g2l = np.arange(n_cells, dtype=int)
    return [
        PartitionInfo(
            rank=0,
            owned_cells=owned,
            ghost_cells=ghosts,
            local_faces=all_faces,
            send_map={},
            recv_map={},
            global_to_local=g2l,
        )
    ]


def _build_partition_infos(
    mesh, membership: np.ndarray, n_parts: int
) -> List[PartitionInfo]:
    """Build full `PartitionInfo` objects from a membership vector.

    `membership` maps inner cell *i* -> partition rank.  Ghost cells in
    the original mesh (indices >= n_inner_cells) are boundary ghosts and
    are replicated on every partition that needs them.
    """
    n_inner = mesh.n_inner_cells
    face_cells = mesh.face_cells  # (2, n_faces)
    n_faces = face_cells.shape[1]

    # --- owned cells per rank -------------------------------------------------
    owned_per_rank: List[np.ndarray] = []
    for p in range(n_parts):
        owned_per_rank.append(np.where(membership == p)[0].astype(int))

    # --- Identify ghost cells and local faces per rank ------------------------
    # A ghost cell for rank p is any cell c that is NOT owned by p but
    # appears across a face from an owned cell of p.
    owned_set_per_rank = [set(o.tolist()) for o in owned_per_rank]

    ghost_sets: List[set] = [set() for _ in range(n_parts)]
    local_face_sets: List[set] = [set() for _ in range(n_parts)]

    for f in range(n_faces):
        a, b = int(face_cells[0, f]), int(face_cells[1, f])
        for p in range(n_parts):
            a_owned = a in owned_set_per_rank[p]
            b_owned = b in owned_set_per_rank[p]
            if a_owned or b_owned:
                local_face_sets[p].add(f)
            # Ghost identification: the *other* cell across the face
            if a_owned and not b_owned:
                ghost_sets[p].add(b)
            if b_owned and not a_owned:
                ghost_sets[p].add(a)

    # --- Build send/recv maps ------------------------------------------------
    # send_map[p][q] = local cells owned by p that are ghosts on q
    # recv_map[p][q] = local indices in p's ghost region coming from q
    partitions: List[PartitionInfo] = []
    for p in range(n_parts):
        owned = owned_per_rank[p]
        ghost_list = sorted(ghost_sets[p])
        ghosts = np.array(ghost_list, dtype=int)

        # global -> local mapping
        n_owned = len(owned)
        g2l = -np.ones(mesh.n_cells, dtype=int)
        for loc, glob in enumerate(owned):
            g2l[glob] = loc
        for loc_offset, glob in enumerate(ghosts):
            g2l[glob] = n_owned + loc_offset

        local_faces = np.array(sorted(local_face_sets[p]), dtype=int)

        # Build recv_map: classify ghost cells by their owning rank
        recv_map: Dict[int, list] = {}
        for gc in ghost_list:
            if gc < n_inner:
                owner = int(membership[gc])
                recv_map.setdefault(owner, []).append(g2l[gc])

        recv_map_final: Dict[int, np.ndarray] = {
            q: np.array(indices, dtype=int) for q, indices in recv_map.items()
        }

        partitions.append(
            PartitionInfo(
                rank=p,
                owned_cells=owned,
                ghost_cells=ghosts,
                local_faces=local_faces,
                send_map={},  # filled in second pass
                recv_map=recv_map_final,
                global_to_local=g2l,
            )
        )

    # Second pass: fill send_maps (requires all partitions' ghost info)
    for p in range(n_parts):
        send_map: Dict[int, np.ndarray] = {}
        for q in range(n_parts):
            if q == p:
                continue
            # Cells owned by p that are ghosts on q
            cells_to_send = []
            for gc in partitions[q].ghost_cells:
                gc = int(gc)
                if gc < n_inner and int(membership[gc]) == p:
                    cells_to_send.append(partitions[p].global_to_local[gc])
            if cells_to_send:
                send_map[q] = np.array(cells_to_send, dtype=int)
        # Rebuild partition with filled send_map
        partitions[p] = PartitionInfo(
            rank=p,
            owned_cells=partitions[p].owned_cells,
            ghost_cells=partitions[p].ghost_cells,
            local_faces=partitions[p].local_faces,
            send_map=send_map,
            recv_map=partitions[p].recv_map,
            global_to_local=partitions[p].global_to_local,
        )

    return partitions


# ---------------------------------------------------------------------------
# Helpers to get geometry arrays from any mesh level
# ---------------------------------------------------------------------------


def _get_geometry(mesh, name):
    """Get a geometry array from mesh, trying property, private attr, then computed method."""
    # Try direct property/attribute (works for FVMMesh/LSQMesh)
    attr = getattr(mesh, name, None)
    if attr is not None and hasattr(attr, 'shape'):
        return np.asarray(attr)
    # Try private attribute (FVMMesh stores as _cell_centers etc.)
    priv = f"_{name}"
    attr = getattr(mesh, priv, None)
    if attr is not None and hasattr(attr, 'shape'):
        return np.asarray(attr)
    # Try computed method (BaseMesh has cell_centers_computed() etc.)
    computed = f"{name}_computed"
    method = getattr(mesh, computed, None)
    if method is not None and callable(method):
        return np.asarray(method())
    raise AttributeError(f"Cannot get '{name}' from {type(mesh).__name__}")


# ---------------------------------------------------------------------------
# Local sub-mesh extraction
# ---------------------------------------------------------------------------


def extract_local_mesh(mesh, partition: PartitionInfo):
    """Build a local mesh for one rank from the global mesh and its
    `PartitionInfo`.

    * Owned cells come first (indices `0 .. n_owned-1`)
    * Ghost cells follow   (indices `n_owned .. n_owned+n_ghost-1`)
    * All topology arrays are remapped to local numbering.
    * Boundary-face bookkeeping is restricted to faces that touch an
      owned cell.

    The returned mesh is an `LSQMesh` (if available) or `FVMMesh`
    with precomputed geometry, ready for `convert_mesh_to_jax`.
    """
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    from zoomy_core.mesh.lsq_reconstruction import (
        least_squares_reconstruction_local,
        scale_lsq_derivative,
    )

    g2l = partition.global_to_local
    owned = partition.owned_cells
    ghosts = partition.ghost_cells
    local_faces = partition.local_faces

    n_owned = len(owned)
    n_ghost = len(ghosts)
    n_local_cells = n_owned + n_ghost

    # Concatenated global cell indices: owned first, then ghosts
    all_cells = np.concatenate([owned, ghosts])

    # ---- geometry arrays from global mesh ----------------------------------
    g_cell_volumes = _get_geometry(mesh, "cell_volumes")
    g_cell_centers = _get_geometry(mesh, "cell_centers")
    g_cell_inradius = _get_geometry(mesh, "cell_inradius")
    g_face_normals = _get_geometry(mesh, "face_normals")
    g_face_volumes = _get_geometry(mesh, "face_volumes")
    g_face_centers = _get_geometry(mesh, "face_centers")
    g_face_subvolumes = _get_geometry(mesh, "face_subvolumes")

    # ---- cell data (reindex) -----------------------------------------------
    cell_volumes = g_cell_volumes[all_cells]
    # cell_centers is (3, n_cells) in FVMMesh
    cell_centers = g_cell_centers[:, all_cells]
    cell_inradius = g_cell_inradius[all_cells]

    # ---- face data (reindex) -----------------------------------------------
    n_local_faces = len(local_faces)
    face_cells_global = mesh.face_cells[:, local_faces]  # (2, n_local_faces)
    face_cells_local = np.stack(
        [g2l[face_cells_global[0]], g2l[face_cells_global[1]]], axis=0
    )
    face_normals = g_face_normals[:, local_faces]
    face_volumes = g_face_volumes[local_faces]
    face_centers = g_face_centers[local_faces]
    face_subvolumes = g_face_subvolumes[local_faces]

    # ---- Build global face -> local face mapping ---------------------------
    gf2lf = -np.ones(mesh.n_faces, dtype=int)
    for lf, gf in enumerate(local_faces):
        gf2lf[gf] = lf

    # ---- cell_faces (only for owned/inner cells, respecting BaseMesh invariant)
    n_fpce = mesh.cell_faces.shape[0]
    cell_faces_local = np.full((n_fpce, n_owned), -1, dtype=int)
    for lc in range(n_owned):
        gc = owned[lc]
        if gc < mesh.cell_faces.shape[1]:
            for i in range(n_fpce):
                gf = mesh.cell_faces[i, gc]
                if 0 <= gf < len(gf2lf) and gf2lf[gf] >= 0:
                    cell_faces_local[i, lc] = gf2lf[gf]

    # ---- cell_neighbors (full: owned + ghost, as in BaseMesh)
    cell_neighbors_local = np.full((n_local_cells, mesh.n_faces_per_cell), -1, dtype=int)
    for lc, gc in enumerate(all_cells):
        if gc < mesh.cell_neighbors.shape[0]:
            for j in range(mesh.n_faces_per_cell):
                gn = mesh.cell_neighbors[gc, j]
                if 0 <= gn < len(g2l) and g2l[gn] >= 0:
                    cell_neighbors_local[lc, j] = g2l[gn]

    # ---- cell_vertices (only for owned/inner cells)
    n_vpce = mesh.cell_vertices.shape[0]
    cell_verts_owned = np.zeros((n_vpce, n_owned), dtype=int)
    for lc in range(n_owned):
        gc = owned[lc]
        if gc < mesh.cell_vertices.shape[1]:
            cell_verts_owned[:, lc] = mesh.cell_vertices[:, gc]
    unique_verts, inv = np.unique(cell_verts_owned.ravel(), return_inverse=True)
    cell_vertices_local = inv.reshape(cell_verts_owned.shape)
    vertex_coordinates = mesh.vertex_coordinates[:, unique_verts]
    n_vertices = len(unique_verts)

    # ---- boundary faces ----------------------------------------------------
    # Two sources: (1) physical boundary faces from the global mesh that
    # touch an owned cell, (2) inter-partition faces where one cell is
    # owned and the other is a ghost from another rank.  Both become
    # regular boundary faces in the local mesh (ghost cell on side B).

    # (1) Physical boundary faces
    phys_bf_cells = []
    phys_bf_ghosts = []
    phys_bf_func_nums = []
    phys_bf_tags = []
    phys_bf_face_indices = []
    for i in range(mesh.n_boundary_faces):
        gc = mesh.boundary_face_cells[i]
        if g2l[gc] >= 0 and g2l[gc] < n_owned:
            gf = mesh.boundary_face_face_indices[i]
            lf = gf2lf[gf]
            if lf >= 0:
                phys_bf_cells.append(g2l[gc])
                phys_bf_ghosts.append(g2l[mesh.boundary_face_ghosts[i]])
                phys_bf_func_nums.append(mesh.boundary_face_function_numbers[i])
                phys_bf_tags.append(mesh.boundary_face_physical_tags[i])
                phys_bf_face_indices.append(lf)

    # (2) Inter-partition faces: faces where cell A is owned, cell B is
    # a ghost from another rank (or vice versa).  These are inner faces
    # in the global mesh that become boundary-like in the local mesh.
    # The ghost cell already exists in the local mesh (indices n_owned..n_local-1).
    # No new ghost cells needed — face_cells already references them.
    # We do NOT add these as boundary faces because the flux loop already
    # covers them via face_cells (ghost values filled by halo exchange).
    # This matches the invariant: boundary_face arrays are for physical BCs only.

    n_phys_bf = len(phys_bf_cells)
    boundary_face_cells = np.array(phys_bf_cells, dtype=int) if phys_bf_cells else np.empty(0, dtype=int)
    boundary_face_ghosts = np.array(phys_bf_ghosts, dtype=int) if phys_bf_ghosts else np.empty(0, dtype=int)
    boundary_face_function_numbers = np.array(phys_bf_func_nums, dtype=int) if phys_bf_func_nums else np.empty(0, dtype=int)
    boundary_face_physical_tags = np.array(phys_bf_tags, dtype=int) if phys_bf_tags else np.empty(0, dtype=int)
    boundary_face_face_indices = np.array(phys_bf_face_indices, dtype=int) if phys_bf_face_indices else np.empty(0, dtype=int)
    n_boundary_faces = n_phys_bf

    # ---- Build the local LSQMesh -------------------------------------------
    dim = mesh.dimension
    lsq_degree = 1

    lsq_gradQ, lsq_neighbors, lsq_bdy_neighbors, lsq_monomial_multi_index = (
        least_squares_reconstruction_local(
            n_local_cells, dim, cell_neighbors_local, cell_centers[:dim, :].T, lsq_degree,
            n_inner_cells=n_owned,
            cell_vertices=cell_vertices_local,
        )
    )
    lsq_scale_factors = scale_lsq_derivative(lsq_monomial_multi_index)

    # Build face_neighbors for the local mesh
    from zoomy_core.mesh.lsq_mesh import _build_face_neighbors
    face_neighbors = _build_face_neighbors(
        face_cells_local, cell_neighbors_local,
        n_local_faces, mesh.n_faces_per_cell, n_local_cells,
    )

    z_ordering = np.array([-1])

    # Construct LSQMesh with all precomputed data
    local_mesh = LSQMesh(
        dimension=mesh.dimension,
        type=mesh.type,
        n_cells=n_local_cells,
        n_inner_cells=n_owned,
        n_faces=n_local_faces,
        n_vertices=n_vertices,
        n_boundary_faces=n_boundary_faces,
        n_faces_per_cell=mesh.n_faces_per_cell,
        vertex_coordinates=vertex_coordinates,
        cell_vertices=cell_vertices_local,
        cell_faces=cell_faces_local,
        face_cells=face_cells_local,
        cell_neighbors=cell_neighbors_local,
        boundary_face_cells=boundary_face_cells,
        boundary_face_ghosts=boundary_face_ghosts,
        boundary_face_function_numbers=boundary_face_function_numbers,
        boundary_face_physical_tags=boundary_face_physical_tags,
        boundary_face_face_indices=boundary_face_face_indices,
        boundary_conditions_sorted_physical_tags=mesh.boundary_conditions_sorted_physical_tags,
        boundary_conditions_sorted_names=list(mesh.boundary_conditions_sorted_names),
        z_ordering=z_ordering,
        # FVMMesh precomputed geometry
        _cell_centers=cell_centers,
        _cell_volumes=cell_volumes,
        _cell_inradius=cell_inradius,
        _face_normals=face_normals,
        _face_volumes=face_volumes,
        _face_centers=face_centers,
        # LSQMesh precomputed stencils
        _lsq_gradQ=lsq_gradQ,
        _lsq_neighbors=lsq_neighbors,
        _lsq_boundary_face_neighbors=lsq_bdy_neighbors,
        _lsq_monomial_multi_index=lsq_monomial_multi_index,
        _lsq_scale_factors=lsq_scale_factors,
        _face_neighbors=face_neighbors,
    )

    return local_mesh
