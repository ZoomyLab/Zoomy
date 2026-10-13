"""SPMD partitioning for `MeshJAX` — produces per-device padded
meshes with halo-shifted indices and remapped LSQ stencils.

Each device gets a `MeshJAX` whose `cell_*` arrays carry
`n_local + 2*halo` cells in this layout:

  [  halo_left  |  n_local owned cells  |  halo_right  ]
     |←  halo →|                        |←  halo →|

The padded slab matches the convention used by
`halo_exchange_inplace`.

LSQ stencil remapping
---------------------
The per-cell LSQ A-matrix (`mesh.lsq_gradQ[g]`) is **invariant** —
it depends only on local geometry, which is identical between the
global cell and its padded-slab image.  What changes is the **index
pointers** (`lsq_neighbors`, `lsq_boundary_face_neighbors`).

For each owned cell at local index `i = halo + j` (global
`g = p*n_local + j`):
  - `part.lsq_gradQ[i] = mesh.lsq_gradQ[g]` (copied unchanged).
  - `part.lsq_neighbors[i, k]` = local-padded index of
    `mesh.lsq_neighbors[g, k]`, via the simple offset map
    `g_idx − p*n_local + halo`.
  - `part.lsq_boundary_face_neighbors[i, k]` remaps the global
    boundary-face index to the per-partition boundary-face index
    (each interior partition has 0; rank-0 has the global left,
    rank-(N-1) has the global right).

Halo cells (`i ∈ [0, halo) ∪ [halo+n_local, halo+n_local+halo)`)
also get an LSQ entry: if the cell's stencil fits in the padded
slab (i.e., halo is wide enough), the real LSQ data is copied;
otherwise the cell falls back to a **self-stencil** (gradient
evaluates to zero — first-order at that cell), and only the owned-
cell + first-halo-cell faces are bit-identical to single-device.

For linear LSQ (radius 1), bit-identical second-order at inter-
partition faces needs `halo ≥ 2`; `halo = 1` is fine for
first-order constant reconstruction but degrades MUSCL reconstruction
to mixed order at the partition boundary.

Partition layout: contiguous block along the cell axis — the only
partition strategy that makes physical sense for a regular
structured 1D mesh.  For unstructured / 2D-3D meshes use the
existing graph-based `zoomy_jax.mesh.partition.partition_mesh`.
"""
from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from zoomy_jax.mesh.mesh import MeshJAX, convert_mesh_to_jax


def _g2l(g_idx: int, p: int, n_local: int, halo: int) -> int:
    """Map a global cell index to a local padded index in partition
    `p`.  Returns -1 if the global cell is outside the padded slab."""
    rel = g_idx - p * n_local
    if 0 <= rel < n_local:
        return halo + rel
    if -halo <= rel < 0:
        return halo + rel  # left halo: rel ∈ [-halo, 0) → local ∈ [0, halo)
    if n_local <= rel < n_local + halo:
        return halo + rel  # right halo: rel ∈ [n_local, n_local+halo)
    return -1


def _bdy_global_to_local(
    g_bdy: int, p: int, n_parts: int
) -> int:
    """Map a global boundary-face index (0 = left, 1 = right) to the
    per-partition boundary-face index.  Each partition holds at most
    one global boundary face; if a global boundary face does not
    appear in this partition return -1."""
    if g_bdy < 0:
        return -1
    if g_bdy == 0 and p == 0:
        return 0
    if g_bdy == 1 and p == n_parts - 1:
        return 0
    return -1


def partition_1d_contiguous(
    mesh: MeshJAX, n_parts: int, halo: int
) -> List[MeshJAX]:
    """Split a 1D `MeshJAX` into `n_parts` contiguous chunks, each
    padded with `halo` cells on both sides; LSQ stencils remapped.

    Per-partition layout (cell axis):
        local cell index 0 .. halo-1            : LEFT halo
        local cell index halo .. halo+n_local-1 : owned
        local cell index halo+n_local .. +halo  : RIGHT halo

    `part.n_inner_cells = n_local + 2*halo` so the reconstruction
    classes iterate over the full padded slab.  The flux operator is
    responsible for restricting cell updates to owned cells
    `[halo .. halo+n_local)` — halo cells get refreshed by
    `halo_exchange_inplace` at the next step.

    Parameters
    ----------
    mesh : MeshJAX
        Global 1D mesh.  Must satisfy ``mesh.n_inner_cells % n_parts
        == 0``.
    n_parts : int
    halo : int
        Halo width on each side.  Must be ≥ 1; ≥ 2 to keep MUSCL
        reconstruction bit-identical at inter-partition faces.

    Returns
    -------
    list of MeshJAX
        `n_parts` per-partition meshes with padded layout and
        remapped LSQ data.
    """
    if mesh.dimension != 1:
        raise NotImplementedError(
            "partition_1d_contiguous: only 1D meshes today.  For 2D "
            "x-axis decomposition (regular nx*ny mesh, partition only "
            "along x): follow the same recipe — owned cells at "
            "ix ∈ [p*n_local_x .. (p+1)*n_local_x), iy ∈ [0..ny), "
            "halo strips of `halo*ny` cells on each x-side; LSQ "
            "stencils remap via the same g_idx -> local mapping "
            "(2D LSQ stencil includes corner neighbors, but with x-"
            "partitioning only those reach into the halo strip, so "
            "halo = 1 is enough as long as y is not also "
            "partitioned).  For unstructured/graph partitioning, "
            "extend the existing pymetis-based "
            "zoomy_jax.mesh.partition.partition_mesh to emit per-"
            "rank MeshJAX with halo cells (it currently emits "
            "PartitionInfo for the MPI pattern)."
        )
    nc_global = int(mesh.n_inner_cells)
    if nc_global % n_parts != 0:
        raise ValueError(
            f"partition_1d_contiguous: n_inner_cells={nc_global} not "
            f"divisible by n_parts={n_parts}."
        )
    n_local = nc_global // n_parts
    n_padded = n_local + 2 * halo

    cell_centers = np.asarray(mesh.cell_centers)
    cell_volumes = np.asarray(mesh.cell_volumes)
    dx = float(cell_volumes[0])

    lsq_neighbors_g = np.asarray(mesh.lsq_neighbors)
    lsq_gradQ_g = np.asarray(mesh.lsq_gradQ)
    lsq_bdy_g = np.asarray(mesh.lsq_boundary_face_neighbors)
    max_nbr = lsq_neighbors_g.shape[1]
    max_bdy = lsq_bdy_g.shape[1]
    n_rows = lsq_gradQ_g.shape[1]  # = max_nbr + max_bdy
    n_mono = lsq_gradQ_g.shape[2]

    parts: List[MeshJAX] = []
    for p in range(n_parts):
        start = p * n_local
        stop = start + n_local
        owned_centers = cell_centers[:, start:stop]

        left_halo_x = np.array(
            [cell_centers[0, start] - (i + 1) * dx for i in range(halo)]
        )[::-1]
        right_halo_x = np.array(
            [cell_centers[0, stop - 1] + (i + 1) * dx for i in range(halo)]
        )
        pad_centers = np.zeros((cell_centers.shape[0], n_padded))
        pad_centers[0, :halo] = left_halo_x
        pad_centers[0, halo:halo + n_local] = owned_centers[0, :]
        pad_centers[0, -halo:] = right_halo_x

        pad_vols = np.full(n_padded, dx)
        pad_inradius = np.full(n_padded, dx * 0.5)

        # Faces: sequential between consecutive padded cells.
        n_faces_local = n_padded - 1
        face_cells = np.zeros((2, n_faces_local), dtype=np.int32)
        face_cells[0, :] = np.arange(n_faces_local)
        face_cells[1, :] = np.arange(1, n_faces_local + 1)

        face_normals = np.zeros((mesh.face_normals.shape[0], n_faces_local))
        face_normals[0, :] = 1.0
        face_volumes = np.ones(n_faces_local)

        face_centers = np.zeros((n_faces_local, mesh.face_centers.shape[1]))
        face_centers[:, 0] = 0.5 * (pad_centers[0, :-1] + pad_centers[0, 1:])

        # Boundary face flagging (global left at face halo-1 for rank 0;
        # global right at face halo+n_local-1 for rank n_parts-1).
        bf_face_idx_list: List[int] = []
        bf_cells_list: List[int] = []
        bf_ghosts_list: List[int] = []
        bf_func_no_list: List[int] = []
        bf_phys_tags_list: List[int] = []
        if p == 0:
            bf_face_idx_list.append(halo - 1)
            bf_cells_list.append(halo)
            bf_ghosts_list.append(halo - 1)
            bf_func_no_list.append(0)
            bf_phys_tags_list.append(1)
        if p == n_parts - 1:
            bf_face_idx_list.append(halo + n_local - 1)
            bf_cells_list.append(halo + n_local - 1)
            bf_ghosts_list.append(halo + n_local)
            bf_func_no_list.append(1)
            bf_phys_tags_list.append(2)

        n_bf = len(bf_face_idx_list)
        bf_face_idx = np.array(bf_face_idx_list, dtype=np.int32)
        bf_cells = np.array(bf_cells_list, dtype=np.int32)
        bf_ghosts = np.array(bf_ghosts_list, dtype=np.int32)
        bf_func_no = np.array(bf_func_no_list, dtype=np.int32)
        bf_phys_tags = np.array(bf_phys_tags_list, dtype=np.int32)

        # ── LSQ stencil remapping ──────────────────────────────────
        part_lsq_neighbors = np.zeros((n_padded, max_nbr), dtype=np.int32)
        part_lsq_gradQ = np.zeros((n_padded, n_rows, n_mono))
        part_lsq_bdy = np.full((n_padded, max_bdy), -1, dtype=np.int32)

        def _remap_cell(local_i: int, g: int) -> bool:
            """Try to remap LSQ data for the cell whose global index
            is `g` into per-partition slot `local_i`.  Returns
            False if the stencil reaches outside the padded slab
            (caller falls back to self-stencil)."""
            if not (0 <= g < nc_global):
                return False
            tmp_nbrs = []
            for k in range(max_nbr):
                lnbr = _g2l(int(lsq_neighbors_g[g, k]), p, n_local, halo)
                if lnbr < 0:
                    return False
                tmp_nbrs.append(lnbr)
            # All neighbors fit — commit.
            for k in range(max_nbr):
                part_lsq_neighbors[local_i, k] = tmp_nbrs[k]
            part_lsq_gradQ[local_i, :, :] = lsq_gradQ_g[g, :, :]
            for k in range(max_bdy):
                part_lsq_bdy[local_i, k] = _bdy_global_to_local(
                    int(lsq_bdy_g[g, k]), p, n_parts
                )
            return True

        def _self_stencil(local_i: int) -> None:
            for k in range(max_nbr):
                part_lsq_neighbors[local_i, k] = local_i
            # gradQ already zero ⇒ gradient = 0 (first-order fallback).
            # bdy already -1.

        # Owned cells.
        for j in range(n_local):
            local_i = halo + j
            g = p * n_local + j
            ok = _remap_cell(local_i, g)
            if not ok:
                raise ValueError(
                    f"partition_1d_contiguous: owned cell at local "
                    f"index {local_i} (global {g}) has an LSQ stencil "
                    f"reaching outside the padded slab; increase halo "
                    f"to ≥ {max_nbr + 1} or use a different stencil."
                )

        # Halo cells: try to remap, else self-stencil fallback.
        # Left halo at local indices [0, halo).  Global = p*n_local +
        # (local_i - halo)  ∈ [p*n_local - halo, p*n_local).
        for local_i in range(halo):
            g = p * n_local + (local_i - halo)
            if not _remap_cell(local_i, g):
                _self_stencil(local_i)
        # Right halo at local indices [halo+n_local, n_padded).  Global =
        # p*n_local + (local_i - halo)  ∈ [(p+1)*n_local, (p+1)*n_local
        # + halo).
        for local_i in range(halo + n_local, n_padded):
            g = p * n_local + (local_i - halo)
            if not _remap_cell(local_i, g):
                _self_stencil(local_i)

        part = MeshJAX(
            dimension=mesh.dimension,
            type=mesh.type,
            n_cells=n_padded,
            n_inner_cells=n_padded,  # full padded slab; flux operator
                                     # restricts updates to owned cells.
            n_faces=n_faces_local,
            n_vertices=n_padded + 1,
            n_boundary_faces=n_bf,
            n_faces_per_cell=mesh.n_faces_per_cell,
            vertex_coordinates=jnp.zeros((mesh.dimension, n_padded + 1)),
            cell_vertices=jnp.zeros((2, n_padded), dtype=jnp.int32),
            cell_faces=jnp.zeros(
                (mesh.n_faces_per_cell, n_padded), dtype=jnp.int32
            ),
            cell_volumes=jnp.asarray(pad_vols),
            cell_centers=jnp.asarray(pad_centers),
            cell_inradius=jnp.asarray(pad_inradius),
            cell_neighbors=jnp.zeros(
                (n_padded, mesh.n_faces_per_cell), dtype=jnp.int32
            ),
            boundary_face_cells=jnp.asarray(bf_cells),
            boundary_face_ghosts=jnp.asarray(bf_ghosts),
            boundary_face_function_numbers=jnp.asarray(bf_func_no),
            boundary_face_physical_tags=jnp.asarray(bf_phys_tags),
            boundary_face_face_indices=jnp.asarray(bf_face_idx),
            face_cells=jnp.asarray(face_cells),
            face_normals=jnp.asarray(face_normals),
            face_volumes=jnp.asarray(face_volumes),
            face_centers=jnp.asarray(face_centers),
            face_subvolumes=jnp.zeros((n_faces_local, 2)),
            face_neighbors=jnp.zeros((n_faces_local, 4), dtype=jnp.int32),
            boundary_conditions_sorted_physical_tags=jnp.asarray([1, 2]),
            boundary_conditions_sorted_names=["left", "right"],
            lsq_gradQ=jnp.asarray(part_lsq_gradQ),
            lsq_neighbors=jnp.asarray(part_lsq_neighbors),
            lsq_boundary_face_neighbors=jnp.asarray(part_lsq_bdy),
            lsq_monomial_multi_index=mesh.lsq_monomial_multi_index,
            lsq_scale_factors=mesh.lsq_scale_factors,
            z_ordering=jnp.zeros(n_padded, dtype=jnp.int32),
        )
        parts.append(part)
    return parts


# ─────────────────────────────────────────────────────────────────────
# x-axis decomposition for structured 1D / 2D / 3D meshes
# ─────────────────────────────────────────────────────────────────────


def partition_xaxis_structured(
    global_mesh: MeshJAX,
    n_parts: int,
    halo: int,
    domain: Tuple[float, ...],
    shape: Tuple[int, ...],
) -> List[MeshJAX]:
    """SPMD x-axis decomposition for a uniform structured mesh in
    1D, 2D, or 3D.  Generalises `partition_1d_contiguous` —
    each rank gets a padded x-strip / x-slab whose faces, geometry,
    and (interior) LSQ stencils are built via `LSQMesh.create_Nd`;
    boundary-aware LSQ data for cells whose stencil fits in the
    slab is then OVERRIDDEN with the global mesh's data
    (index-remapped) so face reconstruction at inter-partition faces
    is bit-identical to a single-device run.

    Cell ordering follows the LSQMesh convention:
      * 1D: `ic = ix`
      * 2D: `ic = ix * ny + iy`
      * 3D: `ic = ix * ny * nz + iy * nz + iz`

    The padded slab's cells are contiguous in this linear cell index,
    so `halo_exchange_inplace` works UNCHANGED — pass
    `halo_cell = halo * x_stride` where `x_stride = ny` in 2D and
    `ny * nz` in 3D.

    Parameters
    ----------
    global_mesh : MeshJAX
        Global uniform structured mesh.  Must satisfy
        `shape[0] % n_parts == 0`.
    n_parts : int
        Number of x-partitions.
    halo : int
        Halo width in **x cells** on each side.  `halo >= 2` is
        needed to keep order-2 LSQ-MUSCL reconstruction bit-identical
        at inter-partition x-faces.
    domain : tuple of float
        Global domain extent.  `(x_min, x_max)` for 1D,
        `(x_min, x_max, y_min, y_max)` for 2D,
        `(x_min, x_max, y_min, y_max, z_min, z_max)` for 3D.
    shape : tuple of int
        Global structured mesh shape.  `(nx,)` for 1D,
        `(nx, ny)` for 2D, `(nx, ny, nz)` for 3D.

    Returns
    -------
    list of MeshJAX
        `n_parts` per-partition padded slabs.  The slab geometry is
        built via `LSQMesh.create_Nd` so face_cells / face_centers /
        normals are automatically correct.  ``n_inner_cells =
        n_padded_x * x_stride`` (the full slab); the flux operator is
        responsible for restricting cell updates to owned cells.

    Limitations (current)
    ---------------------
    * Owned cells whose global LSQ stencil reaches a global y- or
      z-boundary have their `lsq_boundary_face_neighbors` set to
      `-1` rather than mapped to the slab's y/z BC face index.  For
      ICs that are constant in y/z this has no effect; for true
      cross-axis BC-aware reconstruction at owned cells, a per-cell
      face-position lookup is needed (TODO).
    """
    from zoomy_core.mesh import LSQMesh
    dim = len(shape)
    if not 1 <= dim <= 3:
        raise ValueError(f"shape must have 1-3 entries, got {shape}")
    if int(global_mesh.dimension) != dim:
        raise ValueError(
            f"shape implies dim={dim} but global_mesh.dimension="
            f"{int(global_mesh.dimension)}"
        )
    if len(domain) != 2 * dim:
        raise ValueError(
            f"domain must have 2*dim={2*dim} entries, got {domain}"
        )

    nx_global = int(shape[0])
    if nx_global % n_parts != 0:
        raise ValueError(
            f"partition_xaxis_structured: nx_global={nx_global} not "
            f"divisible by n_parts={n_parts}."
        )
    n_local_x = nx_global // n_parts
    n_padded_x = n_local_x + 2 * halo
    x_stride = int(np.prod([int(s) for s in shape[1:]])) if dim > 1 else 1

    x_min = float(domain[0])
    x_max = float(domain[1])
    dx = (x_max - x_min) / nx_global

    g_neighbors = np.asarray(global_mesh.lsq_neighbors)
    g_gradQ = np.asarray(global_mesh.lsq_gradQ)
    g_bdy = np.asarray(global_mesh.lsq_boundary_face_neighbors)
    n_cells_global = int(global_mesh.n_inner_cells)
    max_nbr = g_neighbors.shape[1]
    max_bdy = g_bdy.shape[1]

    parts: List[MeshJAX] = []
    for p in range(n_parts):
        # Slab x-extent (may extend outside global domain for rank-0
        # left halo or rank-(N-1) right halo).
        x_min_slab = x_min + (p * n_local_x - halo) * dx
        x_max_slab = x_min_slab + n_padded_x * dx

        if dim == 1:
            slab_lsq = LSQMesh.create_1d(
                (x_min_slab, x_max_slab), n_padded_x
            )
        elif dim == 2:
            slab_lsq = LSQMesh.create_2d(
                (x_min_slab, x_max_slab, float(domain[2]), float(domain[3])),
                n_padded_x, int(shape[1]),
            )
        else:  # dim == 3
            slab_lsq = LSQMesh.create_3d(
                (x_min_slab, x_max_slab, float(domain[2]), float(domain[3]),
                 float(domain[4]), float(domain[5])),
                n_padded_x, int(shape[1]), int(shape[2]),
            )
        slab_mesh = convert_mesh_to_jax(slab_lsq)
        n_slab = int(slab_mesh.n_inner_cells)
        assert n_slab == n_padded_x * x_stride, (
            f"slab cell count {n_slab} != n_padded_x * x_stride "
            f"{n_padded_x * x_stride}"
        )

        # ── Helper: global → local cell index in this partition ──
        def _g2l_xaxis(ic_global: int) -> int:
            ix_g, rest = divmod(ic_global, x_stride)
            ix_l = ix_g - p * n_local_x + halo
            if 0 <= ix_l < n_padded_x and 0 <= ic_global < n_cells_global:
                return ix_l * x_stride + rest
            return -1

        # ── LSQ override (owned + remappable halo cells) ──
        slab_gradQ = np.asarray(slab_mesh.lsq_gradQ).copy()
        slab_nbr = np.asarray(slab_mesh.lsq_neighbors).copy()
        slab_bdy_local = np.asarray(
            slab_mesh.lsq_boundary_face_neighbors
        ).copy()

        for ic_local in range(n_slab):
            ic_global_candidate = (
                ic_local + (p * n_local_x - halo) * x_stride
            )
            ix_g_cand = ic_global_candidate // x_stride
            if not (0 <= ix_g_cand < nx_global):
                # Outside global mesh (rank-0 left halo or rank-(N-1)
                # right halo on a non-periodic domain) — keep slab's
                # auto-built LSQ.  These cells never feed an owned-cell
                # update via reconstruction (faces between halo cells
                # are not consumed).
                continue
            ic_global = ic_global_candidate
            # Try to remap all neighbors.
            tmp_nbrs = []
            ok = True
            for k in range(max_nbr):
                gnbr = int(g_neighbors[ic_global, k])
                lnbr = _g2l_xaxis(gnbr)
                if lnbr < 0:
                    ok = False
                    break
                tmp_nbrs.append(lnbr)
            if not ok:
                # Outer halo cell with stencil reach beyond the slab.
                continue
            for k in range(max_nbr):
                slab_nbr[ic_local, k] = tmp_nbrs[k]
            slab_gradQ[ic_local, :, :] = g_gradQ[ic_global, :, :]
            # Boundary-face neighbors: clear them.  For owned cells
            # touching a y- or z-boundary the proper mapping would
            # point at the slab's y/z BC face index, but this
            # requires a per-cell face-position lookup — TODO.  Until
            # that lands, ICs that are constant in y/z are unaffected
            # (the would-be BC delta is zero).
            for k in range(max_bdy):
                slab_bdy_local[ic_local, k] = -1

        # ── Boundary face filtering ──
        # Drop slab x-BC faces that are actually inter-partition.
        slab_bf_idx = np.asarray(slab_mesh.boundary_face_face_indices)
        slab_face_centers = np.asarray(slab_mesh.face_centers)
        tol = dx * 1e-6
        keep_mask = np.ones(len(slab_bf_idx), dtype=bool)
        for i_bf, fidx in enumerate(slab_bf_idx):
            fc_x = slab_face_centers[fidx, 0]
            if abs(fc_x - x_min_slab) < tol and p != 0:
                keep_mask[i_bf] = False
            elif abs(fc_x - x_max_slab) < tol and p != n_parts - 1:
                keep_mask[i_bf] = False

        kept_idx = np.where(keep_mask)[0]
        n_bf = len(kept_idx)
        bf_face_idx = np.asarray(
            slab_mesh.boundary_face_face_indices
        )[kept_idx]
        bf_cells = np.asarray(slab_mesh.boundary_face_cells)[kept_idx]
        bf_ghosts = np.asarray(slab_mesh.boundary_face_ghosts)[kept_idx]
        bf_func_no = np.asarray(
            slab_mesh.boundary_face_function_numbers
        )[kept_idx]
        bf_phys_tags = np.asarray(
            slab_mesh.boundary_face_physical_tags
        )[kept_idx]

        # Update lsq_boundary_face_neighbors entries that still
        # reference the slab's BC face list (= cells we didn't
        # override above).  Old slab-BC index → new index in kept_idx.
        old_to_new = -np.ones(len(slab_bf_idx), dtype=np.int32)
        old_to_new[kept_idx] = np.arange(n_bf, dtype=np.int32)
        for ic_local in range(n_slab):
            for k in range(max_bdy):
                v = int(slab_bdy_local[ic_local, k])
                if v >= 0:
                    slab_bdy_local[ic_local, k] = int(old_to_new[v])
                # -1 stays -1.

        # ── Build the per-partition MeshJAX (replace LSQ + BC bits) ──
        from dataclasses import replace
        part = replace(
            slab_mesh,
            n_boundary_faces=n_bf,
            boundary_face_cells=jnp.asarray(bf_cells),
            boundary_face_ghosts=jnp.asarray(bf_ghosts),
            boundary_face_function_numbers=jnp.asarray(bf_func_no),
            boundary_face_physical_tags=jnp.asarray(bf_phys_tags),
            boundary_face_face_indices=jnp.asarray(bf_face_idx),
            lsq_gradQ=jnp.asarray(slab_gradQ),
            lsq_neighbors=jnp.asarray(slab_nbr),
            lsq_boundary_face_neighbors=jnp.asarray(slab_bdy_local),
        )
        parts.append(part)
    return parts
