"""Sanity test for the SPMD 1D mesh partition utility."""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

from zoomy_core.mesh import LSQMesh
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_jax.mesh.partition_jax import partition_1d_contiguous


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_1d_shapes_and_counts():
    """16 cells, 4 partitions, halo=1 → each partition has 6 padded
    cells (4 owned + 2 halo), 5 internal faces, and 0 or 1 boundary
    faces (only the rank-0 and rank-(N-1) own a global-boundary face).

    ``n_inner_cells = n_padded`` (the full padded slab) so the per-
    cell reconstruction iteration covers owned + halo cells; the
    flux operator is responsible for restricting cell updates to
    ``[halo .. halo+n_local)``."""
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=16)
    mesh = convert_mesh_to_jax(mesh_np)
    parts = partition_1d_contiguous(mesh, n_parts=4, halo=1)
    assert len(parts) == 4
    for p in parts:
        assert p.n_inner_cells == 6
        assert p.n_cells == 6
        assert p.n_faces == 5
        assert p.cell_volumes.shape == (6,)
        assert p.face_cells.shape == (2, 5)
    assert int(parts[0].n_boundary_faces) == 1
    assert int(parts[1].n_boundary_faces) == 0
    assert int(parts[2].n_boundary_faces) == 0
    assert int(parts[3].n_boundary_faces) == 1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_owned_centers_match_global():
    """Each partition's owned cell centers must equal the
    corresponding slice of the global mesh."""
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=16)
    mesh = convert_mesh_to_jax(mesh_np)
    parts = partition_1d_contiguous(mesh, n_parts=4, halo=2)
    halo = 2
    n_local = 4
    global_centers = np.asarray(mesh.cell_centers[0, :16])
    for p_idx, part in enumerate(parts):
        owned = np.asarray(part.cell_centers[0, halo:halo + n_local])
        expected = global_centers[p_idx * n_local:(p_idx + 1) * n_local]
        np.testing.assert_array_almost_equal(owned, expected, decimal=12)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_face_topology_inside_padded_slab():
    """face_cells[0, k] = k, face_cells[1, k] = k+1 — faces between
    consecutive cells in the padded slab."""
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=16)
    mesh = convert_mesh_to_jax(mesh_np)
    parts = partition_1d_contiguous(mesh, n_parts=4, halo=1)
    for part in parts:
        fc0 = np.asarray(part.face_cells[0])
        fc1 = np.asarray(part.face_cells[1])
        assert fc0.tolist() == list(range(5))
        assert fc1.tolist() == list(range(1, 6))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_rejects_uneven_split():
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=17)
    mesh = convert_mesh_to_jax(mesh_np)
    with pytest.raises(ValueError, match="not divisible"):
        partition_1d_contiguous(mesh, n_parts=4, halo=1)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_lsq_remapping_owned_cells():
    """Owned cells' lsq_gradQ rows must equal the GLOBAL mesh's
    lsq_gradQ for the corresponding global cell index — the A-matrix
    is geometry-only and identical between global and padded layout.

    Owned cells' lsq_neighbors must point at local-padded indices
    that resolve to the SAME physical cells as the global neighbors
    (verified via cell_centers).
    """
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=16)
    mesh = convert_mesh_to_jax(mesh_np)
    halo = 2
    n_local = 4
    parts = partition_1d_contiguous(mesh, n_parts=4, halo=halo)
    global_gradQ = np.asarray(mesh.lsq_gradQ)
    global_neighbors = np.asarray(mesh.lsq_neighbors)
    global_centers = np.asarray(mesh.cell_centers[0])
    for p_idx, part in enumerate(parts):
        for j in range(n_local):
            local_i = halo + j
            g = p_idx * n_local + j
            np.testing.assert_array_almost_equal(
                np.asarray(part.lsq_gradQ[local_i]),
                global_gradQ[g],
                decimal=6,
                err_msg=f"part {p_idx} owned cell j={j} (global {g}) "
                        f"lsq_gradQ mismatch",
            )
            for k in range(global_neighbors.shape[1]):
                local_nbr = int(part.lsq_neighbors[local_i, k])
                global_nbr = int(global_neighbors[g, k])
                np.testing.assert_almost_equal(
                    float(part.cell_centers[0, local_nbr]),
                    float(global_centers[global_nbr]),
                    decimal=6,
                    err_msg=f"part {p_idx} cell {local_i} neighbor "
                            f"{k}: local idx {local_nbr} (center "
                            f"{float(part.cell_centers[0, local_nbr])}) "
                            f"≠ global idx {global_nbr} (center "
                            f"{float(global_centers[global_nbr])})",
                )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_lsq_inner_halo_cell_has_complete_stencil():
    """With halo=2 and linear LSQ (radius 1), the INNER halo cell
    on each side has a complete stencil within the padded slab and
    its lsq_gradQ matches the global cell.  The OUTER halo cell falls
    back to a self-stencil (gradient = 0)."""
    mesh_np = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=16)
    mesh = convert_mesh_to_jax(mesh_np)
    halo = 2
    n_local = 4
    parts = partition_1d_contiguous(mesh, n_parts=4, halo=halo)
    global_gradQ = np.asarray(mesh.lsq_gradQ)
    # Take an interior partition (no global-boundary cells).
    part = parts[1]
    # Inner left halo cell is at local index halo-1 = 1 (closest to owned).
    # Global = p*n_local + (1 - halo) = 4 + (1-2) = 3.
    np.testing.assert_array_almost_equal(
        np.asarray(part.lsq_gradQ[halo - 1]),
        global_gradQ[3],
        decimal=6,
    )
    # Outer left halo at local index 0 has incomplete stencil
    # (would need a cell at global p*n_local - 3 = 1, which IS in
    # the padded slab via the other direction... actually:
    # for halo=2, p=1, n_local=4, the outer halo at local 0 = global 2.
    # Global 2 has neighbors [1, 3]. global 1 → rel = -3, outside the
    # padded slab (only halo=2 deep) → fallback to self-stencil.
    assert int(part.lsq_neighbors[0, 0]) == 0
    assert int(part.lsq_neighbors[0, 1]) == 0
    np.testing.assert_array_almost_equal(
        np.asarray(part.lsq_gradQ[0]),
        np.zeros_like(global_gradQ[2]),
        decimal=12,
    )
    # Inner right halo cell at local index halo+n_local = 6 (closest
    # to owned).  Global = p*n_local + (6 - halo) = 4 + 4 = 8.
    np.testing.assert_array_almost_equal(
        np.asarray(part.lsq_gradQ[halo + n_local]),
        global_gradQ[8],
        decimal=6,
    )
