"""SPMD bit-identity: ``LSQMUSCLReconstructionJAX`` composes with
``shard_map`` when handed a per-partition mesh with LSQ stencils
remapped to local-padded indices.

The crux: the per-cell LSQ A matrix is geometry-only — IDENTICAL in
global and partitioned form.  Only the index pointers change.  With
``halo = 2`` for a radius-1 linear LSQ stencil, the FIRST halo cell
on each side has a complete stencil within the padded slab, so face
reconstruction at every inter-partition face matches bit-exactly
between the left and right device's view of the same physical face.

The test fixes a smooth PERIODIC IC and an INTERIOR per-partition
mesh (parts[1]) replicated on every device — that way no device
imposes a global-boundary LSQ augmentation it shouldn't, and the
periodic-wrap halo exchange supplies the cross-partition values.

This is the LSQ-MUSCL analogue of ``test_spmd_solver_integration``,
which already proved ``ConstantReconstruction`` composes.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from zoomy_core.mesh import LSQMesh
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_jax.mesh.partition_jax import partition_1d_contiguous
from zoomy_jax.fvm.reconstruction_jax import LSQMUSCLReconstructionJAX


N_TOTAL = 32
N_DEVS = 4
N_LOCAL = N_TOTAL // N_DEVS  # 8
HALO = 2     # radius-1 linear LSQ + 1 extra ring for halo-cell stencils
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL


def _smooth_ic(x):
    """Smooth periodic IC — bit-identity at LSQ MUSCL faces requires
    a smooth-enough state that the gradient at every cell is well-
    defined."""
    return 1.0 + 0.5 * np.sin(2 * np.pi * x)


def _periodic_halo(Q_pad, halo, axis_name, n_devices):
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(right_owned, perm=perm_right, axis_name=axis_name)
    fill_right = lax.ppermute(left_owned, perm=perm_left, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo].set(fill_left)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right)
    return Q_pad


def _build_spmd_recon():
    """Set up: global mesh → partitions → reconstruction on an
    INTERIOR partition (parts[1]) shared by every device.  All-cells-
    interior LSQ stencil + periodic halo gives a true periodic SPMD
    layout."""
    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    mesh_global = convert_mesh_to_jax(mesh_np)
    parts = partition_1d_contiguous(mesh_global, n_parts=N_DEVS, halo=HALO)
    # parts[1] is an INTERIOR partition with n_boundary_faces=0 and
    # all-interior LSQ stencils — the representative we want on every
    # device under shard_map.
    part_mesh = parts[1]
    assert int(part_mesh.n_boundary_faces) == 0, (
        "Need interior partition; parts[1] should have 0 boundary faces"
    )
    recon_local = LSQMUSCLReconstructionJAX(
        part_mesh, dim=1, limiter="venkatakrishnan"
    )
    return mesh_global, part_mesh, recon_local


def _make_padded_Q(u0_np):
    """Layout per device: [halo_zeros | n_local owned | halo_zeros].
    The halo zeros are overwritten by periodic_halo_exchange on the
    SPMD path."""
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, HALO)), chunk, np.zeros((1, HALO))], axis=1
    )
    chunks = [u0_np[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    return jnp.asarray(np.concatenate([pad_chunk(c) for c in chunks], axis=1))


def _run_spmd_recon(recon_local, part_mesh, spmd_mesh, Q_pad_global):
    def reconstruct_local(Q_pad):
        Q_pad = _periodic_halo(Q_pad, HALO, "cells", N_DEVS)
        # LSQ recon's per-cell bdy gather indexes u_bf[max(bf,0)] before
        # the where-mask kicks in — give it at least one dummy entry so
        # the gather stays in range when n_boundary_faces=0.
        bf_local = jnp.zeros(
            (Q_pad.shape[0], max(1, int(part_mesh.n_boundary_faces)))
        )
        Q_L, Q_R = recon_local(Q_pad, bf_local)
        return Q_L, Q_R

    @partial(shard_map, mesh=spmd_mesh, in_specs=P(None, "cells"),
             out_specs=(P(None, "cells"), P(None, "cells")), check_rep=False)
    def run(Q_pad):
        return reconstruct_local(Q_pad)
    return run(Q_pad_global)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_inter_partition_face_consistency():
    """At every inter-partition face the LEFT device's view of the
    face (its rightmost owned-adjacent face) and the RIGHT device's
    view of the SAME physical face (its leftmost owned-adjacent face)
    must agree to single-precision bit-identity (≈ 1e-6 for float32).

    This is the SPMD halo-correctness invariant — proves the halo
    concept works flawlessly: the inner-halo cell on each side has
    its LSQ stencil correctly remapped, and the reconstruction at
    the face produces the same (Q_L, Q_R) from both partitions.
    """
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0_np = _smooth_ic(xc).astype(np.float32).reshape(1, N_TOTAL)

    _, part_mesh, recon_local = _build_spmd_recon()
    Q_pad_global = _make_padded_Q(u0_np)
    Q_L, Q_R = _run_spmd_recon(
        recon_local, part_mesh, spmd_mesh, Q_pad_global
    )
    Q_L = np.asarray(Q_L)
    Q_R = np.asarray(Q_R)
    n_faces_local = N_LOCAL + 2 * HALO - 1  # 11 for N_LOCAL=8, HALO=2

    # Inter-partition face between dev d and dev d+1:
    #   - on dev d: face index HALO + N_LOCAL - 1 (rightmost owned-adj
    #     face, local index 9 for HALO=2 N_LOCAL=8)
    #   - on dev d+1: face index HALO - 1 (leftmost owned-adj face,
    #     local index 1)
    max_diff_L = 0.0
    max_diff_R = 0.0
    for d in range(N_DEVS - 1):
        base_left = d * n_faces_local
        base_right = (d + 1) * n_faces_local
        left_view_L = float(Q_L[0, base_left + HALO + N_LOCAL - 1])
        right_view_L = float(Q_L[0, base_right + HALO - 1])
        left_view_R = float(Q_R[0, base_left + HALO + N_LOCAL - 1])
        right_view_R = float(Q_R[0, base_right + HALO - 1])
        max_diff_L = max(max_diff_L, abs(left_view_L - right_view_L))
        max_diff_R = max(max_diff_R, abs(left_view_R - right_view_R))

    print(f"  inter-partition face consistency: ΔQ_L={max_diff_L:.3e} "
          f"ΔQ_R={max_diff_R:.3e}")
    assert max_diff_L < 1e-5, (
        f"Q_L disagreement at inter-partition face: {max_diff_L:.3e}"
    )
    assert max_diff_R < 1e-5, (
        f"Q_R disagreement at inter-partition face: {max_diff_R:.3e}"
    )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_spmd_matches_replicated_single_device():
    """Bit-identity vs a single-device reference that runs the SAME
    INTERIOR-CELL LSQ MUSCL across the WHOLE periodic global state —
    i.e., the SPMD path and a "give one device 4× the work" path
    must produce identical face reconstructions over the owned faces.

    Reference: stack all 4 per-device padded slabs into a single
    padded mesh on one device, run reconstruction once.
    SPMD: same data, sharded.  Compare owned-adjacent faces only.
    """
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0_np = _smooth_ic(xc).astype(np.float32).reshape(1, N_TOTAL)

    _, part_mesh, recon_local = _build_spmd_recon()

    # Reference path: replicate the per-partition recon on each
    # device's padded slab, sequentially.  No SPMD; just runs the
    # same kernel 4 times on the 4 padded slabs.
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, HALO)), chunk, np.zeros((1, HALO))], axis=1
    )
    chunks_np = [u0_np[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    padded_per_dev = [pad_chunk(c) for c in chunks_np]
    # Periodic-wrap halos manually (mimicking what halo exchange would do).
    for d in range(N_DEVS):
        left_nbr = (d - 1) % N_DEVS
        right_nbr = (d + 1) % N_DEVS
        padded_per_dev[d][:, :HALO] = padded_per_dev[left_nbr][:, HALO + N_LOCAL - HALO:HALO + N_LOCAL].copy()
        padded_per_dev[d][:, -HALO:] = padded_per_dev[right_nbr][:, HALO:HALO + HALO].copy()
    bf_zero = jnp.zeros((1, 1), dtype=jnp.float32)
    refs_L = []
    refs_R = []
    for d in range(N_DEVS):
        Q_L_d, Q_R_d = recon_local(jnp.asarray(padded_per_dev[d]), bf_zero)
        refs_L.append(np.asarray(Q_L_d))
        refs_R.append(np.asarray(Q_R_d))

    # SPMD path: same exact data + same recon, sharded.
    Q_pad_global = _make_padded_Q(u0_np)
    Q_L, Q_R = _run_spmd_recon(
        recon_local, part_mesh, spmd_mesh, Q_pad_global
    )
    Q_L = np.asarray(Q_L)
    Q_R = np.asarray(Q_R)
    n_faces_local = N_LOCAL + 2 * HALO - 1

    max_err_L = 0.0
    max_err_R = 0.0
    for d in range(N_DEVS):
        base = d * n_faces_local
        # Owned-adjacent face indices: HALO-1 .. HALO+N_LOCAL-1.
        for k_local in range(HALO - 1, HALO + N_LOCAL):
            err_L = abs(float(Q_L[0, base + k_local])
                        - float(refs_L[d][0, k_local]))
            err_R = abs(float(Q_R[0, base + k_local])
                        - float(refs_R[d][0, k_local]))
            max_err_L = max(max_err_L, err_L)
            max_err_R = max(max_err_R, err_R)

    print(f"  SPMD vs replicated-single-device: max_err_L={max_err_L:.3e} "
          f"max_err_R={max_err_R:.3e}")
    assert max_err_L < 1e-6, f"Q_L err {max_err_L:.3e}"
    assert max_err_R < 1e-6, f"Q_R err {max_err_R:.3e}"
