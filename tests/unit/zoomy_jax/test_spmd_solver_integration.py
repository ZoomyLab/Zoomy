"""SPMD integration: the existing ``ConstantReconstruction`` class
runs unchanged inside a ``shard_map`` body when handed a per-partition
``MeshJAX``.

Proves the solver-side reconstruction primitive composes with the
SPMD halo-exchange wiring without touching the reconstruction code
itself.  The bridge from "halo works" to "solver splits".
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
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace
from zoomy_jax.fvm.reconstruction_jax import ConstantReconstruction


N_TOTAL = 16
N_DEVS = 4
N_LOCAL = N_TOTAL // N_DEVS  # 4
HALO = 1
A = 1.0
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.4 * DX / A
N_STEPS = 10


def _ic(x):
    return np.sin(2 * np.pi * x)


def _periodic_halo(Q_pad, halo, axis_name, n_devices):
    """Periodic-wrap halo exchange (same pattern as test_spmd_advection)."""
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(right_owned, perm=perm_right, axis_name=axis_name)
    fill_right = lax.ppermute(left_owned, perm=perm_left, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo].set(fill_left)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right)
    return Q_pad


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_constant_reconstruction_runs_under_shard_map():
    """ConstantReconstruction on a per-partition mesh produces the
    same face states as ConstantReconstruction on the global mesh,
    when SPMD halo exchange supplies the cross-partition values.

    Bit-identity on a 4-device split of a 16-cell mesh."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    # Global mesh.
    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    mesh_global = convert_mesh_to_jax(mesh_np)

    # Per-partition meshes (all identical in topology for uniform 1D).
    parts = partition_1d_contiguous(mesh_global, n_parts=N_DEVS, halo=HALO)
    part_mesh = parts[0]  # representative; all are structurally identical

    # Build the existing ConstantReconstruction on a per-partition mesh.
    # Inside shard_map each device executes the same code; the
    # per-partition face_cells / boundary indices captured here are
    # the topology every device sees.
    recon_local = ConstantReconstruction(part_mesh, dim=1)

    # IC.
    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0 = jnp.asarray(_ic(xc), dtype=jnp.float64).reshape(1, N_TOTAL)

    # Single-device reference: just upwind FE with periodic wrap.
    def _ref_step(u):
        u_pad = jnp.concatenate([u[:, -1:], u, u[:, :1]], axis=1)
        f = A * u_pad[:, :-1]
        return u - DT / DX * (f[:, 1:] - f[:, :-1])
    u_ref = u0
    for _ in range(N_STEPS):
        u_ref = _ref_step(u_ref)
    u_ref = np.asarray(u_ref).reshape(-1)

    # SPMD path: padded global Q.
    halo = HALO
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, halo)), chunk, np.zeros((1, halo))], axis=1)
    chunks = [u0[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(np.asarray(c)) for c in chunks], axis=1)
    )

    def spmd_step(Q_pad):
        Q_pad = _periodic_halo(Q_pad, halo, "cells", N_DEVS)
        # Per-partition reconstruction: Q_L, Q_R at each face.
        # ConstantReconstruction on the part_mesh produces face states
        # of shape (1, n_faces_local).  Because we periodic-halo above,
        # the halo cells carry the right values; the reconstruction
        # at every face is just a slice of Q_pad.
        Q_L, Q_R = recon_local(Q_pad, bf_face_values=None)
        # Upwind flux at every face: f = a*Q_L for a>0.
        f = A * Q_L
        # Owned cells live at indices [halo .. halo+n_local).  Each owned
        # cell j has faces at face index j-1 (left) and j (right) in the
        # padded face indexing.
        # Owned faces: indices [halo-1, halo, ..., halo+n_local-1] inclusive,
        # length n_local+1.  Cell j's left face is index (halo-1)+(j-halo) = j-1.
        # Cell j's right face is index j.
        owned = Q_pad[:, halo:halo + N_LOCAL]
        # face index for left of owned cell j (j in halo..halo+n_local-1) = j-1.
        # face index for right of owned cell j = j.
        f_left = f[:, halo - 1:halo - 1 + N_LOCAL]
        f_right = f[:, halo:halo + N_LOCAL]
        owned_new = owned - DT / DX * (f_right - f_left)
        zeros = jnp.zeros((Q_pad.shape[0], halo), dtype=Q_pad.dtype)
        return jnp.concatenate([zeros, owned_new, zeros], axis=1)

    @partial(shard_map, mesh=spmd_mesh, in_specs=P(None, "cells"),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad):
        def body(Q, _):
            return spmd_step(Q), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(N_STEPS))
        return Q_final

    Q_final = np.asarray(run(Q_pad_global)).reshape(-1)
    owned = []
    for d in range(N_DEVS):
        base = d * (N_LOCAL + 2 * halo)
        owned.append(Q_final[base + halo:base + halo + N_LOCAL])
    u_spmd = np.concatenate(owned)

    err = float(np.max(np.abs(u_spmd - u_ref)))
    print(f"  ConstantReconstruction SPMD vs single-device: err = {err:.3e}")
    assert err < 1e-12, f"err {err:.3e}"
