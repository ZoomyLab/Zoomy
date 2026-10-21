"""Standalone unit test of the SPMD halo-exchange primitive.

Confirms that ``halo_exchange_inplace`` correctly fills the halo
slabs from neighboring devices on a fake 4-CPU mesh.

Setup:
  * 16 owned cells split across 4 devices → 4 cells per device.
  * Halo width 1: each device has a padded array of length 6.
  * Initial: owned cells = arange(16) split into 4 contiguous chunks.
    Halo slabs zero.
  * After exchange: each device's left halo holds the right-most
    owned cell of its left neighbor; right halo holds the left-most
    owned cell of its right neighbor.  Edge devices keep zero in the
    outer halo.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import os
# Must be set before jax imports.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace


# ── Helpers ─────────────────────────────────────────────────────────


def _4_device_mesh():
    if jax.device_count() < 4:
        pytest.skip(
            f"Need 4 devices, got {jax.device_count()}. "
            "Run with XLA_FLAGS=--xla_force_host_platform_device_count=4."
        )
    return Mesh(np.array(jax.devices()[:4]), axis_names=("cells",))


# ── Tests ───────────────────────────────────────────────────────────


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_halo_exchange_4_devices_halo1():
    """4 ranks × 4 cells, halo=1.  Verifies interior halos receive
    neighbor edge cell; boundary halos remain zero (no wrap)."""
    mesh = _4_device_mesh()
    n_local = 4
    halo = 1
    n_total = n_local * 4  # 16

    # Global owned-cell field (one variable for clarity).
    owned = jnp.arange(n_total, dtype=jnp.float64).reshape(1, n_total)

    # Build padded array: each device's slab is (1, n_local + 2*halo).
    # Across 4 devices the SHARDED layout is (1, 4*(n_local+2*halo)) = (1, 24).
    # We construct the padded global by inserting zeros into the halo
    # slots between each owned chunk.
    pad_chunk = lambda chunk: jnp.concatenate(
        [jnp.zeros((1, halo)), chunk, jnp.zeros((1, halo))], axis=1)
    chunks = [owned[:, i * n_local:(i + 1) * n_local] for i in range(4)]
    Q_pad_global = jnp.concatenate([pad_chunk(c) for c in chunks], axis=1)
    assert Q_pad_global.shape == (1, 4 * (n_local + 2 * halo))  # (1, 24)

    @partial(shard_map, mesh=mesh, in_specs=P(None, "cells"),
             out_specs=P(None, "cells"), check_rep=False)
    def do_exchange(Q_pad_local):
        return halo_exchange_inplace(
            Q_pad_local, halo=halo, axis_name="cells", n_devices=4)

    out = do_exchange(Q_pad_global)
    out_np = np.asarray(out).reshape(-1)

    # Expected layout: per-device [left_halo, owned×4, right_halo].
    # Device d's owned cells are arange(d*4, d*4+4).
    # Device d's left_halo  = device(d-1)'s right-most owned = (d*4 - 1)
    #                         except d=0 where it stays zero.
    # Device d's right_halo = device(d+1)'s left-most owned = ((d+1)*4)
    #                         except d=3 where it stays zero.
    expected = np.zeros(24, dtype=np.float64)
    for d in range(4):
        base = d * 6  # device slab start
        # left halo
        expected[base + 0] = 0.0 if d == 0 else d * 4 - 1
        # owned
        for k in range(4):
            expected[base + 1 + k] = d * 4 + k
        # right halo
        expected[base + 5] = 0.0 if d == 3 else (d + 1) * 4

    np.testing.assert_array_equal(out_np, expected)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_halo_exchange_2_devices_halo2():
    """2 ranks × 8 cells, halo=2.  Verifies wider halos."""
    if jax.device_count() < 2:
        pytest.skip("Need at least 2 devices")
    mesh2 = Mesh(np.array(jax.devices()[:2]), axis_names=("cells",))
    n_local = 8
    halo = 2
    n_total = n_local * 2

    owned = jnp.arange(n_total, dtype=jnp.float64).reshape(1, n_total)
    pad_chunk = lambda chunk: jnp.concatenate(
        [jnp.zeros((1, halo)), chunk, jnp.zeros((1, halo))], axis=1)
    chunks = [owned[:, i * n_local:(i + 1) * n_local] for i in range(2)]
    Q_pad_global = jnp.concatenate([pad_chunk(c) for c in chunks], axis=1)

    @partial(shard_map, mesh=mesh2, in_specs=P(None, "cells"),
             out_specs=P(None, "cells"), check_rep=False)
    def do_exchange(Q_pad_local):
        return halo_exchange_inplace(
            Q_pad_local, halo=halo, axis_name="cells", n_devices=2)

    out = np.asarray(do_exchange(Q_pad_global)).reshape(-1)

    # Per-device slab: (n_local + 2*halo) = 12 cells.  Total = 24.
    # Device 0: [0, 0, 0..7, 8, 9]   (left halo: 0/0; right halo: 8/9)
    # Device 1: [6, 7, 8..15, 0, 0]  (left halo: 6/7; right halo: 0/0)
    expected = np.array([
        # Device 0
        0.0, 0.0,   0, 1, 2, 3, 4, 5, 6, 7,   8.0, 9.0,
        # Device 1
        6.0, 7.0,   8, 9, 10, 11, 12, 13, 14, 15,   0.0, 0.0,
    ], dtype=np.float64)
    np.testing.assert_array_equal(out, expected)
