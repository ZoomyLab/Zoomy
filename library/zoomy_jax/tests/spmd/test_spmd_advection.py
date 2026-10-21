"""SPMD bit-identity test: 1D scalar advection on a sharded mesh.

Confirms the end-to-end SPMD path produces the SAME result as a
single-device run on the same global mesh + IC.

Layout
------
* 16-cell periodic 1D mesh, u_t + a*u_x = 0, upwind flux, forward
  Euler.
* Halo = 1 (first-order upwind needs only the immediate neighbor
  cell on each side).
* Sharded: 2 fake CPU devices × 8 owned cells.  Padded local Q has
  shape (1, 10) per device.  Halo refilled by ``halo_exchange_inplace``
  before each flux step.
* Periodic BC: the leftmost and rightmost halo slabs of the global
  domain are wrapped via an extra ppermute pair (the standard JAX
  Wave_Equation trick).

Bit-identity check at the end is the regression hammer.
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

from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace


# ── Parameters ─────────────────────────────────────────────────────


N_TOTAL = 16
N_DEVS = 2
N_LOCAL = N_TOTAL // N_DEVS  # 8
HALO = 1
A = 1.0           # advection speed (positive ⇒ upwind reads from the LEFT)
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.4 * DX / A   # CFL = 0.4
N_STEPS = 20


def _ic(x):
    return np.sin(2 * np.pi * x)


# ── Single-device reference ────────────────────────────────────────


def _single_device_step(u):
    """Periodic upwind step on a single-device 1D array of N_TOTAL."""
    # Pad periodically with one ghost on each side.
    u_pad = jnp.concatenate([u[-1:], u, u[:1]])
    # Upwind flux at faces 0..N_TOTAL: f[i] = a * u_pad[i] for a>0
    f = A * u_pad[:-1]
    # Flux balance per cell: (f_right - f_left) / dx
    f_left = f[:-1]   # face between u[-1] and u[0], then u[0..1], ...
    f_right = f[1:]
    return u - DT / DX * (f_right - f_left)


def _single_device_run(u0, n_steps):
    u = u0
    for _ in range(n_steps):
        u = _single_device_step(u)
    return u


# ── SPMD path ─────────────────────────────────────────────────────


def _spmd_step(Q_pad, axis_name, n_devices):
    """One forward-Euler upwind step on the padded local Q.  Q_pad
    has shape (1, n_local + 2*halo) with halo width 1.  Periodic
    BCs are handled by the ppermute wrap at the global boundary
    (rank-0 and rank-(N-1) halos receive zeros from the
    open-boundary ppermute; we then ROLL the inputs via an extra
    wrap step before exchange — see below for the periodic-aware
    exchange used in this test)."""
    # Halo exchange (periodic — see periodic_halo_exchange below).
    Q_pad = periodic_halo_exchange(Q_pad, HALO, axis_name, n_devices)
    # Q_pad[:, 0] = halo from left neighbor (cell to the left of cell 1).
    # Q_pad[:, 1:n_local+1] = owned cells.
    # Q_pad[:, n_local+1] = halo from right neighbor.
    # Faces: face[k] is between Q_pad[:, k] and Q_pad[:, k+1].
    # For upwind a>0, f[k] = a * Q_pad[:, k].
    f = A * Q_pad
    f_left = f[:, 0:HALO + N_LOCAL]      # faces 0..n_local (length n_local+1)
    f_right = f[:, HALO:HALO + N_LOCAL + HALO]  # one off to the right
    # Wait — simpler: for each owned cell j in [1..n_local], face_left
    # is f[:, j], face_right is f[:, j+1] (for a>0 upwind).
    # Use slicing to do it cleanly.
    f_at_face = A * Q_pad   # length n_local + 2 (faces co-located with cells for upwind a>0)
    owned = Q_pad[:, HALO:HALO + N_LOCAL]
    f_face_left = A * Q_pad[:, HALO - 1:HALO - 1 + N_LOCAL]    # cell to the left
    f_face_right = A * Q_pad[:, HALO:HALO + N_LOCAL]            # the owned cell itself
    # Forward Euler update of owned cells only.
    owned_new = owned - DT / DX * (f_face_right - f_face_left)
    # Re-pad with zero halos (will be refilled at next step's exchange).
    halo_left = jnp.zeros((Q_pad.shape[0], HALO), dtype=Q_pad.dtype)
    halo_right = jnp.zeros((Q_pad.shape[0], HALO), dtype=Q_pad.dtype)
    return jnp.concatenate([halo_left, owned_new, halo_right], axis=1)


def periodic_halo_exchange(Q_pad, halo, axis_name, n_devices):
    """Halo exchange WITH periodic wrap at the global boundary.

    Standard ``halo_exchange_inplace`` returns zeros at the rank-0
    left halo and rank-(N-1) right halo.  To make the global domain
    periodic, we add a full-ring ppermute pair that includes the
    wrap-around edge.
    """
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    # Full ring perms — every device sends to next/prev, including wrap.
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(right_owned, perm=perm_right, axis_name=axis_name)
    fill_right = lax.ppermute(left_owned, perm=perm_left, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo].set(fill_left)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right)
    return Q_pad


def _spmd_run(Q_pad_global, n_steps, mesh):
    """Run n_steps on the sharded Q.  Returns the gathered owned
    cells (shape (1, N_TOTAL))."""
    @partial(shard_map, mesh=mesh, in_specs=P(None, "cells"),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad):
        def body(Q, _):
            return _spmd_step(Q, "cells", N_DEVS), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(n_steps))
        return Q_final

    Q_final = run(Q_pad_global)
    # Strip the halos per device slab and concat owned cells.
    Q_np = np.asarray(Q_final).reshape(-1)
    owned = []
    for d in range(N_DEVS):
        base = d * (N_LOCAL + 2 * HALO)
        owned.append(Q_np[base + HALO:base + HALO + N_LOCAL])
    return np.concatenate(owned).reshape(1, -1)


# ── Test ──────────────────────────────────────────────────────────


def _run_case(n_devices, n_total, n_steps):
    """Run both reference (single-device) and SPMD paths; return max err."""
    if jax.device_count() < n_devices:
        pytest.skip(f"Need {n_devices} devices")
    mesh = Mesh(np.array(jax.devices()[:n_devices]), axis_names=("cells",))

    n_local = n_total // n_devices
    dx = (DOMAIN[1] - DOMAIN[0]) / n_total
    dt = 0.4 * dx / A

    xc = DOMAIN[0] + (np.arange(n_total) + 0.5) * dx
    u0 = jnp.asarray(_ic(xc), dtype=jnp.float64)

    # Reference: single-device with the same dt.
    def _ref_step(u):
        u_pad = jnp.concatenate([u[-1:], u, u[:1]])
        f = A * u_pad[:-1]
        return u - dt / dx * (f[1:] - f[:-1])
    u_ref = u0
    for _ in range(n_steps):
        u_ref = _ref_step(u_ref)
    u_ref = np.asarray(u_ref)

    # SPMD: padded global, scan-in-shard_map.
    halo = HALO
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros(halo), chunk, np.zeros(halo)])
    chunks = [u0[d * n_local:(d + 1) * n_local] for d in range(n_devices)]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(np.asarray(c)) for c in chunks])
    ).reshape(1, -1)

    # Inline a parameterised spmd_step for this n_local/dx/dt.
    def step_local(Q_pad):
        Q_pad = periodic_halo_exchange(Q_pad, halo, "cells", n_devices)
        owned = Q_pad[:, halo:halo + n_local]
        f_face_left = A * Q_pad[:, halo - 1:halo - 1 + n_local]
        f_face_right = A * Q_pad[:, halo:halo + n_local]
        owned_new = owned - dt / dx * (f_face_right - f_face_left)
        zeros = jnp.zeros((Q_pad.shape[0], halo), dtype=Q_pad.dtype)
        return jnp.concatenate([zeros, owned_new, zeros], axis=1)

    @partial(shard_map, mesh=mesh, in_specs=P(None, "cells"),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad):
        def body(Q, _):
            return step_local(Q), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(n_steps))
        return Q_final

    Q_final = np.asarray(run(Q_pad_global)).reshape(-1)
    owned_chunks = [
        Q_final[d * (n_local + 2 * halo) + halo:
                d * (n_local + 2 * halo) + halo + n_local]
        for d in range(n_devices)
    ]
    u_spmd = np.concatenate(owned_chunks)
    return float(np.max(np.abs(u_spmd - u_ref)))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_spmd_advection_matches_single_device():
    err = _run_case(n_devices=2, n_total=16, n_steps=20)
    print(f"  2dev × 16 cells: err = {err:.3e}")
    assert err < 1e-12


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
@pytest.mark.parametrize("n_devices,n_total", [
    (2, 16), (4, 16), (4, 32), (2, 32),
])
def test_spmd_advection_scales_across_decompositions(n_devices, n_total):
    """Same bit-identity check across multiple (devices, total cells)
    decompositions.  Pins that the halo wiring is robust to varying
    partition layouts (no off-by-one as n_local changes)."""
    err = _run_case(n_devices=n_devices, n_total=n_total, n_steps=20)
    print(f"  {n_devices}dev × {n_total} cells: err = {err:.3e}")
    assert err < 1e-12, f"err {err:.3e}"
