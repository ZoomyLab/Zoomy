"""JAX-native halo exchange for SPMD-sharded FV solvers.

Pattern copied from JAX's `cloud_tpu_colabs/Wave_Equation.ipynb` and
used in production by JAX-Fluids / Autodesk XLB.  No MPI dependency;
just `jax.lax.ppermute` along a named axis inside `shard_map`.

Storage layout
--------------
Each device holds a **padded** local Q of shape ``(n_var, n_local +
2*halo)``:

  [  halo_left  |  n_local owned cells  |  halo_right  ]
     |←  halo →|                        |←  halo →|

`halo_exchange_inplace` pulls the **owned** edge slab of width
`halo` from each side, ``ppermute``s it to the neighbor, and writes
it into the neighbor's halo slab on the opposite side.  Devices at
the domain boundary receive zeros in their outer halo (free
boundary); the inline BC evaluator on the solver side overwrites
those slots with the BC-evaluated face value, so the same kernel
runs everywhere — no Python branching inside the JIT trace.

The cell axis is always the *last* axis of Q (`(n_var, n_cells)`).

Composition with the existing solver
------------------------------------
Already wired (see `tests/unit/zoomy_jax/`):
  * `test_halo_exchange.py`  — bit-correct halo on 2/4 devices.
  * `test_spmd_advection.py` — bit-identity scalar advection on
    {2,4} devices × {16,32} cells vs single-device.
  * `test_partition_jax.py`  — `partition_1d_contiguous` chops a
    global `MeshJAX` into per-partition padded slabs with shifted
    face indices.
  * `test_spmd_solver_integration.py` — existing
    `ConstantReconstruction` composes with SPMD shard_map when
    handed a per-partition mesh; bit-identity vs single-device.

Open work to lift this into the full `HyperbolicSolver`:
  1. Rebuild LSQ stencils per partition (`LSQMUSCLReconstructionJAX`
     consumes `mesh.lsq_gradQ` / `mesh.lsq_neighbors` /
     `mesh.lsq_boundary_face_neighbors` — partition_jax leaves
     these empty.  Re-run `LSQMesh._build_lsq_stencil` on the
     padded coordinate system of each partition.)
  2. Drop the periodic-wrap demo path and use the inline BC kernel
     at global boundaries (rank-0 left face + rank-(N-1) right face).
     The partition's `boundary_face_*` arrays already carry exactly
     these faces; the existing flux operator's BC fori_loop runs over
     them per shard.
  3. Wrap `HyperbolicSolver.step` in ``shard_map(... in_specs=
     P(None, "cells"), out_specs=P(None, "cells"))`` with the
     halo exchange called inside the body before the flux operator.
  4. For multi-process deployment: `jax.distributed.initialize()`
     at process startup (SLURM auto-detection).  Dev loop runs in
     one process via `XLA_FLAGS=--xla_force_host_platform_device_count=N`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _send_left(x_halo_slab, axis_name, n_devices):
    """Send the halo slab to the LEFT neighbor (lower rank); the
    device at rank 0 receives zeros."""
    perm = [(i, (i - 1) % n_devices) for i in range(1, n_devices)]
    return lax.ppermute(x_halo_slab, perm=perm, axis_name=axis_name)


def _send_right(x_halo_slab, axis_name, n_devices):
    """Send the halo slab to the RIGHT neighbor (higher rank); the
    device at rank N-1 receives zeros."""
    perm = [(i, (i + 1) % n_devices) for i in range(n_devices - 1)]
    return lax.ppermute(x_halo_slab, perm=perm, axis_name=axis_name)


def halo_exchange_inplace(Q_pad, halo, axis_name, n_devices):
    """[Surrounding layout] Refill the halo slabs of `Q_pad` with
    neighbor data via `lax.ppermute`.

    `Q_pad` layout: `[halo_left | n_local owned | halo_right]`.

    Used by the demo SPMD advection / partition tests.  For solver-
    side LSQ-MUSCL composition (where the existing reconstruction
    iterates `jnp.arange(n_inner_cells)` from index 0), use
    `halo_exchange_owned_first` instead — its layout matches
    `LSQMesh`'s convention (ghosts at the end).
    """
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    fill_left_halo = _send_right(right_owned, axis_name, n_devices)
    fill_right_halo = _send_left(left_owned, axis_name, n_devices)
    Q_pad = Q_pad.at[:, :halo].set(fill_left_halo)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right_halo)
    return Q_pad


def halo_exchange_owned_first(Q_pad, n_local, halo, axis_name, n_devices):
    """[Owned-first layout, matches LSQMesh convention] Refill the
    halo slabs of `Q_pad` via `lax.ppermute`.

    Layout:

    ```
        [  owned (n_local)  |  left_halo (halo)  |  right_halo (halo)  ]
           indices 0..n_local-1   n_local..n_local+halo-1   ...
    ```

    The owned region is at the **front** so the existing per-cell
    reconstruction (`jnp.arange(n_inner_cells)` over owned cells)
    works directly.  Halo cells live at the end of the array, mirroring
    how `LSQMesh` stores BC ghost cells at indices
    `[n_inner_cells .. n_cells)`.

    Send pattern:
      * MY leftmost slab `[0:halo]` → fills LEFT neighbor's
        right_halo.
      * MY rightmost slab `[n_local-halo:n_local]` → fills RIGHT
        neighbor's left_halo.

    Receive (after ppermute):
      * MY left_halo  = LEFT neighbor's rightmost owned slab.
      * MY right_halo = RIGHT neighbor's leftmost owned slab.

    At the global domain boundary the corresponding halo slab
    contains zeros — the inline BC kernel on the solver side
    overwrites them.
    """
    my_leftmost = Q_pad[:, 0:halo]
    my_rightmost = Q_pad[:, n_local - halo:n_local]
    # MY left_halo = LEFT neighbor's rightmost slab; data flows RIGHT
    # so I receive it via _send_right.
    fill_left_halo = _send_right(my_rightmost, axis_name, n_devices)
    # MY right_halo = RIGHT neighbor's leftmost slab; data flows LEFT.
    fill_right_halo = _send_left(my_leftmost, axis_name, n_devices)
    Q_pad = Q_pad.at[:, n_local:n_local + halo].set(fill_left_halo)
    Q_pad = Q_pad.at[:, n_local + halo:n_local + 2 * halo].set(fill_right_halo)
    return Q_pad
