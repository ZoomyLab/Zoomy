"""SPMD smoke test: MalpassetSWE on a tiny 2D mesh under shard_map.

Validates that the full Malpasset viscous SWE flux op composes with
partition_xaxis_structured + halo_exchange.  Bit-identity vs
replicated single-device, ONE step.  Target: < 1 min."""
from __future__ import annotations

import os
import sys
import time
from functools import partial

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malpasset_swe_model import MalpassetSWE  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.models.system_model import SystemModel  # noqa: E402
from zoomy_core.numerics import NumericalSystemModel  # noqa: E402
from zoomy_core.numerics.numerical_system_model import (
    ReconstructionSpec,
)  # noqa: E402
from zoomy_jax.fvm.solver_jax import HyperbolicSolver  # noqa: E402
from zoomy_jax.mesh.partition_jax import (
    partition_xaxis_structured,
)  # noqa: E402


def _periodic_halo(Q_pad, halo_cells, axis_name, n_devices):
    left_owned = Q_pad[:, halo_cells:2 * halo_cells]
    right_owned = Q_pad[:, -2 * halo_cells:-halo_cells]
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(
        right_owned, perm=perm_right, axis_name=axis_name
    )
    fill_right = lax.ppermute(
        left_owned, perm=perm_left, axis_name=axis_name
    )
    Q_pad = Q_pad.at[:, :halo_cells].set(fill_left)
    Q_pad = Q_pad.at[:, -halo_cells:].set(fill_right)
    return Q_pad


def _periodic_halo_np(padded_per_dev, halo_cells, n_local_cells):
    out = [arr.copy() for arr in padded_per_dev]
    n = len(out)
    for d in range(n):
        left = (d - 1) % n
        right = (d + 1) % n
        out[d][:, :halo_cells] = out[left][
            :, n_local_cells:n_local_cells + halo_cells
        ]
        out[d][:, -halo_cells:] = out[right][:, halo_cells:halo_cells + halo_cells]
    return out


def main():
    if jax.device_count() < 4:
        raise RuntimeError(
            f"Need 4 devices; got {jax.device_count()}. "
            f"Set XLA_FLAGS='--xla_force_host_platform_device_count=4'."
        )

    NX, NY = 16, 4
    N_DEVS = 4
    N_LOCAL_X = NX // N_DEVS    # 4
    HALO_X = 1
    DOMAIN = (0.0, 1600.0, 0.0, 400.0)
    DT = 0.04

    t0 = time.perf_counter()
    mesh_np = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
    model = MalpassetSWE()
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(order=1),
    )
    solver = HyperbolicSolver()
    Q, Qaux = solver.setup_simulation(mesh_np, nsm)
    print(f"[t={time.perf_counter() - t0:6.2f}s] setup done — n_cells="
          f"{int(solver._rt_mesh.n_inner_cells)}, Q.shape={Q.shape}")

    # Lake-at-rest IC.
    Q = Q.at[0].set(0.0)
    Q = Q.at[1].set(10.0)
    Q = Q.at[2].set(0.0)
    Q = Q.at[3].set(0.0)
    Qaux = solver.update_qaux(
        Q, Qaux, Q, Qaux, solver._rt_mesh, solver._rt_model,
        solver._rt_parameters, 0.0, 1.0,
    )

    # Partition.
    parts = partition_xaxis_structured(
        solver._rt_mesh, n_parts=N_DEVS, halo=HALO_X,
        domain=DOMAIN, shape=(NX, NY),
    )
    part_mesh = parts[1]   # interior partition
    print(f"[t={time.perf_counter() - t0:6.2f}s] partitioned: "
          f"{[int(p.n_inner_cells) for p in parts]} cells per part")

    # Build per-partition flux op via the existing solver method.
    flux_op_part = solver.get_flux_operator(part_mesh, solver._rt_model)
    parameters = solver._rt_parameters

    x_stride = NY
    n_local_cells = N_LOCAL_X * x_stride
    halo_cells = HALO_X * x_stride
    n_padded_cells = n_local_cells + 2 * halo_cells

    # Build per-device padded Q (and Qaux).
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((4, halo_cells)), chunk, np.zeros((4, halo_cells))],
        axis=1,
    )
    chunks = [
        np.asarray(Q[:, d * n_local_cells:(d + 1) * n_local_cells])
        for d in range(N_DEVS)
    ]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(c) for c in chunks], axis=1),
        dtype=Q.dtype,
    )
    # Qaux per device (need 1 aux variable).
    Qaux_chunks = [
        np.asarray(Qaux[:, d * n_local_cells:(d + 1) * n_local_cells])
        for d in range(N_DEVS)
    ]
    pad_chunk_aux = lambda chunk: np.concatenate(
        [np.zeros((1, halo_cells)), chunk, np.zeros((1, halo_cells))],
        axis=1,
    )
    Qaux_pad_global = jnp.asarray(
        np.concatenate([pad_chunk_aux(c) for c in Qaux_chunks], axis=1),
        dtype=Q.dtype,
    )

    print(f"[t={time.perf_counter() - t0:6.2f}s] Q_pad.shape="
          f"{Q_pad_global.shape}")

    # ── Reference: per-device flux op applied sequentially.
    pad_list = [
        np.asarray(Q_pad_global[:, d * n_padded_cells:(d + 1) * n_padded_cells])
        .copy() for d in range(N_DEVS)
    ]
    aux_list = [
        np.asarray(Qaux_pad_global[:, d * n_padded_cells:(d + 1) * n_padded_cells])
        .copy() for d in range(N_DEVS)
    ]
    pad_list = _periodic_halo_np(pad_list, halo_cells, n_local_cells)
    dt_j = jnp.asarray(DT, dtype=Q.dtype)
    t_j = jnp.asarray(0.0, dtype=Q.dtype)
    t0_ref = time.perf_counter()
    ref_new = []
    for d in range(N_DEVS):
        Q_d = jnp.asarray(pad_list[d])
        Qaux_d = jnp.asarray(aux_list[d])
        dQ_d = flux_op_part(
            dt_j, t_j, Q_d, Qaux_d, parameters, jnp.zeros_like(Q_d),
        )
        ref_new.append(np.asarray(Q_d + DT * dQ_d))
    print(f"[t={time.perf_counter() - t0:6.2f}s] reference done "
          f"(compile {time.perf_counter() - t0_ref:.2f}s)")

    # ── SPMD.
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    def spmd_step(Q_pad, Qaux_pad):
        Q_pad = _periodic_halo(Q_pad, halo_cells, "cells", N_DEVS)
        dQ = flux_op_part(
            dt_j, t_j, Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad),
        )
        return Q_pad + DT * dQ

    @partial(shard_map, mesh=spmd_mesh,
             in_specs=(P(None, "cells"), P(None, "cells")),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad, Qaux_pad):
        return spmd_step(Q_pad, Qaux_pad)

    t0_spmd = time.perf_counter()
    Q_spmd = np.asarray(run(Q_pad_global, Qaux_pad_global))
    print(f"[t={time.perf_counter() - t0:6.2f}s] SPMD done "
          f"(compile {time.perf_counter() - t0_spmd:.2f}s)")

    # Bit-identity on owned cells.
    max_err = 0.0
    for d in range(N_DEVS):
        owned_spmd = Q_spmd[
            :, d * n_padded_cells + halo_cells:
               d * n_padded_cells + halo_cells + n_local_cells
        ]
        owned_ref = ref_new[d][:, halo_cells:halo_cells + n_local_cells]
        max_err = max(max_err, float(np.max(np.abs(owned_spmd - owned_ref))))

    print(f"\nMalpasset SWE viscous SPMD vs replicated single-device:")
    print(f"  max_err = {max_err:.3e}")
    print(f"  total wall = {time.perf_counter() - t0:.2f}s")
    assert max_err < 1e-5, f"bit-identity failed: {max_err:.3e}"
    print("SPMD SMOKE PASSED")


if __name__ == "__main__":
    main()
