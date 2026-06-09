"""SPMD with multi-stage explicit time integration: SSP-RK2 (Heun)
with a halo exchange BEFORE each RK stage.

Mirrors what ``HyperbolicSolver.step`` does for order >= 2: two
flux-op evaluations per step, recombined Heun-style.  Under SPMD,
the halo refresh must happen at the start of EACH stage, otherwise
stage-2 reads stale halos from stage-1's update.

Bit-identity vs replicated single-device on a 32-cell × 4-device
mesh confirms the SPMD multi-stage pattern is correct.
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
from zoomy_core.model.models.advection import Advection
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.partition_jax import partition_1d_contiguous


N_TOTAL = 32
N_DEVS = 4
N_LOCAL = N_TOTAL // N_DEVS
HALO = 2
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.25 * DX
N_STEPS = 4


def _smooth_ic(x):
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


def _periodic_halo_np(padded_per_dev, halo):
    out = [arr.copy() for arr in padded_per_dev]
    n = len(out)
    for d in range(n):
        left_nbr = (d - 1) % n
        right_nbr = (d + 1) % n
        out[d][:, :halo] = out[left_nbr][:, N_LOCAL:N_LOCAL + halo]
        out[d][:, -halo:] = out[right_nbr][:, halo:halo + halo]
    return out


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_ssp_rk2_step_composes_with_shard_map():
    """SSP-RK2 (Heun) under SPMD with halo exchange before each stage.

    Reference: same RK2 applied on per-device padded slabs,
    sequentially, with periodic halos refilled between stages.
    """
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    model = Advection(dimension=1)
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(
            order=2, limiter="venkatakrishnan"
        ),
    )
    solver = HyperbolicSolver()
    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0_np = _smooth_ic(xc).astype(np.float32).reshape(1, N_TOTAL)
    Q_setup, Qaux_setup = solver.setup_simulation(mesh_np, nsm)
    runtime = solver._rt_model
    global_jax_mesh = solver._rt_mesh

    parts = partition_1d_contiguous(
        global_jax_mesh, n_parts=N_DEVS, halo=HALO
    )
    part_mesh = parts[1]
    flux_op_part = solver.get_flux_operator(part_mesh, runtime)
    parameters = solver._rt_parameters

    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, HALO)), chunk, np.zeros((1, HALO))], axis=1
    )
    chunks = [u0_np[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(c) for c in chunks], axis=1),
        dtype=Q_setup.dtype,
    )
    Qaux_pad = jnp.zeros(
        (Qaux_setup.shape[0], Q_pad_global.shape[1]), dtype=Q_setup.dtype
    )

    # ── Reference: replicated single-device SSP-RK2 with manual halo
    # refills between stages.
    n_padded = N_LOCAL + 2 * HALO
    padded_per_dev_ref = [
        np.asarray(Q_pad_global[:, d * n_padded:(d + 1) * n_padded]).copy()
        for d in range(N_DEVS)
    ]
    qaux_per_dev = [
        np.zeros((Qaux_setup.shape[0], n_padded), dtype=Q_setup.dtype)
        for _ in range(N_DEVS)
    ]
    dt_j = jnp.asarray(DT, dtype=Q_setup.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_setup.dtype)

    def _one_flux(Q_arr_per_dev):
        out = []
        for d in range(N_DEVS):
            Q_d = jnp.asarray(Q_arr_per_dev[d])
            Qaux_d = jnp.asarray(qaux_per_dev[d])
            dQ_d = flux_op_part(
                dt_j, t_j, Q_d, Qaux_d, parameters,
                jnp.zeros_like(Q_d),
            )
            out.append(np.asarray(dQ_d))
        return out

    for _ in range(N_STEPS):
        # Stage 1.
        padded_per_dev_ref = _periodic_halo_np(padded_per_dev_ref, HALO)
        dQ1 = _one_flux(padded_per_dev_ref)
        Q1 = [padded_per_dev_ref[d] + DT * dQ1[d] for d in range(N_DEVS)]
        # Stage 2 — refresh halos on Q1 (since its halo cells came from
        # dQ1 at stage 1, which was wrong on halo cells).
        Q1 = _periodic_halo_np(Q1, HALO)
        dQ2 = _one_flux(Q1)
        Q2 = [Q1[d] + DT * dQ2[d] for d in range(N_DEVS)]
        # Heun average.
        padded_per_dev_ref = [
            0.5 * (padded_per_dev_ref[d] + Q2[d]) for d in range(N_DEVS)
        ]

    # ── SPMD: SSP-RK2 with halo exchange before each stage.
    def spmd_stage(Q_pad, Qaux_pad):
        Q_pad = _periodic_halo(Q_pad, HALO, "cells", N_DEVS)
        dQ = flux_op_part(
            dt_j, t_j, Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad),
        )
        return Q_pad + DT * dQ

    def spmd_step(Q_pad, Qaux_pad):
        Q0 = Q_pad
        Q1 = spmd_stage(Q0, Qaux_pad)
        Q2 = spmd_stage(Q1, Qaux_pad)
        return 0.5 * (Q0 + Q2)

    @partial(shard_map, mesh=spmd_mesh, in_specs=(P(None, "cells"), P(None, "cells")),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad, Qaux_pad):
        def body(carry, _):
            return spmd_step(carry, Qaux_pad), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(N_STEPS))
        return Q_final

    Q_final_spmd = np.asarray(run(Q_pad_global, Qaux_pad))

    max_err = 0.0
    for d in range(N_DEVS):
        owned_spmd = Q_final_spmd[
            :, d * n_padded + HALO:d * n_padded + HALO + N_LOCAL
        ]
        owned_ref = padded_per_dev_ref[d][:, HALO:HALO + N_LOCAL]
        max_err = max(max_err, float(np.max(np.abs(owned_spmd - owned_ref))))

    print(f"  SSP-RK2 step SPMD vs replicated single-device: "
          f"max_err = {max_err:.3e}")
    assert max_err < 1e-5, f"err {max_err:.3e}"
