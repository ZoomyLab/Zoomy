"""SPMD integration with the actual ``HyperbolicSolver`` flux
operator.  Demonstrates that ``solver.get_flux_operator(part_mesh,
runtime)`` produces a per-partition flux closure that composes
unchanged with ``shard_map`` when wrapped in halo exchange.

This is the milestone for "the solver is set up to split everything
up": no modification to ``HyperbolicSolver`` — just call
``get_flux_operator`` with a per-partition mesh and the same runtime
model that was built once for the global setup.

Parametrised on the reconstruction order:
  * order=1 (ConstantReconstruction, halo=1) — fast JIT (~4s).
  * order=2 (LSQ-MUSCL + Venkatakrishnan, halo=2) — slower JIT
    (~60s) but exercises the full second-order pipeline.
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
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.4 * DX
N_STEPS = 3


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
    """NumPy/JAX periodic halo refill (no SPMD) — for the replicated
    reference path."""
    out = [arr.copy() if hasattr(arr, "copy") else np.array(arr)
           for arr in padded_per_dev]
    n = len(out)
    for d in range(n):
        left_nbr = (d - 1) % n
        right_nbr = (d + 1) % n
        out[d][:, :halo] = out[left_nbr][:, N_LOCAL:N_LOCAL + halo]
        out[d][:, -halo:] = out[right_nbr][:, halo:halo + halo]
    return out


def _run_one_case(reconstruction_order: int, halo: int):
    """Build the solver + per-partition flux op at the given order,
    run N_STEPS in both SPMD and replicated-single-device modes, and
    return max |spmd − ref| over owned cells."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    model = Advection(dimension=1)
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(
            order=reconstruction_order, limiter="venkatakrishnan"
        ),
    )
    solver = HyperbolicSolver()
    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0_np = _smooth_ic(xc).astype(np.float32).reshape(1, N_TOTAL)
    Q_setup, Qaux_setup = solver.setup_simulation(mesh_np, nsm)
    runtime = solver._rt_model
    global_jax_mesh = solver._rt_mesh

    parts = partition_1d_contiguous(
        global_jax_mesh, n_parts=N_DEVS, halo=halo
    )
    part_mesh = parts[1]
    assert int(part_mesh.n_boundary_faces) == 0
    flux_op_part = solver.get_flux_operator(part_mesh, runtime)
    parameters = solver._rt_parameters

    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, halo)), chunk, np.zeros((1, halo))], axis=1
    )
    chunks = [u0_np[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(c) for c in chunks], axis=1),
        dtype=Q_setup.dtype,
    )
    Qaux_pad = jnp.zeros(
        (Qaux_setup.shape[0], Q_pad_global.shape[1]), dtype=Q_setup.dtype
    )

    n_padded = N_LOCAL + 2 * halo
    padded_per_dev_ref = [
        np.asarray(Q_pad_global[:, d * n_padded:(d + 1) * n_padded]).copy()
        for d in range(N_DEVS)
    ]
    qaux_per_dev = [
        np.zeros((Qaux_setup.shape[0], n_padded), dtype=Q_setup.dtype)
        for _ in range(N_DEVS)
    ]
    for _ in range(N_STEPS):
        padded_per_dev_ref = _periodic_halo_np(padded_per_dev_ref, halo)
        for d in range(N_DEVS):
            Q_d = jnp.asarray(padded_per_dev_ref[d])
            Qaux_d = jnp.asarray(qaux_per_dev[d])
            dQ_d = flux_op_part(
                jnp.asarray(DT, dtype=Q_setup.dtype),
                jnp.asarray(0.0, dtype=Q_setup.dtype),
                Q_d, Qaux_d, parameters, jnp.zeros_like(Q_d),
            )
            padded_per_dev_ref[d] = np.asarray(Q_d + DT * dQ_d)

    def spmd_step(Q_pad, Qaux_pad):
        Q_pad = _periodic_halo(Q_pad, halo, "cells", N_DEVS)
        dQ = flux_op_part(
            jnp.asarray(DT, dtype=Q_pad.dtype),
            jnp.asarray(0.0, dtype=Q_pad.dtype),
            Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad),
        )
        return Q_pad + DT * dQ

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
            :, d * n_padded + halo:d * n_padded + halo + N_LOCAL
        ]
        owned_ref = padded_per_dev_ref[d][:, halo:halo + N_LOCAL]
        max_err = max(max_err, float(np.max(np.abs(owned_spmd - owned_ref))))
    return max_err


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_hyperbolic_solver_flux_op_composes_with_shard_map_order1():
    """Order-1 constant reconstruction, halo=1.  Fast compile (~4s)."""
    err = _run_one_case(reconstruction_order=1, halo=1)
    print(f"  HyperbolicSolver order=1 flux op SPMD vs replicated: "
          f"max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_hyperbolic_solver_flux_op_composes_with_shard_map_order2():
    """Order-2 LSQ-MUSCL with Venkatakrishnan limiter, halo=2.
    Compile is heavier — proves the second-order pipeline composes
    end-to-end with shard_map."""
    err = _run_one_case(reconstruction_order=2, halo=2)
    print(f"  HyperbolicSolver order=2 flux op SPMD vs replicated: "
          f"max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"
