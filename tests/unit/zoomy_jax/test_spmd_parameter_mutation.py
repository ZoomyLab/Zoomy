"""SPMD with mutated model parameters: confirm that changing
``solver._rt_parameters`` between SPMD runs propagates correctly
through the sharded flux op — i.e., parameters are passed through
the shard_map call site as a regular argument, not closure-captured.

Important for parameter sweeps + AD (jax.grad against parameters).
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
HALO = 1
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.25 * DX
N_STEPS = 2


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


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_parameter_change_propagates_through_spmd():
    """Run the SPMD flux op TWICE — once with a_x = 1.0, once with
    a_x = 2.0.  Verify the two outputs DIFFER (parameter is not
    frozen as a JIT constant) and that doubling a_x yields a propor-
    tionally larger update (flux ∝ a_x for advection)."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    model = Advection(dimension=1)
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(order=1),
    )
    solver = HyperbolicSolver()
    Q_setup, Qaux_setup = solver.setup_simulation(mesh_np, nsm)
    runtime = solver._rt_model
    global_jax_mesh = solver._rt_mesh

    parts = partition_1d_contiguous(
        global_jax_mesh, n_parts=N_DEVS, halo=HALO
    )
    flux_op_part = solver.get_flux_operator(parts[1], runtime)

    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0_np = _smooth_ic(xc).astype(np.float32).reshape(1, N_TOTAL)
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

    def spmd_step(Q_pad, Qaux_pad, params):
        Q_pad = _periodic_halo(Q_pad, HALO, "cells", N_DEVS)
        dQ = flux_op_part(
            jnp.asarray(DT, dtype=Q_pad.dtype),
            jnp.asarray(0.0, dtype=Q_pad.dtype),
            Q_pad, Qaux_pad, params, jnp.zeros_like(Q_pad),
        )
        return Q_pad + DT * dQ

    @partial(shard_map, mesh=spmd_mesh,
             in_specs=(P(None, "cells"), P(None, "cells"), P()),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad, Qaux_pad, params):
        def body(carry, _):
            return spmd_step(carry, Qaux_pad, params), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(N_STEPS))
        return Q_final

    # Run with original parameters (a_x = 1.0 by default).
    params_orig = solver._rt_parameters
    Q1_a1 = np.asarray(run(Q_pad_global, Qaux_pad, params_orig))

    # Mutate the parameter to a_x = 2.0 and re-run.
    params_x2 = jnp.asarray(np.asarray(params_orig) * 2.0,
                            dtype=params_orig.dtype)
    Q1_a2 = np.asarray(run(Q_pad_global, Qaux_pad, params_x2))

    # The outputs MUST differ — if they're identical, params were
    # constant-folded into the jit trace.
    delta = float(np.max(np.abs(Q1_a1 - Q1_a2)))
    print(f"  Δ(Q | a=1.0 vs a=2.0) = {delta:.3e}")
    assert delta > 1e-6, (
        "Parameter change had no effect — params got constant-folded"
    )

    # Sanity: dQ ∝ a_x for linear advection, so the update increment
    # (Q - Q0) should approximately double.  Use the centre owned cell
    # where the halo doesn't influence.
    n_padded = N_LOCAL + 2 * HALO
    centre_dev = 1
    base = centre_dev * n_padded + HALO
    Q0_centre = np.asarray(Q_pad_global[0, base:base + N_LOCAL])
    dQ_a1 = np.asarray(Q1_a1[0, base:base + N_LOCAL]) - Q0_centre
    dQ_a2 = np.asarray(Q1_a2[0, base:base + N_LOCAL]) - Q0_centre
    ratio = np.max(np.abs(dQ_a2)) / max(np.max(np.abs(dQ_a1)), 1e-12)
    print(f"  ratio max|dQ@a=2| / max|dQ@a=1| = {ratio:.3f}  (expect ≈ 2)")
    assert 1.8 < ratio < 2.2, (
        f"Linear-advection scaling broken: ratio {ratio:.3f} not ≈ 2"
    )
