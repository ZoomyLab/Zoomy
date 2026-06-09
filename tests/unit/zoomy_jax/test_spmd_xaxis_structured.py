"""SPMD x-axis decomposition for STRUCTURED 2D and 3D meshes.

Uses :func:`partition_xaxis_structured` to split a uniform nx*ny
(or nx*ny*nz) mesh along the x axis only; the y (and z) directions
are NOT decomposed.  Halo cells form strips/slabs of ``halo*ny``
(or ``halo*ny*nz``) cells.

Bit-identity vs replicated single-device on a smooth IC that is
constant in y (and z): the y/z dimensions carry no gradient, so
the y/z-BC complication (currently set to -1 in the per-partition
LSQ boundary face neighbors) has no effect on the result.

The actual ``HyperbolicSolver`` flux operator (1D scalar advection
in the active x direction; y and z are inert because the IC and
parameters have no y/z component) composes with shard_map exactly
as in the 1D case.
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
from zoomy_jax.mesh.partition_jax import partition_xaxis_structured


N_DEVS = 4
NX_TOTAL = 16
N_LOCAL_X = NX_TOTAL // N_DEVS
DT_X = 0.25 / NX_TOTAL    # CFL on the active x direction
N_STEPS = 3


def _smooth_ic_x(x):
    """Smooth periodic IC in x — constant in y (and z when 3D)."""
    return 1.0 + 0.5 * np.sin(2 * np.pi * x)


def _periodic_halo_jax(Q_pad, halo_cells, axis_name, n_devices):
    """Generic periodic-wrap halo exchange — operates on the FLAT
    cell axis (cell axis = last axis of Q_pad).  ``halo_cells`` is
    the halo width in **cells** (= halo_x * x_stride for 2D/3D)."""
    left_owned = Q_pad[:, halo_cells:2 * halo_cells]
    right_owned = Q_pad[:, -2 * halo_cells:-halo_cells]
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(right_owned, perm=perm_right, axis_name=axis_name)
    fill_right = lax.ppermute(left_owned, perm=perm_left, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo_cells].set(fill_left)
    Q_pad = Q_pad.at[:, -halo_cells:].set(fill_right)
    return Q_pad


def _periodic_halo_np(padded_per_dev, halo_cells, n_local_cells):
    """NumPy/JAX periodic halo refill (no SPMD) for the replicated
    reference path."""
    out = [arr.copy() for arr in padded_per_dev]
    n = len(out)
    for d in range(n):
        left_nbr = (d - 1) % n
        right_nbr = (d + 1) % n
        out[d][:, :halo_cells] = out[left_nbr][
            :, n_local_cells:n_local_cells + halo_cells
        ]
        out[d][:, -halo_cells:] = out[right_nbr][
            :, halo_cells:halo_cells + halo_cells
        ]
    return out


def _run_xaxis_case(dim: int, ny: int = 3, nz: int = 2,
                    reconstruction_order: int = 1, halo_x: int = 1):
    """Build the solver + per-partition flux op, run N_STEPS in both
    SPMD and replicated-single-device modes, return max owned-cell
    error."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    if dim == 2:
        shape = (NX_TOTAL, ny)
        domain = (0.0, 1.0, 0.0, 1.0)
        mesh_np = LSQMesh.create_2d(domain=domain, nx=NX_TOTAL, ny=ny)
    elif dim == 3:
        shape = (NX_TOTAL, ny, nz)
        domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        mesh_np = LSQMesh.create_3d(
            domain=domain, nx=NX_TOTAL, ny=ny, nz=nz
        )
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    x_stride = int(np.prod(shape[1:]))
    n_local_cells = N_LOCAL_X * x_stride
    halo_cells = halo_x * x_stride
    n_padded_cells = n_local_cells + 2 * halo_cells

    model = Advection(dimension=dim)
    # Make the parameters: a_x = 1, a_y = 0, a_z = 0 — flow only in x.
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(
            order=reconstruction_order, limiter="venkatakrishnan"
        ),
    )
    solver = HyperbolicSolver()
    Q_setup, Qaux_setup = solver.setup_simulation(mesh_np, nsm)
    runtime = solver._rt_model
    global_jax_mesh = solver._rt_mesh

    # Zero out a_y (and a_z if 3D) so flow is purely in x.  Parameters
    # is a 1D array; Advection model has params (a_x, a_y[, a_z]) per
    # the model's __init__ order.
    params_orig = np.asarray(solver._rt_parameters).copy()
    if len(params_orig) >= 2:
        params_orig[1] = 0.0    # a_y
    if len(params_orig) >= 3:
        params_orig[2] = 0.0    # a_z
    parameters = jnp.asarray(params_orig, dtype=solver._rt_parameters.dtype)

    parts = partition_xaxis_structured(
        global_jax_mesh, n_parts=N_DEVS, halo=halo_x,
        domain=domain, shape=shape,
    )
    # Use parts[1] (interior — no global x-BC).
    part_mesh = parts[1]
    flux_op_part = solver.get_flux_operator(part_mesh, runtime)

    # Build IC on the global mesh: u(x,y,z) = f(x) only.
    centers_global = np.asarray(global_jax_mesh.cell_centers)
    x_global = centers_global[0, :int(global_jax_mesh.n_inner_cells)]
    u0_np = _smooth_ic_x(x_global).astype(np.float32).reshape(1, -1)

    # ── Build per-device padded Q.
    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, halo_cells)), chunk, np.zeros((1, halo_cells))], axis=1
    )
    chunks = [
        u0_np[:, d * n_local_cells:(d + 1) * n_local_cells]
        for d in range(N_DEVS)
    ]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(c) for c in chunks], axis=1),
        dtype=Q_setup.dtype,
    )
    Qaux_pad = jnp.zeros(
        (Qaux_setup.shape[0], Q_pad_global.shape[1]), dtype=Q_setup.dtype
    )

    # ── Reference: replicated single-device.
    padded_per_dev_ref = [
        np.asarray(Q_pad_global[:, d * n_padded_cells:(d + 1) * n_padded_cells]).copy()
        for d in range(N_DEVS)
    ]
    qaux_per_dev = [
        np.zeros((Qaux_setup.shape[0], n_padded_cells), dtype=Q_setup.dtype)
        for _ in range(N_DEVS)
    ]
    dt_j = jnp.asarray(DT_X, dtype=Q_setup.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_setup.dtype)

    for _ in range(N_STEPS):
        padded_per_dev_ref = _periodic_halo_np(
            padded_per_dev_ref, halo_cells, n_local_cells
        )
        for d in range(N_DEVS):
            Q_d = jnp.asarray(padded_per_dev_ref[d])
            Qaux_d = jnp.asarray(qaux_per_dev[d])
            dQ_d = flux_op_part(
                dt_j, t_j, Q_d, Qaux_d, parameters, jnp.zeros_like(Q_d),
            )
            padded_per_dev_ref[d] = np.asarray(Q_d + DT_X * dQ_d)

    # ── SPMD path.
    def spmd_step(Q_pad, Qaux_pad):
        Q_pad = _periodic_halo_jax(Q_pad, halo_cells, "cells", N_DEVS)
        dQ = flux_op_part(
            dt_j, t_j, Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad),
        )
        return Q_pad + DT_X * dQ

    @partial(shard_map, mesh=spmd_mesh,
             in_specs=(P(None, "cells"), P(None, "cells")),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad, Qaux_pad):
        def body(carry, _):
            return spmd_step(carry, Qaux_pad), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(N_STEPS))
        return Q_final

    Q_final_spmd = np.asarray(run(Q_pad_global, Qaux_pad))

    # ── Owned-cell error.
    max_err = 0.0
    for d in range(N_DEVS):
        owned_spmd = Q_final_spmd[
            :, d * n_padded_cells + halo_cells:
               d * n_padded_cells + halo_cells + n_local_cells
        ]
        owned_ref = padded_per_dev_ref[d][
            :, halo_cells:halo_cells + n_local_cells
        ]
        max_err = max(
            max_err, float(np.max(np.abs(owned_spmd - owned_ref)))
        )
    return max_err


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_xaxis_2d_order1():
    """2D structured mesh (nx=16, ny=3), x-axis partition with
    halo=1, ConstantReconstruction."""
    err = _run_xaxis_case(dim=2, ny=3, reconstruction_order=1, halo_x=1)
    print(f"  2D x-axis order=1 SPMD vs replicated: max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_xaxis_3d_order1():
    """3D structured mesh (nx=16, ny=3, nz=2), x-axis partition with
    halo=1, ConstantReconstruction."""
    err = _run_xaxis_case(dim=3, ny=3, nz=2, reconstruction_order=1, halo_x=1)
    print(f"  3D x-axis order=1 SPMD vs replicated: max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_xaxis_2d_order2():
    """2D structured mesh, halo=2, LSQ-MUSCL with Venkatakrishnan
    limiter — the full second-order pipeline composes in 2D."""
    err = _run_xaxis_case(dim=2, ny=3, reconstruction_order=2, halo_x=2)
    print(f"  2D x-axis order=2 SPMD vs replicated: max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_partition_xaxis_3d_order2():
    """3D structured mesh, halo=2, LSQ-MUSCL with Venkatakrishnan
    limiter — full second-order pipeline composes in 3D."""
    err = _run_xaxis_case(dim=3, ny=3, nz=2, reconstruction_order=2, halo_x=2)
    print(f"  3D x-axis order=2 SPMD vs replicated: max_err = {err:.3e}")
    assert err < 1e-6, f"err {err:.3e}"
