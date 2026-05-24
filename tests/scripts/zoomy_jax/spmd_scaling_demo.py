"""SPMD scaling-knob demo: same 1D advection problem sharded across
N_DEVS ∈ {1, 2, 4} CPU devices, reporting compile time and steady-
state per-step time.

Run with:

    XLA_FLAGS="--xla_force_host_platform_device_count=8" \\
        python tests/scripts/zoomy_jax/spmd_scaling_demo.py

Run with extra device counts:

    XLA_FLAGS="--xla_force_host_platform_device_count=16" \\
        python tests/scripts/zoomy_jax/spmd_scaling_demo.py --n_devs 1,2,4,8,16

The demo is not a benchmark — fake CPU devices share one physical
core, so wall-clock does not scale.  It exists so a developer can
spot regressions in compile cost and confirm the SPMD path runs
end-to-end at a real problem size.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models.advection import Advection
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.partition_jax import (
    partition_1d_contiguous,
    partition_xaxis_structured,
)
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace


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


def run_one(n_devs: int, dim: int, nx: int, halo: int, n_steps: int,
            reconstruction_order: int, ny: int = 4, nz: int = 4):
    """Run SPMD scaling for a given (dim, nx, n_devs) configuration."""
    if jax.device_count() < n_devs:
        return None
    spmd_mesh = Mesh(
        np.array(jax.devices()[:n_devs]), axis_names=("cells",)
    )
    n_local_x = nx // n_devs
    assert nx % n_devs == 0, f"{nx} % {n_devs} != 0"
    x_stride = 1 if dim == 1 else (ny if dim == 2 else ny * nz)
    n_local_cells = n_local_x * x_stride
    halo_cells = halo * x_stride
    n_padded_cells = n_local_cells + 2 * halo_cells
    n_total_cells = nx * x_stride

    if dim == 1:
        domain = (0.0, 1.0)
        shape = (nx,)
        mesh_np = LSQMesh.create_1d(domain=domain, n_inner_cells=nx)
    elif dim == 2:
        domain = (0.0, 1.0, 0.0, 1.0)
        shape = (nx, ny)
        mesh_np = LSQMesh.create_2d(domain=domain, nx=nx, ny=ny)
    else:
        domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        shape = (nx, ny, nz)
        mesh_np = LSQMesh.create_3d(domain=domain, nx=nx, ny=ny, nz=nz)

    dx = (domain[1] - domain[0]) / nx
    dt = 0.25 * dx

    model = Advection(dimension=dim)
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

    # Zero out a_y / a_z so flow is purely along x.
    params_np = np.asarray(solver._rt_parameters).copy()
    for i in range(1, len(params_np)):
        params_np[i] = 0.0
    parameters = jnp.asarray(params_np, dtype=solver._rt_parameters.dtype)

    parts = partition_xaxis_structured(
        global_jax_mesh, n_parts=n_devs, halo=halo,
        domain=domain, shape=shape,
    )
    part_mesh = parts[min(1, len(parts) - 1)]
    flux_op_part = solver.get_flux_operator(part_mesh, runtime)

    # IC: constant in y/z, smooth in x.
    centers = np.asarray(global_jax_mesh.cell_centers)
    x_global = centers[0, :int(global_jax_mesh.n_inner_cells)]
    u0_np = _smooth_ic(x_global).astype(np.float32).reshape(1, -1)

    pad_chunk = lambda chunk: np.concatenate(
        [np.zeros((1, halo_cells)), chunk, np.zeros((1, halo_cells))], axis=1
    )
    chunks = [
        u0_np[:, d * n_local_cells:(d + 1) * n_local_cells]
        for d in range(n_devs)
    ]
    Q_pad_global = jnp.asarray(
        np.concatenate([pad_chunk(c) for c in chunks], axis=1),
        dtype=Q_setup.dtype,
    )
    Qaux_pad = jnp.zeros(
        (Qaux_setup.shape[0], Q_pad_global.shape[1]), dtype=Q_setup.dtype
    )
    dt_j = jnp.asarray(dt, dtype=Q_setup.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_setup.dtype)

    def spmd_stage(Q_pad, Qaux_pad):
        Q_pad = _periodic_halo(Q_pad, halo_cells, "cells", n_devs)
        dQ = flux_op_part(
            dt_j, t_j, Q_pad, Qaux_pad, parameters, jnp.zeros_like(Q_pad),
        )
        return Q_pad + dt * dQ

    def spmd_step(Q_pad, Qaux_pad):
        if reconstruction_order >= 2:
            Q0 = Q_pad
            Q1 = spmd_stage(Q0, Qaux_pad)
            Q2 = spmd_stage(Q1, Qaux_pad)
            return 0.5 * (Q0 + Q2)
        return spmd_stage(Q_pad, Qaux_pad)

    @partial(shard_map, mesh=spmd_mesh,
             in_specs=(P(None, "cells"), P(None, "cells")),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Q_pad, Qaux_pad):
        def body(carry, _):
            return spmd_step(carry, Qaux_pad), None
        Q_final, _ = lax.scan(body, Q_pad, jnp.arange(n_steps))
        return Q_final

    t0 = time.perf_counter()
    Q1 = run(Q_pad_global, Qaux_pad)
    Q1.block_until_ready()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_repeats = 3
    for _ in range(n_repeats):
        Q1 = run(Q_pad_global, Qaux_pad)
        Q1.block_until_ready()
    t_run = (time.perf_counter() - t0) / n_repeats

    return {
        "n_devs": n_devs,
        "dim": dim,
        "nx": nx,
        "n_total_cells": n_total_cells,
        "halo": halo,
        "n_steps": n_steps,
        "order": reconstruction_order,
        "compile_s": t_compile,
        "run_s": t_run,
        "us_per_step_per_cell": 1e6 * t_run / (n_steps * n_total_cells),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--n_devs", type=str, default="1,2,4")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--order", type=int, default=1)
    args = ap.parse_args()

    n_devs_list = [int(s) for s in args.n_devs.split(",")]
    halo = 2 if args.order >= 2 else 1
    print(f"\nSPMD scaling demo: {args.dim}D Advection, order={args.order}, "
          f"halo={halo}, nx={args.nx}, n_steps={args.n_steps}")
    if args.dim >= 2:
        print(f"                   ny={args.ny}" +
              (f", nz={args.nz}" if args.dim == 3 else ""))
    print(f"jax.devices()={len(jax.devices())} fake CPU devices "
          f"(XLA_FLAGS sets the count)\n")
    print(f"{'n_devs':>8} | {'n_cells':>9} | {'compile (s)':>12} | "
          f"{'run (s)':>10} | {'us/step/cell':>14}")
    print(f"{'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}-+-{'-'*14}")
    for n_devs in n_devs_list:
        if args.nx % n_devs != 0:
            print(f"{n_devs:>8} | (skipped: nx={args.nx} % {n_devs} != 0)")
            continue
        if jax.device_count() < n_devs:
            print(f"{n_devs:>8} | (skipped: jax.device_count()={jax.device_count()})")
            continue
        res = run_one(
            n_devs=n_devs, dim=args.dim, nx=args.nx, ny=args.ny, nz=args.nz,
            halo=halo, n_steps=args.n_steps,
            reconstruction_order=args.order,
        )
        print(f"{res['n_devs']:>8} | {res['n_total_cells']:>9d} | "
              f"{res['compile_s']:>12.3f} | "
              f"{res['run_s']:>10.3f} | {res['us_per_step_per_cell']:>14.3f}")


if __name__ == "__main__":
    main()
