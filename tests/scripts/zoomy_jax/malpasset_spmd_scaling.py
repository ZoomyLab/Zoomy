"""SPMD scaling measurement for MalpassetSWE viscous on a 2D
structured mesh.  Runs N steps single-device and on 2/4/8-shard
SPMD; reports wall-clock + speedup.

Mesh: nx*ny structured (lake-at-rest IC for now; bathymetry
interpolation from the actual Malpasset triangular mesh is a
follow-up, see notes at top of malpasset_swe_model.py).

The MalpassetSWE physics — flux + NCP + Manning friction + eddy
viscosity + KP-desingularised hinv aux + wet/dry eigenvalue gate —
runs unchanged through the existing JAX HyperbolicSolver in both
the single-device and SPMD paths."""
from __future__ import annotations

import os
import sys
import time
import argparse
from functools import partial

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=8"
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


def _build_setup(nx, ny, domain, order=1):
    """Build solver + IC.  Returns (solver, Q, Qaux)."""
    mesh_np = LSQMesh.create_2d(domain=domain, nx=nx, ny=ny)
    model = MalpassetSWE()
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(
            order=order, limiter="venkatakrishnan"
        ),
    )
    solver = HyperbolicSolver()
    Q, Qaux = solver.setup_simulation(mesh_np, nsm)

    # Dam-break style IC: deep water (10 m) for x < midpoint, shallow
    # (1 m) for x > midpoint.  Bathymetry: flat (b = 0).  This is the
    # Malpasset-flavour dam-break minus the actual topography.
    centers = np.asarray(solver._rt_mesh.cell_centers)
    nc = int(solver._rt_mesh.n_inner_cells)
    x_mid = 0.5 * (domain[0] + domain[1])
    h_np = np.where(centers[0, :nc] < x_mid, 10.0, 1.0).astype(
        np.asarray(Q).dtype
    )
    Q = Q.at[0].set(0.0)
    Q = Q.at[1].set(jnp.asarray(h_np))
    Q = Q.at[2].set(0.0)
    Q = Q.at[3].set(0.0)
    Qaux = solver.update_qaux(
        Q, Qaux, Q, Qaux, solver._rt_mesh, solver._rt_model,
        solver._rt_parameters, 0.0, 1.0,
    )
    return solver, Q, Qaux


def run_single_device(nx, ny, domain, n_steps, dt, order=1):
    """Run n_steps single-device.  Returns (t_compile, t_run, Q_final)."""
    solver, Q, Qaux = _build_setup(nx, ny, domain, order=order)
    # Warm-up step (triggers JIT).
    t0 = time.perf_counter()
    Q = solver.step(jnp.asarray(dt, dtype=Q.dtype), 0.0, Q, Qaux)
    Q.block_until_ready()
    t_compile = time.perf_counter() - t0

    # Warm-up again to amortise potential per-call dispatch overhead
    # before the timed loop.
    for _ in range(2):
        Q = solver.step(jnp.asarray(dt, dtype=Q.dtype), 0.0, Q, Qaux)
    Q.block_until_ready()

    # Timed loop.
    t0 = time.perf_counter()
    for s in range(n_steps):
        Q = solver.step(
            jnp.asarray(dt, dtype=Q.dtype),
            (s + 1) * dt, Q, Qaux,
        )
    Q.block_until_ready()
    t_run = time.perf_counter() - t0
    return t_compile, t_run, np.asarray(Q)


def run_spmd(nx, ny, domain, n_steps, dt, n_devs, order=1, halo_x=1):
    """Run n_steps SPMD on n_devs.  Returns (t_compile, t_run,
    owned-cell concatenation)."""
    if jax.device_count() < n_devs:
        raise RuntimeError(f"need {n_devs} devices; got {jax.device_count()}")
    spmd_mesh = Mesh(np.array(jax.devices()[:n_devs]), axis_names=("cells",))

    solver, Q, Qaux = _build_setup(nx, ny, domain, order=order)
    n_local_x = nx // n_devs
    x_stride = ny
    n_local_cells = n_local_x * x_stride
    halo_cells = halo_x * x_stride
    n_padded_cells = n_local_cells + 2 * halo_cells

    parts = partition_xaxis_structured(
        solver._rt_mesh, n_parts=n_devs, halo=halo_x,
        domain=domain, shape=(nx, ny),
    )
    # Use interior partition where possible (n_devs >= 2) so the
    # per-shard mesh has n_bf = (y-boundaries only).  For n_devs == 1
    # use the only partition (it carries the full x-BCs too).
    part_mesh = parts[min(1, len(parts) - 1)]
    flux_op_part = solver.get_flux_operator(part_mesh, solver._rt_model)
    parameters = solver._rt_parameters

    # Build padded Q + Qaux.
    pad_chunk = lambda chunk, n_var: np.concatenate(
        [np.zeros((n_var, halo_cells)), chunk,
         np.zeros((n_var, halo_cells))],
        axis=1,
    )
    Q_np = np.asarray(Q)
    Qaux_np = np.asarray(Qaux)
    chunks = [Q_np[:, d * n_local_cells:(d + 1) * n_local_cells]
              for d in range(n_devs)]
    chunks_aux = [Qaux_np[:, d * n_local_cells:(d + 1) * n_local_cells]
                  for d in range(n_devs)]
    Q_pad_global = jnp.asarray(
        np.concatenate(
            [pad_chunk(c, Q.shape[0]) for c in chunks], axis=1
        ),
        dtype=Q.dtype,
    )
    Qaux_pad_global = jnp.asarray(
        np.concatenate(
            [pad_chunk(c, Qaux.shape[0]) for c in chunks_aux], axis=1
        ),
        dtype=Q.dtype,
    )

    dt_j = jnp.asarray(dt, dtype=Q.dtype)

    def spmd_stage(Q_pad, Qaux_pad, t_j):
        Q_pad = _periodic_halo(Q_pad, halo_cells, "cells", n_devs)
        dQ = flux_op_part(
            dt_j, t_j, Q_pad, Qaux_pad, parameters,
            jnp.zeros_like(Q_pad),
        )
        return Q_pad + dt * dQ

    def spmd_step(Q_pad, Qaux_pad, t_j):
        if order >= 2:
            Q0 = Q_pad
            Q1 = spmd_stage(Q0, Qaux_pad, t_j)
            Q2 = spmd_stage(Q1, Qaux_pad, t_j + dt_j)
            return 0.5 * (Q0 + Q2)
        return spmd_stage(Q_pad, Qaux_pad, t_j)

    def _make_run(n_steps_local):
        @partial(shard_map, mesh=spmd_mesh,
                 in_specs=(P(None, "cells"), P(None, "cells")),
                 out_specs=P(None, "cells"), check_rep=False)
        def run_n_steps(Q_pad, Qaux_pad):
            def body(carry, s):
                t_j = jnp.asarray(s, dtype=Q_pad.dtype) * dt_j
                return spmd_step(carry, Qaux_pad, t_j), None
            Q_final, _ = lax.scan(body, Q_pad, jnp.arange(n_steps_local))
            return Q_final
        return jax.jit(run_n_steps)

    # Warm-up (compile) — separate trace at n_steps=1.
    run_one = _make_run(1)
    t0 = time.perf_counter()
    Q1 = run_one(Q_pad_global, Qaux_pad_global)
    Q1.block_until_ready()
    t_compile = time.perf_counter() - t0

    # Timed loop — new trace at n_steps=N.
    run_n = _make_run(n_steps - 1)
    # Warm up the n_steps-N trace too (don't include in t_run).
    Q1 = run_n(Q_pad_global, Qaux_pad_global)
    Q1.block_until_ready()
    t0 = time.perf_counter()
    Q1 = run_n(Q_pad_global, Qaux_pad_global)
    Q1.block_until_ready()
    t_run = time.perf_counter() - t0

    # Gather owned cells.
    Q_np = np.asarray(Q1)
    owned = []
    for d in range(n_devs):
        owned.append(
            Q_np[:, d * n_padded_cells + halo_cells:
                    d * n_padded_cells + halo_cells + n_local_cells]
        )
    return t_compile, t_run, np.concatenate(owned, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--n_devs", type=str, default="1,2,4,8")
    ap.add_argument("--order", type=int, default=1)
    args = ap.parse_args()

    domain = (0.0, 1600.0, 0.0, 400.0)
    dt = 0.02
    n_devs_list = [int(s) for s in args.n_devs.split(",")
                   if int(s) <= jax.device_count()
                   and args.nx % int(s) == 0]

    print(f"\nMalpasset SWE viscous — SPMD scaling")
    print(f"  mesh: {args.nx}x{args.ny}, n_cells={args.nx * args.ny}, "
          f"order={args.order}, n_steps={args.n_steps}")
    print(f"  domain: {domain}, dt={dt}\n")

    # Single-device reference.
    print(f"  baseline (1 device)...")
    t_c1, t_r1, Q_single = run_single_device(
        args.nx, args.ny, domain, args.n_steps, dt, order=args.order
    )
    print(f"    compile {t_c1:.2f}s, run {t_r1:.2f}s "
          f"({1e6 * t_r1 / (args.n_steps * args.nx * args.ny):.1f} us/step/cell)")

    print(f"\n  SPMD timings:")
    print(f"  {'n_devs':>7} | {'compile':>9} | {'run (s)':>9} | "
          f"{'us/step/cell':>13} | {'speedup':>8}")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*13}-+-{'-'*8}")
    for n_devs in n_devs_list:
        halo = 2 if args.order >= 2 else 1
        t_c, t_r, Q_spmd = run_spmd(
            args.nx, args.ny, domain, args.n_steps, dt, n_devs,
            order=args.order, halo_x=halo,
        )
        speedup = t_r1 / t_r if t_r > 0 else float("nan")
        us_pc = 1e6 * t_r / (args.n_steps * args.nx * args.ny)
        print(f"  {n_devs:>7d} | {t_c:>8.2f}s | {t_r:>8.3f}s | "
              f"{us_pc:>13.1f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
