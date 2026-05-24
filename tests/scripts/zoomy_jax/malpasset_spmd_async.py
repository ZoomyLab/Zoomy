"""SPMD speedup attempt #2: per-rank ``jax.jit`` that fuses
slice-from-global + flux + update in ONE traced program per rank.

Each rank's jit'd function takes the FULL global Q and returns its
updated OWNED slice.  Inputs are placed on per-device buffers via
``jax.device_put``; 4 calls dispatched in sequence return
immediately (JAX async), then we ``block_until_ready`` once at the
end of the step.  Wall-clock ≈ max(per-rank time) instead of sum.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
      python tests/scripts/zoomy_jax/malpasset_spmd_async.py
"""
from __future__ import annotations

import os
import sys
import time

_OUT_ROOT = (
    os.path.expanduser("~/sciebo")
    if os.path.isdir(os.path.expanduser("~/sciebo"))
    else "/tmp"
)
os.environ["ZOOMY_DIR"] = _OUT_ROOT
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malpasset_swe_model import MalpassetSWE  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.models.system_model import SystemModel  # noqa: E402
from zoomy_core.numerics import NumericalSystemModel  # noqa: E402
from zoomy_core.numerics.numerical_system_model import (
    ReconstructionSpec,
)  # noqa: E402
from zoomy_jax.fvm.solver_jax import HyperbolicSolver  # noqa: E402
from zoomy_jax.mesh.partition import (
    partition_mesh, extract_local_mesh,
)  # noqa: E402
from zoomy_jax.mesh.mesh import convert_mesh_to_jax  # noqa: E402


MESH_PATH = "/home/ingo/git/Zoomy/data/malpasset/geo_malpasset-small.msh"


def _load_ic(solver, Q):
    import meshio
    m = meshio.read(MESH_PATH)
    cv = np.asarray(solver._rt_mesh.cell_vertices)
    nc = int(solver._rt_mesh.n_inner_cells)
    cv = cv[:, :nc]
    B, H, U, V = (m.point_data[k] for k in ("B", "H", "U", "V"))
    b = (B[cv[0]] + B[cv[1]] + B[cv[2]]) / 3.0
    h = (H[cv[0]] + H[cv[1]] + H[cv[2]]) / 3.0
    u = (U[cv[0]] + U[cv[1]] + U[cv[2]]) / 3.0
    v = (V[cv[0]] + V[cv[1]] + V[cv[2]]) / 3.0
    Q = Q.at[0, :nc].set(jnp.asarray(b, dtype=Q.dtype))
    Q = Q.at[1, :nc].set(jnp.asarray(h, dtype=Q.dtype))
    Q = Q.at[2, :nc].set(jnp.asarray(h * u, dtype=Q.dtype))
    Q = Q.at[3, :nc].set(jnp.asarray(h * v, dtype=Q.dtype))
    return Q


def main():
    N_STEPS = 50
    devices = jax.devices()
    N_PARTS = len(devices)
    print(f"\nMalpasset SPMD-async — {N_PARTS} devices, {N_STEPS} steps")

    t0 = time.perf_counter()
    mesh_np = LSQMesh.from_msh(MESH_PATH)
    nc_global = int(mesh_np.n_inner_cells)
    print(f"[{time.perf_counter() - t0:6.1f}s] n_inner={nc_global}")

    model = MalpassetSWE()
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(order=1),
    )
    solver = HyperbolicSolver()
    Q_glob, Qaux_glob = solver.setup_simulation(mesh_np, nsm)
    Q_glob = _load_ic(solver, Q_glob)
    Qaux_glob = solver.update_qaux(
        Q_glob, Qaux_glob, Q_glob, Qaux_glob, solver._rt_mesh,
        solver._rt_model, solver._rt_parameters, 0.0, 1.0,
    )
    parameters = solver._rt_parameters
    runtime = solver._rt_model

    dt = jnp.asarray(0.005, dtype=Q_glob.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_glob.dtype)

    # ─── BASELINE.
    flux_op_global = solver._rt_flux_op
    print(f"\n[{time.perf_counter() - t0:6.1f}s] BASELINE compile...")
    t_c0 = time.perf_counter()
    Q_warm = Q_glob + dt * flux_op_global(
        dt, t_j, Q_glob, Qaux_glob, parameters, jnp.zeros_like(Q_glob)
    )
    Q_warm.block_until_ready()
    t_compile_global = time.perf_counter() - t_c0
    print(f"   compile = {t_compile_global:.2f}s")

    print(f"[{time.perf_counter() - t0:6.1f}s] BASELINE {N_STEPS} steps...")
    t_r0 = time.perf_counter()
    Q_cur = Q_glob
    for _ in range(N_STEPS):
        Q_cur = Q_cur + dt * flux_op_global(
            dt, t_j, Q_cur, Qaux_glob, parameters,
            jnp.zeros_like(Q_cur),
        )
    Q_cur.block_until_ready()
    t_run_global = time.perf_counter() - t_r0
    print(f"   wall = {t_run_global:.3f}s ({1e3 * t_run_global / N_STEPS:.2f} ms/step)")

    # ─── SPMD: per-rank fused jit'd functions.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] Partition + local meshes...")
    parts_info = partition_mesh(mesh_np, n_parts=N_PARTS)
    local_meshes = [extract_local_mesh(mesh_np, p) for p in parts_info]
    print(f"[{time.perf_counter() - t0:6.1f}s] local shapes: "
          f"{[(int(lm.n_inner_cells), int(lm.n_cells)) for lm in local_meshes]}")

    # Per-rank: build a fetch_idx array and the local flux op.
    rank_step_fns = []
    rank_owned_idx = []
    for i, (p_info, lm) in enumerate(zip(parts_info, local_meshes)):
        n_local = int(lm.n_cells)
        n_owned = int(lm.n_inner_cells)
        all_cells = np.asarray(p_info.ghost_cells)
        # Build fetch_idx of length n_local: index into Q_global (or
        # 0 for physical-BC ghosts whose values aren't read).
        fetch_idx = np.zeros(n_local, dtype=np.int64)
        fetch_idx[:n_owned] = np.asarray(p_info.owned_cells)
        for k, gc in enumerate(all_cells):
            fetch_idx[n_owned + k] = gc if gc < nc_global else 0
        fetch_idx_j = jnp.asarray(fetch_idx)
        rank_owned_idx.append(jnp.asarray(p_info.owned_cells))

        lm_jax = convert_mesh_to_jax(lm)
        flux_op_local = solver.get_flux_operator(lm_jax, runtime)

        def make_step(flux_op_local, fetch_idx_j, n_owned):
            @jax.jit
            def step(Q_glob_arg, Qaux_glob_arg):
                Q_local = Q_glob_arg[:, fetch_idx_j]
                Qaux_local = Qaux_glob_arg[:, fetch_idx_j]
                dQ_local = flux_op_local(
                    dt, t_j, Q_local, Qaux_local, parameters,
                    jnp.zeros_like(Q_local),
                )
                Q_local_new = Q_local + dt * dQ_local
                return Q_local_new[:, :n_owned]
            return step

        rank_step_fns.append(make_step(flux_op_local, fetch_idx_j, n_owned))

    # Place Q_global on each device.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] Place Q_global on each device + compile...")
    t_c0 = time.perf_counter()
    Q_per_dev = [jax.device_put(Q_glob, devices[i]) for i in range(N_PARTS)]
    Qaux_per_dev = [jax.device_put(Qaux_glob, devices[i]) for i in range(N_PARTS)]
    # Warm-up compile (one call per rank).
    owned_outputs = [
        rank_step_fns[i](Q_per_dev[i], Qaux_per_dev[i])
        for i in range(N_PARTS)
    ]
    for o in owned_outputs:
        o.block_until_ready()
    t_compile_spmd = time.perf_counter() - t_c0
    print(f"   compile = {t_compile_spmd:.2f}s")

    # ─── Timed SPMD loop.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] SPMD {N_STEPS} steps...")
    t_r0 = time.perf_counter()
    Q_global_state = Q_glob
    for s in range(N_STEPS):
        # Place current Q on each device (overlapped with compute via async).
        Q_per_dev_s = [jax.device_put(Q_global_state, devices[i])
                       for i in range(N_PARTS)]
        # Dispatch per-rank jit'd step (async, returns futures).
        owned_outputs = [
            rank_step_fns[i](Q_per_dev_s[i], Qaux_per_dev[i])
            for i in range(N_PARTS)
        ]
        # Gather back to global state.  Each owned_outputs[i] is on
        # device i; .at[].set() with an array on a different device
        # triggers transfer.  For correctness this works; for speed
        # we'd want shard_map's collective gather.
        Q_global_state = Q_glob  # start from a fresh array of the right shape
        for i in range(N_PARTS):
            owned_outputs[i].block_until_ready()
            Q_global_state = Q_global_state.at[:, rank_owned_idx[i]].set(
                jax.device_put(owned_outputs[i], devices[0])
            )
    Q_global_state.block_until_ready()
    t_run_spmd = time.perf_counter() - t_r0
    print(f"   wall = {t_run_spmd:.3f}s ({1e3 * t_run_spmd / N_STEPS:.2f} ms/step)")

    speedup = t_run_global / t_run_spmd if t_run_spmd > 0 else float("nan")
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  Single-device wall:    {t_run_global:.3f}s  "
          f"({1e3 * t_run_global / N_STEPS:.2f} ms/step)")
    print(f"  SPMD-{N_PARTS} async wall:    {t_run_spmd:.3f}s  "
          f"({1e3 * t_run_spmd / N_STEPS:.2f} ms/step)")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"\nTotal wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
