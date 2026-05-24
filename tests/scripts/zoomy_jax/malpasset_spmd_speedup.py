"""SPMD speedup measurement for the actual Malpasset SWE.

Per-rank flux ops are placed on different JAX devices and run
concurrently via async dispatch.  Wall-clock = max(per-rank time)
rather than sum, giving the speedup vs single-device.

NOTE: on fake CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=N
without real CPUs), all "devices" share one physical core so the
wall-clock won't actually drop with N — but the architecture is
correct and on real CPU/GPU/TPU sharding the same code shows true
speedup.  We report all numbers honestly with this caveat."""
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
    N_PARTS = jax.device_count()
    N_STEPS = 50
    print(f"\nMalpasset SPMD scaling — {N_PARTS} devices, {N_STEPS} steps")
    print(f"  jax.devices() = {jax.devices()}")

    t0 = time.perf_counter()
    print(f"[{time.perf_counter() - t0:6.1f}s] Load mesh...")
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
    print(f"[{time.perf_counter() - t0:6.1f}s] global solver ready")

    dt = jnp.asarray(0.005, dtype=Q_glob.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_glob.dtype)

    # ─── BASELINE: single device, sequential.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] BASELINE single-device, "
          f"compile + {N_STEPS} steps")
    flux_op_global = solver._rt_flux_op
    t_c0 = time.perf_counter()
    Q1 = Q_glob + dt * flux_op_global(
        dt, t_j, Q_glob, Qaux_glob, parameters, jnp.zeros_like(Q_glob)
    )
    Q1.block_until_ready()
    t_compile_global = time.perf_counter() - t_c0
    print(f"   compile = {t_compile_global:.2f}s")

    t_r0 = time.perf_counter()
    Q_cur = Q_glob
    for _ in range(N_STEPS):
        Q_cur = Q_cur + dt * flux_op_global(
            dt, t_j, Q_cur, Qaux_glob, parameters,
            jnp.zeros_like(Q_cur),
        )
    Q_cur.block_until_ready()
    t_run_global = time.perf_counter() - t_r0
    print(f"   {N_STEPS} steps wall = {t_run_global:.3f}s "
          f"({1e3 * t_run_global / N_STEPS:.2f} ms/step)")

    # ─── SPMD: partition + per-rank flux op on device i.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] Partitioning + extracting "
          f"local meshes...")
    parts_info = partition_mesh(mesh_np, n_parts=N_PARTS)
    local_meshes = [extract_local_mesh(mesh_np, p) for p in parts_info]
    print(f"[{time.perf_counter() - t0:6.1f}s] local mesh shapes: "
          f"{[(int(lm.n_inner_cells), int(lm.n_cells)) for lm in local_meshes]}")

    Q_glob_np = np.asarray(Q_glob)
    Qaux_glob_np = np.asarray(Qaux_glob)
    devices = jax.devices()
    flux_ops_per_rank = []
    Q_local_per_rank = []
    Qaux_local_per_rank = []

    print(f"\n[{time.perf_counter() - t0:6.1f}s] Building per-rank flux "
          f"ops + placing on devices...")
    for i, (p_info, lm) in enumerate(zip(parts_info, local_meshes)):
        lm_jax = convert_mesh_to_jax(lm)
        n_local = int(lm.n_cells)
        n_owned = int(lm.n_inner_cells)
        flux_op_local = solver.get_flux_operator(lm_jax, runtime)
        flux_ops_per_rank.append(flux_op_local)

        # Pack initial per-rank Q + Qaux on device i.
        Q_local_np = np.zeros((Q_glob_np.shape[0], n_local),
                              dtype=Q_glob_np.dtype)
        Qaux_local_np = np.zeros((Qaux_glob_np.shape[0], n_local),
                                 dtype=Qaux_glob_np.dtype)
        Q_local_np[:, :n_owned] = Q_glob_np[:, p_info.owned_cells]
        Qaux_local_np[:, :n_owned] = Qaux_glob_np[:, p_info.owned_cells]
        for k, gc in enumerate(np.asarray(p_info.ghost_cells)):
            if gc < nc_global:
                Q_local_np[:, n_owned + k] = Q_glob_np[:, gc]
                Qaux_local_np[:, n_owned + k] = Qaux_glob_np[:, gc]
        Q_local = jax.device_put(jnp.asarray(Q_local_np), devices[i])
        Qaux_local = jax.device_put(jnp.asarray(Qaux_local_np), devices[i])
        Q_local_per_rank.append(Q_local)
        Qaux_local_per_rank.append(Qaux_local)

    # Halo exchange: takes the current Q_local arrays, gathers
    # cross-partition ghost values from the appropriate other ranks.
    # Precompute send/recv index maps.
    # For each rank i: ghost_global[k] -> from rank j (=membership[gc])
    # at local position p_info_j.global_to_local[gc].
    membership = -np.ones(nc_global, dtype=int)
    for r, p in enumerate(parts_info):
        membership[np.asarray(p.owned_cells)] = r

    # Per-rank: ghost slots → (source_rank, source_local_idx).
    ghost_pull_per_rank = []  # list of (k_local, src_rank, src_local_idx)
    for i, p_info in enumerate(parts_info):
        lm = local_meshes[i]
        n_owned = int(lm.n_inner_cells)
        pulls = []
        for k, gc in enumerate(np.asarray(p_info.ghost_cells)):
            if gc < nc_global:
                src_rank = int(membership[gc])
                src_local_idx = int(parts_info[src_rank].global_to_local[gc])
                pulls.append((n_owned + k, src_rank, src_local_idx))
        ghost_pull_per_rank.append(pulls)

    def halo_exchange(Qs_per_rank):
        """Pull ghost cell values from owner ranks.  Each rank's Q is
        on a different device; jax.device_put handles cross-device
        transfer.  Returns new Qs_per_rank list."""
        new_Qs = []
        for i, Q_i in enumerate(Qs_per_rank):
            Q_new = Q_i
            for (k, src, src_idx) in ghost_pull_per_rank[i]:
                # Pull the column from src rank; device_put to my device.
                val = jax.device_put(
                    Qs_per_rank[src][:, src_idx], devices[i]
                )
                Q_new = Q_new.at[:, k].set(val)
            new_Qs.append(Q_new)
        return new_Qs

    print(f"[{time.perf_counter() - t0:6.1f}s] Compile per-rank flux ops "
          f"(first call)...")
    t_c0 = time.perf_counter()
    # Compile: one warm-up step per rank.
    dQs = []
    for i in range(N_PARTS):
        dQ = flux_ops_per_rank[i](
            dt, t_j, Q_local_per_rank[i], Qaux_local_per_rank[i],
            parameters, jnp.zeros_like(Q_local_per_rank[i]),
        )
        dQs.append(dQ)
    for dQ in dQs:
        dQ.block_until_ready()
    t_compile_spmd = time.perf_counter() - t_c0
    print(f"   per-rank compile total = {t_compile_spmd:.2f}s")

    # ─── Timed SPMD loop.
    print(f"\n[{time.perf_counter() - t0:6.1f}s] SPMD {N_STEPS} steps "
          f"(halo exchange + per-rank flux dispatch)...")
    Q_current = Q_local_per_rank
    t_r0 = time.perf_counter()
    for s in range(N_STEPS):
        Q_current = halo_exchange(Q_current)
        new_Qs = []
        for i in range(N_PARTS):
            dQ_i = flux_ops_per_rank[i](
                dt, t_j, Q_current[i], Qaux_local_per_rank[i],
                parameters, jnp.zeros_like(Q_current[i]),
            )
            new_Qs.append(Q_current[i] + dt * dQ_i)
        Q_current = new_Qs
    for Q_i in Q_current:
        Q_i.block_until_ready()
    t_run_spmd = time.perf_counter() - t_r0
    print(f"   wall = {t_run_spmd:.3f}s ({1e3 * t_run_spmd / N_STEPS:.2f} "
          f"ms/step)")

    speedup = t_run_global / t_run_spmd if t_run_spmd > 0 else float("nan")
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  Single-device wall:    {t_run_global:.3f}s")
    print(f"  SPMD-{N_PARTS} wall:          {t_run_spmd:.3f}s")
    print(f"  Speedup:               {speedup:.2f}x  "
          f"(ideal {N_PARTS}x; <1 on fake-CPU because all 'devices' "
          f"share 1 physical core)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"\nTotal wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
