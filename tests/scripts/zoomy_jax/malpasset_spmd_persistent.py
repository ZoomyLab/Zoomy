"""SPMD with persistent per-device Q buffers + small halo transfers.

Eliminates the per-step `device_put(Q_global)` broadcast (3.3 MB
per device per step → 14 MB/step total).  Each device keeps its
Q_local LIVE across steps; only the halo cells (~50 per pair × 8
bytes per cell × 4 vars = ~1.6 KB per pair) cross device boundaries
per step.

Per step:
  1. Each device runs ``gather_halo_to_send_i`` (jit'd small gather).
  2. Cross-device transfers: O(neighbors) small transfers per rank.
  3. Each device runs ``step_with_halo_i`` (jit'd halo-apply + flux
     + owned-update — one trace per rank).
  4. block_until_ready at end of step.

Compared to malpasset_spmd_async.py: no global Q reconstruction
per step.
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


MESH_PATH = os.environ.get(
    "MALPASSET_MESH",
    "/home/ingo/git/Zoomy/data/malpasset/geo_malpasset-small.msh",
)


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
    print(f"\nMalpasset SPMD-persistent — {N_PARTS} devices, {N_STEPS} steps")

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
    Q_warm = Q_glob + dt * flux_op_global(
        dt, t_j, Q_glob, Qaux_glob, parameters, jnp.zeros_like(Q_glob)
    )
    Q_warm.block_until_ready()
    t_r0 = time.perf_counter()
    Q_cur = Q_glob
    for _ in range(N_STEPS):
        Q_cur = Q_cur + dt * flux_op_global(
            dt, t_j, Q_cur, Qaux_glob, parameters,
            jnp.zeros_like(Q_cur),
        )
    Q_cur.block_until_ready()
    t_run_global = time.perf_counter() - t_r0
    print(f"BASELINE: {N_STEPS} steps wall = {t_run_global:.3f}s "
          f"({1e3 * t_run_global / N_STEPS:.2f} ms/step)")

    # ─── Partition + local meshes.
    print(f"\nPartition + extract local meshes...")
    parts_info = partition_mesh(mesh_np, n_parts=N_PARTS)
    local_meshes = [extract_local_mesh(mesh_np, p) for p in parts_info]
    print(f"local shapes: "
          f"{[(int(lm.n_inner_cells), int(lm.n_cells)) for lm in local_meshes]}")

    # ─── Build per-rank infrastructure.
    Q_glob_np = np.asarray(Q_glob)
    Qaux_glob_np = np.asarray(Qaux_glob)

    # For each rank: per-neighbor send/recv index lists.
    # send_to[i][j] = local indices in rank i's array that are needed by rank j.
    # recv_from[i][j] = local indices in rank i's array that receive from rank j.
    membership = -np.ones(nc_global, dtype=int)
    for r, p in enumerate(parts_info):
        membership[np.asarray(p.owned_cells)] = r

    send_to = {}     # (i, j) -> np.array of local indices in i's array
    recv_from = {}   # (i, j) -> np.array of local indices in i's array
    for i, p_info in enumerate(parts_info):
        n_owned = int(local_meshes[i].n_inner_cells)
        # recv: which ghost cells (local indices > n_owned) come from each src rank?
        for k, gc in enumerate(np.asarray(p_info.ghost_cells)):
            if gc < nc_global:
                src = int(membership[gc])
                recv_from.setdefault((i, src), []).append(n_owned + k)
        # send: which owned cells (local indices < n_owned) are needed by each dst rank?
        # Cell i_owned is sent to rank j if j's recv_map includes i.
        # Easier: iterate over OTHER ranks' ghost lists.
        for j, q_info in enumerate(parts_info):
            if j == i:
                continue
            for k, gc in enumerate(np.asarray(q_info.ghost_cells)):
                if gc < nc_global and int(membership[gc]) == i:
                    # gc is owned by i.  Its local index in i's array
                    # = position in i's owned_cells.
                    local_idx_in_i = int(np.where(
                        np.asarray(p_info.owned_cells) == gc)[0][0])
                    send_to.setdefault((i, j), []).append(local_idx_in_i)

    for k, v in send_to.items():
        send_to[k] = np.asarray(v, dtype=np.int32)
    for k, v in recv_from.items():
        recv_from[k] = np.asarray(v, dtype=np.int32)

    print(f"\nSend/recv stats:")
    for (i, j), idx in send_to.items():
        print(f"  rank {i} -> rank {j}: {len(idx)} cells")

    # ─── Build per-rank jit'd "step with halo" functions.
    rank_step_with_halo = []
    rank_gather_halo_to_send = {}  # (i, j) -> jit'd gather
    rank_n_owned = []
    rank_n_local = []
    for i, (p_info, lm) in enumerate(zip(parts_info, local_meshes)):
        n_local = int(lm.n_cells)
        n_owned = int(lm.n_inner_cells)
        rank_n_owned.append(n_owned)
        rank_n_local.append(n_local)
        lm_jax = convert_mesh_to_jax(lm)
        flux_op_local = solver.get_flux_operator(lm_jax, runtime)

        # Per-neighbor halo recv positions in this rank's array.
        recv_positions = {
            j: jnp.asarray(recv_from[(i, j)])
            for j in range(N_PARTS) if (i, j) in recv_from
        }

        def make_step(flux_op_local, recv_positions, n_owned, i_rank=i):
            @jax.jit
            def step(Q_local, Qaux_local, *halo_in_per_src):
                # halo_in_per_src: list of halo data arrays, one per
                # source rank (in the order of sorted neighbor IDs).
                Q = Q_local
                sorted_neighbors = sorted(recv_positions.keys())
                for k, src in enumerate(sorted_neighbors):
                    Q = Q.at[:, recv_positions[src]].set(halo_in_per_src[k])
                # Flux op.
                dQ = flux_op_local(
                    dt, t_j, Q, Qaux_local, parameters,
                    jnp.zeros_like(Q),
                )
                # Update owned cells only (halo cells stay as halo
                # data; they get refreshed next step).
                Q_new = Q.at[:, :n_owned].add(dt * dQ[:, :n_owned])
                return Q_new
            return step

        sorted_neighbors_i = sorted(recv_positions.keys())
        rank_step_with_halo.append(
            (make_step(flux_op_local, recv_positions, n_owned),
             sorted_neighbors_i)
        )

        # Per-neighbor gather: given rank i's Q_local, extract data to send to j.
        for j in range(N_PARTS):
            if (i, j) in send_to:
                send_idx = jnp.asarray(send_to[(i, j)])
                def make_gather(send_idx_j):
                    @jax.jit
                    def gather(Q_local):
                        return Q_local[:, send_idx_j]
                    return gather
                rank_gather_halo_to_send[(i, j)] = make_gather(send_idx)

    # ─── Build initial per-device Q_local arrays.
    print(f"\nPlacing Q_local on each device + compiling step functions...")
    Q_local_per_dev = []
    Qaux_local_per_dev = []
    for i, p_info in enumerate(parts_info):
        n_local = rank_n_local[i]
        n_owned = rank_n_owned[i]
        Q_local_np = np.zeros((Q_glob_np.shape[0], n_local), dtype=Q_glob_np.dtype)
        Qaux_local_np = np.zeros((Qaux_glob_np.shape[0], n_local), dtype=Qaux_glob_np.dtype)
        Q_local_np[:, :n_owned] = Q_glob_np[:, p_info.owned_cells]
        Qaux_local_np[:, :n_owned] = Qaux_glob_np[:, p_info.owned_cells]
        for k, gc in enumerate(np.asarray(p_info.ghost_cells)):
            if gc < nc_global:
                Q_local_np[:, n_owned + k] = Q_glob_np[:, gc]
                Qaux_local_np[:, n_owned + k] = Qaux_glob_np[:, gc]
        Q_local_per_dev.append(jax.device_put(jnp.asarray(Q_local_np), devices[i]))
        Qaux_local_per_dev.append(jax.device_put(jnp.asarray(Qaux_local_np), devices[i]))

    # ─── Compile warm-up: one step.
    t_c0 = time.perf_counter()
    halo_pre = {(i, j): rank_gather_halo_to_send[(i, j)](Q_local_per_dev[i])
                for (i, j) in send_to}
    for _, fut in halo_pre.items():
        fut.block_until_ready()
    new_Q_local = []
    for i in range(N_PARTS):
        step_fn, sorted_neighbors = rank_step_with_halo[i]
        halos = [jax.device_put(halo_pre[(src, i)], devices[i])
                 for src in sorted_neighbors]
        new_Q_local.append(
            step_fn(Q_local_per_dev[i], Qaux_local_per_dev[i], *halos)
        )
    for q in new_Q_local:
        q.block_until_ready()
    t_compile = time.perf_counter() - t_c0
    print(f"compile = {t_compile:.2f}s")

    # ─── Timed loop.
    print(f"\nTimed loop ({N_STEPS} steps)...")
    Q_state = list(Q_local_per_dev)
    t_r0 = time.perf_counter()
    for s in range(N_STEPS):
        # 1. Each rank gathers its halo-send data.
        halo_send = {}
        for (i, j) in send_to:
            halo_send[(i, j)] = rank_gather_halo_to_send[(i, j)](Q_state[i])
        # 2. Cross-device transfer.
        # 3. Each rank applies halo + flux + update (one jit'd trace).
        new_Q_state = []
        for i in range(N_PARTS):
            step_fn, sorted_neighbors = rank_step_with_halo[i]
            halos_for_i = [jax.device_put(halo_send[(src, i)], devices[i])
                           for src in sorted_neighbors]
            new_Q_state.append(
                step_fn(Q_state[i], Qaux_local_per_dev[i], *halos_for_i)
            )
        Q_state = new_Q_state
    for q in Q_state:
        q.block_until_ready()
    t_run = time.perf_counter() - t_r0
    print(f"wall = {t_run:.3f}s ({1e3 * t_run / N_STEPS:.2f} ms/step)")

    speedup = t_run_global / t_run if t_run > 0 else float("nan")
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  Single-device:         {t_run_global:.3f}s "
          f"({1e3 * t_run_global / N_STEPS:.2f} ms/step)")
    print(f"  SPMD-{N_PARTS} persistent:   {t_run:.3f}s "
          f"({1e3 * t_run / N_STEPS:.2f} ms/step)")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  ─────────────────────────────────────────────────────")

    # ─── Bit-identity check against single-device.
    Q_global_final = Q_glob
    for _ in range(N_STEPS):
        Q_global_final = Q_global_final + dt * flux_op_global(
            dt, t_j, Q_global_final, Qaux_glob, parameters,
            jnp.zeros_like(Q_global_final),
        )
    Q_global_final.block_until_ready()

    max_err = 0.0
    for i in range(N_PARTS):
        spmd_owned = np.asarray(Q_state[i])[:, :rank_n_owned[i]]
        ref_owned = np.asarray(Q_global_final)[:, parts_info[i].owned_cells]
        err = float(np.max(np.abs(spmd_owned - ref_owned)))
        max_err = max(max_err, err)
    print(f"\n  Bit-identity vs single-device after {N_STEPS} steps: "
          f"max_err = {max_err:.3e}")
    if max_err < 1e-5:
        print("  CORRECTNESS: PASS")
    else:
        print("  CORRECTNESS: FAIL")

    print(f"\nTotal wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
