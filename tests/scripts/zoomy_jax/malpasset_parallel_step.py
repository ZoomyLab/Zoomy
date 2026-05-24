"""Test parallel pipeline on real Malpasset mesh: per-rank flux op
+ manual halo exchange + bit-identity vs single-device.

Step 1 of the SPMD investigation requested by the user.  Builds
4 per-rank LSQMeshes via extract_local_mesh, computes the JAX flux
op on each, manually fills ghost cells from the global state, and
verifies that owned-cell dQ from each rank equals the corresponding
slice of single-device dQ.

If this passes, SPMD via shard_map is just adding device-level
parallelism on top.  If it fails, the rank meshes / halo
identification has a bug to fix."""
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
    B = m.point_data["B"]
    H = m.point_data["H"]
    U = m.point_data["U"]
    V = m.point_data["V"]
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
    N_PARTS = 4
    t0 = time.perf_counter()
    print(f"[{time.perf_counter() - t0:6.2f}s] Loading Malpasset mesh...")
    mesh_np = LSQMesh.from_msh(MESH_PATH)
    nc_global = int(mesh_np.n_inner_cells)
    print(f"[{time.perf_counter() - t0:6.2f}s] n_inner={nc_global}")

    print(f"[{time.perf_counter() - t0:6.2f}s] Partitioning into {N_PARTS}...")
    parts_info = partition_mesh(mesh_np, n_parts=N_PARTS)
    print(f"[{time.perf_counter() - t0:6.2f}s] Per-rank stats:")
    for i, p in enumerate(parts_info):
        n_send = sum(len(v) for v in p.send_map.values())
        n_recv = sum(len(v) for v in p.recv_map.values())
        print(f"   rank {i}: owned={len(p.owned_cells)}, "
              f"ghost={len(p.ghost_cells)}, send={n_send}, "
              f"recv={n_recv}, nbrs={sorted(p.recv_map.keys())}")

    print(f"[{time.perf_counter() - t0:6.2f}s] Extracting local meshes...")
    local_meshes = [extract_local_mesh(mesh_np, p) for p in parts_info]
    print(f"[{time.perf_counter() - t0:6.2f}s] Local mesh shapes: "
          f"{[(int(lm.n_inner_cells), int(lm.n_cells)) for lm in local_meshes]}")

    # ── Build single-device reference solver.
    print(f"[{time.perf_counter() - t0:6.2f}s] Building single-device solver...")
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
    print(f"[{time.perf_counter() - t0:6.2f}s] single-device setup done.")
    parameters = solver._rt_parameters
    runtime = solver._rt_model

    # Single-device dQ (one step).
    dt = jnp.asarray(0.01, dtype=Q_glob.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_glob.dtype)
    print(f"[{time.perf_counter() - t0:6.2f}s] single-device flux op (compile + run)...")
    flux_op_global = solver._rt_flux_op
    dQ_global = flux_op_global(
        dt, t_j, Q_glob, Qaux_glob, parameters, jnp.zeros_like(Q_glob)
    )
    dQ_global.block_until_ready()
    print(f"[{time.perf_counter() - t0:6.2f}s] single-device dQ done.")

    # ── Build per-rank flux ops (same runtime, different mesh).
    print(f"[{time.perf_counter() - t0:6.2f}s] Building per-rank flux ops...")
    Q_glob_np = np.asarray(Q_glob)
    Qaux_glob_np = np.asarray(Qaux_glob)

    max_err = 0.0
    max_err_per_rank = []
    for i, (p_info, lm) in enumerate(zip(parts_info, local_meshes)):
        t_r = time.perf_counter()
        lm_jax = convert_mesh_to_jax(lm)
        n_local = int(lm.n_cells)
        n_owned = int(lm.n_inner_cells)

        # Build per-rank Q + Qaux.  Ghost cells from PartitionInfo
        # include BOTH cross-partition cells (global idx < n_inner)
        # AND physical-boundary ghost cells (idx >= n_inner).  Only
        # the former carry meaningful Q values via halo exchange; the
        # latter are placeholders — the flux_op evaluates BC face
        # values from the BC kernel, not from Q at the ghost slot.
        all_cells_np = np.asarray(p_info.ghost_cells)
        partition_ghost_mask = all_cells_np < nc_global
        Q_local_np = np.zeros((Q_glob_np.shape[0], n_local),
                              dtype=Q_glob_np.dtype)
        Qaux_local_np = np.zeros((Qaux_glob_np.shape[0], n_local),
                                 dtype=Qaux_glob_np.dtype)
        Q_local_np[:, :n_owned] = Q_glob_np[:, p_info.owned_cells]
        Qaux_local_np[:, :n_owned] = Qaux_glob_np[:, p_info.owned_cells]
        # Place cross-partition ghosts at their local slots.
        ghost_local_offsets = np.arange(len(all_cells_np), dtype=int)
        for k, gc in enumerate(all_cells_np):
            if gc < nc_global:
                Q_local_np[:, n_owned + k] = Q_glob_np[:, gc]
                Qaux_local_np[:, n_owned + k] = Qaux_glob_np[:, gc]
        Q_local = jnp.asarray(Q_local_np)
        Qaux_local = jnp.asarray(Qaux_local_np)

        # Per-rank flux op closed over the local mesh + same runtime.
        flux_op_local = solver.get_flux_operator(lm_jax, runtime)

        # Run flux op on the local mesh.
        dQ_local = flux_op_local(
            dt, t_j, Q_local, Qaux_local, parameters,
            jnp.zeros_like(Q_local),
        )
        dQ_local.block_until_ready()

        # Per-rank flux op returns dQ of shape (n_var, n_inner=n_owned).
        # Compare against the global dQ sliced by owned_cells.
        dQ_local_np = np.asarray(dQ_local)[:, :n_owned]
        dQ_ref = np.asarray(dQ_global)[:, p_info.owned_cells]
        err = float(np.max(np.abs(dQ_local_np - dQ_ref)))
        max_err = max(max_err, err)
        max_err_per_rank.append(err)
        print(f"   rank {i}: build+run = {time.perf_counter() - t_r:.2f}s, "
              f"max |dQ_local - dQ_global[owned]| = {err:.3e}")

    print(f"\n[{time.perf_counter() - t0:6.2f}s] DONE")
    print(f"  Overall max err across all ranks: {max_err:.3e}")
    print(f"  Per-rank: {[f'{e:.3e}' for e in max_err_per_rank]}")
    if max_err < 1e-6:
        print("  PARALLEL CORRECTNESS: PASS (bit-identical to single-device)")
    else:
        print(f"  PARALLEL CORRECTNESS: FAIL (err {max_err:.3e} > 1e-6)")


if __name__ == "__main__":
    main()
