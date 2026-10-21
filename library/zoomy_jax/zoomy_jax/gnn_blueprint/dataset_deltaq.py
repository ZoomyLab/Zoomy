import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

try:
    from .common import (
        boundary_class_id,
        build_cell_graph,
        ensure_local_imports,
        global_adaptive_dt,
        load_mesh_jax,
        resolve_mesh_h5,
    )
    from .models_deltaq import DeltaQGNN
except ImportError:
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from common import (
        boundary_class_id,
        build_cell_graph,
        ensure_local_imports,
        global_adaptive_dt,
        load_mesh_jax,
        resolve_mesh_h5,
    )
    from models_deltaq import DeltaQGNN


def run_synthetic_rollout(mesh, n_steps: int, n_fields: int, param_scale: float):
    graph, centers, _ = build_cell_graph(mesh)
    n_inner = int(mesh.n_inner_cells)

    # synthetic initial state with mesh + parameter dependence
    x = centers[:, 0]
    y = centers[:, 1] if centers.shape[1] > 1 else jnp.zeros_like(x)
    q = jnp.zeros((n_fields, n_inner), dtype=jnp.float64)
    for i in range(n_fields):
        q = q.at[i].set((1.0 + 0.15 * i) * jnp.sin(param_scale * x) + (0.2 + 0.05 * i) * jnp.cos(y))

    model = DeltaQGNN(n_fields=n_fields)
    cls = boundary_class_id(mesh)

    X_q = []
    X_dt = []
    X_cls = []
    Y_dq = []

    for _ in range(n_steps):
        dt = global_adaptive_dt(mesh, q[0])
        dq = model(graph, q, dt)
        q_next = q + dq

        X_q.append(np.asarray(q))
        X_dt.append(np.asarray(dt))
        X_cls.append(np.asarray(cls))
        Y_dq.append(np.asarray(dq))

        q = q_next

    return {
        "q": np.asarray(X_q),                # (steps, n_fields, n_cells)
        "dt": np.asarray(X_dt),              # (steps,)
        "class_id": np.asarray(X_cls),       # (steps, n_cells)
        "delta_q": np.asarray(Y_dq),         # (steps, n_fields, n_cells)
    }


def generate_dataset(mesh, n_fields: int, n_steps: int, param_values, out_file: Path):
    shards = []
    for p in param_values:
        shards.append(run_synthetic_rollout(mesh, n_steps=n_steps, n_fields=n_fields, param_scale=float(p)))

    q = np.concatenate([s["q"] for s in shards], axis=0)
    dt = np.concatenate([s["dt"] for s in shards], axis=0)
    class_id = np.concatenate([s["class_id"] for s in shards], axis=0)
    delta_q = np.concatenate([s["delta_q"] for s in shards], axis=0)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        q=q,
        dt=dt,
        class_id=class_id,
        delta_q=delta_q,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic deltaQ dataset with adaptive global dt")
    parser.add_argument("--mesh-name", type=str, default="channel_quad_2d")
    parser.add_argument("--mesh-h5", type=Path, default=Path("meshes/channel_quad_2d/channel_quad_2d.h5"))
    parser.add_argument("--n-fields", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--param-values", type=float, nargs="*", default=[0.8, 1.0, 1.2, 1.4])
    parser.add_argument("--out", type=Path, default=Path("outputs/gnn_blueprint/dataset_deltaq.npz"))
    args = parser.parse_args()

    repo_root = ensure_local_imports()
    mesh_h5 = resolve_mesh_h5(repo_root, args.mesh_name, args.mesh_h5)
    mesh = load_mesh_jax(mesh_h5)
    out = args.out if args.out.is_absolute() else (repo_root / args.out)
    generate_dataset(mesh, args.n_fields, args.n_steps, args.param_values, out)
    print(f"Saved dataset: {out}")


if __name__ == "__main__":
    main()
