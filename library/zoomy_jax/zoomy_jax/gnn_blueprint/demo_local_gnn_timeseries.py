import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jraph
import numpy as np


def _ensure_local_imports() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    core_path = repo_root / "library" / "zoomy_core"
    jax_path = repo_root / "library" / "zoomy_jax"
    for path in (core_path, jax_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class TinyMessagePassing(eqx.Module):
    w_node: jnp.ndarray
    w_edge: jnp.ndarray
    b_node: jnp.ndarray

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        sender_vals = graph.nodes[graph.senders, 0]
        edge_vals = graph.edges[:, 0]
        messages = self.w_edge * edge_vals + self.w_node * sender_vals
        agg = jraph.segment_sum(messages, graph.receivers, num_segments=graph.nodes.shape[0])
        return self.w_node * graph.nodes[:, 0] + agg + self.b_node


def _resolve_mesh_file(repo_root: Path, mesh_name: str, mesh_h5: Path) -> Path:
    if mesh_name:
        _ensure_local_imports()
        from zoomy_core.mesh.mesh_catalog import MeshCatalog
        catalog = MeshCatalog()
        return catalog.download(mesh_name, size="medium", filetype="h5", folder=mesh_h5.parent)
    return mesh_h5 if mesh_h5.is_absolute() else (repo_root / mesh_h5)


def _build_cell_graph(mesh) -> jraph.GraphsTuple:
    n_nodes = int(mesh.n_inner_cells)
    face_cells = np.asarray(mesh.face_cells)
    senders = face_cells[0]
    receivers = face_cells[1]
    valid = (senders < n_nodes) & (receivers < n_nodes)
    senders = senders[valid]
    receivers = receivers[valid]
    rev_senders = receivers.copy()
    rev_receivers = senders.copy()
    senders = np.concatenate([senders, rev_senders])
    receivers = np.concatenate([receivers, rev_receivers])

    centers = np.asarray(mesh.cell_centers)[:, :n_nodes].T
    rel = centers[receivers] - centers[senders]
    edge_len = np.linalg.norm(rel, axis=1, keepdims=True)
    node_scalar = centers[:, :1] + (centers[:, 1:2] if centers.shape[1] > 1 else 0.0)

    return jraph.GraphsTuple(
        nodes=jnp.asarray(node_scalar),
        edges=jnp.asarray(edge_len),
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        n_node=jnp.asarray([n_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )


def run_demo(mesh_file: Path, output_h5: Path, vtk_name: str, n_steps: int, dt: float) -> None:
    _ensure_local_imports()
    from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
    from zoomy_core.misc import io as core_io
    from zoomy_jax.mesh.mesh import convert_mesh_to_jax
    from zoomy_jax.misc.io import get_save_fields

    mesh = Mesh.from_hdf5(str(mesh_file))
    mesh_jax = convert_mesh_to_jax(mesh)
    graph = _build_cell_graph(mesh_jax)

    key = jax.random.PRNGKey(7)
    model = TinyMessagePassing(
        w_node=jax.random.uniform(key, (), minval=0.1, maxval=0.3),
        w_edge=jnp.array(0.5),
        b_node=jnp.array(0.05),
    )

    base_pred = model(graph)
    n_cells = mesh_jax.n_cells

    core_io.init_output_directory(str(output_h5.parent), clean=False)
    mesh_jax.write_to_hdf5(str(output_h5))
    save_fields = get_save_fields(str(output_h5), write_all=True)

    for i in range(n_steps):
        time = float(i * dt)
        signal = 1.0 + 0.2 * np.sin(2.0 * np.pi * time)
        pred_t = base_pred * signal

        Q = jnp.zeros((1, n_cells), dtype=jnp.float64)
        Q = Q.at[0, : mesh_jax.n_inner_cells].set(pred_t)
        Qaux = jnp.zeros((1, n_cells), dtype=jnp.float64)
        Qaux = Qaux.at[0, : mesh_jax.n_inner_cells].set(jnp.array(signal, dtype=jnp.float64))

        _ = save_fields(
            jnp.float64(time),
            jnp.float64(time),
            jnp.float64(i),
            Q,
            Qaux,
        )

    from zoomy_prepost import hdf5_to_vtk

    hdf5_to_vtk(
        str(output_h5),
        str(output_h5.parent),
        basename=vtk_name,
        include_aux=True,
        field_names=["gnn_scalar"],
        aux_field_names=["drive_signal"],
    )

    print(f"Loaded mesh: {mesh_file}")
    print(f"Saved {n_steps} snapshots to: {output_h5}")
    print(f"VTK series: {output_h5.parent / (vtk_name + '.pvd')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Zoomy + Jraph timeseries blueprint")
    repo_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--mesh-name", type=str, default="channel_quad_2d")
    parser.add_argument("--mesh-h5", type=Path, default=Path("meshes/channel_quad_2d/channel_quad_2d.h5"))
    parser.add_argument("--output-h5", type=Path, default=Path("outputs/gnn_blueprint/gnn_blueprint_timeseries.h5"))
    parser.add_argument("--vtk-name", type=str, default="gnn_blueprint_timeseries")
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    mesh_file = _resolve_mesh_file(repo_root, args.mesh_name, args.mesh_h5)
    output_h5 = args.output_h5 if args.output_h5.is_absolute() else (repo_root / args.output_h5)
    run_demo(mesh_file, output_h5, args.vtk_name, args.n_steps, args.dt)


if __name__ == "__main__":
    main()
