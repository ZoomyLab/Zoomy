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
    for p in (repo_root / "library" / "zoomy_core", repo_root / "library" / "zoomy_jax"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _download_mesh_h5(mesh_name: str, out_path: Path) -> Path:
    _ensure_local_imports()
    from zoomy_core.mesh.mesh_catalog import MeshCatalog
    catalog = MeshCatalog()
    return catalog.download(mesh_name, size="medium", filetype="h5", folder=out_path.parent)


class TinyLocalExpert(eqx.Module):
    w_self: jnp.ndarray
    w_msg: jnp.ndarray
    bias: jnp.ndarray

    def __call__(self, self_val: jnp.ndarray, msg_val: jnp.ndarray, dt_global: jnp.ndarray) -> jnp.ndarray:
        # explicit-like update with global adaptive dt per step
        rhs = self.w_self * self_val + self.w_msg * msg_val + self.bias
        return self_val + dt_global * rhs


def _build_graph_and_features(mesh):
    n_inner = int(mesh.n_inner_cells)
    face_cells = np.asarray(mesh.face_cells)

    s = face_cells[0]
    r = face_cells[1]
    valid = (s < n_inner) & (r < n_inner)
    s = s[valid]
    r = r[valid]

    # undirected
    senders = np.concatenate([s, r])
    receivers = np.concatenate([r, s])

    centers = np.asarray(mesh.cell_centers)[:, :n_inner].T
    rel = centers[receivers] - centers[senders]
    edge_len = np.linalg.norm(rel, axis=1, keepdims=True)

    node_scalar = centers[:, :1] + (centers[:, 1:2] if centers.shape[1] > 1 else 0.0)

    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(node_scalar),
        edges=jnp.asarray(edge_len),
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        n_node=jnp.asarray([n_inner], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )

    return graph, jnp.asarray(centers), jnp.asarray(edge_len[:, 0])


def _cell_classes_from_mesh(mesh):
    """
    Class IDs from mesh metadata only:
      0 = interior
      k>0 = boundary function number (k-1) + 1
    """
    n_inner = int(mesh.n_inner_cells)
    cls = np.zeros(n_inner, dtype=np.int32)

    b_cells = np.asarray(mesh.boundary_face_cells)
    b_funcs = np.asarray(mesh.boundary_face_function_numbers)

    # map each boundary-adjacent cell to its BC function number (+1)
    for c, f in zip(b_cells, b_funcs):
        if c < n_inner:
            cls[c] = max(cls[c], int(f) + 1)

    return jnp.asarray(cls)


def _make_experts(n_classes: int):
    experts = []
    for i in range(n_classes):
        # deterministic hand-set weights for trivial blueprint
        experts.append(
            TinyLocalExpert(
                w_self=jnp.array(0.05 + 0.01 * i),
                w_msg=jnp.array(0.03 + 0.005 * i),
                bias=jnp.array(0.01 * (i + 1)),
            )
        )
    return experts


def _global_adaptive_dt(mesh, q_inner):
    """Global adaptive dt from cellwise CFL-like candidates."""
    inradius = jnp.asarray(mesh.cell_inradius[: mesh.n_inner_cells])
    speed = jnp.abs(q_inner) + 1.0
    dt_cells = 0.15 * inradius / speed
    return jnp.min(dt_cells)


def run_demo(mesh_file: Path, output_h5: Path, vtk_name: str, n_steps: int):
    _ensure_local_imports()
    from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
    from zoomy_core.misc import io as core_io
    from zoomy_jax.mesh.mesh import convert_mesh_to_jax
    from zoomy_jax.misc.io import get_save_fields

    mesh = Mesh.from_hdf5(str(mesh_file))
    mesh_jax = convert_mesh_to_jax(mesh)

    graph, centers, edge_lengths = _build_graph_and_features(mesh_jax)
    class_id = _cell_classes_from_mesh(mesh_jax)
    n_classes = int(jnp.max(class_id)) + 1
    experts = _make_experts(n_classes)

    n_cells = int(mesh_jax.n_cells)
    n_inner = int(mesh_jax.n_inner_cells)

    # initial state (one scalar per inner cell)
    q_inner = graph.nodes[:, 0]

    core_io.init_output_directory(str(output_h5.parent), clean=False)
    mesh_jax.write_to_hdf5(str(output_h5))
    save_fields = get_save_fields(str(output_h5), write_all=True)

    for step in range(n_steps):
        # local message aggregation from current state
        msg = jraph.segment_sum(q_inner[graph.senders], graph.receivers, num_segments=n_inner)

        dt_global = _global_adaptive_dt(mesh_jax, q_inner)

        q_next = q_inner
        for cid, expert in enumerate(experts):
            mask = class_id == cid
            upd = expert(q_inner, msg, dt_global)
            q_next = jnp.where(mask, upd, q_next)

        q_inner = q_next

        Q = jnp.zeros((1, n_cells), dtype=jnp.float64)
        Q = Q.at[0, :n_inner].set(q_inner)

         # Qaux[0]=class_id, Qaux[1]=global adaptive dt
        Qaux = jnp.zeros((2, n_cells), dtype=jnp.float64)
        Qaux = Qaux.at[0, :n_inner].set(class_id.astype(jnp.float64))
        Qaux = Qaux.at[1, :n_inner].set(jnp.full((n_inner,), dt_global))

        t = jnp.float64(step)
        _ = save_fields(t, t, jnp.float64(step), Q, Qaux)

    from zoomy_prepost import hdf5_to_vtk

    hdf5_to_vtk(
        str(output_h5),
        str(output_h5.parent),
        basename=vtk_name,
        include_aux=True,
        field_names=["q_local_expert"],
        aux_field_names=["class_id", "dt_global"],
    )

    print(f"Loaded mesh: {mesh_file}")
    print(f"Boundary classes used: {n_classes} (0=interior, >0 from boundary_function_number+1)")
    print(f"Saved {n_steps} snapshots to: {output_h5}")
    print(f"VTK series: {output_h5.parent / (vtk_name + '.pvd')}")


def main():
    parser = argparse.ArgumentParser(description="Local-expert transient GNN blueprint")
    repo_root = Path(__file__).resolve().parents[4]

    parser.add_argument("--mesh-name", type=str, default="channel_quad_2d")
    parser.add_argument("--mesh-h5", type=Path, default=Path("meshes/channel_quad_2d/channel_quad_2d.h5"))
    parser.add_argument("--output-h5", type=Path, default=Path("outputs/gnn_blueprint/gnn_local_experts_transient.h5"))
    parser.add_argument("--vtk-name", type=str, default="gnn_local_experts_transient")
    parser.add_argument("--n-steps", type=int, default=12)
    args = parser.parse_args()

    mesh_h5 = args.mesh_h5 if args.mesh_h5.is_absolute() else (repo_root / args.mesh_h5)
    if args.mesh_name:
        mesh_h5 = mesh_h5.parent / f"{args.mesh_name}.h5"
        if not mesh_h5.exists():
            mesh_h5 = _download_mesh_h5(args.mesh_name, mesh_h5)

    output_h5 = args.output_h5 if args.output_h5.is_absolute() else (repo_root / args.output_h5)
    run_demo(mesh_h5, output_h5, args.vtk_name, args.n_steps)


if __name__ == "__main__":
    main()
