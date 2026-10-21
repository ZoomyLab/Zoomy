import argparse
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jraph
import numpy as np


def _ensure_local_imports() -> None:
    """Allow running this script directly from source checkout."""
    repo_root = Path(__file__).resolve().parents[4]
    core_path = repo_root / "library" / "zoomy_core"
    jax_path = repo_root / "library" / "zoomy_jax"
    for path in (core_path, jax_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _maybe_generate_mesh(mesh_geo: Path, mesh_msh: Path) -> None:
    if mesh_msh.exists():
        return
    cmd = ["gmsh", "-2", str(mesh_geo), "-o", str(mesh_msh)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gmsh not found. Install gmsh or provide an existing .msh file."
        ) from exc


class TinyMessagePassing(eqx.Module):
    """Fixed-weight, no-training scalar message passing model."""

    w_node: jnp.ndarray
    w_edge: jnp.ndarray
    b_node: jnp.ndarray

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        sender_vals = graph.nodes[graph.senders, 0]
        edge_vals = graph.edges[:, 0]

        messages = self.w_edge * edge_vals + self.w_node * sender_vals
        agg = jraph.segment_sum(
            messages,
            graph.receivers,
            num_segments=graph.nodes.shape[0],
        )
        out = self.w_node * graph.nodes[:, 0] + agg + self.b_node
        return out


def _build_cell_graph(mesh) -> jraph.GraphsTuple:
    # Keep only interior cells as graph nodes.
    n_nodes = int(mesh.n_inner_cells)

    face_cells = np.asarray(mesh.face_cells)
    senders = face_cells[0]
    receivers = face_cells[1]

    # Drop faces touching ghost cells.
    valid = (senders < n_nodes) & (receivers < n_nodes)
    senders = senders[valid]
    receivers = receivers[valid]

    # Make graph undirected by adding reverse edges.
    rev_senders = receivers.copy()
    rev_receivers = senders.copy()
    senders = np.concatenate([senders, rev_senders])
    receivers = np.concatenate([receivers, rev_receivers])

    centers = np.asarray(mesh.cell_centers)[:, :n_nodes].T  # (n_nodes, dim)
    rel = centers[receivers] - centers[senders]
    edge_len = np.linalg.norm(rel, axis=1, keepdims=True)

    # Use scalar node feature = x+y and scalar edge feature = edge length.
    if centers.shape[1] == 1:
        node_scalar = centers[:, :1]
    else:
        node_scalar = centers[:, :1] + centers[:, 1:2]

    return jraph.GraphsTuple(
        nodes=jnp.asarray(node_scalar),
        edges=jnp.asarray(edge_len),
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        n_node=jnp.asarray([n_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )


def _load_mesh(mesh_file: Path):
    from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh

    if mesh_file.suffix == ".h5":
        return Mesh.from_hdf5(str(mesh_file))
    if mesh_file.suffix == ".msh":
        return Mesh.from_gmsh(str(mesh_file))
    raise ValueError(f"Unsupported mesh format: {mesh_file}")


def _resolve_mesh_file(repo_root: Path, args) -> Path:
    mesh_geo = args.mesh_geo if args.mesh_geo.is_absolute() else (repo_root / args.mesh_geo)
    mesh_msh = args.mesh_msh if args.mesh_msh.is_absolute() else (repo_root / args.mesh_msh)
    mesh_h5 = args.mesh_h5 if args.mesh_h5.is_absolute() else (repo_root / args.mesh_h5)

    # Preferred path: download/use HDF5 mesh to bypass petsc4py.
    if args.mesh_name:
        from zoomy_core.mesh.mesh_catalog import MeshCatalog
        catalog = MeshCatalog()
        return catalog.download(args.mesh_name, size="medium", filetype="h5", folder=mesh_h5.parent)

    if mesh_h5.exists():
        return mesh_h5

    if args.prefer_h5:
        raise FileNotFoundError(
            f"Preferred HDF5 mesh not found: {mesh_h5}. "
            "Provide --mesh-name to download from Zoomy mesh database or pass --prefer-h5 false to use .msh path."
        )

    if not mesh_msh.exists():
        _maybe_generate_mesh(mesh_geo, mesh_msh)
    return mesh_msh


def run_demo(mesh_file: Path, output_h5: Path, vtk_name: str) -> None:
    _ensure_local_imports()

    from zoomy_core.misc import io as core_io
    from zoomy_jax.mesh.mesh import convert_mesh_to_jax
    from zoomy_jax.misc.io import get_save_fields

    mesh = _load_mesh(mesh_file)
    mesh_jax = convert_mesh_to_jax(mesh)

    graph = _build_cell_graph(mesh_jax)

    # Fixed deterministic "random" weights for no-training blueprint.
    key = jax.random.PRNGKey(7)
    w_node = jax.random.uniform(key, (), minval=0.1, maxval=0.3)
    w_edge = jnp.array(0.5)
    b_node = jnp.array(0.05)
    model = TinyMessagePassing(w_node=w_node, w_edge=w_edge, b_node=b_node)

    pred = model(graph)  # shape: (n_inner_cells,)

    # Match Zoomy field format: Q=(n_fields, n_cells), Qaux=(n_aux, n_cells)
    Q = jnp.zeros((1, mesh_jax.n_cells), dtype=jnp.float64)
    Q = Q.at[0, : mesh_jax.n_inner_cells].set(pred)
    Qaux = jnp.zeros((1, mesh_jax.n_cells), dtype=jnp.float64)

    core_io.init_output_directory(str(output_h5.parent), clean=False)
    mesh_jax.write_to_hdf5(str(output_h5))

    save_fields = get_save_fields(str(output_h5), write_all=True)
    _ = save_fields(
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
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
        aux_field_names=["aux_dummy"],
    )

    print(f"Loaded mesh: {mesh_file}")
    print(f"Wrote HDF5: {output_h5}")
    print(f"Wrote VTK series: {output_h5.parent / (vtk_name + '.pvd')}")


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Zoomy + Jraph no-training blueprint")
    repo_root = Path(__file__).resolve().parents[4]

    parser.add_argument(
        "--mesh-name",
        type=str,
        default="",
        help="Optional mesh basename from Zoomy mesh database (downloads .h5 and bypasses petsc4py)",
    )
    parser.add_argument(
        "--mesh-geo",
        type=Path,
        default=Path("meshes/channel_quad_2d/mesh.geo"),
        help="Path to gmsh .geo file (used if --prefer-h5 false and .msh missing)",
    )
    parser.add_argument(
        "--mesh-msh",
        type=Path,
        default=Path("meshes/channel_quad_2d/mesh.msh"),
        help="Path to gmsh .msh file (requires petsc4py)",
    )
    parser.add_argument(
        "--mesh-h5",
        type=Path,
        default=Path("meshes/channel_quad_2d/mesh.h5"),
        help="Path to mesh .h5 file (preferred; no petsc4py required)",
    )
    parser.add_argument(
        "--prefer-h5",
        type=_str2bool,
        default=True,
        help="If true, require .h5 mesh unless --mesh-name is provided.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=Path("outputs/gnn_blueprint/gnn_blueprint.h5"),
        help="Output HDF5 path (Zoomy-style)",
    )
    parser.add_argument(
        "--vtk-name",
        type=str,
        default="gnn_blueprint",
        help="Base name for VTK files",
    )
    args = parser.parse_args()

    output_h5 = args.output_h5 if args.output_h5.is_absolute() else (repo_root / args.output_h5)
    mesh_file = _resolve_mesh_file(repo_root, args)
    run_demo(mesh_file, output_h5, args.vtk_name)


if __name__ == "__main__":
    main()
