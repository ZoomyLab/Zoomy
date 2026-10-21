import sys
from pathlib import Path

import jax.numpy as jnp
import jraph
import numpy as np


def ensure_local_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    for p in (repo_root / "library" / "zoomy_core", repo_root / "library" / "zoomy_jax"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo_root


def _get_catalog():
    ensure_local_imports()
    from zoomy_core.mesh.mesh_catalog import MeshCatalog
    return MeshCatalog()


def download_mesh_h5(mesh_name: str, out_path: Path) -> Path:
    catalog = _get_catalog()
    return catalog.download(mesh_name, size="medium", filetype="h5", folder=out_path.parent)


def resolve_mesh_h5(repo_root: Path, mesh_name: str, mesh_h5: Path) -> Path:
    p = mesh_h5 if mesh_h5.is_absolute() else (repo_root / mesh_h5)
    if mesh_name:
        p = p.parent / f"{mesh_name}.h5"
        if not p.exists():
            p = download_mesh_h5(mesh_name, p)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")
    return p


def load_mesh_jax(mesh_h5: Path):
    ensure_local_imports()
    from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
    from zoomy_jax.mesh.mesh import convert_mesh_to_jax

    mesh = Mesh.from_hdf5(str(mesh_h5))
    return convert_mesh_to_jax(mesh)


def build_cell_graph(mesh):
    n_inner = int(mesh.n_inner_cells)
    face_cells = np.asarray(mesh.face_cells)

    s = face_cells[0]
    r = face_cells[1]
    valid = (s < n_inner) & (r < n_inner)
    s = s[valid]
    r = r[valid]

    senders = np.concatenate([s, r])
    receivers = np.concatenate([r, s])

    centers = np.asarray(mesh.cell_centers)[:, :n_inner].T
    rel = centers[receivers] - centers[senders]
    edge_len = np.linalg.norm(rel, axis=1, keepdims=True)

    node_base = centers[:, :1] + (centers[:, 1:2] if centers.shape[1] > 1 else 0.0)

    graph = jraph.GraphsTuple(
        nodes=jnp.asarray(node_base),
        edges=jnp.asarray(edge_len),
        senders=jnp.asarray(senders, dtype=jnp.int32),
        receivers=jnp.asarray(receivers, dtype=jnp.int32),
        n_node=jnp.asarray([n_inner], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        globals=None,
    )
    return graph, jnp.asarray(centers), jnp.asarray(edge_len[:, 0])


def boundary_class_id(mesh):
    n_inner = int(mesh.n_inner_cells)
    cls = np.zeros(n_inner, dtype=np.int32)
    b_cells = np.asarray(mesh.boundary_face_cells)
    b_funcs = np.asarray(mesh.boundary_face_function_numbers)
    for c, f in zip(b_cells, b_funcs):
        if c < n_inner:
            cls[c] = max(cls[c], int(f) + 1)
    return jnp.asarray(cls)


def global_adaptive_dt(mesh, q_inner):
    inradius = jnp.asarray(mesh.cell_inradius[: mesh.n_inner_cells])
    speed = jnp.abs(q_inner) + 1.0
    dt_cells = 0.2 * inradius / speed
    return jnp.min(dt_cells)
