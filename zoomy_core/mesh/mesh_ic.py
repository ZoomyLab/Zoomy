"""Initial conditions from mesh-carried gmsh node data: node-to-cell interpolation plus a
`field_map` from state names to mesh fields, backend-agnostic.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

FieldSpec = Union[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]]


def _as_base_mesh(mesh_or_path):
    """Return a mesh carrying node/cell data, from a mesh object or a `.msh` path."""
    if isinstance(mesh_or_path, (str, bytes)) or hasattr(mesh_or_path, "__fspath__"):
        from zoomy_core.mesh.base_mesh import BaseMesh
        return BaseMesh.from_msh(str(mesh_or_path))
    return mesh_or_path


def interpolate_node_data_to_cells(
    mesh,
    node_values: np.ndarray,
    method: str = "nearest",
    query_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Interpolate a per-vertex field onto cells, by `"nearest"` vertex (KD-tree) or
    `"cell_average"` of the cell's vertices.

    Args:
        node_values: `(n_vertices,)` or `(n_vertices, k)`, aligned with
            `mesh.vertex_coordinates`.
        query_points: sample points for `"nearest"`, `(n, dim)`; default the inner cell centres.
    Returns:
        `(n_inner_cells,)` or `(n_inner_cells, k)` array.
    """
    node_values = np.asarray(node_values, dtype=float)
    dim = mesh.dimension
    n_inner = mesh.n_inner_cells

    if method == "cell_average":
        # Mesh-connectivity average: cell value = mean of its vertices.
        cell_verts = np.asarray(mesh.cell_vertices)[:, :n_inner]  # (vpc, n_inner)
        return node_values[cell_verts].mean(axis=0)

    if method == "nearest":
        from scipy.spatial import cKDTree
        verts = np.asarray(mesh.vertex_coordinates)[:dim, :].T  # (n_vertices, dim)
        if query_points is None:
            centers = mesh.cell_centers_computed()[:dim, :n_inner].T  # (n_inner, dim)
        else:
            centers = np.asarray(query_points, dtype=float)[:, :dim]
        tree = cKDTree(verts)
        _, idx = tree.query(centers)
        return node_values[idx]

    raise ValueError(f"unknown interpolation method {method!r} "
                     "(expected 'nearest' or 'cell_average')")


def initial_conditions_from_mesh(
    mesh_or_path,
    field_map: Mapping[str, FieldSpec],
    state_names: Optional[Sequence[str]] = None,
    method: str = "nearest",
    default: float = 0.0,
    query_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the per-cell initial state from mesh node data, mapping state names to mesh fields via
    `field_map`.

    Args:
        mesh_or_path: BaseMesh with `node_data` or a `.msh` path.
        field_map: `{state: mesh_field_name}` or ``{state: callable(node_fields) -> node
            array}``, evaluated at nodes then interpolated.
        state_names: row order of the result (default `field_map` order); rows not in
            `field_map` get `default`.
        method: node-to-cell interpolation, see `interpolate_node_data_to_cells`.
    Returns:
        `(n_state, n_cells)` array.
    """
    mesh = _as_base_mesh(mesh_or_path)
    node_fields: Dict[str, np.ndarray] = dict(getattr(mesh, "node_data", {}) or {})
    if not node_fields:
        raise ValueError(
            "mesh carries no node data ($NodeData); nothing to build an IC from. "
            "Load the .msh via BaseMesh.from_msh so mesh.node_data is populated.")

    if state_names is None:
        names = list(field_map.keys())
    else:
        names = [str(s) for s in state_names]
    idx = {n: k for k, n in enumerate(names)}

    if query_points is None:
        n_cells = mesh.n_inner_cells
    else:
        n_cells = len(query_points)
    Q = np.full((len(names), n_cells), float(default), dtype=float)

    for state_name, spec in field_map.items():
        if state_name not in idx:
            raise KeyError(
                f"field_map targets state {state_name!r} not in state_names "
                f"{names}")
        if callable(spec):
            node_arr = np.asarray(spec(node_fields), dtype=float)
        else:
            if spec not in node_fields:
                raise KeyError(
                    f"mesh field {spec!r} not in node data {list(node_fields)}")
            node_arr = np.asarray(node_fields[spec], dtype=float)
        Q[idx[state_name]] = interpolate_node_data_to_cells(
            mesh, node_arr, method=method, query_points=query_points)

    return Q
