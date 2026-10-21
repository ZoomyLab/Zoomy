"""BC-aware cell→vertex (point) data interpolation for post-processing.

`interpolate_point_data_with_bcs` lifts cell-centred `Q` (and `Qaux`)
fields to mesh vertices while honouring the boundary conditions: each boundary
face contributes its BC-consistent face value (via the BC's `face_value`
kernel) to its vertices, instead of every vertex seeing only interior cells.

For a straight `Wall` this makes the boundary vertex's normal momentum vanish
(no-penetration): the wall `face_value` returns the reflected ghost state
(normal momentum negated), so averaging it with the single interior cell cancels
the normal component exactly.  Interior vertices are untouched (no incident
boundary face) — identical to the plain unweighted cell→vertex mean used
elsewhere (`zoomy_plotting.mesh.faces.cell_to_vert_values`), so C0 continuity
is preserved.

This is a reusable post-processing primitive (3-D reconstruction generator, 2-D
enrichers, the steffler secondary-flow cross-sections — task 0012 / REQ-01).
"""
from __future__ import annotations

import numpy as np

from zoomy_core.mesh.base_mesh import _local_faces_for_type
from zoomy_core.mesh.lsq_reconstruction import build_vertex_to_cells
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, resolve_per_field,
)

# (spatial dim, vertices-per-cell) → element type understood by
# ``_local_faces_for_type``.  The runtime meshes don't carry a ``mesh_type``
# attribute, but the element is unambiguous from these two shapes.
_MESH_TYPE_BY_SHAPE = {
    (1, 2): "line", (2, 3): "triangle", (2, 4): "quad",
    (3, 4): "tetra", (3, 8): "hexahedron",
}


def _infer_mesh_type(mesh) -> str:
    dim = int(np.asarray(mesh.vertex_coordinates).shape[0])
    verts_per_cell = int(np.asarray(mesh.cell_vertices).shape[0])
    try:
        return _MESH_TYPE_BY_SHAPE[(dim, verts_per_cell)]
    except KeyError as exc:
        raise ValueError(
            f"cannot infer element type from dim={dim}, "
            f"verts_per_cell={verts_per_cell}"
        ) from exc


def _boundary_face_vertices(mesh, mesh_type):
    """For each boundary face, the global vertex indices lying on it.

    Recovered exactly from the mesh's own `cell_faces` map and the local
    face→vertex template `_local_faces_for_type` (the same template the
    topology builder used), so the per-face vertex set is consistent with the
    mesh's stored face indexing — no geometric tolerance needed.
    """
    local_faces = _local_faces_for_type(mesh_type)
    cell_faces = np.asarray(mesh.cell_faces)
    cell_vertices = np.asarray(mesh.cell_vertices)
    bf_fidx = np.asarray(mesh.boundary_face_face_indices)
    bf_cells = np.asarray(mesh.boundary_face_cells)
    out = []
    for bf in range(len(bf_fidx)):
        fidx, ic = int(bf_fidx[bf]), int(bf_cells[bf])
        lf_matches = np.where(cell_faces[:, ic] == fidx)[0]
        if lf_matches.size == 0:
            raise ValueError(
                f"boundary face {bf} (global {fidx}) is not a face of its "
                f"adjacent cell {ic}; mesh topology inconsistent."
            )
        lf = int(lf_matches[0])
        out.append([int(cell_vertices[k, ic]) for k in local_faces[lf]])
    return out


def _resolve_bc_dict(bcs, state_names, aliases):
    """Return `{tag_name: BC}` with numpy `face_value` kernels, from a
    resolved `BoundaryConditions`, a `{tag: BC}` dict, or a raw
    per-field list (resolved via `resolve_per_field`)."""
    if bcs is None:
        return {}
    if isinstance(bcs, BoundaryConditions):
        return bcs.boundary_conditions_list_dict
    if isinstance(bcs, dict):
        return bcs
    aliases = {"momentum": "q"} if aliases is None else aliases
    return resolve_per_field(bcs, state_names, aliases).boundary_conditions_list_dict


def interpolate_point_data_with_bcs(
    cell_Q, mesh, boundary_conditions, *,
    state_names,
    cell_Qaux=None,
    aux_names=None,
    aux_boundary_conditions=None,
    parameters=None,
    time=0.0,
    aliases=None,
):
    """Lift cell-centred `Q` (and optionally `Qaux`) to vertices, BC-aware.

    Parameters
    ----------
    cell_Q : array (n_state, n_inner_cells)
        Cell-centred state.  Full state vector per cell — the BC `face_value`
        kernels reflect/replace whole-state slots (a `Wall` needs all momentum
        components together to reflect the face normal).
    mesh : BaseMesh / FVMMesh / LSQMesh
        Unstructured mesh with `cell_vertices` / `cell_faces` /
        `boundary_face_*` connectivity.
    boundary_conditions : BoundaryConditions | dict | list
        The Q boundary conditions.  A raw per-field list (e.g.
        `[Wall("wall", on="momentum"), ...]`) is resolved via
        `resolve_per_field` against `state_names` (default alias
        `{"momentum": "q"}`, matching `resolve_and_attach`).
    state_names : sequence of str
        `[str(s) for s in sm.state]` — needed to resolve a raw BC list.
    cell_Qaux : array (n_aux, n_inner_cells), optional
        Cell-centred aux state.  Required by the Q `face_value` signature; if
        omitted an empty aux vector is used (fine when no BC consumes aux).
    aux_names : sequence of str, optional
        Aux state names; needed only to resolve a raw aux BC list.
    aux_boundary_conditions : BoundaryConditions | dict | list, optional
        The Qaux boundary conditions.  If `None` the aux is extrapolated at
        the boundary (interior value copied) — the framework default.
    parameters : array, optional
        Parameter vector the BC kernels may need (pass the solver's parameter
        array; `sm.parameter_values` is a Zstruct, not an array — convert
        first).  Defaults to an empty array.
    time : float, optional
        Evaluation time for time-dependent BCs.
    aliases : dict, optional
        `on=` alias map for resolving raw BC lists (default `{"momentum": "q"}`).

    Returns
    -------
    vertex_Q : ndarray (n_state, n_vertices)
    vertex_Qaux : ndarray (n_aux, n_vertices)
        BC-aware vertex data.  `vertex_Qaux` has zero rows if no aux was given.
    """
    cell_Q = np.asarray(cell_Q, dtype=float)
    if cell_Q.ndim != 2:
        raise ValueError(
            f"cell_Q must be (n_state, n_inner_cells); got shape {cell_Q.shape}"
        )
    n_state, n_inner = cell_Q.shape
    n_vertices = int(mesh.n_vertices)

    if cell_Qaux is None:
        cell_Qaux = np.zeros((0, n_inner), dtype=float)
    else:
        cell_Qaux = np.asarray(cell_Qaux, dtype=float)
    n_aux = cell_Qaux.shape[0]
    params = np.asarray([] if parameters is None else parameters, dtype=float)

    mesh_type = _infer_mesh_type(mesh)
    v2c = build_vertex_to_cells(np.asarray(mesh.cell_vertices), n_inner)
    bf_verts = _boundary_face_vertices(mesh, mesh_type)

    bc_Q = _resolve_bc_dict(boundary_conditions, state_names, aliases)
    bc_aux = _resolve_bc_dict(aux_boundary_conditions, aux_names, aliases)

    tag_names = list(mesh.boundary_conditions_sorted_names)
    # ``boundary_face_physical_tags`` are raw gmsh physical IDs (e.g. 1000+);
    # the 0-based index into ``boundary_conditions_sorted_names`` is
    # ``boundary_face_function_numbers`` (the same dispatch index the solver uses).
    bf_tags = np.asarray(mesh.boundary_face_function_numbers)
    bf_cells = np.asarray(mesh.boundary_face_cells)
    bf_fidx = np.asarray(mesh.boundary_face_face_indices)
    face_normals = np.asarray(mesh.face_normals_computed())   # (3, n_faces)
    face_centers = np.asarray(mesh.face_centers_computed())   # (n_faces, 3)
    cell_centers = np.asarray(mesh.cell_centers_computed())   # (3, n_cells)
    dim = int(np.asarray(mesh.vertex_coordinates).shape[0])

    sumQ = np.zeros((n_state, n_vertices), dtype=float)
    sumA = np.zeros((n_aux, n_vertices), dtype=float)
    count = np.zeros(n_vertices, dtype=float)

    # interior-cell contributions (unweighted mean, as cell_to_vert_values)
    for v, cells in enumerate(v2c):
        for ic in cells:
            sumQ[:, v] += cell_Q[:, ic]
            if n_aux:
                sumA[:, v] += cell_Qaux[:, ic]
            count[v] += 1.0

    # boundary-face BC contributions: each boundary face feeds its BC-consistent
    # face value to its vertices (Wall ghost reflection → cancels normal momentum)
    for bf, verts in enumerate(bf_verts):
        ic = int(bf_cells[bf])
        fidx = int(bf_fidx[bf])
        tag = tag_names[int(bf_tags[bf])]
        normal = face_normals[:dim, fidx]
        d_face = float(np.linalg.norm(
            face_centers[fidx, :dim] - cell_centers[:dim, ic]))
        Q_in = cell_Q[:, ic]
        Qaux_in = cell_Qaux[:, ic] if n_aux else np.zeros(0)

        bc = bc_Q.get(tag)
        Q_face = (np.asarray(bc.face_value(Q_in, Qaux_in, normal, d_face,
                                           time, params), dtype=float)
                  if bc is not None else Q_in)

        if n_aux:
            abc = bc_aux.get(tag)
            Qaux_face = (np.asarray(abc.face_value(Qaux_in, Q_in, normal,
                                                   d_face, time, params),
                                    dtype=float)
                         if abc is not None else Qaux_in)  # default: extrapolate

        for v in verts:
            sumQ[:, v] += Q_face
            if n_aux:
                sumA[:, v] += Qaux_face
            count[v] += 1.0

    nz = count > 0
    vertex_Q = np.zeros_like(sumQ)
    vertex_Q[:, nz] = sumQ[:, nz] / count[nz]
    vertex_Qaux = np.zeros_like(sumA)
    if n_aux:
        vertex_Qaux[:, nz] = sumA[:, nz] / count[nz]
    return vertex_Q, vertex_Qaux


__all__ = ["interpolate_point_data_with_bcs"]
