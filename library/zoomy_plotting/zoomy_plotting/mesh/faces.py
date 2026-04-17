"""Mesh-primitive helpers consolidated from three older copies in the Zoomy
tree (``library/zoomy_gui/snippets/mesh_mpl.py``, ``mesh_plotly.py``,
``library/zoomy_gui/generate_mesh_previews.py``).

Operates on the canonical in-memory layout: ``cells`` is ``(n_cells, k)``.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Face index maps — vertex indices of each face of each 3-D cell type.
# Order matches zoomy_gui's generate_mesh_previews._FACE_DEFS.
# --------------------------------------------------------------------------
FACE_DEFS = {
    "tetra": [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ],
    "hexahedron": [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ],
    "wedge": [
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [0, 2, 5, 3],
    ],
    "pyramid": [
        [0, 1, 2, 3],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ],
}


def boundary_faces_3d(
    cells: np.ndarray, cell_type: str
) -> Tuple[List[List[int]], List[int]]:
    """Return the boundary faces of a 3-D cell array, plus their parent cell.

    A face is on the boundary iff it is shared by exactly one cell. Returns
    two parallel lists:

    * ``faces``: each entry is a list of vertex indices (preserving the
      original ordering from ``FACE_DEFS`` so outward winding is consistent).
    * ``parent_cells``: the cell index each face came from, useful for
      mapping cell-centered values onto each face.
    """
    if cell_type not in FACE_DEFS:
        raise ValueError(
            f"no face template for 3-D cell type {cell_type!r}; "
            f"known types: {list(FACE_DEFS)}"
        )

    face_defs = FACE_DEFS[cell_type]
    count = Counter()
    face_map: dict[tuple, list[int]] = {}
    parent_map: dict[tuple, int] = {}

    for ci, cell in enumerate(cells):
        for fi in face_defs:
            face_nodes = [int(cell[k]) for k in fi]
            key = tuple(sorted(face_nodes))
            count[key] += 1
            face_map[key] = face_nodes
            parent_map[key] = ci

    faces: list[list[int]] = []
    parents: list[int] = []
    for key, c in count.items():
        if c == 1:
            faces.append(face_map[key])
            parents.append(parent_map[key])
    return faces, parents


def triangulate(cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fan-triangulate a 2-D or 3-surface cell array.

    For each cell with ``k`` vertices, produces ``k-2`` triangles.
    Returns four 1-D arrays ``(i, j, k, parent)`` where ``i/j/k`` are
    vertex indices and ``parent`` is the cell index each triangle came from.
    """
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    parents: list[int] = []
    for ci, cell in enumerate(cells):
        if len(cell) < 3:
            continue
        for t in range(1, len(cell) - 1):
            ii.append(int(cell[0]))
            jj.append(int(cell[t]))
            kk.append(int(cell[t + 1]))
            parents.append(ci)
    return (
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(kk, dtype=np.int64),
        np.asarray(parents, dtype=np.int64),
    )


def cell_to_vert_values(
    n_vertices: int, cells: np.ndarray, cell_values: np.ndarray
) -> np.ndarray:
    """Average cell-centered scalar values onto vertices.

    Each vertex's value is the mean of the values of all cells incident on
    it. Isolated vertices (no incident cell) receive zero.
    """
    vv = np.zeros(n_vertices, dtype=float)
    vc = np.zeros(n_vertices, dtype=float)
    cell_values = np.asarray(cell_values, dtype=float)
    for ci, cell in enumerate(cells):
        val = float(cell_values[ci]) if ci < len(cell_values) else 0.0
        for vi in cell:
            vv[int(vi)] += val
            vc[int(vi)] += 1.0
    vc[vc == 0] = 1.0
    return vv / vc
