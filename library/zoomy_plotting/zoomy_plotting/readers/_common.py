"""Helpers shared across readers.

The only non-trivial reader invariant is the axis convention: internally we
use ``n x dim`` for vertices and ``n x k`` for cells, while zoomy_core's
HDF5 schema stores them transposed. :func:`to_canonical_axes` is the one
and only place where that transpose happens — any future reader must funnel
its mesh data through it so the invariant is preserved.
"""

from __future__ import annotations

import numpy as np


def to_canonical_axes(vertices: np.ndarray, cells: np.ndarray, dim: int):
    """Ensure ``vertices`` is ``(n_vertices, dim)`` and ``cells`` is ``(n_cells, k)``.

    Accepts either orientation on input and returns the canonical form.

    Parameters
    ----------
    vertices
        Raw vertex array of shape ``(n, dim)`` or ``(dim, n)``.
    cells
        Raw cell-vertex array of shape ``(n, k)`` or ``(k, n)``. The cell
        count should equal the *non-dim* axis of ``vertices`` inferred above.
    dim
        Spatial dimension (1, 2, or 3). Used to disambiguate vertex layout
        when ``n == dim``.

    Returns
    -------
    (vertices, cells)
        Both in canonical ``n x ...`` layout.
    """
    vertices = np.asarray(vertices)
    cells = np.asarray(cells)

    if vertices.ndim != 2:
        raise ValueError(
            f"vertices must be 2-D; got {vertices.ndim}-D of shape {vertices.shape}"
        )

    # zoomy_core stores vertex_coordinates as ``(embed_dim, n_vertices)`` where
    # ``embed_dim`` is often 3 even for 1- or 2-D meshes (z = 0). Accept any
    # orientation where one axis equals 3 or ``dim`` and the other is the
    # vertex count. We trim embedded-but-unused axes down to ``dim``.
    a, b = vertices.shape
    oriented = None
    for candidate in (vertices, vertices.T):
        rows, cols = candidate.shape
        if rows in (dim, 3) and cols > max(rows, 3):
            oriented = candidate
            break
    if oriented is None:
        # Fallbacks for the pathological "n_vertices == dim" corner case.
        if vertices.shape[1] == dim:
            oriented = vertices
        elif vertices.shape[0] == dim:
            oriented = vertices.T
    if oriented is None:
        raise ValueError(
            f"cannot infer vertex axis: shape {vertices.shape} with dim={dim}"
        )
    # Transpose so rows are vertices, cols are coords.
    vertices = oriented.T
    # Trim unused embedding dimensions (drop z for dim=2, etc.).
    if vertices.shape[1] > dim:
        vertices = np.ascontiguousarray(vertices[:, :dim])

    # Cells: canonical is (n_cells, k). HDF5 stores (k, n_cells).
    if cells.ndim != 2:
        raise ValueError(
            f"cells must be 2-D; got {cells.ndim}-D of shape {cells.shape}"
        )

    n_vertices = vertices.shape[0]
    # If one of the axes of `cells` already equals n_cells (unknown), we pick
    # the orientation whose *other* axis equals a plausible k in [1, 9] AND
    # whose index values are all < n_vertices.
    a, b = cells.shape
    canonical = None
    for candidate in (cells, cells.T):
        k = candidate.shape[1]
        if 1 <= k <= 16 and candidate.max(initial=-1) < n_vertices:
            canonical = candidate
            break
    if canonical is None:
        raise ValueError(
            f"cannot orient cells array with shape {cells.shape} "
            f"given n_vertices={n_vertices}"
        )
    return vertices, canonical
