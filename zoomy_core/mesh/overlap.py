"""The overlap of a coupled participant (overlapping Schwarz over preCICE, the Chorin march):
its mesh reaches `layers` cell rings across the coupled interface into the peer's domain, the
outer boundary of those rings tagged `tag`.  The rings (the overlap) hold the peer's values; the
next `layers` rings of own cells (the mirror) are what the peer holds as its overlap.
Backend-neutral (numpy).
"""
from __future__ import annotations

import numpy as np


def overlap_layers(mesh, tag, layers):
    """`(overlap, mirror)`: the inner cells of the first `layers` face rings from the boundary
    faces tagged `tag`, and those of the next `layers` rings, each ring by ring in cell order.

    Raises if the mesh has no boundary `tag`, or if it is too small for `2 layers` rings.
    """
    layers = int(layers)
    if layers < 1:
        raise ValueError(f"overlap: {layers} layers, at least 1")
    names = list(mesh.boundary_conditions_sorted_names)
    if tag not in names:
        raise ValueError(f"overlap: no boundary tag {tag!r} in {names}")
    nc, n_bf = int(mesh.n_inner_cells), int(mesh.n_boundary_faces)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)[:n_bf]
    bf_cells = np.asarray(mesh.boundary_face_cells, dtype=int)[:n_bf]
    fc = np.asarray(mesh.face_cells, dtype=int)[:2]
    inner = (fc[0] < nc) & (fc[1] < nc)
    a, b = fc[0, inner], fc[1, inner]
    nbrs = [[] for _ in range(nc)]
    for i, j in zip(a, b):
        nbrs[i].append(j)
        nbrs[j].append(i)
    seen = np.zeros(nc, dtype=bool)
    ring = np.unique(bf_cells[nums == names.index(tag)])
    rings = []
    for _ in range(2 * layers):
        if not len(ring):
            raise ValueError(f"overlap: the mesh has fewer than {2 * layers} cell rings "
                             f"from {tag!r}")
        seen[ring] = True
        rings.append(ring)
        ring = np.unique([j for i in ring for j in nbrs[i] if not seen[j]]).astype(int)
    return np.concatenate(rings[:layers]), np.concatenate(rings[layers:])
