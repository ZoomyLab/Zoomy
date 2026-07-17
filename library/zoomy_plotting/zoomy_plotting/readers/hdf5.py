"""HDF5 reader — lazy field access.

Reads the mesh topology, times, and variable-count metadata eagerly; installs
a closure that pulls single ``(time_step, field_index)`` slices from disk on
demand. A store opened this way uses O(mesh) memory even if the backing
HDF5 file is multi-gigabyte across hundreds of snapshots.

File layout (zoomy_core convention, see
``library/zoomy_core/zoomy_core/misc/io.py``)::

    file.h5
    ├── /mesh/
    │   ├── dimension        (scalar int: 1 | 2 | 3)
    │   ├── type             (bytes: "line" | "triangle" | ...)
    │   ├── n_cells          (scalar int)
    │   ├── n_inner_cells    (scalar int)
    │   ├── vertex_coordinates   (dim, n_vertices)
    │   └── cell_vertices        (k,   n_cells)
    └── /fields/
        ├── iteration_0/
        │   ├── time  (scalar float)
        │   ├── Q     (n_vars, n_cells)
        │   └── Qaux  (n_aux,  n_cells)         [optional]
        ├── iteration_1/ ...
        └── iteration_N/

Variable names are not stored by ``_save_fields_to_hdf5``; this reader
falls back to ``q0, q1, ..., aux_0, aux_1, ...``. The ``aux_`` prefix is
applied unconditionally to keep name resolution unambiguous.
"""

from __future__ import annotations

import re
from typing import Optional

import h5py
import numpy as np

from ..store import SimulationStore, Zstruct
from ._common import to_canonical_axes


_ITERATION_RE = re.compile(r"^iteration_(\d+)$")


def _decode(raw):
    """Turn an h5py scalar that might be bytes into a Python str."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def read_hdf5(path, *, swmr: bool = False) -> SimulationStore:
    """Open an HDF5 file and return a :class:`SimulationStore` with lazy fields.

    The file handle is held by the returned store; call ``store.close()`` or
    use it as a context manager to release it.

    Parameters
    ----------
    path
        Path to a ``.h5`` / ``.hdf5`` file produced by zoomy_core.
    swmr
        Passed through to ``h5py.File(..., swmr=True)`` if the file was
        opened in SWMR write mode.
    """
    h5 = h5py.File(str(path), "r", swmr=swmr)
    try:
        # ----- /mesh/ ----------------------------------------------------
        mesh_g = h5["mesh"]
        dim = int(mesh_g["dimension"][()])
        cell_type = _decode(mesh_g["type"][()])
        n_inner_cells = (
            int(mesh_g["n_inner_cells"][()])
            if "n_inner_cells" in mesh_g
            else None
        )
        raw_vertices = np.asarray(mesh_g["vertex_coordinates"][:])
        raw_cells = np.asarray(mesh_g["cell_vertices"][:])
        vertices, cells = to_canonical_axes(raw_vertices, raw_cells, dim)

        # ----- /fields/ timeline & field-name Zstruct -------------------
        times: Optional[np.ndarray]
        field_ztable = Zstruct()
        n_q = 0
        n_aux = 0
        cell_reader = None

        if "fields" in h5:
            fields_g = h5["fields"]
            iter_keys = []
            for name in fields_g.keys():
                m = _ITERATION_RE.match(name)
                if m:
                    iter_keys.append((int(m.group(1)), name))
            iter_keys.sort()
            if iter_keys:
                times = np.asarray([
                    float(fields_g[name]["time"][()])
                    for _, name in iter_keys
                ])
                # Infer n_vars and n_aux from the first snapshot.
                first = fields_g[iter_keys[0][1]]
                q_shape = first["Q"].shape   # (n_q, n_cells)
                n_q = int(q_shape[0])
                if "Qaux" in first:
                    aux_shape = first["Qaux"].shape
                    n_aux = int(aux_shape[0])

                # Build {name -> index} mapping. Converters that know the
                # variable names store them as ``/fields@names`` (REQ-158:
                # positional resolution silently row-shifts DMPlex stores,
                # whose first CellData entry is ``Rank``); use them when
                # present and consistent, else fall back to q0..qN. The
                # aux_ prefix is explicit to keep resolution unambiguous.
                stored = fields_g.attrs.get("names")
                if stored is not None and len(stored) == n_q:
                    mapping = {
                        (n.decode() if isinstance(n, bytes) else str(n)): i
                        for i, n in enumerate(stored)}
                else:
                    mapping = {f"q{i}": i for i in range(n_q)}
                mapping.update({f"aux_{i}": n_q + i for i in range(n_aux)})
                field_ztable = Zstruct(mapping)

                # Capture names used by the closure so it doesn't need the
                # outer locals after this function returns.
                _iter_lookup = {i: name for i, name in iter_keys}
                _n_q = n_q

                def _read_cell(t: int, idx: int) -> np.ndarray:
                    try:
                        iter_name = _iter_lookup[int(t)]
                    except KeyError:
                        raise IndexError(
                            f"time_step {t} out of range; "
                            f"available: {sorted(_iter_lookup)}"
                        )
                    g = fields_g[iter_name]
                    if idx < _n_q:
                        return np.asarray(g["Q"][idx, :])
                    aux_i = idx - _n_q
                    if "Qaux" not in g:
                        raise IndexError(
                            f"field index {idx} requests Qaux but "
                            f"iteration {t} has none"
                        )
                    return np.asarray(g["Qaux"][aux_i, :])

                cell_reader = _read_cell
            else:
                times = None
        else:
            times = None

        store = SimulationStore(
            dim=dim,
            cell_type=cell_type,
            vertices=vertices,
            cells=cells,
            n_inner_cells=n_inner_cells,
            times=times,
            field=field_ztable,
            point_field=Zstruct(),         # HDF5 has no point data
            _cell_reader=cell_reader,
            _point_reader=None,
            source_path=str(path),
            _resource=h5,
        )
        return store
    except Exception:
        h5.close()
        raise
