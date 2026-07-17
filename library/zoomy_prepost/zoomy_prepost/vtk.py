"""VTK / ParaView ``.pvd`` -> zoomy HDF5 converter.

Backends that emit VTK instead of the zoomy HDF5 (firedrake ``.pvd``, foam /
dmplex ``.vtu``/``.vtk``) call one function here and the result is readable by
the normal postprocessing (``zoomy_plotting.read_hdf5`` / the GUI viz). No
per-backend HDF5 writers.

Target layout (``zoomy_core.misc.io`` convention)::

    /mesh/{dimension, type, n_cells, n_inner_cells,
           vertex_coordinates:(dim,n_vert), cell_vertices:(k,n_cells)}
    /fields/iteration_i/{time, Q:(n_vars,n_cells)}
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

# meshio cell type -> zoomy mesh type (inverse of mesh_util.convert_mesh_type_to_meshio_mesh_type)
_MESHIO_TO_ZOOMY = {
    "line": "line",
    "triangle": "triangle",
    "quad": "quad",
    "tetra": "tetra",
    "hexahedron": "hexahedron",
    "wedge": "wface",
}
_ZOOMY_DIM = {"line": 1, "triangle": 2, "quad": 2, "tetra": 3, "hexahedron": 3, "wface": 3}


def _read_pvd(path):
    """Parse a ParaView ``.pvd`` collection -> [(time, absolute_vtu_path), ...]."""
    base = os.path.dirname(os.path.abspath(path))
    root = ET.parse(path).getroot()
    frames = []
    for i, ds in enumerate(root.iter("DataSet")):
        t = float(ds.get("timestep", i))
        f = ds.get("file")
        frames.append((t, f if os.path.isabs(f) else os.path.join(base, f)))
    frames.sort(key=lambda tf: tf[0])
    return frames


def _frames(source):
    """Normalise the input into an ordered [(time, path), ...] list."""
    if isinstance(source, (list, tuple)):
        return [(float(i), s) for i, s in enumerate(source)]
    source = str(source)
    if source.endswith(".pvd"):
        return _read_pvd(source)
    return [(0.0, source)]  # a single .vtu / .vtk


# VTK cell-type ints (for the pyvista fallback) -> meshio names
_VTK_TO_MESHIO = {3: "line", 5: "triangle", 9: "quad", 10: "tetra", 12: "hexahedron", 13: "wedge"}


def _read(path):
    """Read one VTK frame as a meshio.Mesh. meshio first; on failure (e.g.
    firedrake's VTU version 2.2, which meshio rejects) fall back to the vtk
    library directly (always present where firedrake writes VTU)."""
    import meshio
    try:
        return meshio.read(path)
    except (Exception, SystemExit):
        # meshio raises SystemExit (not Exception) on an unsupported version.
        return _read_via_vtk(path)


def _read_via_vtk(path):
    """Read a .vtu via the ``vtk`` library directly and return a meshio.Mesh.
    Handles the appended/binary VTU 2.2 that firedrake writes and meshio rejects.
    vtk ships with firedrake (its VTK output dep), so this needs no extra install."""
    import numpy as np
    import meshio
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()))
    ca = grid.GetCells()
    conn = vtk_to_numpy(ca.GetConnectivityArray())
    offsets = vtk_to_numpy(ca.GetOffsetsArray())      # (n_cells+1,)
    ctypes = vtk_to_numpy(grid.GetCellTypesArray())   # (n_cells,)

    cells = []
    for t in np.unique(ctypes):
        ti = int(t)
        if ti not in _VTK_TO_MESHIO:
            continue
        idx = np.where(ctypes == t)[0]
        k = int(offsets[idx[0] + 1] - offsets[idx[0]])
        block = np.stack([conn[offsets[i]:offsets[i] + k] for i in idx])
        cells.append((_VTK_TO_MESHIO[ti], block))
    if not cells:
        raise ValueError(f"no supported VTK cell types in {np.unique(ctypes)} ({path})")

    pd = grid.GetPointData()
    point_data = {pd.GetArrayName(i): np.asarray(vtk_to_numpy(pd.GetArray(i)))
                  for i in range(pd.GetNumberOfArrays())}
    cdg = grid.GetCellData()
    cell_data = {cdg.GetArrayName(i): [np.asarray(vtk_to_numpy(cdg.GetArray(i)))]
                 for i in range(cdg.GetNumberOfArrays())}

    return meshio.Mesh(points, cells, cell_data=cell_data, point_data=point_data)


def _pick_block(mesh):
    """Return (block_index, CellBlock) for the top-dimensional supported block."""
    best = None
    for idx, cb in enumerate(mesh.cells):
        z = _MESHIO_TO_ZOOMY.get(cb.type)
        if z is None:
            continue
        if best is None or _ZOOMY_DIM[z] > _ZOOMY_DIM[_MESHIO_TO_ZOOMY[mesh.cells[best].type]]:
            best = idx
    if best is None:
        raise ValueError(f"no supported cell type in {[c.type for c in mesh.cells]}")
    return best, mesh.cells[best]


def _extract_Q(mesh, block_index, cell_idx, n_cells):
    """Stack the frame's fields into ``Q`` of shape (n_vars, n_cells).

    Prefers cell_data (fields already at cell centres); falls back to
    point_data averaged onto each cell's vertices. Vector components are
    split into scalar rows ``name_0, name_1, ...``.
    """
    cols, names = [], []
    for name, blocks in (mesh.cell_data or {}).items():
        arr = np.asarray(blocks[block_index])
        if arr.shape[0] != n_cells:
            continue
        if arr.ndim == 1:
            cols.append(arr); names.append(name)
        else:
            for c in range(arr.shape[1]):
                cols.append(arr[:, c]); names.append(f"{name}_{c}")
    if not cols:
        for name, arr in (mesh.point_data or {}).items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                cols.append(arr[cell_idx].mean(axis=1)); names.append(name)
            else:
                for c in range(arr.shape[1]):
                    cols.append(arr[cell_idx, c].mean(axis=1)); names.append(f"{name}_{c}")
    if not cols:
        raise ValueError("no cell_data / point_data fields found in the VTK frame")
    return np.vstack(cols).astype(float), names


def vtk_to_hdf5(source, out_path, fields=None, exclude=()):
    """Convert a VTK series to the zoomy HDF5 format.

    Field NAMES are written to ``/fields@names`` so readers resolve
    variables by name instead of position (REQ-158: PETSc/DMPlex writes a
    ``Rank`` cell array as the FIRST CellData entry, silently shifting
    every positional row).

    Parameters
    ----------
    source : str | list[str]
        A ``.pvd`` collection, a single ``.vtu``/``.vtk``, or an ordered list
        of frame files.
    out_path : str
        Destination ``.h5`` (overwritten).
    fields : list[str], optional
        Keep exactly these fields, in this order (error if one is missing).
    exclude : tuple[str, ...], optional
        Drop these fields (e.g. ``("Rank",)``); applied when ``fields`` is
        not given.

    Returns
    -------
    str
        ``out_path``.
    """
    import h5py
    import meshio

    frames = _frames(source)
    if not frames:
        raise ValueError(f"no frames in {source!r}")

    mesh0 = _read(frames[0][1])
    bi, cb = _pick_block(mesh0)
    ztype = _MESHIO_TO_ZOOMY[cb.type]
    dim = _ZOOMY_DIM[ztype]
    verts = np.asarray(mesh0.points, dtype=float)[:, :dim].T   # (dim, n_vert)
    cell_idx = np.asarray(cb.data)                             # (n_cells, k)
    cells = cell_idx.T                                         # (k, n_cells)
    n_cells = cells.shape[1]

    with h5py.File(out_path, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("dimension", data=dim)
        g.create_dataset("type", data=ztype)
        g.create_dataset("n_cells", data=n_cells)
        g.create_dataset("n_inner_cells", data=n_cells)
        g.create_dataset("vertex_coordinates", data=verts)
        g.create_dataset("cell_vertices", data=cells)

        fgroup = f.create_group("fields")
        kept = None
        for i, (t, path) in enumerate(frames):
            mesh = mesh0 if i == 0 else _read(path)
            Q, names = _extract_Q(mesh, bi, cell_idx, n_cells)
            if kept is None:
                if fields is not None:
                    missing = [n for n in fields if n not in names]
                    if missing:
                        raise ValueError(f"requested fields not in VTK data: "
                                         f"{missing} (have {names})")
                    kept = list(fields)
                else:
                    kept = [n for n in names if n not in set(exclude)]
                fgroup.attrs["names"] = np.array(kept, dtype="S")
            # select by NAME per frame — robust to field-order changes
            rows = [names.index(n) for n in kept]
            it = fgroup.create_group(f"iteration_{i}")
            it.create_dataset("time", data=float(t), dtype=float)
            it.create_dataset("Q", data=Q[rows])

    return out_path
