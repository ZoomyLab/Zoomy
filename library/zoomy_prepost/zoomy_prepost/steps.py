"""Post-processing pipeline steps.

Two conversions on the zoomy HDF5 result store
(``/mesh/{dimension,type,n_cells,n_inner_cells,vertex_coordinates,
cell_vertices}`` + ``/fields/iteration_i/{time,Q[,Qaux]}``):

* :func:`hdf5_to_vtk` — store -> ``.vtu`` series + ParaView ``.pvd``
  (the inverse of :func:`zoomy_prepost.vtk.vtk_to_hdf5`; the single h5 -> vtk
  writer, optionally naming rows / including ``Qaux``).
* :func:`lift3d` — lift a depth-averaged result to a (d+1)-dimensional
  VTK series through the MODEL's symbolic ``interpolate_to_3d``.

``lift3d`` is the decoupled refactor of the (bit-rotted)
``zoomy_core.postprocessing.postprocessing.vtk_interpolate_to_3d``: same
flow (read store -> extrude the mesh in the vertical -> evaluate the
model's vertical reconstruction per z-layer -> write VTK), but it takes an
explicit model / case folder and explicit paths instead of the old
settings-object + ``get_main_directory`` coupling, and it lambdifies the
``SystemModel`` rows directly (the old ``petscMesh.Mesh.extrude_mesh`` /
``NumpyRuntimeModel`` path no longer exists).  The mesh extrusion reuses
``zoomy_core.mesh.mesh_extrude`` — pure numpy, NO PETSc required (the old
``petscMesh`` import was only an alias name).

Only ``lift3d`` needs zoomy_core (imported inside the function);
``hdf5_to_vtk`` runs on numpy + h5py + meshio alone.
"""
from __future__ import annotations

import os
import importlib.util

import numpy as np

# zoomy mesh type -> meshio cell type (inverse of the map in .vtk)
_ZOOMY_TO_MESHIO = {
    "line": "line",
    "triangle": "triangle",
    "quad": "quad",
    "tetra": "tetra",
    "hexahedron": "hexahedron",
    "hex": "hexahedron",
    "wface": "wedge",
}
#: extruding a d-dim cell (keyed by its vertex count) -> (d+1)-dim zoomy type
_EXTRUDED_TYPE = {2: "quad", 3: "wface", 4: "hexahedron"}
#: canonical interpolate_to_3d profile slots (see to_openfoam._PROFILE_3D_FIELDS)
_PROFILE_NAMES = ("b", "h", "u", "v", "w", "p")


# --------------------------------------------------------------------------- #
# shared store IO
# --------------------------------------------------------------------------- #
def _read_store(h5_path):
    """Read the zoomy HDF5 store -> (mesh dict, [(time, Q, Qaux|None), ...]).

    ``Q``/``Qaux`` are returned as stored (possibly including ghost-cell
    columns); ``mesh["cells"]`` covers the inner cells only.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        g = f["mesh"]
        mtype = g["type"][()]
        mtype = mtype.decode() if isinstance(mtype, bytes) else str(mtype)
        verts = np.asarray(g["vertex_coordinates"][()], dtype=float)  # (dim|3, n_verts)
        cells = np.asarray(g["cell_vertices"][()], dtype=int)         # (k, n_inner)
        mesh = {
            "dimension": int(g["dimension"][()]),
            "type": mtype,
            "n_inner_cells": int(g["n_inner_cells"][()]),
            "points": verts.T,       # (n_verts, dim|3)
            "cells": cells.T,        # (n_inner, k)
        }
        frames = []
        fields = f["fields"]
        for i in range(len(fields.keys())):
            grp = fields[f"iteration_{i}"]
            frames.append((
                float(grp["time"][()]),
                np.asarray(grp["Q"][()], dtype=float),
                np.asarray(grp["Qaux"][()], dtype=float) if "Qaux" in grp else None,
            ))
    if not frames:
        raise ValueError(f"no /fields/iteration_* snapshots in {h5_path}")
    return mesh, frames


def _pad_points_3d(points):
    """(n, d<=3) vertex block -> (n, 3) for meshio."""
    points = np.asarray(points, dtype=float)
    out = np.zeros((points.shape[0], 3))
    out[:, : min(3, points.shape[1])] = points[:, :3]
    return out


def _write_vtu_series(out_base, meshio_type, points, cells, frames):
    """Write ``<out_base>_NNNN.vtu`` + ``<out_base>.pvd``.

    ``frames``: [(time, {field_name: (n_cells,) values}), ...].
    Returns the ``.pvd`` path.
    """
    import meshio

    out_dir = os.path.dirname(os.path.abspath(out_base)) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(out_base)

    entries = []
    for i, (time, named) in enumerate(frames):
        fname = f"{base}_{i:04d}.vtu"
        meshio.Mesh(
            _pad_points_3d(points),
            [(meshio_type, np.asarray(cells, dtype=int))],
            cell_data={k: [np.asarray(v, dtype=float)] for k, v in named.items()},
        ).write(os.path.join(out_dir, fname))
        entries.append((time, fname))

    pvd_path = os.path.join(out_dir, f"{base}.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for time, fname in entries:
            f.write(f'<DataSet timestep="{time}" group="" part="0" file="{fname}"/>\n')
        f.write("</Collection>\n</VTKFile>\n")
    return pvd_path


def _out_base(out, default_basename):
    """Normalise an output argument: directory -> dir/<default>, ``.pvd``
    path -> stripped base, anything else -> used as base path verbatim."""
    out = str(out)
    if os.path.isdir(out) or out.endswith(os.sep):
        return os.path.join(out, default_basename)
    if out.endswith(".pvd"):
        return out[: -len(".pvd")]
    return out


# --------------------------------------------------------------------------- #
# hdf5 -> vtk (inverse of vtk_to_hdf5)
# --------------------------------------------------------------------------- #
def hdf5_to_vtk(h5_path, out_dir, basename="simulation", *,
               include_aux=False, field_names=None, aux_field_names=None):
    """Convert a zoomy HDF5 result store to a ``.vtu`` series + ``.pvd``.

    Each state row ``Q[i]`` becomes a cell_data field (default name ``q<i>``);
    pass ``field_names`` to name them after the model's state symbols so
    ParaView shows ``h``, ``hu``, ...  With ``include_aux=True`` the store's
    ``Qaux`` rows are appended as further cell_data fields (default ``aux_<i>``,
    matching the former ``zoomy_core.misc.io.generate_vtk``); name them via
    ``aux_field_names``.  Snapshot times land in the ``.pvd``.  Ghost-cell
    columns (``Q.shape[1] > n_inner_cells``) are dropped.  Returns the ``.pvd``
    path.

    This is the single h5 -> vtk writer (the legacy ``generate_vtk`` /
    ``_write_to_vtk_from_vertices_edges`` in zoomy_core were removed in favour
    of it); the ``.vtu``/``.pvd`` shape is what the postprocessing adapters and
    ``vtk_to_hdf5`` round-trip on.
    """
    mesh, store_frames = _read_store(h5_path)
    meshio_type = _ZOOMY_TO_MESHIO.get(mesh["type"])
    if meshio_type is None:
        raise ValueError(f"unsupported zoomy mesh type {mesh['type']!r}")
    n_inner = mesh["n_inner_cells"]

    def _name(names, i, default):
        return names[i] if names is not None and i < len(names) else default

    frames = []
    for time, Q, Qaux in store_frames:
        named = {_name(field_names, i, f"q{i}"): Q[i, :n_inner]
                 for i in range(Q.shape[0])}
        if include_aux and Qaux is not None:
            for i in range(Qaux.shape[0]):
                named[_name(aux_field_names, i, f"aux_{i}")] = Qaux[i, :n_inner]
        frames.append((time, named))

    return _write_vtu_series(
        os.path.join(out_dir, basename), meshio_type,
        mesh["points"], mesh["cells"], frames,
    )


# --------------------------------------------------------------------------- #
# lift3d — depth-averaged result -> (d+1)-D VTK via model.interpolate_to_3d
# --------------------------------------------------------------------------- #
def _resolve_model(case_or_model):
    """Accept a model object, a case folder (containing ``model.py``), or a
    path to a model ``.py`` file; return the model object.

    Mirrors ``zoomy_server.adapter.SolverAdapter.resolve_model`` (module-level
    ``model`` preferred — the composed-case format — else the first ``Model``
    subclass) without depending on zoomy_server.
    """
    if not isinstance(case_or_model, (str, os.PathLike)):
        return case_or_model
    path = str(case_or_model)
    model_py = os.path.join(path, "model.py") if os.path.isdir(path) else path
    if not (model_py.endswith(".py") and os.path.exists(model_py)):
        raise FileNotFoundError(f"no model.py found at {case_or_model!r}")
    spec = importlib.util.spec_from_file_location("zoomy_case_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "model"):
        return mod.model
    from zoomy_core.model.basemodel import Model
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
            return obj()
    raise RuntimeError(f"no `model` object or Model subclass in {model_py}")


def _as_system_model(model):
    """Normalise Model / BaseModel / SystemModel / NumericalSystemModel to the
    SystemModel that carries the stored ``interpolate_to_3d`` rows."""
    # A SystemModel stores its operators as data (flux is a ZArray, not a method).
    if hasattr(model, "state") and not callable(getattr(model, "flux", None)):
        return model
    for attr in ("system_model", "sm"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            return _as_system_model(inner)
    # Legacy production Model (callable operator API): lift through from_model.
    from zoomy_core.systemmodel.system_model import SystemModel
    return SystemModel.from_model(model)


def _row_exprs(rows):
    """ZArray / Matrix of shape (n,) or (n, 1) -> flat list of sympy exprs."""
    import sympy as sp
    return [sp.sympify(e) for e in np.asarray(rows, dtype=object).ravel()]


def _compile_interpolate_rows(sm):
    """Lambdify the SystemModel's ``interpolate_to_3d`` rows (the same
    approach as ``zoomy_core.postprocessing.column_plots.read_zoomyfoam``).

    Returns ``(fns, n_state, n_aux, param_values)`` with each ``fn`` taking
    ``(*Q_rows, *Qaux_rows, *params, x, y, z)``.
    """
    import sympy as sp

    rows = getattr(sm, "interpolate_to_3d", None)
    if rows is None:
        raise ValueError(
            "the model defines no interpolate_to_3d (vertical reconstruction) "
            "— lift3d needs one (e.g. SME/VAM models)")
    row_exprs = _row_exprs(rows)

    if getattr(sm, "position", None) is not None:
        pos = list(sm.position.values())
    else:
        pos = list(sp.symbols("x y z", real=True))
    state = list(sm.state)
    aux = list(getattr(sm, "aux_state", None) or [])
    params = list(sm.parameters.values()) if sm.parameters is not None else []
    syms = state + aux + params + pos[:3]

    fns = [sp.lambdify(syms, e, "numpy", dummify=True) for e in row_exprs]
    pv = getattr(sm, "parameter_values", None)
    pvals = ([float(v) for v in pv.values()] if pv is not None
             else [0.0] * len(params))
    return fns, len(state), len(aux), pvals


def lift3d(case_or_model, h5_path, out, Nz=10):
    """Lift a depth-averaged zoomy result to a vertically-extruded VTK series.

    Parameters
    ----------
    case_or_model : model object | case folder path | model.py path
        The model whose symbolic ``interpolate_to_3d`` does the lifting
        (Model / BaseModel with ``.system_model`` / SystemModel all accepted).
    h5_path : str
        The simulation store (``/mesh`` + ``/fields``) of the run.
    out : str
        Output base: a directory (-> ``<out>/simulation_3d``), a ``.pvd``
        path, or a bare base path.  Writes ``<base>_NNNN.vtu`` + ``<base>.pvd``.
    Nz : int
        Number of vertical cell layers on the σ unit interval ``z∈[0,1]``
        (rows are evaluated at the layer centres ``(k+0.5)/Nz``).

    Returns the ``.pvd`` path.

    Notes
    -----
    * The base mesh is extruded with ``zoomy_core.mesh.mesh_extrude``
      (pure numpy — no PETSc; the old ``petscMesh`` name was an alias).
      line -> quad, triangle -> wedge, quad -> hexahedron; the extrusion
      coordinate is appended after the mesh's own (so a 1-D run lifts into
      the x/y VTK plane, a 2-D run into x/y/z).
    * Aux gradient fields the reconstruction needs (``dhdx`` etc.) are read
      from the store's ``Qaux``; missing rows are zero-filled with a warning.
    """
    from zoomy_core.mesh.mesh_extrude import (
        extrude_element_vertices,
        extrude_points,
    )

    sm = _as_system_model(_resolve_model(case_or_model))
    fns, n_state, n_aux, pvals = _compile_interpolate_rows(sm)

    mesh, store_frames = _read_store(h5_path)
    dim = mesh["dimension"]
    if dim >= 3:
        raise ValueError(f"lift3d expects a 1-D or 2-D base mesh, got {dim}-D")
    n_inner = mesh["n_inner_cells"]
    points = mesh["points"][:, :dim]                  # (n_verts, dim)
    cells = mesh["cells"]                             # (n_inner, k)
    vpce = cells.shape[1]
    if vpce not in _EXTRUDED_TYPE:
        raise ValueError(f"cannot extrude cells with {vpce} vertices")

    # -- extrude: Nz cell layers between Nz+1 point levels on z∈[0,1] --------
    z_levels = np.linspace(0.0, 1.0, Nz + 1)
    points_ext = extrude_points(points, z_levels)     # (n_verts*(Nz+1), dim+1)
    cells_ext = extrude_element_vertices(cells, points.shape[0], Nz)
    meshio_type = _ZOOMY_TO_MESHIO[_EXTRUDED_TYPE[vpce]]
    z_centers = (np.arange(Nz) + 0.5) / Nz

    # -- per-cell horizontal positions (the rows may reference x/y) ----------
    centers = points[cells].mean(axis=1)              # (n_inner, dim)
    xc = centers[:, 0]
    yc = centers[:, 1] if dim > 1 else np.zeros(n_inner)

    n_rows = len(fns)
    names = (_PROFILE_NAMES if n_rows == len(_PROFILE_NAMES)
             else tuple(f"f{i}" for i in range(n_rows)))

    warned = False
    frames = []
    for time, Q, Qaux in store_frames:
        Qi = Q[:, :n_inner]
        aux = np.zeros((n_aux, n_inner))
        if n_aux:
            have = 0 if Qaux is None else min(n_aux, Qaux.shape[0])
            if have:
                aux[:have] = Qaux[:have, :n_inner]
            if have < n_aux and not warned:
                import warnings
                warnings.warn(
                    f"store has {have}/{n_aux} aux rows the reconstruction may "
                    "reference — missing rows (e.g. gradients) are zero-filled")
                warned = True
        args_base = [Qi[i] for i in range(n_state)] + [aux[i] for i in range(n_aux)] \
            + pvals + [xc, yc]

        lifted = np.zeros((n_rows, n_inner * Nz))
        for iz, zc in enumerate(z_centers):
            sl = slice(iz * n_inner, (iz + 1) * n_inner)
            for r, fn in enumerate(fns):
                val = np.asarray(fn(*args_base, zc), dtype=float)
                lifted[r, sl] = np.broadcast_to(val, (n_inner,))
        frames.append((time, {names[r]: lifted[r] for r in range(n_rows)}))

    return _write_vtu_series(
        _out_base(out, "simulation_3d"), meshio_type, points_ext, cells_ext, frames,
    )
