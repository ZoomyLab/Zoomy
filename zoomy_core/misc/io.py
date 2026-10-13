"""HDF5 settings, field checkpoints, and mesh I/O helpers used by solvers and drivers."""

import os
import numpy as np
import shutil

try:
    import h5py

    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

# import zoomy_core.mesh.fvm_mesh as fvm_mesh
from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
from zoomy_core.misc.misc import Zstruct, Settings
from zoomy_core.misc import misc as misc
from zoomy_core.misc.logger_config import logger


def init_output_directory(path, clean):
    """Create (and optionally wipe) a run directory under the Zoomy project root."""
    main_dir = misc.get_main_directory()

    path = os.path.join(main_dir, path)
    os.makedirs(path, exist_ok=True)
    if clean:
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            if os.path.isdir(os.path.join(path, f)):
                shutil.rmtree(os.path.join(path, f))
            else:
                os.remove(os.path.join(path, f))


def get_hdf5_type(value):
    """Get hdf5 type."""
    out = type(value)
    if isinstance(value, str):
        out = h5py.string_dtype()
    return out


def write_dict_to_hdf5(group, d):
    """Recursively store nested dicts / `Zstruct`-like objects into an HDF5 group."""
    for key, value in d.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            write_dict_to_hdf5(subgroup, value)
        elif isinstance(value, (str, int, float, bool)):
            group.create_dataset(key, data=value, dtype=get_hdf5_type(value))
        elif isinstance(value, (list, tuple)):
            group.create_dataset(key, data=value)
        elif isinstance(value, type(np.ndarray)):
            group.create_dataset(key, data=value)
        elif hasattr(value, "as_dict"):
            subgroup = group.create_group(key)
            write_dict_to_hdf5(subgroup, value.as_dict())
        else:
            logger.warning(f"Skipping unsupported type for key: {key} -> {type(value)}")


def load_hdf5_to_dict(group):
    """Inverse of `write_dict_to_hdf5` for reading settings or metadata."""
    d = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            d[key] = load_hdf5_to_dict(value)
        elif isinstance(value, h5py.Dataset):
            if value.dtype == h5py.string_dtype():
                d[key] = value[()].decode("utf-8")
            else:
                d[key] = value[()]
        else:
            logger.warning(f"Skipping unsupported type for key: {key} -> {type(value)}")

    return d


def save_settings(settings):
    """Write `settings` to `settings.h5` inside the configured output directory."""
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, settings.output.directory)
    with h5py.File(os.path.join(filepath, "settings.h5"), "w") as f:
        write_dict_to_hdf5(f, settings.as_dict(recursive=True))


def load_settings(filepath):
    """Load a `Settings` object from `settings.h5`."""
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    with h5py.File(os.path.join(filepath, "settings.h5"), "r") as f:
        d = load_hdf5_to_dict(f)

    settings = Settings.from_dict(d)
    return settings


def load_settings2(filepath):
    """Legacy HDF5 layout: model/solver/output groups plus callbacks attribute."""
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    with h5py.File(os.path.join(filepath, "settings.h5"), "r") as f:
        model = f["model"]
        solver = f["solver"]
        output = f["output"]

        d_model = {}
        if "parameters" in model:
            parameters = {k: v[()] for k, v in model["parameters"].items()}
            parameters = Zstruct(**parameters)
        for k in model.keys():
            if k != "parameters":
                v = model[k][()]
                if isinstance(v, (str, int, float, bool)):
                    d_model[k] = v
                else:
                    raise ValueError(
                        f"Unsupported type for model attribute {k}: {type(v)}"
                    )
        d_model["parameters"] = parameters
        model = Zstruct(**d_model)
        d_solver = {}
        for k in solver.keys():
            v = solver[k][()]
            if isinstance(v, (str, int, float, bool)):
                d_solver[k] = v
            else:
                raise ValueError(
                    f"Unsupported type for solver attribute {k}: {type(v)}"
                )
        solver = Zstruct(**d_solver)

        d_output = {}
        for k in output.keys():
            v = output[k][()]
            if isinstance(v, (str, int, float, bool)):
                d_output[k] = v
            else:
                raise ValueError(
                    f"Unsupported type for output attribute {k}: {type(v)}"
                )
        output = Zstruct(**d_output)

        settings = Settings(model=model, solver=solver, output=output)

        # parameters = {k: v[()] for k, v in f["parameters"].items()}
        # name = f["name"][()]
        # output_dir = f["output_dir"][()]
        # output_snapshots = f["output_snapshots"][()]
        # output_write_all = f["output_write_all"][()]
        # output_clean_dir = f["output_clean_dir"][()]
        # truncate_last_time_step = f["truncate_last_time_step"][()]
        callbacks = f["callbacks"][()]
    return settings


def clean_files(filepath, filename=".vtk"):
    """Clean files."""
    main_dir = misc.get_main_directory()

    abs_filepath = os.path.join(main_dir, filepath)
    if os.path.exists(abs_filepath):
        for file in os.listdir(abs_filepath):
            if file.endswith(filename):
                os.remove(os.path.join(abs_filepath, file))


def _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux=None, overwrite=True):
    i_snap = int(i_snapshot)
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    with h5py.File(filepath, "a") as f:
        if i_snap == 0 and "fields" not in f.keys():
            fields = f.create_group("fields")
        else:
            fields = f["fields"]
        group_name = "iteration_" + str(i_snap)
        if group_name in fields:
            if overwrite:
                del fields[group_name]
            else:
                raise ValueError(f"Group {group_name} already exists in {filepath}")
        attrs = fields.create_group(group_name)
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)
    return i_snapshot + 1.0


def get_save_fields_simple(_filepath, write_all, overwrite=True):
    """Return a plain callback that appends one snapshot to HDF5 (no time gating)."""

    def _save_hdf5(i_snapshot, time, Q, Qaux):
        i_snap = int(i_snapshot)
        main_dir = misc.get_main_directory()

        filepath = os.path.join(main_dir, _filepath)

        with h5py.File(filepath, "a") as f:
            if i_snap == 0 and "fields" not in f.keys():
                fields = f.create_group("fields")
            else:
                fields = f["fields"]
            group_name = "iteration_" + str(i_snap)
            if group_name in fields:
                if overwrite:
                    del fields[group_name]
                else:
                    raise ValueError(f"Group {group_name} already exists in {filepath}")
            attrs = fields.create_group(group_name)
            attrs.create_dataset("time", data=time, dtype=float)
            attrs.create_dataset("Q", data=Q)
            if Qaux is not None:
                attrs.create_dataset("Qaux", data=Qaux)
        return i_snapshot + 1.0

    return _save_hdf5


def _save_hdf5(_filepath, i_snapshot, time, Q, Qaux, overwrite=True):
    i_snap = int(i_snapshot)
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, _filepath)

    with h5py.File(filepath, "a") as f:
        if i_snap == 0 and "fields" not in f.keys():
            fields = f.create_group("fields")
        else:
            fields = f["fields"]
        group_name = "iteration_" + str(i_snap)
        if group_name in fields:
            if overwrite:
                del fields[group_name]
            else:
                raise ValueError(f"Group {group_name} already exists in {filepath}")
        attrs = fields.create_group(group_name)
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)
    return i_snapshot + 1.0


def write_mesh_to_hdf5(_filepath, mesh, names=None):
    """Initialise the solver output h5 with the mesh topology under `/mesh/` so
    the file is SELF-DESCRIBING for the postprocessing readers
    (`zoomy_plotting.read_hdf5` → `columns_from_slice` / the column & mesh
    plots).  Written once at setup (fresh file); the per-snapshot field writer
    then appends `/fields/`.  No-op without h5py.

    `names` — the state variable names, in `Q` row order — are stored as
    `/fields@names` (the convention `zoomy_prepost` writes and
    `zoomy_plotting.read_hdf5` resolves), so a reader can ask the store for
    `"h"` instead of guessing that `q1` is the depth."""
    if not _HAVE_H5PY:
        return
    main_dir = misc.get_main_directory()
    filepath = os.path.join(main_dir, _filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with h5py.File(filepath, "w") as f:
        g = f.create_group("mesh")
        g.attrs["format_version"] = 2
        g.create_dataset("dimension", data=mesh.dimension)
        g.create_dataset("type", data=mesh.type)
        g.create_dataset("n_cells", data=mesh.n_cells)
        g.create_dataset("n_inner_cells", data=mesh.n_inner_cells)
        g.create_dataset("vertex_coordinates", data=mesh.vertex_coordinates)
        g.create_dataset("cell_vertices", data=mesh.cell_vertices)
        if names is not None:
            f.create_group("fields").attrs["names"] = np.array(
                [str(n) for n in names], dtype="S")


def get_save_fields(_filepath, write_all=False, overwrite=True):
    """Factory for a time-gated field writer compatible with solver time-stepping loops."""
    if _HAVE_H5PY:

        def save(time, next_write_at, i_snapshot, Q, Qaux):
            if write_all or time >= next_write_at:
                return _save_hdf5(
                    _filepath, i_snapshot, time, Q, Qaux, overwrite=overwrite
                )
            else:
                return i_snapshot
    else:

        def save(time, next_write_at, i_snapshot, Q, Qaux):
            """Save."""
            if write_all or time >= next_write_at:
                return i_snapshot + 1
            else:
                return i_snapshot

    return save


def save_fields_test(a):
    """Test helper: tuple interface around `_save_fields_to_hdf5`."""
    filepath, time, next_write_at, i_snapshot, Q, Qaux, write_all = a
    if not write_all and time < next_write_at:
        return i_snapshot

    _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux)
    return i_snapshot + 1


def load_mesh_from_hdf5(filepath):
    """Load a `LSQMesh` from disk via `Mesh.from_hdf5`."""
    mesh = Mesh.from_hdf5(filepath)
    return mesh


def load_fields_from_hdf5(filepath, i_snapshot=-1):
    """Read `Q`, `Qaux`, and time for one snapshot (default: latest)."""
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    with h5py.File(filepath, "r") as f:
        fields = f["fields"]
        if i_snapshot == -1:
            i_snapshot = len(fields.keys()) - 1
        else:
            i_snapshot = i_snapshot
        group = fields[f"iteration_{i_snapshot}"]
        time = group["time"][()]
        Q = group["Q"][()]
        Qaux = group["Qaux"][()]
    return Q, Qaux, time


def load_timeline_of_fields_from_hdf5(filepath):
    """Load all field snapshots plus mesh geometry from a combined HDF5 file."""
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    l_time = []
    l_Q = []
    l_Qaux = []
    mesh = Mesh.from_hdf5(filepath)
    with h5py.File(filepath, "r") as f:
        fields = f["fields"]
        n_snapshots = len(fields.keys())
        for i in range(n_snapshots):
            group = fields[f"iteration_{i}"]
            time = group["time"][()]
            Q = group["Q"][()]
            Qaux = group["Qaux"][()]
            l_time.append(time)
            l_Q.append(Q)
            l_Qaux.append(Qaux)
    return mesh.cell_centers[0], np.array(l_Q), np.array(l_Qaux), np.array(l_time)


# h5 -> vtk export lives in a single place now:
# `zoomy_prepost.steps.hdf5_to_vtk` (writes a `.vtu` series + ParaView `.pvd`).
# The former `generate_vtk` / `_write_to_vtk_from_vertices_edges` (legacy `.vtk`
# + `.vtk.series`) were removed; callers import zoomy_prepost lazily.
