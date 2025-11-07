import os
import numpy as np
import json
import shutil

try:
    import meshio

    _HAVE_MESHIO = True
except:
    _HAVE_MESHIO = False

try:
    import h5py

    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

# import zoomy_core.mesh.fvm_mesh as fvm_mesh
from zoomy_core.mesh.mesh import Mesh
import zoomy_core.mesh.mesh_util as mesh_util
from zoomy_core.misc.misc import Zstruct, Settings
from zoomy_core.misc import misc as misc
from zoomy_core.misc.logger_config import logger


def init_output_directory(path, clean):
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
    out = type(value)
    if isinstance(value, str):
        out = h5py.string_dtype()
    return out


def write_dict_to_hdf5(group, d):
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
            logger.warning(f"Skipping unsupported type for key: {
                           key} -> {type(value)}")


def load_hdf5_to_dict(group):
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
            logger.warning(f"Skipping unsupported type for key: {
                           key} -> {type(value)}")

    return d


def save_settings(settings):
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, settings.output.directory)
    with h5py.File(os.path.join(filepath, "settings.h5"), "w") as f:
        write_dict_to_hdf5(f, settings.as_dict(recursive=True))


def load_settings(filepath):
    main_dir = misc.get_main_directory()

    filepath = os.path.join(main_dir, filepath)
    with h5py.File(os.path.join(filepath, "settings.h5"), "r") as f:
        d = load_hdf5_to_dict(f)

    settings = Settings.from_dict(d)
    return settings


def load_settings2(filepath):
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
        if i_snap == 0 and not "fields" in f.keys():
            fields = f.create_group("fields")
        else:
            fields = f["fields"]
        group_name = "iteration_" + str(i_snap)
        if group_name in fields:
            if overwrite:
                del fields[group_name]
            else:
                raise ValueError(
                    f"Group {group_name} already exists in {filepath}")
        attrs = fields.create_group(group_name)
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)
    return i_snapshot + 1.0


def get_save_fields_simple(_filepath, write_all, overwrite=True):
    def _save_hdf5(i_snapshot, time, Q, Qaux):
        i_snap = int(i_snapshot)
        main_dir = misc.get_main_directory()

        filepath = os.path.join(main_dir, _filepath)

        with h5py.File(filepath, "a") as f:
            if i_snap == 0 and not "fields" in f.keys():
                fields = f.create_group("fields")
            else:
                fields = f["fields"]
            group_name = "iteration_" + str(i_snap)
            if group_name in fields:
                if overwrite:
                    del fields[group_name]
                else:
                    raise ValueError(
                        f"Group {group_name} already exists in {filepath}")
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
        if i_snap == 0 and not "fields" in f.keys():
            fields = f.create_group("fields")
        else:
            fields = f["fields"]
        group_name = "iteration_" + str(i_snap)
        if group_name in fields:
            if overwrite:
                del fields[group_name]
            else:
                raise ValueError(
                    f"Group {group_name} already exists in {filepath}")
        attrs = fields.create_group(group_name)
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)
    return i_snapshot + 1.0


def get_save_fields(_filepath, write_all=False, overwrite=True):
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
            if write_all or time >= next_write_at:
                return i_snapshot + 1
            else:
                return i_snapshot

    return save


def save_fields_test(a):
    filepath, time, next_write_at, i_snapshot, Q, Qaux, write_all = a
    if not write_all and time < next_write_at:
        return i_snapshot

    _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux)
    return i_snapshot + 1


def load_mesh_from_hdf5(filepath):
    mesh = Mesh.from_hdf5(filepath)
    return mesh


def load_fields_from_hdf5(filepath, i_snapshot=-1):
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


def _write_to_vtk_from_vertices_edges(
    filepath,
    mesh_type,
    vertex_coordinates,
    cell_vertices,
    fields=None,
    field_names=None,
    point_fields=None,
    point_field_names=None,
):
    if not _HAVE_MESHIO:
        raise RuntimeError(
            "_write_to_vtk_from_vertices_edges requires meshio, which is not available."
        )
    assert (
        mesh_type == "triangle"
        or mesh_type == "quad"
        or mesh_type == "wface"
        or mesh_type == "hexahedron"
        or mesh_type == "line"
        or mesh_type == "tetra"
    )
    d_fields = {}
    n_inner_elements = cell_vertices.shape[0]
    if fields is not None:
        if field_names is None:
            field_names = [str(i) for i in range(fields.shape[0])]
        for i_fields, _ in enumerate(fields):
            d_fields[field_names[i_fields]] = [
                fields[i_fields, :n_inner_elements]]
    point_d_fields = {}
    if point_fields is not None:
        if point_field_names is None:
            point_field_names = [str(i) for i in range(point_fields.shape[0])]
        for i_fields, _ in enumerate(point_fields):
            point_d_fields[point_field_names[i_fields]
                           ] = point_fields[i_fields]
    meshout = meshio.Mesh(
        vertex_coordinates,
        [(mesh_util.convert_mesh_type_to_meshio_mesh_type(mesh_type), cell_vertices)],
        cell_data=d_fields,
        point_data=point_d_fields,
    )
    path, filename = os.path.split(filepath)
    filename_base, filename_ext = os.path.splitext(filename)
    os.makedirs(path, exist_ok=True)
    meshout.write(filepath + ".vtk")


def generate_vtk(
    filepath: str,
    field_names=None,
    aux_field_names=None,
    skip_aux=False,
    filename="out",
    warp=False,
):
    main_dir =  misc.get_main_directory()
    abs_filepath = os.path.join(main_dir, filepath)
    path = os.path.dirname(abs_filepath)
    full_filepath_out = os.path.join(path, filename)
    # abs_filepath = os.path.join(main_dir, filepath)
    # with h5py.File(os.path.join(filepath, 'mesh'), "r") as file_mesh, h5py.File(os.path.join(filepath, 'fields'), "r") as file_fields:
    file = h5py.File(os.path.join(main_dir, filepath), "r")
    file_fields = file["fields"]
    mesh = Mesh.from_hdf5(abs_filepath)
    snapshots = list(file_fields.keys())
    # init timestamp file
    vtk_timestamp_file = {"file-series-version": "1.0", "files": []}

    def get_iteration_from_datasetname(name):
        return int(name.split("_")[1])

    # write out vtk files for each timestamp
    for snapshot in snapshots:
        time = file_fields[snapshot]["time"][()]
        Q = file_fields[snapshot]["Q"][()]

        if not skip_aux:
            Qaux = file_fields[snapshot]["Qaux"][()]
        else:
            Qaux = np.empty((Q.shape[0], 0))
        output_vtk = f"{filename}.{get_iteration_from_datasetname(snapshot)}"

        # TODO callout to compute pointwise data?
        point_fields = None
        point_field_names = None

        if field_names is None:
            field_names = [str(i) for i in range(Q.shape[0])]
        if aux_field_names is None:
            aux_field_names = ["aux_{}".format(
                str(i)) for i in range(Qaux.shape[0])]

        fields = np.concatenate((Q, Qaux), axis=0)
        field_names = field_names + aux_field_names

        vertex_coordinates_3d = np.zeros((mesh.vertex_coordinates.shape[1], 3))
        vertex_coordinates_3d[:, : mesh.dimension] = mesh.vertex_coordinates.T

        _write_to_vtk_from_vertices_edges(
            os.path.join(path, output_vtk),
            mesh.type,
            vertex_coordinates_3d,
            mesh.cell_vertices.T,
            fields=fields,
            field_names=field_names,
            point_fields=point_fields,
            point_field_names=point_field_names,
        )

        vtk_timestamp_file["files"].append(
            {
                "name": output_vtk + ".vtk",
                "time": time,
            }
        )
    # finalize vtk
    with open(os.path.join(path, f"{filename}.vtk.series"), "w") as f:
        json.dump(vtk_timestamp_file, f)

    file.close()
