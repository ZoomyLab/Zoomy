import os
import numpy as np
import jax
import jax.numpy as jnp
import h5py
import meshio
import json
import shutil

# import library.mesh.fvm_mesh as fvm_mesh
from library.mesh.mesh import *


def init_output_directory(path, clean):
    main_dir = os.getenv("SMS")
    path = os.path.join(main_dir, path)
    os.makedirs(path, exist_ok=True)
    if clean:
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            if os.path.isdir(os.path.join(path, f)):
                shutil.rmtree(os.path.join(path, f))
            else:
                os.remove(os.path.join(path, f))


def save_settings(filepath, settings):
    main_dir = os.getenv("SMS")
    filepath = os.path.join(main_dir, filepath)
    with h5py.File(os.path.join(filepath, "settings.hdf5"), "w") as f:
        attrs = f.create_group("parameters")
        for k, v in settings.parameters.items():
            attrs.create_dataset(k, data=v)
        f.create_dataset(
            "parameter_values",
            data=np.array(list(settings.parameters.values()), dtype=float),
        )
        f.create_dataset("name", data=settings.name)
        f.create_dataset("output_dir", data=settings.output_dir)
        f.create_dataset("output_snapshots", data=settings.output_snapshots)
        f.create_dataset("output_write_all", data=settings.output_write_all)
        f.create_dataset("output_clean_dir", data=settings.output_clean_dir)
        f.create_dataset(
            "truncate_last_time_step", data=settings.truncate_last_time_step
        )
        # f.create_dataset('reconstruction', data.settings.reconstruction.__name__, dtype=h5py.string_dtype())
        # f.create_dataset('reconstruction_edge', data.settings.reconstruction_edge.__name__, dtype=h5py.string_dtype())
        # f.create_dataset('num_flux', data.settings.num_flux.__name__, dtype=h5py.string_dtype())
        # f.create_dataset('nc_flux', data.settings.nc_flux.__name__, dtype=h5py.string_dtype())
        # f.create_dataset('compute_dt', data.settings.compute_dt.__name__, dtype=h5py.string_dtype())
        # f.create_dataset('compute_dt_args', data=settings.compute_dt_args)
        f.create_dataset("time_end", data=settings.time_end)
        f.create_dataset("callbacks", data=settings.callbacks)


def clean_files(filepath, filename=".vtk"):
    main_dir = os.getenv("SMS")
    abs_filepath = os.path.join(main_dir, filepath)
    if os.path.exists(abs_filepath):
        for file in os.listdir(abs_filepath):
            if file.endswith(filename):
                os.remove(os.path.join(abs_filepath, file))


def _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux=None, overwrite=True):
    i_snap = int(i_snapshot)
    jax.debug.print("SAVING")
    main_dir = os.getenv("SMS")
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
                raise ValueError(f"Group {group_name} already exists in {filename}")
        attrs = fields.create_group(group_name)
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)
    return i_snapshot + 1.0


def get_save_fields(_filepath, write_all, overwrite=True):
    def _save_hdf5(i_snapshot, time, Q, Qaux):
        i_snap = int(i_snapshot)
        main_dir = os.getenv("SMS")
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
                    raise ValueError(f"Group {group_name} already exists in {filename}")
            attrs = fields.create_group(group_name)
            attrs.create_dataset("time", data=time, dtype=float)
            attrs.create_dataset("Q", data=Q)
            if Qaux is not None:
                attrs.create_dataset("Qaux", data=Qaux)
        return i_snapshot + 1.0

    def save_fields(time, next_write_at, i_snapshot, Q, Qaux):
        condition = jnp.logical_and(not write_all, time < next_write_at)

        def do_nothing(_):
            return i_snapshot

        def do_save(_):
            # We define a small custom_jvp function that does the side effect
            @jax.custom_jvp
            def _save(i_snapshot, time, Q, Qaux):
                return jax.pure_callback(
                    _save_hdf5,
                    jax.ShapeDtypeStruct((), jnp.float64),
                    i_snapshot,
                    time,
                    Q,
                    Qaux,
                )
                # return i_snapshot + 1.

            @_save.defjvp
            def _save_jvp(primals, tangents):
                # Primal evaluation
                primal_out = _save(*primals)
                # Derivative is zero. We match shape/dtype (float64 scalar)
                # so returning 0.0 is valid.
                return primal_out, jnp.float64(0.0)

            # Then return the new integer snapshot
            return _save(i_snapshot, time, Q, Qaux)

        return jax.lax.cond(condition, do_nothing, do_save, operand=None)

    return save_fields


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
    main_dir = os.getenv("SMS")
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
    main_dir = os.getenv("SMS")
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
    assert (
        mesh_type == "triangle"
        or mesh_type == "quad"
        or mesh_type == "wface"
        or mesh_type == "hex"
        or mesh_type == "line"
        or mesh_type == "tetra"
    )
    d_fields = {}
    n_inner_elements = cell_vertices.shape[0]
    if fields is not None:
        if field_names is None:
            field_names = [str(i) for i in range(fields.shape[0])]
        for i_fields, _ in enumerate(fields):
            d_fields[field_names[i_fields]] = [fields[i_fields, :n_inner_elements]]
    point_d_fields = {}
    if point_fields is not None:
        if point_field_names is None:
            point_field_names = [str(i) for i in range(point_fields.shape[0])]
        for i_fields, _ in enumerate(point_fields):
            point_d_fields[point_field_names[i_fields]] = point_fields[i_fields]
    meshout = meshio.Mesh(
        vertex_coordinates,
        [(mesh_type, cell_vertices)],
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
):
    main_dir = os.getenv("SMS")
    abs_filepath = os.path.join(main_dir, filepath)
    path = os.path.dirname(abs_filepath)
    filename_out = "out"
    full_filepath_out = os.path.join(path, filename_out)
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
        output_vtk = f"{filename_out}.{get_iteration_from_datasetname(snapshot)}"

        # TODO callout to compute pointwise data?
        point_fields = None
        point_field_names = None

        if field_names is None:
            field_names = [str(i) for i in range(Q.shape[0])]
        if aux_field_names is None:
            aux_field_names = ["aux_{}".format(str(i)) for i in range(Qaux.shape[0])]

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
    with open(os.path.join(path, f"{full_filepath_out}.vtk.series"), "w") as f:
        json.dump(vtk_timestamp_file, f)

    file.close()
