import os 
import numpy as np
import h5py
import meshio
import json

import library.mesh.fvm_mesh as fvm_mesh


def init_output_directory(path, clean):
    os.makedirs(path, exist_ok=True)
    if clean:
        filelist = [ f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))

def save_settings(filepath, settings):
    with h5py.File(os.path.join(filepath, 'settings.hdf5'), "w") as f:
        attrs = f.create_group('parameters')
        for k, v in settings.parameters.items():
            attrs.create_dataset(k, data=v)

def save_fields(filepath, time, next_write_at, i_snapshot, Q, Qaux, write_all, filename='fields.hdf5'):
    if not write_all and  time < next_write_at:
        return i_snapshot

    _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux, filename=filename)
    return i_snapshot +1

def _save_fields_to_hdf5(filepath, i_snapshot, time, Q, Qaux=None, filename='fields.hdf5'):
    with h5py.File(os.path.join(filepath, filename), "a") as f:
        attrs = f.create_group(str(i_snapshot))
        attrs.create_dataset("time", data=time, dtype=float)
        attrs.create_dataset("Q", data=Q)
        if Qaux is not None:
            attrs.create_dataset("Qaux", data=Qaux)

def load_fields_from_hdf5(filepath, i_snapshot = -1):
    with h5py.File(filepath, "r") as f:
        if i_snapshot == -1:
            i_snapshot = len(f.keys())-1
        else:
            i_snapshot = i_snapshot
        group = f[str(i_snapshot)]
        time = group['time'][()]
        Q = group['Q'][()]
        Qaux = group['Qaux'][()]
    return Q, Qaux, time


def _write_to_vtk_from_vertices_edges(
    filepath,
    mesh_type,
    vertex_coordinates,
    element_vertices,
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
    if fields is not None:
        if field_names is None:
            field_names = [str(i) for i in range(fields.shape[0])]
        for i_fields, _ in enumerate(fields.T):
            d_fields[field_names[i_fields]] = [fields[:, i_fields]]
    point_d_fields = {}
    if point_fields is not None:
        if point_field_names is None:
            point_field_names = [str(i) for i in range(point_fields.shape[0])]
        for i_fields, _ in enumerate(point_fields):
            point_d_fields[point_field_names[i_fields]] = point_fields[i_fields]
    meshout = meshio.Mesh(
        vertex_coordinates,
        [(fvm_mesh.convert_mesh_type_to_meshio_mesh_type(mesh_type), element_vertices)],
        cell_data=d_fields,
        point_data=point_d_fields,
    )
    path, filename = os.path.split(filepath)
    filename_base, filename_ext = os.path.splitext(filename)
    os.makedirs(path, exist_ok=True)
    meshout.write(filepath + ".vtk")
    

def generate_vtk(filepath: str, field_names=None, aux_field_names=None, filename_fields='fields.hdf5', filename_out = 'out'):
    # with h5py.File(os.path.join(filepath, 'mesh'), "r") as file_mesh, h5py.File(os.path.join(filepath, 'fields'), "r") as file_fields:
    file_fields =  h5py.File(os.path.join(filepath, filename_fields), "r")
    mesh = fvm_mesh.Mesh.from_hdf5(os.path.join(filepath, 'mesh.hdf5'))
    snapshots = list(file_fields.keys())
    # init timestamp file
    vtk_timestamp_file = {"file-series-version": "1.0", "files": []}

    # write out vtk files for each timestamp
    for snapshot in snapshots:
        time = file_fields[snapshot]['time'][()]
        Q = file_fields[snapshot]['Q'][()]
        Qaux = file_fields[snapshot]['Qaux'][()]
        filename = f'{filename_out}.{int(snapshot)}'
        fullpath = os.path.join(filepath, filename )

        #TODO callout to compute pointwise data?
        point_fields = None
        point_field_names = None


        if field_names is None:
            field_names = [str(i) for i in range(Q.shape[1])]
        if aux_field_names is None:
            aux_field_names = ['aux_{}'.format(str(i)) for i in range(Qaux.shape[1])]

        fields = np.concatenate((Q, Qaux), axis=-1)
        field_names = field_names + aux_field_names

        _write_to_vtk_from_vertices_edges(fullpath, mesh.type, mesh.vertex_coordinates, mesh.element_vertices, fields=fields, field_names=field_names,point_fields=point_fields, point_field_names=point_field_names)

        vtk_timestamp_file["files"].append(
            {
                "name": filename + '.vtk',
                "time": time,
            }
        )

    #finalize vtk
    with open(os.path.join(filepath , f"{filename_out}.vtk.series"), "x") as f:
        json.dump(vtk_timestamp_file, f)

    file_fields.close()
