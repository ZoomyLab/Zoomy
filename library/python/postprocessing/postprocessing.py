import os
import numpy as np
import h5py
from sympy.abc import x
from sympy import lambdify, integrate

from library.python.fvm.reconstruction import GradientMesh
import library.python.mesh.fvm_mesh as fvm_mesh
import library.python.mesh.mesh as petscMesh
import library.python.misc.io as io
from library.python.misc.logger_config import logger
from library.model.models.shallow_moments import reconstruct_uvw

def vtk_project_2d_to_3d(
    model, settings, Nz=10, start_at_time=0, scale_h=1.0, filename='out_3d'
):
    main_dir = os.getenv("ZOOMY_DIR")
    path_to_simulation = os.path.join(main_dir, os.path.join(settings.output.directory, f"{settings.output.filename}.h5"))    
    sim = h5py.File(path_to_simulation, "r")
    settings = io.load_settings(settings.output.directory)
    fields = sim["fields"]
    mesh = petscMesh.Mesh.from_hdf5(path_to_simulation)
    n_snapshots = len(list(fields.keys()))

    Z = np.linspace(0, 1, Nz)

    mesh_extr = petscMesh.Mesh.extrude_mesh(mesh, Nz)
    output_path = os.path.join(main_dir, settings.output.directory + f"/{filename}.h5")
    mesh_extr.write_to_hdf5(output_path)
    save_fields = io.get_save_fields_simple(output_path, True)

    #mesh = GradientMesh.fromMesh(mesh)
    i_count = 0
    pde = model._get_pde(printer='numpy')
    for i_snapshot in range(n_snapshots):
        group_name = "iteration_" + str(i_snapshot)
        group = fields[group_name]
        time = group["time"][()]
        if time < start_at_time:
            continue
        Q = group["Q"][()]
        Qaux = group["Qaux"][()]

        rhoUVWP = np.zeros((Q.shape[1] * Nz, 6), dtype=float)

        #for i_elem, (q, qaux) in enumerate(zip(Q.T, Qaux.T)):
        #    for iz, z in enumerate(Z):
        #        rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.project_2d_to_3d(np.array([0, 0, z]), q, qaux, parameters)
        for iz, z in enumerate(Z):
        
            #rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.project_2d_to_3d(np.array([0, 0, z]), q, qaux, parameters)
            # rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), 0] = Q[0, :mesh.n_inner_cells]
            Qnew = pde.project_2d_to_3d(np.array([0, 0, z]), Q[:, :mesh.n_inner_cells], Qaux[:, :mesh.n_inner_cells], model.parameter_values).T
            rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), :] = Qnew

        # rhoUVWP[mesh.n_inner_cells:mesh.n_inner_cells+mesh.n_inner_cells, 0] = Q[0, :mesh.n_inner_cells]

        qaux = np.zeros((Q.shape[1]*Nz, 1), dtype=float)
        _ = save_fields(i_snapshot, time, rhoUVWP.T, qaux.T)
        i_count += 1
        
        logger.info(f"Converted snapshot {i_snapshot}/{n_snapshots}")

    io.generate_vtk(output_path, filename=filename)
    logger.info(f"Output is written to: {output_path}/{filename}.*.vtk")


def write_to_calibration_dataformat(
    input_folderpath: str, output_filepath: str, field_names=None, aux_field_names=None
):
    fields = h5py.File(os.path.join(input_folderpath, "fields.hdf5"), "r")
    settings = h5py.File(os.path.join(input_folderpath, "settings.hdf5"), "r")
    # mesh =  h5py.File(os.path.join(input_folderpath, 'mesh.hdf5'), "r")
    mesh = fvm_mesh.Mesh.from_hdf5(os.path.join(input_folderpath, "mesh.hdf5"))
    snapshots = list(fields.keys())

    n_variables = fields[str(0)]["Q"][()].shape[1]
    n_aux_variables = fields[str(0)]["Qaux"][()].shape[1]

    if field_names is None:
        field_names = [str(i) for i in range(n_variables)]
    if aux_field_names is None:
        aux_field_names += [str(i) for i in range(n_aux_variables)]
    # convert back to dict
    parameters = {key: value[()] for key, value in settings["parameters"].items()}
    # parameters = settings['parameters'][()]

    main_dir = os.getenv("ZOOMY_DIR")
    f = h5py.File(os.path.join(main_dir, output_filepath), "w")

    # write static data, e.g. mesh, parameters, name
    attrs = f.create_group("mesh")
    attrs.create_dataset("centers", data=mesh.element_center)
    attrs = f.create_group("parameters")
    for k, v in parameters.items():
        attrs.create_dataset(k, data=v)

    grp = f.create_group("timeseries")
    for i_snapshot in range(len(snapshots)):
        # load timeseries data
        time = fields[str(i_snapshot)]["time"][()]
        Q = fields[str(i_snapshot)]["Q"][()]
        Qaux = fields[str(i_snapshot)]["Qaux"][()]

        # write timeseries data
        attrs = grp.create_group(str(i_snapshot))
        attrs.create_dataset("time", data=time, dtype=float)
        for i, field_name in enumerate(field_names):
            attrs.create_dataset(field_name, data=Q[:, i])
        for i, field_name in enumerate(aux_field_names):
            attrs.create_dataset(field_name, data=Qaux[:, i])

    f.close()
    settings.close()
    fields.close()
