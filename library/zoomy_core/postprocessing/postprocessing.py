import os
import numpy as np

try:
    import h5py

    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

import zoomy_core.mesh.mesh as petscMesh
import zoomy_core.misc.io as io
from zoomy_core.misc.logger_config import logger
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

def vtk_project_2d_to_3d(
    model, settings, start_at_time=0, scale_h=1.0, filename='out_3d'
):
    if not _HAVE_H5PY:
        raise ImportError("h5py is required for vtk_project_2d_to_3d function.")
    Nz = model.number_of_points_3d
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
    pde = NumpyRuntimeModel(model)
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
            Qnew = pde.project_2d_to_3d(z, Q[:, :mesh.n_inner_cells], Qaux[:, :mesh.n_inner_cells], model.parameter_values).T

            #rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.project_2d_to_3d(np.array([0, 0, z]), q, qaux, parameters)
            # rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), 0] = Q[0, :mesh.n_inner_cells]
            rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), :] = Qnew[iz, :]

        # rhoUVWP[mesh.n_inner_cells:mesh.n_inner_cells+mesh.n_inner_cells, 0] = Q[0, :mesh.n_inner_cells]

        qaux = np.zeros((Q.shape[1]*Nz, 1), dtype=float)
        _ = save_fields(i_snapshot, time, rhoUVWP.T, qaux.T)
        i_count += 1
        
        logger.info(f"Converted snapshot {i_snapshot}/{n_snapshots}")

    io.generate_vtk(output_path, filename=filename)
    logger.info(f"Output is written to: {output_path}/{filename}.*.vtk")

