import os
import numpy as np
import h5py
from sympy.abc import x
from sympy import lambdify, integrate

from library.fvm.reconstruction import GradientMesh
import library.mesh.fvm_mesh as fvm_mesh
import library.mesh.mesh as petscMesh
import library.misc.io as io
from library.model.models.shallow_moments import reconstruct_uvw
from library.model.models.base import RuntimeModel

def vtk_interpolate_3d(
    model, output_path, path_to_simulation, Nz=10, start_at_time=0, scale_h=1.0
):
    sim = h5py.File(path_to_simulation, "r")
    parameters = io.load_settings(os.path.join(path_to_simulation, "settings.hdf5"))
    fields = sim["fields"]
    mesh = petscMesh.Mesh.from_hdf5(path_to_simulation)
    n_snapshots = len(list(fields.keys()))

    Z = np.linspace(0, 1, Nz)

    mesh_extr = petscMesh.Mesh.extrude_mesh(mesh, Nz)
    output_path = 'output/out3d.h5'
    mesh_extr.write_to_hdf5(output_path)
    save_fields = io.get_save_fields_simple(output_path, True)

    #mesh = GradientMesh.fromMesh(mesh)
    i_count = 0
    pde = model.get_pde(printer='numpy')
    for i_snapshot in range(n_snapshots):
        group_name = "iteration_" + str(i_snapshot)
        group = fields[group_name]
        time = group["time"][()]
        if time < start_at_time:
            continue
        Q = group["Q"][()]
        Qaux = group["Qaux"][()]

        rhoUVWP = np.zeros((Q.shape[1] * Nz, 5), dtype=float)

        #for i_elem, (q, qaux) in enumerate(zip(Q.T, Qaux.T)):
        #    for iz, z in enumerate(Z):
        #        rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.interpolate_3d(np.array([0, 0, z]), q, qaux, parameters)
        for iz, z in enumerate(Z):
        
            #rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.interpolate_3d(np.array([0, 0, z]), q, qaux, parameters)
            # rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), 0] = Q[0, :mesh.n_inner_cells]
            Qnew = pde.interpolate_3d(np.array([0, 0, z]), Q[:, :mesh.n_inner_cells], Qaux[:, :mesh.n_inner_cells], parameters).T
            rhoUVWP[(iz * mesh.n_inner_cells):((iz+1) * mesh.n_inner_cells), :] = Qnew

        # rhoUVWP[mesh.n_inner_cells:mesh.n_inner_cells+mesh.n_inner_cells, 0] = Q[0, :mesh.n_inner_cells]

        qaux = np.zeros((Q.shape[1]*Nz, 1), dtype=float)
        _ = save_fields(i_snapshot, time, rhoUVWP.T, qaux.T)
        i_count += 1
        print("converted {}".format(str(i_snapshot)))

    io.generate_vtk(output_path)
    print(f"write 3d: {output_path}")


def vtk_interpolate_3d_old(
    model, output_path, path_to_simulation, Nz=10, start_at_time=0, scale_h=1.0
):
    sim = h5py.File(path_to_simulation, "r")
    parameters = io.load_settings(os.path.join(path_to_simulation, "settings.hdf5"))
    # mesh = sim['mesh']
    fields = sim["fields"]
    mesh = petscMesh.Mesh.from_hdf5(path_to_simulation)
    n_snapshots = len(list(fields.keys()))

    Z = np.linspace(0, 1, Nz)

    #mesh = GradientMesh.fromMesh(mesh)
    i_count = 0
    basis = model.basismatrices.basisfunctions
    lvl = model.levels
    phi = lambda z: np.array(
        [lambdify(x, basis.basis(i, x))(z) for i in range(lvl + 1)]
    )
    psi = lambda z: np.array(
        [
            lambdify(x, integrate(basis.basis(i, x), (x, 0, 1)))(z)
            for i in range(lvl + 1)
        ]
    )
    pde = model.get_pde(printer='numpy')
    #for i_snapshot in range(n_snapshots):
    i_snapshot = n_snapshots-1 
    if True:
        group_name = "iteration_" + str(i_snapshot)
        group = fields[group_name]
        time = group["time"][()]
        #if time < start_at_time:
        #    continue
        Q = group["Q"][()]
        Qaux = group["Qaux"][()]


        rhoUVWP = np.zeros((Q.shape[1] * Nz, 5), dtype=float)
        for i_elem, (q, qaux) in enumerate(zip(Q.T, Qaux.T)):
            #u, v, w = reconstruct_uvw(q, gradq, model.levels, phi, psi)

            for iz, z in enumerate(Z):
                rhoUVWP[i_elem + (iz * mesh.n_cells), :] = pde.interpolate_3d(np.array([0, 0, z]), q, qaux, parameters)
        # io._save_fields_to_hdf5(
        #     os.path.join(output_path, 'fields3d.h5'), i_count, time, rhoUVWP"
        # )
        i_count += 1
        print("converted {}".format(str(i_snapshot)))
    print("write 3d")

    (points_3d, element_vertices_3d, mesh_type) = (
        fvm_mesh.extrude_2d_element_vertices_mesh(
            mesh.type, mesh.vertex_coordinates, mesh.cell_vertices, Q[0] * scale_h, Nz
        )
    )
    io._write_to_vtk_from_vertices_edges(
        os.path.join(output_path, "mesh3d.vtk"),
        mesh_type,
        points_3d,
        element_vertices_3d.T,
        fields=rhoUVWP.T,
    )

def recover_3d_from_smm_as_vtk(
    model, output_path, path_to_simulation, Nz=10, start_at_time=0, scale_h=1.0
):
    sim = h5py.File(path_to_simulation, "r")
    fields = sim["fields"]
    # mesh = sim['mesh']
    mesh = petscMesh.Mesh.from_hdf5(path_to_simulation)
    n_snapshots = len(list(fields.keys()))

    Z = np.linspace(0, 1, Nz)

    mesh = GradientMesh.fromMesh(mesh)
    i_count = 0
    for i_snapshot in range(n_snapshots):
        group = fields[str(i_snapshot)]
        time = group["time"][()]
        if time < start_at_time:
            continue
        Q = group["Q"][()]
        Qaux = group["Qaux"][()]
        gradQ = mesh.gradQ(Q)

        UVW = np.zeros((Q.shape[0] * Nz, 3), dtype=float)
        for i_elem, (q, gradq) in enumerate(zip(Q, gradQ)):
            u, v, w = reconstruct_uvw(q, gradq, model.levels, phi, psi)
            for iz, z in enumerate(Z):
                UVW[i_elem + (iz * mesh.n_elements), 0] = u(z)
                UVW[i_elem + (iz * mesh.n_elements), 1] = v(z)
                UVW[i_elem + (iz * mesh.n_elements), 2] = w(z)
        io._save_fields_to_hdf5(
            output_path, i_count, time, UVW, filename="fields3d.hdf5"
        )
        i_count += 1
        print("converted {}".format(str(i_snapshot)))
    print("write 3d")

    (points_3d, element_vertices_3d, mesh_type) = (
        fvm_mesh.extrude_2d_element_vertices_mesh(
            mesh.type, mesh.vertex_coordinates, mesh.element_vertices, Q[:, 0], Nz
        )
    )
    io._write_to_vtk_from_vertices_edges(
        os.path.join(output_path, "mesh3d.vtk"),
        mesh_type,
        points_3d,
        element_vertices_3d,
        fields=UVW,
    )


def append_custom_fields_to_aux_fields_for_hdf5(input_folderpath, custom_functions):
    fields = h5py.File(os.path.join(input_folderpath, "fields.hdf5"), "r+")
    settings = h5py.File(os.path.join(input_folderpath, "settings.hdf5"), "r")
    mesh = fvm_mesh.Mesh.from_hdf5(os.path.join(input_folderpath, "mesh.hdf5"))
    snapshots = list(fields.keys())
    n_fields = fields[str(0)]["Q"].shape[1]
    n_aux_fields = fields[str(0)]["Qaux"][()].shape[1]

    parameters = {key: value[()] for key, value in settings["parameters"].items()}
    Qaux_new = np.zeros(
        (mesh.n_elements, n_aux_fields + len(custom_functions)), dtype=float
    )

    for i_snapshot in range(len(snapshots)):
        # load timeseries data
        time = fields[str(i_snapshot)]["time"][()]
        Q = fields[str(i_snapshot)]["Q"][()]
        Qaux = fields[str(i_snapshot)]["Qaux"][()]
        Qaux_new[:, :n_aux_fields] = Qaux
        for i, (name, func) in enumerate(custom_functions):
            Qaux_new[:, n_aux_fields + i] = func(
                mesh.element_center, Q, Qaux, parameters
            )
        del fields[str(i_snapshot)]["Qaux"]
        fields[str(i_snapshot)].create_dataset("Qaux", data=Qaux_new)


def write_to_calibration_dataformat(
    input_folderpath: str, output_filepath: str, field_names=None, aux_field_names=None
):
    fields = h5py.File(os.path.join(input_folderpath, "fields.hdf5"), "r")
    settings = h5py.File(os.path.join(input_folderpath, "settings.hdf5"), "r")
    # mesh =  h5py.File(os.path.join(input_folderpath, 'mesh.hdf5'), "r")
    mesh = fvm_mesh.Mesh.from_hdf5(os.path.join(input_folderpath, "mesh.hdf5"))
    snapshots = list(fields.keys())

    n_fields = fields[str(0)]["Q"][()].shape[1]
    n_aux_fields = fields[str(0)]["Qaux"][()].shape[1]

    if field_names is None:
        field_names = [str(i) for i in range(n_fields)]
    if aux_field_names is None:
        aux_field_names += [str(i) for i in range(n_aux_fields)]
    # convert back to dict
    parameters = {key: value[()] for key, value in settings["parameters"].items()}
    # parameters = settings['parameters'][()]

    main_dir = os.getenv("SMS")
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


# TODO unfinished
# def combine_calibration_hdf5_files(filepath_list, output_filepath, combine_along_parameter=['nm']):
#     main_dir = os.getenv("SMS")
#     f =  h5py.File(os.path.join(main_dir , output_filepath), "w")
#     # load the first mesh to extract the mesh and parameters
#     fin = h5py.File(filepath_list[0], "r")

#     attrs = f.create_group("mesh")
#     attrs.create_dataset("centers", data=f['mesh']['centers'][()])

#     parameters = {key: value[()] for key, value in settings['parameters'].items()}
#     for param in combine_along_parameter:
#         del parameters[param]
#     attrs = f.create_group("parameters")
#     for k, v in parameters.items():
#         attrs.create_dataset(k, data=v)


#     grp = f.create_group('simulations')
#     for i, filepath in enumerate(filepath_list):
#         fin = h5py.File(filepath, "r")
#         params = [(param, parameters[param]) for param in combine_along_parameter]
#         sim = grp.create_group(f'{i}')
#         for (p_name, p_value) in params:
#             sim.create_dataset(p_name, data=p_value, dtype=float)
#             timeseries = grp.create_group('timeseries')
#             snapshots = list(fin['timeseries'].keys())
#             for i_snapshot in range(len(snapshots)):
#                 timeseries.create_dataset("time", data=fin['timeseries']['0']['time'][()], dtype=float)


#         fin.close()

#     f.close()
