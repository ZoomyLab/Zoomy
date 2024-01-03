import os
import numpy as np
import h5py
from sympy.abc import x
from sympy import lambdify, integrate

from library.pysolver.reconstruction import GradientMesh
import library.mesh.fvm_mesh as fvm_mesh
import library.misc.io as io
from library.model.models.shallow_moments import reconstruct_uvw

def recover_3d_from_smm_as_vtk(model, output_path, path_to_mesh, path_to_fields, Nz = 10, start_at_time=0, scale_h = 1.):
    fields =  h5py.File(path_to_fields, "r")
    mesh = fvm_mesh.Mesh.from_hdf5(path_to_mesh)
    n_snapshots = len(list(fields.keys()))

    Z = np.linspace(0, 1, Nz)

    mesh = GradientMesh.fromMesh(mesh)
    i_count = 0
    basis = model.basis
    lvl = model.levels
    phi = lambda z: np.array([lambdify(x, basis.basis(i, x))(z) for i in range(lvl + 1)])
    psi = lambda z: np.array([lambdify(x, integrate(basis.basis(i, x), (x, 0, 1)))(z) for i in range(lvl + 1)])
    print('init phi psi and mesh')
    for i_snapshot in range(n_snapshots):
        group = fields[str(i_snapshot)]
        time = group['time'][()]
        if time < start_at_time:
            continue
        Q = group['Q'][()]
        Qaux = group['Qaux'][()]
        gradQ = mesh.gradQ(Q)

        UVW = np.zeros((Q.shape[0]*Nz, 3), dtype=float)
        for i_elem, (q, gradq) in enumerate(zip(Q, gradQ)):
            u, v, w = reconstruct_uvw(q, gradq, model.levels, phi, psi)
            for iz, z in enumerate(Z):
                UVW[i_elem + (iz*mesh.n_elements), 0] = u(z)
                UVW[i_elem + (iz*mesh.n_elements), 1] = v(z)
                UVW[i_elem + (iz*mesh.n_elements), 2] = w(z)
        io._save_fields_to_hdf5(output_path,i_count, time, UVW, filename='fields3d.hdf5')
        i_count +=1
        print('converted {}'.format(str(i_snapshot)))
    print('write 3d')
        
    (points_3d, element_vertices_3d, mesh_type) = fvm_mesh.extrude_2d_element_vertices_mesh(mesh.type, mesh.vertex_coordinates, mesh.element_vertices, Q[:,0], Nz)
    io._write_to_vtk_from_vertices_edges(os.path.join(output_path, 'mesh3d.vtk'), mesh_type, points_3d, element_vertices_3d, fields =UVW)