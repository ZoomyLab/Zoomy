
import numpy as np
import re
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import pyvista as pv
import os

def get_interface_position(data, pos):
    def objective(interface_pos):
        # f_data = CubicSpline(pos, data)
        f_data = lambda x: np.interp(x, pos, data)
        f_recon = lambda x: 1.*(x <= interface_pos) + 0.
        integral, error = quad(lambda x: (f_data(x) - f_recon(x))**2, pos[0], pos[-1], points=pos, limit=2*pos.shape[0])
        # assert error < 10**(-6)
        return integral

    res = minimize_scalar(objective, [pos[0], 0.5*(pos[0] + pos[-1]), pos[-1]], [pos[0], pos[-1]])
    interface_pos = res.x
    return interface_pos


def read_vtk(filepath):
    reader = pv.get_reader(filepath)
    mesh = reader.read()
    scalar_field_names = mesh.cell_data.keys()
    return mesh, scalar_field_names

def extract_smm_data_1d(directory, filename='internal.vtu', pos = [2.2, 0, 0], stride=10):
    file_names = [name for name in os.listdir(directory) if name.endswith('.vtk')]
    def sorting_key(name):
        number = re.finall(r'\d+', name)
        return number
    file_names = sorted(folder_names, key=sorting_key)
    file_names = file_names[::stride]
    l_q = []
    iteration = []
    for file in file_names:
        path = os.path.join(directory,file)
        mesh, _ = read_vtk(path)

        x = np.array(mesh.cell_centers().points)[:, 0]
        q = np.array(mesh.cell_data)
        sort_order = np.argsort(x)
        x = x[sort_order]
        q = q[sort_order]

        l_q.append(q)
        iteration.append(sorting_key(file))
    return x, np.array(l_q), np.array(iteration)


def extract_1d_slice(mesh, pos=[15, 0, 0]):
    """
    extract the data at pos 15
    also, reconstruct h and select U that is only part of the water (and resample it for later plotting)
    """
    mesh2d = mesh.slice(normal = [0,0,1], origin=[0, 0, 0.05])
    slice1d = mesh2d.slice(normal=[1,0,0], origin=pos)
    x = np.array(slice1d.cell_centers().points)[:, 1]
    alpha = np.array(slice1d.cell_data['alpha.water'])
    U = np.array(slice1d.cell_data['U'])
    sort_order = np.argsort(x)
    x = x[sort_order]
    alpha = alpha[sort_order]
    U = U[sort_order]
    h = get_interface_position(alpha, x)

    # cut away the air and resample
    indices_water = np.array(list(range(x.shape[0])))[x <= h]
    U = U[indices_water]
    u = U[:, 0]
    w = U[:, 1]
    x_unitheight = np.linspace(0,1,100)
    u = np.interp(h*x_unitheight,x[indices_water],  u[indices_water])
    w = np.interp(h*x_unitheight,x[indices_water],  w[indices_water])
    return x, alpha, h, u, w

def extract_1d_data(directory, filename='internal.vtu', pos = [15, 0, 0], stride=10):
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    def sorting_key(name):
        prefix = 'nozzle_openfoam_'
        if name.startswith(prefix):
            return int(name[len(prefix):])
        else:
            return name
    folder_names = sorted(folder_names, key=sorting_key)
    folder_names = folder_names[::stride]
    l_h = []
    l_u = []
    l_w = []
    iteration = []
    for folder in folder_names:
        path = os.path.join(os.path.join(directory,folder), filename)
        mesh, _ = read_vtk(path)
        x, alpha, h, u, w = extract_1d_slice(mesh, pos)
        l_h.append(h)
        l_u.append(u)
        l_w.append(w)
        iteration.append(sorting_key(folder))
    return x, np.array(l_h), np.array(l_u), np.array(l_w), np.array(iteration)

def project_to_smm(u, z, basis=[lambda z:1.*np.ones_like(z), lambda z:(1-2*z)]):
    u_spline = CubicSpline(z, u)

    moments = np.zeros((len(basis)), dtype=float)

    for i, b in enumerate(basis):
        nominator, error = quad(lambda z: u_spline(z) * b(z), z[0], z[-1]) 
        assert error < 10**(-6)
        denominator, error =  quad(lambda z: b(z) * b(z), z[0], z[-1])
        assert error < 10**(-6)
        moments[i] = nominator / denominator
    
    return moments

def test_project_to_smm():
    z = np.linspace(0, 1, 100)
    # u = 1. * np.ones_like(z)
    u = 1. * np.ones_like(z) + 0.5 * (1-2*z)
    moments = project_to_smm(u, z)
    print(moments)

if __name__ == '__main__':
    test_project_to_smm()

