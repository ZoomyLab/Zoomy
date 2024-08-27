
import numpy as np
import re
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import pyvista as pv
import os

from library.model.model import *

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

12
def extract_1d_slice(mesh, pos=[15, 0, 0]):
    """
    extract the data at pos 15
    also, reconstruct h and select U that is only part of the water (and resample it for later plotting)
    """
    mesh2d = mesh.slice(normal = [0,0,1], origin=[0, 0, mesh.cell_centers().points[0,2]])
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

def extract_1d_data_foam11(directory, pos = [15, 0, 0], stride=10):
    file_names = [name for name in os.listdir(directory) if name.endswith('.vtk')]
    def sorting_key(name):
        numbers = re.findall(r'\d+', name)
        assert len(numbers) == 1
        return int(numbers[0])
    file_names = sorted(file_names, key=sorting_key)
    file_names = file_names[::stride]
    l_h = []
    l_u = []
    l_w = []
    iteration = []
    for i, file in enumerate(file_names):
        path = os.path.join(directory,file)
        mesh, _ = read_vtk(path)
        x, alpha, h, u, w = extract_1d_slice(mesh, pos)
        l_h.append(h)
        l_u.append(u)
        l_w.append(w)
        iteration.append(i * stride)
    return x, np.array(l_h), np.array(l_u), np.array(l_w), np.array(iteration)

def extract_1d_data(directory, filename='internal.vtu', pos = [15, 0, 0], stride=10, foam11=True):
    if foam11:
        return extract_1d_data_foam11(directory, pos, stride)
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    def sorting_key(name):
        prefix = 'nozzle_openfoam_'
        # prefix = 'openfoam_'
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

def project_to_smm(u, z, basis=[lambda z:1.*np.ones_like(z), lambda z:(1-2*z)], weights='uniform', continuous=False):
    moments = np.zeros((len(basis)), dtype=float)

    w = lambda z, order: 1.
    if weights=='linear':
        w = lambda z, order: (1-z)**order
    if continuous:
        u_spline = CubicSpline(z, u)

        for i, b in enumerate(basis):
            nominator, error = quad(lambda z: u_spline(z) * b(z) * w(z, i), z[0], z[-1], limit=100) 
            assert error < 10**(-4)
            denominator, error =  quad(lambda z: b(z) * b(z), z[0], z[-1], limit=100)
            assert error < 10**(-4)
            moments[i] = nominator / denominator
    else:
        u_sampled = u

        for i, b in enumerate(basis):
            nominator = simpson(u_sampled * b(z) * w(z, i), x=z) 
            denominator=  simpson(b(z) * b(z), x=z)
            moments[i] = nominator / denominator

    
    return moments

def test_project_to_smm():
    z = np.linspace(0, 1, 100)
    # u = 1. * np.ones_like(z)
    u = 1. * np.ones_like(z) + 0.5 * (1-2*z)
    moments = project_to_smm(u, z)

def project_openfoam_to_smm(directory, pos=[0.5, 0, 0], stride=60, dt=0.01, level=10, output_uw=False, weights='uniform'):
    pos, h, u, w, iteration = extract_1d_data(directory, pos=pos, stride=stride)
    iteration_times = dt * iteration
    basis_analytical =Legendre_shifted(order=level+1)
    basis = [basis_analytical.get_lambda(k) for k in range(level+1)]

    moments = []
    moments_w = []
    z = np.linspace(0, 1, u.shape[1])
    for u_n in u:
        moments.append(project_to_smm(u_n, z, basis=basis, weights=weights))
    for w_n in w:
        moments_w.append(project_to_smm(w_n, z, basis=basis))
    moments = np.array(moments)
    moments_w = np.array(moments_w)
    if output_uw:
        return h, moments, moments_w, iteration_times, u, w
    return h, moments, moments_w, iteratio_extendedn_times

def shear_at_bottom_moments(Q):
    level = Q.shape[1]-2
    basis_analytical =Legendre_shifted(order=level)
    basis_analytical.get_diff_basis()
    basis_0 = np.array([basis_analytical.eval(k, 0) for k in range(level+1)], dtype=float)
    shear =  np.zeros(Q.shape[0], dtype=float)
    h = Q[:, 0]
    moments = Q[:, 1:] / np.repeat(h, level+1).reshape((h.shape[0], level+1))

    for i in range(level+1):
        shear += moments[:, i] * basis_0[i]
    return shear

def shear_at_bottom_moments_FD(Q, dz=0.05):
    level = Q.shape[1]-2
    basis_analytical =Legendre_shifted(order=level)
    basis_0 = np.array([basis_analytical.eval(k, dz) for k in range(level+1)], dtype=float)
    shear =  np.zeros(Q.shape[0], dtype=float)
    h = Q[:, 0]
    moments = Q[:, 1:] / np.repeat(h, level+1).reshape((h.shape[0], level+1))

    for i in range(level+1):
        shear += moments[:, i] * basis_0[i]
    shear /= dz

    return shear

def sync_timelines(Q1, T1, Q2, T2, fixed_order = False):
    if (T1.shape[0] > T2.shape[0]) or (fixed_order):
        Tlarge = T1
        Tsmall = T2
        Qlarge = Q1
        Qsmall = Q2
    else:
        Tlarge = T2
        Tsmall = T1
        Qlarge = Q2
        Qsmall = Q1
        
    Qinterp = np.zeros_like(Qsmall)
    if len(Qsmall.shape) == 2:
        for i in range(Qsmall.shape[1]):
            Qinterp[:, i] = np.interp(Tsmall, Tlarge, Qlarge[:, i])
    elif len(Qsmall.shape) == 3:
        for i in range(Qsmall.shape[1]):
            for j in range(Qsmall.shape[2]):
                Qinterp[:, i, j] = np.interp(Tsmall, Tlarge, Qlarge[:, i, j])
    else:
        assert False
    return Tsmall, Qinterp, Qsmall

def compute_error_over_h_and_u(Q1, Q2, T1, T2 , z, dx):
    """
    Q: [time, field, space]
    ERROR: Does not work. I do not have openfoam data at each spatial position
    ATTENTION: I assume Q are primary variables
    """
    T, q1, q2, b_first = sync_timelines(Q1, Q2, T1, T2, swap=False)
    level = q1.shape[1]-1
    basis_analytical =Legendre_shifted(order=level)
    basis = np.zeros((level+1, z.shape[0]), dtype=float)
    # evaluate basis
    for i in range(z.shape[0]):
        basis[:, i] = np.array([basis_analytical.eval(k, z[i]) for k in range(level+1)], dtype=float)
    error = np.zeros_like(T)
    error_h = np.zeros_like(T)
    for it in range(T.shape[0]):
        for ix in range(q1.shape[2]):
            h1 = q1[:, 0, :]
            h2 = q2[:, 0, :]
            u1 = np.zeros_like(z)
            u2 = np.zeros_like(z)
            for k, (m1, m2) in enumerate(zip(q1[it, 1:, ix], q2[it, 1:, ix])):
                u1 += m1 * basis[k,:]
                u2 += m2 * basis[k,:]
        # norm_u = simpson(np.abs(u_of - u_smm)[:20], x=z[:20])
            norm_u = simpson(np.abs(u1 - u2), x=z)
            error[it] += dx * norm_u
            error_h[it] += dx * np.abs(h1-h2)
    return error_h, error

        

def compute_error(m_smm, t_smm, m_of, t_of, z):
    T, M1, M2, b_first = sync_timelines(m_of, t_of, m_smm, t_smm, swap=True)
    if b_first:
        m_of = M1
        m_smm = M2
    else:
        m_of = M2
        m_smm = M1
    level = m_of.shape[1]-1
    basis_analytical =Legendre_shifted(order=level)
    basis = np.zeros((level+1, z.shape[0]), dtype=float)
    for i in range(z.shape[0]):
        basis[:, i] = np.array([basis_analytical.eval(k, z[i]) for k in range(level+1)], dtype=float)
    error = np.zeros_like(T)
    for it in range(T.shape[0]):
        u_of = np.zeros_like(z)
        u_smm = np.zeros_like(z)
        for k, (m1, m2) in enumerate(zip(m_of[it], m_smm[it])):
            u_of += m1 * basis[k,:]
            u_smm += m2 * basis[k,:]
        # norm_u = simpson(np.abs(u_of - u_smm)[:20], x=z[:20])
        norm_u = simpson(np.abs(u_of - u_smm), x=z)
        error[it] = norm_u
    return error

if __name__ == '__main__':
    test_project_to_smm()

    



