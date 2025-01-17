import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from copy import deepcopy
from matplotlib.colors import Normalize
import matplotlib
from matplotlib.ticker import ScalarFormatter
import seaborn 
import pickle
seaborn.set_context('talk')
from scipy.integrate import quad, simpson


import pytest
from types import SimpleNamespace
import argparse


from library.model.model import *
# from library.pysolver.solver import *
# import library.model.initial_conditions as IC
# import library.model.boundary_conditions as BC
# from library.pysolver.ode import RK1
import library.misc.io as io
# from library.pysolver.reconstruction import GradientMesh
# import library.mesh.mesh as petscMesh
# import library.postprocessing.postprocessing as postprocessing

from code_11_ijshs import *
from library.model.model import *






def extract_from_openfoam(experiments, x = 0.3, stride=10):
    pos, h, u, w, iteration = extract_1d_data_fast(directory, pos=[x, 0, 0], stride=stride)
    dt = 0.01
    time = dt * iteration
    experiments[str(x)] = {"x": pos.copy(), "h": h.copy(), "u":u.copy(), "w": w.copy(), "time":time}
    
def extract_from_precice(path, interface, stride=10):
    directory_interface = os.path.join(main_dir, path)
    pos, u, iteration = read_precice_vtus(directory_interface, stride=stride)
    dt = 0.005
    time = dt * iteration
    interface= {"x": pos.copy(), "u":u.copy(), "time":time}
    return interface
    
    
def plot_reconstruct_moments_from_of(u, z, lGevel=8):
    basis_analytical =Legendre_shifted(order=level+1)
    basis_readable = [basis_analytical.get(k) for k in range(level+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(level+1)]

    u_test = u
    moments = project_to_smm(u_test, z, basis=basis)
    reconstructions = []
    for k in range(level+1):
        reconst = moments[0] * basis[0](z)
        for i in range(1,k+1):
            reconst += moments[i] * basis[i](z)
        reconstructions.append(reconst)
    fig, ax = plt.subplots()
    plt.plot(u, z, '*', label='openfoam')
    # for k in range(level+1):
    #     ax.plot(reconstructions[k], z, label=f'level {k}')

    for k in [0, 1, 2, 8]:
        ax.plot(reconstructions[k], z, label=f'level {k}')

    errors = []
    rel_errors = []
    error_0 = np.trapz((u_test - reconstructions[0])**2, z)
    for k in range(level+1):
        error = np.trapz((u_test - reconstructions[k])**2, z)
        errors.append(error)
        rel_errors.append(error_0/error)
    
    
    plt.title('Velocity projection at 3d/2d interface')
    plt.legend()
    print(f'errors: {errors}')
    print(f'relative errors: {rel_errors}')
    return fig, ax, errors, rel_errors

def project_openfoam_to_smm(experiments, x=0.5, level=10, output_uw=False):
    # pos, h, u, w, iteration = extract_1d_data(directory, pos=pos, stride=stride)
    pos = experiments[str(x)]['x']
    h = experiments[str(x)]['h']
    u = experiments[str(x)]['u']
    w = experiments[str(x)]['w']
    iteration_times = experiments[str(x)]['time']    
    basis_analytical = Legendre_shifted(order=level+1)
    basis = [basis_analytical.get_lambda(k) for k in range(level+1)]

    moments = []
    moments_w = []
    z = np.linspace(0, 1, u.shape[1])
    for u_n in u:
        moments.append(project_to_smm(u_n, z, basis=basis))
    for w_n in w:
        moments_w.append(project_to_smm(w_n, z, basis=basis))
    moments = np.array(moments)
    for l in range(moments.shape[1]):
        moments[:, l] *= h
    moments_w = np.array(moments_w)
    for l in range(moments_w.shape[1]):
        moments_w[:, l] *= h
    if output_uw:
        return h, moments, moments_w, iteration_times, u, w
    return h, moments, moments_w, iteration_times

def project_openfoam(openfoam_projected, experiments,  x=0.5):  
    h_of ,moments_of, moments_w_of, timeline_of, u_of, w_of = project_openfoam_to_smm(experiments, x=x, output_uw=True)
    
    data = {'h': h_of, 'moments': moments_of, 'moments_w': moments_w_of, 'time': timeline_of, 'u': u_of, 'w': w_of}
    openfoam_projected[str(x)] = data
    return openfoam_projected
    
        
def shear_at_bottom_moments(moments):
    level = moments.shape[1]-1
    basis_analytical =Legendre_shifted(order=level)
    basis_analytical.get_diff_basis()
    basis_0 = np.array([basis_analytical.eval(k, 0) for k in range(level+1)], dtype=float)

    shear =  np.zeros(moments.shape[0], dtype=float)
    for i in range(level+1):
        shear += moments[:, i] * basis_0[i]

    return shear



# def simulate(level, inflow_bc, friction=["newtonian_boundary_layer", "newtonian"], param={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6), "rho": 1, "lamda": 3, "beta": 0.0100}, N = 100):
#     mesh = petscMesh.Mesh.create_1d((0.5, 3.0), N)
#     level = level
#     offset = level+1

#     h_inflow = inflow_bc['h']
#     moments_inflow = inflow_bc['moments']
#     timeline_inflow = inflow_bc['time']
#     data_dict = {}
#     data_dict[0] = h_inflow 
#     for i in range(level+1):
#         data_dict[1+i] = moments_inflow[:, i]

#     bcs = BC.BoundaryConditions(
#         [
#             BC.FromData(physical_tag='left', prescribe_fields=data_dict, timeline=timeline_inflow),
#             BC.Wall(physical_tag="right", momentum_field_indices=[[i+1] for i in range(level+1)], wall_slip=0.),
#         ]
#     )

#     ic = IC.Constant(
#         constants=lambda n_fields: np.array(
#             [0.02, 0.0] + [0.0 for i in range(n_fields - 2)]
#         )
#     )


#     settings = Settings(
#         name="ShallowMoments",
#         parameters=param,
#         reconstruction=recon.constant,
#         num_flux=flux.LLF(),
#         nc_flux=nonconservative_flux.segmentpath(1),
#         compute_dt=timestepping.adaptive(CFL=.3),
#         time_end=5.,
#         output_snapshots=100,
#         output_clean_dir=True,
#         output_dir=f"outputs/ijrewhs_{level}",
#     )


#     model = ShallowMoments(
#         dimension=1,
#         fields=2 + level,
#         aux_fields=0,
#         parameters=settings.parameters,
#         boundary_conditions=bcs,
#         initial_conditions=ic,
#         settings={"friction": friction},
#         basis=Basis(basis=Legendre_shifted(order=level)),
#     )
#     jax_fvm_unsteady_semidiscrete(
#         mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
#     )
#     io.generate_vtk(os.path.join(settings.output_dir, 'ShallowMoments.h5'))

def load_simulation_data(simulation_data, level):
    x_smm, Q_smm, Qaux_smm, timeline_smm = io.load_timeline_of_fields_from_hdf5(os.path.join(f'outputs/ijrewhs_cpl_{level}', "ShallowMoments.h5" ))
    simulation_data[str(level)] = {'x': x_smm, 'Q': Q_smm, 'Qaux': Qaux_smm, 'time': timeline_smm}
    return simulation_data

def get_x(x, X):
    i = np.argmin((X-x)**2)
    return i

def get_t(t, T):
    i = np.argmin((T-t)**2)
    return i
    

def plot_h(ax, x, experiments, openfoam_projected, simulation_data):
    h_of = experiments[str(x)]['h']
    time_of = experiments[str(x)]['time']
    ax.plot(time_of, h_of, '*', label=f'openfoam')
    for level in simulation_data.keys():
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:, 0, x_smm]
        time_smm = simulation_data[level]['time']
        ax.plot(time_smm, h_smm, '--', label=f'level {level}')
    
def plot_u_mean(ax, x, experiments, openfoam_projected, simulation_data):
    h_of = openfoam_projected[str(x)]['h']
    u_of = openfoam_projected[str(x)]['moments'][:, 0] / h_of
    time_of = openfoam_projected[str(x)]['time']
    
    ax.plot(time_of, u_of, '*', label=f'openfoam')
    for level in simulation_data.keys():
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:,0, x_smm]
        u_smm = simulation_data[level]['Q'][:,1, x_smm] / h_smm
        time_smm = simulation_data[level]['time']
        ax.plot(time_smm, u_smm, '--', label=f'level {level}')
    
def plot_moment(ax, x, field, experiments, openfoam_projected, simulation_data):
    h_of = openfoam_projected[str(x)]['h']
    u_of = openfoam_projected[str(x)]['moments'][:, field] / h_of
    time_of = openfoam_projected[str(x)]['time']
    
    ax.plot(time_of, u_of, '*', label=f'openfoam')
    for level in simulation_data.keys():
        if int(level) < field:
            continue
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:,0, x_smm]
        u_smm = simulation_data[level]['Q'][:,field+1, x_smm] / h_smm
        time_smm = simulation_data[level]['time']
        ax.plot(time_smm, u_smm, '--', label=f'level {level}')
    
def plot_bottom_velocity(ax, x, experiments, openfoam_projected, simulation_data):
    for i in range(2,5):
        u_of = experiments[str(x)]['u'][:, i]
        time_of = openfoam_projected[str(x)]['time']
        ax.plot(time_of, u_of, '*', label=f'openfoam_{i}')
    for level in simulation_data.keys():
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:,0, x_smm]
        u_smm = np.sum(simulation_data[level]['Q'][:,1:, x_smm], axis=1) / h_smm
        time_smm = simulation_data[level]['time']
        ax.plot(time_smm, u_smm, '--', label=f'level {level}')

def error_bottom_velocity(x, experiments, openfoam_projected, simulation_data):
    bottom_values = [0]
    errors = np.zeros((len(simulation_data.keys()), len(bottom_values)))
    U_of = []
    T_of = []
    def compute_error(u_of, t_of, u_smm, t_smm):
        u_of_sampled = np.interp(t_smm, t_of, u_of)
        dt = t_smm[1:] - t_smm[:-1]
        return np.sqrt(np.sum(dt * (u_of_sampled[1:] - u_smm[1:])**2))
    for i in bottom_values:
        u_of = experiments[str(x)]['u'][:, i]
        time_of = openfoam_projected[str(x)]['time']
        # ax.plot(time_of, u_of, '*', label=f'openfoam_{i}')
        U_of.append(u_of)
        T_of.append(time_of)
    for il, level in enumerate(simulation_data.keys()):
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:,0, x_smm]
        u_smm = np.sum(simulation_data[level]['Q'][:,1:, x_smm], axis=1) / h_smm
        time_smm = simulation_data[level]['time']
        # ax.plot(time_smm, u_smm, '--', label=f'level {level}')
        for i_of, (u_of, t_of) in enumerate(zip(U_of, T_of)):
            errors[il, i_of] = compute_error(u_of, t_of, u_smm, time_smm)
    return errors



def plot_velocity_profile(ax, x, t, experiments, openfoam_projected, simulation_data):
    lvl = np.array(list(simulation_data.keys()), dtype=int).max()
    basis_analytical =Legendre_shifted(order=lvl+1)
    basis_readable = [basis_analytical.get(k) for k in range(lvl+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(lvl+1)]
    
    t_of = get_t(t, experiments[str(x)]['time'])
    u_of = experiments[str(x)]['u'][t_of, :]
    # time_of = openfoam_projected[str(x)]['time']
    ax.plot(u_of, z, '*', label=f'openfoam')
    
    for level in simulation_data.keys():
        x_smm = get_x(x, simulation_data[level]['x'])
        t_smm = get_t(t, simulation_data[level]['time'])
        h_smm = simulation_data[level]['Q'][t_smm,0, x_smm]
        moments = simulation_data[level]['Q'][t_smm,1:, x_smm] / h_smm
        u_smm = moments[0] * basis[0](z)
        for i in range(1,int(level)+1):
            u_smm += moments[i] * basis[i](z)
        ax.plot(u_smm.copy(), z, '--', label=f'level {level}')
        
def plot_velocity_profile_projected(ax, x, t, experiments, openfoam_projected, simulation_data):
    lvl = 10
    basis_analytical =Legendre_shifted(order=lvl+1)
    basis_readable = [basis_analytical.get(k) for k in range(lvl+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(lvl+1)]
    
    t_of = get_t(t, experiments[str(x)]['time'])
    u_of = experiments[str(x)]['u'][t_of, :]
    # time_of = openfoam_projected[str(x)]['time']
    
    ax.plot(u_of, z, '*', label=f'OF')
    
    for level in list(range(6,lvl)):
        # x_smm = get_x(x, openfoam_projected['x'])
        print(openfoam_projected.keys())
        t_smm = get_t(t, openfoam_projected[str(x)]['time'])
        h_smm = openfoam_projected[str(x)]['h'][t_smm]
        moments = openfoam_projected[str(x)]['moments'][t_smm,:] / h_smm
        u_smm = moments[0] * basis[0](z)
        for i in range(1,int(level)+1):
            u_smm += moments[i] * basis[i](z)
        ax.plot(u_smm.copy(), z, '--', label=f'level {level}')
        
def interpolate_noslip(u, z, z1=0.0005):
    n = u.shape[0]
    ur = np.zeros((n+1))
    zr = np.zeros((n+1))
    zr[1:] = np.linspace(z1, 1, n)
    ur[1:] = u
    return zr, ur

        
def plot_velocity_profile_interface(ax, t, interface, simulation_data):
    lvl = 10
    basis_analytical =Legendre_shifted(order=lvl+1)
    basis_readable = [basis_analytical.get(k) for k in range(lvl+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(lvl+1)]
    
    t_if = get_t(t, interface['time'])
    u_if = interface['u'][t_if, :]
    
    # zp = np.linspace(0, 1, 101)
    # dz = zp[1]-zp[0]
    # up_of = np.zeros_like(zp)
    # up_of[1:] = u_if
    
    moments = project_velocity_profile(u_if, z, 10)
    
    # z_of , u_of = interpolate_noslip(u_if, z)
    u_of = u_if
    z_of = z
        
    ax.plot(u_of, z_of, '*', label=f'openfoam')
    
    levels = [0, 2, 4, 8]
    for level in levels:
        u_smm = moments[0] * basis[0](z)
        for i in range(1,int(level)+1):
            u_smm += moments[i] * basis[i](z)
        # z_smm , u_smm = interpolate_noslip(u_smm, z)
        # up = np.zeros_like(zp)
        # up[1:] = u_smm
        ax.plot(u_smm.copy(), z, '--', label=f'level {level}')
        
def error_velocity_profile_interface(t, interface, simulation_data):
    lvl = 10
    error_U = np.zeros(lvl)
    error_UU = np.zeros(lvl)
    basis_analytical =Legendre_shifted(order=lvl+1)
    basis_readable = [basis_analytical.get(k) for k in range(lvl+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(lvl+1)]
    
    t_if = get_t(t, interface['time'])
    u_if = interface['u'][t_if, :]

    # z_of , u_of = interpolate_noslip(u_if, z)
    u_of = u_if
    z_of = z

    moments = project_velocity_profile(u_if, z, 10)
        
    dz = z_of[1]-z_of[0]

    levels = list(range(lvl))
    for il, level in enumerate(levels):
        u_smm = moments[0] * basis[0](z)
        for i in range(1,int(level)+1):
            u_smm += moments[i] * basis[i](z)
        # z_smm , u_smm = interpolate_noslip(u_smm, z)

        # error_U[il] = np.sum(dz * np.abs(u_of[1:] - u_smm[1:]))
        error_U[il] = np.sqrt(simpson((u_of - u_smm)**2, x=z))
        error_UU[il] = np.sqrt(simpson((u_of*u_of - u_smm*u_smm)**2, x=z))

        # error_UU[il] = np.sum(dz * np.abs(u_of*u_of - u_smm*u_smm))

    return error_U, error_UU
        # ax.plot(up.copy(), zp, '--', label=f'level {level}')
        
        
def compute_error_at_x_t(error, x, t, experiments, openfoam_projected, simulation_data):
    lvl = np.array(list(simulation_data.keys()), dtype=int).max()
    basis_analytical =Legendre_shifted(order=lvl+1)
    basis_readable = [basis_analytical.get(k) for k in range(lvl+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(lvl+1)]
    
    t_of = get_t(t, experiments[str(x)]['time'])
    u_of = experiments[str(x)]['u'][t_of, :]
    dz = z[1]-z[0]
    time_of = openfoam_projected[str(x)]['time']    
    for il, level in enumerate(simulation_data.keys()):
        x_smm = get_x(x, simulation_data[level]['x'])
        t_smm = get_t(t, simulation_data[level]['time'])
        h_smm = simulation_data[level]['Q'][t_smm,0, x_smm]
        moments = simulation_data[level]['Q'][t_smm,1:, x_smm] / h_smm
        u_smm = moments[0] * basis[0](z)
        for i in range(1,int(level)+1):
            u_smm += moments[i] * basis[i](z)
        
        error[il] = np.sqrt(np.sum(dz * (u_of - u_smm)**2))
        z_of = np.linspace(0, 1, u_of.shape[0])
        z_smm = np.linspace(0, 1, u_smm.shape[0])
        # fig, ax = plt.subplots()
        # ax.plot(u_of, z_of)
        # ax.plot(u_smm, z_smm)
        # plt.show()
        
        

        
def get_experiments(compute_exp, stride):
    if compute_exp:
        experiments={}
        for x in samples_experiments:
            print(x)
            extract_from_openfoam(experiments, x=x, stride=stride)
        with open(f'of_experiments_{folder}.pkl', 'wb') as f:
            pickle.dump(experiments, f)
        print('Extract openfoam data')

    with open(f'of_experiments_{folder}.pkl', 'rb') as f:
        experiments = pickle.load(f)
    return experiments

def get_interface(path, compute_exp, stride):
    if compute_interface:
        interface={}
        interface = extract_from_precice(path, interface, stride=stride)
        with open(f'interface_{folder}.pkl', 'wb') as f:
            pickle.dump(interface, f)
        print('Extract interface data')

    with open(f'interface_{folder}.pkl', 'rb') as f:
        interface = pickle.load(f)
    return interface
    
    
def get_of_projected(compute_proj):
    openfoam_projected = {}

    if compute_proj:
        openfoam_projected = {}
        for x in samples_experiments:
            print(x)
            openfoam_projected = project_openfoam(openfoam_projected, experiments, x=x)
        with open(f'openfoam_projected_{folder}.pkl', 'wb') as f:
            pickle.dump(openfoam_projected, f)
        print('Project OF to moments')
    
    with open(f'openfoam_projected_{folder}.pkl', 'rb') as f:
        openfoam_projected = pickle.load(f)
    return openfoam_projected


def get_simulation(compute_sim, stride):
    if compute_sim:
        simulation_data = {}
        for level in sim_levels:
            # print(level)
            # simulate(level=level, inflow_bc = openfoam_projected['0.5'], friction=['newtonian', 'newtonian_boundary_layer'], param={"g": 9.81, "nu": 1.034*10**(-6), "rho": 1, "eta": 0., "eta_bulk": 0.})
            simulation_data = load_simulation_data(simulation_data, level)
        with open(f'simulation_data_{folder}.pkl', 'wb') as f:
            pickle.dump(simulation_data, f)
        print('SIMULATION COMPLETE')
    with open(f'simulation_data_{folder}.pkl', 'rb') as f:
        simulation_data = pickle.load(f)

    return simulation_data

if __name__ == '__main__':
    ## Folder
    main_dir = os.getenv("SMS")
    folder_base = 'outputs'
    # folder = 'openfoam_nozzle_2d'
    # folder = 'VTK_laminar'
    # folder = 'VTK_turbulent'
    # folder = 'VTK_bottom'
    folder = 'exp_reference'
    stride=50
    compute_exp = False
    compute_proj = False
    compute_sim = True
    compute_interface = False
    #offset_extraction = 5
    samples_experiments = [0.5, 1., 1.5, 2., 2.5]
    VP_time = [2., 3., 4., 5.]
    # VP_time = [0.5, 1.5, 2.5, 3.5, 4.5]

    VP_pos = samples_experiments
    sim_levels = [4]
    

    directory = os.path.join(main_dir, os.path.join(folder_base, folder))
    z = np.linspace(0, 1, 100)

    
    experiments = get_experiments(compute_exp, stride)
    openfoam_projected = get_of_projected(compute_proj)
    print('LOAD OpenFOAM complete')
    simulation_data = get_simulation(compute_sim, stride)
    print('LOAD SMM complete')
    interface_data = get_interface('export2', compute_interface, stride)
    print('LOAD interface complete')
    
    # timeline_exp = experiments['0.5']['time']
    # timeline_dat = simulation_data['4']['time']
    # print(timeline_exp)
    # print(timeline_dat)
    # print(np.intersect1d(timeline_exp, timeline_dat))

    #--------------------------------------------------------------------------------------------------------
    #----------------------------------------------Inflow----------------------------------------------------
    #--------------------------------------------------------------------------------------------------------
    # fig, ax, errors, rel_errors = plot_reconstruct_moments_from_of(experiments['0.5']['u'][-1], z, level=8)
    # plt.show()
    # fig.savefig('images/ProjectionInflow.png')
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Bottom velocities----------------------------------------------
    #--------------------------------------------------------------------------------------------------------
    # fig, ax = plt.subplots(len(VP_pos), 3)
    # for i, x in enumerate(VP_pos):
    #     plot_h(ax[i,0], x, experiments, openfoam_projected, simulation_data)
    #     plot_u_mean(ax[i,1], x, experiments, openfoam_projected, simulation_data)
    #     plot_bottom_velocity(ax[i,2], x, experiments, openfoam_projected, simulation_data)
    # plt.show()
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Errors Bottom velocities---------------------------------------
    #--------------------------------------------------------------------------------------------------------
    # for i, x in enumerate(VP_pos):
    
    # errors = error_bottom_velocity(0.5, experiments, openfoam_projected, simulation_data)
    # print(errors)
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Velocity profiles----------------------------------------------
    #--------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots(len(VP_time),len(VP_pos))
    for it, t in enumerate(VP_time):
        for ix, x in enumerate(VP_pos):
            ax[it, ix].set_xlim(0., 0.3)
            plot_velocity_profile(ax[it, ix], x, t, experiments, openfoam_projected, simulation_data)
    plt.show()
    
    # fig, ax = plt.subplots()
    # plot_velocity_profile(ax, 1., 2., experiments, openfoam_projected, simulation_data)
    # plt.show()
    
    #--------------------------------------------------------------------------------------------------------
    #---------------------------------STEADY STATE Velocity profiles-----------------------------------------
    #--------------------------------------------------------------------------------------------------------
    
    # fig, ax = plt.subplots(len(VP_time))
    # x = 0.5
    # for it, t in enumerate(VP_time):
    #     ax[it].set_xlim(0., 0.3)
    #     plot_velocity_profile_projected(ax[it], x, t, experiments, openfoam_projected, simulation_data)
    # plt.show()
    
    # x = 0.5
    # for it, t_ in enumerate(VP_time):
    #     t = get_t(t_, openfoam_projected[str(x)]['time'])
    #     h = openfoam_projected[str(x)]['h'][t]
    #     moments = openfoam_projected[str(x)]['moments'][t, :] / h
    #     print(f'{t_}: {moments}')


    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Errors in velocity profiles------------------------------------
    #--------------------------------------------------------------------------------------------------------
    # levels = simulation_data.keys()
    # error = np.zeros((len(levels), len(VP_time),len(VP_pos)), dtype=float)
    # print(VP_pos)
    # print(VP_time)
    # for it, t in enumerate(VP_time):
    #     for ix, x in enumerate(VP_pos):
    #        compute_error_at_x_t(error[:, it, ix], x, t, experiments, openfoam_projected, simulation_data)
    # for il, lvl in enumerate(levels):
    #     print(f'Level: {lvl}')
    #     print(error[il, :, 0])
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Velocity at interface----------------------------------------------
    #--------------------------------------------------------------------------------------------------------

    # fig, ax = plt.subplots(len(VP_time))
    # for it, t in enumerate(VP_time):
    #     ax[it].set_xlim(0., 0.3)
    #     plot_velocity_profile_interface(ax[it], t, interface_data , simulation_data)
    #     error_U, error_UU = error_velocity_profile_interface(t, interface_data, simulation_data)
    #     print(t)
    #     print(error_U)
    #     print(error_U[0]/error_U)
    #     print(error_UU)
    #     print(error_UU[0]/error_UU)
    #     print('=========================')
    # plt.show()

# TODO
# compare interface and reference VP at 0.5 for OF
# Find out why the projection does not to look properly? Before, I assumed that the pressue was 
# bad -> inflow that distored the VP in the coupled simulation -> bad VP at the interface BUT
# a correct projection. 
# Now, the VP still looks bad (as if I miss a coeff. But I already checked that). 
# disable source terms ?
# plot coupling VP as well, maybe they match that and the issue is on the coupling?