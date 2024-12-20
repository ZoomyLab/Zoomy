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

import pytest
from types import SimpleNamespace
import argparse


from library.model.model import *
from library.pysolver.solver import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing

from code_11_ijshs import *
from library.model.model import *



## Folder
main_dir = os.getenv("SMS")
folder = 'outputs/openfoam_nozzle_2d'
directory = os.path.join(main_dir, folder)
z = np.linspace(0, 1, 100)


def extract_from_openfoam(experiments, x = 0.3, stride=10):
    pos, h, u, w, iteration = extract_1d_data(directory, pos=[x, 0, 0], stride=stride)
    dt = 0.01
    time = dt * iteration
    experiments[str(x)] = {"x": pos.copy(), "h": h.copy(), "u":u.copy(), "w": w.copy(), "time":time}
    
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



def simulate(level, inflow_bc, friction=["newtonian_boundary_layer", "newtonian"], param={"g": 9.81, "C": 30.0, "nu": 1.034*10**(-6), "rho": 1, "lamda": 3, "beta": 0.0100}, N = 100):
    mesh = petscMesh.Mesh.create_1d((0.5, 3.0), N)
    level = level
    offset = level+1

    h_inflow = inflow_bc['h']
    moments_inflow = inflow_bc['moments']
    timeline_inflow = inflow_bc['time']
    data_dict = {}
    data_dict[0] = h_inflow 
    for i in range(level+1):
        data_dict[1+i] = moments_inflow[:, i]

    bcs = BC.BoundaryConditions(
        [
            BC.FromData(physical_tag='left', prescribe_fields=data_dict, timeline=timeline_inflow),
            BC.Wall(physical_tag="right", momentum_field_indices=[[i+1] for i in range(level+1)], wall_slip=0.),
        ]
    )

    ic = IC.Constant(
        constants=lambda n_fields: np.array(
            [0.02, 0.0] + [0.0 for i in range(n_fields - 2)]
        )
    )


    settings = Settings(
        name="ShallowMoments",
        parameters=param,
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=.3),
        time_end=5.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=f"outputs/ijrewhs_{level}",
    )


    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": friction},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, 'ShallowMoments.h5'))

def load_simulation_data(simulation_data, level):
    x_smm, Q_smm, Qaux_smm, timeline_smm = io.load_timeline_of_fields_from_hdf5(os.path.join(f'outputs/ijrewhs_{level}', "ShallowMoments.h5" ))
    simulation_data[str(level)] = {'x': x_smm, 'Q': Q_smm, 'Qaux': Qaux_smm, 'time': timeline_smm}
    return simulation_data

def get_x(x, X):
    i = np.argmin((X-x)**2)
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
    for i in range(3):
        u_of = experiments[str(x)]['u'][:, i]
        time_of = openfoam_projected[str(x)]['time']
        ax.plot(time_of, u_of, '*', label=f'openfoam_{i}')
    for level in simulation_data.keys():
        x_smm = get_x(x, simulation_data[level]['x'])
        h_smm = simulation_data[level]['Q'][:,0, x_smm]
        u_smm = np.sum(simulation_data[level]['Q'][:,1:, x_smm], axis=1) / h_smm
        time_smm = simulation_data[level]['time']
        ax.plot(time_smm, u_smm, '--', label=f'level {level}')
        
def get_experiments(compute_exp, stride):
    if compute_exp:
        experiments={}
        for x in [0.3, 0.5, 0.7, 0.95, 1.3, 2, 2.5, 2.8]:
            print(x)
            extract_from_openfoam(experiments, x=x, stride=stride)
        with open('of_experiments.pkl', 'wb') as f:
            pickle.dump(experiments, f)
        print('Extract openfoam data')

    with open('of_experiments.pkl', 'rb') as f:
        experiments = pickle.load(f)
    return experiments
    
    
def get_of_projected(compute_proj):
    if compute_proj:
        openfoam_projected = {}
        for x in [0.3, 0.5, 0.7, 0.95, 1.3, 2, 2.5, 2.8]:
            print(x)
            openfoam_projected = project_openfoam(openfoam_projected, experiments, x=x)
        with open('openfoam_projected.pkl', 'wb') as f:
            pickle.dump(openfoam_projected, f)
        print('Project OF to moments')
    
    with open('openfoam_projected.pkl', 'rb') as f:
        openfoam_projected = pickle.load(f)
    return openfoam_projected


def get_simulation(compute_sim, stride):
    if compute_sim:
        simulation_data = {}
        for level in [0, 2,6]:
            print(level)
            simulate(level=level, inflow_bc = openfoam_projected['0.5'], friction=['newtonian', 'newtonian_boundary_layer'], param={"g": 9.81, "nu": 1.034*10**(-6), "rho": 1, "eta": 2.5})
            simulation_data = load_simulation_data(simulation_data, level)
        with open('simulation_data.pkl', 'wb') as f:
            pickle.dump(simulation_data, f)
        print('SIMULATION COMPLETE')
    with open('simulation_data.pkl', 'rb') as f:
        simulation_data = pickle.load(f)
    return simulation_data

if __name__ == '__main__':
    stride=1
    compute_exp = False
    compute_proj = False
    compute_sim = False
    
    experiments = get_experiments(compute_exp, stride)
    openfoam_projected = get_of_projected(compute_proj)
    print('LOAD OpenFOAM complete')
    simulation_data = get_simulation(compute_sim, stride)
    print('LOAD SMM complete')

    #--------------------------------------------------------------------------------------------------------
    #----------------------------------------------Inflow----------------------------------------------------
    #--------------------------------------------------------------------------------------------------------
    # fig, ax, errors, rel_errors = plot_reconstruct_moments_from_of(experiments['0.5']['u'][-1], z, level=8)
    # plt.show()
    # fig.savefig('images/ProjectionInflow.png')
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Bottom velocities----------------------------------------------
    #--------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(4,3)
    for i, x in enumerate([0.5, 0.7, 1.3, 2]):
        plot_h(ax[i,0], x, experiments, openfoam_projected, simulation_data)
        plot_u_mean(ax[i,1], x, experiments, openfoam_projected, simulation_data)
        plot_bottom_velocity(ax[i,2], x, experiments, openfoam_projected, simulation_data)
    plt.show()
    
    #--------------------------------------------------------------------------------------------------------
    #-----------------------------------------Velocity profiles----------------------------------------------
    #--------------------------------------------------------------------------------------------------------
