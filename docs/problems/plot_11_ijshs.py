# ---
# title: 'IJSHS2024'
# author: IS
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#| code-fold: true
#| code-summary: "Packages and plotting settings"
#| output: false

# %load_ext autoreload
# %autoreload 2

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
seaborn.set_context('talk')
seaborn.color_palette('colorblind')
import pytest
from types import SimpleNamespace
import argparse
import pickle

main_dir = os.getenv("SMS")
from code_11_ijshs import *
from library.model.model import *
from library.pysolver.solver import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing

# ccc = seaborn.color_palette('colorblind')
ccc = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=ccc) 


# path_to_openfoam = 'outputs/nozzle_openfoam_jonas/VTK'
path_to_openfoam = 'outputs/openfoam_nozzle_2d'
directory = os.path.join(main_dir, path_to_openfoam)


def compute_shear_at_position_OF(u, z):
    shear = (u[:, 3] - u[:, 0])/(z[3]-z[0])
    return shear


def extract_data_from_openfoam(pos=[0.3, 0, 0], dt=0.1, stride=10, level=20):
    # pos, h, u, w, iteration = extract_1d_data(directory, pos=pos, stride=stride)
    h ,moments, moments_w, timeline, u, w = project_openfoam_to_smm(directory, pos=pos, stride=stride, output_uw=True, level=level)
    Q = np.zeros((h.shape[0], level+2), dtype=float)
    Q[:, 0] = h
    Q[:, 1:] = (moments.T * h).T
    z = np.linspace(0, 1, 100)
    shear = compute_shear_at_position_OF(u, z)
    return {"pos": np.array(pos), "h": h.copy(), "u":u.copy(), "w": w.copy(), "timeline":timeline.copy(), 'moments': moments.copy(), 'moments_w': moments_w.copy(), 'Q': Q.copy(), 'shear': shear.copy()}

def plot_velocity_profiles_at_position(ax, directory, data, regions=[0, 5, 20, 1000], stride=10, dt=0.01):

    h = data['h']
    time = data['timeline']
    u = data['u']
    w = data['w']

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('h')
    ax[1].set_xlabel('u')
    ax[1].set_ylabel('z')
    ax[2].set_xlabel('w')
    ax[2].set_ylabel('z')

    z = np.linspace(0, 1, 100)

    ax[0].plot(time, h, 'x')
    ax[0].fill_between(time[regions[0]:regions[1]+1], h[regions[0]:regions[1]+1], color=ccc[0], alpha=0.3)
    ax[0].fill_between(time[regions[1]:regions[2]+1], h[regions[1]:regions[2]+1], color=ccc[1], alpha=0.3)
    ax[0].fill_between(time[regions[2]:regions[3]+1], h[regions[2]:regions[3]+1], color=ccc[2], alpha=0.3)
    I = np.linspace(1, u.shape[0], u.shape[0], dtype=int)
    for i in range(u.shape[0]):
        if i not in I[::stride]:
            continue
        if i >= regions[0] and i < regions[1]:
            ax[1].plot(u[i], z, ccc[0])
            ax[2].plot(w[i], z, ccc[0])
        elif i >= regions[1] and i < regions[2]:
            ax[1].plot(u[i], z, ccc[1])
            ax[2].plot(w[i], z, ccc[1])
        elif i >= regions[2] and i <= regions[3]:
            ax[1].plot(u[i], z, ccc[2])
            ax[2].plot(w[i], z, ccc[2])

    return ax


def reconstruct_velocity_profiles_OF(level, u, z):
    basis_analytical =Legendre_shifted(order=level+1)
    basis_readable = [basis_analytical.get(k) for k in range(level+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(level+1)]

    moments = project_to_smm(u, z, basis=basis)
    reconstructions = []
    for k in range(level+1):
        reconst = moments[0] * basis[0](z)
        for i in range(1,k+1):
            reconst += moments[i] * basis[i](z)
        reconstructions.append(reconst)

    errors = []
    rel_errors = []
    error_0 = np.trapz((u - reconstructions[0])**2, z)
    for k in range(level+1):
        error = np.trapz((u - reconstructions[k])**2, z)
        errors.append(error)
        rel_errors.append(error_0/error)
    return reconstructions, errors, rel_errors

def reconstruct_velocity_profile_SMM(moments):
    level = moments.shape[0]-1
    basis_analytical =Legendre_shifted(order=level+1)
    basis_readable = [basis_analytical.get(k) for k in range(level+1)]
    basis = [basis_analytical.get_lambda(k) for k in range(level+1)]
    z = np.linspace(0, 1,100)
    reconstruction = np.zeros_like(z)
    for i in range(0,level+1):
        reconstruction += moments[i] * basis[i](z)
    return reconstruction


def plot_reconstructed_VP(ax, data, time, levels=[0, 1, 2, 4, 6, 8]):
    timeline = data['timeline']
    it = ((timeline - time)**2).argmin()
    u = data['u'][it]
    z = np.linspace(0, 1, 100)

    reconstructions, errors, rel_errors = reconstruct_velocity_profiles_OF(np.max(np.array(levels)), u, z)
    for k in levels:
        ax.plot(reconstructions[k], z, label=f'level {k}')
    ax.plot(u, z, 'b*', linestyle='', label='openfoam')

    ax.set_xlabel('$u$')
    ax.set_ylabel('$z$')

    plt.title('Velocity projection at 3d/2d interface')
    plt.legend()

    print(f'errors: {errors}')
    print(f'relative errors: {rel_errors}')
    return ax, u, z


def setup_SMM(data_inflow, level, dir='output', name='ShallowMoments'):

    mesh = petscMesh.Mesh.create_1d((0.5, 1.0), 100)

    h_inflow = data_inflow['h']
    moments_inflow = data_inflow['moments']
    timeline_inflow = data_inflow['timeline']

    offset = level+1

    data_dict = {}
    data_dict[0] = h_inflow 
    for i in range(level+1):
        data_dict[1+i] = h_inflow * moments_inflow[:, i]

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
        name=name,
        parameters={"g": 9.81, "C": 10.0, "nu": 1.034*10**(-6), "rho": 1, "lamda": 3, "beta": 0.0100},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=.9),
        time_end=5.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=f"outputs/{dir}",
    )

    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": []},
        # settings={"friction": ["newtonian"]},
        # settings={"friction": ["chezy"]},
        # settings={"friction": ["shear"]},
        # settings={"friction": ["shear_crazy"]},
        # settings={"friction": ["shear", "newtonian"]},
        settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["chezy"]},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )

    return model, settings, mesh

def load_and_align_SMM_with_OF(path, pos, experiments):
    pos = 0.75
    data = experiments[str(pos)]
    Q_OF = data['Q']
    timeline_OF = data['timeline']

    x_smm, Q_smm, Qaux_smm, timeline_smm = io.load_timeline_of_fields_from_hdf5(path)

    i_pos = ((x_smm-pos)**2).argmin()
    Q = np.zeros((Q_smm.shape[0], Q_OF.shape[1]))
    Q[:, :Q_smm.shape[1]] = Q_smm[:, :, i_pos]

    T, Q_smm_synced, Q_OF = sync_timelines(Q, timeline_smm, Q_OF, timeline_OF, fixed_order=True)
    shear_smm = shear_at_bottom_moments(Q_smm_synced)
    return {'Q': Q_smm_synced, 'timeline': T, 'shear': shear_smm}




def run_SMM(mesh, model, settings):
    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )

    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

def create_timeline_summary():
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    ax = fig.subplot_mosaic(
        [[0, 1], ["legend", "legend"]],
        # per_subplot_kw={
        #     0: {"box_aspect": 0.6},
        #     1: {"box_aspect": 0.6},
        #     2: {"box_aspect": 1.0},
            # 3: {"box_aspect": 0.6},
        # },
        height_ratios=[1.3, 0.3],
    )
    ax[0].set_title('$h(t, x=0.75)$')
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$h$')
    ax[1].set_title('$u(t, x=0.75)$')
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$u$')
    # ax[2].set_title('$ |u_{SMM}-u_{OF}|_{L2}$')
    # ax[2].set_xlabel('$x$')
    # ax[2].set_ylabel('$|u_{SMM}-u_{OF}|_{L2}$')
    # ax[3].set_title('$|\partial_x u^b_{SMM}-u^b_{OF}|_{L2}$')
    # ax[3].set_xlabel('$x$')
    # ax[3].set_ylabel('$|\partial_x u^b_{SMM}-u^b_{OF}|_{L2}$')

    # ax[2].set_ylim(10**(-3), 0.01)
    handles, labels = ax[0].get_legend_handles_labels()
    ax['legend'].legend(handles, labels, ncol=6, mode="expand", loc='lower left')
    ax["legend"].axis("off")

    # for i in range(3): 
    #     ax[i].ticklabel_format(style='sci', scilimits=(0, 10))

    return fig, ax


def add_timeline_summary(ax, data, label):
    # makers = ['*', '', '', '', '', '', '', '']
    # linestyles = ['b', '', '', '', '', '', '', '']
    Q = data['Q']
    level = Q.shape[1]-2
    timeline = data['timeline']
    h = Q[:, 0]
    a = Q[:, 1:]/np.repeat(h, level+1).reshape((h.shape[0], level+1))
    if label=='openfoam':
        ax[0].plot(timeline, h, 'b*', linestyle='', label=label)
        ax[1].plot(timeline, a[:, 0], 'b*', linestyle='', label=label)
    else:
        ax[0].plot(timeline, h, linestyle='', label=label)
        ax[1].plot(timeline, a[:, 0], label=label)
    # ax[4].plot(timeline, shear, '*', linestyle='', label=label)


def add_timeline_summary_error(ax, data, label):
    ax[2].semilogy(timeline, error, label=label)

if __name__=='__main__':

    ## extract data
    # stride = 1
    # experiments = {}
    experiments['0.3'] = extract_data_from_openfoam(pos=[0.3, 0, 0], stride=stride)
    # experiments['0.4'] = extract_data_from_openfoam(pos=[0.4, 0, 0], stride=stride)
    # experiments['0.5'] = extract_data_from_openfoam(pos=[0.5, 0, 0], stride=stride)
    # experiments['0.75'] = extract_data_from_openfoam(pos=[0.75, 0, 0], stride=stride)
    # experiments['0.9'] = extract_data_from_openfoam(pos=[0.9, 0, 0], stride=stride)

    # with open('experiments.pkl', 'wb')  as f:
    #     pickle.dump(experiments, f)

    with open('experiments.pkl', 'rb')  as f:
        experiments = pickle.load(f)

    print('loaded experiments')


    ### simulation

    # data_inflow = {'h': experiments['0.5']['h'], 'moments': experiments['0.5']['moments'], 'timeline': experiments['0.5']['timeline']}
    # model_0, settings_0, mesh_0 = setup_smm(data_inflow, 0, dir='lvl0', name='chezy10')
    # model_1, settings_1, mesh_1 = setup_smm(data_inflow, 1, dir='lvl1', name='chezy10')
    # model_2, settings_2, mesh_2 = setup_smm(data_inflow, 2, dir='lvl2', name='chezy10')
    # model_4, settings_4, mesh_4 = setup_smm(data_inflow, 4, dir='lvl4', name='chezy10')
    # model_6, settings_6, mesh_6 = setup_smm(data_inflow, 6, dir='lvl6', name='chezy10')
    # model_8, settings_8, mesh_8 = setup_smm(data_inflow, 8, dir='lvl8', name='chezy10')

    # run_smm(mesh_0, model_0, settings_0)
    # run_smm(mesh_1, model_1, settings_1)
    # run_smm(mesh_2, model_2, settings_2)
    # run_smm(mesh_4, model_4, settings_4)
    # run_smm(mesh_6, model_6, settings_6)
    # run_smm(mesh_8, model_8, settings_8)



    # ### align of and smm datag
    simulations = {}
    def load_and_align(pos, simulations):
        # new = load_and_align_smm_with_of(os.path.join('outputs/lvl0', "chezy10.h5" ), pos, experiments)
        lvl0 = load_and_align_smm_with_of(os.path.join('outputs/lvl0', "chezy10.h5" ), pos, experiments)
        lvl1 = load_and_align_smm_with_of(os.path.join('outputs/lvl1', "chezy10.h5" ), pos, experiments)
        lvl2 = load_and_align_smm_with_of(os.path.join('outputs/lvl2', "chezy10.h5" ), pos, experiments)
        lvl4 = load_and_align_smm_with_of(os.path.join('outputs/lvl4', "chezy10.h5" ), pos, experiments)
        lvl6 = load_and_align_smm_with_of(os.path.join('outputs/lvl6', "chezy10.h5" ), pos, experiments)
        lvl8 = load_and_align_smm_with_of(os.path.join('outputs/lvl8', "chezy10.h5" ), pos, experiments)
        # lvl0 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl0_chezy10.h5" ), pos, experiments)
        # lvl1 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl1_chezy10.h5" ), pos, experiments)
        # lvl2 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl2_chezy10.h5" ), pos, experiments)
        # lvl4 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl4_chezy10.h5" ), pos, experiments)
        # lvl6 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl6_chezy30.h5" ), pos, experiments)
        # lvl8 = load_and_align_smm_with_of(os.path.join('outputs/save_ijshs', "lvl8_chezy10.h5" ), pos, experiments)
        simulations[str(pos)] = {'0': lvl0, '1': lvl1, '2': lvl2, '4': lvl4, '6': lvl6, '8': lvl8}
        return simulations
    
    # simulations = load_and_align(0.75, simulations)
    # simulations = load_and_align(0.5, simulations)
    # simulations = load_and_align(0.9, simulations)


    # with open('simulations.pkl', 'wb')  as f:
    #     pickle.dump(simulations, f)


    with open('simulations.pkl', 'rb')  as f:
        simulations = pickle.load(f)

    print('loaded simulations')


    # ### velocity profiles
    # fig, ax = plt.subplots(1, 3, constrained_layout=true, figsize=(12,7.5))
    # fig.suptitle(f'velocity profiles at x = {0.3} m')
    # ax = plot_velocity_profiles_at_position(ax, directory, experiments['0.3'], regions=[0, 5, 20, 1000], stride=10) 
    # fig.savefig(os.path.join(main_dir, 'docs/problems/images/vp03.png'))

    # fig, ax = plt.subplots(1, 3, constrained_layout=true, figsize=(12,7.5))
    # fig.suptitle(f'velocity profiles at x = {0.4} m')
    # ax = plot_velocity_profiles_at_position(ax, directory, experiments['0.4'], regions=[0, 2, 12, 1000], stride=10) 
    # fig.savefig(os.path.join(main_dir, 'docs/problems/images/vp04.png'))

    # fig, ax = plt.subplots(1, 3, constrained_layout=true, figsize=(12,7.5))
    # fig.suptitle(f'velocity profiles at x = {0.5} m')
    # ax = plot_velocity_profiles_at_position(ax, directory, experiments['0.5'], regions=[0, 6, 15, 1000], stride=10) 
    # fig.savefig(os.path.join(main_dir, 'docs/problems/images/vp05.png'))


    # fig, ax = plt.subplots(1, 3, constrained_layout=true, figsize=(12,7.5))
    # fig.suptitle(f'velocity profiles at x = {0.75} m')
    # ax = plot_velocity_profiles_at_position(ax, directory, experiments['0.75'], regions=[0, 33, 45, 1000], stride=10) 
    # fig.savefig(os.path.join(main_dir, 'docs/problems/images/vp75.png'))

    # fig, ax = plt.subplots(1, 3, constrained_layout=true, figsize=(12,7.5))
    # fig.suptitle(f'velocity profiles at x = {0.9} m')
    # ax = plot_velocity_profiles_at_position(ax, directory, experiments['0.9'], regions=[0, 33, 45, 1000], stride=10) 
    # fig.savefig(os.path.join(main_dir, 'docs/problems/images/vp90.png'))

    # ### reconstructions

    fig, ax = plt.subplots()
    time = 3.
    ax = plot_reconstructed_vp(ax, experiments['0.3'], time, levels=[0, 1, 2, 4, 8])
    fig.savefig(os.path.join(main_dir, 'docs/problems/images/projectioninflow03.png'))

    fig, ax = plt.subplots()
    ax = plot_reconstructed_vp(ax, experiments['0.4'], time,  levels=[0, 1, 2, 4, 8])
    fig.savefig(os.path.join(main_dir, 'docs/problems/images/projectioninflow04.png'))

    fig, ax = plt.subplots()
    ax = plot_reconstructed_vp(ax, experiments['0.5'], time, levels=[0, 1, 2, 4, 8])
    fig.savefig(os.path.join(main_dir, 'docs/problems/images/projectioninflow05.png'))



    # ### compute errors

    # ### timeline summary
    fig, ax = create_timeline_summary()
    add_timeline_summary(ax, experiments['0.75'], 'openfoam')
    add_timeline_summary(ax, simulations['0.75']['8'], 'lvl8')


    # ### shear plot
    fig, ax = plt.subplots()
    ax.plot(experiments['0.75']['timeline'], experiments['0.75']['shear'], '*', label='openfoam')
    ax.plot(simulations['0.75']['8']['timeline'], simulations['0.75']['8']['shear'], label='lvl8')
    plt.suptitle(f"shear at x = {0.75}")
    plt.legend()
    fig.savefig(os.path.join(main_dir, 'docs/problems/images/shear_075.png'))



