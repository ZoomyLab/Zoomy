import numpy as np
from types import SimpleNamespace
import pyprog
from attr import define, field
from typing import Callable, Optional, Type
from copy import deepcopy
from time import time as gettime
import os

import logging
logging.basicConfig(level=logging.DEBUG)

from library.model.model import *
from library.model.models.base import register_parameter_defaults
from library.model.models.shallow_moments import reconstruct_uvw
import library.pysolver.reconstruction as recon
import library.pysolver.flux as flux
import library.pysolver.nonconservative_flux as nonconservative_flux
import library.pysolver.timestepping as timestepping
import library.misc.io as io
import library.mesh.fvm_mesh as fvm_mesh
from library.pysolver.ode import *


@define(slots=True, frozen=False, kw_only=True)
class Settings():
    name : str = 'Simulation'
    momentum_eqns: list[int] = [0]
    parameters : dict = {}
    reconstruction : Callable = recon.constant
    reconstruction_edge: Callable = recon.constant_edge
    num_flux : Callable = flux.LF()
    nc_flux: Callable = nonconservative_flux.segmentpath()
    compute_dt : Callable = timestepping.constant(dt=0.1)
    time_end : float = 1.0
    truncate_last_time_step: bool = True
    output_snapshots: int = 10
    output_write_all: bool = False
    output_dir: str = 'output'
    output_clean_dir: bool= True

def _initialize_problem(model, mesh):
    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_fields = model.n_fields
    n_elements = mesh.n_elements

    Q = np.empty((n_elements, n_fields), dtype=float)
    Qaux = np.zeros((Q.shape[0], model.aux_variables.length()))

    Q = model.initial_conditions.apply(mesh.element_center, Q)
    return Q, Qaux

def _get_compute_max_abs_eigenvalue(mesh, runtime_model, boundary_conditions, settings):
    reconstruction = settings.reconstruction
    reconstruction_edge = settings.reconstruction_edge
    def compute_max_abs_eigenvalue(Q, Qaux, parameters):

        max_abs_eigenvalue = -np.inf
        eigenvalues_i = np.empty(Q.shape[1], dtype=float)
        eigenvalues_j = np.empty(Q.shape[1], dtype=float)
        # Loop over the inner elements
        for i_elem in range(mesh.n_elements):
            for i_th_neighbor in range(mesh.element_n_neighbors[i_elem]):
                i_neighbor = mesh.element_neighbors[i_elem, i_th_neighbor]
                if i_elem > i_neighbor:
                    continue
                i_face = mesh.element_neighbors_face_index[i_elem, i_th_neighbor]
                # reconstruct 
                [Qi, Qauxi], [Qj, Qauxj] = reconstruction(mesh, [Q, Qaux], i_elem, i_th_neighbor)

                #TODO PROBLEM: the C interface has eigenvalues as reference, python interface
                # needs to return a value, because it cant do by-reference
                ev_i = runtime_model.eigenvalues(
                    Qi, Qauxi, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_i
                )
                if ev_i is not None:
                    eigenvalues_i = ev_i
                max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_i)))

                ev_j = runtime_model.eigenvalues(
                    Qj, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_j
                )
                if ev_j is not None:
                    eigenvalues_j = ev_j
                max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_j)))

        assert max_abs_eigenvalue > 10**(-6)
        # Loop over boundary faces
        for i in range(mesh.n_boundary_elements):
            i_elem = mesh.boundary_face_corresponding_element[i]
            i_face = mesh.boundary_face_element_face_index[i]
            Q_ghost = boundary_conditions.apply(i, i_elem, Q, mesh.element_face_normals[i_elem, i_face], settings.momentum_eqns)
            # reconstruct 
            [Qi, Qauxi], [Qj, Qauxj] = reconstruction_edge(mesh, [Q, Qaux], Q_ghost, i_elem )

            runtime_model.eigenvalues(
                Qi, Qauxi, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_i
            )
            max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_j)))
            runtime_model.eigenvalues(
                Qj, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_j
            )
            max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_j)))

        return max_abs_eigenvalue
    return compute_max_abs_eigenvalue

def _get_source(mesh, runtime_model, settings):
    def source(dt, Q, Qaux, parameters, dQ):
        # Loop over the inner elements
        for i_elem in range(mesh.n_elements):
            runtime_model.source(Q[i_elem], Qaux[i_elem], parameters, dQ[i_elem])
    return source

def _get_source_jac(mesh, runtime_model, settings):
    def source_jac(dt, Q, Qaux, parameters, dQ):
        # Loop over the inner elements
        for i_elem in range(mesh.n_elements):
            runtime_model.source_jacobian(Q[i_elem], Qaux[i_elem], parameters, dQ[i_elem])
    return source_jac


def _get_semidiscrete_solution_operator(mesh, runtime_model, boundary_conditions, settings):
    compute_num_flux = settings.num_flux
    compute_nc_flux = settings.nc_flux
    reconstruction = settings.reconstruction
    reconstruction_edge = settings.reconstruction_edge
    def operator_rhs_split(dt, Q, Qaux, parameters, dQ):

        # Loop over the inner elements
        for i_elem in range(mesh.n_elements):
            for i_th_neighbor in range(mesh.element_n_neighbors[i_elem]):
                i_neighbor = mesh.element_neighbors[i_elem, i_th_neighbor]
                if i_elem > i_neighbor:
                    continue
                i_face = mesh.element_neighbors_face_index[i_elem, i_th_neighbor]
                # reconstruct 
                [Qi, Qauxi], [Qj, Qauxj] = reconstruction(mesh, [Q, Qaux], i_elem, i_th_neighbor)

                #TODO callout to a requirement of the flux
                mesh_props = SimpleNamespace(dt_dx= dt / (2*mesh.element_inradius[i_elem]))
                flux, failed = compute_num_flux(
                    Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model, mesh_props=mesh_props
                )
                assert not failed
                nc_flux, failed = compute_nc_flux(
                    Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model
                )
                assert not failed
            
            
                # TODO index map (elem_edge_index) such that I do not need:
                # and if statement to avoid double edge computation (as I do now)
                # avoid double edge computation
                dQ[i_elem] -= (flux+nc_flux) * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_elem]
                dQ[i_neighbor] += (flux-nc_flux) * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_neighbor]

        # Loop over boundary faces
        for i in range(mesh.n_boundary_elements):
            i_elem = mesh.boundary_face_corresponding_element[i]
            i_face = mesh.boundary_face_element_face_index[i]
            Q_ghost = boundary_conditions.apply(i, i_elem, Q, mesh.element_face_normals[i_elem, i_face], settings.momentum_eqns)
            # i_neighbor = mesh.element_neighbors[i_elem, i_face]
            # reconstruct 
            [Qi, Qauxi], [Qj, Qauxj] = reconstruction_edge(mesh, [Q, Qaux], Q_ghost, i_elem )

            # callout to a requirement of the flux
            mesh_props = SimpleNamespace(dt_dx= dt / (2*mesh.element_inradius[i_elem]))
            flux, failed = compute_num_flux(
                Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model, mesh_props=mesh_props
            )
            assert not failed
            nc_flux, failed = compute_nc_flux(
                Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model
            )
            assert not failed

            dQ[i_elem] -= (flux+nc_flux) * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_elem]
        return dQ
    return operator_rhs_split
    

def fvm_unsteady_semidiscrete(mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1, runtime_model = None):
    iteration = 0
    time = 0.0

    assert model.dimension == mesh.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()
    # force=True is needed in order to disable old logging root handlers (if multiple test cases are run by one file)
    # otherwise only a logfile will be created for the first testcase but the log file gets deleted by rmtree
    # logging.basicConfig(
    #     filename=os.path.join(
    #         os.path.join(main_dir, self.output_dir), "logfile.log"
    #     ),
    #     filemode="w",
    #     level=self.logging_level,
    #     force=True,
    # )
    # logger = logging.getLogger(__name__ + ":solve_steady")

    Q, Qaux = _initialize_problem(model, mesh)
    parameters = model.parameter_values
    Qnew = deepcopy(Q)

    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    i_snapshot = 0
    dt_snapshot = settings.time_end/(settings.output_snapshots-1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    i_snapshot = io.save_fields(settings.output_dir, time, 0, i_snapshot, Qnew, Qaux, settings.output_write_all)
    mesh.write_to_hdf5(settings.output_dir)

    time_start = gettime()

    if runtime_model is None:
        model_functions = model.get_runtime_model()
        _ = model.create_c_interface()
        runtime_model = model.load_c_model()

    # map_elements_to_edges = recon.create_map_elements_to_edges(mesh)
    # on_edges_normal, on_edges_length = recon.get_edge_geometry_data(mesh)
    # space_solution_operator = get_semidiscrete_solution_operator(mesh, runtime_model, settings, on_edges_normal, on_edges_length, map_elements_to_edges)
    space_solution_operator = _get_semidiscrete_solution_operator(mesh, runtime_model, model.boundary_conditions, settings)
    compute_source = _get_source(mesh, runtime_model, settings)
    compute_source_jac = _get_source_jac(mesh, runtime_model, settings)
    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(mesh, runtime_model, model.boundary_conditions, settings)
    min_inradius = np.min(mesh.element_inradius)

    # print(f'hi from process {os.getpid()}')
    while (time < settings.time_end):
        # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue)
        assert dt > 10**(-6)

        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)
        Qnew = ode_solver_source(compute_source, Qnew, Qaux, parameters, dt, func_jac = compute_source_jac)

        # if time > 0.8:
        #     gradQ = mesh.gradQ(Q)
        #     for q, gradq in zip(Q, gradQ):
        #         u, v, w = reconstruct_uvw(q, gradq, model.levels, model.basis)
        # time, Qnew, Qaux, parameters, settings = callback_post_solve(time, Qnew, Qaux, parameters, settings)
        # Qavg = np.mean(Qavg_5steps, axis=0)
        # error = (
        #     self.compute_Lp_error(Qnew - Qavg, p=2, **kwargs)
        #     / (self.compute_Lp_error(Qavg, p=2, **kwargs) + 10 ** (-10))
        # ).max()
        # Qavg_5steps.pop()
        # Qavg_5steps.insert(0, Qnew)

        # Update solution and time
        time += dt
        iteration += 1

        # for callback in self.callback_function_list_post_solvestep:
        #     Qnew, kwargs = callback(self, Qnew, **kwargs)

        i_snapshot = io.save_fields(settings.output_dir, time, (i_snapshot+1)*dt_snapshot, i_snapshot, Qnew, Qaux, settings.output_write_all)
        
        # logger.info(
        #     "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
        #         iteration, gettime() - time_start, time, dt, error
        #     )
        # )
        # print(f'finished timestep: {os.getpid()}')
        progressbar.set_stat(min(time, settings.time_end))
        progressbar.update()

    progressbar.end()
    return settings
