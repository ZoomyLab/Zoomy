import numpy as np
from types import SimpleNamespace
import pyprog
from attr import define, field
from typing import Callable, Optional, Type
from copy import deepcopy
from time import time as gettime

from library.model import *
from library.models.base import register_parameter_defaults
import library.reconstruction as recon
import library.flux as flux
import library.timestepping as timestepping
import library.io as io
import library.fvm_mesh as fvm_mesh


@define(slots=True, frozen=False, kw_only=True)
class Settings():
    name : str = 'Simulation'
    momentum_eqns: list[int] = [0]
    parameters : dict = {}
    reconstruction : Callable = recon.constant
    reconstruction_edge: Callable = recon.constant_edge
    num_flux : Callable = flux.LF
    compute_dt : Callable = timestepping.constant(dt=0.1)
    time_end : float = 1.0
    truncate_last_time_step: bool = True
    output_snapshots: int = 10
    output_write_all: bool = False
    output_dir: str = 'output'
    output_clean_dir: bool= True

def initialize_problem(model, mesh):
    n_ghosts = model.boundary_conditions.initialize(mesh)

    # n_all_elements = mesh.n_elements + n_ghosts
    n_all_elements = mesh.n_elements
    n_fields = model.n_fields

    Q = np.empty((n_all_elements, n_fields), dtype=float)
    Qaux = np.zeros((Q.shape[0], model.aux_variables.length()))

    model.initial_conditions.apply(mesh.element_center, Q[:mesh.n_elements])
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

                runtime_model.eigenvalues(
                    Qi, Qauxi, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_i
                )
                max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_i)))

                runtime_model.eigenvalues(
                    Qj, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], eigenvalues_j
                )
                max_abs_eigenvalue = max(max_abs_eigenvalue, np.max(np.abs(eigenvalues_j)))

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


def get_semidiscrete_solution_operator(mesh, runtime_model, boundary_conditions, settings):
    num_flux = settings.num_flux
    reconstruction = settings.reconstruction
    reconstruction_edge = settings.reconstruction_edge
    def solution_operator(dt, Q, Qaux, parameters, dQ):

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
                flux, failed = num_flux(
                    Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model, mesh_props=mesh_props
                )
                assert not failed
            
            
                # TODO index map (elem_edge_index) such that I do not need:
                # and if statement to avoid double edge computation (as I do now)
                # avoid double edge computation
                dQ[i_elem] -= flux * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_elem]
                dQ[i_neighbor] += flux * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_neighbor]

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
            flux, failed = num_flux(
                Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_face_normals[i_elem, i_face], runtime_model, mesh_props=mesh_props
            )
            assert not failed

            dQ[i_elem] -= flux * mesh.element_face_areas[i_elem, i_face] / mesh.element_volume[i_elem]
        return dQ
    return solution_operator
    

def fvm_unsteady_semidiscrete(mesh, model, settings, time_ode_solver):
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
    # progressbar.progress_explain = ""
    # progressbar.update()
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

    Q, Qaux = initialize_problem(model, mesh)
    parameters = model.parameter_values
    Qnew = deepcopy(Q)

    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    i_snapshot = 0
    dt_snapshot = time/(settings.output_snapshots-1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    i_snapshot = io.save_fields(settings.output_dir, time, 0, i_snapshot, Qnew, Qaux, settings.output_write_all)
    mesh.write_to_hdf5(settings.output_dir)

    time_start = gettime()

    model_functions = model.get_runtime_model()
    _ = model.create_c_interface()
    runtime_model = model.load_c_model()

    # map_elements_to_edges = recon.create_map_elements_to_edges(mesh)
    # on_edges_normal, on_edges_length = recon.get_edge_geometry_data(mesh)
    # space_solution_operator = get_semidiscrete_solution_operator(mesh, runtime_model, settings, on_edges_normal, on_edges_length, map_elements_to_edges)
    space_solution_operator = get_semidiscrete_solution_operator(mesh, runtime_model, model.boundary_conditions, settings)
    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(mesh, runtime_model, model.boundary_conditions, settings)
    min_inradius = np.min(mesh.element_inradius)


    while (time < settings.time_end):
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue)

        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        Qnew = time_ode_solver(space_solution_operator, Q, Qaux, parameters, dt)

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
        progressbar.set_stat(min(time, settings.time_end))
        progressbar.update()

    progressbar.end()
    return settings
