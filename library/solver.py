import numpy as np
from types import SimpleNamespace
import pyprog
from attr import define
from typing import Callable, Optional, Type
from copy import deepcopy
from time import time as gettime

from library.model import *
from library.models.base import register_parameter_defaults
import library.reconstruction as recon
import library.flux as flux
import library.timestepping as timestepping


@define(slots=True, frozen=False, kw_only=True)
class Settings():
    name : str = 'Simulation'
    PDE : Type[Advection] = Advection
    momentum_eqns: list[int] = [0]
    parameters : SimpleNamespace = SimpleNamespace()
    reconstruction : Callable = recon.constant
    reconstruction_edge: Callable = recon.constant_edge
    num_flux : Callable = flux.LF
    compute_dt : Callable = timestepping.constant(dt=0.1)
    time_end : float = 1.0
    output_timesteps: int = 10

def initialize_problem(model, mesh):
    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    # n_all_elements = mesh.n_elements
    n_fields = model.n_fields

    Q = np.empty((n_all_elements, n_fields), dtype=float)
    Qaux = np.zeros((Q.shape[0], model.aux_variables.length()))
    # num_normals = mesh.element_n_neighbors
    # normals = np.array(
    #     [mesh.element_edge_normal[:, i] for i in range(mesh.n_nodes_per_element)]
    # )

    model.initial_conditions.apply(mesh.element_centers, Q[:mesh.n_elements])
    model.boundary_conditions.apply(Q)
    # return Q, Qaux, normals
    return Q, Qaux


def get_semidiscrete_solution_operator_new(mesh, runtime_model, settings):
    num_flux = settings.num_flux
    reconstruction = settings.reconstruction
    reconstruction_edge = settings.reconstruction_edge
    # on_edge_incircle = recon.constant(mesh, [mesh.element_incircle])
    def solution_operator(dt, Q, Qaux, parameters, dQ):
        # Qi, Qj, Qauxi, Qauxj = reconstruction(mesh, Q, Qaux)


        # Loop over the inner elements
        # for i_elem in range(mesh.n_elements):
        #     for i_edge in range(mesh.element_n_neighbors[i_elem]):
        for i_elem, i_edge in mesh.inner_edge_list:
            # reconstruct 
            [Qi, Qauxi], [Qj, Qauxj] = reconstruction(mesh, [Q, Qaux], i_elem, i_edge)

            # callout to a requirement of the flux
            mesh_props = SimpleNamespace(dt_dx= dt / (mesh.element_incircle[i_elem]))
            flux, failed = num_flux(
                Qi, Qj, Qauxi, Qauxj, parameters, mesh.element_edge_normal[i_elem, i_edge], runtime_model, mesh_props=mesh_props
            )
            assert not failed
            
            
            # TODO index map (elem_edge_index) such that I do not need:
            # and if statement to avoid double edge computation (as I do now)
            # avoid double edge computation
            dQ[i_elem] -= flux * mesh.element_edge_length[i_elem, i_edge] / mesh.element_volume[i_elem]
            i_neighbor = mesh.element_neighbors[i_elem, i_edge]
            dQ[i_neighbor] += flux * mesh.element_edge_length[i_elem, i_edge] / mesh.element_volume[i_neighbor]

        # Loop over boundary elements
        for i, i_elem in enumerate(mesh.boundary_edge_elements):
            i_neighbor = mesh.boundary_edge_neighbors[i]
            # reconstruct 
            [Qi, Qauxi], [Qj, Qauxj] = reconstruction_edge(mesh, [Q, Qaux], i_elem, i_neighbor)

            # callout to a requirement of the flux
            mesh_props = SimpleNamespace(dt_dx= dt / (mesh.element_incircle[i_elem]))
            flux, failed = num_flux(
                Qi, Qj, Qauxi, Qauxj, parameters, mesh.boundary_edge_normal[i], runtime_model, mesh_props=mesh_props
            )
            assert not failed

            dQ[i_elem] -= flux * mesh.boundary_edge_length[i] / mesh.element_volume[i_elem]

        # # Add flux (edges) to elements
        # (
        #     map_elements_to_edges_plus,
        #     map_elements_to_edges_minus,
        # ) = map_elements_to_edges
        # for i, elem in enumerate(map_elements_to_edges_plus):
        #     #TODO if hack: element_volume of of bounds error
        #     if elem < mesh.n_elements:
        #         dQ[elem] += flux[i] * on_edges_length[i] / mesh.element_volume[elem]
        # for i, elem in enumerate(map_elements_to_edges_minus):
        #     if elem < mesh.n_elements:
        #         dQ[elem] -= flux[i] * on_edges_length[i] / mesh.element_volume[elem]
        return dQ
    return solution_operator
    

def get_semidiscrete_solution_operator(mesh, runtime_model, settings, on_edges_normal, on_edges_length, map_elements_to_edges):
    num_flux = settings.num_flux
    reconstruction = settings.reconstruction
    # on_edge_incircle = recon.constant(mesh, [mesh.element_incircle])
    def solution_operator(dt, Q, Qaux, parameters, dQ):
        # Qi, Qj, Qauxi, Qauxj = reconstruction(mesh, Q, Qaux)
        [Qi, Qauxi], [Qj, Qauxj] = reconstruction(mesh, [Q, Qaux])

        # callout to a requirement of the flux
        mesh_props = SimpleNamespace(dt_dx= dt / (0.1*on_edges_length))
        flux, failed = num_flux(
            Qi, Qj, Qauxi, Qauxj, parameters, on_edges_normal, runtime_model, mesh_props=mesh_props
        )
        assert not failed

        # Add flux (edges) to elements
        (
            map_elements_to_edges_plus,
            map_elements_to_edges_minus,
        ) = map_elements_to_edges
        for i, elem in enumerate(map_elements_to_edges_plus):
            #TODO if hack: element_volume of of bounds error
            if elem < mesh.n_elements:
                dQ[elem] += flux[i] * on_edges_length[i] / mesh.element_volume[elem]
        for i, elem in enumerate(map_elements_to_edges_minus):
            if elem < mesh.n_elements:
                dQ[elem] -= flux[i] * on_edges_length[i] / mesh.element_volume[elem]
        return dQ
    return solution_operator

def fvm_unsteady_semidiscrete(mesh, model, settings, time_ode_solver):
    iteration = 0
    time = 0.0

    assert model.dimension == mesh.dimension
    # self.reinitialize()
    # self.solver.initialize(model=self.model, mesh=self.mesh)
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

    # if model.dimension > 1 and self.cfl > 0.5:
    #     logger.error(
    #         "CFL = {} > 1/Dimension = 1/{}".format(self.cfl, self.model.dimension)
    #     )

    # Qnew = initial_condition.initialize(
    #     self.model.initial_conditions,
    #     self.model.n_fields,
    #     self.mesh.element_centers,
    # )
    # Q, Qaux, normals = initialize_problem(model, mesh)
    Q, Qaux = initialize_problem(model, mesh)
    parameters = register_parameter_defaults(settings.parameters)
    Qnew = deepcopy(Q)

    # dt = self.dtmin
    # kwargs = {
    #     "model": self.model,
    #     "callback_parameters": self.callback_parameters,
    #     "aux_fields": {},
    #     "mesh": self.mesh,
    #     "time": time,
    #     "iteration": iteration,
    # }
    # self.init_output_timesteps()
    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    # output preparation
    # timeseries_output = np.zeros(
    #     (self.output_timesteps.shape[0], Q.shape[0], Q.shape[1])
    # )
    # timeseries_Q = self.save_output(time, Q, timeseries_Q)

    timeseries_output = []
    timeseries_output.append(( Q, Qaux, parameters, time ))

    time_start = gettime()

    # Qavg_5steps = [Q, Q, Q, Q, Q]
    # error = np.inf

    # convert element_edge_normals (dict of elements keys -> list of edge normals) to array
    # in case there are less than 4 (quad), 3(tri) normals in list due to bounary elements, fill array with existing normal
    # normals = np.zeros((self.mesh.num_nodes_per_element, 3, Qnew.shape[1]))
    # for k, v in self.mesh.element_edge_normal.items():
    #     for i in range(len(v)):
    #         normals[i, :, k] = v[i]
    #     # fill remaining with copy of last normal (since I do not want to have zero eigenvalues, rather copies of existing ones)
    #     for i in range(len(v), self.mesh.num_nodes_per_element):
    #         normals[i, :, k] = v[-1]

    model_functions = model.get_runtime_model()
    _ = model.create_c_interface()
    runtime_model = model.load_c_model()

    # map_elements_to_edges = recon.create_map_elements_to_edges(mesh)
    # on_edges_normal, on_edges_length = recon.get_edge_geometry_data(mesh)
    # space_solution_operator = get_semidiscrete_solution_operator(mesh, runtime_model, settings, on_edges_normal, on_edges_length, map_elements_to_edges)
    space_solution_operator = get_semidiscrete_solution_operator_new(mesh, runtime_model, settings)



    while (time < settings.time_end):
        Q = deepcopy(Qnew)
        # Time step estimation
        #TODO Hack
        ev_abs_max = 0.
        min_incircle = 0.
        dt = settings.compute_dt(ev_abs_max, min_incircle)

        # Repeat timestep for imaginary numbers
        # EVs, imaginary = self.solver.compute_eigenvalues(Qnew, normals)

        # TODO use this to avoid EV computation in flux. Problem
        # finding the appropriate neigbor edge it to find EVj!
        # kwargs["aux_fields"].update({"EVs": EVs})
        # EVmax = np.abs(kwargs["aux_fields"]["EVs"]).max()
        # dt = self.solver.compute_timestep_size(Qnew, self.cfl, EVmax)
        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # add 0.001 safty measure to avoid very small time steps
        if time + dt * 1.001 > settings.time_end:
            dt = settings.time_end - time + 10 ** (-10)
        # kwargs.update({"time": time, "dt": dt, "Qold": Q})
        # Qnew, _ = self.solver.step(Qnew, **kwargs)

        Qnew = time_ode_solver(space_solution_operator, Q, Qaux, parameters, dt)
        model.boundary_conditions.apply(Qnew)

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

        # kwargs.update({"time": time, "iteration": iteration})
        # for callback in self.callback_function_list_post_solvestep:
        #     Qnew, kwargs = callback(self, Qnew, **kwargs)

        # timeseries_Q = self.save_output(time, Qnew, timeseries_Q)
        timeseries_output.append(( Qnew, Qaux, parameters, dt ))

        # logger.info(
        #     "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
        #         iteration, gettime() - time_start, time, dt, error
        #     )
        # )
        progressbar.set_stat(min(time, settings.time_end))
        progressbar.update()

    # self.vtk_write_timestamp_file()
    progressbar.end()
    return timeseries_output, mesh 