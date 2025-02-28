# import numpy as np
# import os
# import json
# import logging
# import pyprog

# # import ray
# from time import time as gettime
# from copy import deepcopy
# import shutil

# from library.solver.baseclass import BaseYaml
# from library.solver.model import *
# from library.solver.solver import FiniteVolumeMethod
# import library.solver.mesh as mesh
# import library.solver.initial_condition as initial_condition

# # import library.solver.aux_fields as aux_fields
# import library.solver.boundary_conditions as boundary_conditions
# import library.solver.misc as misc
# from library.solver.limiter import limiter
# import library.solver.callbacks as callbacks

# eps = 10 ** (-14)
# main_dir = os.getenv("SMPYTHON")

# # ray.init(num_cpus=4, runtime_env={"env_vars": {"PYTHONPATH": main_dir}}, ignore_reinit_error=True)
# # # ray.init(num_cpus=4)

# # Currently used for counting logging calls


# class Controller(BaseYaml):
#     yaml_tag = "!Controller"

#     def set_default_parameters(self):
#         self.name = "Simulation"
#         self.output_dir = "output"
#         self.output_write_vtk = False
#         self.output_snapshots = 2

#         self.debug_animation = False
#         self.debug_animation_fields = [0]
#         self.debug_animation_range = [[]]
#         self.debug_animation_pause = 0.05

#         self.cfl = 0.9
#         self.dtmin = 10 ** (-5)
#         self.dtfix = None
#         self.dt_relaxation_factor = 0.1
#         self.warmup_interations = 0

#         self.time_end_sharp = True
#         self.time_end = 1.0
#         self.error_threshold = 10 ** (-10)
#         self.iteration_max = np.inf
#         self.callback_parameters = {}
#         self.callback_list_init = []
#         self.callback_list_post_solvestep = []

#         # DEBUG < INFO < WARNING < ERROR < CRITICAL
#         self.logging_level = "DEBUG"

#         self.mesh = mesh.Mesh1D()
#         self.model = ShallowMomentsWithBottom()
#         self.solver = FiniteVolumeMethod()

#     def set_runtime_variables(self):
#         self.callback_function_list_init = callbacks.get_callback_function_list(
#             self.callback_list_init
#         )
#         self.callback_function_list_post_solvestep = (
#             callbacks.get_callback_function_list(self.callback_list_post_solvestep)
#         )

#         boundary_conditions.initialize_boundary_conditions(
#             self.model.boundary_conditions, self.mesh
#         )
#         self.vtk_timestamp_file = {"file-series-version": "1.0", "files": []}

#     def save_config(self):
#         filename = "config.yaml"
#         self.write_class_to_file(
#             filepath=os.path.join(main_dir, self.output_dir), filename=filename
#         )

#     def load_config(filepath=main_dir + "/output/", filename="config.yaml"):
#         return Controller.read_class_from_file(filepath, filename)

#     def reinitialize(self, remove_data=True):
#         if remove_data:
#             shutil.rmtree(os.path.join(main_dir, self.output_dir), ignore_errors=True)
#         self.save_config()
#         new_controller = Controller.load_config(
#             filepath=os.path.join(main_dir, self.output_dir)
#         )
#         # Ugly way to update the current class with the runtime variables from new_controller
#         self.__dict__ = {**self.__dict__, **new_controller.__dict__}

#     def init_output_timesteps(self):
#         if self.output_snapshots == 1:
#             self.output_timesteps = np.array([self.time_end])
#         else:
#             self.output_timesteps = np.linspace(
#                 0.0, self.time_end, self.output_snapshots
#             )
#         self.output_next_save_at = 0

#     # def compute_time_step_size(self, EVmax, **kwargs):
#     #     # TODO incircle is the radius. This is not consistent with 1D -> double check!
#     #     if self.dtfix is None:
#     #         return self.cfl * np.min(kwargs["mesh"]["element_incircle"] / np.abs(EVmax))
#     #     return self.dtfix

#     # def reset_timestep(
#     #     self, Q, Qold, time, dt, count_imaginary, count_unresolved_imaginary
#     # ):
#     #     count_imaginary += 1
#     #     if dt > self.dtmin:
#     #         time -= dt
#     #         Q = deepcopy(Qold)
#     #         dt *= self.dt_relaxation_factor
#     #         dt = max(dt, self.dtmin)
#     #     else:
#     #         count_unresolved_imaginary += 1
#     #         dt = self.dtmin
#     #         if self.debug_output_level > 2:
#     #             print(
#     #                 "WARNING: imaginary EVs and dt=dtmin. Use linearized equations w.r.t. last time step with real eigenvalues."
#     #             )
#     #     return Q, Qold, time, dt, count_imaginary, count_unresolved_imaginary
#     #
#     # def compute_timestep_size(
#     #     self,
#     #     Q,
#     #     **kwargs,
#     # ):
#     #     EVmax = -np.inf
#     #     for d in range(self.model.dimension):
#     #         normal = np.eye(self.model.dimension)[d]
#     #         EVs, imag = self.model.eigenvalues(Q, normal)
#     #         imaginary = imaginary or imag
#     #         EVmax = max(EVmax, np.max(np.abs(EVs)))
#     #         imaginary = False
#     #     assert np.isfinite(EVmax)
#     #     if imaginary or ode_solver_failed:
#     #         return (
#     #             *self.reset_timestep(
#     #                 Q, Qold, time, dt, count_imaginary, count_unresolved_imaginary
#     #             ),
#     #             EVmax,
#     #             imaginary,
#     #             False,
#     #         )
#     #     else:
#     #         dt = self.compute_time_step_size(EVmax, **kwargs)
#     #         Qold = deepcopy(Q)
#     #         if dt < self.dtmin:
#     #             dt = self.dtmin
#     #             if self.debug_output_level > 3:
#     #                 print("WARNING: dt < dtmin. Continue with dtmin")
#     #         if iteration < self.warmup_interations:
#     #             dt = self.dtmin
#     #
#     #     if time + dt > self.time_end and self.time_end_sharp:
#     #         dt = self.time_end - time
#     #     # in case I time_end - time+dt is very small, some schemes will give bad results, e.g. lax-friedrichs. Therefore I finish the simulation, if I am nearly done
#     #     if self.time_end - (time + dt) < 0.01 * self.dtmin:
#     #         dt = self.time_end - time
#     #     return (
#     #         Q,
#     #         Qold,
#     #         time,
#     #         dt,
#     #         count_imaginary,
#     #         count_unresolved_imaginary,
#     #         EVmax,
#     #         imaginary,
#     #         False,
#     #     )

#     # def time_step_estimation_and_recovery(
#     #     self,
#     #     Q,
#     #     Qold,
#     #     count_imaginary,
#     #     count_unresolved_imaginary,
#     #     time,
#     #     dt,
#     #     iteration,
#     #     ode_solver_failed,
#     #     **kwargs,
#     # ):
#     #     EVmax = -np.inf
#     #     imaginary = False
#     #     for d in range(self.model.dimension):
#     #         normal = np.eye(self.model.dimension)[d]
#     #         EVs, imag = self.model.eigenvalues(Q, normal)
#     #         imaginary = imaginary or imag
#     #         EVmax = max(EVmax, np.max(np.abs(EVs)))
#     #         imaginary = False
#     #     assert np.isfinite(EVmax)
#     #     if imaginary or ode_solver_failed:
#     #         return (
#     #             *self.reset_timestep(
#     #                 Q, Qold, time, dt, count_imaginary, count_unresolved_imaginary
#     #             ),
#     #             EVmax,
#     #             imaginary,
#     #             False,
#     #         )
#     #     else:
#     #         dt = self.compute_time_step_size(EVmax, **kwargs)
#     #         Qold = deepcopy(Q)
#     #         if dt < self.dtmin:
#     #             dt = self.dtmin
#     #             if self.debug_output_level > 3:
#     #                 print("WARNING: dt < dtmin. Continue with dtmin")
#     #         if iteration < self.warmup_interations:
#     #             dt = self.dtmin
#     #
#     #     if time + dt > self.time_end and self.time_end_sharp:
#     #         dt = self.time_end - time
#     #     # in case I time_end - time+dt is very small, some schemes will give bad results, e.g. lax-friedrichs. Therefore I finish the simulation, if I am nearly done
#     #     if self.time_end - (time + dt) < 0.01 * self.dtmin:
#     #         dt = self.time_end - time
#     #     return (
#     #         Q,
#     #         Qold,
#     #         time,
#     #         dt,
#     #         count_imaginary,
#     #         count_unresolved_imaginary,
#     #         EVmax,
#     #         imaginary,
#     #         False,
#     #     )

#     def compute_Lp_error(self, Q, p=1, **kwargs):
#         result = np.zeros(Q.shape[0])
#         if not "filtered_elements" in kwargs.keys():
#             filtered_elements = np.array(np.ones((Q.shape[1])), dtype=bool)
#         else:
#             filtered_elements = kwargs["filtered_elements"]
#         for i_elem in range(Q.shape[1]):
#             if filtered_elements[i_elem]:
#                 volume = kwargs["mesh"].element_volume[i_elem]
#                 result += volume * (np.abs(Q[:, i_elem])) ** p
#         result = result ** (1 / p)
#         for r in result:
#             if np.isnan(r).any():
#                 r = np.inf
#         return result

#     def solve_unsteady(self):
#         iteration = 0
#         time = 0.0

#         assert self.model.dimension == self.mesh.dim
#         self.reinitialize()
#         self.solver.initialize(model=self.model, mesh=self.mesh)
#         progressbar = pyprog.ProgressBar(
#             "{}: ".format(self.name),
#             "\r",
#             self.time_end,
#             complete_symbol="█",
#             not_complete_symbol="-",
#         )
#         progressbar.progress_explain = ""
#         progressbar.update()
#         # force=True is needed in order to disable old logging root handlers (if multiple test cases are run by one file)
#         # otherwise only a logfile will be created for the first testcase but the log file gets deleted by rmtree
#         logging.basicConfig(
#             filename=os.path.join(
#                 os.path.join(main_dir, self.output_dir), "logfile.log"
#             ),
#             filemode="w",
#             level=self.logging_level,
#             force=True,
#         )
#         logger = logging.getLogger(__name__ + ":solve_steady")

#         if self.model.dimension > 1 and self.cfl > 0.5:
#             logger.error(
#                 "CFL = {} > 1/Dimension = 1/{}".format(self.cfl, self.model.dimension)
#             )

#         Qnew = initial_condition.initialize(
#             self.model.initial_conditions,
#             self.model.n_fields,
#             self.mesh.element_centers,
#         )
#         Q = deepcopy(Qnew)

#         dt = self.dtmin
#         kwargs = {
#             "model": self.model,
#             "callback_parameters": self.callback_parameters,
#             "aux_fields": {},
#             "mesh": self.mesh,
#             "time": time,
#             "iteration": iteration,
#         }
#         self.init_output_timesteps()
#         for callback in self.callback_function_list_init:
#             Qnew, kwargs = callback(self, Qnew, **kwargs)

#         # output preparation
#         timeseries_Q = np.zeros(
#             (self.output_timesteps.shape[0], Q.shape[0], Q.shape[1])
#         )
#         timeseries_Q = self.save_output(time, Q, timeseries_Q)

#         time_start = gettime()

#         Qavg_5steps = [Q, Q, Q, Q, Q]
#         error = np.inf

#         # convert element_edge_normals (dict of elements keys -> list of edge normals) to array
#         # in case there are less than 4 (quad), 3(tri) normals in list due to bounary elements, fill array with existing normal
#         normals = np.zeros((self.mesh.num_nodes_per_element, 3, Qnew.shape[1]))
#         for k, v in self.mesh.element_edge_normal.items():
#             for i in range(len(v)):
#                 normals[i, :, k] = v[i]
#             # fill remaining with copy of last normal (since I do not want to have zero eigenvalues, rather copies of existing ones)
#             for i in range(len(v), self.mesh.num_nodes_per_element):
#                 normals[i, :, k] = v[-1]

#         while (
#             time < self.time_end
#             and error > self.error_threshold
#             and iteration < self.iteration_max
#         ):
#             Q = np.array(Qnew)
#             # Time step estimation
#             # Repeat timestep for imaginary numbers
#             EVs, imaginary = self.solver.compute_eigenvalues(Qnew, normals)
#             # TODO use this to avoid EV computation in flux. Problem
#             # finding the appropriate neigbor edge it to find EVj!
#             kwargs["aux_fields"].update({"EVs": EVs})
#             EVmax = np.abs(kwargs["aux_fields"]["EVs"]).max()
#             dt = self.solver.compute_timestep_size(Qnew, self.cfl, EVmax)
#             assert not np.isnan(dt) and np.isfinite(dt)
#             if iteration < self.warmup_interations:
#                 dt *= self.dt_relaxation_factor
#             if dt < self.dtmin:
#                 dt = self.dtmin
#             # add 0.001 safty measure to avoid very small time steps
#             if self.time_end_sharp and time + dt * 1.001 > self.time_end:
#                 dt = self.time_end - time + 10 ** (-10)
#             kwargs.update({"time": time, "dt": dt, "Qold": Q})
#             Qnew, _ = self.solver.step(Qnew, **kwargs)

#             Qavg = np.mean(Qavg_5steps, axis=0)
#             error = (
#                 self.compute_Lp_error(Qnew - Qavg, p=2, **kwargs)
#                 / (self.compute_Lp_error(Qavg, p=2, **kwargs) + 10 ** (-10))
#             ).max()
#             Qavg_5steps.pop()
#             Qavg_5steps.insert(0, Qnew)

#             # Update solution and time
#             time += dt
#             iteration += 1

#             kwargs.update({"time": time, "iteration": iteration})
#             for callback in self.callback_function_list_post_solvestep:
#                 Qnew, kwargs = callback(self, Qnew, **kwargs)

#             timeseries_Q = self.save_output(time, Qnew, timeseries_Q)

#             logger.info(
#                 "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
#                     iteration, gettime() - time_start, time, dt, error
#                 )
#             )
#             progressbar.set_stat(min(time, self.time_end))
#             progressbar.update()

#         self.vtk_write_timestamp_file()
#         progressbar.end()
#         return timeseries_Q, self.mesh.element_centers, time, kwargs

#     def postprocessing(self):
#         assert self.model.dimension == self.mesh.dim

#         # find number of vtks
#         path = main_dir + "/" + self.output_dir
#         files = [f for f in os.listdir(path) if "out." in f]
#         n_vtks = len(files)

#         time = 0
#         iteration = 0

#         self.reinitialize(remove_data=False)
#         self.solver.initialize(model=self.model, mesh=self.mesh)
#         progressbar = pyprog.ProgressBar(
#             "{}: ".format("Postprocessing"),
#             "\r",
#             n_vtks,
#             complete_symbol="█",
#             not_complete_symbol="-",
#         )
#         progressbar.progress_explain = ""
#         progressbar.update()
#         logging.basicConfig(
#             filename=os.path.join(
#                 os.path.join(main_dir, self.output_dir), "logfile.log"
#             ),
#             filemode="w",
#             level=self.logging_level,
#             force=True,
#         )
#         logger = logging.getLogger(__name__ + ":postprocessing")

#         ic = {
#             "scheme": "file",
#             "filename": os.path.join(
#                 os.path.join(main_dir, self.output_dir), "out.{}.vtk".format(iteration)
#             ),
#             "dim": 2,
#             "map_fields": list(
#                 np.linspace(0, self.model.n_fields - 1, self.model.n_fields, dtype=int)
#             ),
#         }
#         Qnew = initial_condition.initialize(
#             ic,
#             self.model.n_fields,
#             self.mesh.element_centers,
#         )

#         kwargs = {
#             "model": self.model,
#             "callback_parameters": self.callback_parameters,
#             "aux_fields": {},
#             "mesh": self.mesh,
#         }
#         self.init_output_timesteps()
#         kwargs.update({"time": time, "iteration": iteration})

#         for callback in self.callback_function_list_init:
#             Qnew, kwargs = callback(self, Qnew, **kwargs)

#         # convert element_edge_normals (dict of elements keys -> list of edge normals) to array
#         # in case there are less than 4 (quad), 3(tri) normals in list due to bounary elements, fill array with existing normal
#         normals = np.zeros((self.mesh.num_nodes_per_element, 3, Qnew.shape[1]))
#         for k, v in self.mesh.element_edge_normal.items():
#             for i in range(len(v)):
#                 normals[i, :, k] = v[i]
#             # fill remaining with copy of last normal (since I do not want to have zero eigenvalues, rather copies of existing ones)
#             for i in range(len(v), self.mesh.num_nodes_per_element):
#                 normals[i, :, k] = v[-1]

#         iteration += 1
#         while iteration < n_vtks:
#             # TODO update time
#             time += 0
#             # load initial condition with file
#             ic = {
#                 "scheme": "file",
#                 "filename": main_dir
#                 + "/"
#                 + self.output_dir
#                 + "out.{}.vtk".format(iteration),
#                 "dim": 2,
#                 "map_fields": list(
#                     np.linspace(
#                         0, self.model.n_fields - 1, self.model.n_fields, dtype=int
#                     )
#                 ),
#             }
#             Qnew = initial_condition.initialize(
#                 ic,
#                 self.model.n_fields,
#                 self.mesh.element_centers,
#             )
#             kwargs.update({"time": time, "iteration": iteration})
#             for callback in self.callback_function_list_post_solvestep:
#                 Qnew, kwargs = callback(self, Qnew, **kwargs)

#             logger.info("Iteration: {:6.0f}, Time: {:2.4f}".format(iteration, time))
#             iteration += 1
#             progressbar.set_stat(min(iteration, n_vtks))
#             progressbar.update()

#         progressbar.end()

#     def save_output(self, time, Q, Qlist):
#         # recursion increases the counter too much at the last time step
#         if self.output_next_save_at >= self.output_timesteps.shape[0]:
#             return Qlist
#         if time >= self.output_timesteps[self.output_next_save_at]:
#             if self.output_write_vtk:
#                 filename = (
#                     main_dir
#                     + "/"
#                     + self.output_dir
#                     + "/out."
#                     + str(self.output_next_save_at)
#                     + ".vtk"
#                 )
#                 Qout = np.array(Q)
#                 # if "coordinate_transform" in self.solver.scheme:
#                 #     Qtrans = np.zeros_like(Q)
#                 #     for i in range(Q.shape[1]):
#                 #         normal = self.mesh.element_centers[i][:2]
#                 #         normal /= np.linalg.norm(normal)
#                 #         Qtrans[:, i] = self.model.compute_Q_in_normal_transverse(
#                 #             Q[:, i][:, np.newaxis], normal[:, np.newaxis]
#                 #         )
#                 #     Qout = Qtrans

#                 self.mesh.write_to_file(
#                     filename,
#                     fields=Qout,
#                     field_names=self.model.field_names,
#                 )
#                 self.vtk_timestamp_file["files"].append(
#                     {
#                         "name": "out." + str(self.output_next_save_at) + ".vtk",
#                         "time": time,
#                     }
#                 )
#             Qlist[self.output_next_save_at] = Q
#             self.output_next_save_at += 1
#             # in case dt captures more than 1 time_stamp, I want to write out 2 times the same solution
#             return self.save_output(time, Q, Qlist)
#         return Qlist

#     def vtk_write_timestamp_file(self):
#         if self.output_write_vtk:
#             with open(main_dir + "/" + self.output_dir + "/time.vtk.series", "x") as f:
#                 json.dump(self.vtk_timestamp_file, f)
