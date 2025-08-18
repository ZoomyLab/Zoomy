# import os
# import sys
# import numpy as np
# from copy import deepcopy as deepcopy
# from multiprocessing import Pool

# from library.solver.baseclass import BaseYaml
# import library.solver.reconstruction as reconstruction
# import library.solver.flux as flux
# import library.solver.flux_direct as flux_direct
# import library.solver.quasilinear as quasilinear
# import library.solver.non_conservative as non_conservative
# import library.solver.ode as ode
# import library.solver.boundary_conditions as boundary_conditions

# main_dir = os.getenv("SMPYTHON")

# eps = 10 ** (-10)


# class FiniteVolumeMethod(BaseYaml):
#     yaml_tag = "!FVM"

#     def set_default_default_parameters(self):
#         self.scheme = "explicit_split_source_vectorized"
#         self.reconstruction = reconstruction.Reconstruction()
#         self.flux = flux.Flux()
#         self.flux_direct = flux_direct.FluxDirect()
#         self.quasilinear = quasilinear.Quasilinear()
#         self.nc = non_conservative.NonConservativeTerms()
#         self.integrator = ode.ODE()

#     def initialize(self, **kwargs):
#         self.model = kwargs["model"]
#         self.dimension = self.model.dimension
#         self.flux_func = self.flux.evaluate(self.model.flux, **kwargs)
#         self.nc_func = self.nc.evaluate(self.model)
#         self.source_func = self.model.rhs
#         self.bc_func = lambda boundary_edge_index, Q, **kwargs: boundary_conditions.get_boundary_value(
#             self.model.boundary_conditions,
#             boundary_edge_index,
#             Q,
#             dim=self.dimension,
#             **kwargs,
#         )
#         self.bc_flux_func = lambda boundary_edge_index, Q, **kwargs: boundary_conditions.get_boundary_flux_value(
#             self.model.boundary_conditions_flux,
#             boundary_edge_index,
#             Q,
#             dim=self.dimension,
#             **kwargs,
#         )
#         self.mesh = kwargs["mesh"]
#         self.step = self.get_step_function()

#     def compute_max_eigenvalue(self, Q):
#         EVmax = -np.inf
#         imaginary = False
#         for d in range(self.dimension):
#             normal = np.repeat(
#                 np.eye(self.dimension)[d][:, np.newaxis], Q.shape[1], axis=1
#             )
#             EVs, imag = self.model.eigenvalues(Q, normal)
#             imaginary = imaginary or imag
#             EVmax = max(EVmax, np.max(np.abs(EVs)))
#         assert np.isfinite(EVmax)
#         # assert imaginary is False
#         return EVmax

#     def compute_eigenvalues(self, Q, normals):
#         # number_of_eigenvalues = self.model.eigenvalues(
#         #     Q[:, 0][:, np.newaxis], normals[0, :, 0][:, np.newaxis]
#         # )[0].shape[0]
#         number_of_eigenvalues = Q.shape[0]
#         EVs = np.zeros(
#             (Q.shape[1], normals.shape[0], number_of_eigenvalues), dtype=float
#         )
#         imaginary = False

#         for edge in range(normals.shape[0]):
#             ev, imag = self.model.eigenvalues(Q, normals[edge])
#             imaginary = imaginary or imag
#             # assert imaginary is False
#             assert np.isfinite(ev).all()
#             EVs[:, edge, :] = np.real(ev).T
#         return EVs, imaginary

#     def compute_timestep_size(self, Q, cfl, EVmax):
#         return cfl * np.min(self.mesh.element_incircle / np.abs(EVmax))

#     def get_step_function(self):
#         return getattr(self, "step_" + self.scheme)

#     def loop_inner_explicit(self, Q, **kwargs):
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]
#         # loop over all inner elements
#         for i_elem in range(self.mesh.n_elements):
#             for i_edge, _ in enumerate(self.mesh.element_neighbors[i_elem]):
#                 i_neighbor = (self.mesh.element_neighbors[i_elem])[i_edge]
#                 n_ij = self.mesh.element_edge_normal[i_elem][i_edge][:, np.newaxis][
#                     : self.mesh.dim
#                 ]
#                 # if this assertion failes, the normal was not set correctly
#                 assert np.linalg.norm(n_ij) < 1.0 + 10 ** (-6)
#                 Qi = np.array(Q[:, i_elem])[:, np.newaxis]
#                 Qj = np.array(Q[:, i_neighbor])[:, np.newaxis]

#                 # TODO HACK
#                 # Qi[:-1] = np.where(Qi[0] > 0, Qi[:-1], np.zeros_like(Qi[:-1]))
#                 # Qj[:-1] = np.where(Qj[0] > 0, Qj[:-1], np.zeros_like(Qj[:-1]))

#                 if Qi[0] > eps or Qj[0] > eps:
#                     Fn, step_failed = self.flux_func(Qi, Qj, n_ij, **kwargs)
#                     NCn = self.nc_func(Qi, Qj, n_ij, **kwargs)
#                     # if np.allclose(Fn, np.zeros_like(Fn)):
#                     #     NCn = np.zeros_like(Fn)
#                     dQelem = -(
#                         dt
#                         / self.mesh.element_volume[i_elem]
#                         * self.mesh.element_edge_length[i_elem][i_edge]
#                         * (Fn + NCn).flatten()
#                     )
#                     dQ[:, i_elem] += dQelem
#         return dQ

#     # TODO: currently, inner fluxes are computed twice
#     # TODO: some arrays can be precomputed
#     def loop_fluxes_vectorized(self, Q, **kwargs):
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]

#         n_edges = int((self.mesh.n_elements * self.mesh.num_nodes_per_element + self.mesh.n_boundary_edges)/2)
#         n_edges_inner = n_edges - self.mesh.n_boundary_edges

#         # n_edges = int((self.mesh.n_elements * self.mesh.num_nodes_per_element))
#         Qi = np.zeros((Q.shape[0], n_edges))
#         Qj = np.zeros((Q.shape[0], n_edges))
#         n_ij = np.zeros((3, n_edges))
#         edge_length = np.zeros((n_edges))
#         element_index_inner = np.zeros((n_edges_inner), dtype=int)
#         element_index_inner_neighbor = np.zeros((n_edges_inner), dtype=int)
#         element_index_boundary = np.zeros((self.mesh.n_boundary_edges), dtype=int)
#         i = 0
#         # loop over all inner elements
#         for i_elem in range(self.mesh.n_elements):
#             for i_edge, _ in enumerate(self.mesh.element_neighbors[i_elem]):
#                 i_neighbor = (self.mesh.element_neighbors[i_elem])[i_edge]
#                 if i_neighbor > i_elem:
#                     continue
#                 n_ij[:, i] = self.mesh.element_edge_normal[i_elem][i_edge]                # if this assertion failes, the normal was not set correctly
#                 Qi[:, i] = np.array(Q[:, i_elem])
#                 Qj[:, i] = np.array(Q[:, i_neighbor])
#                 edge_length[i] = self.mesh.element_edge_length[i_elem][i_edge]
#                 element_index_inner[i] = i_elem
#                 element_index_inner_neighbor[i] = i_neighbor
#                 i += 1

#         i_boundary = 0
#         for i_bp, i_elem in enumerate(self.mesh.boundary_edge_element):
#             n_ij[:, i] = self.mesh.boundary_edge_normal[i_bp]
#             Qi[:,i] = np.array(Q[:, i_elem])
#             Qj[:,i] = self.bc_func(i_bp, Q, **kwargs)

#             edge_length[i] = self.mesh.boundary_edge_length[i_bp]

#             element_index_boundary[i_boundary] = i_elem
#             i += 1
#             i_boundary +=1

#         n_ij = n_ij[:self.mesh.dim, :]
#         Fn, step_failed = self.flux_func(Qi, Qj, n_ij, **kwargs)
#         NCn = self.nc_func(Qi, Qj, n_ij, **kwargs)

#         # Reduce edges to cells
#         for i in range(n_edges_inner):
#             i_elem = element_index_inner[i]
#             i_neighbor = element_index_inner_neighbor[i]
#             dQ[:, i_elem] -= dt / self.mesh.element_volume[i_elem] * edge_length[i] * (Fn + NCn)[:,i]
#             dQ[:, i_neighbor] += dt / self.mesh.element_volume[i_neighbor] * edge_length[i] * (Fn - NCn)[:,i]
#         for j in range(self.mesh.n_boundary_edges):
#             i_elem = element_index_boundary[j]
#             dQ[:, i_elem] -= dt / self.mesh.element_volume[i_elem] * edge_length[j+n_edges_inner] * (Fn + NCn)[:,j+n_edges_inner]
#         return dQ

#     def loop_fluxes_vectorized_coordinate_transform(self, Q, **kwargs):
#         print(Q.shape)
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]

#         n_edges = int((self.mesh.n_elements * self.mesh.num_nodes_per_element + self.mesh.n_boundary_edges)/2)
#         n_edges_inner = n_edges - self.mesh.n_boundary_edges

#         # n_edges = int((self.mesh.n_elements * self.mesh.num_nodes_per_element))
#         Qi = np.zeros((Q.shape[0], n_edges))
#         Qj = np.zeros((Q.shape[0], n_edges))
#         n_ij = np.zeros((3, n_edges))
#         edge_length = np.zeros((n_edges))
#         element_index_inner = np.zeros((n_edges_inner), dtype=int)
#         element_index_inner_neighbor = np.zeros((n_edges_inner), dtype=int)
#         element_index_boundary = np.zeros((self.mesh.n_boundary_edges), dtype=int)
#         i = 0
#         # loop over all inner elements
#         for i_elem in range(self.mesh.n_elements):
#             for i_edge, _ in enumerate(self.mesh.element_neighbors[i_elem]):
#                 i_neighbor = (self.mesh.element_neighbors[i_elem])[i_edge]
#                 if i_neighbor > i_elem:
#                     continue
#                 n_ij[:, i] = self.mesh.element_edge_normal[i_elem][i_edge]                # if this assertion failes, the normal was not set correctly
#                 Qi[:, i] = np.array(Q[:, i_elem])
#                 Qj[:, i] = np.array(Q[:, i_neighbor])
#                 Qi[:,i] = self.model.compute_Q_in_normal_transverse(Qi[:, i][:, np.newaxis], n_ij[:, i][:, np.newaxis])
#                 Qj[:,i] = self.model.compute_Q_in_normal_transverse(Qj[:, i][:, np.newaxis], n_ij[:, i][:, np.newaxis])
#                 edge_length[i] = self.mesh.element_edge_length[i_elem][i_edge]
#                 element_index_inner[i] = i_elem
#                 element_index_inner_neighbor[i] = i_neighbor
#                 i += 1

#         i_boundary = 0
#         for i_bp, i_elem in enumerate(self.mesh.boundary_edge_element):
#             n_ij[:, i] = self.mesh.boundary_edge_normal[i_bp]
#             Qi[:,i] = self.model.compute_Q_in_normal_transverse(np.array(Q[:,i_elem])[:, np.newaxis], n_ij[:, i][:, np.newaxis])
#             Qj[:,i] = self.model.compute_Q_in_normal_transverse(self.bc_func(i_bp, Q, **kwargs)[:, np.newaxis], n_ij[:, i][:, np.newaxis])

#             edge_length[i] = self.mesh.boundary_edge_length[i_bp]

#             element_index_boundary[i_boundary] = i_elem
#             i += 1
#             i_boundary +=1

#         n_ij = n_ij[:self.mesh.dim, :]
#         n_ij_transformed = np.zeros_like(n_ij)
#         n_ij_transformed[0, :] = 1.0
#         Fn, step_failed = self.flux_func(Qi, Qj, n_ij_transformed, **kwargs)
#         NCn = self.nc_func(Qi, Qj, n_ij_transformed, **kwargs)

#         # Reduce edges to cells
#         for i in range(n_edges_inner):
#             i_elem = element_index_inner[i]
#             i_neighbor = element_index_inner_neighbor[i]
#             dQelem = -dt / self.mesh.element_volume[i_elem] * edge_length[i] * (Fn + NCn)[:,i]
#             dQ[:, i_elem] += self.model.compute_Q_in_x_y(dQelem[:, np.newaxis], n_ij[:,i][:, np.newaxis])
#             dQneighbor = dt / self.mesh.element_volume[i_neighbor] * edge_length[i] * (Fn - NCn)[:,i]
#             dQ[:, i_neighbor] += self.model.compute_Q_in_x_y(dQneighbor[:, np.newaxis], n_ij[:,i][:, np.newaxis])
#         for j in range(self.mesh.n_boundary_edges):
#             i_elem = element_index_boundary[j]
#             dQelem = -dt / self.mesh.element_volume[i_elem] * edge_length[j+n_edges_inner] * (Fn + NCn)[:,j+n_edges_inner]
#             dQ[:, i_elem] += self.model.compute_Q_in_x_y(dQelem[:, np.newaxis], n_ij[:,j + n_edges_inner][:, np.newaxis])
#         return dQ

#     def loop_inner_explicit_coordinate_transform(self, Q, **kwargs):
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]
#         # loop over all inner elements
#         for i_elem in range(self.mesh.n_elements):
#             for i_edge, _ in enumerate(self.mesh.element_neighbors[i_elem]):
#                 i_neighbor = (self.mesh.element_neighbors[i_elem])[i_edge]
#                 n_ij = self.mesh.element_edge_normal[i_elem][i_edge][:, np.newaxis][
#                     : self.mesh.dim
#                 ]
#                 # if this assertion failes, the normal was not set correctly
#                 assert np.linalg.norm(n_ij) < 1.0 + 10 ** (-6)
#                 Qi = np.array(Q[:, i_elem])[:, np.newaxis]
#                 Qj = np.array(Q[:, i_neighbor])[:, np.newaxis]
#                 Qi = self.model.compute_Q_in_normal_transverse(Qi, n_ij)
#                 Qj = self.model.compute_Q_in_normal_transverse(Qj, n_ij)
#                 Qi = Qi[:, np.newaxis]
#                 Qj = Qj[:, np.newaxis]
#                 n_normal = np.zeros_like(n_ij)
#                 n_normal[0] = 1.0

#                 # TODO HACK
#                 # Qi[:-1] = np.where(Qi[0] > 0, Qi[:-1], np.zeros_like(Qi[:-1]))
#                 # Qj[:-1] = np.where(Qj[0] > 0, Qj[:-1], np.zeros_like(Qj[:-1]))

#                 if Qi[0] > eps or Qj[0] > eps:
#                     Fn, step_failed = self.flux_func(Qi, Qj, n_normal, **kwargs)
#                     NCn = self.nc_func(Qi, Qj, n_normal, **kwargs)
#                     dQelem = (
#                         dt
#                         / self.mesh.element_volume[i_elem]
#                         * self.mesh.element_edge_length[i_elem][i_edge]
#                         * (Fn + NCn).flatten()
#                     )
#                     dQelem = dQelem[:, np.newaxis]
#                     dQelem = self.model.compute_Q_in_x_y(dQelem, n_ij)
#                     dQ[:, i_elem] -= dQelem
#         return dQ

#     def loop_boundary_explicit(self, Q, **kwargs):
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]
#         for i_bp, elem in enumerate(self.mesh.boundary_edge_element):
#             kwargs["i_elem"] = elem
#             # TODO WRONG!!
#             kwargs["i_edge"] = i_bp
#             n_ij = self.mesh.boundary_edge_normal[i_bp][: self.mesh.dim][:, np.newaxis]
#             kwargs["i_neighbor"] = elem

#             edge_length = self.mesh.boundary_edge_length[i_bp]

#             Qi = np.array(Q[:, elem])[:, np.newaxis]
#             Qj = self.bc_func(i_bp, Q, **kwargs)[:, np.newaxis]
#             if Qi[0] > eps or Qj[0] > eps:
#                 Fn, step_failed = self.flux_func(Qi, Qj, n_ij, **kwargs)
#                 NCn = self.nc_func(Qi, Qj, n_ij, **kwargs)
#                 dQelem = (
#                     dt
#                     / self.mesh.element_volume[elem]
#                     * edge_length
#                     * (Fn + NCn).flatten()
#                 )
#                 dQ[:, elem] -= dQelem
#         return dQ

#     def loop_boundary_explicit_coordinate_transform(self, Q, **kwargs):
#         dQ = np.zeros_like(Q)
#         dt = kwargs["dt"]
#         for i_bp, elem in enumerate(self.mesh.boundary_edge_element):
#             kwargs["i_elem"] = elem
#             # TODO WRONG!!
#             kwargs["i_edge"] = i_bp
#             n_ij = self.mesh.boundary_edge_normal[i_bp][: self.mesh.dim][:, np.newaxis]
#             kwargs["i_neighbor"] = elem

#             edge_length = self.mesh.boundary_edge_length[i_bp]

#             Qi = np.array(Q[:, elem])[:, np.newaxis]
#             Qj = self.bc_func(i_bp, Q, **kwargs)[:, np.newaxis]
#             Qi = self.model.compute_Q_in_normal_transverse(Qi, n_ij)
#             Qj = self.model.compute_Q_in_normal_transverse(Qj, n_ij)
#             Qi = Qi[:, np.newaxis]
#             Qj = Qj[:, np.newaxis]
#             n_normal = np.zeros_like(n_ij)
#             n_normal[0] = 1.0
#             if Qi[0] > eps or Qj[0] > eps:
#                 Fn, step_failed = self.flux_func(Qi, Qj, n_normal, **kwargs)
#                 NCn = self.nc_func(Qi, Qj, n_normal, **kwargs)
#                 dQelem = (
#                     dt
#                     / self.mesh.element_volume[elem]
#                     * edge_length
#                     * (Fn + NCn).flatten()
#                 )
#                 dQelem = dQelem[:, np.newaxis]
#                 dQelem = self.model.compute_Q_in_x_y(dQelem, n_ij)
#                 dQ[:, elem] -= dQelem
#         return dQ

#     def loop_source(self, Q, **kwargs):
#         Qnew = np.array(Q)
#         if self.integrator.order == -1:
#             kwargs_source = {
#                 **kwargs,
#                 "func_jac": lambda t, Q, **kwargs: self.model.rhs_jacobian(
#                     t, Q, **kwargs
#                 ),
#             }
#         else:
#             kwargs_source = kwargs
#         for i_elem in range(self.mesh.n_elements):
#             kwargs_source_elem = kwargs_source.copy()
#             kwargs_source_elem["aux_variables"] = elementwise_aux_variables(
#                 i_elem, kwargs_source["aux_variables"]
#             )
#             if Qnew[0, i_elem] > 0:
#                 Qnew[:, i_elem] = self.integrator.evaluate(
#                     self.source_func,
#                     kwargs["time"],
#                     Qnew[:, i_elem][:, np.newaxis],
#                     **kwargs_source_elem,
#                 ).flatten()
#         return Qnew

#     def step_explicit_split_source_vectorized(self, Q, **kwargs):
#         Qnew = fix_negative_height(Q, self.model.yaml_tag)

#         dQ = self.loop_fluxes_vectorized(np.array(Qnew), **kwargs)
#         Qnew = self.loop_source(Qnew, **kwargs)
#         Qnew += dQ

#         step_failed = False
#         return Qnew, step_failed

#     def step_explicit_split_source_vectorized_coordinate_transform(self, Q, **kwargs):
#         # def work_on_chunk(index_list):
#         #     Qchunk = Q[:, index_list]
#         #     dQ = self.loop_fluxes_vectorized_coordinate_transform(np.array(Qchunk), **kwargs)
#         #     Qnew[:, index_list] += dQ
#         def get_Q_chunk(Q, i, n_chunks):
#             index_list = list(range(Q.shape[1]))
#             chunksize = int(len(index_list)/n_chunks)+1
#             indices = index_list[i*chunksize: (i+1)*chunksize]
#             return indices, Q[:, indices]

#         def write_result(result):
#             print(result)
#             dQ = result
#             Qnew[:, index_list] += dQ


#         Qnew = fix_negative_height(Q, self.model.yaml_tag)
#         n_chunks = 2
#         pool = Pool(5)
#         # pool.map(task, list(range(10)))
#         for i_chunk in range(n_chunks):
#         #     index_list, Q_chunk = get_Q_chunk(Q, i_chunk, n_chunks)
#             # pool.apply_async(self.loop_fluxes_vectorized_coordinate_transform, args = (Q_chunk), kwds=kwargs, callback=write_result)
#             result = pool.apply_async(task)
#             r =  result.get()
#             print(r)

#             # Qnew[:, index_list] += write_result()

#         pool.close()
#         pool.join()


#         # with Pool(2) as p:
#         #     Qchunk =
#         #     p.map(work_on_chunk,  index_list, chunksize = int(len(index_list)/2)+1 )
#         # dQ = self.loop_fluxes_vectorized_coordinate_transform(np.array(Qnew), **kwargs)
#         Qnew = self.loop_source(Qnew, **kwargs)
#         # Qnew += dQ

#         step_failed = False
#         return Qnew, step_failed


#     def step_explicit_split_source(self, Q, **kwargs):
#         Qnew = fix_negative_height(Q, self.model.yaml_tag)

#         dQinner = self.loop_inner_explicit(Qnew, **kwargs)
#         # dQboundary = self.loop_boundary_explicit(Qnew, **kwargs)
#         # TODO copying Qnew is a hack. There must be a problem in boundary_loop wher Qnew is overwritten
#         dQboundary = self.loop_boundary_explicit(np.array(Qnew), **kwargs)
#         Qnew = self.loop_source(Qnew, **kwargs)
#         Qnew += dQinner + dQboundary

#         step_failed = False
#         return Qnew, step_failed

#     def step_explicit_split_source_coordinate_transform(self, Q, **kwargs):
#         Qnew = fix_negative_height(Q, self.model.yaml_tag)

#         dQinner = self.loop_inner_explicit_coordinate_transform(Qnew, **kwargs)
#         dQboundary = self.loop_boundary_explicit_coordinate_transform(Qnew, **kwargs)
#         Qnew = self.loop_source(Qnew, **kwargs)
#         Qnew += dQinner + dQboundary

#         step_failed = False
#         return Qnew, step_failed


# def fix_negative_height(Q, model_tag):
#     Qnew = np.array(Q)
#     for i_elem in range(Q.shape[1]):
#         if Qnew[0, i_elem] < 0.0:
#             if "WithBottom" in model_tag:
#                 Qnew[:-1, i_elem] = np.zeros_like(Qnew[:-1, i_elem])
#             else:
#                 Qnew[:, i_elem] = np.zeros_like(Qnew[:, i_elem])
#     return Qnew


# def elementwise_aux_variables(elem, aux_field_dict):
#     aux_out = deepcopy(aux_field_dict)
#     for key, item in aux_out.items():
#         aux_out[key] = item[elem]
#     return aux_out

# def task():
#     print('hi:')
#     return 1.

# def callback_result(result):
#     print('callback')
#     print(result)
