# import sys
# import os
# import numpy as np
# import logging
# import matplotlib.pyplot as plt

# from library.solver.reconstruction import (
#     compute_gradient,
#     compute_gradient_field_2d,
#     reconstruct_uvw,
# )

# main_dir = os.getenv("SMPYTHON")


# def get_callback_function_list(callback_name_list):
#     output = []
#     for callback in callback_name_list:
#         output.append(getattr(sys.modules[__name__], callback))
#     return output


# def controller_template(cls, Q, **kwargs):
#     return Q, kwargs


# def controller_init_topo_const(cls, Q, **kwargs):
#     mesh = kwargs["mesh"].element_centers
#     H = np.zeros((mesh.shape[0]))
#     kwargs["aux_variables"].update({"H": H, "dHdx": H, "dHdy": H})
#     return Q, kwargs


# def controller_init_topo_bump(cls, Q, **kwargs):
#     mesh = kwargs["mesh"].element_centers
#     x0 = mesh[0, 0]
#     x1 = mesh[-1, 0]
#     to_TC_domain = lambda x: 8 * x - 4
#     to_unity = lambda x: (x - x0) / (x1 - x0)
#     t = lambda x, y: to_TC_domain(to_unity(x))
#     dtdx = lambda x, y: 8 / (x1 - x0)
#     dtdy = lambda x, y: 0.0
#     func = lambda x, y: 0.3 * np.exp(-3 * t(x, y) ** 2)
#     dfuncdx = (
#         lambda x, y: 0.3 * (-3 * 2 * t(x, y)) * dtdx(x, y) * np.exp(-3 * t(x, y) ** 2)
#     )
#     dfuncdy = lambda x, y: 0.0 * np.ones_like(x)
#     out = np.zeros(mesh.shape[0])
#     H = func(mesh[:, 0], mesh[:, 1])
#     dHdx = dfuncdx(mesh[:, 0], mesh[:, 1])
#     dHdy = dfuncdy(mesh[:, 0], mesh[:, 1])
#     kwargs["aux_variables"].update({"H": H, "dHdx": dHdx, "dHdy": dHdy})
#     return Q, kwargs


# # def controller_init_point_data(cls, Q, **kwargs):
# #     n_variables = Q.shape[0]
# #     point_data = {}
# #     data = np.zeros((kwargs["mesh"].element_vertices.shape[0]))
# #     for i in n_variables:
# #         point_data.update({str(i): data})
# #     kwargs["point_data"].update({"point_data": point_data})
# #     return Q, kwargs


# def controller_compute_point_data(cls, Q, **kwargs):
#     n_variables = Q.shape[0]
#     area = np.zeros((kwargs["mesh"].element_vertices.max() + 1))
#     point_data = {}
#     for i in range(n_variables):
#         point_data.update({str(i): np.zeros_like(area)})

#     for i_elem in range(cls.mesh.n_elements):
#         for i_vertex in cls.mesh.element_vertices[i_elem]:
#             area[i_vertex] += cls.mesh.element_volume[i_elem]
#             for i in range(n_variables):
#                 point_data[str(i)][i_vertex] += (
#                     cls.mesh.element_volume[i_elem] * Q[i, i_elem]
#                 )
#     for i_bp, i_elem in enumerate(cls.mesh.boundary_edge_element):
#         Qj = cls.solver.bc_func(i_bp, Q, **kwargs)
#         for i_vertex in cls.mesh.boundary_edge_vertices[i_bp]:
#             area[i_vertex] += cls.mesh.element_volume[i_elem]
#             for i in range(n_variables):
#                 point_data[str(i)][i_vertex] += cls.mesh.element_volume[i_elem] * Qj[i]
#     for field in point_data.values():
#         field /= area
#     cls.callback_default_parameters.update({"point_data": point_data})
#     filename = (
#         main_dir + "/" + cls.output_dir + "/out." + str(kwargs["iteration"]) + ".vtk"
#     )
#     cls.mesh.write_to_file(
#         filename,
#         fields=Q,
#         point_data=point_data,
#     )
#     return Q, kwargs


# def controller_compute_vertical_velocity_pointwise(cls, Q, **kwargs):
#     n_variables = Q.shape[0]
#     try:
#         level = cls.model.level
#     except:
#         level = 0
#     point_data = cls.callback_default_parameters["point_data"]
#     # convert point data to array
#     points = cls.mesh.vertex_coordinates
#     values = np.zeros((Q.shape[0], points.shape[0]))
#     for i, q in point_data.items():
#         values[int(i)] = q

#     # compute gradients
#     grad = compute_gradient_field_2d(points[:, :2], values)

#     # evaluate H, U, V, W
#     N = 10
#     dz = 1.0 / (N - 1)
#     Z = np.linspace(0, 1, N)
#     H = values[0]
#     U = np.zeros((N * values.shape[1]))
#     V = np.zeros((N * values.shape[1]))
#     W = np.zeros((N * values.shape[1]))
#     n_points = values.shape[1]
#     n_vertices = n_points
#     points_3d = np.zeros(
#         (
#             points.shape[0] * (N),
#             3,
#         )
#     )
#     UVW = np.zeros((values.shape[1] * (N), 3))

#     if "scale_3d_height" in cls.callback_default_parameters:
#         scale_z = cls.callback_default_parameters["scale_3d_height"]
#     else:
#         scale_z = 1.0

#     for i in range(n_vertices):
#         points_3d[i * N : (i + 1) * N, :] = cls.mesh.vertex_coordinates[i]
#         points_3d[i * N : (i + 1) * N, 2] = H[i] * scale_z * Z
#         u, v, w = reconstruct_uvw(
#             values[:, i], grad[:, :, i], level, kwargs["model"].matrices
#         )
#         UVW[i * N : (i + 1) * N, 0] = u(Z)
#         UVW[i * N : (i + 1) * N, 1] = v(Z)
#         UVW[i * N : (i + 1) * N, 2] = w(Z)

#     # compute connectivity for mesh (element_vertices)
#     n_elements = cls.mesh.element_vertices.shape[0]
#     element_vertices_3d = np.zeros(
#         (n_elements * (N - 1), cls.mesh.num_nodes_per_element * 2)
#     )
#     for i_layer in range(N - 1):
#         element_vertices_3d[
#             i_layer * n_elements : (i_layer + 1) * n_elements,
#             : cls.mesh.num_nodes_per_element,
#         ] = i_layer + cls.mesh.element_vertices * (N)
#         element_vertices_3d[
#             i_layer * n_elements : (i_layer + 1) * n_elements,
#             cls.mesh.num_nodes_per_element :,
#         ] = (
#             i_layer + 1 + cls.mesh.element_vertices * (N)
#         )

#     # write to file
#     point_fields = UVW.T
#     filename = (
#         main_dir + "/" + cls.output_dir + "/out3d." + str(kwargs["iteration"]) + ".vtk"
#     )
#     cls.mesh.write_to_file_3d(
#         filename, points_3d, element_vertices_3d, point_fields=point_fields
#     )

#     return Q, kwargs


# def controller_compute_vertical_velocity(cls, Q, **kwargs):
#     n_variables = Q.shape[0]
#     try:
#         level = cls.model.level
#     except:
#         level = 0
#     point_data = cls.callback_default_parameters["point_data"]
#     # convert point data to array
#     points = np.zeros((Q.shape[1], cls.mesh.num_nodes_per_element, 3))
#     values = np.zeros((Q.shape[1], cls.mesh.num_nodes_per_element, n_variables))
#     p0 = np.zeros((Q.shape[1], 3))
#     v0 = np.zeros((Q.shape[1], n_variables))
#     for i in range(n_variables):
#         for i_elem in range(cls.mesh.n_elements):
#             for i_vertex, vertex in enumerate(cls.mesh.element_vertices[i_elem]):
#                 values[i_elem, i_vertex, i] = point_data[str(i)][vertex]
#     p0 = cls.mesh.element_centers
#     v0 = Q.T
#     for i_elem in range(cls.mesh.n_elements):
#         for i in range(n_variables):
#             for i_vertex, vertex in enumerate(cls.mesh.element_vertices[i_elem]):
#                 values[i_elem, i_vertex, i] = point_data[str(i)][vertex]
#                 points[i_elem, i_vertex] = cls.mesh.vertex_coordinates[vertex, :]
#     # compute gradient fields
#     grad = np.zeros((n_variables, 2, Q.shape[1]))
#     for i_elem in range(cls.mesh.n_elements):
#         grad[:, :, i_elem] = compute_gradient(
#             points[i_elem], values[i_elem], p0[i_elem], v0[i_elem]
#         )

#     # evaluate H, U, V, W
#     N = 10
#     dz = 1.0 / N
#     Z = np.arange(dz / 2, 1, dz)
#     Zp = np.linspace(0, 1, N + 1)
#     H = point_data["0"]
#     U = np.zeros((N * Q.shape[1]))
#     V = np.zeros((N * Q.shape[1]))
#     W = np.zeros((N * Q.shape[1]))
#     n_elements = Q.shape[1]
#     n_vertices = cls.mesh.vertex_coordinates.shape[0]
#     vertex_coords_3d = np.zeros(
#         (
#             cls.mesh.vertex_coordinates.shape[0] * (N + 1),
#             cls.mesh.vertex_coordinates.shape[1],
#         )
#     )
#     element_vertices_3d = np.zeros(
#         (cls.mesh.element_vertices.shape[0] * N, cls.mesh.element_vertices.shape[1] * 2)
#     )
#     element_midpoints_3d = np.zeros((cls.mesh.element_vertices.shape[0] * N, 3))
#     for i_layer in range(N + 1):
#         for i in range(n_vertices):
#             vertex_coords_3d[i_layer * n_vertices + i, :] = cls.mesh.vertex_coordinates[
#                 i
#             ]
#             vertex_coords_3d[i_layer * n_vertices + i, 2] = (
#                 H[i] * cls.callback_default_parameters["scale_3d_height"] * Zp[i_layer]
#             )
#     for i_layer in range(N):
#         for i in range(n_elements):
#             element_vertices_3d[
#                 i_layer * n_elements + i, : cls.mesh.num_nodes_per_element
#             ] = (cls.mesh.element_vertices[i] + i_layer * n_vertices)
#             element_vertices_3d[
#                 i_layer * n_elements + i, cls.mesh.num_nodes_per_element :
#             ] = (cls.mesh.element_vertices[i] + (i_layer + 1) * n_vertices)
#             u, v, w = reconstruct_uvw(
#                 Q[:, i], grad[:, :, i], level, kwargs["model"].matrices
#             )
#             U[i_layer * n_elements + i] = u(Z[i_layer])
#             V[i_layer * n_elements + i] = v(Z[i_layer])
#             W[i_layer * n_elements + i] = w(Z[i_layer])
#             element_midpoints_3d[
#                 i_layer * n_elements + i, :
#             ] = cls.mesh.element_centers[i, :]
#             element_midpoints_3d[i_layer * n_elements + i, 2] = 0.5 * (
#                 Zp[i_layer] + Zp[i_layer + 1]
#             )

#     # create streamline plots
#     if cls.callback_default_parameters["streamlines"]:
#         positions = cls.callback_default_parameters["streamlines_config"]["positions"]
#         normals = cls.callback_default_parameters["streamlines_config"]["normals"]
#         for position, normal in zip(positions, normals):
#             # get elements from positions
#             # get from vertex_coordinates
#             xmin = position[0]
#             xmax = position[1]
#             ymin = position[2]
#             ymax = position[3]
#             vertex_indices = list(range(element_vertices_3d.shape[0]))
#             element_centers = element_midpoints_3d
#             selection_x = np.logical_and(
#                 element_centers[:, 0] >= xmin, element_centers[:, 0] <= xmax
#             )
#             selection_y = np.logical_and(
#                 element_centers[:, 1] >= ymin, element_centers[:, 1] <= ymax
#             )
#             selection = np.logical_and(selection_x, selection_y)
#             assert selection.sum() > 0
#             indices_selection = np.array(vertex_indices)[selection]
#             vertices_selection = element_centers[selection, :]
#             # sort by x, y, z
#             sort_order = np.lexsort(
#                 (
#                     vertices_selection[:, 2],
#                     vertices_selection[:, 1],
#                     vertices_selection[:, 0],
#                 )
#             )
#             indices_selection = indices_selection[sort_order]
#             vertices_selection = vertices_selection[sort_order, :]

#             U_selection = U[indices_selection]
#             V_selection = V[indices_selection]
#             W_selection = W[indices_selection]
#             # project U, V along normal direction
#             normal_normal = np.cross(normal, [0, 0, 1])
#             UVn_selection = (
#                 U_selection * normal_normal[0] + V_selection * normal_normal[1]
#             )
#             # create streamline plot using UVn and W
#             speed = np.sqrt(UVn_selection**2 + W_selection**2)
#             fig, axes = plt.subplots(
#                 1, 1, sharex=False, sharey=False, constrained_layout=True
#             )
#             N_z = N
#             N_xy = int(W_selection.shape[0] / N_z)
#             assert N_xy * N_z == W_selection.shape[0]
#             # X, Y = np.meshgrid(np.linspace(0, 1, N_xy), np.linspace(0, 1, N_z))
#             X = (
#                 vertices_selection[:, 0] * normal_normal[0]
#                 + vertices_selection[:, 1] * normal_normal[1]
#             )
#             Y = vertices_selection[:, 2]
#             out_X = np.zeros((N_z, N_xy))
#             out_Y = np.zeros((N_z, N_xy))
#             out_U = np.zeros((N_z, N_xy))
#             out_V = np.zeros((N_z, N_xy))
#             out_color = np.zeros((N_z, N_xy))
#             # for row in range(N_z):
#             #     for col in range(N_xy):
#             #         out_X[row, col] = X[col * N_xy + row]
#             #         out_Y[row, col] = Y[col * N_xy + row]
#             #         out_U[row, col] = UVn_selection[col * N_xy + row]
#             #         out_V[row, col] = W_selection[col * N_xy + row]
#             #         out_color[row, col] = speed[col * N_xy + row]
#             # out_X = X.reshape((N_xy, N_z)).T
#             # out_Y = Y.reshape((N_xy, N_z)).T
#             out_X, out_Y = np.meshgrid(np.linspace(0, 1, N_xy), np.linspace(0, 1, N_z))
#             out_U = UVn_selection.reshape((N_xy, N_z)).T
#             out_V = W_selection.reshape((N_xy, N_z)).T
#             out_color = speed.reshape((N_xy, N_z)).T
#             strm = axes.streamplot(
#                 out_X,
#                 out_Y,
#                 out_U,
#                 out_V,
#                 color=out_color,
#                 broken_streamlines=True,
#             )
#             plt.show()
#         # save plot using iteration number

#     # write to file
#     fields = np.array([U, V, W])
#     filename = (
#         main_dir + "/" + cls.output_dir + "/out3d." + str(kwargs["iteration"]) + ".vtk"
#     )
#     cls.mesh.write_to_file_3d(
#         filename, vertex_coords_3d, element_vertices_3d, fields=fields
#     )

#     return Q, kwargs


# def controller_timedependent_topography(cls, Q, **kwargs):
#     mesh = kwargs["mesh"].element_centers
#     time = kwargs["time"]
#     # compute new values
#     H = 0.1 * (np.sin(mesh[:, 0] * np.pi + time * 2 * np.pi))
#     dHdx = 0.1 * (np.cos(mesh[:, 0] * np.pi + time * 2 * np.pi))
#     dHdy = np.zeros_like(H)

#     # update output
#     kwargs["aux_variables"].update({"H": H, "dHdx": dHdx, "dHdy": dHdy})
#     return Q, kwargs


# def controller_clip_small_moments(cls, Q, **kwargs):
#     dim = kwargs["model"].dimension
#     n_variables = kwargs["model"].n_variables
#     scalar_fields = 1
#     if "WithBottom" in kwargs["model"].yaml_tag:
#         scalar_fields += 1
#     level = int((n_variables - scalar_fields) / dim - 1)
#     offset = level + 1
#     assert (level + 1) * dim + scalar_fields == n_variables

#     Q[1 : 1 + dim * offset] = np.where(
#         np.abs(Q[1 : 1 + dim * offset]) < 10 ** (-8), 0, Q[1 : 1 + dim * offset]
#     )
#     return Q, kwargs


# def controller_damp_oszillations_at_stiffler_inlet(cls, Q, **kwargs):
#     dim = kwargs["model"].dimension
#     n_variables = kwargs["model"].n_variables
#     scalar_fields = 1
#     if "WithBottom" in kwargs["model"].yaml_tag:
#         scalar_fields += 1
#     level = int((n_variables - scalar_fields) / dim - 1)
#     offset = level + 1
#     assert (level + 1) * dim + scalar_fields == n_variables

#     # find out relevant elements
#     filtered_elements = kwargs["mesh"].element_centers[:, 0] > 4.2

#     # clip vertical velocities
#     # Q[2,filtered_elements] = 0.

#     # relaxation
#     Q[2, filtered_elements] = 0.5 * (
#         Q[2, filtered_elements] + kwargs["Qold"][2, filtered_elements]
#     )

#     return Q, kwargs


# def controller_filter_elements_for_L2_error(cls, Q, **kwargs):
#     dim = kwargs["model"].dimension
#     n_variables = kwargs["model"].n_variables
#     scalar_fields = 1
#     if "WithBottom" in kwargs["model"].yaml_tag:
#         scalar_fields += 1
#     level = int((n_variables - scalar_fields) / dim - 1)
#     offset = level + 1
#     assert (level + 1) * dim + scalar_fields == n_variables

#     # find out relevant elements
#     filtered_elements = np.array(kwargs["mesh"].element_centers[:, 0] < 4.3, dtype=bool)

#     kwargs.update({"filtered_elements": filtered_elements})

#     return Q, kwargs
