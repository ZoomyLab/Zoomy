import numpy as np
from sympy import integrate, diff
from sympy.abc import x
from sympy import lambdify

import zoomy_core.mesh.mesh_util as mesh_util
# from zoomy_core.model.models.shallow_moments import Basis


def _find_bounding_element(mesh, position):
    """
    Strategy: the faces of the elements are outward facing. If I compute compte the intersection of the normal with the point,
    resulting in  alpha * normal = minimal_distance; then the alpha needs to be negative for all faces
    see https://en.wikipedia.org/wiki/Hesse_normal_form
    """
    mesh_type = mesh.type
    for i_elem, vertices in enumerate(mesh.element_vertices):
        faces = mesh_util._face_order(vertices, mesh_type)
        face_centers = [
            mesh_util.center(mesh.vertex_coordinates, np.array(face)) for face in faces
        ]
        vector_origin_to_plane = [
            face_center - position for face_center in face_centers
        ]
        face_normals = mesh.element_face_normals[i_elem]

        if _is_point_inside_bounding_faces(face_normals, vector_origin_to_plane):
            return i_elem

    # outside of domain
    assert False


def _is_point_inside_bounding_faces(outward_face_normals, vectors_OP):
    for n, p in zip(outward_face_normals, vectors_OP):
        if np.dot(n, p) < 0:
            return False
    return True


def to_new_mesh(fields, mesh_old, mesh_new, interp="const", map_fields=None):
    assert interp == "const"

    fields_new = np.zeros_like(fields)

    for i_elem in range(mesh_new.n_elements):
        element_center = mesh_new.element_center[i_elem]
        i_elem_old = _find_bounding_element(mesh_old, element_center)
        fields_new[i_elem] = fields[i_elem_old]
    return fields_new


# # comute gradients based on FD using scattered pointwise data
# def compute_gradient_field_2d(points, fields):
#     def in_hull(points, probe):
#         n_points = points.shape[0]
#         n_dim = points.shape[1]
#         c = np.zeros(n_points)
#         A = np.r_[points.T, np.ones((1, n_points))]
#         b = np.r_[probe, np.ones(1)]
#         lp = linprog(c, A_eq=A, b_eq=b)
#         return lp.success

#     assert points.shape[1] == 2
#     grad = np.zeros((fields.shape[0], 2, fields.shape[1]))
#     eps_x = (points[:, 0].max() - points[:, 0].min()) / 100.0
#     eps_y = (points[:, 1].max() - points[:, 1].min()) / 100.00

#     # generate evaluation 'stencil' for central differences
#     xi_0 = np.array(points)
#     xi_xp = np.array(points)
#     xi_xp[:, 0] += eps_x
#     xi_xm = np.array(points)
#     xi_xm[:, 0] -= eps_x
#     xi_yp = np.array(points)
#     xi_yp[:, 1] += eps_y
#     xi_ym = np.array(points)
#     xi_ym[:, 1] -= eps_y
#     factors_x = 2.0 * np.ones((points.shape[0]))
#     factors_y = 2.0 * np.ones((points.shape[0]))
#     # correct boundary points with single sided differences
#     for i in range(xi_xp.shape[0]):
#         if not in_hull(points, xi_xp[i]):
#             xi_xp[i, 0] -= eps_x
#             factors_x[i] = 1.0
#         if not in_hull(points, xi_xm[i]):
#             xi_xm[i, 0] += eps_x
#             factors_x[i] = 1.0
#         if not in_hull(points, xi_yp[i]):
#             xi_yp[i, 1] -= eps_y
#             factors_y[i] = 1.0
#         if not in_hull(points, xi_ym[i]):
#             xi_ym[i, 1] += eps_y
#             factors_y[i] = 1.0

#     for i_field, values in enumerate(fields):
#         f = griddata(points, values, xi_0)
#         f_xp = griddata(points, values, xi_xp)
#         f_xm = griddata(points, values, xi_xm)
#         f_yp = griddata(points, values, xi_yp)
#         f_ym = griddata(points, values, xi_ym)

#         dfdx = (f_xp - f_xm) / (factors_x * eps_x + 10 ** (-10))
#         dfdy = (f_yp - f_ym) / (factors_y * eps_y + 10 ** (-10))

#         grad[i_field, 0, :] = dfdx
#         grad[i_field, 1, :] = dfdy

#     assert (np.isnan(grad) == False).all()
#     assert (np.isfinite(grad) == True).all()
#     return grad


# # ps, vs: values at the boundary points
# # p0, v0, value at the cell_center
# def compute_gradient(ps, vs, p0, v0, limiter=lambda r: 1.0):
#     points = np.zeros((ps.shape[0] + 1, 2))
#     points[:-1, :] = ps[:, :2]
#     points[-1, :] = p0[:2]
#     values = np.zeros((vs.shape[0] + 1, vs.shape[1]))
#     values[:-1, :] = vs
#     values[-1, :] = v0

#     f = LinearNDInterpolator(points, values)
#     eps_x = (points[:, 0].max() - points[:, 0].min()) / 100.0
#     eps_y = (points[:, 1].max() - points[:, 1].min()) / 100.00
#     x0 = p0[0]
#     y0 = p0[1]

#     dfdx = (f(x0 + eps_x, y0) - f(x0 - eps_x, y0)) / (2 * eps_x + 10 ** (-10))
#     rx = (f(x0, y0) - f(x0 - eps_x, y0)) / (f(x0 + eps_x, y0) - f(x0, y0) + 10 ** (-10))
#     phix = limiter(rx)
#     dfdy = (f(x0, y0 + eps_y) - f(x0, y0 - eps_y)) / (2 * eps_y + 10 ** (-10))
#     ry = (f(x0, y0) - f(x0, y0 - eps_y)) / (f(x0, y0 + eps_y) - f(x0, y0) + 10 ** (-10))
#     phiy = limiter(ry)

#     grad = np.array([phix * dfdx, phiy * dfdy]).T
#     assert (np.isnan(grad) == False).all()
#     assert (np.isfinite(grad) == True).all()
#     return grad
