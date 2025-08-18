# import numpy as np
# import os
# import logging


# from library.solver.baseclass import BaseYaml
# from library.solver.models.base import *
# from library.solver.boundary_conditions import *
# import library.solver.initial_condition as initial_condition
# import library.solver.smm_model as smm
# import library.solver.smm_model_hyperbolic as smmh
# import library.solver.smm_model_exner as smm_exner
# import library.solver.smm_model_exner_hyperbolic as smm_exner_hyper

# import sympy
# from sympy import Symbol, Matrix, lambdify
# from sympy import zeros, ones

# main_dir = os.getenv("SMPYTHON")


# class ShallowWater(Model):
#     yaml_tag = "!ShallowWater"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 2

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def flux(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         return np.array([hu, hu * u + self.g * self.ez * h * h / 2]).reshape(
#             2, 1, Q.shape[1]
#         )

#     def flux_jacobian(self, Q):
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         u = Q[1] / Q[0]
#         # dF1_dh=0
#         # dF1_dhu
#         out[0, 1, 0] = 1.0
#         # dF2_dh
#         out[1, 0, 0] = -u * u + self.g * h
#         # dF2_dhu
#         out[1, 1, 0] = 2 * u
#         return out

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         assert (h > 0).all()
#         c = np.sqrt(self.g * h)
#         return np.array([u - c, u + c]), imaginary

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         dHdx = kwargs["aux_variables"]["dHdx"]
#         h = Q[0]
#         output[1] = h * self.g * (self.ex - self.ez * dHdx)

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         # vectorized or single element?
#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))

#         # Topography
#         dHdx = kwargs["aux_variables"]["dHdx"]
#         h = Q[0]
#         # dR1_dh=0
#         # dR1_dhu=0
#         # dR2_dh
#         output[1, 0] = self.g * (self.ex - self.ez * dHdx)
#         # dR2_dhu=0

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction + "_jacobian")(t, Q, **kwargs)

#         return output

#     def newtonian(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / Q[0]
#         nu = self.parameters["nu"]
#         output[1] = -nu * u
#         return output

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         # vectorized?
#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         else:
#             output = np.zeros((Q.shape[0], Q.shape[0]))
#         nu = self.parameters["nu"]
#         h = Q[0]
#         u_inv = Q[0] / Q[1]
#         # dR1_dh=0
#         # dR1_dhu=0
#         # dR2_dh
#         output[1, 0] = nu * u_inv * u_inv
#         # dR2_dhu=0
#         output[1, 1] = -nu / h
#         return output

#     def manning(
#         self, t, Q, **kwargs
#     ):  # Manning Friction defined as per: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/wrcr.20366
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / Q[0]
#         nu = self.parameters["nu"]
#         output[1] = -self.g * (nu**2) * Q[1] * np.abs(Q[1]) / Q[0] ** (7 / 3)
#         return output

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         return np.array([h, u])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         return np.array([h, h * u])


# class ShallowWaterWithBottom(Model):
#     yaml_tag = "!ShallowWaterWithBottom"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 3

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def flux(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         return np.array([hu, hu * u + self.g * self.ez * h * h / 2, np.zeros_like(h)])

#     def flux_jacobian(self, Q):
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         # dF1_dh=0
#         # dF1_dhu
#         out[0, 1, 0] = 1.0
#         # dF1_dh_b = 0
#         # dF2_dh
#         out[1, 0, 0] = -u * u + self.g * h
#         # dF2_dhu
#         out[1, 1, 0] = 2 * u
#         # dF2_dh_b = 0
#         # dF3_dh =0
#         # dF3_du = 0
#         # dF3_dh_b =0
#         return out

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         h = Q[0]
#         hu = Q[1]
#         h = np.where(h <= 0, 0.0, h)
#         u = np.where(h <= 0, 0.0, hu / h)
#         # assert (h > 0).all()
#         c = np.sqrt(self.g * h)
#         return np.array([u - c, np.zeros_like(u), u + c]), imaginary

#     def nonconservative_matrix(self, Q, **kwargs):
#         result = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         h = np.where(h <= 0, 0.0, h)
#         result[1, 2] = -h * self.g * self.ez
#         return result

#     # def nonconservative_matrix(self, Q, **kwargs):
#     #     result = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#     #     h = Q[0]
#     #     h = np.where(h <= 0, 0.0, h)
#     #     result[1, 2] = -h * self.g * self.ez
#     #     return result

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         h = Q[0]
#         h = np.where(h <= 0, 0.0, h)
#         output[1] = h * self.g * self.ex

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         # h = np.where(h <= 0.0, 0.0, h)
#         # u = np.where(h <= 0.0, 0.0, hu / h)

#         # Topography
#         out[1, 0, 0] = self.g * self.ex
#         # Friction
#         for friction in self.friction_models:
#             out += getattr(self, friction + "_jacobian")(t, Q, **kwargs)

#         return out

#     def newtonian(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         hu = Q[1]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         nu = self.parameters["nu"]
#         output[1] = -nu * u
#         return output

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         nu = self.parameters["nu"]
#         out[1, 0, 0] = +nu * hu / h / h
#         out[1, 1, 0] = -nu / h
#         return out

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         h_b = Q[2]
#         h = Q[0]
#         hu = Q[1]
#         h = np.where(h <= 0, 0.0, h)
#         u = np.where(h <= 0, 0.0, hu / h)
#         return np.array([h, u, h_b])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         h_b = U[2]
#         return np.array([h, h * u, h_b])


# class ShallowWaterWithBottom2d(Model2d):
#     yaml_tag = "!ShallowWaterWithBottom2d"
#     dimension = 2

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 4

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def flux(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         return np.arraw(
#             [
#                 [hu, hu * u + self.g * h * h / 2, hu * v, np.zeros_like(h)],
#                 [hv, hv * u, hv * v + self.g * h * h / 2, np.zeros_like(h)],
#             ]
#         ).swapaxes(0, 1)

#     def flux_jacobian(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         c = self.g * h
#         zero = np.zeros_like(h)
#         one = np.ones_like(h)
#         A_x = np.array(
#             [
#                 [zero, one, zero, zero],
#                 [-(u**2) + c, 2 * u, zero, zero],
#                 [-u * v, v, u, zero],
#                 [zero, zero, zero, zero],
#             ]
#         )
#         A_y = np.array(
#             [
#                 [zero, zero, one, zero],
#                 [-u * v, v, u, zero],
#                 [-(v**2) + c, zero, 2 * v, zero],
#                 [zero, zero, zero, zero],
#             ]
#         )
#         A = np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))
#         A[:, :, 0, :] = A_x
#         A[:, :, 1, :] = A_y
#         return A

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         un = u * nij[0] + v * nij[1]
#         c = np.sqrt(self.g * h)
#         return np.array([un - c, np.zeros_like(h), un + c]), imaginary

#     def eigensystem(self, Q, nij):
#         imaginary = False
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         c = np.sqrt(self.g * h)
#         zero = np.zeros_like(h)
#         one = np.ones_like(h)
#         eps = 10 ** (-18) * one
#         ev_x = np.array([zero, u, u - c, u + c])
#         R_x = np.array(
#             [
#                 [-(c**2), zero, one, one],
#                 [zero, zero, -c + u, c + u],
#                 [-(c**2) * v, one, v, v],
#                 [c**2 - u**2, zero, zero, zero],
#             ]
#         )
#         iR_x = np.array(
#             [
#                 [zero, zero, zero, 1.0 / (c**2 - u**2 + eps)],
#                 [-v, zero, one, zero],
#                 [
#                     (c + u) / (2 * c + eps),
#                     -1.0 / (2 * c + eps),
#                     zero,
#                     c / (2 * c - 2 * u + eps),
#                 ],
#                 [
#                     (c - u) / (2 * c + eps),
#                     1.0 / (2 * c + eps),
#                     zero,
#                     c / (2 * (c + u + eps)),
#                 ],
#             ]
#         )
#         ev_y = np.array([zero, v, v - c, v + c])
#         R_y = np.array(
#             [
#                 [-(c**2), zero, -one, one],
#                 [-(c**2) * u, one, -u, u],
#                 [zero, zero, c - v, c + v],
#                 [c**2 - v**2, zero, zero, zero],
#             ]
#         )
#         iR_y = np.array(
#             [
#                 [zero, zero, zero, 1.0 / (c**2 - v**2 + eps)],
#                 [-u, one, zero, zero],
#                 [
#                     -(c + v) / (2 * c + eps),
#                     zero,
#                     1.0 / (2 * c + eps),
#                     -c / (2 * c - 2 * v + eps),
#                 ],
#                 [
#                     (c - v) / (2 * c + eps),
#                     zero,
#                     1.0 / (2 * c + eps),
#                     c / (2 * (c + v) + eps),
#                 ],
#             ]
#         )
#         ev = np.zeros((Q.shape[0], 2, Q.shape[1]))
#         R = np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))
#         iR = np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))
#         ev[:, 0, :] = ev_x
#         ev[:, 1, :] = ev_y
#         R[:, :, 0, :] = R_x
#         R[:, :, 1, :] = R_y
#         iR[:, :, 0, :] = iR_x
#         iR[:, :, 1, :] = iR_y

#         ev = np.einsum("ik..., k... -> i...", ev, nij)
#         R = np.einsum("ijk..., k... -> ij...", R, nij)
#         iR = np.einsum("ijk..., k... -> ij...", iR, nij)
#         A_rec = R[:, :, 0] @ np.diag(ev[:, 0]) @ iR[:, :, 0]
#         err = np.linalg.norm(
#             np.einsum("ijk..., k... -> ij...", self.quasilinear_matrix(Q), nij)[:, :, 0]
#             - A_rec
#         )
#         return ev, R, iR, err

#     def nonconservative_matrix(self, Q, **kwargs):
#         result = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         h = Q[0]
#         h = np.where(h <= 0, 0.0, h)
#         result[1, 3, 0] = -h * self.g * self.ez
#         result[2, 3, 1] = -h * self.g * self.ez
#         return result

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         h = Q[0]
#         h = np.where(h <= 0, 0.0, h)
#         output[1] = h * self.g * self.ex
#         output[2] = h * self.g * self.ey

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         # h = np.where(h <= 0.0, 0.0, h)
#         # u = np.where(h <= 0.0, 0.0, hu / h)

#         # Topography
#         out[1, 0] = self.g * self.ex
#         out[2, 0] = self.g * self.ey
#         # Friction
#         for friction in self.friction_models:
#             out += getattr(self, friction + "_jacobian")(t, Q, **kwargs)

#         return out

#     def newtonian(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         nu = self.parameters["nu"]
#         output[1] = -nu * u
#         output[2] = -nu * v
#         return output

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         nu = self.parameters["nu"]
#         out[1, 0] = +nu * hu / h / h
#         out[1, 1] = -nu / h
#         out[2, 0] = +nu * hv / h / h
#         out[2, 2] = -nu / h
#         return out

#     def bc_chezy(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         # C = kwargs["aux_variables"]["ChezyCoef"]
#         C = kwargs["model"].parameters["ChezyCoef"]
#         u_sq = np.sqrt(u**2 + v**2)
#         output[1] = -1.0 / C**2 * u * u_sq
#         output[2] = -1.0 / C**2 * v * u_sq
#         return output

#     def bc_chezy_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         C = kwargs["model"].parameters["ChezyCoef"]
#         # C = kwargs["aux_variables"]["ChezyCoef"]
#         u_sq = np.sqrt(u**2 + v**2)
#         eps = 10 ** (-10)
#         out[1, 0] = +1.0 / C**2 * 2 * u * u_sq / h
#         out[1, 1] = -1.0 / C**2 * (u / h / (u_sq + eps) * u + u_sq / h)
#         out[1, 2] = -1.0 / C**2 * (u / h / (u_sq + eps) * v)
#         out[2, 0] = +1.0 / C**2 * 2 * v * u_sq / h
#         out[2, 1] = -1.0 / C**2 * (v / h / (u_sq + eps) * u)
#         out[2, 2] = -1.0 / C**2 * (v / h / (u_sq + eps) * v + u_sq / h)
#         return out

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         nu = self.parameters["nu"]
#         out[1, 0] = +nu * hu / h / h
#         out[1, 1] = -nu / h
#         out[2, 0] = +nu * hv / h / h
#         out[2, 2] = -nu / h
#         return out

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         h_b = Q[3]
#         h = np.where(h <= 0.0, 0.0, h)
#         u = np.where(h <= 0.0, 0.0, hu / h)
#         v = np.where(h <= 0.0, 0.0, hv / h)
#         return np.array([h, u, v, h_b])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         v = U[2]
#         h_b = U[3]
#         return np.array([h, h * u, h * v, h_b])

#     def compute_Q_in_normal_transverse(self, Q_, n_, **kwargs):
#         assert Q_.shape[1] == 1
#         Q = np.array(Q_[:, 0])
#         n = np.array(n_[:, 0])
#         dim = self.dimension
#         offset = 1
#         assert dim == 2
#         result = np.array(Q)
#         mom = np.zeros((offset, dim))
#         for d in range(dim):
#             mom[:, d] = Q[1 + d * offset : 1 + (d + 1) * offset]
#         mom_normal = np.einsum("ik... ,k... -> i...", mom, n[:dim])
#         n_trans = np.cross(np.array([n[0], n[1], 0.0]), np.array([0.0, 0.0, 1.0]))
#         mom_trans = np.einsum("ik... ,k... -> i...", mom, n_trans[:dim])
#         result[1 : 1 + offset] = mom_normal
#         result[1 + offset : 1 + 2 * offset] = mom_trans
#         return result

#     def compute_Q_in_x_y(self, Q_, n_, **kwargs):
#         assert Q_.shape[1] == 1
#         Q = np.array(Q_[:, 0])
#         n = np.array(n_[:, 0])
#         n = np.array([n[0], n[1], 0.0])
#         dim = self.dimension
#         offset = 1
#         assert dim == 2
#         result = np.array(Q)
#         mom = np.zeros((offset, dim))
#         mom_normal = Q[1 : 1 + offset]
#         mom_trans = Q[1 + offset : 1 + 2 * offset]
#         n_trans = np.cross(n, np.array([0.0, 0.0, 1.0]))
#         nx = np.array([1.0, 0.0, 0.0])
#         ny = np.array([0.0, 1.0, 0.0])
#         result[1 : 1 + offset] = mom_normal * np.dot(n, nx)
#         result[1 : 1 + offset] += mom_trans * np.dot(n_trans, nx)
#         result[1 + offset : 1 + 2 * offset] = mom_normal * np.dot(n, ny)
#         result[1 + offset : 1 + 2 * offset] += mom_trans * np.dot(n_trans, ny)
#         return result


# class ShallowWater2d(Model2d):
#     yaml_tag = "!ShallowWater2d"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 3

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def flux(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         u = hu / h
#         v = hv / h
#         return np.array(
#             [
#                 [hu, hu * u + self.g * h * h / 2, hu * v],
#                 [hv, hv * u, hv * v + self.g * h * h / 2],
#             ]
#         ).swapaxes(0, 1)

#     def flux_jacobian(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         u = hu / h
#         v = hv / h

#         result = np.zeros((3, 3, 2))
#         dfdq = np.array([[0, 1, 0], [-(u**2) + self.g * h, 2 * u, 0], [-u * v, v, u]])
#         dgdq = np.array([[0, 0, 1], [-u * v, v, u], [-(v**2) + self.g * h, 0, 2 * v]])
#         result[:, :, 0] = dfdq
#         result[:, :, 1] = dgdq
#         return result

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         # evs = np.zeros_like(Q)
#         # for i in range(Q.shape[1]):
#         #     evs[:,i] = np.linalg.eigvals(np.dot(self.flux_jacobian(Q[:,i]), nij.flatten()))
#         # return evs, imaginary
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         u = hu / h
#         v = hv / h
#         un = u + v
#         assert (h > 0).all()
#         c = np.sqrt(self.g * h)
#         return np.array([un - c, un, un + c]), imaginary

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         h = Q[0]
#         output[1] = h * self.g * self.ex
#         output[2] = h * self.g * self.ey

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]

#         # Topography
#         out[1, 0] = self.g * self.ex
#         out[2, 0] = self.g * self.ey
#         # Friction
#         for friction in self.friction_models:
#             out += getattr(self, friction + "_jacobian")(t, Q, **kwargs)

#         return out

#     def newtonian(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / h
#         v = Q[2] / h
#         nu = self.parameters["nu"]
#         output[1] = -nu * u
#         output[2] = -nu * v
#         return output

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         nu = self.parameters["nu"]
#         out[1, 0] = +nu * hu / h / h
#         out[1, 1] = -nu / h
#         out[2, 0] = +nu * hv / h / h
#         out[2, 2] = -nu / h
#         return out

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         u = hu / h
#         v = hv / h
#         return np.array([h, u, v])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         v = U[2]
#         return np.array([h, h * u, h * v])


# class ShallowWaterSympy(Model):
#     yaml_tag = "!ShallowWaterSympy"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 2

#     def set_runtime_variables(self):
#         super().set_runtime_variables()
#         self.initialize_sympy_model()

#     def initialize_sympy_model(self):
#         numEquations = 1
#         h = sympy.symbols("h")
#         halpha = Matrix([sympy.symbols("halpha%d" % i) for i in range(numEquations)])
#         Q = Matrix([h, *(halpha)])

#         def f():
#             flux = Matrix([0 for i in range(Q.shape[0])])
#             h = Q[0]
#             hu = Q[1]
#             flux[0] = hu
#             flux[1] = hu**2 / h + self.g * self.ez * h * h / 2
#             return flux

#         def jac_f():
#             return f().jacobian(Q)

#         def rhs():
#             output = Matrix([0 for i in range(Q.shape[0])])
#             h = Q[0]
#             for friction in self.friction_models:
#                 output += getattr(self, friction)
#             return output

#         def jac_rhs():
#             return rhs().jacobian(Q)

#         def newtonian():
#             result = flux = Matrix([0 for i in range(Q.shape[0])])
#             h = Q[0]
#             hu = Q[1]
#             nu = self.parameters["nu"]
#             result[1] = -nu * hu / h
#             return result

#         self.EVs = simplify(jac_f().eigenvals())
#         self.F = lambdify(Q, f(), "numpy")
#         self.JacF = lambdify(Q, jac_f(), "numpy")
#         self.rhs = lambdify(Q, rhs(), "numpy")
#         self.Jac_rhs = lambdify(Q, jac_rhs(), "numpy")

#     def flux(self, Q):
#         return self.F(*Q)

#     def flux_jacobian(self, Q):
#         return self.JacF(*Q)

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         assert (h > 0).all()
#         c = np.sqrt(self.g * h)
#         return np.array([u - c, u + c]), imaginary

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         dHdx = kwargs["aux_variables"]["dHdx"]
#         h = Q[0]
#         output[1] = h * self.g * (self.ex - self.ez * dHdx)

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         # vectorized or single element?
#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))

#         # Topography
#         dHdx = kwargs["aux_variables"]["dHdx"]
#         h = Q[0]
#         # dR1_dh=0
#         # dR1_dhu=0
#         # dR2_dh
#         output[1, 0] = self.g * (self.ex - self.ez * dHdx)
#         # dR2_dhu=0

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction + "_jacobian")(t, Q, **kwargs)

#         return output

#     def newtonian(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / Q[0]
#         nu = self.parameters["nu"]
#         output[1] = -nu * u
#         return output

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         # vectorized?
#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         else:
#             output = np.zeros((Q.shape[0], Q.shape[0]))
#         nu = self.parameters["nu"]
#         h = Q[0]
#         u_inv = Q[0] / Q[1]
#         # dR1_dh=0
#         # dR1_dhu=0
#         # dR2_dh
#         output[1, 0] = nu * u_inv * u_inv
#         # dR2_dhu=0
#         output[1, 1] = -nu / h
#         return output

#     def manning(
#         self, t, Q, **kwargs
#     ):  # Manning Friction defined as per: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/wrcr.20366
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / Q[0]
#         nu = self.parameters["nu"]
#         output[1] = -self.g * (nu**2) * Q[1] * np.abs(Q[1]) / Q[0] ** (7 / 3)
#         return output

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         return np.array([h, u])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         return np.array([h, h * u])


# class ShallowWaterExner(Model):
#     yaml_tag = "!ShallowWaterExner"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 3
#         self.parameters = {
#             "sediment_density": 1580,
#             "water_density": 1000,
#             "sediment_dia": 0.0039,
#             "critical_shield": 0.047,
#             "manning": 0.0365,
#             "porosity": 0.47,
#         }

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def characteristic_discharge(self):
#         return self.parameters["sediment_dia"] * np.sqrt(
#             self.g
#             * self.parameters["sediment_dia"]
#             * (
#                 self.parameters["sediment_density"] / self.parameters["water_density"]
#                 - 1
#             )
#         )

#     def flux(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h

#         q = self.characteristic_discharge()
#         S_f = (self.parameters["manning"] ** 2) * u * np.abs(u) / h ** (4 / 3)
#         tau = self.parameters["water_density"] * self.g * h * S_f
#         theta = (
#             np.abs(tau)
#             * (self.parameters["sediment_dia"] ** 2)
#             / (
#                 self.g
#                 * (
#                     self.parameters["sediment_density"]
#                     - self.parameters["water_density"]
#                 )
#                 * self.parameters["sediment_dia"] ** 3
#             )
#         )

#         theta_flag = np.array(theta >= self.parameters["critical_shield"], dtype=float)
#         delta = (theta_flag * (theta - self.parameters["critical_shield"])) ** 1.5

#         # delta = np.abs(theta - self.critical_shield)**1.5

#         q_b = q * np.sign(tau) * (8 / (1 - self.parameters["porosity"])) * delta

#         return np.array([hu, hu * u + self.g * h * h / 2, q_b])

#     def flux_jacobian(self, Q):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         h = Q[0]
#         u = Q[1] / Q[0]
#         S_f = (self.parameters["manning"] ** 2) * u * np.abs(u) / h ** (4 / 3)
#         tau = self.parameters["water_density"] * self.g * h * S_f
#         theta = (
#             np.abs(tau)
#             * (self.parameters["sediment_dia"] ** 2)
#             / (
#                 self.g
#                 * (
#                     self.parameters["sediment_density"]
#                     - self.parameters["water_density"]
#                 )
#                 * self.parameters["sediment_dia"] ** 3
#             )
#         )
#         q = self.characteristic_discharge()

#         theta_flag = np.array(theta >= self.parameters["critical_shield"], dtype=float)
#         delta = (theta_flag * (theta - self.parameters["critical_shield"])) ** 0.5

#         # delta = np.abs(theta - self.critical_shield)**0.5

#         dq = (
#             q
#             * 24
#             * (self.parameters["manning"] ** 2)
#             * delta
#             * u
#             / (
#                 (1 - self.parameters["porosity"])
#                 * (
#                     (
#                         self.parameters["sediment_density"]
#                         / self.parameters["water_density"]
#                     )
#                     - 1
#                 )
#                 * self.parameters["sediment_dia"]
#                 * (h ** (4 / 3))
#             )
#         )
#         dh = (-7 / 6) * u * dq

#         out[0, 1] = 1
#         out[1, 0] = self.g * h - u**2
#         out[1, 1] = 2 * u
#         # out[1,2] = self.g * h
#         out[2, 0] = dh
#         out[2, 1] = dq

#         return out

#     def nonconservative_matrix(self, Q, **kwargs):
#         h = Q[0]
#         NC = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         NC[1, 2] = -self.g * h
#         return NC

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         h = Q[0]
#         u = Q[1] / Q[0]
#         output[1] = (
#             -self.g * self.parameters["manning"] ** 2 * np.abs(u) * u / h ** (1 / 3)
#         )

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         h = Q[0]
#         u = Q[1] / Q[0]

#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         else:
#             output = np.zeros((Q.shape[0], Q.shape[0]))

#         output[1, 0] = (
#             (7 / 3)
#             * self.g
#             * (self.parameters["manning"] ** 2)
#             * u
#             * np.abs(u)
#             / h ** (4 / 3)
#         )
#         output[1, 1] = (
#             -self.g
#             * (self.parameters["manning"] ** 2)
#             * (np.abs(u) + (u * u / np.abs(u)))
#             / h ** (4 / 3)
#         )

#         return output

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         b = Q[2]
#         return np.array([h, u, b])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         b = U[2]
#         return np.array([h, h * u, b])
