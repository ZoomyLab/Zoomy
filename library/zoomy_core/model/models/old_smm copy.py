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
# from sympy import *
# from sympy import zeros, ones

# main_dir = os.getenv("SMPYTHON")


# class ShallowMomentReference2d(Model2d):
#     yaml_tag = "!ShallowMomentReference2d"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 2

#     def set_runtime_variables(self):
#         super().set_runtime_variables()

#     def flux(self, Q, **kwargs):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         hum = kwargs["aux_variables"]["hu_mean"]
#         hvm = kwargs["aux_variables"]["hv_mean"]
#         result = np.zeros((2, 2, h.shape[0]))
#         result[0, 0] = hum
#         result[0, 1] = hu * u + self.g * h * h / 2
#         return result

#     def flux_jacobian(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         hv = Q[2]
#         u = hu / h
#         v = hv / h

#         result = np.zeros((2, 2, 2, h.shape[0]))
#         result[0, 1, 0] = 1
#         result[1, 0, 0] = -(u**2) + self.g * h
#         result[1, 1, 0] = 2 * u
#         return result

#     def primitive_variables(self, Q):
#         h = Q[0]
#         hu = Q[1]
#         u = hu / h
#         return np.array([h, u])

#     def conservative_variables(self, U):
#         h = U[0]
#         u = U[1]
#         return np.array([h, h * u])

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)

#         # vertical coupling

#         # Topography

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(
#                 t,
#                 Q,
#                 **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#             )

#         return output

#     def update_hum_and_omega(self, Q, N, M, **kwargs):
#         U = self.primitive_variables(Q)
#         U_zz = np.zeros(U.shape[1])
#         omega = np.zeros(U.shape[1])
#         Umean = np.zeros(U.shape[1])
#         Umean_dict = {}
#         vol_dict = {}

#         assert kwargs["mesh"]["type"] == "quad"
#         for i_elem in range(kwargs_in["mesh"]["n_elements"]):
#             Ui = np.array(U[:, i_elem])
#             U_zz[i_elem] += Ui
#             xpos = kwargs["mesh"]["element_centers"][i_elem][0]
#             vol = kwargs["mesh"]["element_volume"][i_elem]
#             Umean_dict[xpos] += Ui * vol
#             vol_dict[xpos] += vol
#             for i_edge, _ in enumerate(kwargs_in["mesh"]["element_neighbors"][i_elem]):
#                 i_neighbor = (kwargs_in["mesh"]["element_neighbors"][i_elem])[i_edge]
#                 n_ij = kwargs_in["mesh"]["element_edge_normal"][i_elem, i_edge, :][
#                     :, np.newaxis
#                 ][: kwargs_in["mesh"]["dim"]]
#                 # reshape to [dim,1]
#                 Uj = np.array(U[:, i_neighbor])
#                 Fn, step_failed = kwargs_in["flux"](Qi, Qj, n_ij, **kwargs)
#                 NCn = kwargs_in["nc"](Qi, Qj, n_ij, **kwargs)
#                 Qnew[:, i_elem] -= (
#                     dt
#                     / kwargs_in["mesh"]["element_volume"][i_elem]
#                     * kwargs_in["mesh"]["element_edge_length"][i_elem, i_edge]
#                     * (Fn + NCn).flatten()
#                 )
#             S = kwargs_in["source"](kwargs_in["time"], Qi, **kwargs).flatten()
#             Qnew[:, i_elem] += (
#                 dt * kwargs_in["mesh"]["element_volume"][i_elem] * S.flatten()
#             )

#         for i_bp, elem in enumerate(kwargs_in["mesh"]["boundary_edge_element"]):
#             kwargs["aux_variables"] = elementwise_aux_variables(i_elem, **kwargs_in)
#             kwargs["i_elem"] = elem
#             # TODO WRONG!!
#             kwargs["i_edge"] = i_bp
#             n_ij = kwargs_in["mesh"]["boundary_edge_normal"][i_bp][
#                 : kwargs_in["mesh"]["dim"]
#             ][:, np.newaxis]
#             kwargs["i_neighbor"] = elem

#             edge_length = kwargs_in["mesh"]["boundary_edge_length"][i_bp]

#             Qi = np.array(Q[:, elem])[:, np.newaxis]
#             Qj = kwargs_in["bc"](
#                 kwargs_in["model"].boundary_conditions,
#                 i_bp,
#                 Q,
#                 dim=kwargs_in["mesh"]["dim"],
#             )[:, np.newaxis]
#             Fn, step_failed = kwargs_in["flux"](Qi, Qj, n_ij, **kwargs)
#             NCn = kwargs_in["nc"](Qi, Qj, n_ij, **kwargs)
#             Qnew[:, elem] -= (
#                 dt
#                 / kwargs_in["mesh"]["element_volume"][elem]
#                 * edge_length
#                 * (Fn + NCn).flatten()
#             )


# class ShallowMoments(Model):
#     yaml_tag = "!ShallowMoments"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 1 + 2 * self.dimension
#         self.basis = "legendre"
#         self.bc_coupling_type = "weak"
#         self.nc_treatment = "flux_increment"

#     def set_runtime_variables(self):
#         super().set_runtime_variables()
#         self.level = int((self.n_variables - 1) / self.dimension) - 1
#         # TODO it is not nice that the parameters exist in both!
#         self.matrices = smm.Matrices(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def flux(self, Q):
#         kwargs = {"g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         return self.matrices.flux(Q, **kwargs)

#     def flux_jacobian(self, Q):
#         kwargs = {"g": self.g, "ex": self.ex, "ez": self.ez}
#         return self.matrices.flux_jac(Q, **kwargs)

#     # def eigenvalues(self, Q, nij):
#     #     assert (Q[0] > 0).all()
#     #     kwargs = {"g": self.g, "ez": self.ez}
#     #     return self.matrices.eigenvalues(Q, nij, **kwargs)

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         output += self.matrices.rhs_topo(
#             Q, **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         )

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(
#                 t,
#                 Q,
#                 **{
#                     **kwargs,
#                     "g": self.g,
#                     "ex": self.ex,
#                     "ey": self.ey,
#                     "ez": self.ez,
#                     **self.parameters,
#                 }
#             )

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))

#         # Topography
#         output += self.matrices.rhs_topo_jacobian(
#             Q, **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         )

#         # Friction
#         for friction in self.friction_models:
#             # print(friction, getattr(self, friction + "_jacobian")(
#             #     t,
#             #     Q,
#             #     **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#             # ).shape)
#             output += getattr(self, friction + "_jacobian")(
#                 t,
#                 Q,
#                 **{
#                     **kwargs,
#                     "g": self.g,
#                     "ex": self.ex,
#                     "ey": self.ey,
#                     "ez": self.ez,
#                     **self.parameters,
#                 }
#             )

#         return output

#     def nonconservative_matrix(self, Q, **kwargs):
#         kwargs_in = {"g": self.g, "ez": self.ez}
#         return self.matrices.nonconservative_matrix(Q, **kwargs_in)

#     def quasilinear_matrix(self, Q, **kwargs):
#         return self.flux_jacobian(Q) - ShallowMoments.nonconservative_matrix(
#             self, Q, **kwargs
#         )

#     def newtonian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian(Q, **kwargs)

#     def bingham(self, t, Q, **kwargs):
#         return self.matrices.rhs_bingham(Q, **kwargs)

#     def bingham_rowdependent(self, t, Q, **kwargs):
#         return self.matrices.rhs_bingham_rowdependent(Q, **kwargs)

#     def bingham_depthaveraged(self, t, Q, **kwargs):
#         return self.matrices.rhs_bingham_depthaveraged(Q, **kwargs)

#     def bingham_bottom(self, t, Q, **kwargs):
#         return self.matrices.rhs_bingham_bottom(Q, **kwargs)

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian_jacobian(Q, **kwargs)

#     def primitive_variables(self, Q):
#         U = np.array(Q)
#         U[1:] = np.where(U[0] > 0, U[1:] / U[0], 0.0)
#         return U

#     def conservative_variables(self, U):
#         Q = np.array(U)
#         Q[1:] *= Q[0]
#         return Q

#     def get_massmatrix(self):
#         return self.matrices.MM

#     def get_massmatrix_inverse(self):
#         return self.matrices.Minv


# class ShallowMomentsHyperbolic(Model):
#     yaml_tag = "!ShallowMomentsHyperbolic"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 1 + 2 * self.dimension
#         self.basis = "legendre"
#         self.bc_coupling_type = "weak"
#         self.nc_treatment = "flux_increment"
#         self.friction_models = []
#         self.parameters = {}

#     def set_runtime_variables(self):
#         super().set_runtime_variables()
#         self.level = int((self.n_variables - 1) / self.dimension) - 1
#         # TODO it is not nice that the parameters exist in both!
#         self.matrices = smmh.Matrices(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def flux(self, Q):
#         kwargs = {"g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         return self.matrices.flux(Q, **kwargs)

#     def flux_jacobian(self, Q):
#         kwargs = {"g": self.g, "ex": self.ex, "ez": self.ez}
#         return self.matrices.flux_jac(Q, **kwargs)

#     def eigenvalues(self, Q, nij):
#         assert (Q[0] > 0).all()
#         kwargs = {"g": self.g, "ez": self.ez}
#         return self.matrices.eigenvalues_hyperbolic(Q, nij, **kwargs)

#     def eigenvalues_analytical(self, Q, nij):
#         assert (Q[0] > 0).all()
#         kwargs = {"g": self.g, "ez": self.ez}
#         return self.matrices.eigenvalues_hyperbolic_analytical(Q, nij, **kwargs)

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         output += self.matrices.rhs_topo(
#             Q, **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         )

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(
#                 t,
#                 Q,
#                 **{**kwargs, "g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#             )

#         return output

#     def nonconservative_matrix(self, Q):
#         kwargs = {"g": self.g, "ex": self.ex, "ey": self.ey, "ez": self.ez}
#         return self.matrices.nonconservative_matrix_hyperbolic(Q, **kwargs)

#     def quasilinear_matrix(self, Q):
#         Q[3:, :] = 0.0
#         return self.flux_jacobian(Q) - self.nonconservative_matrix(Q)

#     def newtonian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian(Q, **kwargs)

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian_jacobian(Q, **kwargs)

#     def primitive_variables(self, Q):
#         U = np.array(Q)
#         U[1:] /= Q[0]
#         return U

#     def conservative_variables(self, U):
#         Q = np.array(U)
#         Q[1:] *= Q[0]
#         return Q

#     def get_massmatrix(self):
#         return self.matrices.MM

#     def get_massmatrix_inverse(self):
#         return self.matrices.Minv


# class ShallowMomentsExner(Model):
#     yaml_tag = "!ShallowMomentsExner"

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()
#         self.n_variables = 3 + 2
#         self.basis = "legendre"
#         self.bc_coupling_type = "weak"
#         self.nc_treatment = "flux_increment"
#         self.friction_models = []
#         self.friction_parameters = {}

#         self.sediment_density = 1580
#         self.water_density = 1000
#         self.sediment_dia = 0.0039
#         self.critical_shield = 0.047
#         self.manning = 0.0365
#         self.porosity = 0.47

#     def set_runtime_variables(self):
#         super().set_runtime_variables()
#         self.level = int((self.n_variables - 1) / self.dimension) - 2
#         # TODO it is not nice that the parameters exist in both!
#         self.matrices = smm_exner.Matrices_exner(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def flux(self, Q):
#         kwargs = {
#             "g": self.g,
#             "ex": self.ex,
#             "ez": self.ez,
#             "sediment_density": self.sediment_density,
#             "water_density": self.water_density,
#             "sediment_dia": self.sediment_dia,
#             "critical_shield": self.critical_shield,
#             "manning": self.manning,
#             "porosity": self.porosity,
#         }
#         return self.matrices.flux(Q, **kwargs)

#     def flux_jacobian(self, Q):
#         kwargs = {
#             "g": self.g,
#             "ex": self.ex,
#             "ez": self.ez,
#             "sediment_density": self.sediment_density,
#             "water_density": self.water_density,
#             "sediment_dia": self.sediment_dia,
#             "critical_shield": self.critical_shield,
#             "manning": self.manning,
#             "porosity": self.porosity,
#         }
#         return self.matrices.flux_jac(Q, **kwargs)

#     def eigenvalues(self, Q, nij):
#         assert (Q[0] > 0).all()
#         kwargs = {
#             "g": self.g,
#             "ex": self.ex,
#             "ez": self.ez,
#             "sediment_density": self.sediment_density,
#             "water_density": self.water_density,
#             "sediment_dia": self.sediment_dia,
#             "critical_shield": self.critical_shield,
#             "manning": self.manning,
#             "porosity": self.porosity,
#         }
#         return self.matrices.eigenvalues(Q, nij, **kwargs)

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         # Topography
#         output += self.matrices.rhs_topo(
#             Q, **{**kwargs, "g": self.g, "ex": self.ex, "ez": self.ez}
#         )

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(
#                 t,
#                 Q,
#                 **{
#                     **kwargs,
#                     "g": self.g,
#                     "ex": self.ex,
#                     "ez": self.ez,
#                     "manning": self.manning,
#                     **self.friction_parameters,
#                 }
#             )

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         if len(Q.shape) == 2:
#             output = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         else:
#             output = np.zeros((Q.shape[0], Q.shape[0]))

#         # Topography
#         if "dHdx" in kwargs:
#             output += self.matrices.rhs_topo_jacobian(
#                 Q, **{**kwargs, "g": self.g, "ex": self.ex, "ez": self.ez}
#             )

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction + "_jacobian")(
#                 t,
#                 Q,
#                 **{
#                     **kwargs,
#                     "g": self.g,
#                     "ex": self.ex,
#                     "ez": self.ez,
#                     **self.friction_parameters,
#                 }
#             )

#         return output

#     def nonconservative_matrix(self, Q):
#         kwargs = {"g": self.g}
#         return self.matrices.nonconservative_matrix(Q, **kwargs)

#     def quasilinear_matrix(self, Q):
#         return self.flux_jacobian(Q) - self.nonconservative_matrix(Q)

#     def newtonian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian(Q, **kwargs)

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian_jacobian(Q, **kwargs)

#     def primitive_variables(self, Q):
#         U = np.array(Q)
#         U[1:-1] /= Q[0]
#         return U

#     def conservative_variables(self, U):
#         Q = np.array(U)
#         Q[1:-1] *= Q[0]
#         return Q

#     def get_default_parameters(self):
#         return self.friction_parameters

#     def get_massmatrix(self):
#         return self.matrices.MM

#     def get_massmatrix_inverse(self):
#         return self.matrices.Minv


# class HyperbolicShallowMoments(ShallowMoments):
#     yaml_tag = "!HyperbolicShallowMoments"

#     def eigenvalues(self, Q, nij):
#         assert (Q[0] > 0).all()
#         kwargs = {"g": self.g, "ez": self.ez}
#         Qhyp = np.array(Q)
#         Qhyp[2:, :] = 0
#         return self.matrices.eigenvalues_hyperbolic_analytical(Qhyp, nij, **kwargs)

#     def quasilinear_matrix(self, Q, **kwargs):
#         Qhyp = np.array(Q)
#         Qhyp[2:, :] = 0
#         return self.flux_jacobian(Qhyp) - super().nonconservative_matrix(Qhyp, **kwargs)

#     def nonconservative_matrix(self, Q, **kwargs):
#         Qhyp = np.array(Q)
#         Qhyp[2:, :] = 0
#         return -self.quasilinear_matrix(Qhyp, **kwargs) + self.flux_jacobian(Q)


# class ShallowMomentsWithBottom(ShallowMoments):
#     yaml_tag = "!ShallowMomentsWithBottom"
#     dimension = 1

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()

#     def set_runtime_variables(self):
#         # I need to run the parent.parent.set_runtime_variables, since I need to call a different smm.Matrices !!
#         # since this is not possible? (super().super() does not exist), i call it call it directly
#         self.compute_unit_vectors()
#         self.level = int((self.n_variables - 2) / self.dimension) - 1

#         self.matrices = smm.MatricesWithBottom(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def primitive_variables(self, Q):
#         res = np.zeros_like(Q)
#         res[:-1] = super().primitive_variables(Q[:-1])
#         res[-1] = Q[-1]
#         return res

#     def conservative_variables(self, U):
#         res = np.zeros_like(U)
#         res[:-1] = super().conservative_variables(U[:-1])
#         res[-1] = U[-1]
#         return res

#     # def eigenvalues(self, Q, nij):
#     #     kwargs = {"g": self.g, "ez": self.ez, "ex": self.ex}
#     #     return self.matrices.eigenvalues(Q, nij, **kwargs)


# class ShallowMoments2d(ShallowMoments):
#     yaml_tag = "!ShallowMoments2d"

#     dimension = 2

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()

#     def set_runtime_variables(self):
#         # I need to run the parent.parent.set_runtime_variables, since I need to call a different smm.Matrices !!
#         # since this is not possible? (super().super() does not exist), i call it call it directly
#         self.compute_unit_vectors()
#         self.level = int((self.n_variables - 1) / self.dimension) - 1

#         self.matrices = smm.Matrices2d(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def get_alpha_beta(Q):
#         prim = self.primitive_variables(Q)
#         alphabeta = prim[1:]
#         alpha = alphabeta[: (self.level + 1) * self.dimension]
#         beta = alphabeta[(self.level + 1) * self.dimension :]
#         return alpha, beta

#     def bc_slip(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_slip(Q, **kwargs)

#     def bc_slip_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_slip_jacobian(Q, **kwargs)

#     def bc_newtonian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_newtonian(Q, **kwargs)

#     def bc_newtonian_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_newtonian_jacobian(Q, **kwargs)


# class ShallowMomentsWithBottom2d(ShallowMoments2d):
#     yaml_tag = "!ShallowMomentsWithBottom2d"
#     dimension = 2

#     def set_default_default_parameters(self):
#         super().set_default_default_parameters()

#     def set_runtime_variables(self):
#         # I need to run the parent.parent.set_runtime_variables, since I need to call a different smm.Matrices !!
#         # since this is not possible? (super().super() does not exist), i call it call it directly
#         self.compute_unit_vectors()
#         self.level = int((self.n_variables - 2) / self.dimension) - 1

#         self.matrices = smm.MatricesWithBottom2d(level=self.level, basistype=self.basis)
#         self.matrices.bc_type = self.bc_coupling_type
#         self.matrices.basistype = self.basis
#         self.matrices.set_runtime_variables()

#     def get_alpha_beta(Q):
#         return super().get_alpha_beta(Q[:-1])

#     def primitive_variables(self, Q):
#         res = np.zeros_like(Q)
#         res[:-1] = super().primitive_variables(Q[:-1])
#         res[-1] = Q[-1]
#         return res

#     def conservative_variables(self, U):
#         res = np.zeros_like(U)
#         res[:-1] = super().conservative_variables(U[:-1])
#         res[-1] = U[-1]
#         return res

#     def flux(self, Q):
#         res = np.zeros((Q.shape[0], self.dimension, Q.shape[1]))
#         kwargs = {"g": self.g, "ez": self.ez}
#         res = self.matrices.flux(Q, **kwargs)
#         offset = 1 + self.level
#         # h = np.where(Q[0] > 0, Q[0], 0)
#         # res[1 : 1 + offset, 0] -= (
#         #     1 / 2 * self.g * self.ez * np.einsum("k,...->k...", self.matrices.W, h**2)
#         # )
#         # res[1 + offset : 1 + 2 * offset, 1] -= (
#         #     1 / 2 * self.g * self.ez * np.einsum("k,...->k...", self.matrices.W, h**2)
#         # )
#         return res

#     def flux_jacobian(self, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#         kwargs_in = {"g": self.g, "ez": self.ez}
#         out = self.matrices.flux_jac(Q, **kwargs_in)
#         # offset = 1 + self.level
#         # h = np.where(Q[0] > 0, Q[0], 0)
#         # out[1 : 1 + offset, -1, 0] -= (
#         #     h * self.g * self.ez * self.matrices.W[:]
#         # ).reshape(out[1 : 1 + offset, -1, 0].shape)
#         # out[1 + offset : 1 + 2 * offset, -1, 1] = (
#         #     h * self.g * self.ez * self.matrices.W[:]
#         # ).reshape(out[1 : 1 + offset, -1, 0].shape)
#         return out

#     # # TODO HACK COMPATIBILITY
#     # def nonconservative_matrix(self, Q, **kwargs):
#     #     result = np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))
#     #     h = Q[0]
#     #     h = np.where(h <= 0, 0.0, h)
#     #     result[1, 3, 0] = -h * self.g * self.ez
#     #     result[2, 3, 1] = -h * self.g * self.ez
#     #     return result

#     def nonconservative_matrix(self, Q, **kwargs):
#         out = super().nonconservative_matrix(Q, **kwargs)
#         h = Q[0]
#         offset = 1 + self.level
#         # out[1 : 1 + offset, 0, 0] -= (
#         #     h * self.g * self.ez * self.matrices.W[:]
#         # ).reshape(out[1 : 1 + offset, -1, 0].shape)
#         # out[1 + offset : 1 + 2 * offset, 0, 1] -= (
#         #     h * self.g * self.ez * self.matrices.W[:]
#         # ).reshape(out[1 : 1 + offset, -1, 0].shape)
#         out[1 : 1 + offset, -1, 0, :] -= (
#             self.g * self.ez * np.outer(self.matrices.W[:], h)
#         ).reshape(out[1 : 1 + offset, -1, 0, :].shape)
#         out[1 + offset : 1 + 2 * offset, -1, 1, :] -= (
#             self.g * self.ez * np.outer(self.matrices.W[:], h)
#         ).reshape(out[1 : 1 + offset, -1, 0, :].shape)
#         return out

#     def rhs(self, t, Q_, **kwargs):
#         output = np.zeros_like(Q_)
#         # Topography
#         Q = np.array(Q_)
#         Q[:-1] = np.where(Q_[0] <= 0, np.zeros_like(Q_[:-1]), Q_[:-1])
#         h = Q[0]
#         offset = 1 + self.level
#         output[1 : 1 + offset] = (h * self.g * self.ex * self.matrices.W[:])[
#             :, np.newaxis
#         ]
#         output[1 + offset : 1 + 2 * offset] = (
#             h * self.g * self.ey * self.matrices.W[:]
#         )[:, np.newaxis]

#         # Friction
#         for friction in self.friction_models:
#             output += getattr(self, friction)(t, Q, **kwargs)

#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         out = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         # Topography
#         offset = 1 + self.level
#         out[1 : 1 + offset, 0] = (self.g * self.ex * self.matrices.W)[:, np.newaxis]
#         out[1 + offset : 1 + 2 * offset, 0] = (self.g * self.ey * self.matrices.W)[
#             :, np.newaxis
#         ]
#         # Friction
#         for friction in self.friction_models:
#             out += getattr(self, friction + "_jacobian")(t, Q, **kwargs)
#         return out

#     def bc_grad_slip(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_grad_slip(Q, **kwargs)

#     def bc_grad_slip_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_grad_slip_jacobian(Q, **kwargs)

#     # # TODO HACK COMPATIBILITY
#     # def bc_chezy(self, t, Q, **kwargs):
#     #     output = np.zeros_like(Q)
#     #     h = Q[0]
#     #     hu = Q[1]
#     #     hv = Q[2]
#     #     h = np.where(h <= 0.0, 0.0, h)
#     #     u = np.where(h <= 0.0, 0.0, hu / h)
#     #     v = np.where(h <= 0.0, 0.0, hv / h)
#     #     # C = kwargs["aux_variables"]["ChezyCoef"]
#     #     C = kwargs["model"].parameters["ChezyCoef"]
#     #     u_sq = np.sqrt(u**2 + v**2)
#     #     output[1] = -1.0 / C**2 * u * u_sq
#     #     output[2] = -1.0 / C**2 * v * u_sq
#     #     return output

#     # TODO HACK COMPATIBILITY
#     def bc_chezy_jacobian_swe(self, t, Q, **kwargs):
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

#     def bc_chezy(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_chezy(Q, **kwargs)

#     def bc_chezy_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_chezy_jacobian(Q, **kwargs)

#     def bc_slip(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_slip(Q, **kwargs)

#     def bc_slip_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_bc_slip_jacobian(Q, **kwargs)

#     def newtonian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian(Q, **kwargs)

#     def newtonian_jacobian(self, t, Q, **kwargs):
#         return self.matrices.rhs_newtonian_jacobian(Q, **kwargs)

#     def compute_Q_in_normal_transverse(self, Q_, n_, **kwargs):
#         assert Q_.shape[1] == 1
#         Q = np.array(Q_[:, 0])
#         n = np.array(n_[:, 0])
#         dim = self.dimension
#         offset = self.level + 1
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
#         offset = self.level + 1
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


# class ShallowMomentsHyperbolicWithBottom2d(ShallowMomentsWithBottom2d):
#     yaml_tag = "!ShallowMomentsHyperbolicWithBottom2d"
#     dimension = 2

#     def flux(self, Q):
#         res = np.zeros((Q.shape[0], self.dimension, Q.shape[1]))
#         kwargs = {"g": self.g, "ez": self.ez}
#         offset = 1 + self.level
#         res = self.matrices.flux(Q, **kwargs)
#         return res

#     def nonconservative_matrix(self, Q, **kwargs):
#         offset = 1 + self.level
#         Qlin = np.array(Q)
#         Qlin[1 + 1 : 1 + offset] = 0.0
#         Qlin[1 + 1 + offset : 1 + 2 * offset] = 0.0
#         out = super().flux_jacobian(Q, **kwargs) - super().quasilinear_matrix(
#             Qlin, **kwargs
#         )
#         return out

#     def quasilinear_matrix(self, Q, **kwargs):
#         offset = 1 + self.level
#         Qlin = np.array(Q)
#         Qlin[1 + 1 : 1 + offset] = 0.0
#         Qlin[1 + 1 + offset : 1 + 2 * offset] = 0.0
#         out = super().quasilinear_matrix(Qlin, **kwargs)
#         return out
