# import os
# import numpy as np
# import numpy.polynomial.legendre as L
# import numpy.polynomial.chebyshev as C
# import matplotlib.pyplot as plt
# import inspect
# import pytest
# from copy import deepcopy
# from scipy import interpolate

# from library.solver.baseclass import BaseYaml  # nopep8

# main_dir = os.getenv("SMPYTHON")
# eps = 10 ** (-10)


# # TODO outsource parameters into model?
# class Matrices(BaseYaml):
#     dim = 1

#     def set_default_parameters(self):
#         self.level = 2
#         self.basistype = "legendre"
#         self.bc_type = "weak"

#     def set_runtime_variables(self):
#         if self.bc_type == "strong":
#             assert self.level >= 2
#         if self.basistype == "legendre":
#             self.basis = lambda order: self.legendre_basis(order)
#         elif self.basistype == "chebyshev":
#             self.basis = lambda order: self.chebyshev_basis(order)
#         else:
#             print("EFluxROR: basis ", self.basistype, " not implemented. Abort!")
#             assert False

#         # initialze matrices
#         self.A = np.zeros((self.level + 1, self.level + 1, self.level + 1))
#         self.B = np.zeros((self.level + 1, self.level + 1, self.level + 1))
#         self.C = np.zeros((self.level + 1, self.level + 1))
#         self.M = np.zeros((self.level + 1, self.level + 1))
#         self.Minv = np.zeros(
#             (1 + (self.level + 1) * self.dim, 1 + (self.level + 1) * self.dim)
#         )
#         self.MM = np.zeros(
#             (1 + (self.level + 1) * self.dim, 1 + (self.level + 1) * self.dim)
#         )
#         self.W = np.zeros(self.level + 1)
#         self.Bottom = np.zeros(self.level + 1)
#         self.Bottom_deriv = np.zeros(self.level + 1)
#         self.Bottom_granular = np.zeros(self.level + 1)
#         self.StrongBottom = np.zeros((self.level + 1, self.level + 1))
#         self.StrongTop = np.zeros((self.level + 1, self.level + 1))
#         self.P = np.zeros((self.level + 1, self.level + 1))

#         # compute matrices
#         self.create_A()
#         self.create_B()
#         self.create_C()
#         self.create_W()
#         self.create_M()
#         self.create_Bottom()
#         self.create_Bottom_deriv()
#         self.create_Bottom_granular()
#         self.create_StrongBottom()
#         self.create_StrongTop()
#         self.create_P()
#         self.create_MM()
#         self.create_Minv()

#     def legendre_basis(self, order):
#         # Legendre basis rescaled to [0.,1.]
#         domain = [0.0, 1.0]
#         # Normalize by swapping window (window is the range of the orig. Leg. Poy
#         # Since I want the range where it is orthogonal, I use [-1, 1] or [1, -1] normalized.
#         window = [1.0, -1.0]
#         basis = L.Legendre.basis(order, domain, window)
#         return basis

#     def legendre_basis_integrate(self, func):
#         return func.integ(1, [], 0.0)

#     def legendre_basis_derivative(self, func):
#         return func.deriv(1)

#     def legendre_weight(self):
#         return lambda x: 1

#     def chebyshev_basis(self, order):
#         domain = [0.0, 1.0]
#         window = [-1.0, 1.0]
#         basis = C.Chebyshev.basis(order, domain, window)
#         return basis

#     def chebyshev_basis_integrate(self, func):
#         return func.integ(1, [], 0.0)

#     def chebyshev_basis_derivative(self, func):
#         return func.deriv(1)

#     def chebyshev_weight(self):
#         return lambda x: C.weight()(x)

#     def basis_integrate(self, func):
#         if self.basistype == "legendre":
#             return self.legendre_basis_integrate(func)
#         elif self.basistype == "chebyshev":
#             return self.chebyshev_basis_integrate(func)
#         else:
#             print("EFluxROR: basis ", self.basistype, " not implemented. Abort!")
#             assert False

#     def basis_derivative(self, func):
#         if self.basistype == "legendre":
#             return self.legendre_basis_derivative(func)
#         elif self.basistype == "chebyshev":
#             return self.chebyshev_basis_derivative(func)
#         else:
#             print("EFluxROR: basis ", self.basistype, " not implemented. Abort!")
#             assert False

#     def basis_weight(self):
#         if self.basistype == "legendre":
#             return lambda x: self.legendre_weight()(x)
#         elif self.basistype == "chebyshev":
#             return lambda x: self.chebyshev_weight(func)(x)
#         else:
#             print("EFluxROR: basis ", self.basistype, " not implemented. Abort!")
#             assert False

#     def A_calc(self, i, j, k):
#         # Define basis functions
#         phi_i = self.basis(i)
#         phi_j = self.basis(j)
#         phi_k = self.basis(k)

#         # compute Aijk according to (B.1) in original shallow moments paper
#         phimult = phi_i * phi_j * phi_k
#         phi_int = self.basis_integrate(phimult)
#         return phi_int(1.0)

#     def create_A(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for j in range(0, self.level + 1):
#                 for k in range(0, self.level + 1):
#                     self.A[i, j, k] = self.A_calc(i, j, k)

#     def B_calc(self, i, j, k):
#         # Define basis functions
#         phi_i = self.basis(i)
#         phi_j = self.basis(j)
#         phi_k = self.basis(k)

#         # compute Bijk according to (B.4) in original shallow moments paper
#         phi_k_div = self.basis_derivative(phi_k)
#         phi_j_int = self.basis_integrate(phi_j)
#         phimult = phi_i * phi_j_int * phi_k_div
#         phi_int = self.basis_integrate(phimult)
#         value = phi_int(1.0)
#         return value

#     def create_B(self):
#         for i in range(0, self.level + 1):
#             for j in range(0, self.level + 1):
#                 for k in range(0, self.level + 1):
#                     self.B[i, j, k] = self.B_calc(i, j, k)

#     def C_calc(self, i, j):
#         # Define basis
#         phi_i = self.basis(i)
#         phi_j = self.basis(j)

#         # compute Cijk according to (B.6) in original shallow moments paper
#         phi_i_div = self.basis_derivative(phi_i)
#         phi_j_div = self.basis_derivative(phi_j)
#         phimult = phi_i_div * phi_j_div
#         phi_int = self.basis_integrate(phimult)
#         return phi_int(1.0)

#     def create_C(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for j in range(0, self.level + 1):
#                 self.C[i, j] = self.C_calc(i, j)

#     def W_calc(self, k):
#         # Define basis
#         phi_i = self.basis(k)
#         # compute W_k
#         phi_int = self.basis_integrate(phi_i)
#         return phi_int(1.0)

#     def create_W(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             self.W[i] = self.W_calc(i)

#     def StrongBottom_calc(self, i, k):
#         # Define basis
#         phi_i = self.basis(i)
#         phi_k = self.basis(k)
#         phi_i_deriv = self.basis_derivative(phi_i)
#         return phi_k(0) * phi_i_deriv(0)

#     def StrongTop_calc(self, i, k):
#         # Define basis
#         phi_i = self.basis(i)
#         phi_k = self.basis(k)
#         phi_i_deriv = self.basis_derivative(phi_i)
#         return phi_k(1) * phi_i_deriv(1)

#     def create_StrongBottom(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for k in range(0, self.level + 1):
#                 self.StrongBottom[i, k] = self.StrongBottom_calc(i, k)

#     def create_StrongTop(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for k in range(0, self.level + 1):
#                 self.StrongTop[i, k] = self.StrongTop_calc(i, k)

#     def Bottom_calc(self, k):
#         # Define basis
#         phi_i = self.basis(k)
#         return phi_i(0)

#     def create_Bottom(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             self.Bottom[i] = self.Bottom_calc(i)

#     def Bottom_deriv_calc(self, k):
#         # Define basis
#         phi_i = self.basis_derivative(self.basis(k))
#         return phi_i(0)

#     def create_Bottom_deriv(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             self.Bottom_deriv[i] = self.Bottom_deriv_calc(i)

#     def Bottom_granular_calc(self, k):
#         # Define basis
#         phi_i = self.basis_derivative(self.basis(k))
#         return phi_i(0)

#     def create_Bottom_granular(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             self.Bottom_granular[i] = self.Bottom_granular_calc(i)

#     def M_calc(self, i, j):
#         # Define basis
#         phi_i = self.basis(i)
#         phi_j = self.basis(j)
#         # compute M_k
#         phi_mult = phi_i * phi_j
#         phi_int = self.basis_integrate(phi_mult)
#         return phi_int(1.0)

#     def create_MM(self):
#         self.MM[0, 0] = 1
#         self.MM[1:, 1:] = self.M

#     def create_Minv(self):
#         self.Minv[0, 0] = 1
#         self.Minv[1:, 1:] = np.linalg.inv(self.M)

#     def create_M(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for j in range(0, self.level + 1):
#                 self.M[i, j] = self.M_calc(i, j)

#     def P_calc(self, i, j):
#         # Define basis
#         phi_i = self.basis(i)
#         phi_j = self.basis(i)
#         zeta = (self.basis(0) - self.basis(1)) / 2
#         # compute M_k
#         phi_i_deriv = self.basis_derivative(phi_i)
#         phi_mult = phi_i_deriv * phi_j * zeta
#         phi_int = self.basis_integrate(phi_mult)
#         return phi_int(1.0)

#     def create_P(self):
#         # populate and return array
#         for i in range(0, self.level + 1):
#             for j in range(0, self.level + 1):
#                 self.P[i, j] = self.P_calc(i, j)

#     # TODO legendre only -> rework
#     def flux(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         g = kwargs["g"]
#         ez = kwargs["ez"]
#         # initialize return array
#         F_value = np.zeros([2 + 1 * self.level, self.dim, Q.shape[1]])
#         h = Q[0]
#         alpha = np.where(Q[0] > 0, Q[1:] / Q[0], 0)

#         # # continuity equation
#         F_value[0, 0] = Q[1]

#         # # moments
#         # F_value[1] = h * (u**2 +  np.einsum('ij, i..., j...->...', self.A[1:,1:,0], alpha[1:], alpha[1:])) + 1/2*g*ez*h**2
#         # for k in range(2,Q.shape[0]):
#         #     F_value[k] = h*(2*u*alpha_unscaled[k-1] + np.einsum('ij, i..., j...->...', self.A[1:,1:,k-1], alpha[1:], alpha[1:]))

#         F_value[1:, 0] = np.einsum(
#             "ijk, i..., j...->k...", self.A, Q[1:], alpha
#         ) + 1 / 2 * g * ez * np.einsum("k,...->k...", self.W, h**2)

#         return F_value

#     def flux_jac(self, Q, multiply_by_Minv=True, **kwargs):
#         # There is no need to rescale Q, since
#         # dx(F(MinvQ)) = dF(MinvQ)/d(MinvQ) d(MinvQ)/dx = dF(Q)/dQ d(MinvQ)/dx
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         g = kwargs["g"]
#         ez = kwargs["ez"]

#         h = Q[0]
#         alpha = np.where(Q[0] > 0, Q[1:] / Q[0], 0)

#         J = np.zeros((self.level + 2, self.level + 2, self.dim, Q.shape[1]))
#         # dF0/dh = 0
#         # dF0/dhalpha_m
#         J[0, 1, 0] = 1
#         # dFk/dh
#         J[1:, 0, 0] = -np.einsum(
#             "ijk, i..., j...->k...", self.A, alpha, alpha
#         ) + np.einsum("k, ...->k...", g * ez * self.W, h)
#         # dFk/dalpha_m
#         J[1:, 1:, 0] = 2 * np.einsum("ijk, i...->jk...", self.A, alpha)

#         # Scale J in order to avoid rescaling dQdx!
#         if multiply_by_Minv:
#             return np.einsum("ki, kml...->iml...", self.Minv, J)
#         else:
#             return J

#     def rhs_nonconservative(self, Q, **kwargs):
#         Qunscaled = deepcopy(Q)
#         # Q = np.einsum('ij, j...->i...', self.Minv, Q)
#         dx = kwargs["dx"]
#         if len(Q.shape) > 1:
#             Ql = np.roll(Q, 1, axis=1)
#             Qr = np.roll(Q, -1, axis=1)
#             dQdx = (Qr - Ql) / 2 / dx
#         else:
#             dQdx = kwargs["dQdx"]
#         return np.einsum(
#             "ij..., j...->i...", self.nonconservative_matrix(Qunscaled, **kwargs), dQdx
#         )

#     def nonconservative_matrix(self, Q, multiply_by_Minv=True, **kwargs):
#         Q = np.einsum(
#             "ij, j...->i...", self.Minv[: self.level + 2, : self.level + 2], Q
#         )
#         alpha = np.where(Q[0] > 0, Q[1:] / Q[0], 0)

#         dim = self.dim

#         # using the derivation as done in the paper from Kowalski

#         if len(Q.shape) == 1:
#             NC = np.zeros((self.level + 2, self.level + 2, dim))
#         else:
#             NC = np.zeros((self.level + 2, self.level + 2, dim, Q.shape[1]))
#         NC[2:, 2:, 0] = np.einsum("ki, ...->ki...", self.M[1:, 1:], alpha[0])
#         # NC[2:,2:] =  np.einsum('ki, ...->ki...',np.eye(Q.shape[0]-1)[1:,1:], alpha[0])
#         # i j_int, k_div
#         NC[2:, 2:, 0] -= np.einsum("ijk, i...->kj...", self.B[1:, 1:, 1:], alpha[1:])

#         # TODO try out my own derivations for comparison
#         # NC[1:,1:] = np.einsum('ijk,k...->ij...', self.B + self.A, alpha)
#         # NC[1:,2:] = np.einsum('ijk,k...->ij...', self.B[:,1:,:], alpha)

#         # Scale NC in order to avoid rescaling dQdx!
#         # Note: Minv acts on dQdx -> Therefore it is multiplied to NC from the right
#         # return NC
#         # return np.einsum('km..., mi->ki...',NC,  self.Minv)
#         if multiply_by_Minv:
#             return np.einsum("kml..., mi->kil...", NC, self.Minv)
#         else:
#             return NC

#     def eigenvalues(self, Q, nij, **kwargs):
#         A = self.flux_jac(Q, multiply_by_Minv=False, **kwargs)
#         NC = self.nonconservative_matrix(Q, multiply_by_Minv=False, **kwargs)
#         EVs = np.zeros_like(Q)
#         imaginary = False
#         # TODO vectorize?
#         for i in range(Q.shape[1]):
#             # ev = np.linalg.eigvals(A[:,:,i]-NC[:,:,i])
#             ev = np.linalg.eigvals(
#                 np.einsum("ij, jk->ik", self.Minv, A[:, :, 0, i] - NC[:, :, 0, i])
#             )
#             assert np.isfinite(ev).all()
#             # assert(np.isreal(ev).all())
#             if not np.isreal(ev).all():
#                 imaginary = True
#                 # TODO enable with debug_level!
#                 # print('WARNING: imaginary eigenvalues: ', str(ev), ' for Q: ', Q[:,i])
#                 ev = 10**6
#             EVs[:, i] = ev
#             del ev
#         return EVs, imaginary

#     def eigenvalues_hyperbolic_analytical(self, Q, nij, **kwargs):
#         u_m = Q[1] / Q[0]
#         h = Q[0]
#         g = kwargs["g"]
#         alpha1 = Q[2] / Q[0]
#         imaginary = False

#         EVs1 = np.zeros_like(Q)

#         if self.level == 3:
#             EVs1[0, :] = u_m + np.sqrt(g * h + alpha1**2)
#             EVs1[1, :] = u_m - np.sqrt(g * h + alpha1**2)
#             EVs1[2, :] = u_m + np.sqrt(3 / 7) * alpha1
#             EVs1[3, :] = u_m
#             EVs1[4, :] = u_m - np.sqrt(3 / 7) * alpha1
#         elif self.level == 5:
#             EVs1[0, :] = u_m + np.sqrt(g * h + alpha1**2)
#             EVs1[1, :] = u_m - np.sqrt(g * h + alpha1**2)
#             EVs1[2, :] = u_m + np.sqrt((15 + 2 * np.sqrt(15)) / 33) * alpha1
#             EVs1[3, :] = u_m + np.sqrt((15 - 2 * np.sqrt(15)) / 33) * alpha1
#             EVs1[4, :] = u_m
#             EVs1[5, :] = u_m - np.sqrt((15 + 2 * np.sqrt(15)) / 33) * alpha1
#             EVs1[6, :] = u_m - np.sqrt((15 - 2 * np.sqrt(15)) / 33) * alpha1
#         else:
#             assert False

#         assert np.isfinite(EVs1).all()
#         if not np.isreal(EVs1).all():
#             imaginary = True

#         return EVs1, imaginary

#     def rhs_topo(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ez = kwargs["ez"]
#         dHdx = kwargs["aux_fields"]["dHdx"]
#         if len(Q.shape) == 1:
#             topo = np.zeros((self.level + 2))
#         else:
#             topo = np.zeros((self.level + 2, Q.shape[1]))
#         h = Q[0]
#         topo[1:] = np.einsum("..., i->i...", (g * ex - g * ez * dHdx) * h, self.W)
#         return topo

#     def rhs_topo_jacobian(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ez = kwargs["ez"]
#         dHdx = kwargs["aux_fields"]["dHdx"]
#         Jac = np.zeros((self.level + 2, self.level + 2, Q.shape[1]))
#         Jac[1:, 0] = g * ex - g * ez * dHdx * np.outer(self.W, np.ones((Q.shape[1])))
#         return Jac

#     def rhs_newtonian(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         # alpha = np.where(Q[0] > 0, Q[1:] / Q[0], 0)
#         lamda = kwargs["lamda"]
#         nu = kwargs["nu"]
#         rho = kwargs["rho"]

#         alpha = Q[1:]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...", (self.StrongTop - self.StrongBottom), nu * alpha / hv
#             )

#         R[1:] -= nu * np.einsum("ij,j...->i...", self.C, alpha / hv)
#         return R

#     def invariant_bingham_bottom(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         # RTest = self.rhs_newtonian(Q_in,**kwargs)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(5)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = kwargs["nu_p"]
#         tau_0 = kwargs["tau_0"]
#         rho = kwargs["rho"]
#         kwargs["nu"] = nu_p
#         lamda = kwargs["lamda"]

#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 rho / lamda * alpha / hv,
#             )

#         # compute the 2nd invariant

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudzeta(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         def nu(zeta, Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             absdudzeta = np.abs(dudzeta(zeta, alpha))
#             eps = 10.0 ** (-8)
#             nu_adaptive = (
#                 h * tau_0 / (absdudzeta + eps) * (1.0 - np.exp(-m / h * absdudzeta))
#             )
#             return nu_adaptive + nu_p

#         # invariant for each level
#         def invariant(Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             result = np.zeros_like(Q)
#             dudz0 = np.zeros(Q.shape[1])
#             assert (h > 0).all()
#             for i in range(0, alpha.shape[0]):
#                 dudz0 += alpha[i] * dphi(i)(0.0)
#             result[:] = 0.5 * np.sqrt(
#                 2.0 * ((tau_0 / h * np.sign(dudz0) + nu_p / h * dudz0) ** 2)
#             )
#             return result, dudz0

#         invar, dudz0 = invariant(Q)

#         return invar

#     def rhs_bingham_bottom(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         # RTest = self.rhs_newtonian(Q_in,**kwargs)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(5)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = kwargs["model"].parameters["nu_p"]
#         tau_0 = kwargs["model"].parameters["tau_0"]
#         rho = kwargs["model"].parameters["rho"]
#         lamda = kwargs["model"].parameters["lamda"]

#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 rho / lamda * alpha / hv,
#             )

#         # compute the 2nd invariant

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudzeta(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         def nu(zeta, Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             absdudzeta = np.abs(dudzeta(zeta, alpha))
#             eps = 10.0 ** (-8)
#             nu_adaptive = (
#                 h * tau_0 / (absdudzeta + eps) * (1.0 - np.exp(-m / h * absdudzeta))
#             )
#             return nu_adaptive + nu_p

#         # invariant for each level
#         def invariant(Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             result = np.zeros_like(Q)
#             dudz0 = np.zeros(Q.shape[1])
#             assert (h > 0).all()
#             for i in range(0, alpha.shape[0]):
#                 dudz0 += alpha[i] * dphi(i)(0.0)
#             result[:] = 0.5 * np.sqrt(
#                 2.0 * ((tau_0 / h * np.sign(dudz0) + nu_p / h * dudz0) ** 2)
#             )
#             return result, dudz0

#         invar, dudz0 = invariant(Q)
#         kwargs["invariant"] = invar
#         Rplastic = np.zeros_like(Q)
#         for k in range(0, alpha.shape[0]):
#             # Rplastic[k+1,:] = tau_0 *invar[0,:] *(self.basis(k)(1.) - self.basis(k)(0.))
#             Rplastic[k + 1, :] = (
#                 tau_0 * np.sign(dudz0) * (self.basis(k)(1.0) - self.basis(k)(0.0))
#             )

#         Rviscous = np.zeros_like(Q)
#         Rviscous[1:] = nu_p * np.einsum("ij,j...->i...", self.C, alpha / hv)

#         indicator = np.zeros_like(Q)
#         for i in range(invar.shape[0]):
#             for j in range(invar.shape[1]):
#                 indicator[i, j] = max(0.0, float(invar[i, j] >= tau_0))

#         return R - indicator * (Rplastic + Rviscous)

#     def invariant_bingham_depthaveraged(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         # RTest = self.rhs_newtonian(Q_in,**kwargs)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(5)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = kwargs["nu_p"]
#         tau_0 = kwargs["tau_0"]
#         rho = kwargs["rho"]
#         kwargs["nu"] = nu_p
#         lamda = kwargs["lamda"]

#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 rho / lamda * alpha / hv,
#             )

#         # compute the 2nd invariant

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudz(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         # invariant for each level
#         def invariant(Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             result = np.zeros(Q.shape[1])
#             for i in range(0, alpha.shape[0]):
#                 phi_i = dphi(i)
#                 for q in range(quad_w.shape[0]):
#                     integrand = lambda z: 0.5 * np.sqrt(
#                         2.0
#                         * (
#                             (
#                                 tau_0 / h * np.sign(dudz(z, alpha))
#                                 + nu_p / h * dudz(z, alpha)
#                             )
#                             ** 2
#                         )
#                     )
#                     result += integr_gp(integrand, q)
#             return result

#         invar = invariant(Q)

#         return invar

#     def rhs_bingham_depthaveraged(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         # RTest = self.rhs_newtonian(Q_in,**kwargs)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(5)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = kwargs["model"].parameters["nu_p"]
#         tau_0 = kwargs["model"].parameters["tau_0"]
#         rho = kwargs["model"].parameters["rho"]
#         lamda = self.parameters["lamda"]

#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 rho / lamda * alpha / hv,
#             )

#         # compute the 2nd invariant

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudz(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         def nu(zeta, Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             absdudzeta = np.abs(dudzeta(zeta, alpha))
#             eps = 10.0 ** (-8)
#             nu_adaptive = (
#                 h * tau_0 / (absdudzeta + eps) * (1.0 - np.exp(-m / h * absdudzeta))
#             )
#             return nu_adaptive + nu_p

#         # invariant for each level
#         def invariant(Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             result = np.zeros(Q.shape[1])
#             for i in range(0, alpha.shape[0]):
#                 phi_i = dphi(i)
#                 for q in range(quad_w.shape[0]):
#                     integrand = lambda z: 0.5 * np.sqrt(
#                         2.0
#                         * (
#                             (
#                                 tau_0 / h * np.sign(dudz(z, alpha))
#                                 + nu_p / h * dudz(z, alpha)
#                             )
#                             ** 2
#                         )
#                     )
#                     result += integr_gp(integrand, q)
#             return result

#         invar = invariant(Q)
#         kwargs["invariant"] = invar
#         Rplastic = np.zeros_like(Q)
#         for k in range(0, alpha.shape[0]):
#             Rplastic[k + 1, :] = tau_0 * (self.basis(k)(1.0) - self.basis(k)(0.0))

#         Rviscous = np.zeros_like(Q)
#         Rviscous[1:] = nu_p * np.einsum("ij,j...->i...", self.C, alpha / hv)

#         indicator = np.zeros(Q.shape[1])
#         for i in range(invar.shape[0]):
#             indicator[i] = max(0.0, invar[i])

#         R = R - Rplastic - indicator * Rviscous

#         return R

#     def rhs_bingham_rowdependent(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         # RTest = self.rhs_newtonian(Q_in,**kwargs)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(5)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = self.parameters["nu_p"]
#         tau_0 = self.parameters["tau_0"]
#         rho = kwargs["rho"]
#         kwargs["nu"] = nu_p
#         lamda = self.parameters["lamda"]

#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         if self.bc_type == "weak":
#             R[1:] = (
#                 -1.0
#                 / lamda
#                 / rho
#                 * np.einsum(
#                     "k, ...->k...",
#                     self.Bottom,
#                     np.einsum("k, k...->...", self.Bottom, alpha),
#                 )
#             )
#         else:
#             R[1:] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 rho / lamda * alpha / hv,
#             )

#         # compute the 2nd invariant

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudzeta(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         def nu(zeta, Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             absdudzeta = np.abs(dudzeta(zeta, alpha))
#             eps = 10.0 ** (-8)
#             nu_adaptive = (
#                 h * tau_0 / (absdudzeta + eps) * (1.0 - np.exp(-m / h * absdudzeta))
#             )
#             return nu_adaptive + nu_p

#         # invariant for each level
#         def invariant(Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             result = np.zeros_like(Q)
#             for k in range(0, alpha.shape[0]):
#                 phi_k = self.basis(k)
#                 for i in range(0, alpha.shape[0]):
#                     phi_i = dphi(i)
#                     for q in range(quad_w.shape[0]):
#                         # integrand = lambda z: phi_k(z) * np.abs(tau_0 + (nu_p / h * dphi(i)(z) * alpha[i]))
#                         integrand = (
#                             lambda z: phi_k(z)
#                             * 0.5
#                             * np.sqrt(
#                                 2.0
#                                 * ((tau_0 * np.sign(dudz(z)) + nu_p / h * dudz(z)) ** 2)
#                             )
#                         )
#                         result[k + 1] += integr_gp(integrand, q)
#             return result

#         invar = invariant(Q)
#         Rplastic = np.zeros_like(Q)
#         for k in range(0, alpha.shape[0]):
#             Rplastic[k + 1, :] = tau_0 * (self.basis(k)(1.0) - self.basis(k)(0.0))

#         Rviscous = np.zeros_like(Q)
#         Rviscous[1:] = nu_p * np.einsum("ij,j...->i...", self.C, alpha / hv)

#         indicator = np.zeros_like(Q)
#         for i in range(invar.shape[0]):
#             for j in range(invar.shape[1]):
#                 indicator[i, j] = max(0.0, invar[i, j])

#         R = R - Rplastic - indicator * Rviscous

#         return R

#     def rhs_bingham(self, Q_in, **kwargs):
#         # quadrature weights and poitns for x in (-1,1)
#         Q = np.array(Q_in)
#         quad_x, quad_w = np.polynomial.legendre.leggauss(3)

#         def integr_gp(f, i):
#             return 0.5 * quad_w[i] * f(0.5 * quad_x[i] + 0.5)

#         nu_p = kwargs["model"].parameters["nu_p"]
#         tau_0 = kwargs["model"].parameters["tau_0"]
#         rho = kwargs["model"].parameters["rho"]
#         lamda = kwargs["model"].parameters["lamda"]
#         m = 100.0

#         Rconst = self.rhs_newtonian(Q, **kwargs)
#         Q = np.einsum("ij, j...->i...", self.Minv, Q_in)

#         alpha = Q[1:] / Q[0]
#         h = Q[0]

#         if len(Q.shape) == 1:
#             R = np.zeros((self.level + 2))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((self.level + 2, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         dphi = lambda order: self.basis_derivative(self.basis(order))

#         def dudzeta(zeta, alpha):
#             result = 0.0
#             for l in range(alpha.shape[0]):
#                 result += alpha[l] * dphi(l)(zeta)
#             return result

#         def nu(zeta, Q):
#             alpha = Q[1:] / Q[0]
#             h = Q[0]
#             absdudzeta = np.abs(dudzeta(zeta, alpha))
#             eps = 10.0 ** (-8)
#             return h * tau_0 / (absdudzeta + eps) * (1.0 - np.exp(-m / h * absdudzeta))

#         for l in range(1, alpha.shape[0]):
#             phi_k = dphi(l)
#             for i in range(1, alpha.shape[0]):
#                 phi_i = dphi(i)
#                 for q in range(quad_w.shape[0]):
#                     integrand = lambda z: phi_k(z) * phi_i(z) * nu(z, Q) / h * alpha[l]
#                     R[l] = integr_gp(integrand, q)
#         return Rconst - R

#     def rhs_newtonian_jacobian(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         nu = kwargs["nu"]
#         lamda = kwargs["lamda"]

#         alpha = Q[1:] / Q[0]
#         h = Q[0]
#         Jac = np.zeros((self.level + 2, self.level + 2, Q.shape[1]))
#         hv = h * np.ones_like(alpha)

#         if self.bc_type == "weak":
#             # dR/dh
#             Jac[1:, 0] = -nu / lamda * np.dot(self.Bottom, self.Bottom) * (-alpha / hv)
#             # dR/dhalpha
#             Jac[1:, 1:] = -nu * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, 1.0 / hv),
#             )
#         else:
#             # dR/dh
#             Jac[1:, 0] = np.einsum(
#                 "ik,i...->k...",
#                 (self.StrongTop - self.StrongBottom),
#                 -2 * nu * alpha / hv**2,
#             )
#             # dR/dhalpha
#             Jac[1:, 1:] = np.einsum(
#                 "ik,i...->k...", (self.StrongTop - self.StrongBottom), nu / hv**2
#             )

#         # dR/dh
#         Jac[1:, 0] -= -2 * nu / h**2 * np.einsum("ij,j...->i...", self.C, alpha)
#         # dR/dhalpha
#         Jac[1:, 1:] -= np.einsum("ij, ...->ij...", self.C, nu / h**2)
#         return Jac

#     def recover_moments_from_profile(self, U, Z):
#         result = np.zeros((self.level + 1, U.shape[0]))
#         dz = Z[1] - Z[0]
#         for x in range(0, U.shape[0]):
#             for i in range(0, self.level):
#                 basis = interpolate.interp1d(Z, self.basis(i)(Z))
#                 f = interpolate.interp1d(Z, U[x, :])
#                 func = lambda z: basis(z) * f(z)
#                 result[i, x] += np.sum(dz * func(Z))
#         return result

#     def recover_velocity_profile(self, Q, flag_substract_mean=False):
#         Q = np.einsum("ij,j...->i...", self.Minv, Q)
#         coefs = Q[1:] / Q[0]
#         if flag_substract_mean:
#             coefs[0] = 0
#         assert len(Q.shape) == 1
#         Z = np.linspace(0, 1, 1000)
#         U = np.zeros_like(Z)
#         for i, c in enumerate(coefs):
#             U += c * self.basis(i)(Z)
#         return U, Z

#     def plot_basis(self, flag="all"):
#         fig = plt.figure(figsize=(10, 8))
#         plt.title("Basis")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.grid()

#         # x-data
#         x_data = np.linspace(0, 1, 1000)

#         # define coefficient array
#         if flag == "single":
#             phi = self.basis(self.level)
#             plt.plot(x_data, phi(x_data), "b", label="phi_" + str(self.level))
#             plt.legend()
#         elif flag == "all":
#             for i in range(1, self.level + 1):
#                 phi = self.basis(i)
#                 plt.plot(x_data, phi(x_data), "b", label="phi_" + str(i))
#                 # plt.legend()
#         plt.show()

#     def strongBoundaryCoefficients(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         lamda = kwargs["model"].parameters["lamda"]
#         h = Q[0]
#         alpha = Q[1:] / Q[0]

#         phi = lambda order: self.basis(order)
#         dphi = lambda order: self.basis_derivative(phi(order))
#         km = alpha.shape[0] - 2
#         k = alpha.shape[0] - 1

#         # [top:free surface, bottom:slip bc] => rewrite as linear system (A [alpha_km, alpha_k] = b, where b constains the lower order alpha)
#         A = np.array(
#             [
#                 [
#                     np.ones(h.shape[0]) * dphi(km)(1.0),
#                     np.ones(h.shape[0]) * dphi(k)(1.0),
#                 ],
#                 [
#                     lamda * dphi(km)(0.0) - h * phi(km)(0.0),
#                     lamda * dphi(k)(0.0) - h * phi(k)(0.0),
#                 ],
#             ]
#         )
#         b = np.array(
#             [
#                 -np.sum([h * dphi(i)(1.0) * alpha[i, :] for i in range(km)], axis=0),
#                 -np.sum(
#                     [
#                         lamda * dphi(i)(0.0) * alpha[i, :]
#                         - h * phi(i)(0.0) * alpha[i, :]
#                         for i in range(km)
#                     ],
#                     axis=0,
#                 ),
#             ]
#         )
#         if len(alpha.shape) == 1:
#             result = np.linalg.solve(A, b)
#             alpha[km] = result[0]
#             alpha[k] = result[1]
#         else:
#             for i in range(alpha.shape[1]):
#                 result = np.linalg.solve(A[:, :, i], b[:, i])
#                 alpha[km, i] = result[0]
#                 alpha[k, i] = result[1]
#         result = np.array(Q, dtype=float)
#         result[1:] = alpha * Q[0]
#         result = np.einsum("ij, j...->i...", self.MM, result)
#         return result


# @pytest.mark.parametrize("a", ["legendre", "chebyshev"])
# def test_basic_matrices(a):
#     b = Matrices(level=2, basistype=a)
#     b.set_runtime_variables()
#     # b.plot_basis()

#     function_name = inspect.currentframe().f_code.co_name
#     misc.write_field_to_npy(
#         b.A,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="A.npy",
#     )
#     misc.write_field_to_npy(
#         b.B,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="B.npy",
#     )
#     misc.write_field_to_npy(
#         b.C,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="C.npy",
#     )
#     misc.write_field_to_npy(
#         b.W,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="W.npy",
#     )
#     misc.write_field_to_npy(
#         b.M,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="M.npy",
#     )
#     misc.write_field_to_npy(
#         b.Bottom,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="Bottom.npy",
#     )
#     misc.write_field_to_npy(
#         b.P,
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="P.npy",
#     )
#     Aref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="A.npy",
#     )
#     Bref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="B.npy",
#     )
#     Cref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="C.npy",
#     )
#     Mref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="M.npy",
#     )
#     Pref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="P.npy",
#     )
#     Bottomref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="Bottom.npy",
#     )
#     Wref = misc.load_npy(
#         filepath=main_dir
#         + "/unittests/unittest_data/"
#         + function_name
#         + "_"
#         + b.basistype
#         + "/",
#         filename="W.npy",
#     )

#     # print(np.linalg.norm(b.A-Aref), 10**(-12))
#     assert np.linalg.norm(b.A - Aref) < 10 ** (-12)
#     assert np.linalg.norm(b.B - Bref) < 10 ** (-12)
#     assert np.linalg.norm(b.C - Cref) < 10 ** (-12)
#     assert np.linalg.norm(b.W - Wref) < 10 ** (-12)
#     assert np.linalg.norm(b.M - Mref) < 10 ** (-12)
#     assert np.linalg.norm(b.P - Pref) < 10 ** (-12)
#     assert np.linalg.norm(b.Bottom - Bottomref) < 10 ** (-12)


# def test_recover_velocity_profile():
#     b = Matrices(level=3)
#     b.set_runtime_variables()

#     coefs = np.array([0.2, 1, 0.5, 0.1, 0.04])
#     c1 = np.array([0.2, 1, 0.0, 0.0, 0])
#     c2 = np.array([0.2, 0, 0.5, 0.0, 0])
#     c3 = np.array([0.2, 0, 0.0, 0.1, 0])
#     c4 = np.array([0.2, 0, 0.0, 0.0, 0.04])
#     U, Z = b.recover_velocity_profile(coefs)

#     function_name = inspect.currentframe().f_code.co_name
#     misc.write_field_to_npy(
#         U,
#         filepath=main_dir + "/unittests/unittest_data/" + function_name + "/",
#         filename="U.npy",
#     )
#     Uref = misc.load_npy(
#         filepath=main_dir + "/unittests/unittest_data/" + function_name + "/",
#         filename="U.npy",
#     )
#     assert np.linalg.norm(U - Uref) < 10 ** (-12)

#     # U_nomean, Z = b.recover_velocity_profile(coefs, flag_substract_mean=True)
#     # U1, Z = b.recover_velocity_profile(c1)
#     # U2, Z = b.recover_velocity_profile(c2)
#     # U3, Z = b.recover_velocity_profile(c3)
#     # U4, Z = b.recover_velocity_profile(c4)
#     # import matplotlib.pyplot as plt
#     # plt.plot(Z, U)
#     # plt.plot(Z, U_nomean)
#     # plt.plot(Z, U1)
#     # plt.plot(Z, U2)
#     # plt.plot(Z, U3)
#     # plt.plot(Z, U4)
#     # plt.show()


# # TODO outsource parameters into model?
# class Matrices2d(Matrices):
#     dim = 2
#     quad_x, quad_w = np.polynomial.legendre.leggauss(5)
#     quax_x = 0.5 * (quad_x + 1)

#     def set_default_parameters(self):
#         super().set_default_parameters()

#     def set_runtime_variables(self):
#         self.offset = self.level + 1
#         super().set_runtime_variables()

#     # TODO legendre only -> rework
#     def flux(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         g = kwargs["g"]
#         ez = kwargs["ez"]
#         dim = self.dim
#         offset = self.offset
#         # initialize return array
#         F_value = np.zeros([1 + dim * (self.level + 1), dim, Q.shape[1]])

#         h = np.where(Q[0] > 0, Q[0], 0)
#         alpha = np.where(h > 0, Q[1 : 1 + offset] / h, 0)
#         beta = np.where(h > 0, Q[1 + offset :] / h, 0)

#         # d/dx!!
#         # # continuity equation
#         F_value[0, 0] = Q[1]
#         # # momentum equations
#         F_value[1 : 1 + offset, 0] = np.einsum(
#             "ijk, i..., j...->k...", self.A, h * alpha, alpha
#         ) + 1 / 2 * g * ez * np.einsum("k,...->k...", self.W, h**2)
#         F_value[1 + offset :, 0] = np.einsum(
#             "ijk, i..., j...->k...", self.A, h * beta, alpha
#         )

#         # d/dy!!
#         F_value[0, 1] = Q[1 + offset]
#         F_value[1 : 1 + offset, 1] = np.einsum(
#             "ijk, i..., j...->k...", self.A, h * alpha, beta
#         )
#         F_value[1 + offset :, 1] = np.einsum(
#             "ijk, i..., j...->k...", self.A, h * beta, beta
#         ) + 1 / 2 * g * ez * np.einsum("k,...->k...", self.W, h**2)

#         return F_value

#     def flux_jac(self, Q, multiply_by_Minv=True, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         # There is no need to rescale Q, since
#         # dx(F(MinvQ)) = dF(MinvQ)/d(MinvQ) d(MinvQ)/dx = dF(Q)/dQ d(MinvQ)/dx
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         g = kwargs["g"]
#         ez = kwargs["ez"]

#         dim = self.dim
#         offset = self.offset
#         h = np.where(Q[0] > 0, Q[0], 0)
#         alpha = np.where(Q[0] > 0, Q[1 : 1 + offset] / h, 0)
#         beta = np.where(Q[0] > 0, Q[1 + offset :] / h, 0)

#         J = np.zeros(
#             (1 + (1 + self.level) * dim, 1 + (1 + self.level) * dim, dim, Q.shape[1])
#         )
#         dfdq = np.zeros((1 + offset * dim, 1 + offset * dim, Q.shape[1]))
#         dgdq = np.zeros((1 + offset * dim, 1 + offset * dim, Q.shape[1]))

#         # dF0 / dh = 0
#         # dF0 / d(halpha)
#         dfdq[0, 1] = 1
#         # dF0 / d(hbeta) = 0

#         # dF(halpha) / dh
#         dfdq[1 : 1 + offset, 0] = -np.einsum(
#             "ijk, i..., j...->k...", self.A, alpha, alpha
#         ) + np.einsum("k, ...->k...", g * ez * self.W, h)
#         # dF(halpha) / d(halpha)
#         dfdq[1 : 1 + offset, 1 : 1 + offset] = 2 * np.einsum(
#             "ijk, i...->jk...", self.A, alpha
#         )
#         # dF(halpha) / d(hbeta) = 0

#         # dF(hbeta) / dh
#         dfdq[1 + offset :, 0] = -np.einsum("ijk, i..., j...->k...", self.A, beta, alpha)
#         # dF(hbeta) / d(halpha)
#         dfdq[1 + offset :, 1 : 1 + offset] = np.einsum("ijk, i...->jk...", self.A, beta)
#         # dF(hbeta) / d(hbeta)
#         dfdq[1 + offset :, 1 + offset :] = np.einsum("ijk, i...->jk...", self.A, alpha)

#         # dG0 / dh = 0
#         # dG0 / d(halpha)
#         # dG0 / d(hbeta) = 0
#         dgdq[0, 1 + offset] = 1

#         # dG(halpha) / dh
#         dgdq[1 : 1 + offset, 0] = -np.einsum(
#             "ijk, i..., j...->k...", self.A, alpha, beta
#         )
#         # dG(halpha) / d(halpha)
#         dgdq[1 : 1 + offset, 1 : 1 + offset] = np.einsum(
#             "ijk, i...->jk...", self.A, beta
#         )
#         # dG(halpha) / d(hbeta) = 0
#         dgdq[1 : 1 + offset, 1 + offset : 1 + 2 * offset] = np.einsum(
#             "ijk, i...->jk...", self.A, alpha
#         )

#         # dG(hbeta) / dh
#         dgdq[1 + offset :, 0] = -np.einsum(
#             "ijk, i..., j...->k...", self.A, beta, beta
#         ) + np.einsum("k, ...->k...", g * ez * self.W, h)
#         # dG(hbeta) / d(halpha) = 0
#         # dG(hbeta) / d(hbeta)
#         dgdq[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] = 2 * np.einsum(
#             "ijk, i...->jk...", self.A, beta
#         )

#         J[:, :, 0] = dfdq
#         J[:, :, 1] = dgdq

#         # Scale J in order to avoid rescaling dQdx!
#         if multiply_by_Minv:
#             J[:, :, 0] = np.einsum("ki, km...->im...", Minv, J[:, :, 0])
#             J[:, :, 1] = np.einsum("ki, km...->im...", Minv, J[:, :, 1])
#         return J

#     def nonconservative_matrix(self, Q, multiply_by_Minv=True, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         h = np.where(Q[0] > 0, Q[0], 0)
#         alpha = np.where(Q[0] > 0, Q[1 : 1 + offset] / h, 0)
#         beta = np.where(h > 0, Q[1 + offset :], 0)

#         # using the derivation as done in the paper from Kowalski
#         if len(Q.shape) == 1:
#             NC = np.zeros((1 + dim * (self.level + 1), 1 + dim * (self.level + 1), dim))
#         else:
#             NC = np.zeros(
#                 (
#                     1 + dim * (self.level + 1),
#                     1 + dim * (self.level + 1),
#                     dim,
#                     Q.shape[1],
#                 )
#             )
#         NC[2 : 1 + offset, 2 : 1 + offset, 0] = np.einsum(
#             "ki, ...->ki...", self.M[1:, 1:], alpha[0]
#         )
#         NC[2 : 1 + offset, 2 + offset : 1 + 2 * offset, 1] = np.einsum(
#             "ki, ...->ki...", self.M[1:, 1:], alpha[0]
#         )
#         NC[2 + offset : 1 + 2 * offset, 2 : 1 + offset, 0] = np.einsum(
#             "ki, ...->ki...", self.M[1:, 1:], beta[0]
#         )
#         NC[2 + offset : 1 + 2 * offset, 2 + offset : 1 + 2 * offset, 1] = np.einsum(
#             "ki, ...->ki...", self.M[1:, 1:], beta[0]
#         )
#         # i j_int, k_div
#         NC[2 : 1 + offset, 2 : 1 + offset, 0] -= np.einsum(
#             "ijk, i...->kj...", self.B[1:, 1:, 1:], alpha[1:]
#         )
#         NC[2 : 1 + offset, 2 + offset : 1 + 2 * offset, 1] -= np.einsum(
#             "ijk, i...->kj...", self.B[1:, 1:, 1:], alpha[1:]
#         )
#         NC[2 + offset : 1 + 2 * offset, 2 : 1 + offset, 0] -= np.einsum(
#             "ijk, i...->kj...", self.B[1:, 1:, 1:], beta[1:]
#         )
#         NC[2 + offset : 1 + 2 * offset, 2 + offset : 1 + 2 * offset, 1] -= np.einsum(
#             "ijk, i...->kj...", self.B[1:, 1:, 1:], beta[1:]
#         )

#         if multiply_by_Minv:
#             NC[:, :, 0, :] = np.einsum("km..., mi->ki...", NC[:, :, 0, :], Minv)
#             NC[:, :, 1, :] = np.einsum("km..., mi->ki...", NC[:, :, 1, :], Minv)
#         return NC

#     def rhs_topo(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ey = kwargs["ey"]
#         ez = kwargs["ez"]
#         dHdx = kwargs["aux_fields"]["dHdx"]
#         dHdy = kwargs["aux_fields"]["dHdy"]
#         if len(Q.shape) == 1:
#             topo = np.zeros((1 + dim * offset))
#         else:
#             topo = np.zeros((1 + dim * offset, Q.shape[1]))
#         h = Q[0]
#         topo[1 : 1 + offset] = np.einsum(
#             "..., i->i...", (g * ex - g * ez * dHdx) * h, self.W
#         )
#         topo[1 + offset :] = np.einsum(
#             "..., i->i...", (g * ey - g * ez * dHdy) * h, self.W
#         )
#         return topo

#     def rhs_topo_jacobian(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         offset = self.offset
#         g = kwargs["g"]
#         h = Q[0]
#         ex = kwargs["ex"]
#         ey = kwargs["ey"]
#         ez = kwargs["ez"]
#         dHdx = kwargs["aux_fields"]["dHdx"]
#         dHdy = kwargs["aux_fields"]["dHdy"]
#         Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset, Q.shape[1]))
#         Jac[1 : 1 + offset, 0] = np.einsum(
#             "..., i->i...", (g * ex - g * ez * dHdx) * np.ones_like(h), self.W
#         )
#         Jac[1 + offset : 1 + 2 * offset, 0] = np.einsum(
#             "..., i->i...", (g * ey - g * ez * dHdy) * np.ones_like(h), self.W
#         )
#         return Jac

#     def rhs_mu_i(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         d = kwargs["model"].parameters["d"]
#         mu_s = kwargs["model"].parameters["mu_s"]
#         mu_2 = kwargs["model"].parameters["mu_2"]
#         I0 = kwargs["model"].parameters["I0"]
#         g = kwargs["g"]
#         ez = kwargs["ez"]
#         paramA = mu_s + (mu_2 - mu_s) / I0 * g * ez
#         paramB = (mu_2 - mu_s) / I0 * d * np.sqrt(g * ez)

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h

#         alpha_b = (np.einsum("k, k...->...", self.Bottom_granular, alpha),)
#         beta_b = (np.einsum("k, k...->...", self.Bottom_granular, beta),)

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         dphik_dzeta = lambda k, x: self.basis_derivative(self.basis(k))(x)
#         du_dzeta = lambda x: np.sum(
#             [
#                 self.basis_derivative(self.basis(i))(x) * alpha[i]
#                 for i in range(self.level + 1)
#             ]
#         )
#         dv_dzeta = lambda x: np.sum(
#             [
#                 self.basis_derivative(self.basis(i))(x) * beta[i]
#                 for i in range(self.level + 1)
#             ]
#         )

#         for xi, wi in zip(self.quad_x, self.quad_w):
#             for k in range(0, offset):
#                 R[k + 1] += (
#                     paramA / h * dphik_dzeta(k, xi) * (1 - xi) * np.sign(du_dzeta(xi))
#                 )
#                 # can be done beforehand
#                 R[k + 1] += (
#                     paramB / h * dphik_dzeta(k, xi) * np.sqrt((1 - xi)) * du_dzeta(xi)
#                 )
#         return R

#     def rhs_bc_mu_i(self, Q, **kwargs):
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         d = kwargs["model"].parameters["d"]
#         mu_s = kwargs["model"].parameters["mu_s"]
#         mu_2 = kwargs["model"].parameters["mu_2"]
#         I0 = kwargs["model"].parameters["I0"]
#         g = kwargs["g"]
#         ez = kwargs["ez"]
#         paramA = mu_s + (mu_2 - mu_s) / I0 * g * ez
#         paramB = (mu_2 - mu_s) / I0 * d * np.sqrt(g * ez)

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h

#         alpha_b = (np.einsum("k, k...->...", self.Bottom_granular, alpha),)
#         beta_b = (np.einsum("k, k...->...", self.Bottom_granular, beta),)

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)
#         R[1 : 1 + offset] = (
#             -self.Bottom / h * paramA * np.sign(alpha_b)
#             - self.Bottom / h * paramB * alpha_b
#         )
#         R[1 + offset : 1 + 2 * offset] = (
#             -self.Bottom / h * paramA * np.sign(beta_b)
#             - self.Bottom / h * paramB * beta_b
#         )
#         return R

#     def rhs_bc_chezy(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         C = kwargs["model"].parameters["ChezyCoef"]

#         h = Q[0]
#         alpha = np.where(Q[0] > 0., Q[1 : 1 + offset] / h, 0.)
#         beta = np.where(Q[0] > 0., Q[1 + offset : 1 + 2 * offset] / h, 0.)

#         if len(Q.shape) == 1:
#             R = np.zeros((Q.shape[0]))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((Q.shape[0], Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)
#         if h <= 0:
#             return R
#         u_star = 0
#         for i, a_i in enumerate(alpha):
#             for j, a_j in enumerate(alpha):
#                 u_star += self.Bottom[i] * self.Bottom[j] * a_i * a_j
#         for i, b_i in enumerate(beta):
#             for j, b_j in enumerate(beta):
#                 u_star += self.Bottom[i] * self.Bottom[j] * b_i * b_j
#         u_star = np.sqrt(u_star)
#         R[1 : 1 + offset] = (
#             -1.0 / C**2 * np.einsum("k, k...->...", self.Bottom, alpha * u_star),
#         )
#         R[1 + offset : 1 + 2 * offset] = (
#             -1.0 / C**2 * np.einsum("k, k...->...", self.Bottom, beta * u_star),
#         )
#         return R

#     def rhs_bc_chezy_jacobian(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         offset = self.offset
#         C = kwargs["model"].parameters["ChezyCoef"]

#         h = Q[0]
#         alpha = np.where(Q[0] > 0, Q[1 : 1 + offset] / h , 0)
#         beta = np.where(Q[0] > 0, Q[1 + offset : 1 + 2 * offset] / h, 0)
#         u_star = 0
#         a_sq = 0
#         b_sq = 0
#         a_sum = 0
#         b_sum = 0
#         ab = 0
#         for i, a_i in enumerate(alpha):
#             a_sum += self.Bottom[i] * a_i
#             for j, a_j in enumerate(alpha):
#                 u_star += self.Bottom[i] * self.Bottom[j] * a_i * a_j
#                 a_sq += self.Bottom[i] * self.Bottom[j] * a_i * a_j
#         for i, b_i in enumerate(beta):
#             b_sum += self.Bottom[i] * b_i
#             for j, b_j in enumerate(beta):
#                 u_star += self.Bottom[i] * self.Bottom[j] * b_i * b_j
#                 b_sq += self.Bottom[i] * self.Bottom[j] * b_i * b_j
#         for i, a_i in enumerate(alpha):
#             for j, b_j in enumerate(beta):
#                 ab += self.Bottom[i] * self.Bottom[j] * a_i * b_j
#         u_star = np.sqrt(u_star) + eps
#         if len(Q.shape) == 1:
#             Jac = np.zeros((Q.shape[0], Q.shape[0]))
#             h_ext = h * np.ones_like(alpha)
#         else:
#             Jac = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#             h_ext = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         # dR/dh
#         Jac[1 : 1 + offset, 0] = 1.0 * 2 / C**2 * a_sum * u_star / h
#         # 1.0
#         # / C**2
#         # * np.einsum("k, k...->...", self.Bottom, 2 * alpha * u_star / h_ext),
#         Jac[1 + offset : 1 + 2 * offset, 0] = 1.0 * 2 / C**2 * b_sum * u_star / h
#         #     1.0
#         #     / C**2
#         #     * np.einsum("k, k...->...", self.Bottom, 2 * beta * u_star / h_ext),
#         # )
#         # dR/dhalpha
#         Jac[1 : 1 + offset, 1 : 1 + offset] = (
#             -1.0 / C**2 * (u_star / h + a_sum**2 / u_star / h)
#         )
#         Jac[1 : 1 + offset, 1 + offset : 1 + 2 * offset] = (
#             -1.0 / C**2 * (a_sum * b_sum / u_star / h)
#         )
#         # Jac[1 : 1 + offset, 1 : 1 + offset] = -1.0 / C**2 * np.einsum(
#         #     "k, ...->k...", self.Bottom, u_star/h
#         # ) - 1.0 / C**2 * np.einsum("k, ...->k...", self.Bottom , a_sq / h / u_star)
#         # Jac[1 : 1 + offset, 1 + offset : 1 + 2 * offset] = (
#         #     -1.0
#         #     / C**2
#         #     * np.einsum(
#         #         "k, ...->k...",
#         #         self.Bottom,
#         #         np.einsum("k, ...->...", self.Bottom, a_sum * b_sum / u_star / h),
#         #     )
#         # )
#         # dR/hbeta
#         Jac[1 + offset : 1 + 2 * offset, 1 : 1 + offset] = (
#             -1.0 / C**2 * (a_sum * b_sum / u_star / h)
#         )
#         Jac[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] = (
#             -1.0 / C**2 * (u_star / h + b_sum**2 / u_star / h)
#         )
#         # Jac[1 + offset : 1 + 2 * offset, 1 : 1 + offset] = (
#         #     -1.0
#         #     / C**2
#         #     * np.einsum(
#         #         "k, ...->k...",
#         #         self.Bottom,
#         #         np.einsum("k, ...->...", self.Bottom, b_sum * a_sum / u_star / h),
#         #     )
#         # )
#         # Jac[
#         #     1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset
#         # ] = -1.0 / C**2 * np.einsum(
#         #     "k, ...->k...", self.Bottom, u_star / h_ext
#         # ) - 1.0 / C**2 * np.einsum(
#         #     "k, ...->k...",
#         #     self.Bottom,
#         #     np.einsum("k, k...->...", self.Bottom, b_sum / u_star / h_ext),
#         # )
#         return Jac

#     def rhs_bc_grad_slip(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         sliplength = kwargs["model"].parameters["sliplength"]
#         rho = kwargs["model"].parameters["rho"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)
#         dz = 0.05
#         R[1 : 1 + offset] = (
#             -1.0
#             / rho
#             / sliplength
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, alpha / h),
#             )
#             / dz
#         )
#         R[1 + offset : 1 + 2 * offset] = (
#             -1.0
#             / rho
#             / sliplength
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, beta / h),
#             )
#             / dz
#         )
#         return R

#     def rhs_bc_grad_slip_jacobian(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         offset = self.offset
#         sliplength = kwargs["model"].parameters["sliplength"]
#         rho = kwargs["model"].parameters["rho"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h
#         if len(Q.shape) == 1:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset))
#             h_ext = h * np.ones_like(alpha)
#         else:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset, Q.shape[1]))
#             h_ext = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         # dR/dh
#         Jac[1 : 1 + offset, 0] = (
#             -1.0
#             / sliplength
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, -2 * alpha / h_ext),
#             )
#         )
#         Jac[1 + offset : 1 + 2 * offset, 0] = (
#             -1.0
#             / sliplength
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, -2 * beta / h_ext),
#             )
#         )
#         # dR/dhalpha
#         Jac[1 : 1 + offset, 1 : 1 + offset] = (
#             -1.0
#             / sliplength
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, 1.0 / h_ext**2),
#             )
#         )
#         # dR/dbeta
#         Jac[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] = (
#             -1.0
#             / sliplength
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, 1.0 / h_ext**2),
#             )
#         )
#         return Jac

#     def rhs_bc_newtonian(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         nu = kwargs["model"].parameters["nu"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)
#         R[1 : 1 + offset] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, alpha / h),
#             )
#         )
#         R[1 + offset : 1 + 2 * offset] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, beta / h),
#             )
#         )
#         return R

#     def rhs_bc_newtonian_jacobian(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         offset = self.offset
#         nu = kwargs["model"].parameters["nu"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h
#         if len(Q.shape) == 1:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset))
#             h_ext = h * np.ones_like(alpha)
#         else:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset, Q.shape[1]))
#             h_ext = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         # dR/dh
#         Jac[1 : 1 + offset, 0] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, -2 * alpha / h_ext**2),
#             )
#         )
#         Jac[1 + offset : 1 + 2 * offset, 0] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, -2 * beta / h_ext**2),
#             )
#         )
#         # dR/dhalpha
#         Jac[1 : 1 + offset, 1 : 1 + offset] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, 1.0 / h_ext**2),
#             )
#         )
#         # dR/dbeta
#         Jac[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] = (
#             -1.0
#             * nu
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom_deriv, 1.0 / h_ext**2),
#             )
#         )
#         return Jac

#     def rhs_bc_slip(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         lamda = kwargs["model"].parameters["sliplength"]
#         rho = kwargs["model"].parameters["rho"]

#         h = Q[0]
#         alpha = np.where(Q[0] > 0, Q[1 : 1 + offset] / h, 0)
#         beta = np.where(Q[0] > 0, Q[1 + offset : 1 + 2 * offset] / h, 0)

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)
#         R[1 : 1 + offset] = (
#             -1.0
#             / lamda
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, alpha),
#             )
#         )
#         R[1 + offset : 1 + 2 * offset] = (
#             -1.0
#             / lamda
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, beta),
#             )
#         )
#         return R

#     def rhs_bc_slip_jacobian(self, Q, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q)
#         offset = self.offset
#         lamda = kwargs["model"].parameters["sliplength"]
#         rho = kwargs["model"].parameters["rho"]

#         h = Q[0]
#         alpha = np.where(Q[0] > 0, Q[1 : 1 + offset] / h, 0)
#         beta = np.where(Q[0] > 0, Q[1 + offset : 1 + 2 * offset] / h, 0)
#         if len(Q.shape) == 1:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset))
#             h_ext = h * np.ones_like(alpha)
#         else:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset, Q.shape[1]))
#             h_ext = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         # dR/dh
#         Jac[1 : 1 + offset, 0] = (
#             -1.0
#             / lamda
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, -alpha / h_ext),
#             )
#         )
#         Jac[1 + offset : 1 + 2 * offset, 0] = (
#             -1.0
#             / lamda
#             / rho
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, -beta / h_ext),
#             )
#         )
#         # dR/dhalpha
#         Jac[1 : 1 + offset, 1 : 1 + offset] = (
#             -1.0
#             / rho
#             / lamda
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, 1.0 / h_ext),
#             )
#         )
#         # dR/dbeta
#         Jac[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] = (
#             -1.0
#             / rho
#             / lamda
#             * np.einsum(
#                 "k, ...->k...",
#                 self.Bottom,
#                 np.einsum("k, k...->...", self.Bottom, 1.0 / h_ext),
#             )
#         )
#         return Jac

#     def rhs_newtonian(self, Q_, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q_)
#         dim = self.dim
#         offset = self.offset
#         nu = kwargs["model"].parameters["nu"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h

#         if len(Q.shape) == 1:
#             R = np.zeros((1 + offset * dim))
#             hv = h * np.ones_like(alpha)
#         else:
#             R = np.zeros((1 + offset * dim, Q.shape[1]))
#             hv = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         R[1 : 1 + offset] -= nu * np.einsum("ij,j...->i...", self.C, alpha / hv)
#         R[1 + offset : 1 + 2 * offset] -= nu * np.einsum(
#             "ij,j...->i...", self.C, beta / hv
#         )
#         return R

#     def rhs_newtonian_jacobian(self, Q_, **kwargs):
#         Minv = self.Minv
#         if "Minv" in kwargs:
#             Minv = kwargs["Minv"]
#         Q = np.einsum("ij, j...->i...", Minv, Q_)
#         offset = self.offset
#         nu = kwargs["model"].parameters["nu"]

#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset : 1 + 2 * offset] / h
#         if len(Q.shape) == 1:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset))
#             h_ext = h * np.ones_like(alpha)
#         else:
#             Jac = np.zeros((1 + 2 * offset, 1 + 2 * offset, Q.shape[1]))
#             h_ext = np.repeat(h[np.newaxis, :], alpha.shape[0], axis=0)

#         # dR/dh
#         Jac[1 : 1 + offset, 0] -= (
#             -2 * nu / h**2 * np.einsum("ij,j...->i...", self.C, alpha)
#         )
#         Jac[1 + offset : 1 + 2 * offset, 0] -= (
#             -2 * nu / h**2 * np.einsum("ij,j...->i...", self.C, beta)
#         )
#         # dR/dhalpha
#         Jac[1 : 1 + offset, 1 : 1 + offset] -= np.einsum(
#             "ij, ...->ij...", self.C, nu / h**2
#         )
#         Jac[1 + offset : 1 + 2 * offset, 1 + offset : 1 + 2 * offset] -= np.einsum(
#             "ij, ...->ij...", self.C, nu / h**2
#         )
#         return Jac

#     def recover_velocity_profile(self, Q, flag_substract_mean=False):
#         Q = np.einsum("ij,j...->i...", self.Minv, Q)
#         dim = self.dim
#         offset = self.offset
#         h = Q[0]
#         alpha = Q[1 : 1 + offset] / h
#         beta = Q[1 + offset :] / h
#         if flag_substract_mean:
#             alpha[0] = 0
#             beta[0] = 0
#         assert len(Q.shape) == 1
#         Z = np.linspace(0, 1, 1000)
#         U = np.zeros_like(Z)
#         V = np.zeros_like(Z)
#         for i, c in enumerate(alpha):
#             U += c * self.basis(i)(Z)
#         for i, c in enumerate(beta):
#             V += c * self.basis(i)(Z)
#         return U, V, Z

#     def create_MM(self):
#         self.MM[0, 0] = 1
#         self.MM[1 : 1 + self.offset, 1 : 1 + self.offset] = self.M
#         self.MM[1 + self.offset :, 1 + self.offset :] = self.M

#     def create_Minv(self):
#         self.Minv[0, 0] = 1
#         self.Minv[1 : 1 + self.offset, 1 : 1 + self.offset] = np.linalg.inv(self.M)
#         self.Minv[1 + self.offset :, 1 + self.offset :] = self.Minv[
#             1 : 1 + self.offset, 1 : 1 + self.offset
#         ]

#     def eigenvalues(self, Q, nij, **kwargs):
#         # Q = np.where(np.abs(Q) < 10 ** (-8), 0, Q)
#         A = self.flux_jac(Q, multiply_by_Minv=False, **kwargs)
#         NC = self.nonconservative_matrix(Q, multiply_by_Minv=False, **kwargs)
#         ANC = np.einsum("ijk..., k...->ij...", A - NC, nij)
#         EVs = np.zeros_like(Q)
#         imaginary = False
#         # TODO vectorize?
#         for i in range(Q.shape[1]):
#             # ev = np.linalg.eigvals(A[:,:,i]-NC[:,:,i])
#             ev = np.linalg.eigvals(np.einsum("ij, jk->ik", self.Minv, ANC[:, :, i]))
#             assert np.isfinite(ev).all()
#             # assert(np.isreal(ev).all())
#             if not np.isreal(ev).all():
#                 imaginary = True
#                 # TODO enable with debug_level!
#                 # print('WARNING: imaginary eigenvalues: ', str(ev), ' for Q: ', Q[:,i])
#                 ev = 10**6
#                 ev = 0.0
#             EVs[:, i] = ev
#             del ev
#         return EVs, imaginary


# class MatricesWithBottom(Matrices):
#     dim = 1

#     def fix_wet_dry(self, Q):
#         Qin = np.array(Q)
#         for q in Qin:
#             q = np.where(Qin[0] <= 0, 0, q)
#         return Qin

#     def set_default_parameters(self):
#         super().set_default_parameters()

#     def set_runtime_variables(self):
#         self.offset = self.level + 1
#         super().set_runtime_variables()
#         self.create_MM()
#         self.create_Minv()

#     def create_MM(self):
#         self.MM[0, 0] = 1
#         self.MM[1:, 1:] = self.M

#     def create_Minv(self):
#         self.Minv[0, 0] = 1
#         self.Minv[1:, 1:] = np.linalg.inv(self.M)

#     #
#     # TODO legendre only -> rework
#     def flux(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         F_value = np.zeros([2 + self.dim * (self.level + 1), self.dim, Q.shape[1]])
#         F_value[:-1] = super().flux(Q[:-1], **kwargs)
#         return F_value

#     def flux_jac(self, Q, multiply_by_Minv=True, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         J = np.zeros((self.level + 3, self.level + 3, self.dim, Q.shape[1]))
#         J[:-1, :-1, :, :] = super().flux_jac(
#             Q[:-1], multiply_by_Minv=multiply_by_Minv, **kwargs
#         )
#         return J

#     def nonconservative_matrix(self, Q, multiply_by_Minv=True, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         Q[:-1] = np.einsum("ij, j...->i...", self.Minv, Q[:-1])
#         h = Q[0]
#         alpha = Q[1:-1] / Q[0]
#         g = kwargs["g"]
#         ez = kwargs["ez"]

#         dim = self.dim

#         # using the derivation as done in the paper from Kowalski

#         # if len(Q.shape) == 1:
#         #     NC = np.zeros((self.level + 3, self.level + 3, dim))
#         # else:
#         NC = np.zeros((self.level + 3, self.level + 3, dim, Q.shape[1]))
#         NC[2:-1, 2:-1, 0] = np.einsum("ki, ...->ki...", self.M[1:, 1:], alpha[0])
#         NC[2:-1, 2:-1, 0] -= np.einsum(
#             "ijk, i...->kj...", self.B[1:, 1:, 1:], alpha[1:]
#         )

#         Minv_ext = np.zeros((Q.shape[0], Q.shape[0]))
#         Minv_ext[:-1, :-1] = self.Minv
#         Minv_ext[-1, -1] = 1.0
#         NC[1 : 1 + self.level + 1, -1] = -g * ez * np.outer(self.W, h)[:, np.newaxis, :]
#         if multiply_by_Minv:
#             NC[:, :] = np.einsum("kml..., mi->kil...", NC, Minv_ext)
#             return NC

#         else:
#             return NC

#     def rhs_topo(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         Q[:-1] = np.einsum("ij, j...->i...", self.Minv, Q[:-1])
#         dim = self.dim
#         offset = self.offset
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ey = kwargs["ey"]
#         ez = kwargs["ez"]
#         if len(Q.shape) == 1:
#             topo = np.zeros((2 + dim * offset))
#         else:
#             topo = np.zeros((2 + dim * offset, Q.shape[1]))
#         h = Q[0]
#         topo[1 : 1 + offset] = (g * ex) * h * self.W[:, np.newaxis]
#         return topo

#     def rhs_topo_jacobian(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         Q[:-1] = np.einsum("ij, j...->i...", self.Minv, Q[:-1])
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ez = kwargs["ez"]
#         Jac = np.zeros((self.level + 3, self.level + 3, Q.shape[1]))
#         Jac[1:-1, 0, :] = g * ex * self.W[:, np.newaxis]
#         return Jac

#     def rhs_newtonian(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         R = np.zeros(((self.level + 3), Q.shape[1]))
#         R[:-1] = super().rhs_newtonian(Q[:-1], **kwargs)
#         return R

#     def rhs_newtonian_jacobian(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         res = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         res[:-1, :-1] = super().rhs_newtonian_jacobian(Q[:-1], **kwargs)
#         return res

#     def eigenvalues(self, Q, nij, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         Minv_ext = np.zeros((Q.shape[0], Q.shape[0]))
#         Minv_ext[:-1, :-1] = self.Minv
#         Minv_ext[-1, -1] = 1.0
#         A = self.flux_jac(Q, multiply_by_Minv=False, **kwargs)
#         NC = self.nonconservative_matrix(Q, multiply_by_Minv=False, **kwargs)
#         ANC = np.einsum("ijk..., k...->ij...", A - NC, nij)
#         EVs = np.zeros_like(Q)
#         imaginary = False
#         # TODO vectorize?
#         for i in range(Q.shape[1]):
#             if Q[0, i] > 0:
#                 # ev = np.linalg.eigvals(A[:,:,i]-NC[:,:,i])
#                 ev = np.linalg.eigvals(np.einsum("ij, jk->ik", Minv_ext, ANC[:, :, i]))
#                 assert np.isfinite(ev).all()
#                 # assert(np.isreal(ev).all())
#                 if not np.isreal(ev).all():
#                     imaginary = True
#                     # TODO enable with debug_level!
#                     # print('WARNING: imaginary eigenvalues: ', str(ev), ' for Q: ', Q[:,i])
#                     ev = 10**6
#                 EVs[:, i] = ev
#                 del ev
#         return EVs, imaginary


# class MatricesWithBottom2d(Matrices2d):
#     dim = 2

#     def fix_wet_dry(self, Q):
#         Qin = np.array(Q)
#         for q in Qin:
#             q = np.where(Qin[0] <= 0, 0, q)
#         return Qin

#     def set_default_parameters(self):
#         super().set_default_parameters()

#     def set_runtime_variables(self):
#         self.offset = self.level + 1
#         super().set_runtime_variables()
#         self.create_MM()
#         self.create_Minv()

#     def create_MM(self):
#         self.MM = np.zeros(
#             (2 + (self.level + 1) * self.dim, 2 + (self.level + 1) * self.dim)
#         )
#         offset = 1 + self.level
#         self.MM[0, 0] = 1
#         self.MM[1 : 1 + offset, 1 : 1 + offset] = self.M
#         self.MM[1 + offset : -1, 1 + offset : -1] = self.M
#         self.MM[-1, -1] = 1

#     def create_Minv(self):
#         self.Minv = np.zeros(
#             (2 + (self.level + 1) * self.dim, 2 + (self.level + 1) * self.dim)
#         )
#         self.Minv = np.linalg.inv(self.MM)

#     #
#     # TODO legendre only -> rework
#     def flux(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         res = np.zeros((Q.shape[0], self.dim, Q.shape[1]))
#         res[:-1] = super().flux(
#             self.fix_wet_dry(Q)[:-1], Minv=self.Minv[:-1, :-1], **kwargs
#         )
#         return res

#     def flux_jac(self, Q, multiply_by_Minv=True, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dim, Q.shape[1]))
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         out[:-1, :-1] = super().flux_jac(self.fix_wet_dry(Q)[:-1], **kwargs)
#         return out

#     def nonconservative_matrix(self, Q, multiply_by_Minv=True, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         out = np.zeros((Q.shape[0], Q.shape[0], self.dim, Q.shape[1]))
#         h = Q[0]
#         g = kwargs["g"]
#         ez = kwargs["ez"]
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         out[:-1, :-1] = super().nonconservative_matrix(
#             self.fix_wet_dry(Q)[:-1], **kwargs
#         )
#         offset = 1 + self.level
#         out[1 : 1 + offset, -1, 0] = -g * ez * np.einsum("..., k->k...", h, self.W[:])
#         out[1 + offset : 1 + 2 * offset, -1, 1] = (
#             -g * ez * np.einsum("..., k->k...", h, self.W[:])
#         )
#         return out

#     def rhs_topo(self, Q, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         Q = np.einsum("ij, j...->i...", self.Minv, Q)
#         Q = self.fix_wet_dry(Q)
#         dim = self.dim
#         offset = self.offset
#         g = kwargs["g"]
#         ex = kwargs["ex"]
#         ey = kwargs["ey"]
#         ez = kwargs["ez"]
#         if len(Q.shape) == 1:
#             topo = np.zeros((1 + dim * offset))
#         else:
#             topo = np.zeros((1 + dim * offset, Q.shape[1]))
#         h = Q[0]
#         topo[1 : 1 + offset] = np.einsum("..., i->i...", (g * ex) * h, self.W)
#         topo[1 + offset :] = np.einsum("..., i->i...", (g * ey) * h, self.W)
#         return topo

#     def rhs_newtonian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros_like(Q)
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1] = super().rhs_newtonian(Q[:-1], **kwargs)
#         return res

#     def rhs_newtonian_jacobian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         offset = self.offset
#         res = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1, :-1] = super().rhs_newtonian_jacobian(Q[:-1], **kwargs)
#         return res

#     def rhs_bc_slip(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros_like(Q)
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1] = super().rhs_bc_slip(self.fix_wet_dry(Q)[:-1], **kwargs)
#         return res

#     def rhs_bc_slip_jacobian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1, :-1] = super().rhs_bc_slip_jacobian(self.fix_wet_dry(Q)[:-1], **kwargs)
#         return res

#     def rhs_bc_newtonian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros_like(Q)
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1] = super().rhs_bc_newtonian(Q[:-1], **kwargs)
#         return res

#     def rhs_bc_newtonian_jacobian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1, :-1] = super().rhs_bc_newtonian_jacobian(Q[:-1], **kwargs)
#         return res

#     def rhs_bc_grad_slip(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros_like(Q)
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1] = super().rhs_bc_grad_slip(Q[:-1], **kwargs)
#         return res

#     def rhs_bc_grad_slip_jacobian(self, Q_, **kwargs):
#         Q = self.fix_wet_dry(Q_)
#         res = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))
#         kwargs["Minv"] = self.Minv[:-1, :-1]
#         res[:-1, :-1] = super().rhs_bc_grad_slip_jacobian(Q[:-1], **kwargs)
#         return res

#     # def eigenvalues(self, Q_, nij, **kwargs):
#     #     Q = self.fix_wet_dry(Q_)
#     #     return super().eigenvalues(Q, nij, **kwargs)
#     def eigenvalues(self, Q, nij, **kwargs):
#         Q = self.fix_wet_dry(Q)
#         A = self.flux_jac(Q, multiply_by_Minv=False, Minv=self.Minv, **kwargs)
#         NC = self.nonconservative_matrix(
#             Q, multiply_by_Minv=False, Minv=self.Minv, **kwargs
#         )
#         ANC = np.einsum("ijk..., k...->ij...", A - NC, nij)
#         EVs = np.zeros_like(Q)
#         imaginary = False
#         # TODO vectorize?
#         for i in range(Q.shape[1]):
#             # ev = np.linalg.eigvals(A[:,:,i]-NC[:,:,i])
#             if Q[0, i] > 10 ** (-8):
#                 ev = np.linalg.eigvals(np.einsum("ij, jk->ik", self.Minv, ANC[:, :, i]))
#                 assert np.isfinite(ev).all()
#                 # assert(np.isreal(ev).all())
#                 ev = np.real_if_close(ev, tol=10**10)
#                 if not np.isreal(ev).all():
#                     imaginary = True
#                     # TODO enable with debug_level!
#                     # print('WARNING: imaginary eigenvalues: ', str(ev), ' for Q: ', Q[:,i])
#                     ev = 10**6
#                 EVs[:, i] = ev
#                 del ev
#         return EVs, imaginary


# @pytest.mark.parametrize("level", [1, 2, 3])
# def test_smm_vs_paper(level):
#     b = Matrices(level=level)
#     b.set_runtime_variables()

#     Q = np.linspace(2, 2 + level + 1, 2 + level).reshape((2 + level, 1))
#     Qscaled = np.einsum("ij,j...->i...", b.Minv, Q)
#     nu = 0.1
#     lamda = 0.2
#     g = 1
#     kwargs = {
#         "nu": nu,
#         "lamda": lamda,
#         "g": g,
#         "dHdx": np.zeros(Q.shape[1]),
#         "ex": 0,
#         "ez": 1.0,
#     }

#     FluxRef = paper.F(Qscaled.T, g=g).reshape((2 + b.level))
#     Flux = b.flux(Q, **kwargs)
#     # print(np.round(Flux.reshape((2+level)),2), '\n', np.round(np.array(FluxRef).reshape((2+level)),2))
#     assert np.allclose(
#         Flux.reshape((2 + level)), np.array(FluxRef).reshape((2 + level))
#     )

#     RRef = -paper.P(Qscaled.T, lamda=lamda, nu=nu).reshape((2 + b.level))
#     R = b.rhs_newtonian(Q, **kwargs)
#     # print(np.round(R.reshape((2+level)),2), '\n', np.round(np.array(RRef).reshape((2+level)),2))
#     assert np.allclose(R.reshape((2 + level)), np.array(RRef).reshape((2 + level)))

#     NCRef = paper.Q(Qscaled.T).reshape((2 + level, 2 + level)).T
#     NC = b.nonconservative_matrix(Q, **kwargs)
#     # print(np.round(NC.reshape((2+level, 2+level)),2), '\n', np.round(np.array(NCRef).reshape((2+level, 2+level)),2))
#     assert np.allclose(
#         NC.reshape((2 + level, 2 + level)),
#         np.array(NCRef).reshape((2 + level, 2 + level)),
#     )

#     FluxJacRef = paper.A(Qscaled.T).reshape((2 + level, 2 + level))
#     FluxJac = b.flux_jac(Q, **kwargs)
#     # print(np.round(FluxJac.reshape((2+level, 2+level)),2), '\n', np.round(np.array(FluxJacRef).reshape((2+level, 2+level)),2))
#     assert np.allclose(
#         FluxJac.reshape((2 + level, 2 + level)),
#         np.array(FluxJacRef).reshape((2 + level, 2 + level)),
#     )

#     SysMatrixRef = FluxJacRef - NCRef
#     SysMatrix = (FluxJac - NC)[:, :, 0]
#     EVsRef = np.zeros_like(Q)
#     EVs = np.zeros_like(Q)
#     for i in range(Q.shape[1]):
#         EVsRef = np.linalg.eigvals(SysMatrixRef)
#         EVs = np.linalg.eigvals(SysMatrix)
#     assert np.allclose(EVs, EVsRef)


# def test_strong_boundary_coefficients():
#     Q = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).T
#     b = Matrices(level=3)
#     b.set_runtime_variables()
#     # I set MM and Minv to unity, since the Mathematica code (I compared to) does not account for scaling
#     b.MM = np.eye(5)
#     b.Minv = np.eye(5)
#     kwargs = {"lamda": 0.1}
#     coefs = b.strongBoundaryCoefficients(0.0, Q, **kwargs)
#     # print(coefs)
#     assert np.abs(coefs[-2, 0] - (-1.0 - 2.0 / 3)) < 10 ** (-8)
#     assert np.abs(coefs[-1, 0] - (-1.0 - 1.0 / 3)) < 10 ** (-8)


# def test_recover_moments_from_profile():
#     b = Matrices(level=4)
#     b.set_runtime_variables()
#     U = np.linspace(1, 50, 50).reshape((5, 10))
#     Z = np.linspace(0, 1, 10)
#     res = b.recover_moments_from_profile(U, Z)
#     # print(res)


# def test_newtonian_1D_vs_2D():
#     d1 = Matrices(level=1)
#     d1.set_runtime_variables()
#     d2 = Matrices2d(level=1)
#     d2.set_runtime_variables()

#     kwargs = {"nu": 0.1, "lamda": 0.1}
#     Qd1 = np.array([1.1, -0.1, 0.0])
#     Qd2 = np.array([1.1, -0.1, 0.0, 0.0, 0.0])
#     print(d1.rhs_newtonian(Qd1, **kwargs))
#     print(d2.rhs_newtonian(Qd2, **kwargs))


# if __name__ == "__main__":
#     # test_basic_matrices('legendre')
#     # test_basic_matrices('chebyshev')
#     # test_recover_velocity_profile()
#     # test_smm_vs_paper(1)
#     # test_smm_vs_paper(2)
#     # test_smm_vs_paper(3)
#     # test_strong_boundary_coefficients()

#     test_newtonian_1D_vs_2D()

#     # TODO untested
#     # test_recover_moments_from_profile()
