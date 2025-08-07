# import numpy as np
# import os
# import sys
# import yaml
# import scipy.linalg
# import logging

# from library.solver.baseclass import BaseYaml

# main_dir = os.getenv("SMPYTHON")


# class Flux(BaseYaml):
#     yaml_tag = "!Flux"

#     def set_default_default_parameters(self):
#         self.scheme = "rusanov"
#         self.limiter = None

#     # TODO simplify interface with kwargs
#     def evaluate(self, F, **kwargs):
#         if self.scheme is None:
#             return np.zeros_like(Ql)
#         return getattr(sys.modules[__name__], self.scheme)(F, **kwargs)


# # TODO fix
# # def local_lax_friedrichs(F, Ql, Qr, EVs, **kwargs):
# #     EVmax = np.max(EVs, axis=0)
# #     Qout = 1.0 / 2.0 * (F(Ql) + F(Qr)) - np.repeat(
# #         EVmax[np.newaxis, :], Ql.shape[0], axis=0
# #     ) / 2 * (Qr - Ql)
# #     return Qout


# # TODO fix
# # def rusanov(F, Ql, Qr, EVs, **kwargs):
# #     EVmax = np.max(EVs, axis=0)
# #     Qout = 1.0 / 2.0 * (F(Ql) + F(Qr)) - np.max(np.abs(EVmax)) * (Qr - Ql) / 2
# #     return Qout


# def price_alphaC2(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         dt = kwargs["dt"]
#         elem = kwargs["i_elem"]
#         neighbor = kwargs["i_neighbor"]
#         dx = kwargs["mesh"]["element_volume"][elem]

#         Ahat = 1.0
#         Ahat_sq = 1.0
#         I = 1.0
#         Am = 0.25 * (2 * Ahat - dx / dt * I - dt / dx * Ahat_sq)
#         # Ap = 0.25 * (2*Ahat + dx/dt * I + dt/dx * Ahat_sq)
#         Ap = 0.25 * (2 * Ahat * nij - dx / dt * I - dt / dx * Ahat_sq)
#         if elem < neighbor:
#             Qout = Am * (Qj - Qi)
#         elif elem > neighbor:
#             Qout = Ap * (-Qi + Qj)
#         else:
#             assert False
#         return Qout, False

#     return f


# # see e.g. Seminar Thesis from Koellermeier's student
# def price_alphaC(F, **kwargs):
#     # approx. Roe matrix using segment path integration
#     # obtain quadrature points and weights
#     quad_x, quad_w = np.polynomial.legendre.leggauss(5)
#     # TODO ASSUMING uniform elements

#     def A_hat(Qi, Qj, **kwargs):
#         # flatten the last dimension to use quadrature
#         A = lambda Q: kwargs["model"].quasilinear_matrix(Q)
#         Qs = lambda s: (Qi + (Qj - Qi) * s)
#         res = 0.5 * quad_w[0] * A(Qs(0.5 * quad_x[0] + 0.5))
#         # mapped quad_x from quad in (-1, 1) to Qs in (0, 1)
#         for i in range(1, quad_x.shape[0]):
#             res += 0.5 * quad_w[i] * A(Qs(0.5 * quad_x[i] + 0.5))
#         return res

#     def f(Qi, Qj, nij, **kwargs):
#         dt = kwargs["dt"]
#         Sij = kwargs["Sij"]
#         Vi = kwargs["Vi"]
#         Vj = kwargs["Vj"]

#         Ahat = A_hat(Qi, Qj, **kwargs)
#         Ahat = np.einsum("ijk..., k...->ij...", Ahat, nij)
#         Ahat_sq = np.einsum("ij..., jl...->il...", Ahat, Ahat)

#         I = np.eye(Qi.shape[0])[:, :, np.newaxis]
#         I = np.repeat(I, Qi.shape[1], axis=2)
#         Am = (
#             0.5 * Ahat
#             - (Vi * Vj) / (Vi + Vj) / (dt * Sij) * I
#             - 0.25 * dt * Sij / (Vi + Vj) * Ahat_sq
#         )
#         Qout = np.einsum("ij...,  j...->i...", Am, Qj - Qi)
#         return Qout, False

#     return f


# # see PRICE paper from Toro
# def price_lf(F, **kwargs):
#     def A_hat(**kwargs):
#         # inside the domain
#         elem = kwargs["i_elem"]
#         Q = kwargs["Qn"]
#         Q_neighbors = []
#         for i_edge, _ in enumerate(kwargs["mesh"]["element_neighbors"][elem]):
#             i_neighbor = (kwargs["mesh"]["element_neighbors"][elem])[i_edge]
#             Q_neighbors.append(Q[:, i_neighbor])

#         # TODO SLOW: I need a map from elem->physical boundary id (None for none)
#         # boundary
#         for i_bp, i_elem in enumerate(kwargs["mesh"]["boundary_edge_element"]):
#             if i_elem != elem:
#                 continue
#             Qj = kwargs["bc"](
#                 kwargs["model"].boundary_conditions, i_bp, Q, dim=kwargs["mesh"]["dim"]
#             )
#             Q_neighbors.append(Qj)

#         Q_neighbors = np.array(Q_neighbors)
#         Qstar = np.mean(Q_neighbors, axis=0)[:, np.newaxis]
#         return kwargs["model"].quasilinear_matrix(Qstar)

#     def f(Qi, Qj, nij, **kwargs):
#         dt = kwargs["dt"]
#         dx = kwargs["mesh"]["element_incircle"][kwargs["i_elem"]]
#         A = A_hat(**kwargs)
#         # Qout = 0.5*np.einsum('ijk..., k..., j...->i...', A, nij, Qi+Qj) - 0.5*dx/dt*(Qj-Qi)
#         A = lambda Q: kwargs["model"].quasilinear_matrix(Q)
#         Qout = (
#             0.5 * np.einsum("ijk..., k..., j...->i...", A(Qi), nij, Qi)
#             + 0.5 * np.einsum("ijk..., k..., j...->i...", A(Qj), nij, Qj)
#             - 0.5 * dx / dt * (Qj - Qi)
#         )
#         return Qout, False

#     return f


# # see PRICE paper from Toro
# # TODO inefficient, since I need to evaluate A(Qi) for Qstar on all edges. While it could be just done once, if I had not this elemnt wise approach
# def price_lw(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         dt = kwargs["dt"]
#         dx = kwargs["mesh"]["element_incircle"][kwargs["i_elem"]]
#         alpha = 0.5
#         A = lambda Q: kwargs["model"].quasilinear_matrix(Q)
#         Qstar = 0.5 * (Qi + Qj) - alpha * dt / dx * (
#             np.einsum("ijk..., k..., j...->i...", A(0.5 * (Qi + Qj)), nij, Qj - Qi)
#         )
#         return np.einsum("ijk..., k..., j...->i...", A(Qi), nij, Qstar), False

#     return f


# def price_f(F, **kwargs):
#     f_lf = price_lf(F, **kwargs)
#     f_lw = price_lw(F, **kwargs)

#     def f(Qi, Qj, nij, **kwargs):
#         q_lw, err_lw = f_lw(Qi, Qj, nij, **kwargs)
#         q_lf, err_lf = f_lf(Qi, Qj, nij, **kwargs)
#         return 0.5 * (q_lw + q_lf), err_lw or err_lf

#     return f


# def osher_segmentpath(F, **kwargs):
#     samples, weights = np.polynomial.legendre.leggauss(3)
#     # shift from [-1, 1] to [0,1]
#     samples = 0.5 * (samples + 1)
#     weights *= 0.5

#     def f(Qi, Qj, nij, **kwargs):
#         Dn = np.zeros_like(Qi)

#         model = kwargs["model"]
#         A = lambda s: np.einsum(
#             "ijk..., k...->ij...",
#             model.quasilinear_matrix(Qi + s * (Qj - Qi), **kwargs),
#             nij,
#         )[:, :, 0]

#         def pos(A):
#             E, vr = scipy.linalg.eig(A, left=False, right=True)

#             err = 0
#             singular = False
#             try:
#                 err = np.linalg.norm(
#                     np.einsum("ij,jk,kl->il", vr, np.diag(E), np.linalg.inv(vr)) - A
#                 )
#             except:
#                 singular = True
#             if err > 10 ** (-4) or singular:
#                 print("Error: Eigenvalue decomposition failed", err)
#                 if np.abs(Qi[1]) < 10 ** (-6) and np.abs(Qj[1] < 10 ** (-6)):
#                     print("Qi", Qi)
#                     print("Qj", Qj)
#                     # assert False
#                 return np.zeros_like(vr), True

#             return (
#                 np.real_if_close(
#                     np.einsum(
#                         "ij,jk,kl->il", vr, np.diag(np.abs(E)), np.linalg.inv(vr)
#                     ),
#                     tol=10**10,
#                 ),
#                 False,
#             )

#         A_s = [A(s) for s in samples]
#         int_posA_s = np.zeros_like(A_s[0])
#         error = False
#         for As, w in zip(A_s, weights):
#             res, err = pos(As)
#             error = error or err
#             int_posA_s += w * res

#         if error:
#             return np.zeros_like(Dn), False
#         Dn = 0.5 * np.einsum("ij...,j...->i...", (F(Qi) + F(Qj)), nij)
#         Dn -= 0.5 * np.einsum("ij, j...->i...", (int_posA_s), (Qj - Qi))
#         return Dn, False

#     return f


# def roe_segmentpath(F, **kwargs):
#     samples, weights = np.polynomial.legendre.leggauss(5)
#     # shift from [-1, 1] to [0,1]
#     samples = 0.5 * (samples + 1)
#     weights *= 0.5

#     def f(Qi, Qj, nij, **kwargs):
#         model = kwargs["model"]
#         Dn = np.zeros_like(Qi)

#         A = lambda s: np.einsum(
#             "ijk..., k...->ij...",
#             model.quasilinear_matrix(Qi + s * (Qj - Qi), **kwargs),
#             nij,
#         )[:, :, 0]

#         def pos(A):
#             # U, s, Vh = scipy.linalg.svd(A)
#             # S = np.diag(np.where(s > 0.01 * s[0], s, 0.0))
#             # AA = U @ S @ Vh
#             E, vr = scipy.linalg.eig(A, left=False, right=True)
#             # vr[np.abs(vr) < 10**(-8)] = 0

#             try:
#                 err = np.linalg.norm(
#                     np.einsum("ij,jk,kl->il", vr, np.diag(E), np.linalg.inv(vr)) - A
#                 )
#             except:
#                 print("err")
#             if err > 10 ** (-8):
#                 logger = logging.getLogger(__name__ + ":roe_segmentpath")
#                 logger.error('Eigenvalue decomposition failed for Qi={} and Qj={}.'.format(Qi, Qj))
#                 # assert False
#                 return np.zeros_like(vr), True
#             out = np.real_if_close(
#                 np.einsum("ij,jk,kl->il", vr, np.diag(np.abs(E)), np.linalg.inv(vr)),
#                 tol=10**10,
#             )
#             assert np.isreal(out).all()
#             return out, False

#         A_s = [A(s) for s in samples]
#         intA_s = np.zeros_like(A_s[0])
#         for As, w in zip(A_s, weights):
#             intA_s += w * As

#         dissipation, dis_err = pos(intA_s)
#         Dn = 0.5 * np.einsum("ij...,j...->i...", (F(Qi) + F(Qj)), nij)
#         if not dis_err:
#             Dn -= 0.5 * np.einsum("ij, j...->i...", (dissipation), (Qj - Qi))

#         return Dn, False

#     return f


# def roe_segmentpath_swe_2d(F, **kwargs):
#     samples, weights = np.polynomial.legendre.leggauss(5)
#     # shift from [-1, 1] to [0,1]
#     samples = 0.5 * (samples + 1)
#     weights *= 0.5

#     def f(Qi, Qj, nij, **kwargs):
#         model = kwargs["model"]
#         Dn = np.zeros_like(Qi)
#         Q = lambda s: Qi + s * (Qj - Qi)
#         absA = np.zeros((Qi.shape[0], Qi.shape[0]), dtype=float)
#         for w, s in zip(weights, samples):
#             ev, R, iR, err = model.eigensystem(Q(s), nij)
#             if err > 10 ** (2) and Q(s)[0] > 10 ** (-4):
#                 ev, R, iR, err = model.eigensystem(Q(s), nij)
#                 return Dn, False
#             absA += w * (R[:, :, 0] @ np.diag(np.abs(ev[:, 0])) @ iR[:, :, 0])

#         Dn = 0.5 * np.einsum("ij...,j...->i...", (F(Qi) + F(Qj)), nij)
#         Dn -= 0.5 * np.einsum("ij, j...->i...", (absA), (Qj - Qi))
#         return Dn, False

#     return f


# def roe_segmentpath_full_jacobian(F, **kwargs):
#     samples, weights = np.polynomial.legendre.leggauss(1)
#     # shift from [-1, 1] to [0,1]
#     samples = 0.5 * (samples + 1)
#     weights *= 0.5

#     def f(Qi, Qj, nij, **kwargs):
#         Dn = np.zeros_like(Qi)

#         model = kwargs["model"]
#         A = lambda s: np.einsum(
#             "ijk..., k...->ij...",
#             model.quasilinear_matrix(Qi + s * (Qj - Qi), **kwargs),
#             nij,
#         )[:, :, 0]

#         def pos(A):
#             # U, s, Vh = scipy.linalg.svd(A)
#             # S = np.diag(np.where(s > 0.01 * s[0], s, 0.0))
#             # AA = U @ S @ Vh
#             E, vr = scipy.linalg.eig(A, left=False, right=True)
#             # vr[np.abs(vr) < 10**(-8)] = 0

#             err = 0.0
#             singular = False
#             try:
#                 err = np.linalg.norm(
#                     np.einsum("ij,jk,kl->il", vr, np.diag(E), np.linalg.inv(vr)) - A
#                 )
#             except:
#                 singular = True
#             if err > 10 ** (-8) or singular:
#                 print("Error: Eigenvalue decomposition failed", err)
#                 print("Qi", Qi)
#                 print("Qj", Qj)
#                 # assert False
#                 return np.zeros_like(vr)
#             out = np.real_if_close(
#                 np.einsum("ij,jk,kl->il", vr, np.diag(np.abs(E)), np.linalg.inv(vr)),
#                 tol=10**10,
#             )
#             assert np.isreal(out).all()
#             return out

#         A_s = [A(s) for s in samples]
#         intA_s = np.zeros_like(A_s[0])
#         for As, w in zip(A_s, weights):
#             intA_s += w * As

#         Dn += 0.5 * np.einsum("ij, j...->i...", (intA_s), (Qj - Qi))
#         Dn -= 0.5 * np.einsum("ij, j...->i...", (pos(intA_s)), (Qj - Qi))
#         return Dn, False

#     return f


# # roe osher with segment path and trapezoidal integration
# def roe_osher_trapezoidal(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         # solve by dimensional splitting
#         model = kwargs["model"]
#         A = lambda Q: model.quasilinear_matrix(Q)

#         def pos(A):
#             E, vl, vr = scipy.linalg.eig(A[:, :, 0], left=True, right=True)
#             return (
#                 np.einsum("ij,jk,kl->il", vr, np.diag(np.abs(E)), np.linalg.inv(vr)),
#             )

#         Qout = np.zeros_like(Qi)
#         Ai = A(Qi)
#         Aj = A(Qj)
#         F1 = 0.5 * (F(Qi) + F(Qj))
#         for d, n in enumerate(nij):
#             F2 = 0.5 * (pos(Ai[:, :, d]) + pos(Aj[:, :, d]))
#             Qout[:, d] += F1[:, d] - np.dot(
#                 np.einsum("ij, j...->j...", F2, (Qj - Qi)[:]), n
#             )
#         return Qout, False

#     return f


# # def local_lax_friedrichs(F, Ql, Qr, dt, dx, **kwargs):
# #     EVi, imag_i = kwargs['model'].eigenvalues(Qi, nij)
# #     EVj, imag_j = kwargs['model'].eigenvalues(Qj, nij)
# #     assert(imag_i==False and imag_j == False)
# #     EVmax = np.max(np.abs(np.vstack([EVi, EVj])), axis=0)
# #     Qout = 1./2. * (F(Ql) + F(Qr)) - np.repeat(EVmax[np.newaxis,:], Ql.shape[0], axis=0)/2*(Qr-Ql)
# #     return Qout


# def rusanov(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         EVi, imag_i = kwargs["model"].eigenvalues(Qi, nij)
#         EVj, imag_j = kwargs["model"].eigenvalues(Qj, nij)
#         # assert imag_i == False and imag_j == False
#         smax = np.max(np.abs(np.vstack([EVi, EVj])))
#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - 0.5 * smax * (Qj - Qi)
#         return Qout, False

#     return f


# def rusanov_WB_swe_1d(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         assert Qi.shape[0] == 3
#         # Well balanced matrix for 1d swe system with Q = (h, hu, h_b)
#         IWB = np.eye(3)
#         IWB[0, 2] = 1.0
#         IWB[2, 2] = 0.0
#         EVi, imag_i = kwargs["model"].eigenvalues(Qi, nij)
#         EVj, imag_j = kwargs["model"].eigenvalues(Qj, nij)
#         assert imag_i == False and imag_j == False
#         smax = np.max(np.abs(np.vstack([EVi, EVj])))
#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - 0.5 * smax * np.einsum("ij, j...->i...", IWB, Qj - Qi)
#         # Qout = 0.5 * np.einsum(
#         #     "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         # ) - 0.5 * smax * np.einsum('ij, j...->i...', IWB,  Qj - Qi)
#         return Qout, False

#     return f


# def rusanov_WB_swe_2d(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         assert Qi.shape[0] == 4
#         # Well balanced matrix for 1d swe system with Q = (h, hu, h_b)
#         IWB = np.eye(4)
#         IWB[0, 3] = 1.0
#         if Qi[0] <= 10 ** (-12) and Qj[0] + Qj[3] < Qi[3]:
#             IWB[0, :] = 0.0
#         if Qj[0] <= 10 ** (-12) and Qi[0] + Qi[3] < Qj[3]:
#             IWB[0, :] = 0.0
#         IWB[3, 3] = 0.0
#         EVi, imag_i = kwargs["model"].eigenvalues(Qi, nij)
#         EVj, imag_j = kwargs["model"].eigenvalues(Qj, nij)
#         assert imag_i == False and imag_j == False
#         smax = np.max(np.abs(np.vstack([EVi, EVj])))
#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - 0.5 * smax * np.einsum("ij, j...->i...", IWB, Qj - Qi)
#         return Qout, False

#     return f

# def rusanov_fast_WB_smm_2d(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         EVi = kwargs['EVi']
#         EVj = kwargs['EVj']
#         IWB = np.eye(Qi.shape[0])
#         IWB[-1, -1] = 0.0
#         IWB[0, -1] = 1.0
#         smax = np.max(np.abs(np.vstack([EVi, EVj])))
#         if Qi[0] <= 10 ** (-12) and Qj[0] + Qj[-1] < Qi[-1]:
#             IWB[0, :] = 0.0
#         if Qj[0] <= 10 ** (-12) and Qi[0] + Qi[-1] < Qj[-1]:
#             IWB[0, :] = 0.0
#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - 0.5 * smax * np.einsum("ij, j...->i...", IWB, Qj - Qi)
#         return Qout, False

#     return f

# def rusanov_WB_smm_2d(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         IWB = np.eye(Qi.shape[0])
#         IWB[-1, -1] = 0.0
#         IWB[0, -1] = 1.0
#         IWB = np.repeat(IWB[:,:,np.newaxis], Qi.shape[1], axis=2)
#         EVi, imag_i = kwargs["model"].eigenvalues(Qi, nij)
#         EVj, imag_j = kwargs["model"].eigenvalues(Qj, nij)
#         EVi = np.real(EVi)
#         EVj = np.real(EVj)
#         smax = np.max(np.abs(np.vstack([EVi, EVj])), axis=0)
#         # if np.max(np.abs(EVi)) == 0 or np.max(np.abs(EVj)) == 0:
#         #     smax = 0
#         # if Qi[0] <= 10 ** (-12) and Qj[0] + Qj[-1] < Qi[-1]:
#         #     IWB[0, :] = 0.0
#         # if Qj[0] <= 10 ** (-12) and Qi[0] + Qi[-1] < Qj[-1]:
#         #     IWB[0, :] = 0.0
#         model = kwargs['model']
#         if "ShallowMoments" in model.yaml_tag or "ShallowWater" in model.yaml_tag:
#             set_zero = np.logical_and((Qi[0] <= 10 ** (-12)) ,(Qj[0] + Qj[-1] < Qi[-1]))
#             set_zero = np.logical_or(set_zero, np.logical_and((Qi[0] <= 10 ** (-12)) ,(Qj[0] + Qj[-1] < Qi[-1])))
#             IWB[:,:,set_zero] = 0.0

#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - 0.5 * smax * np.einsum("ij..., j...->i...", IWB, Qj - Qi)
#         return Qout, False

#     return f


# def lax_friedrichs(F, **kwargs):
#     def f(Qi, Qj, nij, **kwargs):
#         dx = np.min(kwargs["mesh"].element_incircle)
#         dt = kwargs["dt"]
#         Qout = 0.5 * np.einsum(
#             "ij...,j...->i...", (F(Qi) + F(Qj)), nij
#         ) - dx / 2.0 / dt * (Qj - Qi)
#         return Qout, False

#     return f


# def hll_shm(F, Ql, Qr, EVs, dx, dt):
#     threshold = 0
#     len = Ql.shape[0]

#     s_slow = np.min(EVs, axis=0)
#     s_fast = np.max(EVs, axis=0)

#     flux0_flag = np.array(s_slow >= -threshold, dtype=float)
#     flux1_flag = np.array(s_fast <= threshold, dtype=float)

#     s_slow_extruded = np.tile(s_slow, (len, 1))
#     s_fast_extruded = np.tile(s_fast, (len, 1))
#     flux0_flag_extruded = np.tile(flux0_flag, (len, 1))
#     flux1_flag_extruded = np.tile(flux1_flag, (len, 1))

#     fluxintermediate_flag_extruded = (
#         np.ones(flux0_flag_extruded.shape) - flux1_flag_extruded - flux0_flag_extruded
#     )

#     flux0_flag = np.array(s_slow >= -threshold, dtype=float)
#     flux1_flag = np.array(s_fast <= threshold, dtype=float)

#     Qout = s_fast_extruded * F(Ql)
#     Qout = Qout - s_slow_extruded * F(Qr)

#     Qout = Qout + s_fast_extruded * s_slow_extruded * (Qr - Ql)
#     eps_extruded = np.full_like(s_fast_extruded, 10 ** (-12))

#     Qout = Qout / (s_fast_extruded - s_slow_extruded + eps_extruded)
#     Qout = Qout * fluxintermediate_flag_extruded

#     Qout = Qout + F(Qr) * flux1_flag_extruded
#     Qout = Qout + F(Ql) * flux0_flag_extruded

#     return Qout


