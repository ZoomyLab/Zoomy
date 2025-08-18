# import numpy as np
# import os
# import sys

# from library.solver.baseclass import BaseYaml  # nopep8

# main_dir = os.getenv('SMPYTHON')


# class FluxDirect(BaseYaml):
#     yaml_tag = "!FluxDirect"

#     def set_default_default_parameters(self):
#         self.scheme = None
#         self.limiter = None

#     # TODO rework interface, shift params into kwargs
#     def evaluate(self, F, Q, Ql, Qr, **kwargs):
#         if self.scheme is None:
#             return np.zeros_like(Ql[:, 1:])
#         return getattr(sys.modules[__name__], self.scheme)(F, Q, Ql, Qr, **kwargs)


# def lax_friedrichs(F, Q, Ql, Qr, **kwargs):
#     LF = lambda ql, qr, dx, dt: 0.5 * (F(ql) + F(qr)) - dx / (2.0 * dt) * (qr - ql)
#     Ql = np.roll(Q, 1, axis=1)
#     Qr = np.roll(Q, -1, axis=1)
#     Fl = LF(Ql, Q, dx, dt)
#     # Fr = np.roll(Fl, -1, axis=1)
#     Fr = LF(Q, Qr, dx, dt)
#     # U + dt/dx will be taken case of in the subsequent routines
#     return Fr - Fl


# def price_alphaC_hyperbolic(F, Q, Ql, Qr, **kwargs):
#     # approx. Roe matrix using segment path integration
#     # obtain quadrature points and weights
#     quad_x, quad_w = np.polynomial.legendre.leggauss(1)
#     # TODO ASSUMING uniform elements

#     nij = np.array([1.0])

#     def A_hat(Qi, Qj, **kwargs):
#         Qi[3:, :] = 0.0
#         Qj[3:, :] = 0.0
#         # flatten the last dimension to use quadrature
#         A = lambda Q: kwargs["model"].quasilinear_matrix(Q)
#         Qs = lambda s: (Qi + (Qj - Qi) * s)
#         res = 0.5 * quad_w[0] * A(Qs(0.5 * quad_x[0] + 0.5))
#         # mapped quad_x from quad in (-1, 1) to Qs in (0, 1)
#         for i in range(1, quad_x.shape[0]):
#             res += 0.5 * quad_w[i] * A(Qs(0.5 * quad_x[i] + 0.5))
#         return res

#     def Aplus(Qi, Qj, nij, **kwargs):
#         dx = kwargs["dx"]
#         dt = kwargs["dt"]
#         Ahat = A_hat(Qi, Qj, **kwargs)
#         Ahat_sq = np.einsum("ij..., jl...->il...", Ahat, Ahat)
#         I = np.eye(Qi.shape[0])[:, :, np.newaxis]
#         I = np.repeat(I, Qi.shape[1], axis=2)
#         Am = 0.25 * (2 * Ahat + dx / dt * I + dt / dx * Ahat_sq)
#         Qout = np.einsum("ij...,  j...->i...", Am, Qj - Qi)
#         return Qout

#     def Aminus(Qi, Qj, nij, **kwargs):
#         dx = kwargs["dx"]
#         dt = kwargs["dt"]
#         Ahat = A_hat(Qi, Qj, **kwargs)
#         Ahat_sq = np.einsum("ij..., jl...->il...", Ahat, Ahat)
#         I = np.eye(Qi.shape[0])[:, :, np.newaxis]
#         I = np.repeat(I, Qi.shape[1], axis=2)
#         Am = 0.25 * (2 * Ahat - dx / dt * I - dt / dx * Ahat_sq)
#         Qout = np.einsum("ij...,  j...->i...", Am, Qj - Qi)
#         return Qout

#     Qim = Ql[:, :-1]
#     Qi = Qr[:, :-1]
#     Qj = Ql[:, 1:]
#     Qjp = Qr[:, 1:]
#     return Aplus(Qim, Qi, nij, **kwargs) + Aminus(Qj, Qjp, nij, **kwargs)


# # see e.g. Seminar Thesis from Koellermeier's student
# def price_alphaC(F, Q, Ql, Qr, **kwargs):
#     # approx. Roe matrix using segment path integration
#     # obtain quadrature points and weights
#     quad_x, quad_w = np.polynomial.legendre.leggauss(1)
#     # TODO ASSUMING uniform elements

#     nij = np.array([1.0])

#     def A_hat(Qi, Qj, **kwargs):
#         # flatten the last dimension to use quadrature
#         A = lambda Q: kwargs["model"].quasilinear_matrix(Q)
#         Qs = lambda s: (Qi + (Qj - Qi) * s)
#         res = 0.5 * quad_w[0] * A(Qs(0.5 * quad_x[0] + 0.5))
#         # mapped quad_x from quad in (-1, 1) to Qs in (0, 1)
#         for i in range(1, quad_x.shape[0]):
#             res += 0.5 * quad_w[i] * A(Qs(0.5 * quad_x[i] + 0.5))
#         return res

#     def Aplus(Qi, Qj, nij, **kwargs):
#         dx = kwargs["dx"]
#         dt = kwargs["dt"]
#         Ahat = A_hat(Qi, Qj, **kwargs)
#         Ahat_sq = np.einsum("ij..., jl...->il...", Ahat, Ahat)
#         I = np.eye(Qi.shape[0])[:, :, np.newaxis]
#         I = np.repeat(I, Qi.shape[1], axis=2)
#         Am = 0.25 * (2 * Ahat + dx / dt * I + dt / dx * Ahat_sq)
#         Qout = np.einsum("ij...,  j...->i...", Am, Qj - Qi)
#         return Qout

#     def Aminus(Qi, Qj, nij, **kwargs):
#         dx = kwargs["dx"]
#         dt = kwargs["dt"]
#         Ahat = A_hat(Qi, Qj, **kwargs)
#         Ahat_sq = np.einsum("ij..., jl...->il...", Ahat, Ahat)
#         I = np.eye(Qi.shape[0])[:, :, np.newaxis]
#         I = np.repeat(I, Qi.shape[1], axis=2)
#         Am = 0.25 * (2 * Ahat - dx / dt * I - dt / dx * Ahat_sq)
#         Qout = np.einsum("ij...,  j...->i...", Am, Qj - Qi)
#         return Qout

#     Qim = Ql[:, :-1]
#     Qi = Qr[:, :-1]
#     Qj = Ql[:, 1:]
#     Qjp = Qr[:, 1:]
#     return Aplus(Qim, Qi, nij, **kwargs) + Aminus(Qj, Qjp, nij, **kwargs)

