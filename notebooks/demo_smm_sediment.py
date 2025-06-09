# ---
# title: 'Introduction to SMS'
# author: Ingo Steldermann
# date: 09/18/2024
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] vscode={"languageId": "raw"}
# ## Aim
#
# We want to solve a PDE of the form
#
# $$
# \partial_t \mathbf{Q} + \partial_{x_i} \mathbf{F}_i = NC_i \partial_{x_i} \mathbf{Q} + \mathbf{S}
# $$
#
# where $\mathbf{F}$ is a flux, $NC$ is a non-conservative matrix, $\mathbf{S}$ is a source term and dimension $i=[1]$ for 1d, $i=[1,2]$ for 2d and $i = [1, 2,3]$ for 3d.
#
#
# -

# ## Imports

# +
# | code-fold: true
# | code-summary: "Load packages"
# | output: false

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from copy import deepcopy
import seaborn

seaborn.set_context("talk")
from IPython.display import Math
import sympy as sym

main_dir = os.getenv("SMS")
import pytest
from types import SimpleNamespace

from library.model.model import *
from library.pysolver.solver import *
from library.pysolver.solver import jax_fvm_unsteady_semidiscrete as fvm_unsteady
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing

# -

# ## 1d Model Simple

# ### Model definition

# +


class ShallowMomentsSedimentSimple(Model):
    """
    Shallow Moments Sediment 1d

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters={},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis(),
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 3
        self.basis = Basis(basis=Legendre_shifted(order=self.levels + 1))
        self.basis.compute_matrices(self.levels)
        if type(fields) == type([]):
            fields_smm = fields[:-1]
        elif type(fields) == int:
            fields_smm = fields - 1
        else:
            assert False
        self.smm = ShallowMoments(
            dimension=dimension,
            fields=fields_smm,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            basis=Basis(basis=Legendre_shifted(order=self.levels + 1)),
        )
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def quasilinear_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:-1]
        a = [_ha / h for _ha in ha]
        u = a[0]
        p = self.parameters
        ub = 0
        for i in range(self.levels + 1):
            ub += a[i]

        def my_abs(x):
            return sym.Abs(x)

        tau = p.epsilon * p.rho * my_abs(ub) * ub
        theta = (my_abs(tau) * p.d_s**2) / (p.g * (p.rho_s - p.rho) * p.d_s**3)

        def pos(x):
            return sym.Piecewise((x, x > 0), (0, True))

        def sign(x):
            return sym.sign(x)

        delta_q = (
            (24 * p.Q)
            / (1 - p.phi)
            * sign(tau)
            * p.epsilon
            / (p.g * (1 / p.r - 1) * p.d_s)
            * (pos(theta - p.theta_c)) ** (1 / 2)
            * ub
            / h
        )

        nc[0, 1] = 1
        nc[1, 0] = p.g * h - u**2

        nc[1, 1] = 2 * u
        if self.levels > 0:
            nc[1, 2] = 2 / 3 * u
        nc[1, -1] = p.g * h

        if self.levels > 0:
            nc[2, 0] = -2 * a[0] * a[1]
            nc[2, 1] = 2 * a[1]
        if self.levels > 1:
            nc[3, 0] = -2 / 3 * a[1] ** 2

        for k in range(1, self.levels + 1):
            nc[1 + k, 1 + k] = a[0]
        for k in range(1, self.levels + 1 - 1):
            N = k + 1
            nc[1 + k, 1 + k + 1] = (N + 1) / (2 * N + 1) * a[1]
            nc[1 + k + 1, 1 + k] = (N - 1) / (2 * N - 1) * a[1]

        nc[-1, 0] = -ub * delta_q
        nc[-1, 1] = delta_q
        for k in range(1, self.levels + 1):
            nc[-1, 1 + k] = delta_q
        return [-nc]

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_fields)])
        # h = self.variables[0]
        # hu = self.variables[1]
        # u = hu/h
        # p = self.parameters
        # evs[0] = sym.sqrt(p.g * h) + u
        # evs[1] = sym.sqrt(p.g * h) - u
        evs[:-1, 0] = self.smm.eigenvalues()
        return evs

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def newtonian(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        out[:-1, 0] = self.smm.newtonian()
        return out

    def manning(self):
        """
        :gui:
            - requires_parameter: ('epsilon', 1000.0)
        """
        assert "epsilon" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        a = [_ha / h for _ha in ha]
        p = self.parameters
        ub = 0
        for i in range(self.levels + 1):
            ub += a[i]

        def my_abs(x):
            return sym.Abs(x)
            # return sym.piecewise_fold(sym.Piecewise((-x, x<0), (x, True)))
            # return (sym.Piecewise((-x, x<0), (x, True)))

        for k in range(1 + self.levels):
            out[1 + k] += -(p.epsilon * self.basis.M[k, k]) * ub * my_abs(ub)
        return out


# -

# ### Mesh and model construction

# +
# | code-fold: true
# | code-summary: "Initialize model and mesh"
mesh = petscMesh.Mesh.create_1d((-6, 6), 600)

bcs = BC.BoundaryConditions(
    [
        BC.Extrapolation(physical_tag="left"),
        BC.Extrapolation(physical_tag="right"),
    ]
)


def define_model(level=0):
    def custom_ic(x):
        Q = np.zeros(1 + level + 1 + 1, dtype=float)
        Q[0] = np.where(x[0] < 0, 1.0, 0.05)
        return Q

    ic = IC.UserFunction(custom_ic)

    d_s = 3.9 * 10 ** (-3)
    rho = 1000.0
    rho_s = 1580.0
    r = rho / rho_s
    g = 9.81
    phi = 0.47
    # n = 1.
    # d_50 = 1.
    # epsilon = (g * n**2)/((7/6)**2 * d_50**(1/3))
    epsilon = 0.0324
    Q = d_s * np.sqrt(g * (1 / r - 1) * d_s)
    # Q = 0.
    theta_c = 0.047
    # theta_c = 0

    fields = ["h"] + [f"hu_{l}" for l in range(level + 1)] + ["b"]
    model = ShallowMomentsSedimentSimple(
        boundary_conditions=bcs,
        initial_conditions=ic,
        fields=fields,
        settings={"friction": ["manning", "newtonian"]},
        # settings={"friction": []},
        parameters={
            "g": g,
            "ez": 1.0,
            "d_s": d_s,
            "rho": rho,
            "rho_s": rho_s,
            "r": r,
            "phi": phi,
            "epsilon": epsilon,
            "Q": Q,
            "theta_c": theta_c,
            "nu": 0.01,
        },
    )
    return model


# -

# ### Simulation

# +
# | code-fold: false
# | code-summary: "Simulation"
# | output: false


def run_model(level):
    model = define_model(level=level)
    level = model.levels

    settings = Settings(
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nonconservative_flux.segmentpath(3),
        compute_dt=timestepping.adaptive(CFL=0.5),
        time_end=1.0,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=f"outputs/demo/swe-lvl{level}",
    )

    solver_price_c(
        mesh,
        model,
        settings,
        ode_solver_source=RKimplicit,
        # solver_price_c(mesh, model, settings, ode_solver_source=RK1
    )


# -

# ### Postprocessing

# +
# | code-fold: true
# | code-summary: "Postprocessing"


def append_to_plot(ax, X, Q, T, prefix=""):
    i_time = 50
    levels = Q.shape[1] - 3
    h = Q[i_time, 0, :]
    b = Q[i_time, -1, :]
    a = Q[i_time, 1 : 2 + levels, :] / h
    ax[0, 0].plot(X, h + b, label=f"h_{prefix} + b_{prefix}")
    ax[0, 0].plot(X, b, label=f"b_{prefix}")
    ax[0, 0].set_title(f"Time: {np.round(T[i_time], 1)}")
    for level in range(levels + 1):
        ax[1, 0].plot(X, a[level], label=f"a{level}_{prefix}")

    i_time = -1
    h = Q[i_time, 0, :]
    b = Q[i_time, -1, :]
    a = Q[i_time, 1 : 2 + levels, :] / h
    ax[0, 1].plot(X, h + b, label=f"h_{prefix} + b_{prefix}")
    ax[0, 1].plot(X, b, label=f"b_{prefix}")
    ax[0, 1].set_title(f"Time: {np.round(T[i_time], 1)}")
    for level in range(levels + 1):
        ax[1, 1].plot(X, a[level], label=f"a{level}_{prefix}")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    return ax


def load_and_plot(level, ax):
    filepath = os.path.join(f"outputs/demo/swe-lvl{level}", "Simulation.h5")
    X, Q, Qaux, T = io.load_timeline_of_fields_from_hdf5(filepath)
    # remove the boundary points
    Q = Q[:, :, :-2]
    X = X[:-2]
    ax = append_to_plot(ax, X.copy(), Q.copy(), T.copy(), prefix=f"lvl{level}")


run_model(3)
run_model(0)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
load_and_plot(0, ax)
load_and_plot(3, ax)
plt.show()


# # -
#
# # ## 1d Model
#
# # ### Model definition
#
# # +
#
# class ShallowMomentsSediment(Model):
#     """
#     Shallow Moments Sediment 1d
#
#     :gui:
#     - tab: model
#     - requires: [ 'mesh.dimension': 1 ]
#
#     """
#     def __init__(
#         self,
#         boundary_conditions,
#         initial_conditions,
#         aux_initial_conditions=IC.Constant(),
#         dimension=1,
#         fields=3,
#         aux_fields=0,
#         parameters = {},
#         parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
#         settings={},
#         settings_default={"topography": False, "friction": []},
#         basis=Basis()
#     ):
#         self.basis = basis
#         self.variables = register_sympy_attribute(fields, "q")
#         self.n_fields = self.variables.length()
#         self.levels = self.n_fields - 3
#         self.basis.compute_matrices(self.levels)
#         if type(fields) == type([]):
#             fields_smm = fields[:-1]
#         elif type(fields) == int:
#             fields_smm = fields-1
#         else:
#             assert False
#         self.smm = ShallowMoments(
#             dimension=dimension,
#             fields=fields_smm,
#             aux_fields=aux_fields,
#             parameters=parameters,
#             parameters_default = parameters_default,
#             boundary_conditions=boundary_conditions,
#             initial_conditions=initial_conditions,
#             aux_initial_conditions=aux_initial_conditions,
#         )
#         super().__init__(
#             dimension=dimension,
#             fields=fields,
#             aux_fields=aux_fields,
#             parameters=parameters,
#             parameters_default = parameters_default,
#             boundary_conditions=boundary_conditions,
#             initial_conditions=initial_conditions,
#             aux_initial_conditions=aux_initial_conditions,
#             settings={**settings_default, **settings},
#         )
#
#     def flux(self):
#         flux = Matrix([0 for i in range(self.n_fields)])
#         # smm flux
#         flux[:-1,0] = self.smm.sympy_flux[0]
#
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         a = [_ha/h for _ha in ha]
#         p = self.parameters
#         # bedload equation
#         ub = a[0]
#         def my_abs(x):
#             return sym.Abs(x)
#             # return sym.Piecewise((x, x>=0), (-x, True))
#         for i in range(self.levels):
#             ub += a[i+1]
#         tau = p.epsilon * p.rho * my_abs(ub) * ub
#         theta = (my_abs(tau)*p.d_s**2)/(p.g * (p.rho_s - p.rho) * d_s**3)
#         def pos(x):
#             # return x**2
#             return sym.Piecewise((x, x>0), (0, True))
#         def sign(x):
#             return sym.Piecewise((-1, x<0), (+1, x>0), (0, True))
#             # return sym.sign(x)
#             # return x
#         # theta = p.theta_c
#         # flux[-1, 0]  = p.Q * (sign(tau)*8/(1-p.phi) * (pos(theta - p.theta_c))**(3/2) )
#         # flux[-1, 0]  = p.Q * (sign(tau)*8/(1-p.phi) * (pos(theta - p.theta_c)))
#         flux[-1, 0]  = 10**(-6) * p.Q * (sign(tau)*8/(1-p.phi) * ((theta - p.theta_c)**2) )
#         return [-flux]
#
#     def nonconservative_matrix(self):
#         nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         a = [_ha / h for _ha in ha]
#         p = self.parameters
#         um = ha[0]/h
#         ub = a[0]
#         def my_abs(x):
#             return sym.Abs(x)
#             # return sym.Piecewise((-x, x<0), (x, True))
#         for i in range(self.levels):
#             ub += a[i+1]
#         tau = p.epsilon * p.rho * my_abs(ub) * ub
#         theta = (my_abs(tau)*p.d_s**2)/(p.g * (p.rho_s - p.rho) * d_s**3)
#         def pos(x):
#             return sym.Piecewise((x, x>0), (0, True))
#         def sign(x):
#             # return sym.Piecewise((-1, x<0), (+1, x>0), (0, True))
#             return sym.sign(x)
#         # theta = p.theta_c
#         # delta_q = (24 * p.Q)/(1-p.phi) * sign(tau) * p.epsilon / (p.g * (1/p.r - 1) * p.d_s) * (pos(theta - p.theta_c))**(1/2) * ub / h
#         delta_q = 10**(-6) * (24 * p.Q)/(1-p.phi) * sign(tau) * p.epsilon / (p.g * (1/p.r - 1) * p.d_s) * (sym.Abs(theta))**(1/2) * ub / h
#         nc[-1, 0]= - ub * delta_q
#         for l in range(self.levels+1):
#             nc[-1, l+1] = delta_q
#         return [nc]
#
#     def eigenvalues(self):
#         A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
#         for d in range(1, self.dimension):
#             A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
#         if self.levels > 1:
#             alpha_erase = self.variables[2:-1]
#             for alpha_i in alpha_erase:
#                 A = A.subs(alpha_i, 0)
#         for i in range(self.n_fields):
#             A[-1, i] = 0
#         return eigenvalue_dict_to_matrix(A.eigenvals())
#
#     def source(self):
#         out = Matrix([0 for i in range(self.n_fields)])
#         if self.settings.topography:
#             out += self.topography()
#         if self.settings.friction:
#             for friction_model in self.settings.friction:
#                 out += getattr(self, friction_model)()
#         return out
#
#
#     def newtonian(self):
#         """
#         :gui:
#             - requires_parameter: ('nu', 0.0)
#         """
#         assert "nu" in vars(self.parameters)
#         out = Matrix([0 for i in range(self.n_fields)])
#         out[:-1,0] = self.smm.newtonian()
#         return out
#
#     def manning(self):
#         """
#         :gui:
#             - requires_parameter: ('epsilon', 1000.0)
#         """
#         assert "epsilon" in vars(self.parameters)
#         out = Matrix([0 for i in range(self.n_fields)])
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         a = [_ha/h for _ha in ha]
#         p = self.parameters
#         ub = a[0]
#         def my_abs(x):
#             return sym.Abs(x)
#             # return sym.piecewise_fold(sym.Piecewise((-x, x<0), (x, True)))
#             # return (sym.Piecewise((-x, x<0), (x, True)))
#         for i in range(self.levels):
#             ub += a[i+1]
#         for k in range(1+self.levels):
#             for l in range(1+self.levels):
#                 out[1+k] += -(p.epsilon * self.basis.M[k,k]) * ub * my_abs(ub)
#         return out
# # -
#
# # ### Mesh and model construction
#
# # +
# #| code-fold: true
# #| code-summary: "Initialize model and mesh"
# mesh = petscMesh.Mesh.create_1d((-6, 6), 200)
#
# bcs = BC.BoundaryConditions(
#     [
#         BC.Extrapolation(physical_tag='left'),
#         BC.Extrapolation(physical_tag="right"),
#     ]
# )
#
# level = 0
# def custom_ic(x):
#     Q = np.zeros(1 + level+1+1, dtype=float)
#     Q[0] = np.where(x[0] < 0, 1., 0.95)
#     return Q
#
# ic = IC.UserFunction(custom_ic)
#
#
# d_s = 3.9*10**(-3)
# rho = 1000.
# rho_s = 1580.
# r = rho/rho_s
# g = 9.81
# phi = 0.47
# n = 1.
# d_50 = 1.
# # epsilon = (g * n**2)/((7/6)**2 * d_50**(1/3))
# epsilon = 0.0324
# Q = d_s * np.sqrt(g * (1/r-1) * d_s)
# # Q = 0.
# # theta_c = 0.047
# theta_c = 0
#
#
# fields = ['h'] + [f'hu_{l}' for l in range(level+1)] + ['b']
# model = ShallowMomentsSediment(
#     boundary_conditions=bcs,
#     initial_conditions=ic,
#     fields=fields,
#     settings={"friction": ['manning']},
#     parameters = {"nu":0.01, "g": g, "ez": 1.0, "d_s":d_s, "rho":rho, "rho_s":rho_s, "r": r, "phi": phi, "n":n, "d_50":d_50, "epsilon": epsilon, "Q": Q, "theta_c":theta_c}
# )
#
# # -
#
# # ### Display model
#
# # +
#
# #| code-fold: true
# #| code-summary: "Display model"
# display(Math(r'\large{' + 'Flux \, in \, x' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_flux[0])) + '}'))
# # display(Math(r'\large{' + 'Nonconservative \, matrix \, in \, x' + '}'))
# # display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_nonconservative_matrix[0])) + '}'))
# display(Math(r'\large{' + 'Eigenvalues' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_eigenvalues)) + '}'))
# display(Math(r'\large{' + 'Source' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_source)) + '}'))
# # -
#
# # ### Simulation
#
# # +
# #| code-fold: false
# #| code-summary: "Simulation"
# #| output: false
#
# settings = Settings(
#     reconstruction=recon.constant,
#     num_flux=flux.Zero(),
#     nc_flux=nonconservative_flux.segmentpath(3),
#     compute_dt=timestepping.adaptive(CFL=.1),
#     time_end=1.,
#     output_snapshots=100,
#     output_clean_dir=True,
#     output_dir="outputs/demo/swe",
# )
#
#
# solver_price_c(mesh, model, settings)
#
# # -
#
# # ### Postprocessing
#
# # +
#
# #| code-fold: true
# #| code-summary: "Postprocessing"
# filepath = os.path.join(settings.output_dir, 'Simulation.h5')
# X, Q, Qaux, T = io.load_timeline_of_fields_from_hdf5(filepath)
# # remove the boundary points
# Q = Q[:, :, :-2]
# X = X[:-2]
#
# # +
# #| code-fold: false
# #| code-summary: "Plot with Matplotlib"
# fig, ax = plt.subplots(1, 2, figsize=(12, 8))
# i_time = 0
# # ax[0].plot(X, Q[i_time, 0, :], label='h')
# # ax[0].plot(X, Q[i_time, 1, :]/Q[i_time, 0, :], label='u')
# ax[0].plot(X, Q[i_time, 2, :], label='b')
# ax[0].set_title(f'Time: {T[i_time]}')
#
# i_time = -1
# # ax[1].plot(X, Q[i_time, 0, :], label='h')
# # ax[1].plot(X, Q[i_time, 1, :]/Q[i_time, 0, :], label='u')
# ax[1].plot(X, Q[i_time, 2, :], label='b')
# ax[1].set_title(f'Time: {np.round(T[i_time], 1)}')
# plt.legend()
#
#
# # -
#
# # ## 2d Model
#
# # ### Model
#
# # +
#
# class ShallowMomentsSediment2d(Model):
#     def __init__(
#         self,
#         boundary_conditions,
#         initial_conditions,
#         dimension=2,
#         fields=3,
#         aux_fields=0,
#         parameters = {},
#         parameters_default={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
#         settings={},
#         settings_default={"topography": False, "friction": []},
#         basis=Basis()
#     ):
#         self.basis = basis
#         self.variables = register_sympy_attribute(fields, "q")
#         self.n_fields = self.variables.length()
#         self.levels = int((self.n_fields - 1)/2)-1
#         self.basis.compute_matrices(self.levels)
#         super().__init__(
#             dimension=dimension,
#             fields=fields,
#             aux_fields=aux_fields,
#             parameters=parameters,
#             parameters_default = parameters_default,
#             boundary_conditions=boundary_conditions,
#             initial_conditions=initial_conditions,
#             settings={**settings_default, **settings},
#         )
#
#     def flux(self):
#         offset = self.levels+1
#         flux_x = Matrix([0 for i in range(self.n_fields)])
#         flux_y = Matrix([0 for i in range(self.n_fields)])
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         hb = self.variables[1+self.levels+1:1+2*(self.levels+1)]
#         p = self.parameters
#         flux_x[0] = ha[0]
#         flux_x[1] = p.g * p.ez * h * h / 2
#         for k in range(self.levels+1):
#             for i in range(self.levels+1):
#                 for j in range(self.levels+1):
#                     # TODO avoid devision by zero
#                     flux_x[k+1] += ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
#         for k in range(self.levels+1):
#             for i in range(self.levels+1):
#                 for j in range(self.levels+1):
#                     # TODO avoid devision by zero
#                     flux_x[k+1+offset] += hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
#
#         flux_y[0] = hb[0]
#         flux_y[1+offset] = p.g * p.ez * h * h / 2
#         for k in range(self.levels+1):
#             for i in range(self.levels+1):
#                 for j in range(self.levels+1):
#                     # TODO avoid devision by zero
#                     flux_y[k+1] += hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
#         for k in range(self.levels+1):
#             for i in range(self.levels+1):
#                 for j in range(self.levels+1):
#                     # TODO avoid devision by zero
#                     flux_y[k+1+offset] += hb[i] * hb[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
#         return [flux_x, flux_y]
#
#     def nonconservative_matrix(self):
#         offset = self.levels+1
#         nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
#         nc_y = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         hb = self.variables[1+offset:1+offset+self.levels+1]
#         p = self.parameters
#         um = ha[0]/h
#         vm = hb[0]/h
#         for k in range(1, self.levels+1):
#             nc_x[k+1, k+1] += um
#             nc_y[k+1, k+1+offset] += um
#         for k in range(self.levels+1):
#             for i in range(1, self.levels+1):
#                 for j in range(1, self.levels+1):
#                     nc_x[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
#                     nc_y[k+1, i+1+offset] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
#
#         for k in range(1, self.levels+1):
#             nc_x[k+1+offset, k+1] += vm
#             nc_y[k+1+offset, k+1+offset] += vm
#         for k in range(self.levels+1):
#             for i in range(1, self.levels+1):
#                 for j in range(1, self.levels+1):
#                     nc_x[k+1+offset, i+1] -= hb[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
#                     nc_y[k+1+offset, i+1+offset] -= hb[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
#         return [nc_x, nc_y]
#
#     def eigenvalues(self):
#         # we delete heigher order moments (level >= 2) for analytical eigenvalues
#         offset = self.levels+1
#         A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
#         for d in range(1, self.dimension):
#             A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
#         alpha_erase = self.variables[2:2+self.levels]
#         beta_erase = self.variables[2+offset : 2+offset+self.levels]
#         for alpha_i in alpha_erase:
#             A = A.subs(alpha_i, 0)
#         for beta_i in beta_erase:
#             A = A.subs(beta_i, 0)
#         return eigenvalue_dict_to_matrix(A.eigenvals())
#
#     def source(self):
#         out = Matrix([0 for i in range(self.n_fields)])
#         if self.settings.topography:
#             out += self.topography()
#         if self.settings.friction:
#             for friction_model in self.settings.friction:
#                 out += getattr(self, friction_model)()
#         return out
#
#     def topography(self):
#         assert "dhdx" in vars(self.aux_variables)
#         assert "dhdy" in vars(self.aux_variables)
#         offset = self.levels+1
#         out = Matrix([0 for i in range(self.n_fields)])
#         h = self.variables[0]
#         p = self.parameters
#         dhdx = self.aux_variables.dhdx
#         dhdy = self.aux_variables.dhdy
#         out[1] = h * p.g * (p.ex - p.ez * dhdx)
#         out[1+offset] = h * p.g * (p.ey - p.ez * dhdy)
#         return out
#
#
#     def newtonian(self):
#         assert "nu" in vars(self.parameters)
#         out = Matrix([0 for i in range(self.n_fields)])
#         offset = self.levels+1
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         hb = self.variables[1+offset:1+self.levels+1+offset]
#         p = self.parameters
#         for k in range(1+self.levels):
#             for i in range(1+self.levels):
#                 out[1+k] += -p.nu/h * ha[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
#                 out[1+k+offset] += -p.nu/h * hb[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
#         return out
#
#
#     def slip(self):
#         assert "lamda" in vars(self.parameters)
#         assert "rho" in vars(self.parameters)
#         out = Matrix([0 for i in range(self.n_fields)])
#         offset = self.levels+1
#         h = self.variables[0]
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         hb = self.variables[1+offset:1+self.levels+1+offset]
#         p = self.parameters
#         for k in range(1+self.levels):
#             for i in range(1+self.levels):
#                 out[1+k] += -1./p.lamda/p.rho * ha[i]  / h / self.basis.M[k, k]
#                 out[1+k+offset] += -1./p.lamda/p.rho * hb[i]  / h / self.basis.M[k, k]
#         return out
#
#     def chezy(self):
#         assert "C" in vars(self.parameters)
#         out = Matrix([0 for i in range(self.n_fields)])
#         offset = self.levels+1
#         h = self.variables[0]
#         ha = self.variables[1:1+self.levels+1]
#         hb = self.variables[1+offset:1+self.levels+1+offset]
#         p = self.parameters
#         tmp = 0
#         for i in range(1+self.levels):
#             for j in range(1+self.levels):
#                 tmp += ha[i] * ha[j] / h / h + hb[i] * hb[j] / h / h
#         sqrt = sympy.sqrt(tmp)
#         for k in range(1+self.levels):
#             for l in range(1+self.levels):
#                 out[1+k] += -1./(p.C**2 * self.basis.M[k,k]) * ha[l] * sqrt / h
#                 out[1+k+offset] += -1./(p.C**2 * self.basis.M[k,k]) * hb[l] * sqrt / h
#         return out
# # -
#
# # ### Mesh
#
# # +
# #| code-fold: false
# #| code-summary: "Load Gmsh mesh"
#
# # mesh = petscMesh.Mesh.from_gmsh(os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"))
# mesh = petscMesh.Mesh.from_gmsh(os.path.join(main_dir, "meshes/quad_2d/mesh_fine.msh"))
# # mesh = petscMesh.Mesh.from_gmsh(os.path.join(main_dir, "meshes/quad_2d/mesh_finest.msh"))
# # mesh = petscMesh.Mesh.from_gmsh(os.path.join(main_dir, "meshes/triangle_2d/mesh_finest.msh"))
# print(f"physical tags in gmsh file: {mesh.boundary_conditions_sorted_names}")
#
# # -
#
# # ### Model
#
# # +
# #| code-summary: "Model"
# #| code-fold: false
#
# bcs = BC.BoundaryConditions(
#     [
#         BC.Periodic(physical_tag='left', periodic_to_physical_tag='right'),
#         BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
#         BC.Periodic(physical_tag='top', periodic_to_physical_tag='bottom'),
#         BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
#     ]
# )
#
#
# def custom_ic(x):
#     Q = np.zeros(3, dtype=float)
#     Q[0] = np.where(x[0]**2 + x[1]**2 < 0.1, 2., 1.)
#     return Q
#
# ic = IC.UserFunction(custom_ic)
#
#
# model = ShallowMomentsSediment2d(
#     boundary_conditions=bcs,
#     initial_conditions=ic,
#     fields=['h', 'hu', 'hv'],
#     settings={"friction": ["chezy"]},
#     parameters = {"C": 1.0, "g": 9.81, "ez": 1.0}
# )
#
#
#
# # -
#
# # ### Automatic tex generation
#
# #| code-summary: "Display Model"
# #| code-fold: true
# display(Math(r'\large{' + 'Flux \, in \, x' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_flux[0])) + '}'))
# display(Math(r'\large{' + 'Flux \, in \, y' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_flux[1])) + '}'))
# display(Math(r'\large{' + 'Nonconservative \, matrix \, in \, x' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_nonconservative_matrix[0])) + '}'))
# display(Math(r'\large{' + 'Nonconservative \, matrix \, in \, y' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_nonconservative_matrix[1])) + '}'))
# display(Math(r'\large{' + 'Quasilinear \, matrix \, in \, x' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_quasilinear_matrix[0])) + '}'))
# display(Math(r'\large{' + 'Quasilinear \, matrix \, in \, y' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_quasilinear_matrix[1])) + '}'))
# display(Math(r'\large{' + 'Eigenvalues' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_eigenvalues)) + '}'))
# display(Math(r'\large{' + 'Source' + '}'))
# display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_source)) + '}'))
#
# # ### Simulation
#
# # +
# #| code-fold: false
# #| code-summary: "Simulation"
# #| output: false
#
# settings = Settings(
#     reconstruction=recon.constant,
#     num_flux=flux.Zero(),
#     nc_flux=nonconservative_flux.segmentpath(3),
#     compute_dt=timestepping.adaptive(CFL=.45),
#     time_end=1.,
#     output_snapshots=100,
#     output_clean_dir=True,
#     output_dir="outputs/demo/swe2d",
# )
#
#
# solver_price_c(mesh, model, settings)
#
# # -
#
# # ### Postprocessing
#
# #| code-fold: true
# #| code-summary: "Postprocessing"
# io.generate_vtk(os.path.join(os.path.join(main_dir, settings.output_dir), f'{settings.name}.h5'))
# out_0 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.0.vtk'))
# out_10 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.10.vtk'))
# out_98 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.98.vtk'))
# field_names = out_0.cell_data.keys()
# print(f'Field names: {field_names}')
#
# # +
# #| code-fold: true
# #| code-summary: "Plot VTK"
# p = pv.Plotter(shape=(1,3), notebook=True)
#
# p.subplot(0, 0)
# p.add_mesh(out_0, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=0)'})
# p.enable_parallel_projection()
# p.enable_image_style()
# p.view_xy()
#
# p.subplot(0, 1)
# p.add_mesh(out_10, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=0.1)'})
# p.enable_parallel_projection()
# p.enable_image_style()
# p.view_xy()
#
# p.subplot(0, 2)
# p.add_mesh(out_98, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=1)'})
# p.enable_parallel_projection()
# p.enable_image_style()
# p.view_xy()
#
# p.show(jupyter_backend='static')
#
