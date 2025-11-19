# ---
# title: "SWEL Parabolic Bowl"
# author: Ingo Steldermann
# date: 11/16/2025
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %%
# | code-fold: true
# | code-summary: "Load packages"
# | output: false


import os
import numpy as np
import os
import numpy as np
from sympy import Matrix, sqrt, Piecewise
import sympy as sp
import pytest
from attr import define, field
from sympy import MutableDenseNDimArray as Arr


from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.model.basemodel import Model, eigenvalue_dict_to_matrix
import zoomy_core.model.initial_conditions as IC
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.misc.misc import Zstruct, ZArray
import zoomy_core.misc.misc as misc
import zoomy_firedrake.firedrake_solver_animate_amr as dg_amr
import zoomy_firedrake.firedrake_solver as dg



# %%
@define(frozen=True, slots=True, kw_only=True)
class SWE(Model):
    dimension: int = 2
    variables: Zstruct = field(init=False)
    aux_variables: Zstruct = field(default=1)
    _default_parameters: dict = field(
        init=False, factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0, "rho": 1000.0, "n": 0., "eps":1e-4}
    )
    
    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", self.dimension + 2)
        super().__attrs_post_init__()

    def project_2d_to_3d(self):
        out = ZArray.zeros(6)
        p = self.parameters
        dim = self.dimension
        z = self.position[2]
        b = self.aux_variables[0]
        h = self.variables[1]
        U = [hu / h for hu in self.variables[2 : 2 + dim]]
        out[0] = b
        out[1] = h
        out[2] = U[0]
        out[3] = 0 if dim == 1 else U[1]
        out[4] = 0
        out[5] = p.rho * p.g * h * (1 - z)
        return out
    
    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1/h
        U = Matrix([hu * hinv for hu in self.variables[2 : 2 + dim]])
        return b, h, U, hinv

    def flux(self):
        dim = self.dimension
        b, h, U, hinv = self.get_primitives()
        g = self.parameters.g
        I = Matrix.eye(dim)
        F = Matrix.zeros(self.variables.length(), dim)
        # F[1, :] = h * U.T
        F[1, :] = sp.Matrix(self.variables[2: 2 + dim]).T
        # F[2:, :] = h * U * U.T + g / 2 * h**2 * I
        F[2:, :] = h * U * U.T
        return ZArray(F)
    
    def nonconservative_matrix(self):
        dim = self.dimension
        b, h, U, hinv = self.get_primitives()
        U = Matrix([hu * hinv for hu in self.variables[2 : 2 + dim]])
        g = self.parameters.g
        N = ZArray.zeros(self.n_variables, self.n_variables, dim)
        for d in range(dim):
            N[2+d, 0, d] = g * h # g * h * grad(b)
            N[2+d, 1, d] = g * h # g * h * grad(h)
        return ZArray(N)
    
    def source(self):
        eps = 1e-4
        dim = self.dimension
        _, _, U, _ = self.get_primitives()
        hU = Matrix(self.variables[2 : 2 + dim])
        hinv = self.aux_variables[0]
        # Uold = Matrix(self.aux_variables[1 : 1 + dim])
        g = self.parameters.g
        n = self.parameters.n
        abs_hu = sqrt(hU.dot(hU) + eps)
        S = Matrix.zeros(self.n_variables, 1)
        # S[2:, 0] = -n**2 * g * hinv**(1/3) * U[:, 0] * abs_u
        S[2:, 0] = n**2 * g  * (hinv**(7/3) + eps) * hU[:, 0] * abs_hu
        return ZArray(S).reshape(self.n_variables,)
    
@define(frozen=True, slots=True, kw_only=True)
class NumericSWE(SWE):
    disable_differentiation: bool = False
    
    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = self.aux_variables[0]
        U = Matrix([hu * hinv for hu in self.variables[2 : 2 + dim]])
        
        return b, h, U, hinv
    
    def eigenvalues(self):
        ev = super().eigenvalues()
        h = self.variables[1]
        return sp.Function('conditional')(h > self.parameters.eps, ev, ZArray.zeros(*ev.shape))
    
    # def source(self):
    #     delta = self.parameters.eps  # or smaller
    #     h = self.variables[1]
    #     smooth = sp.Rational(1,2)*(1 + sp.tanh((h - self.parameters.eps)/delta))

    #     S = super().source()
    #     zeros = ZArray.zeros(*S.shape)

    #     return smooth * S + (1 - smooth) * zeros
    
    def source(self):
        delta = self.parameters.eps  # or smaller
        h = self.variables[1]
        smooth = sp.Rational(1,2)*(1 + sp.tanh((h - self.parameters.eps)/delta))

        S = super().source()
        S2 = sp.Matrix(S)
        S2 = S2.subs({h: self.parameters.eps})
        Sreg = ZArray.zeros(*S.shape)
        for i in range(self.n_variables):
            Sreg[i] = S2[i,0]
        zeros = ZArray.zeros(*S.shape)
        # return Sreg
        # return smooth * S + (1 - smooth) * Sreg
        return sp.Function('conditional')(h > self.parameters.eps, -S, zeros)
        # return -S
    
    def source_jacobian_wrt_aux_variables(self):
        return ZArray.zeros(
            self.n_variables
        )
    
    def source_jacobian_wrt_variables(self):
        return ZArray.zeros(
            self.n_variables
        )
                



# %% [markdown] vscode={"languageId": "raw"}
# # Transformation to UFL Code (Medium)

# %% [markdown]
# ### Map from Sympy to UFL

# %%


bcs = BC.BoundaryConditions(
    [
        BC.Extrapolation(tag="wall"),
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ]
)

### Initial condition
def ic_q(x):
    R = 3
    r = np.sqrt((x[0])**2 + (x[1])**2)
    # b = 0.1 * x[0] + 0.5 * np.sin(2 * np.pi * x[0] / 5)
    # b = 0.3 * x[0]
    # b = r**2 / 100 * 3
    b = 0
    h = np.where(r <= R, 2., 0.) -b
    h = np.where(h <= 0, 0, h)
    return np.array([b, h , 0.*x[0], 0.*x[0]])

ic = IC.UserFunction(ic_q)

model = NumericSWE(
    dimension=2,
    boundary_conditions=bcs,
    initial_conditions=ic,
)

settings = Settings(name="Firedrake", output=Zstruct(directory="outputs/firedrake", snapshots=10000, filename='dg', clean_directory=True))


# %%
import ufl 
IdentityMatrix = ufl.as_tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

solver = dg.FiredrakeHyperbolicSolver(settings=settings, time_end = 10.0, CFL=0.2, IdentityMatrix=IdentityMatrix)
# solver = dg_amr.FiredrakeHyperbolicSolverAMR(settings=settings, time_end = 10.0, CFL=0.45, IdentityMatrix=IdentityMatrix, refine_every=20, enable_amr=False)

# %%
main_dir = misc.get_main_directory()
path_to_mesh = os.path.join(main_dir, "meshes", "square", "mesh.msh")
solver.solve(path_to_mesh, model)

# %%
