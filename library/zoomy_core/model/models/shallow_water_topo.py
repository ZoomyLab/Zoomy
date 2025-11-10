import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, Abs, sqrt

from sympy import zeros, ones

from attr import define, field
from typing import Optional
from types import SimpleNamespace

from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.model.initial_conditions import InitialConditions, Constant
from zoomy_core.misc.custom_types import FArray
from zoomy_core.model.basemodel import Model, eigenvalue_dict_to_matrix

from zoomy_core.misc.misc import Zstruct

@define(frozen=True, slots=True, kw_only=True)
class ShallowWaterEquationsWithTopo1D(Model):
    dimension: int = 1
    variables: Zstruct = field(default=3)
    aux_variables: Zstruct = field(default=0)

@define(frozen=True, slots=True, kw_only=True)
class ShallowWaterEquationsWithTopo(Model):
    dimension: int=2
    variables: Zstruct = field(default=4)
    aux_variables: Zstruct = field(default=0)
    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0}
        )
    
    def __attrs_post_init__(self):
        if type(self.variables)==int:
            object.__setattr__(self, "variables",self.dimension+2)
        super().__attrs_post_init__()
        
    def compute_hinv(self):
        h = self.variables[1]
        return 1 / h
        
    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = self.compute_hinv()
        U = [self.variables[2 + i] * hinv for i in range(dim)]
        return b, h, U, hinv


    def project_2d_to_3d(self):
        out = Matrix([0 for i in range(5)])
        dim = self.dimension
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]
        b, h, U, hinv = self.get_primitives()
        rho_w = 1000.
        g = 9.81
        out[0] = h
        out[1] = U[0]
        out[2] = 0 if dim == 1 else U[1]
        out[3] = 0
        out[4] = rho_w * g * h * (1-z)
        return out

        

    def flux(self):
        dim = self.dimension
        h = self.variables[1]
        U = Matrix([hu / h for hu in self.variables[2:2+dim]])
        g = self.parameters.g
        I = Matrix.eye(dim)
        F = Matrix.zeros(self.variables.length(), dim)
        F[1, :] = (h * U).T
        F[2:, :] = h * U * U.T + g/2 * h**2 * I
        return [F[:, d] for d in range(dim)]
    
    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        nc_y = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        b, h, U, hinv = self.get_primitives()
        p = self.parameters
        nc_x[1, 0] = p.g * h 
        nc_y[1, 0] = p.g * h
        return [nc_x, nc_y][:self.dimension]

    
    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out
    
    def chezy(self):
        assert "C" in vars(self.parameters)
        dim = self.dimension
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hU = self.variables[1:1+dim]
        U = Matrix([hu / h for hu in hU])
        p = self.parameters
        u_sq = sqrt(U.dot(U))
        out[2:2+dim] = -1.0 / p.C**2 * U * u_sq
        return out
    
        return self._simplify(eigenvalue_dict_to_matrix(A.eigenvals()))
    