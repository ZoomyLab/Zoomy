import numpy as np
import os
# import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, Abs, sqrt

from sympy import zeros, ones

from attr import define
from typing import Optional
from types import SimpleNamespace

from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.model.initial_conditions import InitialConditions, Constant
from zoomy_core.misc.custom_types import FArray

# from zoomy_core.misc import vectorize  # type: ignore
from zoomy_core.model.basemodel import Model


class Advection(Model):
    def flux(self):
        if self.dimension == 1:
            F = Matrix([0 for i in range(self.n_variables)])
            for i_field in range(self.n_variables):
                F[i_field] = self.variables[i_field] * self.parameters[0]
            return [F]
        elif self.dimension == 2:
            F = Matrix([0 for i in range(self.n_variables)])
            G = Matrix([0 for i in range(self.n_variables)])
            for i_field in range(self.n_variables):
                F[i_field] = self.variables[i_field] * self.parameters[0]
                G[i_field] = self.variables[i_field] * self.parameters[1]
            return [F, G]
        elif self.dimension == 3:
            F = Matrix([0 for i in range(self.n_variables)])
            G = Matrix([0 for i in range(self.n_variables)])
            H = Matrix([0 for i in range(self.n_variables)])
            for i_field in range(self.n_variables):
                F[i_field] = self.variables[i_field] * self.parameters[0]
                G[i_field] = self.variables[i_field] * self.parameters[1]
                H[i_field] = self.variables[i_field] * self.parameters[2]
            return [F, G, H]
        else:
            assert False

    # def eigenvalues(self):
    #     assert self.normal.shape[0] == self.parameters.shape[0]
    #     ev = self.normal[0] * self.parameters[0]
    #     for d in range(1, self.dimension):
    #         ev += self.normal[d] * self.parameters[d]
    #     self.sympy_eigenvalues = Matrix[[ev for i in range(self.n_variables)]]
