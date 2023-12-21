import numpy as np
import os
# import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, Abs, sqrt

from sympy import zeros, ones

from attr import define
from typing import Optional
from types import SimpleNamespace

from library.boundary_conditions import BoundaryConditions, Periodic
from library.initial_conditions import InitialConditions, Constant
from library.custom_types import FArray
# from library.misc import vectorize  # type: ignore
from library.models.base import Model

class Advection(Model):
    def flux(self):
        # assume that the first variables.length() parameters are the corresponding advection speeds
        assert self.parameters.length() >= self.variables.length()
        # assert self.n_fields == self.dimension
        if self.dimension == 1:
            F = Matrix([0 for i in range(self.n_fields)])
            for i_field in range(self.n_fields):
                F[i_field] = self.variables[i_field] * self.parameters[0]
            return [F]
        elif self.dimension == 2:
            F = Matrix([0 for i in range(self.n_fields)])
            G = Matrix([0 for i in range(self.n_fields)])
            for i_field in range(self.n_fields):
                F[i_field] = self.variables[i_field] * self.parameters[0]
                G[i_field] = self.variables[i_field] * self.parameters[1]
            return [F, G]
        elif self.dimension == 3:
            F = Matrix([0 for i in range(self.n_fields)])
            G = Matrix([0 for i in range(self.n_fields)])
            H = Matrix([0 for i in range(self.n_fields)])
            for i_field in range(self.n_fields):
                F[i_field] = self.variables[i_field] * self.parameters[0]
                G[i_field] = self.variables[i_field] * self.parameters[1]
                H[i_field] = self.variables[i_field] * self.parameters[2]
            return [F, G, H]
        else:
            assert False