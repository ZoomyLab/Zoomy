import numpy as np
import os
import logging
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
import matplotlib.pyplot as plt
import inspect
import pytest
from copy import deepcopy
from scipy import interpolate
import sympy
from sympy import Symbol, Matrix, lambdify
from sympy import *
from sympy import zeros, ones

from library.model.models.base import register_sympy_attribute
from library.model.models.base import Model


# from library.solver.baseclass import BaseYaml
from library.model import *
# from library.solver.boundary_conditions import *
# import library.solver.initial_condition as initial_condition
# import library.solver.smm_model as smm
# import library.solver.smm_model_hyperbolic as smmh
# import library.solver.smm_model_exner as smm_exner
# import library.solver.smm_model_exner_hyperbolic as smm_exner_hyper

# from library.solver.baseclass import BaseYaml  # nopep8

# main_dir = os.getenv("SMPYTHON")
# eps = 10 ** (-10)

from attr import define
from sympy import integrate, diff
from sympy.abc import x

from sympy import legendre

def legendre_shifted(order, x):
    return legendre(order, 2*x-1)

class Basis():
    def __init__(self, basis=legendre_shifted):
        self.basis = basis

    """ 
    Compute <phi_k, phi_i>
    """
    def M(self, k, i):
        return integrate(self.basis(k, x) * self.basis(i, x), (x, 0, 1))

    """ 
    Compute <phi_k, phi_i, phi_j>
    """
    def A(self, k, i, j):
        return integrate(self.basis(k, x) * self.basis(i, x) * self.basis(j, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, phi_j, int(phi)_j>
    """
    def B(self, k, i, j):
        return integrate(diff(self.basis(k, x), x) * integrate(self.basis(j, x), x) * self.basis(i, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, (phi')_j>
    """
    def D(self, k, i):
        return integrate(diff(self.basis(k, x), x) * diff(self.basis(i, x), x), (x, 0, 1))

class ShallowMoments(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=2,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 2
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings=settings,
            settings_default=settings_default,
        )

    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # avoid devision by zero 
                    flux[k+1] += ha[i] * ha[j] / h * self.basis.A(k, i, j) / self.basis.M(k, k)
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:]
        p = self.parameters
        um = ha[0]/h
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -= ha[j]/h*self.basis.B(k, i, j)/self.basis.M(k, k)

        return [nc]

    # def eigenvalues(self):
    #     return None