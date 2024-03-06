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
# from sympy import *
from sympy import zeros, ones
from sympy import bspline_basis, bspline_basis_set
from sympy.abc import x
import h5py

from library.model.models.base import register_sympy_attribute, eigenvalue_dict_to_matrix
from library.model.models.base import Model


from library.model import *

from attr import define
from sympy import integrate, diff

from sympy import legendre

class ShearShallowFlow(Model):
    """
    Shallow Moments 

    :gui: 
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """
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
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=3,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        P11 = self.variables[2]
        p = self.parameters
        flux_x[0] = hu
        flux_x[1] = hu * u + p.g * h**2/2 + P11
        flux_x[2] = u * P11

        return [flux_x]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        P11 = self.variables[2]

        nc_x[2, 0] = -P11/h 
        nc_x[2, 1] = +P11/h
        return [nc_x]

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        # out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        P11 = self.variables[2]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3*P11)

        evs[0] = u
        evs[1] = u + a
        evs[2] = u - a

        return evs

    def friction_paper(self):
        assert "phi" in vars(self.parameters)   
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        P11 = self.variables[2]
        p = self.parameters

        abs_u = sympy.sqrt(u**2)
        trace_P = P11 
        grad_b = [- sympy.tan(p.theta)]
        # alpha = max(0, p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2))
        alpha = sympy.Piecewise((p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2), p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2) > 0), (0, p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2) <= 0))
        D11 = -2 * alpha / h * abs_u**3 * P11
        Q =  alpha * trace_P * abs_u**3


        out[1] = -h * p.g * grad_b[0] - p.Cr * u * abs_u
        out[2] = D11
        return out

    
        




class ShearShallowFlow2d(Model):
    """
    Shallow Moments 2d

    :gui: 
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=6,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_fields)])
        flux_y = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu/h
        v = hv/h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters
        flux_x[0] = hu
        flux_x[1] = hu * u + p.g * h**2/2 + P11
        flux_x[2] = hu * v + h * P12
        flux_x[3] = u * P11
        flux_x[4] = u * P12
        flux_x[5] = 0

        flux_y[0] = hv
        flux_y[1] = hu*v + P12
        flux_y[2] = hv*v + p.g * h**2/2 + P22
        flux_y[3] = 0
        flux_y[4] = v * P12
        flux_y[5] = v * P22
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        nc_y = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu/h
        v = hv/h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]

        nc_x[3, 0] = -P11/h 
        nc_x[3, 1] = +P11/h
        nc_y[3, 0] = -2*P11/h
        nc_y[3, 1] = +2*P11/h
        nc_y[3, 3] = +v

        nc_x[4, 0] = -P11/h
        nc_x[4, 2] = +P11/h
        nc_y[4, 0] = -P22/h
        nc_y[4, 1] = +P22/h

        nc_x[5, 0] = -2*P12/h
        nc_x[5, 2] = +2*P12/h
        nc_y[5, 0] = -P22/h
        nc_y[5, 2] = +P22/h
        nc_x[5, 5] = u
        return [nc_x, nc_y]

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        # out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu/h
        v = hv/h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3*P11)

        evs[0] = u
        evs[1] = u
        evs[2] = u + b
        evs[3] = u - b
        evs[4] = u + a
        evs[5] = u - a

        return evs

    def friction_paper(self):
        assert "phi" in vars(self.parameters)   
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu/h
        v = hv/h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters

        abs_u = sympy.sqrt(u**2 + v**2)
        trace_P = P11 + P22
        grad_b = [- sympy.tan(p.theta), 0]
        alpha = max(0, p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2))
        D11 = -2 * alpha / h * abs_u**3 * P11
        D12 = -2 * alpha / h * abs_u**3 * P12
        D22 = -2 * alpha / h * abs_u**3 * P22
        Q =  alpha * trace_P * abs_u**3


        out[1] = -h * p.g * grad_b[0] - p.Cr * u * abs_u
        out[2] = -h * p.g * grad_b[1] - p.Cr * v * abs_u
        out[3] = h * D11
        out[4] = h * D12
        out[5] = h * D22
        return out

    
        


