import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, Abs, sqrt

from sympy import zeros, ones

from attr import define
from typing import Optional
from types import SimpleNamespace

from library.model.boundary_conditions import BoundaryConditions, Periodic
from library.model.initial_conditions import InitialConditions, Constant
from library.misc.custom_types import FArray
from library.misc.misc import vectorize  # type: ignore
from library.model.models.shallow_water import ShallowWater, ShallowWater2d
from library.model.models.base import eigenvalue_dict_to_matrix


@define(slots=True, frozen=False, kw_only=True)
class ShallowWaterTopo(ShallowWater):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
    ):
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
        hu = self.variables[1]
        p = self.parameters
        flux[0] = hu
        flux[1] = hu**2 / h 
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        p = self.parameters
        nc[1,0] = -p.g * p.ez * h 
        nc[1,2] =  -p.g * p.ez * h 
        return [nc]



class ShallowWaterTopo2d(ShallowWater2d):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters={},
        parameters_default={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
    ):
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings=settings,
            settings_default=settings_default,
        )

    def flux(self):
        fx = Matrix([0 for i in range(self.n_fields)])
        fy = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        p = self.parameters
        fx[0] = hu
        fx[1] = hu**2 / h 
        fx[2] = hu * hv / h
        fy[0] = hv
        fy[1] = hu * hv / h
        fy[2] = hv**2 / h
        return [fx, fy]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        nc_y = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        p = self.parameters
        nc_x[1,0] = -p.g * p.ez * h 
        nc_x[1,3] = -p.g * p.ez * h 
        nc_y[2,0] = -p.g * p.ez * h 
        nc_y[2,3] = -p.g * p.ez * h 
        return [nc_x, nc_y]

