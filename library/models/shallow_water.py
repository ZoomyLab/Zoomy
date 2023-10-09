import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, Abs

from sympy import zeros, ones

from attr import define
from typing import Optional
from types import SimpleNamespace

from library.boundary_conditions import BoundaryConditions, Periodic
from library.initial_conditions import InitialConditions, Constant
from library.custom_types import FArray
from library.misc import vectorize  # type: ignore
from library.models.base import Model


@define(slots=True, frozen=False, kw_only=True)
class ShallowWater(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=2,
        aux_fields=0,
        parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={"topography": True, "friction": ['manning']},
    ):
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings=settings,
        )

    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        p = self.parameters
        flux[0] = hu
        flux[1] = hu**2 / h + p.g * p.ez * h * h / 2
        return [flux]

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
        hu = self.variables[1]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out
    
    def newtonian(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        p = self.parameters
        out[1] = -p.nu * hu/h
        return out

    def manning(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        p = self.parameters
        out[1] = -p.g * (p.nu**2) * hu * Abs(u)**(7 / 3)
        return out


class ShallowWater2d(Model):
    settings = {"topography": False, "friction": []}

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={"topography": False, "friction": []},
    ):
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings=settings,
        )

    def flux(self):
        fx = Matrix([0 for i in range(self.n_fields)])
        fy = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        p = self.parameters
        fx[0] = hu
        fx[1] = hu**2 / h + p.g * p.ez * h * h / 2
        fx[2] = hu * hv / h
        fy[0] = hv
        fy[1] = hu * hv / h
        fy[2] = hv**2 / h + p.g * p.ez * h * h / 2
        return [fx, fy]

    def source(self):
        return zeros(self.n_fields, 1)
