import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose

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
        parameters={"g": 1.0, "ez": 1.0},
    ):
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
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
        return zeros(self.n_fields, 1)


class ShallowWater2d(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters={"g": 1.0, "ez": 1.0},
    ):
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
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
