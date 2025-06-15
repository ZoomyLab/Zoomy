import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, sqrt
from sympy.abc import x

from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify

from library.model import *
from library.model.models.base import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from library.model.models.base import Model
import library.model.initial_conditions as IC
from library.model.models.basisfunctions import *
from library.model.models.basismatrices import *

class VAMHyperbolic(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=6,
        aux_fields=['hw2'],
        parameters={},
        parameters_default={"g": 1},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )
        
    def flux(self):
        fx = Matrix([0 for i in range(self.n_fields)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        
        fx[0] = hu0
        fx[1] = hu0 * u0 + 1/3 * hu1 * u1 * u1
        fx[2] = hu0 * w0 + 1/3 * hu1 * w1
        fx[3] = 2*hu0 * u1
        fx[4] = hu0 * w1 + u1 * (hw0 + 2/5*hw2)
        
        return [fx]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])

        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 / h

        nc[1, 0] = param.g * h
        nc[3, 2] = u0
        nc[4, 2] = - 1/5 * w2 + w0
        nc[1, 5] = param.g * h
        return [-nc]
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h

        ev[0] = u0
        ev[1] = u0 + 1/sqrt(3) * u1
        ev[2] = u0 - 1/sqrt(3) * u1
        ev[3] = u0 + sqrt(param.g * h + u1**2)
        ev[4] = u0 - sqrt(param.g * h + u1**2)
        ev[5] = 0
        
        return ev


class VAMPoisson(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=8,
        aux_fields=['dhdt', 'dhu0dt', 'dhu1dt', 'dhw0dt', 'dhw1dt', 'dhdx', 'dhu0dx', 'dhu1dx', 'dhw0dx', 'dhw1dx', 'dhp0dx', 'dhp1dx', 'dbdx', 'du0dx', 'hw2', 'lap_p0', 'lap_p1'],
        parameters={},
        parameters_default={"g": 1},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_fields)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        b = self.variables[5]
        p0 = self.variables[6]
        p1 = self.variables[7]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 /h  

        dhdt   = self.aux_variables.dhdt   
        dhu0dt = self.aux_variables.dhu0dt 
        dhu1dt = self.aux_variables.dhu1dt 
        dhw0dt = self.aux_variables.dhw0dt 
        dhw1dt = self.aux_variables.dhw1dt 
        dhdx   = self.aux_variables.dhdx   
        dhu0dx = self.aux_variables.dhu0dx 
        dhu1dx = self.aux_variables.dhu1dx 
        dhw0dx = self.aux_variables.dhw0dx 
        dhw1dx = self.aux_variables.dhw1dx 
        dhp0dx = self.aux_variables.dhp0dx 
        dhp1dx = self.aux_variables.dhp1dx 
        dbdx   = self.aux_variables.dbdx   
        du0dx  = self.aux_variables.du0dx
        lap_p0 = self.aux_variables.lap_p0
        lap_p1 = self.aux_variables.lap_p1

        R[0] = dhdt 
        R[1] = dhu0dt + dhp0dx + 2 * p1 * dbdx 
        R[2] = dhu1dt -2*p1
        R[3] = dhw0dt + dhp1dx - (3*p0 - p1)*dhdx  -6*(p0-p1)*dbdx
        R[4] = 6*(p0-p1)
        R[5] = 0.
        R[6] = h*du0dx + 1/3 * dhu1dx + 1/3 * u1 * dhdx + 2*(w0 - u0 * dbdx)
        R[7] = h * du0dx + u1*dhdx + 2*(u1 * dbdx - w1)
        
        delta = 0.000001
        R[1] += + delta * h* (lap_p0 + lap_p1)
        R[2] += + delta * h* (lap_p0 + lap_p1)
        R[3] += + delta * h* (lap_p0 + lap_p1)
        R[4] += + delta * h* (lap_p0 + lap_p1)
        R[6] += + delta * h* (lap_p0 + lap_p1)
        R[7] += + delta * h* (lap_p0 + lap_p1)

        return R
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_fields)])
        return ev
    
    
