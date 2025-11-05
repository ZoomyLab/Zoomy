import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, sqrt
from sympy.abc import x

from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify, Rational


from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
import zoomy_core.model.initial_conditions as IC
import zoomy_core.model.boundary_conditions as BC

class GN(Model):
    def __init__(
        self,
        boundary_conditions= None,
        initial_conditions=None,
        dimension=1,
        fields=2,
        # D = h^3 / 3 * (dt * dx * u + u * dx^2 u - (dx u)^2)
        aux_variables=['dD_dx'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )
        
    def flux(self):
        fx = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]

        param = self.parameters
        
        fx[0] = hu
        fx[1] = hu**2 / h + 1/2 * param.g * h**2 
        return [fx]

    

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])
        dD_dx = self.aux_variables.dD_dx


        R[0] = 0
        R[1] = dD_dx
        return R
        
