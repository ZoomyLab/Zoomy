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

class GN(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=2,
        aux_fields=['dq0dt','D1', 'dtF0', 'dtF1'],
        parameters={},
        parameters_default={"g": 9.81},
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
        nu = self.variables[0]
        u = self.variables[1]

        param = self.parameters
        
        fx[0] = u + nu * u
        fx[1] = 1/2 * u**2

        return [fx]

    

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_fields)])
        dq0dt = self.aux_variables.dq0dt
        D1 = self.aux_variables.D1
        dtF0 = self.aux_variables.dtF0
        dtF1 = self.aux_variables.dtF1


        R[0] = dq0dt + dtF0
        R[1] = D1 + dtF1
        return R
        
