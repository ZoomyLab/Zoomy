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


from library.model.models.base import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from library.model.models.base import Model
import library.model.initial_conditions as IC

class Poisson(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=1,
        aux_variables=['ddTdxx'],
        parameters={},
        _default_parameters={},
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
        
    def residual(self):
        R = Matrix([0 for i in range(self.n_variables)])
        T = self.variables[0]
        ddTdxx = self.aux_variables.ddTdxx
        param = self.parameters

        R[0] = - ddTdxx + 2
        return R
        

    
    
