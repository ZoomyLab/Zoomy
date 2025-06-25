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

class Poisson(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=1,
        aux_fields=['dTdt', 'dTdx', 'ddTdxx', 'kappa'],
        parameters={},
        parameters_default={},
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
        T = self.variables[0]
        dTdt = self.aux_variables.dTdt
        dTdx = self.aux_variables.dTdx
        ddTdxx = self.aux_variables.ddTdxx
        param = self.parameters
        kappa = self.aux_variables.kappa

        R[0] = - kappa * ddTdxx
        return R
        

    
    
