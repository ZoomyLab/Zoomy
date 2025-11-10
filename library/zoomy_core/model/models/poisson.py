import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, sqrt
from sympy.abc import x
from attr import define, field

from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
import zoomy_core.model.initial_conditions as IC


@define(kw_only=True, slots=True, frozen=True)
class Poisson(Model):
    dimension: int = 1
    variables: Zstruct = field(init=False, default=1)
    aux_variables: Zstruct = field(factory = lambda: ['ddTdxx', 'ddTdyy', 'ddTdzz'])
    
            
    def residual(self):
        R = Matrix([0 for i in range(self.n_variables)])
        T = self.variables[0]
        ddTdxx = self.aux_variables.ddTdxx
        param = self.parameters

        R[0] = - ddTdxx + 2
        return R
        

    
    
