import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix
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

class CoupledConstrained(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=2,
        aux_fields=4,
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
        out = Matrix([0 for i in range(2)])
        u = self.variables[0]
        p = self.variables[1]
        param = self.parameters
        dudt = self.aux_variables.dudx
        dudx = self.aux_variables.dudx
        dpdx = self.aux_variables.dpdx
        f = self.aux_variables.f
        #out[0] = dudt + dpdx
        #out[1] = dudx + f
        out[0] = dudx -1.
        out[1] = dpdx
        return out
