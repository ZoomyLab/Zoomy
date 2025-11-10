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

from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
import zoomy_core.model.initial_conditions as IC

class CoupledConstrained(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=2,
        aux_variables=4,
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


    def source_implicit(self):
        out = Matrix([0 for i in range(2)])
        u = self.variables[0]
        p = self.variables[1]
        param = self.parameters
        dudt = self.aux_variables.dudx
        dudx = self.aux_variables.dudx
        dpdx = self.aux_variables.dpdx
        f = self.aux_variables.f
        out[0] = dudt + dpdx  + 1
        out[1] = dudx + f
        # out[0] = dudx -1.
        # out[1] = dpdx
        return out
