import sympy
from sympy import Matrix

# from sympy import *

from zoomy_core.model.basemodel import (
    register_sympy_attribute,
)
from zoomy_core.model.basemodel import Model






class ShearShallowFlow(Model):
    """
    Shallow Moments

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=3,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
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
        flux_x = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[2]
        p = self.parameters
        flux_x[0] = hu
        flux_x[1] = hu * u + p.g * h**2 / 2 + h * P11
        # flux_x[1] = hu * u + h*P11
        # flux_x[2] = 0.
        flux_x[2] = 2 * P11 * u

        return [flux_x]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[2]
        p = self.parameters

        # nc_x[2, 0] = - p.g * h
        nc_x[2, 0] = 0
        nc_x[2, 1] = 0
        nc_x[2, 2] = u
        return [nc_x]

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        # out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[2]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3 * P11)

        evs[0] = u
        evs[1] = u + a
        evs[2] = u - a

        return evs

    def friction_paper(self):
        assert "phi" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[2]
        p = self.parameters

        abs_u = sympy.sqrt(u**2)
        trace_P = P11
        grad_b = [-sympy.tan(p.theta)]
        # alpha = max(0, p.Cr * (trace_P/h**2 - p.phi)/(trace_P**2/h**2))
        alpha = sympy.Piecewise(
            (
                p.Cr * (trace_P / h**2 - p.phi) / (trace_P**2 / h**2),
                p.Cr * (trace_P / h**2 - p.phi) / (trace_P**2 / h**2) > 0,
            ),
            (0, p.Cr * (trace_P / h**2 - p.phi) / (trace_P**2 / h**2) <= 0),
        )
        D11 = -2 * alpha / h * abs_u**3 * P11
        Q = alpha * trace_P * abs_u**3

        out[1] = -h * p.g * grad_b[0] - p.Cr * u * abs_u
        out[2] = Q
        return out

    def chezy(self):
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        Q = self.variables

        u = Q[1] / Q[0]

        p = self.parameters
        abs_u = sympy.sqrt(u**2)
        out[1] = -p.Cf * abs_u * u
        # out[2] = - p.Cf * abs_u**3
        return out


class ShearShallowFlowEnergy(Model):
    """
    Shallow Moments

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=3,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
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
        flux_x = Matrix([0 for i in range(self.n_variables)])

        p = self.parameters
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        E = self.variables[2]
        R = 2 * E - h * u**2
        flux_x[0] = hu
        flux_x[1] = hu * u + p.g * h**2 / 2 + R
        flux_x[2] = (E + R) * u
        return [flux_x]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        p = self.parameters
        Q = self.variables
        h = Q[0]
        u = Q[1] / h
        nc_x[2, 0] = -p.g * h * u
        return [nc_x]

    def chezy(self):
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        Q = self.variables

        u = Q[1] / Q[0]

        p = self.parameters
        abs_u = sympy.sqrt(u**2)
        out[1] = -p.Cf * abs_u * u
        out[2] = -p.Cf * abs_u**3
        return out


class ShearShallowFlowPathconservative(Model):
    """
    Shallow Moments

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=6,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
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

    def get_primitives(self, Q):
        h = Q[0]
        u = Q[1] / Q[0]
        v = Q[2] / Q[0]
        E11 = Q[3]
        E12 = Q[4]
        E22 = Q[5]
        R11 = 2 * E11 - Q[1] ** 2 / Q[0]
        R12 = 2 * E12 - Q[1] * Q[2] / Q[0]
        R22 = 2 * E22 - Q[2] ** 2 / Q[0]
        P11 = R11 / h
        P12 = R12 / h
        P22 = R22 / h
        return h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22

    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_variables)])

        h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = self.get_primitives(
            self.variables.get_list()
        )
        p = self.parameters

        flux_x[0] = h * u
        flux_x[1] = R11 + h * u**2 + 1 / 2 * p.g * h**2
        flux_x[2] = R12 + h * u * v
        flux_x[3] = (E11 + R11) * u
        flux_x[4] = E12 * u + 1 / 2 * (R11 * v + R12 * u)
        flux_x[5] = E22 * u + R12 * v
        return [flux_x]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])

        h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = self.get_primitives(
            self.variables.get_list()
        )
        p = self.parameters

        nc_x[3, 0] = -p.g * h * u
        nc_x[4, 0] = -1 / 2 * p.g * h * v
        return [nc_x]

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        # out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_variables)])

        h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = self.get_primitives(
            self.variables.get_list()
        )
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3 * P11)

        evs[0] = u - a
        evs[1] = u - b
        evs[2] = u
        evs[3] = u
        evs[4] = u + b
        evs[5] = u + a

        return evs

    def chezy(self):
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])

        h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = self.get_primitives(
            self.variables.get_list()
        )
        p = self.parameters

        abs_u = sympy.sqrt(u**2 + v**2)
        out[1] = p.Cf * abs_u * u
        out[2] = -p.Cf * abs_u * v
        out[3] = p.Cf * abs_u * u**2
        out[4] = p.Cf * abs_u * u * v
        out[5] = p.Cf * abs_u * v
        return out

    def friction_paper(self):
        assert "Cf" in vars(self.parameters)
        assert "Cr" in vars(self.parameters)
        assert "g" in vars(self.parameters)
        assert "theta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])

        h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = self.get_primitives(
            self.variables.get_list()
        )
        p = self.parameters

        abs_u = sympy.sqrt(u**2 + v**2)
        grad_b = [-sympy.tan(p.theta)]
        trace_P = P11 + P22
        expr = p.Cr * (trace_P / h**2 - p.phi) / (trace_P**2 / h**2)
        alpha = sympy.Piecewise((0, expr < 0), (expr, True))

        out[1] = -h * p.g * grad_b[0] - p.Cf * abs_u * u
        out[2] = -p.Cf * abs_u * v
        out[3] = -alpha * abs_u**3 * P11 - p.g * h * u * grad_b[0] - p.Cf * abs_u * u**2
        out[4] = (
            -alpha * abs_u**3 * P12 - p.g * h * v * grad_b[0] - p.Cf * abs_u * u * v
        )
        out[5] = -alpha * abs_u**3 * P12 - p.Cf * abs_u * v
        return out


class ShearShallowFlowPathconservative2(ShearShallowFlowPathconservative):
    """
    Shallow Moments

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=6,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
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
        SSF = ShearShallowFlowPathconservative(
            boundary_conditions=self.boundary_conditions,
            initial_conditions=self.initial_conditions,
            parameters=self._default_parameters,
        )
        SSF.init_derived_sympy_functions()
        flux_x = 0.0 * SSF.sympy_flux[0]
        return [flux_x]

    def nonconservative_matrix(self):
        SSF = ShearShallowFlowPathconservative(
            boundary_conditions=self.boundary_conditions,
            initial_conditions=self.initial_conditions,
            parameters=self._default_parameters,
        )
        SSF.init_derived_sympy_functions()
        nc_x = SSF.sympy_nonconservative_matrix[0] - 1.0 * SSF.sympy_flux_jacobian[0]
        return [nc_x]


class ShearShallowFlow2d(Model):
    """
    Shallow Moments 2d

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=6,
        aux_variables=0,
        parameters={},
        _default_parameters={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
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
        flux_x = Matrix([0 for i in range(self.n_variables)])
        flux_y = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu / h
        v = hv / h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters
        flux_x[0] = hu
        flux_x[1] = hu * u + p.g * h**2 / 2 + P11
        flux_x[2] = hu * v + h * P12
        flux_x[3] = u * P11
        flux_x[4] = u * P12
        flux_x[5] = 0

        flux_y[0] = hv
        flux_y[1] = hu * v + P12
        flux_y[2] = hv * v + p.g * h**2 / 2 + P22
        flux_y[3] = 0
        flux_y[4] = v * P12
        flux_y[5] = v * P22
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        nc_x = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        nc_y = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu / h
        v = hv / h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]

        nc_x[3, 0] = -P11 / h
        nc_x[3, 1] = +P11 / h
        nc_y[3, 0] = -2 * P11 / h
        nc_y[3, 1] = +2 * P11 / h
        nc_y[3, 3] = +v

        nc_x[4, 0] = -P11 / h
        nc_x[4, 2] = +P11 / h
        nc_y[4, 0] = -P22 / h
        nc_y[4, 1] = +P22 / h

        nc_x[5, 0] = -2 * P12 / h
        nc_x[5, 2] = +2 * P12 / h
        nc_y[5, 0] = -P22 / h
        nc_y[5, 2] = +P22 / h
        nc_x[5, 5] = u
        return [nc_x, nc_y]

    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        # out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def eigenvalues(self):
        evs = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu / h
        v = hv / h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3 * P11)

        evs[0] = u
        evs[1] = u
        evs[2] = u + b
        evs[3] = u - b
        evs[4] = u + a
        evs[5] = u - a

        return evs

    def friction_paper(self):
        assert "phi" in vars(self.parameters)
        assert "theta" in vars(self.parameters)
        assert "Cr" in vars(self.parameters)
        assert "g" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu = self.variables[1]
        hv = self.variables[2]
        u = hu / h
        v = hv / h
        P11 = self.variables[3]
        P12 = self.variables[4]
        P22 = self.variables[5]
        p = self.parameters

        abs_u = sympy.sqrt(u**2 + v**2)
        trace_P = P11 + P22
        grad_b = [-sympy.tan(p.theta), 0]
        alpha = max(0, p.Cr * (trace_P / h**2 - p.phi) / (trace_P**2 / h**2))
        D11 = -2 * alpha / h * abs_u**3 * P11
        D12 = -2 * alpha / h * abs_u**3 * P12
        D22 = -2 * alpha / h * abs_u**3 * P22
        Q = alpha * trace_P * abs_u**3

        out[1] = -h * p.g * grad_b[0] - p.Cr * u * abs_u
        out[2] = -h * p.g * grad_b[1] - p.Cr * v * abs_u
        out[3] = h * D11
        out[4] = h * D12
        out[5] = h * D22
        return out
