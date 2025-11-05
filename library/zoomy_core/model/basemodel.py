from typing import Callable, Union

import numpy as np
import sympy
from attrs import define, field
from sympy import init_printing, powsimp, zeros


from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function

init_printing()


def default_simplify(expr):
    return powsimp(expr, combine="all", force=False, deep=True)


@define(frozen=True, slots=True, kw_only=True)
class Model:
    """
    Generic (virtual) model implementation.
    """

    boundary_conditions: BoundaryConditions

    name: str = "Model"
    dimension: int = 1

    initial_conditions: InitialConditions = field(factory=Constant)
    aux_initial_conditions: InitialConditions = field(factory=Constant)

    parameters: Zstruct = field(factory=lambda: Zstruct())

    time: sympy.Symbol = field(
        init=False, factory=lambda: sympy.symbols("t", real=True)
    )
    distance: sympy.Symbol = field(
        init=False, factory=lambda: sympy.symbols("dX", real=True)
    )
    position: Zstruct = field(
        init=False, factory=lambda: register_sympy_attribute(3, "X")
    )
    number_of_points_3d: int = field(factory=lambda: 10)

    _simplify: Callable = field(factory=lambda: default_simplify)

    # Derived fields initialized in __attrs_post_init__
    _default_parameters: dict = field(init=False, factory=dict)
    n_variables: int = field(init=False)
    n_aux_variables: int = field(init=False)
    n_parameters: int = field(init=False)
    variables: Zstruct = field(init=False, default=1)

    positive_variables: Union[dict, list, None] = field(default=None)
    aux_variables: Zstruct = field(default=0)
    parameter_values: FArray = field(init=False)
    normal: ZArray = field(init=False)

    z_3d: Zstruct = field(init=False, default=number_of_points_3d)
    u_3d: Zstruct = field(init=False, default=number_of_points_3d)
    p_3d: Zstruct = field(init=False, default=number_of_points_3d)
    alpha_3d: Zstruct = field(init=False, default=number_of_points_3d)

    _flux: Function = field(init=False)
    _dflux: Function = field(init=False)
    _nonconservative_matrix: Function = field(init=False)
    _quasilinear_matrix: Function = field(init=False)
    _eigenvalues: Function = field(init=False)
    _left_eigenvectors: Function = field(init=False)
    _right_eigenvectors: Function = field(init=False)
    _source: Function = field(init=False)
    _source_jacobian_wrt_variables: Function = field(init=False)
    _source_jacobian_wrt_aux_variables: Function = field(init=False)
    _residual: Function = field(init=False)
    _project_2d_to_3d: Function = field(init=False)
    _project_3d_to_2d: Function = field(init=False)
    _boundary_conditions: Function = field(init=False)

    def __attrs_post_init__(self):
        updated_default_parameters = {
            **self._default_parameters,
            **self.parameters.as_dict(),
        }

        # Use object.__setattr__ because class is frozen
        object.__setattr__(
            self,
            "variables",
            register_sympy_attribute(self.variables, "q", self.positive_variables),
        )
        object.__setattr__(
            self, "aux_variables", register_sympy_attribute(self.aux_variables, "qaux")
        )

        object.__setattr__(
            self,
            "parameters",
            register_sympy_attribute(updated_default_parameters, "p"),
        )
        object.__setattr__(
            self,
            "parameter_values",
            register_parameter_values(updated_default_parameters),
        )
        object.__setattr__(
            self,
            "normal",
            register_sympy_attribute(
                ["n" + str(i) for i in range(self.dimension)], "n"
            ),
        )

        object.__setattr__(self, "n_variables", self.variables.length())
        object.__setattr__(self, "n_aux_variables", self.aux_variables.length())
        object.__setattr__(self, "n_parameters", self.parameters.length())

        object.__setattr__(
            self, "z_3d", register_sympy_attribute(self.number_of_points_3d, "z")
        )
        object.__setattr__(
            self, "u_3d", register_sympy_attribute(self.number_of_points_3d, "u")
        )
        object.__setattr__(
            self, "p_3d", register_sympy_attribute(self.number_of_points_3d, "p")
        )
        object.__setattr__(
            self,
            "alpha_3d",
            register_sympy_attribute(self.number_of_points_3d, "alpha"),
        )

        object.__setattr__(
            self,
            "_flux",
            Function(
                name="flux",
                definition=self.flux(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_dflux",
            Function(
                name="dflux",
                definition=self.dflux(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )

        object.__setattr__(
            self,
            "_nonconservative_matrix",
            Function(
                name="nonconservative_matrix",
                definition=self.nonconservative_matrix(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_quasilinear_matrix",
            Function(
                name="quasilinear_matrix",
                definition=self.quasilinear_matrix(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_eigenvalues",
            Function(
                name="eigenvalues",
                definition=self.eigenvalues(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                    normal=self.normal,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_left_eigenvectors",
            Function(
                name="left_eigenvectors",
                definition=self.left_eigenvectors(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                    normal=self.normal,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_right_eigenvectors",
            Function(
                name="right_eigenvectors",
                definition=self.right_eigenvectors(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                    normal=self.normal,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_source",
            Function(
                name="source",
                definition=self.source(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_source_jacobian_wrt_variables",
            Function(
                name="source_jacobian_wrt_variables",
                definition=self.source_jacobian_wrt_variables(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_source_jacobian_wrt_aux_variables",
            Function(
                name="source_jacobian_wrt_aux_variables",
                definition=self.source_jacobian_wrt_aux_variables(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )

        object.__setattr__(
            self,
            "_residual",
            Function(
                name="residual",
                definition=self.residual(),
                args=Zstruct(
                    time=self.time,
                    position=self.position,
                    distance=self.distance,
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_project_2d_to_3d",
            Function(
                name="project_2d_to_3d",
                definition=self.project_2d_to_3d(),
                args=Zstruct(
                    Z=self.position[2],
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_project_3d_to_2d",
            Function(
                name="project_3d_to_2d",
                definition=self.project_3d_to_2d(),
                args=Zstruct(
                    variables=self.variables,
                    aux_variables=self.aux_variables,
                    parameters=self.parameters,
                ),
            ),
        )
        object.__setattr__(
            self,
            "_boundary_conditions",
            self.boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
            ),
        )

    def flux(self):
        return ZArray.zeros(
            self.n_variables,
            self.dimension,
        )

    def dflux(self):
        return ZArray.zeros(
            self.n_variables,
            self.dimension,
        )

    def nonconservative_matrix(self):
        return ZArray.zeros(
            self.n_variables,
            self.n_variables,
            self.dimension,
        )

    def source(self):
        return ZArray.zeros(self.n_variables)
        return zeros(self.n_variables, 1)

    def quasilinear_matrix(self):
        """generated automatically unless explicitly provided"""
        # NC = [ZArray.zeros(self.n_variables, self.n_variables) for _ in range(self.dimension)]
        # NC = ZArray.zeros(
        #     self.n_variables, self.n_variables, self.dimension
        # )
        JacF = ZArray(sympy.derive_by_array(self.flux(), self.variables.get_list()))
        for d in range(self.dimension):
            JacF_d = JacF[:, :, d]
            JacF_d = ZArray(JacF_d.tomatrix().T)
            JacF[:, :, d] = JacF_d
        return self._simplify(
            JacF + self.nonconservative_matrix()
        )

    def source_jacobian_wrt_variables(self):
        """generated automatically unless explicitly provided"""
        return self._simplify(
            sympy.derive_by_array(self.source(), self.variables.get_list())
        )

    def source_jacobian_wrt_aux_variables(self):
        """generated automatically unless explicitly provided"""
        return self._simplify(
            sympy.derive_by_array(self.source(), self.aux_variables.get_list())
        )

    def residual(self):
        return ZArray.zeros(self.n_variables)

    def project_2d_to_3d(self):
        return ZArray.zeros(6)

    def project_3d_to_2d(self):
        return ZArray.zeros(self.n_variables)

    def eigenvalues(self):
        A = self.normal[0] * self.quasilinear_matrix()[:, :, 0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[:, :, d]
        return self._simplify(eigenvalue_dict_to_matrix(sympy.Matrix(A).eigenvals()))

    def left_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)


def substitute_precomputed_denominator(self, expr, sym, sym_inv):
    """Recursively replace denominators involving `sym` with precomputed `sym_inv`.
    Works for scalar, Matrix, or Array of any rank.
    """
    # --- Case 1: Matrix (2D) ---
    if isinstance(expr, sympy.MatrixBase):
        return expr.applyfunc(
            lambda e: self.substitute_precomputed_denominator(e, sym, sym_inv)
        )

    # --- Case 2: Array (any rank) ---
    if isinstance(expr, sympy.Array):
        # Recursively apply substitution elementwise
        new_data = [
            self.substitute_precomputed_denominator(e, sym, sym_inv) for e in expr
        ]
        return sympy.ZArray(new_data).reshape(*expr.shape)

    # --- Case 3: Scalar or general SymPy expression ---
    num, den = sympy.fraction(expr)

    if den.has(sym):
        # split denominator into sym-dependent and independent parts
        den_sym, den_rest = den.as_independent(sym, as_Add=False)
        # swap naming (as_independent returns independent first)
        den_rest, den_sym = den_sym, den_rest

        # replace sym by sym_inv in the sym-dependent part
        den_sym_repl = den_sym.xreplace({sym: sym_inv})

        return (
            self.substitute_precomputed_denominator(num, sym, sym_inv)
            * den_sym_repl
            / den_rest
        )

    # recurse through function arguments
    elif hasattr(expr, "args") and expr.args:
        return expr.func(
            *[
                self.substitute_precomputed_denominator(arg, sym, sym_inv)
                for arg in expr.args
            ]
        )

    # base case: atomic expression
    else:
        return expr


def transform_positive_variable_intput_to_list(argument, positive, n_variables):
    out = [False for _ in range(n_variables)]
    if positive is None:
        return out
    if type(positive) == type({}):
        assert type(argument) == type(positive)
        for i, a in enumerate(argument.keys()):
            if a in positive.keys():
                out[i] = positive[a]
    if type(positive) == list:
        for i in positive:
            out[i] = True
    return out


def register_sympy_attribute(argument, string_identifier="q_", positives=None):
    if type(argument) == int:
        positive = transform_positive_variable_intput_to_list(
            argument, positives, argument
        )
        attributes = {
            string_identifier + str(i): sympy.symbols(
                string_identifier + str(i), real=True, positive=positive[i]
            )
            for i in range(argument)
        }
    elif type(argument) == type({}):
        positive = transform_positive_variable_intput_to_list(
            argument, positives, len(argument)
        )
        attributes = {
            name: sympy.symbols(str(name), real=True, positive=pos)
            for name, pos in zip(argument.keys(), positive)
        }
    elif type(argument) == list:
        positive = transform_positive_variable_intput_to_list(
            argument, positives, len(argument)
        )
        attributes = {
            name: sympy.symbols(str(name), real=True, positive=pos)
            for name, pos in zip(argument, positive)
        }
    else:
        assert False
    return Zstruct(**attributes)


def register_parameter_values(parameters):
    if type(parameters) == int:
        default_values = np.zeros(parameters, dtype=float)
    elif type(parameters) == type({}):
        default_values = np.array([value for value in parameters.values()])
    else:
        assert False
    return default_values


def eigenvalue_dict_to_matrix(eigenvalues, simplify=default_simplify):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(simplify(ev))
    return ZArray(evs)
