import jax.numpy as jnp
import numpy as np
import numpy.ctypeslib as npct
from scipy.linalg import eigvals
import os
from copy import deepcopy
from ctypes import cdll

import sympy
from sympy import (
    Matrix,
    lambdify,
    powsimp,
    MatrixSymbol,
    fraction,
    cancel,
    latex,
    init_printing,
)
from sympy import zeros

from attr import define
from typing import Optional, Union, Callable
from types import SimpleNamespace


from library.model.boundary_conditions import BoundaryConditions, Extrapolation
from library.model.initial_conditions import InitialConditions, Constant
from library.misc.custom_types import FArray
from library.misc.misc import Zstruct
from library.model.sympy2c import create_module

init_printing()


def vectorize_constant_sympy_expressions(expr, Q, Qaux):
    symbol_list = Q.get_list() + Qaux.get_list()
    q0 = Q[0]
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            if not any(symbol in expr[i, j].free_symbols for symbol in symbol_list):
                if expr[i, j] == 0:
                    expr[i, j] = 10 ** (-20) * q0
                elif expr[i, j]:
                    expr[i, j] = expr[i, j] + 10 ** (-20) * q0
                    # print('I don\'t know how to vectorize this yet')
                    # assert False
    return expr


def custom_simplify(expr):
    return powsimp(expr, combine="all", force=False, deep=True)


@define(slots=True, frozen=False, kw_only=True)
class RuntimeModel:
    name: str
    dimension: int = 1
    n_fields: int
    n_aux_fields: int
    n_parameters: int
    parameters: FArray

    flux: list[Callable]
    flux_jacobian: list[Callable]
    source: Callable
    source_jacobian: Callable
    nonconservative_matrix: list[Callable]
    quasilinear_matrix: list[Callable]
    eigenvalues: Callable
    left_eigenvectors: Optional[Callable]
    right_eigenvectors: Optional[Callable]
    source_imlicit: Callable
    interpolate_3d: Callable

    bcs: list[Callable]

    @classmethod
    def from_model(cls, model):
        pde = model.get_pde()
        # bcs = model.create_python_boundary_interface(printer='numpy')
        bcs = model.get_boundary_conditions()
        return cls(
            model.name,
            model.dimension,
            model.n_fields,
            model.n_aux_fields,
            model.n_parameters,
            pde.flux,
            pde.flux_jacobian,
            pde.source,
            pde.source_jacobian,
            pde.nonconservative_matrix,
            pde.quasilinear_matrix,
            pde.source,
            pde.eigenvalues,
            pde.left_eigenvectors,
            pde.right_eigenvectors,
            pde.source_implicit,
            pde.interpolate_3d,
            bcs,
        )


@define(slots=True, frozen=False, kw_only=True)
class Model:
    """
    Generic (virtual) model implementation.
    """

    name: str
    dimension: int = 1
    boundary_conditions: BoundaryConditions = BoundaryConditions(
        [
            Extrapolation(physical_tag="left"),
            Extrapolation(physical_tag="right"),
        ]
    )
    initial_conditions: InitialConditions = Constant()
    aux_initial_conditions: InitialConditions = Constant()

    n_fields: int
    n_aux_fields: int
    n_parameters: int
    variables: Zstruct
    aux_variables: Zstruct
    parameters: Zstruct
    parameters_default: dict
    parameter_values: FArray
    sympy_normal: Matrix

    settings: Zstruct
    settings_default_dict: dict

    sympy_flux: list[Matrix]
    sympy_flux_jacobian: list[Matrix]
    sympy_source: Matrix
    sympy_source_jacobian: Matrix
    sympy_nonconservative_matrix: list[Matrix]
    sympy_quasilinear_matrix: list[Matrix]
    sympy_eigenvalues: Matrix
    sympy_left_eigenvectors: Optional[Matrix]
    sympy_right_eigenvectors: Optional[Matrix]
    sympy_source_implicit: Matrix
    sympy_interpolate_3d: Matrix

    def __init__(
        self,
        dimension: int,
        fields: Union[int, list],
        aux_fields: Union[int, list],
        boundary_conditions: BoundaryConditions,
        initial_conditions: InitialConditions,
        aux_initial_conditions: InitialConditions = Constant(),
        parameters: dict = {},
        parameters_default: dict = {},
        settings: dict = {},
        settings_default: dict = {"eigenvalue_mode": "symbolic"},
    ):
        self.name = "Model"
        self.dimension = dimension
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.aux_initial_conditions = aux_initial_conditions

        self.variables = register_sympy_attribute(fields, "q")
        self.aux_variables = register_sympy_attribute(aux_fields, "aux")
        self.time = sympy.symbols("t", real=True)
        self.position = register_sympy_attribute(self.dimension, "X")
        self.position_3d = register_sympy_attribute(3, "X")
        self.distance = sympy.symbols("dX", real=True)
        updated_parameters = {**parameters_default, **parameters}
        self.parameters = register_sympy_attribute(updated_parameters, "p")
        self.parameters_default = parameters_default
        self.parameter_values = register_parameter_defaults(updated_parameters)
        self.sympy_normal = register_sympy_attribute(
            ["n" + str(i) for i in range(self.dimension)], "n"
        )

        self.settings = SimpleNamespace(**{**settings_default, **settings})

        self.n_fields = self.variables.length()
        self.n_aux_fields = self.aux_variables.length()
        self.n_parameters = self.parameters.length()

        self.init_sympy_functions()

    def get_latex(self):
        return latex(self.flux)

    def init_sympy_functions(self):
        self.sympy_flux = self.flux()
        if self.flux_jacobian() is None:
            self.sympy_flux_jacobian = [
                zeros(self.n_fields, self.n_fields) for i in range(self.dimension)
            ]
            for d in range(self.dimension):
                self.sympy_flux_jacobian[d] = Matrix(
                    custom_simplify(
                        Matrix(self.sympy_flux[d]).jacobian(self.variables),
                    )
                )
        else:
            self.sympy_flux_jacobian = self.flux_jacobian()
        self.sympy_source = self.source()
        if self.source_jacobian() is None:
            self.sympy_source_jacobian = Matrix(
                custom_simplify(
                    Matrix(self.sympy_source).jacobian(self.variables),
                )
            )
        else:
            self.sympy_source_jacobian = self.source_jacobian()
        self.sympy_nonconservative_matrix = self.nonconservative_matrix()
        if self.quasilinear_matrix() is None:
            self.sympy_quasilinear_matrix = [
                self.sympy_flux_jacobian[d] - self.sympy_nonconservative_matrix[d]
                for d in range(self.dimension)
            ]
        else:
            self.sympy_quasilinear_matrix = self.quasilinear_matrix()
        self.sympy_source_implicit = self.source_implicit()
        self.sympy_interpolate_3d = self.interpolate_3d()
        # self.sympy_quasilinear_matrix = self.quasilinear_matrix()
        # TODO check case imaginary
        # TODO check case not computable
        if self.settings.eigenvalue_mode == "symbolic":
            self.sympy_eigenvalues = self.eigenvalues()
        else:
            self.sympy_eigenvalues = None
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None


    def get_boundary_conditions(self, printer="numpy"):
        """Returns a runtime boundary_conditions for numpy arrays from the symbolic model."""
        n_boundary_functions = len(self.boundary_conditions.boundary_functions)
        # l_bcs = [
        #     lambdify(
        #         [
        #             self.time,
        #             self.position.get_list(),
        #             self.distance,
        #             self.variables.get_list(),
        #             self.aux_variables.get_list(),
        #             self.parameters.get_list(),
        #             self.sympy_normal.get_list(),
        #         ],
        #         vectorize_constant_sympy_expressions(
        #             self.boundary_conditions.boundary_functions[d], self.variables, self.aux_variables
        #         ),
        #         printer,
        #     )
        #     for d in range(n_boundary_functions)
        # ]
        # # the func=func part is necessary, because of https://stackoverflow.com/questions/46535577/initialising-a-list-of-lambda-functions-in-python/46535637#46535637
        # bcs = [
        #     lambda time,
        #         position,
        #         distance,
        #         q,
        #         qaux,
        #         p,
        #         n, f=l_bcs[d]: jnp.squeeze(
        #         np.array(f(time, position, distance, q, qaux, p, n)), axis=1
        #     )
        #     for d in range(n_boundary_functions)
        # ]
        bcs = []
        for i in range(n_boundary_functions):
            func_bc = lambdify(
                [
                    self.time,
                    self.position.get_list(),
                    self.distance,
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                    self.sympy_normal.get_list(),
                ],
                vectorize_constant_sympy_expressions(self.boundary_conditions.boundary_functions[i], self.variables, self.aux_variables),
                #self.boundary_conditions.boundary_functions[i],
                printer,
            )
            # the func=func part is necessary, because of https://stackoverflow.com/questions/46535577/initialising-a-list-of-lambda-functions-in-python/46535637#46535637
            f = (
                lambda time,
                position,
                distance,
                q,
                qaux,
                p,
                n,
                func=func_bc: jnp.squeeze(
                    jnp.array(func(time, position, distance, q, qaux, p, n)), axis=1
                )
            )
            bcs.append(f)
        return bcs

    def get_pde(self, printer="numpy"):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = [
            lambdify(
                (
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                ),
                vectorize_constant_sympy_expressions(
                    self.sympy_flux[d], self.variables, self.aux_variables
                ),
                printer,
            )
            for d in range(self.dimension)
        ]
        # the f=l_flux[d] part is necessary, because of https://stackoverflow.com/questions/46535577/initialising-a-list-of-lambda-functions-in-python/46535637#46535637
        flux = [
            lambda Q, Qaux, param, f=l_flux[d]: jnp.squeeze(
                np.array(f(Q, Qaux, param)), axis=1
            )
            for d in range(self.dimension)
        ]
        l_flux_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_flux_jacobian,
            printer,
        )
        flux_jacobian = l_flux_jacobian

        l_nonconservative_matrix = [
            lambdify(
                [
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                ],
                vectorize_constant_sympy_expressions(
                    self.sympy_nonconservative_matrix[d],
                    self.variables,
                    self.aux_variables,
                ),
                printer,
            )
            for d in range(self.dimension)
        ]
        nonconservative_matrix = [
            lambda Q, Qaux, param, f=l_nonconservative_matrix[d]: f(Q, Qaux, param)
            for d in range(self.dimension)
        ]

        l_quasilinear_matrix = [
            lambdify(
                [
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                ],
                vectorize_constant_sympy_expressions(
                    self.sympy_quasilinear_matrix[d], self.variables, self.aux_variables
                ),
                printer,
            )
            for d in range(self.dimension)
        ]
        quasilinear_matrix = l_quasilinear_matrix

        if self.settings.eigenvalue_mode == "symbolic":
            l_eigenvalues = lambdify(
                [
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                    self.sympy_normal.get_list(),
                ],
                vectorize_constant_sympy_expressions(
                    self.sympy_eigenvalues, self.variables, self.aux_variables
                ),
                printer,
            )

            def eigenvalues(Q, Qaux, param, normal):
                return jnp.squeeze(
                    jnp.array(l_eigenvalues(Q, Qaux, param, normal)), axis=1
                )

        elif self.settings.eigenvalue_mode == "numerical":
            eigenvalues = None

        l_source = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.sympy_source, self.variables, self.aux_variables
            ),
            printer,
        )

        def source(Q, Qaux, param):
            return jnp.squeeze(jnp.array(l_source(Q, Qaux, param)), axis=1)


        l_source_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.sympy_source_jacobian, self.variables, self.aux_variables
            ),
            printer,
        )
        source_jacobian = l_source_jacobian

        l_source_implicit = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.sympy_source_implicit, self.variables, self.aux_variables
            ),
            printer,
        )
        def source_implicit(Q, Qaux, param):
            return jnp.squeeze(jnp.array(l_source_implicit(Q, Qaux, param)), axis=1)

        l_interpolate_3d = lambdify(
            [
                self.position_3d.get_list(),
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.sympy_interpolate_3d, self.variables, self.aux_variables
            ),
            printer,
        )
        def interpolate_3d(X, Q, Qaux, param):
            return jnp.squeeze(jnp.array(l_interpolate_3d(X, Q, Qaux, param)), axis=1)



        left_eigenvectors = None
        right_eigenvectors = None
        d = {
            "flux": flux,
            "flux_jacobian": flux_jacobian,
            "nonconservative_matrix": nonconservative_matrix,
            "quasilinear_matrix": quasilinear_matrix,
            "eigenvalues": eigenvalues,
            "left_eigenvectors": left_eigenvectors,
            "right_eigenvectors": right_eigenvectors,
            "source": source,
            "source_jacobian": source_jacobian,
            "source_implicit": source_implicit,
            "interpolate_3d": interpolate_3d,
        }
        return SimpleNamespace(**d)

    def flux(self):
        return [Matrix(self.variables[:]) for d in range(self.dimension)]

    def nonconservative_matrix(self):
        return [zeros(self.n_fields, self.n_fields) for d in range(self.dimension)]

    def source(self):
        return zeros(self.n_fields, 1)

    def flux_jacobian(self):
        # generated automatically unless explicitly provided
        return None

    def quasilinear_matrix(self):
        # generated automatically unless explicitly provided
        return None

    def source_jacobian(self):
        # generated automatically unless explicitly provided
        return None

    def source_implicit(self):
        return zeros(self.n_fields, 1)

    def interpolate_3d(self):
        return zeros(5, 1)


    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def get_default_setup(self):
        text = """
        def setup():
            level = 0
            settings = Settings(
                name="ShallowMoments2d",
                parameters={"g": 1.0, "C": 1.0, "nu": 0.1},
                reconstruction=recon.constant,
                num_flux=flux.LLF(),
                compute_dt=timestepping.adaptive(CFL=0.45),
                time_end=1.0,
                output_snapshots=100, output_dir='outputs/test')
            
            inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
            inflow_dict[1] = 0.36
            outflow_dict = {0: 1.0}
            
            bcs = BC.BoundaryConditions(
                [
                    BC.Wall(physical_tag="top"),
                    BC.Wall(physical_tag="bottom"),
                    BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
                    BC.InflowOutflow(physical_tag="right", prescribe_fields= outflow_dict),
                ]
            )
            ic = IC.Constant(
                constants=lambda n_fields: np.array(
                    [1.0, 0.1, 0.1] + [0.0 for i in range(n_fields - 3)]
                )
            )
        
            return ic, bcs, settings
        """
        return text


def register_sympy_attribute(argument, string_identifier="q_"):
    if type(argument) == int:
        attributes = {
            string_identifier + str(i): sympy.symbols(
                string_identifier + str(i), real=True
            )
            for i in range(argument)
        }
    elif type(argument) == type({}):
        attributes = {
            name: sympy.symbols(str(name), real=True) for name in argument.keys()
        }
    elif type(argument) == list:
        attributes = {name: sympy.symbols(str(name), real=True) for name in argument}
    else:
        assert False
    return Zstruct(**attributes)


def register_parameter_defaults(parameters):
    if type(parameters) == int:
        default_values = np.zeros(parameters, dtype=float)
    elif type(parameters) == type({}):
        default_values = np.array([value for value in parameters.values()])
    else:
        assert False
    return default_values


def substitute_sympy_attributes_with_symbol_matrix(
    expr: Matrix, attr: Zstruct, attr_matrix: MatrixSymbol
):
    if expr is None:
        return None
    assert attr.length() == attr_matrix.shape[0]
    for i, k in enumerate(attr.get_list()):
        expr = Matrix(expr).subs(k, attr_matrix[i])
    return expr


def eigenvalue_dict_to_matrix(eigenvalues):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(custom_simplify(ev))
    return Matrix(evs)


class ModelGUI(Model):
    """
    :gui:
    { 'parameters': { 'I': {'type': 'int', 'value': 1, 'step': 1}, 'F': {'type': 'float', 'value': 1., 'step':0.1}, 'S': {'type': 'string', 'value': 'asdf'}, 'Arr':{'type': 'array', 'value': np.array([[1., 2.], [3., 4.]])} },}
    """
