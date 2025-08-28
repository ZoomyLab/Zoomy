import jax.numpy as jnp
import numpy as np
import os
from copy import deepcopy
from ctypes import cdll
import dolfinx

import sympy
from sympy import (
    Matrix,
    lambdify,
    powsimp,
    MatrixSymbol,
    latex,
    init_printing,
    zeros
)

from typing import Optional, Union, Callable
from types import SimpleNamespace
from attrs import define, field



from library.model.boundary_conditions import BoundaryConditions, Extrapolation
from library.model.initial_conditions import InitialConditions, Constant
from library.misc.custom_types import FArray
from library.misc.misc import Zstruct
from library.model.sympy2c import create_module

init_printing()


def vectorize_constant_sympy_expressions(expr, Q, Qaux):
    symbol_list = Q.get_list() + Qaux.get_list()
    q0 = Q[0]

    rows, cols = expr.shape

    new_data = []

    for i in range(rows):
        row = []
        for j in range(cols):
            entry = expr[i, j]
            if not any(symbol in entry.free_symbols for symbol in symbol_list):
                if entry == 0:
                    row.append(10 ** (-20) * q0)
                else:
                    row.append(entry + 10 ** (-20) * q0)
            else:
                row.append(entry)
        new_data.append(row)

    return Matrix(new_data)




@define(kw_only=True, slots=True, frozen=True)
class JaxRuntimeModel:
    name: str = field()
    n_variables: int = field()
    n_aux_variables: int = field()
    n_parameters: int = field()
    parameters: FArray = field()
    flux: list[Callable] = field()
    flux_jacobian: list[Callable] = field()
    source: Callable = field()
    source_jacobian: Callable = field()
    nonconservative_matrix: list[Callable] = field()
    quasilinear_matrix: list[Callable] = field()
    eigenvalues: Callable = field()
    source_implicit: Callable = field()
    residual: Callable = field()
    interpolate_3d: Callable = field()
    bcs: list[Callable] = field()
    dimension: int = field()
    left_eigenvectors: Optional[Callable] = field(default=None)
    right_eigenvectors: Optional[Callable] = field(default=None)

    @classmethod
    def from_model(cls, model):
        pde = model._get_pde()
        bcs = model._get_boundary_conditions()
        return cls(
            name=model.name,
            dimension=model.dimension,
            n_variables=model.n_variables,
            n_aux_variables=model.n_aux_variables,
            n_parameters=model.n_parameters,
            parameters=model.parameter_values,
            flux=pde.flux,
            flux_jacobian=pde.flux_jacobian,
            source=pde.source,
            source_jacobian=pde.source_jacobian,
            nonconservative_matrix=pde.nonconservative_matrix,
            quasilinear_matrix=pde.quasilinear_matrix,
            eigenvalues=pde.eigenvalues,
            left_eigenvectors=pde.left_eigenvectors,
            right_eigenvectors=pde.right_eigenvectors,
            source_implicit=pde.source_implicit,
            residual=pde.residual,
            interpolate_3d=pde.interpolate_3d,
            bcs=bcs,
        )
        




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


    time: sympy.Symbol = field(init=False, factory=lambda: sympy.symbols("t", real=True))
    distance: sympy.Symbol = field(init=False, factory=lambda: sympy.symbols("dX", real=True))
    position: Zstruct = field(init=False, factory=lambda: register_sympy_attribute(3, "X"))

    _simplify: Callable = field(factory=lambda: default_simplify)

    # Derived fields initialized in __attrs_post_init__
    _default_parameters: dict = field(init=False, factory=dict)
    n_variables: int = field(init=False)
    n_aux_variables: int = field(init=False)
    n_parameters: int = field(init=False)
    variables: Zstruct = field(init=False, default=1)
    aux_variables: Zstruct = field(default=0)
    parameter_values: FArray = field(init=False)
    normal: Matrix = field(init=False)
    # position: Zstruct = field(init=False)
    


    def __attrs_post_init__(self):
        updated_default_parameters = {**self._default_parameters, **self.parameters}

        # Use object.__setattr__ because class is frozen
        object.__setattr__(self, "variables", register_sympy_attribute(self.variables, "q"))
        object.__setattr__(self, "aux_variables", register_sympy_attribute(self.aux_variables, "qaux"))
        # object.__setattr__(self, "position", register_sympy_attribute(self.dimension, "X"))

        object.__setattr__(self, "parameters", register_sympy_attribute(updated_default_parameters, "p"))
        object.__setattr__(self, "parameter_values", register_parameter_values(updated_default_parameters))
        object.__setattr__(
            self, "normal",
            register_sympy_attribute(["n" + str(i) for i in range(self.dimension)], "n")
        )

        object.__setattr__(self, "n_variables", self.variables.length())
        object.__setattr__(self, "n_aux_variables", self.aux_variables.length())
        object.__setattr__(self, "n_parameters", self.parameters.length())
        
    def get_boundary_conditions_matrix_inputs(self):
        """
        Returns the inputs for the boundary conditions matrix.
        """
        return (
            self.time,
            self.position,
            self.distance,
            self.variables,
            self.aux_variables,
            self.parameters,
            self.normal,
        )
        
    def get_boundary_conditions_matrix_inputs_as_list(self):
        """
        Returns the inputs for the boundary conditions matrix where the Zstructs are converted to lists.
        """
        return [
            self.time,
            self.position.get_list(),
            self.distance,
            self.variables.get_list(),
            self.aux_variables.get_list(),
            self.parameters.get_list(),
            self.normal.get_list(),
        ]


    def _get_boundary_conditions(self, printer="jax"):
        """Returns a runtime boundary_conditions for jax arrays from the symbolic model."""
        n_boundary_functions = len(self.boundary_conditions.boundary_functions)
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
                    self.normal.get_list(),
                ],
                vectorize_constant_sympy_expressions(self.boundary_conditions.boundary_functions[i], self.variables, self.aux_variables),
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

    def _get_pde(self, printer="jax"):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = [
            lambdify(
                (
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                ),
                vectorize_constant_sympy_expressions(
                    self.flux()[d], self.variables, self.aux_variables
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
            self.flux_jacobian(),
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
                    self.nonconservative_matrix()[d],
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
                    self.quasilinear_matrix()[d], self.variables, self.aux_variables
                ),
                printer,
            )
            for d in range(self.dimension)
        ]
        quasilinear_matrix = l_quasilinear_matrix

        l_eigenvalues = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
                self.normal.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.eigenvalues(), self.variables, self.aux_variables
            ),
            printer,
        )

        def eigenvalues(Q, Qaux, param, normal):
            return jnp.squeeze(
                jnp.array(l_eigenvalues(Q, Qaux, param, normal)), axis=1
            )

        l_left_eigenvectors = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
                self.normal.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.left_eigenvectors(), self.variables, self.aux_variables
            ),
            printer,
        )

        def left_eigenvectors(Q, Qaux, param, normal):
            return jnp.squeeze(
                jnp.array(l_left_eigenvectors(Q, Qaux, param, normal)), axis=1
            )


        l_right_eigenvectors = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
                self.normal.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.right_eigenvectors(), self.variables, self.aux_variables
            ),
            printer,
        )

        def right_eigenvectors(Q, Qaux, param, normal):
            return jnp.squeeze(
                jnp.array(l_right_eigenvectors(Q, Qaux, param, normal)), axis=1
            )
            
        

        l_source = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.source(), self.variables, self.aux_variables
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
                self.source_jacobian(), self.variables, self.aux_variables
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
                self.source_implicit(), self.variables, self.aux_variables
            ),
            printer,
        )
        def source_implicit(Q, Qaux, param):
            return jnp.squeeze(jnp.array(l_source_implicit(Q, Qaux, param)), axis=1)
        
        l_residual = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.residual(), self.variables, self.aux_variables
            ),
            printer,
        )
        def residual(Q, Qaux, param):
            return jnp.squeeze(jnp.array(l_residual(Q, Qaux, param)), axis=1)


        l_interpolate_3d = lambdify(
            [
                self.position.get_list(),
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(
                self.interpolate_3d(), self.variables, self.aux_variables
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
            "residual": residual,
            "interpolate_3d": interpolate_3d,
        }
        return SimpleNamespace(**d)

    def flux(self):
        return [Matrix(self.variables[:]) for d in range(self.dimension)]

    def nonconservative_matrix(self):
        return [zeros(self.n_variables, self.n_variables) for d in range(self.dimension)]

    def source(self):
        return zeros(self.n_variables, 1)

    def flux_jacobian(self):
        """ generated automatically unless explicitly provided """
        return [ self._simplify(
                Matrix(self.flux()[d]).jacobian(self.variables),
            ) for d in range(self.dimension) ]

    def quasilinear_matrix(self):
        """ generated automatically unless explicitly provided """
        return [ self._simplify(
                Matrix(self.flux_jacobian()[d] + self.nonconservative_matrix()[d],
            )
        ) for d in range(self.dimension) ]

    def source_jacobian(self):
        """ generated automatically unless explicitly provided """
        return self._simplify(
                Matrix(self.source()).jacobian(self.variables),
            )

    def source_implicit(self):
        return zeros(self.n_variables, 1)

    def residual(self):
        return zeros(self.n_variables, 1)
    
    def interpolate_3d(self):
        return zeros(6, 1)


    def eigenvalues(self):
        A = self.normal[0] * self.quasilinear_matrix()[0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[d]
        return eigenvalue_dict_to_matrix(A.eigenvals())
    
    def left_eigenvectors(self):
        return zeros(self.n_variables, self.n_variables)
    
    def right_eigenvectors(self):
        return zeros(self.n_variables, self.n_variables)


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
    return Matrix(evs)
