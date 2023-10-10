import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, powsimp

from sympy import zeros, ones

from attr import define
from typing import Optional, Any, Type, Union
from types import SimpleNamespace

from library.boundary_conditions import BoundaryConditions, Periodic
from library.initial_conditions import InitialConditions, Constant
from library.custom_types import FArray
from library.misc import vectorize  # type: ignore
from library.misc import IterableNamespace


@define(slots=True, frozen=False, kw_only=True)
class Model:
    dimension: int = 1
    boundary_conditions: BoundaryConditions = BoundaryConditions(
        [
            Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]
    )
    initial_conditions: InitialConditions = Constant()
    aux_initial_conditions: InitialConditions = Constant()

    n_fields: int
    n_aux_fields: int
    n_parameters: int
    variables: IterableNamespace
    aux_variables: IterableNamespace
    parameters: IterableNamespace
    parameter_defaults: FArray
    sympy_normal: Matrix

    settings: SimpleNamespace
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

    def __init__(
        self,
        dimension: int,
        fields: Union[int, list],
        aux_fields: Union[int, list],
        parameters: Union[int, list, dict],
        boundary_conditions: BoundaryConditions,
        initial_conditions: InitialConditions,
        aux_initial_conditions: InitialConditions = Constant(),
        settings: dict = {},
        settings_default: dict = {},
    ):
        self.dimension = dimension
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.aux_initial_conditions = aux_initial_conditions

        self.variables = register_sympy_attribute(fields, "q")
        self.aux_variables = register_sympy_attribute(aux_fields, "aux")
        self.parameters = register_sympy_attribute(parameters, "p")
        self.parameter_defaults = register_parameter_defaults(parameters)
        self.sympy_normal = register_sympy_attribute(
            ["n" + str(i) for i in range(self.dimension)], "n"
        )

        self.settings = SimpleNamespace(**{**settings_default, **settings})

        self.n_fields = self.variables.length()
        self.n_aux_fields = self.aux_variables.length()
        self.n_parameters = self.parameters.length()

        self.init_sympy_functions()

    def init_sympy_functions(self):
        self.sympy_flux = self.flux()
        if self.flux_jacobian() is None:
            self.sympy_flux_jacobian = [
                zeros(self.n_fields, self.n_fields) for i in range(self.dimension)
            ]
            for d in range(self.dimension):
                self.sympy_flux_jacobian[d] = Matrix(
                    powsimp(
                        self.sympy_flux[d].jacobian(self.variables),
                        combine="all",
                        force=True,
                    )
                )
        else:
            self.sympy_flux_jacobian = self.flux_jacobian()
        self.sympy_source = self.source()
        if self.source_jacobian() is None:
            self.sympy_source_jacobian = powsimp(
                self.sympy_source.jacobian(self.variables), combine="all", force=True
            )
        else:
            self.sympy_source_jacobian = self.source_jacobian()
        self.sympy_nonconservative_matrix = self.nonconservative_matrix()
        self.sympy_quasilinear_matrix = [
            self.sympy_flux_jacobian[d] - self.sympy_nonconservative_matrix[d]
            for d in range(self.dimension)
        ]
        # TODO case imaginary
        # TODO case not computable
        if self.eigenvalues() is None:
            A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
            for d in range(1, self.dimension):
                A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
            self.sympy_eigenvalues = eigenvalue_dict_to_matrix(A.eigenvals())
        else:
            self.sympy_eigenvalues = self.eigenvalues()
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None

    def get_runtime_model(self):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_flux,
            modules="numpy",
        )
        flux = vectorize(l_flux)
        l_flux_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_flux_jacobian,
            "numpy",
        )
        flux_jacobian = vectorize(l_flux_jacobian)

        l_nonconservative_matrix = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_nonconservative_matrix,
            "numpy",
        )
        nonconservative_matrix = vectorize(l_nonconservative_matrix)

        l_quasilinear_matrix = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_quasilinear_matrix,
            "numpy",
        )
        quasilinear_matrix = vectorize(l_quasilinear_matrix)

        l_eigenvalues = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.sympy_normal.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_eigenvalues,
            "numpy",
        )
        eigenvalues = vectorize(l_eigenvalues, n_arguments=4)

        l_source = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_source,
            "numpy",
        )
        source = vectorize(l_source)

        l_source_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_source_jacobian,
            "numpy",
        )
        source_jacobian = vectorize(l_source_jacobian)

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
        }
        return SimpleNamespace(**d)

    def flux(self):
        return [Matrix(self.variables[:]) for d in range(self.dimension)]

    def nonconservative_matrix(self):
        return [zeros(self.n_fields, self.n_fields) for d in range(self.dimension)]

    def source(self):
        return zeros(self.n_fields, 1)

    def flux_jacobian(self):
        return None

    def source_jacobian(self):
        return None

    def eigenvalues(self):
        return None


class Advection(Model):
    def flux(self):
        # assume that the first variables.length() parameters are the corresponding advection speeds
        assert self.parameters.length() >= self.variables.length()
        assert self.n_fields == self.dimension
        flux = []
        for d in range(self.dimension):
            f = []
            for i in range(self.n_fields):
                f.append(self.variables[i] * self.parameters[i])
            flux.append(f)
        return flux


def register_sympy_attribute(argument, string_identifier="q_"):
    if type(argument) == int:
        attributes = {
            string_identifier
            + str(i): sympy.symbols(string_identifier + str(i), real=True)
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
    return IterableNamespace(**attributes)


def register_parameter_defaults(parameters):
    if type(parameters) == int:
        default_values = np.zeros(parameters, dtype=float)
    elif type(parameters) == type({}):
        default_values = np.array([value for value in parameters.values()])
    else:
        assert False
    return default_values


def eigenvalue_dict_to_matrix(eigenvalues):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(powsimp(ev, combine="all", force=True))
    return Matrix(evs)
