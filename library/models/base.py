import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify, transpose
from sympy import *
from sympy import zeros, ones

from jax import vmap

from attr import define
from typing import Optional
from types import SimpleNamespace

from library.boundary_conditions import BoundaryConditions, Periodic
from library.initial_conditions import InitialConditions, Constant
from library.custom_types import FArray
from library.misc import vectorize # type: ignore


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
    n_fields: int
    n_aux_fields: int
    n_parameters: int

    variables: list[Symbol]
    aux_variables: list[Symbol]
    parameters: list[Symbol]

    sympy_flux: Matrix
    sympy_flux_jacobian: Matrix
    sympy_source: Matrix
    sympy_source_jacobian: Matrix
    sympy_nonconservative_matrix: Matrix
    sympy_quasilinear_matrix: Matrix
    sympy_eigenvalues: Matrix
    sympy_left_eigenvectors: Optional[Matrix]
    sympy_right_eigenvectors: Optional[Matrix]

    def __init__(
        self,
        dimension,
        n_fields,
        n_aux_fields,
        n_parameters,
        boundary_conditions,
        initial_conditions,
    ):
        self.dimension = dimension
        self.n_fields = n_fields
        self.n_aux_fields = n_aux_fields
        self.n_parameters = n_parameters
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions

        self.variables = register_sympy_variables(n_fields)
        self.aux_variables = register_sympy_auxilliary_variables(n_aux_fields)
        self.parameters = register_sympy_parameters(n_parameters)

        self.sympy_flux = self.flux()
        if self.flux_jacobian() is None:
            self.sympy_flux_jacobian = self.sympy_flux.jacobian(self.variables)
        else:
            self.sympy_flux_jacobian = self.flux_jacobian()
        self.sympy_source = self.source()
        if self.source_jacobian() is None:
            self.sympy_source_jacobian = self.sympy_source.jacobian(self.variables)
        else:
            self.sympy_source_jacobian = self.source_jacobian()
        self.sympy_nonconservative_matrix = self.nonconservative_matrix()
        self.sympy_quasilinear_matrix = (
            self.sympy_flux_jacobian - self.sympy_nonconservative_matrix
        )
        # TODO case imaginary
        # TODO case not computable
        if self.eigenvalues() is None:
            self.sympy_eigenvalues = eigenvalue_dict_to_matrix(
                self.sympy_quasilinear_matrix.eigenvals()
            )
        else:
            self.sympy_eigenvalues = self.eigenvalues()
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None

    def get_runtime_model(self):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_flux,
            modules="numpy",
        )
        flux = vectorize(l_flux)
        l_flux_jacobian = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_flux_jacobian,
            "numpy",
        )
        flux_jacobian = vectorize(l_flux_jacobian)

        l_nonconservative_matrix = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_nonconservative_matrix,
            "numpy",
        )
        nonconservative_matrix = vectorize(l_nonconservative_matrix)

        l_quasilinear_matrix = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_quasilinear_matrix,
            "numpy",
        )
        quasilinear_matrix = vectorize(l_quasilinear_matrix)

        l_eigenvalues = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_eigenvalues,
            "numpy",
        )
        eigenvalues = vectorize(l_eigenvalues)

        l_source = lambdify(
            [self.variables, self.aux_variables, self.parameters],
            self.sympy_source,
            "numpy",
        )
        source = vectorize(l_source)

        l_source_jacobian = lambdify(
            [self.variables, self.aux_variables, self.parameters],
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
        flux = []
        for q in self.variables:
            flux.append(q)
        return Matrix(flux)

    def nonconservative_matrix(self):
        return zeros(self.n_fields, self.n_fields)

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
        assert len(self.parameters) >= len(self.variables)
        flux = []
        for q, p in zip(self.variables, self.parameters):
            flux.append(p * q)
        return Matrix(flux)


def register_sympy_variables(number_of_unknowns, string_identifier="q_"):
    return [
        sympy.symbols(string_identifier + str(v)) for v in range(number_of_unknowns)
    ]


def register_sympy_auxilliary_variables(number_of_aux_unknowns, string_identifier="a_"):
    return [
        sympy.symbols(string_identifier + str(v)) for v in range(number_of_aux_unknowns)
    ]


def register_sympy_parameters(number_of_parameters, string_identifier="p_"):
    return [
        sympy.symbols(string_identifier + str(v)) for v in range(number_of_parameters)
    ]


def eigenvalue_dict_to_matrix(eigenvalues):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(ev)
    return Matrix(evs)
