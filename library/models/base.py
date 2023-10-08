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


from collections import UserDict


from library.boundary_conditions import BoundaryCondition, Periodic
from library.initial_conditions import InitialConditions, Constant
from library.custom_types import FArray
from library.misc import vectorize


@define(slots=True, frozen=False, kw_only=True)
class Model:
    dimension: int = 1
    boundary_conditions: BoundaryCondition = [
        Periodic(physical_tag="left", periodic_to_physical_tag="right"),
        Periodic(physical_tag="right", periodic_to_physical_tag="left"),
    ]
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
        self.sympy_eigenvalues = list(self.sympy_quasilinear_matrix.eigenvals().keys())
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None

    # TODO we currently need lambda to transpse the input. Otherwise sympy fails. better options? Maybe compile to c and load c instead?
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
        source = vectorize(l_eigenvalues)

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
        return UserDict(d)

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


class Advection(Model):
    def flux(self):
        assert len(self.parameters) >= len(self.variables)
        flux = []
        for q, p in zip(self.variables, self.parameters):
            flux.append(p * q)
        return Matrix(flux)


# class Model(BaseYaml):
#     yaml_tag = "!Model"
#     dimension = 1

#     def set_default_parameters(self):
#         self.boundary_conditions = [
#             Periodic(physical_tag="left", periodic_to_physical_tag="right"),
#             Periodic(physical_tag="right", periodic_to_physical_tag="left"),
#         ]

#         self.initial_conditions = Constant()

#         self.initial_conditions = {"scheme": "func", "name": "one"}
#         self.nc_treatment = None

#         self.inclination_angle = 0.0
#         self.rotation_angle_xy = 0.0
#         self.g = 1.0

#         self.n_fields = 1
#         self.field_names = None

#         self.friction_models = []
#         self.parameters = {}

#     def set_runtime_variables(self):
#         self.compute_unit_vectors()

#     def compute_unit_vectors(self):
#         self.ex = np.sin(self.inclination_angle * 2 * np.pi / 360.0) * np.cos(
#             self.rotation_angle_xy * 2 * np.pi / 360.0
#         )
#         self.ey = np.sin(self.inclination_angle * 2 * np.pi / 360.0) * np.sin(
#             self.rotation_angle_xy * 2 * np.pi / 360.0
#         )
#         self.ez = np.sqrt(1 - (self.ex**2 + self.ey**2))

#     def flux(self, Q):
#         return np.zeros_like(Q)

#     def flux_jacobian(self, Q):
#         return np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))

#     def eigenvalues(self, Q, nij):
#         imaginary = False
#         evs = np.zeros_like(Q)
#         An = np.einsum(
#             "ijk..., k...->...ij", self.quasilinear_matrix(Q), nij[: self.dimension]
#         )
#         EVs = np.linalg.eigvals(An[:, :-1, :-1])
#         for i in range(Q.shape[1]):
#             ev = EVs[i, :]
#             imaginary = imaginary or not (np.isreal(ev).all())
#             logger = logging.getLogger(__name__ + ":eigenvalues")
#             if not np.isreal(ev).all() and np.abs(np.imag(ev)).max() > 10 ** (-8):
#                 logger.error(
#                     "imaginary eigenvalues encountered with Evs = {} for Q = {}".format(
#                         ev, Q[:, i]
#                     )
#                 )
#             evs[:-1, i] = np.real(ev)
#         # for i in range(Q.shape[1]):
#         #     ev = np.linalg.eigvals(
#         #         np.einsum(An[:,:,i])
#         #     )
#         #     imaginary = imaginary or not (np.isreal(ev).all())
#         #     logger = logging.getLogger(__name__ + ":eigenvalues")
#         #     if not np.isreal(ev).all():
#         #         logger.error(
#         #             "imaginary eigenvalues encountered with Evs = {} for Q = {}".format(
#         #                 ev, Q[:, i]
#         #             )
#         #         )
#         #     evs[:, i] = np.real(ev)
#         return evs, imaginary

#     def rhs(self, t, Q, **kwargs):
#         output = np.zeros_like(Q)
#         return output

#     def rhs_jacobian(self, t, Q, **kwargs):
#         return np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))

#     def nonconservative_matrix(self, Q, **kwargs):
#         return np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))

#     def quasilinear_matrix(self, Q, **kwargs):
#         return self.flux_jacobian(Q) - self.nonconservative_matrix(Q, **kwargs)

#     def primitive_variables(self, Q):
#         return Q

#     def conservative_variables(self, U):
#         return U

#     def get_massmatrix(self):
#         return np.eye(self.n_fields)

#     def get_massmatrix_inverse(self):
#         return np.eye(self.n_fields)

#     def get_boundary_tags(self):
#         physical_tags = []
#         for bc in self.boundary_conditions:
#             physical_tags.append(bc.physical_tag)
#         physical_tags = np.unique(physical_tags)
#         return physical_tags

#     # def proj_Q_onto_normal(self, Q, normal):
#     #     result = np.array(Q)
#     #     dim = self.dimension
#     #     result[1:1+dim] = Q[1:1+dim]*normal[:dim]
#     #     return result


# class Model2d(Model):
#     yaml_tag = "!Model2D"
#     dimension = 2

#     def set_default_parameters(self):
#         self.boundary_conditions = [
#             Periodic(physical_tag="left", periodic_to_physical_tag="right"),
#             Periodic(physical_tag="right", periodic_to_physical_tag="left"),
#             Periodic(physical_tag="bottom", periodic_to_physical_tag="top"),
#             Periodic(physical_tag="top", periodic_to_physical_tag="bottom"),
#         ]

#         self.friction_models = []
#         self.parameters = {}

#         # support e.g. lambda, file, array, TC (from initial_fields library)
#         self.initial_conditions = {"scheme": "func", "name": "one"}
#         self.nc_treatment = None

#         self.inclination_angle = 0.0
#         self.rotation_angle_xy = 0.0
#         self.g = 1.0

#         self.n_fields = 1
#         self.field_names = None

#         self.parameters = {}

#     def flux(self, Q):
#         return np.zeros((Q.shape[0], 2, Q.shape[1]))

#     def flux_jacobian(self, Q):
#         return np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))

#     def nonconservative_matrix(self, Q, **kwargs):
#         return np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))


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


class ModelSympy(Model):
    yaml_tag = "!ModelSympy"
    dimension = 1

    def set_default_parameters(self):
        self.boundary_conditions = [
            Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]
        self.boundary_conditions_flux = []

        self.initial_conditions = {"scheme": "func", "name": "one"}
        self.nc_treatment = None

        self.inclination_angle = 0.0
        self.rotation_angle_xy = 0.0
        self.g = 1.0

        self.n_fields = 2
        self.n_aux_fields = 0
        self.field_names = ["h", "hu"]

        self.friction_models = []
        self.parameters = {}

    def set_runtime_variables(self):
        self.variables = register_sympy_variables(self.n_fields)
        self.lambda_flux_func = lambdify(self.variables, self.sympy_flux(), "numpy")
        self.sympy_jacobian = self.sympy_flux().jacobian(self.variables)
        self.lambda_flux_jac_func = lambdify(
            self.variables, self.sympy_jacobian, "numpy"
        )
        self.sympy_nonconservative_matrix = self.sympy_nonconservative_matrix()
        self.lambda_nonconservative_matrix = lambdify(
            self.variables, self.sympy_nonconservative_matrix, "numpy"
        )
        self.sympy_quasilinar_matrix = (
            self.sympy_jacobian - self.sympy_nonconservative_matrix
        )
        self.lambda_quasiliner_matrix_func = lambdify(
            self.variables, self.sympy_quasilinear_matrix, "numpy"
        )
        # TODO case imaginary
        # TODO case not computable
        # TODO nij dependence
        self.sympy_eigenvalues = list(self.sympy_quasilinar_matrix.eigenvals().keys())
        self.lambda_eigenvalues = lambdify(
            self.variables, self.sympy_eigenvalues, "numpy"
        )

    def sympy_flux(self):
        h = self.variables[0]
        hu = self.variables[1]
        u = hu / h
        flux = []
        flux.append(hu)
        flux.append(h * u**2 + self.g * h**2 / 2)
        return Matrix(flux)

    def sympy_nonconservative_matrix(self):
        return zeros(self.n_fields, n_fields)

    def sympy_source(self):
        return zeros(self.n_fields, n_fields)

    def flux(self, Q):
        return self.lambda_flux_func(*Q)

    def flux_jacobian(self, Q):
        return self.lambda_flux_jac_func(*Q)

    def eigenvalues(self, Q, nij):
        imaginary = False
        evs = self.lambda_eigenvalues(*Q)
        return evs, imaginary

    def rhs(self, t, Q, **kwargs):
        output = np.zeros_like(Q)
        return output

    def rhs_jacobian(self, t, Q, **kwargs):
        return np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]))

    def nonconservative_matrix(self, Q, **kwargs):
        return np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))

    def quasilinear_matrix(self, Q, **kwargs):
        return self.flux_jacobian(Q) - self.nonconservative_matrix(Q, **kwargs)

    def primitive_variables(self, Q):
        return Q

    def conservative_variables(self, U):
        return U
