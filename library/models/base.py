import numpy as np
import os
import logging

import sympy
from sympy import Symbol, Matrix, lambdify
from sympy import *
from sympy import zeros, ones

from attr import define


from library.baseclass import BaseYaml
from library.boundary_conditions import BoundaryCondition, Periodic
from library.initial_condition import InitialCondition, Default


@define(slots=True, frozen=True, kw_only=True)
class Model:
    dimension: int = 1
    boundary_conditions: BoundaryCondition = [
        Periodic(physical_tag="left", periodic_to_physical_tag="right"),
        Periodic(physical_tag="right", periodic_to_physical_tag="left"),
    ]
    initial_conditions: InitialCondition = InitialCondition()


class Model(BaseYaml):
    yaml_tag = "!Model"
    dimension = 1

    def set_default_parameters(self):
        self.boundary_conditions = [
            Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            Periodic(physical_tag="right", periodic_to_physical_tag="left"),
        ]

        self.initial_conditions = initial_condition.Constant()

        self.initial_conditions = {"scheme": "func", "name": "one"}
        self.nc_treatment = None

        self.inclination_angle = 0.0
        self.rotation_angle_xy = 0.0
        self.g = 1.0

        self.n_fields = 1
        self.field_names = None

        self.friction_models = []
        self.parameters = {}

    def set_runtime_variables(self):
        self.compute_unit_vectors()

    def compute_unit_vectors(self):
        self.ex = np.sin(self.inclination_angle * 2 * np.pi / 360.0) * np.cos(
            self.rotation_angle_xy * 2 * np.pi / 360.0
        )
        self.ey = np.sin(self.inclination_angle * 2 * np.pi / 360.0) * np.sin(
            self.rotation_angle_xy * 2 * np.pi / 360.0
        )
        self.ez = np.sqrt(1 - (self.ex**2 + self.ey**2))

    def flux(self, Q):
        return np.zeros_like(Q)

    def flux_jacobian(self, Q):
        return np.zeros((Q.shape[0], Q.shape[0], self.dimension, Q.shape[1]))

    def eigenvalues(self, Q, nij):
        imaginary = False
        evs = np.zeros_like(Q)
        An = np.einsum(
            "ijk..., k...->...ij", self.quasilinear_matrix(Q), nij[: self.dimension]
        )
        EVs = np.linalg.eigvals(An[:, :-1, :-1])
        for i in range(Q.shape[1]):
            ev = EVs[i, :]
            imaginary = imaginary or not (np.isreal(ev).all())
            logger = logging.getLogger(__name__ + ":eigenvalues")
            if not np.isreal(ev).all() and np.abs(np.imag(ev)).max() > 10 ** (-8):
                logger.error(
                    "imaginary eigenvalues encountered with Evs = {} for Q = {}".format(
                        ev, Q[:, i]
                    )
                )
            evs[:-1, i] = np.real(ev)
        # for i in range(Q.shape[1]):
        #     ev = np.linalg.eigvals(
        #         np.einsum(An[:,:,i])
        #     )
        #     imaginary = imaginary or not (np.isreal(ev).all())
        #     logger = logging.getLogger(__name__ + ":eigenvalues")
        #     if not np.isreal(ev).all():
        #         logger.error(
        #             "imaginary eigenvalues encountered with Evs = {} for Q = {}".format(
        #                 ev, Q[:, i]
        #             )
        #         )
        #     evs[:, i] = np.real(ev)
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

    def get_massmatrix(self):
        return np.eye(self.n_fields)

    def get_massmatrix_inverse(self):
        return np.eye(self.n_fields)

    def get_boundary_tags(self):
        physical_tags = []
        for bc in self.boundary_conditions:
            physical_tags.append(bc.physical_tag)
        physical_tags = np.unique(physical_tags)
        return physical_tags

    # def proj_Q_onto_normal(self, Q, normal):
    #     result = np.array(Q)
    #     dim = self.dimension
    #     result[1:1+dim] = Q[1:1+dim]*normal[:dim]
    #     return result


class Model2d(Model):
    yaml_tag = "!Model2D"
    dimension = 2

    def set_default_parameters(self):
        self.boundary_conditions = [
            Periodic(physical_tag="left", periodic_to_physical_tag="right"),
            Periodic(physical_tag="right", periodic_to_physical_tag="left"),
            Periodic(physical_tag="bottom", periodic_to_physical_tag="top"),
            Periodic(physical_tag="top", periodic_to_physical_tag="bottom"),
        ]

        self.friction_models = []
        self.parameters = {}

        # support e.g. lambda, file, array, TC (from initial_fields library)
        self.initial_conditions = {"scheme": "func", "name": "one"}
        self.nc_treatment = None

        self.inclination_angle = 0.0
        self.rotation_angle_xy = 0.0
        self.g = 1.0

        self.n_fields = 1
        self.field_names = None

        self.parameters = {}

    def flux(self, Q):
        return np.zeros((Q.shape[0], 2, Q.shape[1]))

    def flux_jacobian(self, Q):
        return np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))

    def nonconservative_matrix(self, Q, **kwargs):
        return np.zeros((Q.shape[0], Q.shape[0], 2, Q.shape[1]))


def register_variables(number_of_unknowns, string_identifier="q_"):
    return [
        sympy.symbols(string_identifier + str(v)) for v in range(number_of_unknowns)
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
        self.variables = register_variables(self.n_fields)
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
