import numpy as np
import numpy.ctypeslib as npct
from scipy.linalg import eigvals
import os
import logging
from copy import deepcopy
from ctypes import cdll

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, powsimp, MatrixSymbol
from sympy import zeros, ones
from sympy.utilities.autowrap import autowrap, ufuncify, make_routine
from sympy.abc import x, y 

from attr import define
from typing import Optional, Any, Type, Union
from types import SimpleNamespace


from library.model.boundary_conditions import BoundaryConditions, Periodic
from library.model.initial_conditions import InitialConditions, Constant
from library.misc.custom_types import FArray
from library.misc.misc import vectorize  # type: ignore
from library.misc.misc import IterableNamespace
from library.model.sympy2c import create_module

def get_numerical_eigenvalues(dim , n_fields, quasilinear_matrix):
    def numerical_eigenvalues(Q, Qaux, parameters, normal, Qout):
        A = np.empty((n_fields, n_fields), dtype=float)
        An = np.zeros((n_fields, n_fields), dtype=float)
        for d in range(dim):
            quasilinear_matrix[d](Q, Qaux, parameters, A)
            An += normal[d] * A
        # Qout = np.linalg.eigvals(An)
        Qout = eigvals(An)
        if np.iscomplex(Qout).any():
            print(Qout)
            print(Q)
            print(An)
            assert False
        return np.array(Qout, dtype=float)
    return numerical_eigenvalues

"""
Warning: This is only needed because the C-function and the numerical_eigenvalues are not comparible (C-pointer vs return value)
"""
def get_sympy_eigenvalues(eigenvalue_func):
    def numerical_eigenvalues(Q, Qaux, parameters, normal):
        Qout = np.zeros((Q.shape[0], 1), dtype=float)
        Qaux = np.zeros((Q.shape[0]), dtype=float)
        eigenvalue_func(Q, Qaux, parameters, normal, Qout)
        return Qout
    return numerical_eigenvalues
    # return eigenvalue_func

    


@define(slots=True, frozen=False, kw_only=True)
class Model:
    name: str
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
    parameters_default: dict
    parameter_values: FArray
    sympy_normal: Matrix

    settings: IterableNamespace
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
        boundary_conditions: BoundaryConditions,
        initial_conditions: InitialConditions,
        aux_initial_conditions: InitialConditions = Constant(),
        parameters: dict = {},
        parameters_default: dict = {},
        settings: dict = {},
        settings_default: dict = {"eigenvalue_mode": "symbolic" },
    ):
        self.name = 'Model' + '_{}'.format(dimension)
        self.dimension = dimension
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.aux_initial_conditions = aux_initial_conditions

        self.variables = register_sympy_attribute(fields, "q")
        self.aux_variables = register_sympy_attribute(aux_fields, "aux")
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


    def init_sympy_functions(self):
        self.sympy_flux = self.flux()
        if self.flux_jacobian() is None:
            self.sympy_flux_jacobian = [
                zeros(self.n_fields, self.n_fields) for i in range(self.dimension)
            ]
            for d in range(self.dimension):
                self.sympy_flux_jacobian[d] = Matrix(
                    powsimp(
                        Matrix(self.sympy_flux[d]).jacobian(self.variables),
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
        # TODO check case imaginary
        # TODO check case not computable
        if self.settings.eigenvalue_mode == 'symbolic':
            self.sympy_eigenvalues = self.eigenvalues()
        else:
            self.sympy_eigenvalues = None
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None

    def create_c_interface(self, path='./.tmp/'):
        # define matrix symbols that will be used as substitutions for the currelty used symbols in the
        # expressions
        Q = MatrixSymbol('Q', self.n_fields, 1)
        Qaux = MatrixSymbol('Qaux', self.n_aux_fields, 1)
        parameters = MatrixSymbol('parameters', self.n_parameters, 1)
        normal = MatrixSymbol('normal', self.dimension, 1)

        # copy in order to not mess up the sypy expressions of the class - in case I want to call the
        # function a second time
        sympy_flux = deepcopy(self.sympy_flux)
        sympy_flux_jacobian = deepcopy(self.sympy_flux_jacobian)
        sympy_source = deepcopy(self.sympy_source)
        sympy_source_jacobian = deepcopy(self.sympy_source_jacobian)
        sympy_nonconservative_matrix = deepcopy(self.sympy_nonconservative_matrix)
        sympy_quasilinear_matrix = deepcopy(self.sympy_quasilinear_matrix)
        sympy_eigenvalues = deepcopy(self.sympy_eigenvalues)
        sympy_left_eigenvectors = deepcopy(self.sympy_left_eigenvectors)
        sympy_right_eigenvectors = deepcopy(self.sympy_right_eigenvectors)

        # make all dimension dependent functions 3d to simplify the C part of the interface
        if self.dimension == 1:
            sympy_flux = [sympy_flux[0], sympy_flux[0], sympy_flux[0]]
            sympy_flux_jacobian = [sympy_flux_jacobian[0], sympy_flux_jacobian[0], sympy_flux_jacobian[0]]
            sympy_nonconservative_matrix = [sympy_nonconservative_matrix[0], sympy_nonconservative_matrix[0], sympy_nonconservative_matrix[0]]
            sympy_quasilinear_matrix = [sympy_quasilinear_matrix[0], sympy_quasilinear_matrix[0], sympy_quasilinear_matrix[0]]
        elif self.dimension == 2:
            sympy_flux = [sympy_flux[0], sympy_flux[1], sympy_flux[0]]
            sympy_flux_jacobian = [sympy_flux_jacobian[0], sympy_flux_jacobian[1], sympy_flux_jacobian[0]]
            sympy_nonconservative_matrix = [sympy_nonconservative_matrix[0], sympy_nonconservative_matrix[1], sympy_nonconservative_matrix[0]]
            sympy_quasilinear_matrix = [sympy_quasilinear_matrix[0], sympy_quasilinear_matrix[1], sympy_quasilinear_matrix[0]]
        elif self.dimension == 3:
            pass
        else:
            assert False

        # aggregate all expressions that are substituted and converted to C into the right data structure
        list_matrix_symbols = [Q, Qaux, parameters]
        list_attributes = [self.variables, self.aux_variables, self.parameters]
        list_expressions = sympy_flux + sympy_flux_jacobian + sympy_nonconservative_matrix + sympy_quasilinear_matrix + [sympy_source, sympy_source_jacobian]
        list_expression_names = ['flux_x', 'flux_y', 'flux_z', 
                                 'flux_jacobian_x', 'flux_jacobian_y', 'flux_jacobian_z',
                                 'nonconservative_matrix_x', 'nonconservative_matrix_y', 'nonconservative_matrix_z',
                                 'quasilinear_matrix_x', 'quasilinear_matrix_y', 'quasilinear_matrix_z',
                                 'source', 'source_jacobian']
        list_matrix_symbols_incl_normal = [Q, Qaux, parameters, normal]
        list_attributes_incl_normal = [self.variables, self.aux_variables, self.parameters, self.sympy_normal]
        if sympy_eigenvalues is not None:
            list_expressions_incl_normal = [sympy_eigenvalues]
            list_expression_names_incl_normal = ['eigenvalues']
        else:
            list_expressions_incl_normal = []
            list_expression_names_incl_normal = []

        # convert symbols to matrix symbols
        for i in range(len(list_expressions)):
            for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
                list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
        for i in range(len(list_expressions_incl_normal)):
            for attr, matrix_symbol in zip(list_attributes_incl_normal, list_matrix_symbols_incl_normal):
                list_expressions_incl_normal[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_incl_normal[i], attr, matrix_symbol)
                

        # aggregate data structure to be passed to the C converter module
        expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]
        expression_name_tuples += [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names_incl_normal, list_expressions_incl_normal)]

        directory = os.path.join(path, self.name)
        module_name = 'model'

        # create c module
        create_module(module_name, expression_name_tuples, directory)
    
    def load_c_model(self, path='./.tmp'):
        folder = os.path.join(path, self.name)
        files_found = []
        for file in os.listdir(folder):
            if file.endswith(".so"):
                files_found.append(os.path.join(folder, file))

        assert len(files_found) == 1
        library_path = files_found[0]

        c_model = cdll.LoadLibrary(library_path)

        # the C module is constructed to use 3d functions, filled with dummies for dim<3. 
        # therefore we extract the correct dimension dependent functions now
        if self.dimension == 1:
            flux = [c_model.flux_x]
            flux_jacobian = [c_model.flux_jacobian_x]
            nonconservative_matrix = [c_model.nonconservative_matrix_x]
            quasilinear_matrix = [c_model.quasilinear_matrix_x]
        elif self.dimension == 2:
            flux = [c_model.flux_x, c_model.flux_y]
            flux_jacobian = [c_model.flux_jacobian_x, c_model.flux_jacobian_y]
            nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y]
            quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y]
        elif self.dimension == 3:
            flux = [c_model.flux_x, c_model.flux_y, c_model.flux_z]
            flux_jacobian = [c_model.flux_jggacobian_x, c_model.flux_jacobian_y, c_model.flux_jacobian_z]
            nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y, c_model.nonconservative_matrix_z]
            quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y, c_model.quasilinear_matrix_z]
        else:
            assert False

        source = c_model.source
        source_jacobian = c_model.source_jacobian
        if self.settings.eigenvalue_mode == 'symbolic':
            eigenvalues = c_model.eigenvalues
            # eigenvalues = get_sympy_eigenvalues( c_model.eigenvalues)
        elif self.settings.eigenvalue_mode == 'numerical':
            eigenvalues = get_numerical_eigenvalues(self.dimension, self.n_fields, quasilinear_matrix)
        else:
            assert False

        # define array prototypes
        array_2d_double = npct.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS')
        array_1d_double = npct.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
    
        # define function prototype
        for d in range(self.dimension):
            flux[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
            flux[d].restype = None
            flux_jacobian[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
            flux_jacobian[d].restype = None
            nonconservative_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
            nonconservative_matrix[d].restype = None
            quasilinear_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
            quasilinear_matrix[d].restype = None
        source.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
        source.restype = None
        source_jacobian.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
        source_jacobian.restype = None
        eigenvalues.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double, array_1d_double]
        eigenvalues.restype = None

        

        out = {'flux': flux, 'flux_jacobian': flux_jacobian, 'nonconservative_matrix':nonconservative_matrix, 'quasilinear_matrix':quasilinear_matrix, 'source': source, 'source_jacobian': source_jacobian, 'eigenvalues':eigenvalues}
        return SimpleNamespace(**out)


    def get_runtime_model(self):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = [lambdify(
            (
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ),
            self.sympy_flux[d],
            "numpy",
        ) for d in range(self.dimension)]
        # l_flux = lambda Q, Qaux, param: np.array(_l_flux(Q, Qaux, param))[:,:,:,0]
        # flux = vectorize(l_flux)
        flux = l_flux
        l_flux_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_flux_jacobian,
            "numpy",
        )
        # flux_jacobian = vectorize(l_flux_jacobian)
        flux_jacobian = l_flux_jacobian

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

        if self.settings.eigenvalue_mode == 'symbolic':
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
        elif self.settings.eigenvalue_mode == 'numerical':
            eigenvalues = None

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
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        return eigenvalue_dict_to_matrix(A.eigenvals())




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

def substitute_sympy_attributes_with_symbol_matrix(expr: Matrix, attr: IterableNamespace, attr_matrix: MatrixSymbol):
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
            evs.append(powsimp(ev, combine="all", force=True))
    return Matrix(evs)



