import jax.numpy as jnp
import numpy as np
import numpy.ctypeslib as npct
from scipy.linalg import eigvals
import os
import logging
from copy import deepcopy
from ctypes import cdll
from functools import partial

import sympy
from sympy import Symbol, Matrix, lambdify, transpose, powsimp, MatrixSymbol, fraction, cancel
from sympy import zeros, ones
from sympy.utilities.autowrap import autowrap, ufuncify, make_routine
from sympy.abc import x, y 

from attr import define
from typing import Optional, Any, Type, Union, Callable
from types import SimpleNamespace


from library.model.boundary_conditions import BoundaryConditions, Extrapolation
from library.model.initial_conditions import InitialConditions, Constant
from library.misc.custom_types import FArray
from library.misc.misc import vectorize  # type: ignore
from library.misc.misc import IterableNamespace
from library.model.sympy2c import create_module

def vectorize_constant_sympy_expressions(expr, Q, Qaux):
    symbol_list = Q.get_list() + Qaux.get_list()
    q0 = Q[0]
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            if not any(symbol in expr[i, j].free_symbols for symbol in symbol_list):
                if expr[i, j] == 0:
                    expr[i, j] = 10**(-20) * q0
                elif expr[i, j]:
                    expr[i, j] = expr[i, j] + 10**(-20) * q0
                    # print('I don\'t know how to vectorize this yet')
                    # assert False
    return expr

def vectorize_nonconservative_matrix(expr, lambdified_expr):
    nonconservative_matrix = []
    for e, le in zip(expr, lambdified_expr):
        if len(list(e.free_symbols)) > 0:
            nonconservative_matrix.append(lambda Q, Qaux, param, f=le:  f(Q, Qaux, param))
        else:
            #constant matrix. Needs to be vectorized.
            constant_matrix = np.array(e, dtype=float)
            nonconservative_matrix.append(lambda Q, Qaux, param, f=le:  np.stack([ constant_matrix for i in range(Q.shape[1])], axis=-1))
    return nonconservative_matrix


def custom_simplify(expr):
    return powsimp(expr, combine="all", force=False, deep=True)

def regularize_denominator(expr, regularization_constant = 10**(-4), regularize = True):
    if not regularize:
        return expr
    def regularize(expr):
        (nom, den) = fraction(cancel(expr))
        return nom * den / (den*2 + regularization_constant)
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            expr[i,j] = regularize(expr[i,j])
    return expr



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

    bcs: list[Callable]

    @classmethod
    def from_model(cls, model):
        pde = model.get_pde()
        # bcs = model.create_python_boundary_interface(printer='numpy')
        bcs.get_boundary_conditions()
        return cls(model.name, model.dimension, model.n_fields, model.n_aux_fields, model.n_parameters, pde.flux, pde.flux_jacobian, pde.source, pde.source_jacobian, pde.nonconservative_matrix, pde.quasilinear_matrix, pde.eigenvalues, pde.left_eigenvectors, pde.right_eigenvectors, bcs)


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
        self.name = 'Model'
        self.dimension = dimension
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.aux_initial_conditions = aux_initial_conditions

        self.variables = register_sympy_attribute(fields, "q")
        self.aux_variables = register_sympy_attribute(aux_fields, "aux")
        self.time = sympy.symbols('time', real=True)
        self.position = register_sympy_attribute(dimension, "X")
        self.distance = sympy.symbols('dX', real=True)
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
        # self.sympy_quasilinear_matrix = self.quasilinear_matrix()
        # TODO check case imaginary
        # TODO check case not computable
        if self.settings.eigenvalue_mode == 'symbolic':
            self.sympy_eigenvalues = self.eigenvalues()
        else:
            self.sympy_eigenvalues = None
        self.sympy_left_eigenvectors = None
        self.sympy_right_eigenvectors = None

    def load_c_boundary_conditions(self, n_boundary_functions, path='./.tmp'):
        folder = os.path.join(path, self.name)
        files_found = []
        for file in os.listdir(folder):
            if file.startswith("boundary_conditions") and file.endswith(".so"):
                files_found.append(os.path.join(folder, file))

        assert len(files_found) == 1
        library_path = files_found[0]

        c_bcs = cdll.LoadLibrary(library_path)

        boundary_functions = [None for i in range(n_boundary_functions)]
        for i in range(n_boundary_functions):
            boundary_functions[i] = eval(f"c_bcs.boundary_condition_{i}")

        # define array prototypes
        array_1d_double = npct.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
    
        # define function prototype
        for i in range(n_boundary_functions):
            boundary_functions[i].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double, array_1d_double]
            boundary_functions[i].restype = None

        return boundary_functions
        
    # def create_python_boundary_interface(self, printer='numpy'):
    #     # define matrix symbols that will be used as substitutions for the currelty used symbols in the
    #     # expressions
    #     assert self.boundary_conditions.initialized
    #     Q = MatrixSymbol('Q', self.n_fields, 1)
    #     Q_ghost = MatrixSymbol('Qg', self.n_fields, 1)
    #     Qaux = MatrixSymbol('Qaux', self.n_aux_fields, 1)
    #     parameters = MatrixSymbol('parameters', self.n_parameters, 1)
    #     normal = MatrixSymbol('normal', self.dimension, 1)


    #     sympy_boundary_functions = deepcopy(self.boundary_conditions.boundary_functions)

    #     # aggregate all expressions that are substituted and converted to C into the right data structure
    #     list_matrix_symbols = [Q, Qaux, parameters, normal]
    #     list_attributes = [self.variables, self.aux_variables, self.parameters, self.sympy_normal]
    #     list_expressions = sympy_boundary_functions
    #     list_expression_names = [f'boundary_condition_{i}' for i in range(len(sympy_boundary_functions))]

    #     # convert symbols to matrix symbols
    #     for i in range(len(list_expressions)):
    #         for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
    #             list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
                

    #     # aggregate data structure to be passed to the C converter module
    #     expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]

    #     runtime_bc_functions = []
    #     for name, expr, [Q, Qaux, parameters, normal] in expression_name_tuples
    #         runtime_bc_functions.append(sympy.lambdify([Q, Qaux, parameters, normal], expr, modules=printer))
    #     return runtime_bc_functions


    #TODO create c_boundary_interface similat to create_c_interface
    def create_c_boundary_interface(self, path='.tmp/'):
        # define matrix symbols that will be used as substitutions for the currelty used symbols in the
        # expressions
        Q = MatrixSymbol('Q', self.n_fields, 1)
        Q_ghost = MatrixSymbol('Qg', self.n_fields, 1)
        Qaux = MatrixSymbol('Qaux', self.n_aux_fields, 1)
        parameters = MatrixSymbol('parameters', self.n_parameters, 1)
        normal = MatrixSymbol('normal', self.dimension, 1)


        sympy_boundary_functions = deepcopy(self.boundary_conditions.boundary_functions)
        #TODO create module
        #TODO delete the code below

        # aggregate all expressions that are substituted and converted to C into the right data structure
        list_matrix_symbols = [Q, Qaux, parameters, normal]
        list_attributes = [self.variables, self.aux_variables, self.parameters, self.sympy_normal]
        list_expressions = sympy_boundary_functions
        list_expression_names = [f'boundary_condition_{i}' for i in range(len(sympy_boundary_functions))]

        # convert symbols to matrix symbols
        for i in range(len(list_expressions)):
            for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
                list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
                

        # aggregate data structure to be passed to the C converter module
        expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]

        directory = os.path.join(path, self.name)
        module_name = 'boundary_conditions'

        # create c module
        create_module(module_name, expression_name_tuples, directory)

    #TODO write to output/.tmp
    #TODO rename to c_model_interface
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
        sympy_source_jacobian = deepcopy(regularize_denominator(self.sympy_source_jacobian))
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
            if file.startswith("model") and file.endswith(".so"):
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

    def get_boundary_conditions(self, printer="numpy"):
        """Returns a runtime boundary_conditions for numpy arrays from the symbolic model."""
        n_boundary_functions = len(self.boundary_conditions.boundary_functions)
        # bcs = [None for i in range(n_boundary_functions)]
        # l_bcs = []
        bcs = []
        for i in range(n_boundary_functions):
            func =lambdify(
            [
                self.time,
                self.position.get_list(),
                self.distance,
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
                self.sympy_normal.get_list(),
            ],
            # vectorize_constant_sympy_expressions(self.boundary_conditions.boundary_functions[i], self.variables, self.aux_variables),
            self.boundary_conditions.boundary_functions[i],
                modules={"jax.numpy": jnp})
            # the func=func part is necessary, because of https://stackoverflow.com/questions/46535577/initialising-a-list-of-lambda-functions-in-python/46535637#46535637
            f = lambda time, position, distance, q, qaux, p, n, func=func: jnp.squeeze(func(time, position, distance, q, qaux, p, n ), axis=-1)
            bcs.append(f)
        return bcs


    def get_pde(self, printer='numpy'):
        """Returns a runtime model for numpy arrays from the symbolic model."""
        l_flux = [lambdify(
            (
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ),
            vectorize_constant_sympy_expressions(self.sympy_flux[d], self.variables, self.aux_variables),
            printer,
        ) for d in range(self.dimension)]
        # the f=l_flux[d] part is necessary, because of https://stackoverflow.com/questions/46535577/initialising-a-list-of-lambda-functions-in-python/46535637#46535637
        # flux = [lambda Q, Qaux, param, f=l_flux[d]:  np.squeeze(np.array(f(Q, Qaux, param)), axis=-1) for d in range(self.dimension)]
        flux = [lambda Q, Qaux, param, f=l_flux[d]:  np.squeeze(np.array(f(Q, Qaux, param)), axis=1) for d in range(self.dimension)]
        # flux = [lambda Q, Qaux, param, f=l_flux[d]:  jnp.squeeze(jnp.array(f(Q, Qaux, param)), axis=-1) for d in range(self.dimension)]
        # flux = l_flux
        l_flux_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            self.sympy_flux_jacobian,
            printer,
        )
        # flux_jacobian = vectorize(l_flux_jacobian)
        flux_jacobian = l_flux_jacobian

        l_nonconservative_matrix = [lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(self.sympy_nonconservative_matrix[d], self.variables, self.aux_variables),
            printer,
        ) for d in range(self.dimension)]
        nonconservative_matrix = [lambda Q, Qaux, param, f=l_nonconservative_matrix[d]:  f(Q, Qaux, param) for d in range(self.dimension)]
        # nonconservative_matrix = vectorize_nonconservative_matrix(self.sympy_nonconservative_matrix, l_nonconservative_matrix)
        # nonconservative_matrix = l_nonconservative_matrix
        # nonconservative_matrix = vectorize(l_nonconservative_matrix)

        l_quasilinear_matrix = [lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(self.sympy_quasilinear_matrix[d], self.variables, self.aux_variables),
            printer,
        ) for d in range(self.dimension)]
        quasilinear_matrix = l_quasilinear_matrix
        # quasilinear_matrix = vectorize(l_quasilinear_matrix)

        if self.settings.eigenvalue_mode == 'symbolic':
            l_eigenvalues = lambdify(
                [
                    self.variables.get_list(),
                    self.aux_variables.get_list(),
                    self.parameters.get_list(),
                    self.sympy_normal.get_list(),
                ],
                vectorize_constant_sympy_expressions(self.sympy_eigenvalues, self.variables, self.aux_variables),
                printer,
            )
            # eigenvalues = lambda Q, Qaux, param, normal :  np.squeeze(np.array(l_eigenvalues(Q, Qaux, param, normal)), axis=-1)
            eigenvalues = lambda Q, Qaux, param, normal :  np.squeeze(np.array(l_eigenvalues(Q, Qaux, param, normal)), axis=1)
            # eigenvalues = vectorize(l_eigenvalues, n_arguments=4)
        elif self.settings.eigenvalue_mode == 'numerical':
            eigenvalues = None

        l_source = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(self.sympy_source, self.variables, self.aux_variables),
            printer,
        )
        source = lambda Q, Qaux, param:  np.squeeze(np.array(l_source(Q, Qaux, param)), axis=1)
        # source = vectorize(l_source)

        l_source_jacobian = lambdify(
            [
                self.variables.get_list(),
                self.aux_variables.get_list(),
                self.parameters.get_list(),
            ],
            vectorize_constant_sympy_expressions(self.sympy_source_jacobian, self.variables, self.aux_variables),
            printer,
        )
        source_jacobian = l_source_jacobian
        # source_jacobian = vectorize(l_source_jacobian)

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

    def quasilinear_matrix(self):
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
            evs.append(custom_simplify(ev))
    return Matrix(evs)




class ModelGUI(Model):
    """
    :gui:
    { 'parameters': { 'I': {'type': 'int', 'value': 1, 'step': 1}, 'F': {'type': 'float', 'value': 1., 'step':0.1}, 'S': {'type': 'string', 'value': 'asdf'}, 'Arr':{'type': 'array', 'value': np.array([[1., 2.], [3., 4.]])} },}
    """
