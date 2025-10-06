import sympy as sp
import ufl
import math
import os
from sympy import MatrixSymbol, fraction, cancel, Matrix, symbols
from sympy.utilities.lambdify import lambdify
from copy import deepcopy
from typing import Callable
from attrs import define, field
import dolfinx

import numpy as np

import ufl
import basix
from dolfinx import mesh, fem
from mpi4py import MPI

from library.misc.misc import Zstruct
from library.model.sympy2c import create_module
from library.transformation.helpers import regularize_denominator, substitute_sympy_attributes_with_symbol_matrix

ufl_map = {
    "math.exp": ufl.exp,
    "math.log": ufl.ln,
    "math.sin": ufl.sin,
    "math.cos": ufl.cos,
    "math.tan": ufl.tan,
    "sqrt": ufl.sqrt,
    "ImmutableDenseMatrix": ufl.as_vector,
}


def sympy_matrix_to_ufl(expr_matrix, sympy_vars, ufl_vars, mapping=ufl_map):
    """
    Convert a SymPy Matrix expression (or list of expressions) to UFL.

    Parameters
    ----------
    expr_matrix : list or sympy.Matrix
        SymPy expressions, one per component.
    sympy_vars : list of sympy.Symbol
        Symbols used in the expressions.
    ufl_vars : list
        Corresponding UFL variables.
    extra_mapping : dict, optional
        Extra mappings for math functions to UFL.

    Returns
    -------
    ufl_vector : list of ufl.Expr
        Converted UFL expressions.
    """

    # Ensure expr_matrix is a flat list
    if isinstance(expr_matrix, sp.Matrix):
        expr_list = list(expr_matrix)
    else:
        expr_list = expr_matrix


    ufl_exprs = []

    for expr in expr_list:
        # Use lambdify with custom UFL mapping
        f = lambdify(sympy_vars, expr, modules=[mapping])
    
        # Evaluate the lambda on UFL variables
        # ufl_expr = f(*ufl_vars)
        # ufl_exprs.append(ufl_expr)

    # Pack into a UFL vector
    # return ufl.as_vector(ufl_exprs)
    return ufl.as_tensor(f)

def to_ufl(model, settings):
    # define matrix symbols that will be used as substitutions for the currelty used symbols in the
    # expressions
    Q = MatrixSymbol('Q', model.n_variables, 1)
    Qaux = MatrixSymbol('Qaux', model.n_aux_variables, 1)
    parameters = MatrixSymbol('parameters', model.n_parameters, 1)
    normal = MatrixSymbol('normal', 3, 1)
    position = MatrixSymbol('position', 3, 1)
    time, distance = symbols(['time', 'distance'])
    

    # deepcopy to not mess up the sympy expressions of the class - in case I want to call the function a second time
    flux = deepcopy(model.flux())
    flux_jacobian = deepcopy(model.flux_jacobian())
    source = deepcopy(model.source())
    source_jacobian = deepcopy(regularize_denominator(model.source_jacobian()))
    nonconservative_matrix = deepcopy(model.nonconservative_matrix())
    quasilinear_matrix = deepcopy(model.quasilinear_matrix())
    eigenvalues = deepcopy(model.eigenvalues())
    left_eigenvectors = deepcopy(model.left_eigenvectors())
    right_eigenvectors = deepcopy(model.right_eigenvectors())
    source_implicit = deepcopy(model.source_implicit())
    residual = deepcopy(model.residual())
    interpolate_3d = deepcopy(model.interpolate_3d())
    boundary_conditions = deepcopy(model.boundary_conditions.get_boundary_function_matrix(model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, normal))

    # make all dimension dependent functions 3d to simplify the C part of the interface
    if model.dimension == 1:
        flux = [flux[0], flux[0], flux[0]]
        flux_jacobian = [flux_jacobian[0], flux_jacobian[0], flux_jacobian[0]]
        nonconservative_matrix = [nonconservative_matrix[0], nonconservative_matrix[0], nonconservative_matrix[0]]
        quasilinear_matrix = [quasilinear_matrix[0], quasilinear_matrix[0], quasilinear_matrix[0]]
    elif model.dimension == 2:
        flux = [flux[0], flux[1], flux[0]]
        flux_jacobian = [flux_jacobian[0], flux_jacobian[1], flux_jacobian[0]]
        nonconservative_matrix = [nonconservative_matrix[0], nonconservative_matrix[1], nonconservative_matrix[0]]
        quasilinear_matrix = [quasilinear_matrix[0], quasilinear_matrix[1], quasilinear_matrix[0]]
    elif model.dimension == 3:
        pass
    else:
        assert False

    # aggregate all expressions that are substituted and converted to UFL into the right data structure
    list_matrix_symbols = [Q, Qaux, parameters]
    list_attributes = [model.variables, model.aux_variables, model.parameters]
    list_expressions = flux + flux_jacobian + nonconservative_matrix + quasilinear_matrix + [source, source_jacobian, source_implicit, residual, interpolate_3d]
    list_expression_names = ['flux_x', 'flux_y', 'flux_z', 
                             'flux_jacobian_x', 'flux_jacobian_y', 'flux_jacobian_z',
                             'nonconservative_matrix_x', 'nonconservative_matrix_y', 'nonconservative_matrix_z',
                             'quasilinear_matrix_x', 'quasilinear_matrix_y', 'quasilinear_matrix_z',
                             'source', 'source_jacobian', 'source_implicit', 'residual']
    
    list_matrix_symbols_incl_normal = [Q, Qaux, parameters, normal]
    list_attributes_incl_normal = [model.variables, model.aux_variables, model.parameters, model.normal]
    if eigenvalues is not None:
        list_expressions_incl_normal = [eigenvalues, left_eigenvectors, right_eigenvectors]
        list_expression_names_incl_normal = ['eigenvalues', 'left_eigenvectors', 'right_eigenvectors']
    else:
        list_expressions_incl_normal = []
        list_expression_names_incl_normal = []
    
    list_matrix_symbols_incl_position = [Q, Qaux, parameters, position]
    list_attributes_incl_position = [model.variables, model.aux_variables, model.parameters, model.position]
    list_expressions_incl_position = [interpolate_3d]
    list_expression_names_incl_position = ['interpolate_3d']
    
    list_matrix_symbols_bc = [time, position, distance, Q, Qaux, parameters, normal]
    list_attributes_bc = [model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, model.normal]
    list_expressions_bc = [boundary_conditions]
    list_expression_names_bc = ['boundary_conditions']


    # convert symbols to matrix symbols
    for i in range(len(list_expressions)):
        for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
            list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
    for i in range(len(list_expressions_incl_normal)):
        for attr, matrix_symbol in zip(list_attributes_incl_normal, list_matrix_symbols_incl_normal):
            list_expressions_incl_normal[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_incl_normal[i], attr, matrix_symbol)
    for i in range(len(list_expressions_incl_position)):
        for attr, matrix_symbol in zip(list_attributes_incl_position, list_matrix_symbols_incl_position):
            list_expressions_incl_position[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_incl_position[i], attr, matrix_symbol)
    for i in range(len(list_expressions_bc)):
        for attr, matrix_symbol in zip(list_attributes_bc, list_matrix_symbols_bc):
            list_expressions_bc[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_bc[i], attr, matrix_symbol)    
            

    # aggregate data structure to be passed to the C converter module
    expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]
    expression_name_tuples += [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names_incl_normal, list_expressions_incl_normal)]
    expression_name_tuples += [(expr_name, expr, [Q, Qaux, parameters, position]) for (expr_name, expr) in zip(list_expression_names_incl_position, list_expressions_incl_position)]
    expression_name_tuples += [(expr_name, expr, list_matrix_symbols_bc) for (expr_name, expr) in zip(list_expression_names_bc, list_expressions_bc)]
    
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    
    elem_Q = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(3,))
    V_Q = fem.functionspace(domain, elem_Q)
    
    elem_Qaux = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(1,))
    V_Qaux = fem.functionspace(domain, elem_Qaux)

    
    # parameters can be simple numbers
    parameters = [9.81]
    
    Q = fem.Function(V_Q)
    Qaux = fem.Function(V_Qaux) 
    
    def ic_q(x):
        return np.array([1.*np.ones_like(x[0]), 2.*np.ones_like(x[0]), 3.*np.ones_like(x[0])])
    Q.interpolate(ic_q)

    # 3. Map SymPy symbols to UFL expressions
    ufl_vars = [Q[i] for i in range(3)] + [parameters[0]]
    
    a = sympy_matrix_to_ufl(flux[0], list_matrix_symbols, ufl_vars)
    print(a)
    return a


def get_ufl_boundary_functions(model):
    boundary_function_matrix = model.boundary_conditions.get_boundary_function_matrix(*model.get_boundary_conditions_matrix_inputs())
    n_bcs = boundary_function_matrix.shape[0]
    bc_funcs = []
    for i in range(n_bcs):
        bc_funcs.append(
            lambdify(
                # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
                (model.variables.get_list(),
                 model.normal.get_list()
                 ), # The (),) comma is crutial!
                boundary_function_matrix[i, :].T,
                ufl_map
            )
        )
    return bc_funcs

def get_lambda_function(model, function):
    f = lambdify(
        # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
        (
            model.variables.get_list(),
            model.aux_variables.get_list(),
            model.parameters.get_list(),
        ), # The (),) comma is crutial!
        function,
        ufl_map
    )
    return f
    
def get_lambda_function_with_normal(model, function):
    f = lambdify(
        # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
        (
            model.variables.get_list(),
            model.aux_variables.get_list(),
            model.parameters.get_list(),
            model.normal.get_list()
        ), # The (),) comma is crutial!
        function,
        ufl_map
    )
    return f

def get_lambda_function_with_position(model, function):
    f = lambdify(
        # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
        (
            model.position.get_list(),
            model.variables.get_list(),
            model.aux_variables.get_list(),
            model.parameters.get_list(),
        ), # The (),) comma is crutial!
        function,
        ufl_map
    )
    return f

def get_lambda_function_boundary(model, function):
    f = lambdify(
        # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
        (
            model.time,
            model.position.get_list(),
            model.distance,
            model.variables.get_list(),
            model.aux_variables.get_list(),
            model.parameters.get_list(),
            model.normal.get_list(),
        ), # The (),) comma is crutial!
        function,
        ufl_map
    )
    return f

# def get_ufl_boundary_functions(model):
#     boundary_function_matrix = model.boundary_conditions.get_boundary_function_matrix(*model.get_boundary_conditions_matrix_inputs())
#     n_bcs = boundary_function_matrix.shape[0]
#     bc_funcs = []
#     for i in range(n_bcs):
#         bc_funcs.append(
#             lambdify(
#                 # tuple(model.get_boundary_conditions_matrix_inputs_as_list()),
#                 (model.variables.get_list(),
#                  model.normal.get_list()
#                  ), # The (),) comma is crutial!
#                 boundary_function_matrix[i, :].T,
#                 trafo.ufl_map
#             )
#         )
#     return bc_funcs
    
@define(kw_only=True, slots=True, frozen=True)
class FenicsXRuntimeModel:
    name: str = field()
    n_variables: int = field()
    n_aux_variables: int = field()
    n_parameters: int = field()
    parameters = field()
    flux: Callable = field()
    flux_jacobian: Callable = field()
    source: Callable = field()
    source_jacobian: Callable = field()
    nonconservative_matrix: Callable = field()
    quasilinear_matrix: Callable = field()
    eigenvalues: Callable = field()
    source_implicit: Callable = field()
    residual: Callable = field()
    interpolate_3d: Callable = field()
    bcs: Callable = field()
    dimension: int = field()
    left_eigenvectors: Callable = field(default=None)
    right_eigenvectors: Callable = field(default=None)

    @classmethod
    def from_model(cls, domain, model):
        parameters = [dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(p)) for p in model.parameter_values]
        
        
        return cls(
            name=model.name,
            dimension=model.dimension,
            n_variables=model.n_variables,
            n_aux_variables=model.n_aux_variables,
            n_parameters=model.n_parameters,
            parameters=parameters,
            flux=get_lambda_function(model, Matrix.hstack(*model.flux())),
            flux_jacobian=get_lambda_function(model, Matrix.hstack(*model.flux_jacobian())),
            source=get_lambda_function(model, model.source()),
            source_jacobian=get_lambda_function(model, model.source_jacobian()),
            nonconservative_matrix=get_lambda_function(model, Matrix.hstack(*model.nonconservative_matrix())),
            quasilinear_matrix=get_lambda_function(model, Matrix.hstack(*model.quasilinear_matrix())),
            eigenvalues=get_lambda_function_with_normal(model, model.eigenvalues()),
            left_eigenvectors=get_lambda_function_with_normal(model, model.left_eigenvectors()),
            right_eigenvectors=get_lambda_function_with_normal(model, model.right_eigenvectors()),
            source_implicit=get_lambda_function(model, model.source_implicit()),
            residual=get_lambda_function(model, model.residual()),
            interpolate_3d=get_lambda_function_with_position(model, model.interpolate_3d()),
            bcs=get_lambda_function_boundary(model, model.boundary_conditions.get_boundary_function_matrix(*model.get_boundary_conditions_matrix_inputs()))
        )