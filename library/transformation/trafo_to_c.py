import os
from sympy import MatrixSymbol, fraction, cancel, Matrix
from copy import deepcopy

from library.misc.misc import Zstruct
from library.model.sympy2c import create_module

def regularize_denominator(expr, regularization_constant = 10**(-4), regularize = False):
    if not regularize:
        return expr
    def regularize(expr):
        (nom, den) = fraction(cancel(expr))
        return nom * den / (den*2 + regularization_constant)
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            expr[i,j] = regularize(expr[i,j])
    return expr

def substitute_sympy_attributes_with_symbol_matrix(expr: Matrix, attr: Zstruct, attr_matrix: MatrixSymbol):
    if expr is None:
        return None
    assert attr.length() == attr_matrix.shape[0]
    for i, k in enumerate(attr.get_list()):
        expr = Matrix(expr).subs(k, attr_matrix[i])
    return expr



def transform_in_place(model, printer='jax'):
    runtime_pde = model._get_pde(printer=printer)
    runtime_bcs = model._get_boundary_conditions(printer=printer)

    return runtime_pde, runtime_bcs

def save_model_to_C(model, settings):
    _ = create_c_model_interface(model, settings)
    # _ = model.create_c_boundary_interface(
    #     path=os.path.join(settings.output.directory, "c_interface")
    #     )

## #TODO create c_boundary_interface similat to create_c_interface
## def create_c_boundary_interface(model, path='.tmp/'):
##     # define matrix symbols that will be used as substitutions for the currelty used symbols in the
##     # expressions
##     Q = MatrixSymbol('Q', model.n_variables, 1)
##     Q_ghost = MatrixSymbol('Qg', model.n_variables, 1)
##     Qaux = MatrixSymbol('Qaux', model.n_aux_variables, 1)
##     parameters = MatrixSymbol('parameters', model.n_parameters, 1)
##     normal = MatrixSymbol('normal', model.dimension, 1)
## 
## 
##     sympy_boundary_functions = deepcopy(model.boundary_conditions.boundary_functions)
##     #TODO create module
##     #TODO delete the code below
## 
##     # aggregate all expressions that are substituted and converted to C into the right data structure
##     list_matrix_symbols = [Q, Qaux, parameters, normal]
##     list_attributes = [model.variables, model.aux_variables, model.parameters, model.sympy_normal]
##     list_expressions = sympy_boundary_functions
##     list_expression_names = [f'boundary_condition_{i}' for i in range(len(sympy_boundary_functions))]
## 
##     # convert symbols to matrix symbols
##     for i in range(len(list_expressions)):
##         for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
##             list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
##             
## 
##     # aggregate data structure to be passed to the C converter module
##     expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]
## 
##     directory = os.path.join(path, model.name)
##     module_name = 'boundary_conditions'
## 
##     # create c module
##     create_module(module_name, expression_name_tuples, directory)

def create_c_model_interface(model, settings):
    # define matrix symbols that will be used as substitutions for the currelty used symbols in the
    # expressions
    Q = MatrixSymbol('Q', model.n_variables, 1)
    Qaux = MatrixSymbol('Qaux', model.n_aux_variables, 1)
    parameters = MatrixSymbol('parameters', model.n_parameters, 1)
    normal = MatrixSymbol('normal', model.dimension, 1)
    position = MatrixSymbol('position', model.position, 1)
    position = MatrixSymbol('position', model.position, 1)
    

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

    # aggregate all expressions that are substituted and converted to C into the right data structure
    list_matrix_symbols = [Q, Qaux, parameters]
    list_attributes = [model.variables, model.aux_variables, model.parameters]
    list_expressions = flux + flux_jacobian + nonconservative_matrix + quasilinear_matrix + [source, source_jacobian, source_implicit, residual, interpolate_3d]
    list_expression_names = ['flux_x', 'flux_y', 'flux_z', 
                             'flux_jacobian_x', 'flux_jacobian_y', 'flux_jacobian_z',
                             'nonconservative_matrix_x', 'nonconservative_matrix_y', 'nonconservative_matrix_z',
                             'quasilinear_matrix_x', 'quasilinear_matrix_y', 'quasilinear_matrix_z',
                             'source', 'source_jacobian', 'source_implicit', 'residual', 'interpolate_3d']
    list_matrix_symbols_incl_normal = [Q, Qaux, parameters, normal]
    list_attributes_incl_normal = [model.variables, model.aux_variables, model.parameters, model.normal]
    if eigenvalues is not None:
        list_expressions_incl_normal = [eigenvalues, left_eigenvectors, right_eigenvectors]
        list_expression_names_incl_normal = ['eigenvalues', 'left_eigenvectors', 'right_eigenvectors']
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

    path = os.path.join(settings.output.directory, '.c_interface')
    directory = os.path.join(path, model.name)
    module_name = 'model'

    # create c module
    create_module(module_name, expression_name_tuples, directory)

##def load_c_model(model, path='./.tmp'):
##    folder = os.path.join(path, model.name)
##    files_found = []
##    for file in os.listdir(folder):
##        if file.startswith("model") and file.endswith(".so"):
##            files_found.append(os.path.join(folder, file))
##
##    assert len(files_found) == 1
##    library_path = files_found[0]
##
##    c_model = cdll.LoadLibrary(library_path)
##
##    # the C module is constructed to use 3d functions, filled with dummies for dim<3. 
##    # therefore we extract the correct dimension dependent functions now
##    if model.dimension == 1:
##        flux = [c_model.flux_x]
##        flux_jacobian = [c_model.flux_jacobian_x]
##        nonconservative_matrix = [c_model.nonconservative_matrix_x]
##        quasilinear_matrix = [c_model.quasilinear_matrix_x]
##    elif model.dimension == 2:
##        flux = [c_model.flux_x, c_model.flux_y]
##        flux_jacobian = [c_model.flux_jacobian_x, c_model.flux_jacobian_y]
##        nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y]
##        quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y]
##    elif model.dimension == 3:
##        flux = [c_model.flux_x, c_model.flux_y, c_model.flux_z]
##        flux_jacobian = [c_model.flux_jggacobian_x, c_model.flux_jacobian_y, c_model.flux_jacobian_z]
##        nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y, c_model.nonconservative_matrix_z]
##        quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y, c_model.quasilinear_matrix_z]
##    else:
##        assert False
##
##    source = c_model.source
##    source_jacobian = c_model.source_jacobian
##    if model.settings.eigenvalue_mode == 'symbolic':
##        eigenvalues = c_model.eigenvalues
##        # eigenvalues = get_sympy_eigenvalues( c_model.eigenvalues)
##    elif model.settings.eigenvalue_mode == 'numerical':
##        eigenvalues = get_numerical_eigenvalues(model.dimension, model.n_variables, quasilinear_matrix)
##    else:
##        assert False
##
##    # define array prototypes
##    array_2d_double = npct.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS')
##    array_1d_double = npct.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
##
##    # define function prototype
##    for d in range(model.dimension):
##        flux[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
##        flux[d].restype = None
##        flux_jacobian[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
##        flux_jacobian[d].restype = None
##        nonconservative_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
##        nonconservative_matrix[d].restype = None
##        quasilinear_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
##        quasilinear_matrix[d].restype = None
##    source.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
##    source.restype = None
##    source_jacobian.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
##    source_jacobian.restype = None
##    eigenvalues.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double, array_1d_double]
##    eigenvalues.restype = None
##
##    
##
##    out = {'flux': flux, 'flux_jacobian': flux_jacobian, 'nonconservative_matrix':nonconservative_matrix, 'quasilinear_matrix':quasilinear_matrix, 'source': source, 'source_jacobian': source_jacobian, 'eigenvalues':eigenvalues}
##    return SimpleNamespace(**out)
