import os
from sympy import MatrixSymbol, fraction, cancel, Matrix, symbols, radsimp, powsimp
import sympy as sp
from copy import deepcopy

from library.misc.misc import Zstruct
from library.model.sympy2c import create_module
from library.transformation.helpers import regularize_denominator, substitute_sympy_attributes_with_symbol_matrix

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
import re
import textwrap


class AmrexPrinter(CXX11CodePrinter):
    """
    After the normal C++ printer has done its job, replace every
    'std::foo(' with 'amrex::Math::foo('  – except if foo is listed
    in 'custom_map'.  No other overrides are necessary.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_Q = {k: f"Q({i})" for i, k in enumerate(model.variables.values())}
        self.map_Qaux = {k: f"Qaux({i})" for i, k in enumerate(model.aux_variables.values())}
        self.map_param = {k: str(float(model.parameter_values[i])) for i, k in enumerate(model.parameters.values())}
        self.map_normal = {k:  f"normal({i})" for i, k in enumerate(model.normal.values())}
        self.map_position = {k:  f"X({i})" for i, k in enumerate(model.position.values())}


        self._custom_map = set({})
        # names that should *keep* their std:: prefix

        # pre-compile regex  std::something(
        self._std_regex = re.compile(r'std::([A-Za-z_]\w*)')
        
    def _print_Symbol(self, s):
        for map in [self.map_Q, self.map_Qaux, self.map_param, self.map_normal, self.map_position]:
            if s in map:
                return map[s]
        return super()._print_Symbol(s)
    
    def _print_Pow(self, expr):
        """
        Print a SymPy Power.

        * integer exponent  -> amrex::Math::powi<EXP>(base)
        * otherwise         -> amrex::Math::pow (run-time exponent)
        """
        base, exp = expr.as_base_exp()

        # integer exponent ------------------------------------------------
        if exp.is_Integer:
            n = int(exp)

            # 0, 1 and negative exponents inlined
            if n == 0:
                return "1.0"
            if n == 1:
                return self._print(base)
            if n < 0:
                # negative integer: 1 / powi<-n>(base)
                return (f"(1.0 / amrex::Math::powi<{abs(n)}>("
                        f"{self._print(base)}))")

            # positive integer
            return f"amrex::Math::powi<{n}>({self._print(base)})"

        # non-integer exponent -------------------------------------------
        return (f"std::pow("
                f"{self._print(base)}, {self._print(exp)})")

    # the only method we override
    def doprint(self, expr, **settings):
        code = super().doprint(expr, **settings)

        # callback that the regex will call for every match
        def _repl(match):
            fname = match.group(1)
            if fname in self._custom_map:
                return self._custom_map[fname]
            else:
                return f'std::{fname}'

        # apply the replacement to the whole code string
        return self._std_regex.sub(_repl, code) 

    def convert_expression_body(self, expr, target='res'):

        tmp_sym   = sp.numbered_symbols('t') 
        temps, simplified = sp.cse(expr, symbols=tmp_sym)  
        lines = []
        for lhs, rhs in temps:
            lines.append(f"amrex::Real {self.doprint(lhs)} = {self.doprint(rhs)};")

        for i in range(expr.rows):
            for j in range(expr.cols):
                lines.append(f"{target}({i},{j}) = {self.doprint(simplified[0][i, j])};")

        body = '\n        '.join(lines)
        return body
    
    def createSmallMatrix(self, rows, cols):
        return f"amrex::SmallMatrix<amrex::Real,{rows},{cols}>"
    
    def create_file_header(self, n_dof_q, n_dof_qaux, dim):
        header = textwrap.dedent(f"""
        #pragma once
        #include <AMReX_Array4.H>
        #include <AMReX_Vector.H>
        
        class Model {{
        public:
            static constexpr int n_dof_q    = {n_dof_q};
            static constexpr int n_dof_qaux = {n_dof_qaux};
            static constexpr int dimension  = {dim};
        """)
        return header
    
    def create_file_footer(self):
        return  """
};
                """
    
    def create_function(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        if type(expr) is list:
            dim = len(expr)
            return [self.create_function(f"{name}_{dir}", expr[i], n_dof_q, n_dof_qaux) for i, dir in enumerate(['x', 'y', 'z'][:dim])]
        res_shape = expr.shape
        body = self.convert_expression_body((expr), target=target)
        text = f"""
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static {self.createSmallMatrix(*res_shape)}
    {name} ( {self.createSmallMatrix(n_dof_q, 1)} const& Q,
    {self.createSmallMatrix(n_dof_qaux, 1)} const& Qaux) noexcept
    {{
        auto {target} = {self.createSmallMatrix(*res_shape)}{{}};
        {body}
        return {target};
    }}
        """
        return text

    def create_function_normal(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        if type(expr) is list:
            dim = len(expr)
            return [self.create_function_normal(f"{name}_{dir}", expr[i], n_dof_q, n_dof_qaux, dim) for i, dir in enumerate(['x', 'y', 'z'][:dim])]
        res_shape = expr.shape
        body = self.convert_expression_body(expr, target=target)
        text = f"""
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static {self.createSmallMatrix(*res_shape)}
    {name} ( {self.createSmallMatrix(n_dof_q, 1)} const& Q,
    {self.createSmallMatrix(n_dof_qaux, 1)} const& Qaux,
    {self.createSmallMatrix(dim, 1)} const& normal) noexcept
    {{
        auto {target} = {self.createSmallMatrix(*res_shape)}{{}};
        {body}
        return {target};

    }}
        """
        return text
    
    def create_function_interpolate(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        res_shape = expr.shape
        body = self.convert_expression_body(expr, target=target)
        text = f"""
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static {self.createSmallMatrix(*res_shape)}
    {name} ( {self.createSmallMatrix(n_dof_q, 1)} const& Q,
    {self.createSmallMatrix(n_dof_qaux, 1)} const& Qaux,
    {self.createSmallMatrix(3, 1)} const& X) noexcept
    {{
        auto {target} = {self.createSmallMatrix(*res_shape)}{{}};
        {body}
        return {target};
    }}

        """
        return text


    def create_function_boundary(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        res_shape = expr.shape
        body = self.convert_expression_body(expr, target=target)
        text = f"""
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static {self.createSmallMatrix(*res_shape)}
    {name} ( {self.createSmallMatrix(n_dof_q, 1)} const& Q,
    {self.createSmallMatrix(n_dof_qaux, 1)} const& Qaux,
    {self.createSmallMatrix(dim, 1)} const& normal, 
    {self.createSmallMatrix(3, 1)} const& position,
    amrex::Real const& time,
    amrex::Real const& dX) noexcept
    {{
        auto {target} = {self.createSmallMatrix(*res_shape)}{{}};
        {body}
        return {target};

    }}
        """
        return text
    
    def create_model(self, model):
        n_dof = model.n_variables
        n_dof_qaux = model.n_aux_variables
        dim =  model.dimension
        module_functions = []
        module_functions += self.create_function('flux', model.flux(), n_dof, n_dof_qaux)
        module_functions += self.create_function('flux_jacobian', model.flux_jacobian(), n_dof, n_dof_qaux)
        module_functions += self.create_function('nonconservative_matrix', model.nonconservative_matrix(), n_dof, n_dof_qaux)
        module_functions += self.create_function('quasilinear_matrix', model.quasilinear_matrix(), n_dof, n_dof_qaux)
        module_functions.append(self.create_function_normal('eigenvalues', model.eigenvalues(), n_dof, n_dof_qaux, dim))
        module_functions.append(self.create_function('left_eigenvectors', model.left_eigenvectors(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function('right_eigenvectors', model.right_eigenvectors(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function('source', model.source(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function('residual', model.residual(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function('source_implicit', model.source_implicit(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function_interpolate('interpolate_3d', model.interpolate_3d(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function_boundary('boundary_conditions', model.boundary_conditions.get_boundary_function_matrix(model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, model.normal), n_dof, n_dof_qaux, dim))
        full = self.create_file_header(n_dof, n_dof_qaux, dim) + '\n\n' + '\n\n'.join(module_functions) + self.create_file_footer()
        return full
    

def to_c(model, settings):
    printer = AmrexPrinter(model)
    expr = printer.create_model(model)
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, settings.output.directory, ".c_interface")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "model.h")
    with open(path, 'w+') as f:
        f.write(expr)

# def to_c(model, settings):
#     # define matrix symbols that will be used as substitutions for the currelty used symbols in the
#     # expressions
#     Q = MatrixSymbol('Q', model.n_variables, 1)
#     Qaux = MatrixSymbol('Qaux', model.n_aux_variables, 1)
#     parameters = MatrixSymbol('parameters', model.n_parameters, 1)
#     normal = MatrixSymbol('normal', 3, 1)
#     position = MatrixSymbol('position', 3, 1)
#     time, distance = symbols(['time', 'distance'])
    

#     # deepcopy to not mess up the sympy expressions of the class - in case I want to call the function a second time
#     flux = deepcopy(model.flux())
#     flux_jacobian = deepcopy(model.flux_jacobian())
#     source = deepcopy(model.source())
#     source_jacobian = deepcopy(regularize_denominator(model.source_jacobian()))
#     nonconservative_matrix = deepcopy(model.nonconservative_matrix())
#     quasilinear_matrix = deepcopy(model.quasilinear_matrix())
#     eigenvalues = deepcopy(model.eigenvalues())
#     left_eigenvectors = deepcopy(model.left_eigenvectors())
#     right_eigenvectors = deepcopy(model.right_eigenvectors())
#     source_implicit = deepcopy(model.source_implicit())
#     residual = deepcopy(model.residual())
#     interpolate_3d = deepcopy(model.interpolate_3d())
#     boundary_conditions = deepcopy(model.boundary_conditions.get_boundary_function_matrix(model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, normal))

#     # make all dimension dependent functions 3d to simplify the C part of the interface
#     if model.dimension == 1:
#         flux = [flux[0], flux[0], flux[0]]
#         flux_jacobian = [flux_jacobian[0], flux_jacobian[0], flux_jacobian[0]]
#         nonconservative_matrix = [nonconservative_matrix[0], nonconservative_matrix[0], nonconservative_matrix[0]]
#         quasilinear_matrix = [quasilinear_matrix[0], quasilinear_matrix[0], quasilinear_matrix[0]]
#     elif model.dimension == 2:
#         flux = [flux[0], flux[1], flux[0]]
#         flux_jacobian = [flux_jacobian[0], flux_jacobian[1], flux_jacobian[0]]
#         nonconservative_matrix = [nonconservative_matrix[0], nonconservative_matrix[1], nonconservative_matrix[0]]
#         quasilinear_matrix = [quasilinear_matrix[0], quasilinear_matrix[1], quasilinear_matrix[0]]
#     elif model.dimension == 3:
#         pass
#     else:
#         assert False

#     # aggregate all expressions that are substituted and converted to C into the right data structure
#     list_matrix_symbols = [Q, Qaux, parameters]
#     list_attributes = [model.variables, model.aux_variables, model.parameters]
#     list_expressions = flux + flux_jacobian + nonconservative_matrix + quasilinear_matrix + [source, source_jacobian, source_implicit, residual, interpolate_3d]
#     list_expression_names = ['flux_x', 'flux_y', 'flux_z', 
#                              'flux_jacobian_x', 'flux_jacobian_y', 'flux_jacobian_z',
#                              'nonconservative_matrix_x', 'nonconservative_matrix_y', 'nonconservative_matrix_z',
#                              'quasilinear_matrix_x', 'quasilinear_matrix_y', 'quasilinear_matrix_z',
#                              'source', 'source_jacobian', 'source_implicit', 'residual']
    
#     list_matrix_symbols_incl_normal = [Q, Qaux, parameters, normal]
#     list_attributes_incl_normal = [model.variables, model.aux_variables, model.parameters, model.normal]
#     if eigenvalues is not None:
#         list_expressions_incl_normal = [eigenvalues, left_eigenvectors, right_eigenvectors]
#         list_expression_names_incl_normal = ['eigenvalues', 'left_eigenvectors', 'right_eigenvectors']
#     else:
#         list_expressions_incl_normal = []
#         list_expression_names_incl_normal = []
    
#     list_matrix_symbols_incl_position = [Q, Qaux, parameters, position]
#     list_attributes_incl_position = [model.variables, model.aux_variables, model.parameters, model.position]
#     list_expressions_incl_position = [interpolate_3d]
#     list_expression_names_incl_position = ['interpolate_3d']
    
#     list_matrix_symbols_bc = [time, position, distance, Q, Qaux, parameters, normal]
#     list_attributes_bc = [model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, model.normal]
#     list_expressions_bc = [boundary_conditions]
#     list_expression_names_bc = ['boundary_conditions']


#     # convert symbols to matrix symbols
#     for i in range(len(list_expressions)):
#         for attr, matrix_symbol in zip(list_attributes, list_matrix_symbols):
#             list_expressions[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions[i], attr, matrix_symbol)
#     for i in range(len(list_expressions_incl_normal)):
#         for attr, matrix_symbol in zip(list_attributes_incl_normal, list_matrix_symbols_incl_normal):
#             list_expressions_incl_normal[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_incl_normal[i], attr, matrix_symbol)
#     for i in range(len(list_expressions_incl_position)):
#         for attr, matrix_symbol in zip(list_attributes_incl_position, list_matrix_symbols_incl_position):
#             list_expressions_incl_position[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_incl_position[i], attr, matrix_symbol)
#     for i in range(len(list_expressions_bc)):
#         for attr, matrix_symbol in zip(list_attributes_bc, list_matrix_symbols_bc):
#             list_expressions_bc[i] = substitute_sympy_attributes_with_symbol_matrix(list_expressions_bc[i], attr, matrix_symbol)    
            

#     # aggregate data structure to be passed to the C converter module
#     expression_name_tuples = [(expr_name, expr, [Q, Qaux, parameters]) for (expr_name, expr) in zip(list_expression_names, list_expressions)]
#     expression_name_tuples += [(expr_name, expr, [Q, Qaux, parameters, normal]) for (expr_name, expr) in zip(list_expression_names_incl_normal, list_expressions_incl_normal)]
#     expression_name_tuples += [(expr_name, expr, [Q, Qaux, parameters, position]) for (expr_name, expr) in zip(list_expression_names_incl_position, list_expressions_incl_position)]
#     expression_name_tuples += [(expr_name, expr, list_matrix_symbols_bc) for (expr_name, expr) in zip(list_expression_names_bc, list_expressions_bc)]
    
#     main_dir = os.getenv("ZOOMY_DIR")
#     path = os.path.join(settings.output.directory, '.c_interface')
#     directory = os.path.join(main_dir, os.path.join(path, model.name))
#     module_name = 'model'

#     # create c module
#     create_module(module_name, expression_name_tuples, directory)

# # def load_c_model(model, path='./.tmp'):
# #    folder = os.path.join(path, model.name)
# #    files_found = []
# #    for file in os.listdir(folder):
# #        if file.startswith("model") and file.endswith(".so"):
# #            files_found.append(os.path.join(folder, file))

# #    assert len(files_found) == 1
# #    library_path = files_found[0]

# #    c_model = cdll.LoadLibrary(library_path)

# #    # the C module is constructed to use 3d functions, filled with dummies for dim<3. 
# #    # therefore we extract the correct dimension dependent functions now
# #    if model.dimension == 1:
# #        flux = [c_model.flux_x]
# #        flux_jacobian = [c_model.flux_jacobian_x]
# #        nonconservative_matrix = [c_model.nonconservative_matrix_x]
# #        quasilinear_matrix = [c_model.quasilinear_matrix_x]
# #    elif model.dimension == 2:
# #        flux = [c_model.flux_x, c_model.flux_y]
# #        flux_jacobian = [c_model.flux_jacobian_x, c_model.flux_jacobian_y]
# #        nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y]
# #        quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y]
# #    elif model.dimension == 3:
# #        flux = [c_model.flux_x, c_model.flux_y, c_model.flux_z]
# #        flux_jacobian = [c_model.flux_jggacobian_x, c_model.flux_jacobian_y, c_model.flux_jacobian_z]
# #        nonconservative_matrix = [c_model.nonconservative_matrix_x, c_model.nonconservative_matrix_y, c_model.nonconservative_matrix_z]
# #        quasilinear_matrix = [c_model.quasilinear_matrix_x, c_model.quasilinear_matrix_y, c_model.quasilinear_matrix_z]
# #    else:
# #        assert False

# #    source = c_model.source
# #    source_jacobian = c_model.source_jacobian
# #    if model.settings.eigenvalue_mode == 'symbolic':
# #        eigenvalues = c_model.eigenvalues
# #        # eigenvalues = get_sympy_eigenvalues( c_model.eigenvalues)
# #    elif model.settings.eigenvalue_mode == 'numerical':
# #        eigenvalues = get_numerical_eigenvalues(model.dimension, model.n_variables, quasilinear_matrix)
# #    else:
# #        assert False

# #    # define array prototypes
# #    array_2d_double = npct.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS')
# #    array_1d_double = npct.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

# #    # define function prototype
# #    for d in range(model.dimension):
# #        flux[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
# #        flux[d].restype = None
# #        flux_jacobian[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
# #        flux_jacobian[d].restype = None
# #        nonconservative_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
# #        nonconservative_matrix[d].restype = None
# #        quasilinear_matrix[d].argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
# #        quasilinear_matrix[d].restype = None
# #    source.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double]
# #    source.restype = None
# #    source_jacobian.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_2d_double]
# #    source_jacobian.restype = None
# #    eigenvalues.argtypes = [array_1d_double, array_1d_double, array_1d_double, array_1d_double, array_1d_double]
# #    eigenvalues.restype = None

   

# #    out = {'flux': flux, 'flux_jacobian': flux_jacobian, 'nonconservative_matrix':nonconservative_matrix, 'quasilinear_matrix':quasilinear_matrix, 'source': source, 'source_jacobian': source_jacobian, 'eigenvalues':eigenvalues}
# #    return SimpleNamespace(**out)
