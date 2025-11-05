import os
from sympy import MatrixSymbol, fraction, cancel, Matrix, symbols, radsimp, powsimp
import sympy as sp
from copy import deepcopy

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.sympy2c import create_module
from zoomy_core.transformation.helpers import regularize_denominator, substitute_sympy_attributes_with_symbol_matrix

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
        module_functions.append(self.create_function_interpolate('project_2d_to_3d', model.project_2d_to_3d(), n_dof, n_dof_qaux))
        module_functions.append(self.create_function_boundary('boundary_conditions', model.boundary_conditions.get_boundary_condition_function(model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, model.normal), n_dof, n_dof_qaux, dim))
        full = self.create_file_header(n_dof, n_dof_qaux, dim) + '\n\n' + '\n\n'.join(module_functions) + self.create_file_footer()
        return full
    

def write_code(model, settings):
    printer = AmrexPrinter(model)
    expr = printer.create_model(model)
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, settings.output.directory, ".amrex_interface")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "Model.h")
    with open(path, 'w+') as f:
        f.write(expr)
