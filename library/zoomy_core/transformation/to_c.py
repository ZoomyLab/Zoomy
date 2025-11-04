import os
import re
import textwrap
import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter

class CPrinter(CXX11CodePrinter):
    """
    Convert SymPy expressions to C code.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables

        # Map variable names to Q[i], Qaux[i]
        self.map_Q = {k: f"Q[{i}]" for i, k in enumerate(model.variables.values())}
        self.map_Qaux = {k: f"Qaux[{i}]" for i, k in enumerate(model.aux_variables.values())}
        self.map_param = {k: str(float(model.parameter_values[i])) for i, k in enumerate(model.parameters.values())}
        
        self.map_normal = {k: f"n[{i}]" for i, k in enumerate(model.normal.values())}
        self.map_position = {k: f"X[{i}]" for i, k in enumerate(model.position.values())}

        self._std_regex = re.compile(r'std::([A-Za-z_]\w*)')

    # --- Symbol printing --------------------------------------------------
    def _print_Symbol(self, s):
        for m in [self.map_Q, self.map_Qaux, self.map_param, self.map_normal, self.map_position]:
            if s in m:
                return m[s]
        return super()._print_Symbol(s)

    # --- Pow printing -----------------------------------------------------
    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return self._print(base)
            if n < 0:
                return f"(1.0 / std::pow({self._print(base)}, {abs(n)}))"
            return f"std::pow({self._print(base)}, {n})"
        return f"std::pow({self._print(base)}, {self._print(exp)})"

    # --- Expression conversion --------------------------------------------
    def convert_expression_body(self, expr, target='res'):
        tmp_sym = sp.numbered_symbols('t')
        temps, simplified = sp.cse(expr, symbols=tmp_sym)
        lines = []
        cols = expr.shape[1]
        for lhs, rhs in temps:
            lines.append(f"double {self.doprint(lhs)} = {self.doprint(rhs)};")
        for i in range(expr.rows):
            for j in range(expr.cols):
                lines.append(f"{target}[{i * cols +j}] = {self.doprint(simplified[0][i, j])};")
        return "\n        ".join(lines)

    # --- Header / Footer --------------------------------------------------
    def create_file_header(self, n_dof_q, n_dof_qaux, dim, list_sorted_function_names):
        return textwrap.dedent(f"""\
        #pragma once

        static const int MODEL_n_dof_q    = {n_dof_q};
        static const int MODEL_n_dof_qaux = {n_dof_qaux};
        static const int MODEL_dimension  = {dim};
        static const int MODEL_n_boundary_tags = {len(list_sorted_function_names)};
        static const char* MODEL_map_boundary_tag_to_function_index[] = {{ {", ".join(f'"{item}"' for item in list_sorted_function_names)} }};
        """)

    def create_file_footer(self):
        return "\n"

    # --- Function generators ---------------------------------------------
    def create_function(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        if isinstance(expr, list):
            dim = len(expr)
            return [self.create_function(f"{name}_{d}", expr[i], n_dof_q, n_dof_qaux)
                    for i, d in enumerate(['x', 'y', 'z'][:dim])]

        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline void {name}(
    const double* Q,
    const double* Qaux,
    double* res
    )
{{
    {body}
}}
        """

    def create_function_normal(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        if isinstance(expr, list):
            return [self.create_function_normal(f"{name}_{d}", expr[i], n_dof_q, n_dof_qaux, dim)
                    for i, d in enumerate(['x', 'y', 'z'][:dim])]

        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline void {name}(
    const double* Q,
    const double* Qaux,
    const double* n,
    double* res)
{{
    {body}
}}
        """

    def create_function_interpolate(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline double* {name}(
    const double* Q,
    const double* Qaux,
    const double* X,
    double* res)
{{
    {body}
}}
        """

    def create_function_boundary(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline double* {name}(
    const double* Q,
    const double* Qaux,
    const double* n,
    const double* X,
    const double time,
    const double dX, 
    double* res)
{{
    {body}
}}
        """

    # --- Full model generation ------------------------------------------
    def create_model(self, model):
        n_dof = model.n_variables
        n_dof_qaux = model.n_aux_variables
        dim = model.dimension
        funcs = []
        funcs += self.create_function('flux', model.flux(), n_dof, n_dof_qaux)
        funcs += self.create_function('flux_jacobian', model.flux_jacobian(), n_dof, n_dof_qaux)
        funcs += self.create_function('nonconservative_matrix', model.nonconservative_matrix(), n_dof, n_dof_qaux)
        funcs += self.create_function('quasilinear_matrix', model.quasilinear_matrix(), n_dof, n_dof_qaux)
        funcs.append(self.create_function_normal('eigenvalues', model.eigenvalues(), n_dof, n_dof_qaux, dim))
        funcs.append(self.create_function('left_eigenvectors', model.left_eigenvectors(), n_dof, n_dof_qaux))
        funcs.append(self.create_function('right_eigenvectors', model.right_eigenvectors(), n_dof, n_dof_qaux))
        funcs.append(self.create_function('source', model.source(), n_dof, n_dof_qaux))
        funcs.append(self.create_function('residual', model.residual(), n_dof, n_dof_qaux))
        funcs.append(self.create_function('source_implicit', model.source_implicit(), n_dof, n_dof_qaux))
        funcs.append(self.create_function_interpolate('interpolate', model.project_2d_to_3d(), n_dof, n_dof_qaux))
        funcs.append(self.create_function_boundary(
            'boundary_conditions',
            model.boundary_conditions.get_boundary_condition_function(
                model.time, model.position, model.distance,
                model.variables, model.aux_variables, model.parameters, model.normal),
            n_dof, n_dof_qaux, dim))

        return self.create_file_header(n_dof, n_dof_qaux, dim, model.boundary_conditions.list_sorted_function_names) + "\n".join(funcs) + self.create_file_footer()


def write_code(model, settings):
    printer = CPrinter(model)
    code = printer.create_model(model)
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, settings.output.directory, ".c_interface")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "Model.H")
    with open(file_path, "w+") as f:
        f.write(code)
