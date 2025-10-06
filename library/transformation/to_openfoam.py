import os
import re
import textwrap
import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter

class FoamPrinter(CXX11CodePrinter):
    """
    Convert SymPy expressions to OpenFOAM 12 compatible C++ code.
    - Q, Qaux: Foam::List<Foam::scalar>
    - Matrices: Foam::List<Foam::List<Foam::scalar>>
    - Vectors: Foam::vector (.x(), .y(), .z())
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables

        # Map variable names to Q[i], Qaux[i]
        self.map_Q = {k: f"Q[{i}]" for i, k in enumerate(model.variables.values())}
        self.map_Qaux = {k: f"Qaux[{i}]" for i, k in enumerate(model.aux_variables.values())}
        self.map_param = {k: str(float(model.parameter_values[i])) for i, k in enumerate(model.parameters.values())}

        # Map normal/position to n.x(), n.y(), n.z()
        self.map_normal = {k: ["n.x()", "n.y()", "n.z()"][i] for i, k in enumerate(model.normal.values())}
        self.map_position = {k: ["X.x()", "X.y()", "X.z()"][i] for i, k in enumerate(model.position.values())}

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
                return f"(1.0 / Foam::pow({self._print(base)}, {abs(n)}))"
            return f"Foam::pow({self._print(base)}, {n})"
        return f"Foam::pow({self._print(base)}, {self._print(exp)})"

    # --- Expression conversion --------------------------------------------
    def convert_expression_body(self, expr, target='res'):
        tmp_sym = sp.numbered_symbols('t')
        temps, simplified = sp.cse(expr, symbols=tmp_sym)
        lines = []
        for lhs, rhs in temps:
            lines.append(f"Foam::scalar {self.doprint(lhs)} = {self.doprint(rhs)};")
        for i in range(expr.rows):
            for j in range(expr.cols):
                lines.append(f"{target}[{i}][{j}] = {self.doprint(simplified[0][i, j])};")
        return "\n        ".join(lines)

    # --- Matrix factory ---------------------------------------------------
    def createMatrix(self, rows, cols):
        return f"Foam::List<Foam::List<Foam::scalar>>({rows}, Foam::List<Foam::scalar>({cols}, 0.0))"

    # --- Header / Footer --------------------------------------------------
    def create_file_header(self, n_dof_q, n_dof_qaux, dim):
        return textwrap.dedent(f"""\
        #pragma once
        #include "List.H"
        #include "vector.H"
        #include "scalar.H"

        namespace Model
        {{
        constexpr int n_dof_q    = {n_dof_q};
        constexpr int n_dof_qaux = {n_dof_qaux};
        constexpr int dimension  = {dim};
        """)

    def create_file_footer(self):
        return "\n} // namespace Model\n"

    # --- Function generators ---------------------------------------------
    def create_function(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        if isinstance(expr, list):
            dim = len(expr)
            return [self.create_function(f"{name}_{d}", expr[i], n_dof_q, n_dof_qaux)
                    for i, d in enumerate(['x', 'y', 'z'][:dim])]

        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline Foam::List<Foam::List<Foam::scalar>> {name}(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{{
    auto {target} = {self.createMatrix(rows, cols)};
    {body}
    return {target};
}}
        """

    def create_function_normal(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        if isinstance(expr, list):
            return [self.create_function_normal(f"{name}_{d}", expr[i], n_dof_q, n_dof_qaux, dim)
                    for i, d in enumerate(['x', 'y', 'z'][:dim])]

        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline Foam::List<Foam::List<Foam::scalar>> {name}(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{{
    auto {target} = {self.createMatrix(rows, cols)};
    {body}
    return {target};
}}
        """

    def create_function_interpolate(self, name, expr, n_dof_q, n_dof_qaux, target='res'):
        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline Foam::List<Foam::List<Foam::scalar>> {name}(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& X)
{{
    auto {target} = {self.createMatrix(rows, cols)};
    {body}
    return {target};
}}
        """

    def create_function_boundary(self, name, expr, n_dof_q, n_dof_qaux, dim, target='res'):
        rows, cols = expr.shape
        body = self.convert_expression_body(expr, target)
        return f"""
inline Foam::List<Foam::List<Foam::scalar>> {name}(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n,
    const Foam::vector& X,
    const Foam::scalar& time,
    const Foam::scalar& dX)
{{
    auto {target} = {self.createMatrix(rows, cols)};
    {body}
    return {target};
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
        funcs.append(self.create_function_interpolate('interpolate', model.interpolate_3d(), n_dof, n_dof_qaux))
        funcs.append(self.create_function_boundary(
            'boundary_conditions',
            model.boundary_conditions.get_boundary_function_matrix(
                model.time, model.position, model.distance,
                model.variables, model.aux_variables, model.parameters, model.normal),
            n_dof, n_dof_qaux, dim))

        return self.create_file_header(n_dof, n_dof_qaux, dim) + "\n".join(funcs) + self.create_file_footer()


def write_code(model, settings):
    printer = FoamPrinter(model)
    code = printer.create_model(model)
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, settings.output.directory, ".foam_interface")
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "Model.h")
    with open(file_path, "w+") as f:
        f.write(code)
