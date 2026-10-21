"""Module `zoomy_core.model.analysis`."""

from sympy import (
    Matrix,
    diff,
    exp,
    I,
    linear_eq_to_matrix,
    solve,
    Eq,
    zeros,
    simplify,
    nsimplify,
    latex,
    symbols,
    Function,
    together,
    Symbol,
)

from zoomy_core.misc.show import show


class _LatexBlock:
    """A display-math block that renders through whatever `display` the
    host provides (Jupyter, the GUI's Pyodide kernel) and prints as LaTeX
    source otherwise. Importing `IPython.display` for this made the module
    unimportable in the Pyodide kernel, which has `display` but no IPython."""

    __slots__ = ("tex",)

    def __init__(self, tex):
        self.tex = tex

    def _repr_latex_(self):
        return self.tex

    def _repr_markdown_(self):
        return self.tex

    def __repr__(self):
        return self.tex


class ModelAnalyser:
    """Linear analysis of a system model: linearisation about a base state,
    plane-wave ansatz, dispersion relation.

    Takes a `SystemModel` and
    nothing else — the analysis lives at the system level and does not
    know whether the system was derived or written down
    (`SystemModel.from_model(model)` is the way in)."""
    def __init__(self, model):
        """Initialize the instance."""
        from zoomy_core.systemmodel.system_model import SystemModel
        if not isinstance(model, SystemModel):
            raise TypeError(
                "ModelAnalyser takes a SystemModel; build one with "
                "SystemModel.from_model(model).")
        self.model = model
        self.t = model.time
        x, y, z = model.position_struct()
        self.x = x
        self.y = y
        self.z = z
        self.equations = None
        self.plane_wave_symbols = []

    def get_equations(self):
        """Get equations."""
        return self.equations

    def print_equations(self):
        """Print equations."""
        if not self.equations:
            print("No equations generated.")
            return
        latex_lines = " \\\\\n".join([f"& {latex(eq)}" for eq in self.equations])
        latex_block = r"$$\begin{align*}" + "\n" + latex_lines + r"\end{align*}$$"
        show(_LatexBlock(latex_block))

    def get_time_space(self):
        """Get time space."""
        return self.t, self.x, self.y, self.z

    def _get_omega_k(self):
        """Internal helper `_get_omega_k`."""
        omega, kx, ky, kz = symbols("omega k_x k_y k_z")
        return omega, kx, ky, kz

    def _get_exponential(self):
        """Internal helper `_get_exponential`."""
        omega, kx, ky, kz = self._get_omega_k()
        t, x, y, z = self.get_time_space()
        exponential = exp(I * (kx * x + ky * y + kz * z - omega * t))
        return exponential

    def get_eps(self):
        """Get eps."""
        eps = symbols("eps")
        return eps

    def create_functions_from_list(self, names):
        """Create functions from list."""
        t, x, y, z = self.get_time_space()
        return [Function(name)(t, x, y, z) for name in names]

    def delete_equations(self, indices):
        """Delete equations."""
        self.equations = [
            self.equations[i] for i in range(len(self.equations)) if i not in indices
        ]

    def insert_plane_wave_ansatz(self, functions_to_replace):
        """
        Apply the ansatz to ALL variables passed in the list.
        """
        exponential = self._get_exponential()
        f_bar_dict = {}
        self.plane_wave_symbols = []

        for f in functions_to_replace:
            f_name = str(f.func)
            f_bar = Symbol(r"\bar{" + f_name + "}")
            f_bar_dict[f] = f_bar * exponential
            self.plane_wave_symbols.append(f_bar)

        self.equations = [eq.xreplace(f_bar_dict).doit() for eq in self.equations]

    def solve_for_dispersion_relation(self):
        """Solve for dispersion relation."""
        assert self.equations is not None
        assert self.plane_wave_symbols, "No plane wave symbols defined."

        A, rhs = linear_eq_to_matrix(self.equations, self.plane_wave_symbols)

        if A.rows != A.cols:
            print(
                f"Warning: System is {A.rows}x{A.cols}. Determinant requires a square matrix."
            )
            return []

        omega = symbols("omega")
        # Berkowitz is generally faster for these types of symbolic matrices
        sol = solve(A.det(method="berkowitz"), omega)
        return sol

    def remove_exponential(self):
        """Remove exponential."""
        exponential = self._get_exponential()
        equations = self.equations
        equations = [
            simplify(Eq(eq.lhs / exponential, eq.rhs / exponential)) for eq in equations
        ]
        self.equations = equations

    def linearize_system(self, q, qaux, source=None, constraints=None):
        """
        Linearize the system given the state vectors q and qaux.

        Args:
            q: State vector (prognostic variables)
            qaux: Auxiliary state vector
            source: (Optional) The source term vector S. If None, S=0.
            constraints: (Optional) The constraint vector C. If None, C=empty.
        """
        model = self.model
        t, x, y, z = self.get_time_space()
        dim = model.dimension
        X = [x, y, z]

        Q = Matrix(list(model.state))
        Qaux = Matrix(list(model.aux_state))

        substitutions = {Q[i]: q[i] for i in range(len(q))}
        substitutions.update({Qaux[i]: qaux[i] for i in range(len(qaux))})

        # 1. Quasilinear Matrix (Flux Jacobian) — the system model's lazily
        #    computed (n_eq, n_state, n_dim) operator.
        A_raw = model.quasilinear_matrix
        A_matrices = []
        for d in range(dim):
            mat = Matrix(A_raw[:, :, d])  # Handle immutable arrays
            mat = mat.xreplace(substitutions)
            A_matrices.append(mat)

        # 2. Source Terms (Optional)
        if source is not None:
            S = Matrix(source)  # Handle immutable arrays
            S = S.xreplace(substitutions)
        else:
            S = zeros(len(q), 1)

        # 3. Constraints (Optional)
        if constraints is not None:
            C = Matrix(constraints)
            C = C.xreplace(substitutions)
            C = C.doit()
        else:
            C = zeros(0, 1)

        gradQ = Matrix(
            [diff(q[i], X[j]) for i in range(len(q)) for j in range(dim)]
        ).reshape(len(q), dim)

        AgradQ = A_matrices[0] * gradQ[:, 0]
        for d in range(1, dim):
            AgradQ += A_matrices[d] * gradQ[:, d]

        # 4. Assemble: Dynamics (lhs + flux - source) stacked on Constraints
        expr = list(Matrix.vstack((diff(q, t) + AgradQ - S), C))

        # Cleanup and Linearization
        for i in range(len(expr)):
            expr[i] = nsimplify(expr[i], rational=True)

        expr = Matrix(expr)
        eps = self.get_eps()
        res = expr.copy()

        for i, e in enumerate(expr):
            collected = e
            collected = collected.series(eps, 0, 2).removeO()
            order_1_term = collected.coeff(eps, 1)
            res[i] = order_1_term

        for r in range(res.shape[0]):
            denom = together(res[r]).as_numer_denom()[1]
            res[r] *= denom
            res[r] = simplify(res[r])

        linearized_system = [Eq((res[i]), 0) for i in range(res.shape[0])]

        self.equations = linearized_system
        return self.equations
    
    def solve_for_constraints(self, list_of_selected_equations, list_of_variables):
        """
        Manually solve specific equations for specific variables and substitute
        the results into the remaining system.

        Args:
            list_of_selected_equations: Indices of equations to solve (e.g., [3, 4, 7])
            list_of_variables: Variables to solve for (e.g., [p0, p1, w2])
        """
        equations = self.equations

        # 1. Solve the selected equations for the selected variables
        # Note: solve returns a dictionary {variable: expression}
        subset_eqs = [equations[i] for i in list_of_selected_equations]
        sol = solve(subset_eqs, list_of_variables)

        # 2. Substitute these solutions into ALL equations
        # .xreplace is generally faster/safer than .subs for exact structural matches
        equations = [eq.xreplace(sol).doit() for eq in equations]

        # 3. Delete the equations we just used
        # We keep only the equations whose indices were NOT in the selected list
        equations = [
            equations[i]
            for i in range(len(equations))
            if i not in list_of_selected_equations
        ]

        self.equations = equations
        return sol
