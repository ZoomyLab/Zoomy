"""Module `zoomy_core.model.analysis_linear`."""

import sympy as sp


class LinearWaveAnalyser:
    """
    Lightweight 1D linear-wave analyser for models with symbolic flux/source.

    Workflow:
    1) Build PDE expressions: d_t(Q) + d_x(F(Q)) - S(Q, D) = 0
    2) Insert plane-wave ansatz for selected fields
    3) Resolve requested derivative symbols (e.g. d_txx_h) from ansatz
    4) Extract O(eps) system and solve for omega, then c = omega / k
    """

    def __init__(self, model):
        """Initialize the instance."""
        self.model = model
        self.t = model.time
        self.x = model.position[0]
        self.eps = sp.symbols("eps", real=True)
        self.k, self.omega = sp.symbols("k omega", real=True)
        self.I = sp.I

    def _exp_wave(self):
        """Internal helper `_exp_wave`."""
        return sp.exp(self.I * (self.k * self.x - self.omega * self.t))

    def _build_equations(self, q_expr_map, aux_expr_map):
        """Internal helper `_build_equations`."""
        Q = list(self.model.variables.values())
        F = self.model.flux()
        S = self.model.source()

        substitutions = {}
        substitutions.update(q_expr_map)
        substitutions.update(aux_expr_map)

        equations = []
        for i, q_i in enumerate(Q):
            q_expr = q_expr_map[q_i]
            # 1D flux component at column 0
            f_expr = sp.sympify(F[i, 0]).xreplace(substitutions)
            s_expr = sp.sympify(S[i]).xreplace(substitutions)
            eq = sp.diff(q_expr, self.t) + sp.diff(f_expr, self.x) - s_expr
            equations.append(sp.simplify(eq))
        return equations

    def _build_quasilinear_equations(self, q_expr_map, aux_expr_map):
        """Internal helper `_build_quasilinear_equations`."""
        Q = list(self.model.variables.values())
        A = sp.Matrix(self.model.quasilinear_matrix()[:, :, 0])
        S = sp.Matrix(self.model.source())

        substitutions = {}
        substitutions.update(q_expr_map)
        substitutions.update(aux_expr_map)

        q_expr_vec = sp.Matrix([q_expr_map[q] for q in Q])
        qx_expr_vec = sp.Matrix([sp.diff(expr, self.x) for expr in q_expr_vec])

        A_sub = A.xreplace(substitutions)
        S_sub = S.xreplace(substitutions)
        eq_vec = sp.diff(q_expr_vec, self.t) + A_sub * qx_expr_vec - S_sub
        return [sp.simplify(eq_vec[i]) for i in range(eq_vec.shape[0])]

    def _linearize_eps(self, equations):
        """Internal helper `_linearize_eps`."""
        out = []
        for eq in equations:
            lin = sp.expand(eq).series(self.eps, 0, 2).removeO().coeff(self.eps, 1)
            out.append(sp.simplify(lin))
        return out

    def _build_aux_expr_map(self, q_expr_map):
        """Internal helper `_build_aux_expr_map`."""
        aux_expr_map = {}
        specs = getattr(self.model, "derivative_specs", [])
        if not specs:
            return aux_expr_map

        Q_syms = self.model.variables
        for spec in specs:
            key = spec.key
            i_aux = self.model.derivative_key_to_index[key]
            aux_sym = self.model.aux_variables[i_aux]
            q_sym = Q_syms[spec.field]
            expr = q_expr_map[q_sym]
            for axis in spec.axes:
                if axis == "t":
                    expr = sp.diff(expr, self.t)
                elif axis == "x":
                    expr = sp.diff(expr, self.x)
                else:
                    raise NotImplementedError(
                        f"Axis '{axis}' not supported in this analyser."
                    )
            aux_expr_map[aux_sym] = sp.simplify(expr)
        return aux_expr_map

    def linearize_from_quasilinear(
        self,
        base_state,
        perturbation_functions,
        assumptions=None,
        print_steps=True,
    ):
        """
        Generic Step 2:
          - Insert q = q0 + eps*q1(t,x) into quasilinear PDE form
          - Expand to O(eps)
        """
        Q_syms = self.model.variables

        q_expr_map = {}
        for name, q_sym in zip(Q_syms.keys(), Q_syms.values()):
            q0 = base_state[name]
            q1 = perturbation_functions[name]
            q_expr_map[q_sym] = q0 + self.eps * q1

        aux_expr_map = self._build_aux_expr_map(q_expr_map)
        equations_after_substitution = self._build_quasilinear_equations(q_expr_map, aux_expr_map)
        linearized_system = self._linearize_eps(equations_after_substitution)
        if assumptions:
            equations_after_substitution = [
                sp.simplify(eq.xreplace(assumptions)) for eq in equations_after_substitution
            ]
            linearized_system = [
                sp.simplify(eq.xreplace(assumptions)) for eq in linearized_system
            ]

        if print_steps:
            print("=" * 80)
            print("Step 2: Linearized system from quasilinear form")
            for i, eq in enumerate(equations_after_substitution):
                print(f"eq_sub[{i}] =", eq)
            print("-" * 80)
            for i, eq in enumerate(linearized_system):
                print(f"lin[{i}] = 0  with  lin[{i}] =", sp.simplify(eq))
            print("=" * 80)

        return linearized_system

    def solve_phase_velocity_from_linearized(
        self,
        linearized_system,
        perturbation_functions,
        assumptions=None,
        print_steps=True,
    ):
        """
        Generic Step 3:
          - Insert q1 = qhat * exp(i(kx - wt)) into already-linearized equations
          - Solve det(A)=0 for omega, return c=omega/k
        """
        E = self._exp_wave()
        amp_symbols = {}
        replacements = {}
        unknowns = []

        for name, q1 in perturbation_functions.items():
            amp = sp.Symbol(f"{name}_hat")
            amp_symbols[name] = amp
            replacements[q1] = amp * E
            unknowns.append(amp)

        equations_pw = [sp.simplify(eq.xreplace(replacements).doit()) for eq in linearized_system]
        if assumptions:
            equations_pw = [sp.simplify(eq.xreplace(assumptions)) for eq in equations_pw]
        equations_pw = [sp.simplify(eq / E) for eq in equations_pw]

        A, _ = sp.linear_eq_to_matrix(equations_pw, unknowns)
        det_eq = sp.simplify(sp.factor(A.det(method="berkowitz")))
        if assumptions:
            det_eq = sp.simplify(det_eq.xreplace(assumptions))
        omega_solutions = sp.solve(sp.Eq(det_eq, 0), self.omega)
        c_solutions = [sp.simplify(sol / self.k) for sol in omega_solutions]

        if print_steps:
            print("=" * 80)
            print("Step 3: Plane-wave system and phase velocity")
            for i, eq in enumerate(equations_pw):
                print(f"pw[{i}] = 0  with  pw[{i}] =", eq)
            print("System matrix A:")
            print(A)
            print("det(A) = 0 with det(A) =", det_eq)
            print("omega solutions:", omega_solutions)
            print("phase velocity solutions:", c_solutions)
            print("=" * 80)

        return {
            "plane_wave_equations": equations_pw,
            "matrix": A,
            "determinant": det_eq,
            "omega_solutions": omega_solutions,
            "phase_velocity_solutions": c_solutions,
            "amplitudes": amp_symbols,
        }

    def analyse_phase_velocity(
        self,
        base_state,
        amplitude_symbols,
        field_to_amplitude,
        print_steps=True,
    ):
        """
        Args:
            base_state: dict like {"h": h0, "hu": hu0}
            amplitude_symbols: list of unknown amplitudes, e.g. [h1, m1]
            field_to_amplitude: dict mapping variable name -> amplitude symbol
                                e.g. {"h": h1, "hu": m1}
        """
        E = self._exp_wave()
        Q_syms = self.model.variables

        q_expr_map = {}
        for name, q_sym in zip(Q_syms.keys(), Q_syms.values()):
            q0 = base_state[name]
            amp = field_to_amplitude[name]
            q_expr_map[q_sym] = q0 + self.eps * amp * E

        aux_expr_map = self._build_aux_expr_map(q_expr_map)

        equations_after_ansatz = self._build_equations(q_expr_map, aux_expr_map)
        linear_system = self._linearize_eps(equations_after_ansatz)
        linear_system = [sp.simplify(eq / E) for eq in linear_system]

        A, _ = sp.linear_eq_to_matrix(linear_system, amplitude_symbols)
        det_eq = sp.simplify(sp.factor(A.det(method="berkowitz")))
        omega_solutions = sp.solve(sp.Eq(det_eq, 0), self.omega)
        c_solutions = [sp.simplify(sol / self.k) for sol in omega_solutions]

        if print_steps:
            print("=" * 80)
            print("Step 1: Equations after ansatz + derivative substitution")
            for i, eq in enumerate(equations_after_ansatz):
                print(f"eq[{i}] =", eq)
            print("-" * 80)
            print("Step 2: Linear O(eps) system to solve")
            for i, eq in enumerate(linear_system):
                print(f"lin[{i}] = 0  with  lin[{i}] =", eq)
            print("System matrix A:")
            print(A)
            print("det(A) = 0 with det(A) =", det_eq)
            print("-" * 80)
            print("Step 3: Phase velocity c = omega / k")
            print("omega solutions:", omega_solutions)
            print("phase velocity solutions:", c_solutions)
            print("=" * 80)

        return {
            "equations_after_ansatz": equations_after_ansatz,
            "linear_system": linear_system,
            "matrix": A,
            "determinant": det_eq,
            "omega_solutions": omega_solutions,
            "phase_velocity_solutions": c_solutions,
        }
