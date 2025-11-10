from sympy import Matrix, diff, exp, I, linear_eq_to_matrix, solve , Eq, zeros, simplify, nsimplify, latex,  symbols, Function, together, Symbol
from IPython.display import display, Latex


class ModelAnalyser():
    def __init__(self, model):
        self.model = model
        self.t = model.time
        x, y, z = model.position
        self.x = x
        self.y = y
        self.z = z
        self.equations = None
        self.plane_wave_symbols = []
        

    def get_equations(self):
        return self.equations

    def print_equations(self):          
        latex_lines = " \\\\\n".join([f"& {latex(eq)}" for eq in self.equations])
        latex_block = r"$$\begin{align*}" + "\n" + latex_lines + r"\end{align*}$$"
        display(Latex(latex_block))
            
    def get_time_space(self):
        x, y, z = self.model.position
        t = self.model.time
        return t, x, y, z
    
    def _get_omega_k(self):
        omega, kx, ky, kz = symbols('omega k_x k_y k_z')
        return omega, kx, ky, kz
    
    def _get_exponential(self):
        omega, kx, ky, kz = self._get_omega_k()
        t, x, y, z = self.get_time_space()
        exponential = exp(I * (kx * x + ky * y + kz * z - omega * t))
        return exponential
    
    def get_eps(self):
        eps = symbols('eps')
        return eps
    
    def create_functions_from_list(self, names):
        t, x, y, z = self.get_time_space()
        return [Function(name)(t, x, y, z) for name in names]
    
    def delete_equations(self, indices):
        self.equations = [self.equations[i] for i in range(len(self.equations)) if i not in indices]


    def solve_for_constraints(self, list_of_selected_equations, list_of_variables):
        equations = self.equations
        sol = solve([equations[i] for i in list_of_selected_equations], list_of_variables)
        equations = [eq.xreplace(sol).doit() for eq in equations]
        # delete used equations from equation system
        equations = [equations[i] for i in range(len(equations)) if i not in list_of_selected_equations]
        self.equations = equations
        return sol

    def insert_plane_wave_ansatz(self, functions_to_replace):
        exponential = self._get_exponential()
        f_bar_dict = {}
        for f in functions_to_replace:
            # Create the base name (e.g., 'f0')
            f_name = str(f.func)  # Get the function name (e.g., 'f0')
            
            # Create a new symbol representing \bar{f0}
            f_bar = Symbol(r'\bar{' + f_name + '}')
            f_bar_dict[f] =  f_bar * exponential
            self.plane_wave_symbols.append(f_bar)
        self.equations =  [eq.xreplace(f_bar_dict).doit() for eq in self.equations ]
        
    def solve_for_dispersion_relation(self):
        assert self.equations is not None, "No equations available to solve for dispersion relation."
        assert self.plane_wave_symbols, "No plane wave symbols available to solve for dispersion relation. Use insert_plane_wave_ansatz first."
        A, rhs = linear_eq_to_matrix(self.equations, self.plane_wave_symbols)
        omega, kx, ky, kz = self._get_omega_k()
        sol = solve(A.det(), omega)
        return sol

    def remove_exponential(self):
        exponential = self._get_exponential()
        equations = self.equations
        equations = [simplify(Eq(eq.lhs / exponential, eq.rhs / exponential)) for eq in equations]
        self.equations = equations

    def linearize_system(self, q, qaux, constraints=None):
        model = self.model
        t, x, y, z = self.get_time_space()
        dim = model.dimension
        X = [x, y, z]
        
        Q = Matrix(model.variables.get_list())
        Qaux = Matrix(model.aux_variables.get_list())

        
        substitutions = {Q[i]: q[i] for i in range(len(q))}
        substitutions.update({Qaux[i]: qaux[i] for i in range(len(qaux))})
        

        A = model.quasilinear_matrix()
        S = model.residual()
        if constraints is not None:
            C = constraints
        else:
            C = zeros(0, 1)
            

        Q = Q.xreplace(substitutions)
        for d in range(dim):
            A[d] = A[d].xreplace(substitutions)
        S = S.xreplace(substitutions)
        C = C.xreplace(substitutions)
        
        C = C.doit()
            
            
        gradQ = Matrix([diff(q[i], X[j]) for i in range(len(q)) for j in range(dim)]).reshape(len(q), dim)
            
        AgradQ = A[0] * gradQ[:, 0]
        for d in range(1, dim):
            AgradQ += A[d] * gradQ[:, d]


        expr = list(Matrix.vstack((diff(q, t) + AgradQ - S) , C))
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
        
        linearized_system = [Eq((res[i]),0) for i in range(res.shape[0])]

        
        self.equations = linearized_system

