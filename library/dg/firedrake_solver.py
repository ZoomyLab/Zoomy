import firedrake as fd
import ufl
from library.python.fvm.solver_jax import Settings
from attrs import field
from library.python.misc.misc import Zstruct





class FiredrakeHyperbolicSolver:
    def __init__(self, CFL=0.45, time_end=0.1):
        settings: Zstruct = field(factory=lambda: Settings.default())
        self.CFL = CFL
        self.time_end = time_end
        IdentityMatrix = field(factory = lambda : ufl.as_tensor([[1, 0, 0],
                                                        [0, 1, 0],
                                                        [0, 0, 1]]))
        
    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, 'settings', defaults)

    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
        return fd.dot(0.5*(model.flux(Ql, Qauxl, parameters)+model.flux(Qr, Qauxr, parameters)), n) \
               - 0.5*self.max_abs_eigenvalue(model, Ql, Qauxl, n, mesh)*(Qr - Ql)

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0, 0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(ev[i, 0] > max_ev, ev[i, 0], max_ev)
        return max_ev

    def solve(self, mshfile, model):
        mesh = fd.Mesh(mshfile)
        x = fd.SpatialCoordinate(mesh)
        n = fd.FacetNormal(mesh)

        V = fd.VectorFunctionSpace(mesh, "DG", 0, dim=model.n_variables)
        Vaux = fd.VectorFunctionSpace(mesh, "DG", 0, dim=model.n_aux_variables)
        Qn = fd.Function(V)
        Qnp1 = fd.Function(V)
        Qaux = fd.Function(Vaux)

        t = fd.Constant(0.0)
        dt = fd.Constant(0.1)

        test_q = fd.TestFunction(V)
        trial_q = fd.TrialFunction(V)

        weak_form = fd.dot(test_q, (trial_q - Qn)/dt)*fd.dx
        weak_form += fd.dot(test_q('+') - test_q('-'),
            self.numerical_flux(model, Qn('+'), Qn('-'), Qaux('+'), Qaux('-'),
                                model.parameters, n('+'), mesh)) * fd.dS

        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        solver = fd.LinearVariationalSolver(problem,
            solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"})

        outfile = fd.File("sim_firedrake.pvd")
        outfile.write(Qn, time=float(t))

        while float(t) < self.time_end:
            solver.solve()
            Qn.assign(Qnp1)
            t.assign(float(t)+float(dt))
            outfile.write(Qn, time=float(t))
