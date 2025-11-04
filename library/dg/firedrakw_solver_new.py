import firedrake as fd
import ufl
import numpy as np
from mpi4py import MPI

from library.zoomy_core.fvm.solver_jax import Settings
from attrs import field
from library.zoomy_core.misc.misc import Zstruct


class FiredrakeHyperbolicSolver:
    def __init__(self, CFL=0.45, time_end=0.1):
        settings: Zstruct = field(factory=lambda: Settings.default())
        self.CFL = CFL
        self.time_end = time_end
        IdentityMatrix = field(
            factory=lambda: ufl.as_tensor(
                [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
        )

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, "settings", defaults)

    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, mesh):
        return fd.dot(
            0.5
            * (model.flux(Ql, Qauxl, parameters) + model.flux(Qr, Qauxr, parameters)),
            n,
        ) - 0.5 * fd.max_value(
            self.max_abs_eigenvalue(model, Ql, Qauxl, n, mesh),
            self.max_abs_eigenvalue(model, Qr, Qauxr, n, mesh),
        ) * self.IdentityMatrix * (Qr - Ql)

    def max_abs_eigenvalue(self, model, Q, Qaux, n, mesh):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0, 0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(ev[i, 0] > max_ev, ev[i, 0], max_ev)
        return max_ev

    def compute_dt(self, Q, Qaux, n, ref_dx, CFL=0.45):
        mesh = Q.function_space().mesh()
        max_abs_ev = self.max_abs_eigenvalue(Q, Qaux, n, mesh)
        lam_max = np.max(max_abs_ev.dat.data_ro)
        lam_max = mesh.comm.allreduce(lam_max, op=MPI.MAX)
        return float(CFL * ref_dx / (lam_max + 1e-8))

    def write_state(q, out, time=0.0, names=None):
        mesh = q.function_space().mesh()
        V_scalar = fd.FunctionSpace(mesh, "DG", 0)
        dim = q.function_space().value_size
        subfuns = [
            fd.project(q[i], V_scalar, name=names[i] if names else f"Q{i}")
            for i in range(dim)
        ]

    # out.write(*subfuns, time=time)

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
        # dt = self.compute_dt(self, Qn, Qaux, n, ref_dx=mesh.hmin(), CFL=self.CFL)

        test_q = fd.TestFunction(V)
        trial_q = fd.TrialFunction(V)

        weak_form = fd.dot(test_q, (trial_q - Qn) / dt) * fd.dx
        weak_form += (
            fd.dot(
                test_q("+") - test_q("-"),
                self.numerical_flux(
                    model,
                    Qn("+"),
                    Qn("-"),
                    Qaux("+"),
                    Qaux("-"),
                    model.parameters,
                    n("+"),
                    mesh,
                ),
            )
            * fd.dS
        )

        a = fd.lhs(weak_form)
        L = fd.rhs(weak_form)
        problem = fd.LinearVariationalProblem(a, L, Qnp1)
        solver = fd.LinearVariationalSolver(
            problem, solver_parameters={"ksp_type": "bcgs", "pc_type": "jacobi"}
        )

        out = fd.VTKFile("output.pvd")
        self.write_state(Qn, out, time=t, names=["b", "h", "hu", "hv"])

        # outfile = fd.File("sim_firedrake.pvd")
        # outfile.write(Qn, time=float(t))

        while float(t) < self.time_end:
            # dt = self.compute_dt(
            #     self, Qn, Qaux, n, ref_dx=mesh.hmin(), CFL=self.CFL
            # )
            solver.solve()
            Qn.assign(Qnp1)
            t.assign(float(t) + float(dt))
            # outfile.write(Qn, time=float(t))
            self.write_state(Qn, out, time=t, names=["b", "h", "hu", "hv"])
