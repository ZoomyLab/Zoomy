import sympy as sp
from sympy import Matrix

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.analysis_linear import LinearWaveAnalyser
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel


class ClassicalGreenNaghdi1D(StructuredDerivativeModel):
    """
    1D nonlinear GN-like model in (h, u) form:
      h_t + (h u)_x = 0
      u_t + (u^2/2 + g h)_x = (h^2/3) * (u_txx + u*u_xxx - u_x*u_xx)
    """

    dimension = 1
    variables = ["h", "u"]
    parameters = {"g": (9.81, "positive")}
    auto_requested_derivatives = True

    def flux(self):
        h = self.Q.h
        u = self.Q.u
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = h * u
        F[1, 0] = sp.Rational(1, 2) * u**2 + g * h
        return ZArray(F)

    def source(self):
        h = self.Q.h
        u = self.Q.u
        u_txx = self.D.dtxx(u)
        u_x = self.D.dx(u)
        u_xx = self.D.dxx(u)
        u_xxx = self.D.diff(u, ("x", "x", "x"))
        S = ZArray.zeros(self.n_variables)
        S[1] = (h**2 * sp.Rational(1, 3)) * (u_txx + u * u_xxx - u_x * u_xx)
        return S


def run():
    model = ClassicalGreenNaghdi1D()
    model.print_model_functions(function_names=["flux", "source"])

    analyser = LinearWaveAnalyser(model)
    t = model.time
    x = model.position[0]
    h1 = sp.Function("h1")(t, x)
    u1 = sp.Function("u1")(t, x)

    h0, u0 = sp.symbols("h0 u0", real=True)
    linearized = analyser.linearize_from_quasilinear(
        base_state={"h": h0, "u": u0},
        perturbation_functions={"h": h1, "u": u1},
        print_steps=True,
    )

    result = analyser.solve_phase_velocity_from_linearized(
        linearized_system=linearized,
        perturbation_functions={"h": h1, "u": u1},
        print_steps=True,
    )

    c_rest = [sp.simplify(c.subs({u0: 0})) for c in result["phase_velocity_solutions"]]
    print("phase velocity for u0=0:", c_rest)
    return result


if __name__ == "__main__":
    run()
