"""Build + print the two solvers; shared by the gate and the participants."""
import sympy as sp
import symsolve as S
import solvers as V
import kernels_swe as K


class P:  # parameter symbols (user data names)
    g, cf, nu, cfl, dx, hb = sp.symbols("g cf nu cfl dx hb")


Q, t, T = sp.symbols("Q t T")
PARAMS = (P.g, P.cf, P.nu, P.cfl, P.dx, P.hb)
H_ARGS = (Q, t, T, *PARAMS)
C_ARGS = (Q, t, *PARAMS)


def printer(lang):
    return S.PyPrinter() if lang == "py" else S.CxxPrinter()


def hyperbolic_src(lang, mode="cse", **kw):
    return printer(lang).procedure(
        "hyperbolic", H_ARGS, lambda Q, t, T, *a: V.hyperbolic(Q, t, T, P, **kw),
        mode=mode, sources=[V.__file__])


def coupled_src(lang, side, **kw):
    return printer(lang).procedure(
        "hyperbolic_precice", C_ARGS,
        lambda Q, t, *a: V.hyperbolic_precice(Q, t, P, side, **kw))


def cpp_program(src, main_body, precice=False):
    inc = '#include "rt_precice.hpp"\n' if precice else '#include "rt.hpp"\n'
    return ("#include <cstdio>\n#include <cstdlib>\n#include <string>\n" + inc +
            "namespace zs {\n" + K.cxx_kernels() + "\n" + src + "\n}\n"
            "int main(int argc, char** v) {\n using namespace zs;\n" + main_body + "}\n")
