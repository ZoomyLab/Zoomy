
from sympy import symbols, Matrix, diff, exp, I, Rational
from zoomy_core.model.model import *
from zoomy_core.model.analysis import linear_stability_analysis

def test_linear_stability_analysis():
    
    model = GN()

    Q = Matrix(model.variables.get_list())
    Qaux = Matrix(model.aux_variables.get_list())
    
    x, y, z = symbols('x y z')
    t, eps = symbols('t eps')
    omega, kx, ky, kz = symbols('omega kx ky kz')

    h0, u0 = symbols('h0 u0')
    h1, u1 = symbols('h1 u1')
    exponential = exp(I * (kx * x + ky * y + kz * z - omega *t))

    h = h0 + h1 *eps* exponential
    u = u1 *eps* exponential
    D = h**3 * 1/3 * (diff(diff(u, x), t) + u * diff(u, x, 2) - (diff(u, x))**2)
    dD_dx = diff(D, x)

    sol = linear_stability_analysis(model, {Q[0]: h , Q[1]: u , Qaux[0]: dD_dx}, linear_system_variables=[h1, u1] )
    assert str(sol) ==  "[-sqrt(3)*kx*sqrt(g*h0/(h0**3*kx**2 + 3)), sqrt(3)*kx*sqrt(g*h0/(h0**3*kx**2 + 3))]"
    
if __name__ == "__main__":
    test_linear_stability_analysis()