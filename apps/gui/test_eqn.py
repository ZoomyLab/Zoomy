from sympy import *
x,y,z = symbols('x y z')
init_printing()
expr = Integral(Array([[x, y], [-y, x]]), x)
def get_model():
    return latex(expr)

