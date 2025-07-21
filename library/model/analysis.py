import os
from sympy import symbols, Matrix, diff, exp, I, linear_eq_to_matrix, pprint, solve , Eq, Float, Rational



def linear_stability_analysis(model, substitutions, linear_system_variables):
    """
    Perform linear stability analysis on the given model with specified substitutions.
    
    Args:
        model (Model): The model to analyze.
        substitutions (dict): Substitutions for the model parameters.
        
    Returns:
        dict: Results of the linear stability analysis.
    """
    
    x, y, z = symbols('x y z')
    t, eps = symbols('t eps')
    omega, kx, ky, kz = symbols('omega kx ky kz')
    
    Q = Matrix(model.variables.get_list())
    dim = model.dimension
    t, x, y, z = symbols('t x y z')
    X = [x, y, z]
    A = model.sympy_quasilinear_matrix 
    S = model.sympy_source_implicit
        
    for k, v in substitutions.items():
        Q = Q.subs(k, v)
        for d in range(dim):
            A[d] = A[d].subs(k, v)
        S = S.subs(k, v)
        
    
        
    gradQ = Matrix([diff(Q[i], X[j]) for i in range(len(Q)) for j in range(dim)]).reshape(len(Q), dim)
        
    AgradQ = A[0] * gradQ[:, 0]
    for d in range(1, dim):
        AgradQ += A[d] * gradQ[:, d]


    expr = (diff(Q, t) + AgradQ - S) 
    for i in range(len(expr)):
        expr[i] = expr[i].replace(lambda expr: isinstance(expr, Float), lambda f: Rational(f))
    
    
    res = expr.copy()
    for i, e in enumerate(expr):
        collected = e
        collected = collected.series(eps, 0, 2).removeO()
        order_1_term = collected.coeff(eps, 1)
        res[i] = order_1_term
    A, rhs = linear_eq_to_matrix(res, linear_system_variables)
    
    detA = A.det()
    sol = solve(Eq(detA, 0), omega)
    
    return sol
