from sympy import MatrixSymbol, fraction, cancel, Matrix

from zoomy_core.misc.misc import Zstruct

def regularize_denominator(expr, regularization_constant = 10**(-4), regularize = False):
    if not regularize:
        return expr
    def regularize(expr):
        (nom, den) = fraction(cancel(expr))
        return nom * den / (den*2 + regularization_constant)
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            expr[i,j] = regularize(expr[i,j])
    return expr

def substitute_sympy_attributes_with_symbol_matrix(expr: Matrix, attr: Zstruct, attr_matrix: MatrixSymbol):
    if expr is None:
        return None
    if type(attr) is Zstruct:
        assert attr.length() <= attr_matrix.shape[0]
        for i, k in enumerate(attr.get_list()):
            expr = Matrix(expr).subs(k, attr_matrix[i])
    else:
        expr = Matrix(expr).subs(attr, attr_matrix)
    return expr
