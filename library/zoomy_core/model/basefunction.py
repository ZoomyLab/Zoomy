
import sympy
import sympy as sp
from attrs import define, field
from sympy import lambdify

from zoomy_core.misc.misc import Zstruct

def listify(expr):
    if type(expr) is sp.Piecewise:
        return expr
    else:
        return list(expr)

def vectorize_constant_sympy_expressions(expr, Q, Qaux):
    """
    Replace entries in `expr` that are constant w.r.t. Q and Qaux
    by entry * ones_like(Q[0]) so NumPy/JAX vectorization works.
    Handles scalars, lists, sympy.Matrix, sympy.Array, and sympy.Piecewise.
    """
    symbol_list = set(Q.get_list() + Qaux.get_list())
    q0 = Q[0]
    ones_like = sp.Function("ones_like")  # symbolic placeholder
    zeros_like = sp.Function("zeros_like")  # symbolic placeholder

    # convert matrices to nested lists (Array handles lists better)
    if isinstance(expr, (sp.MatrixBase, sp.ImmutableDenseMatrix, sp.MutableDenseMatrix)):
        expr = expr.tolist()

    def vectorize_entry(entry):
        """Return entry multiplied by ones_like(q0) if it is constant."""
        # numeric zero
        if entry == 0:
            return zeros_like(q0)

        # numeric constant (int, float, Rational, pi, etc.)
        if getattr(entry, "is_number", False):
            return entry * ones_like(q0)

        # symbolic constant independent of Q and Qaux
        if hasattr(entry, "free_symbols") and entry.free_symbols.isdisjoint(symbol_list):
            return entry * ones_like(q0)

        # otherwise, depends on variables
        return entry

    def recurse(e):
        """Recursively handle Array, Matrix, Piecewise, list, or scalar."""
        # Handle lists (possibly nested)
        if isinstance(e, list):
            return [recurse(sub) for sub in e]

        # Handle Matrices
        if isinstance(e, sp.MatrixBase):
            return sp.Matrix([[recurse(sub) for sub in row] for row in e.tolist()])

        # Handle Arrays (any rank)
        if isinstance(e, sp.Array):
            return sp.Array([recurse(sub) for sub in e])

        # Handle Piecewise
        if isinstance(e, sp.Piecewise):
            # Recursively process all (expr, cond) pairs
            new_args = []
            for expr_i, cond_i in e.args:
                new_expr = recurse(expr_i)
                new_args.append((new_expr, cond_i))
            return sp.Piecewise(*new_args)

        # Scalar or atomic expression
        return vectorize_entry(e)

    # Recurse and then normalize into an N-dimensional array (if possible)
    result = recurse(expr)

    # Convert to Array if possible (this ensures uniform output type)
    if isinstance(result, list):
        try:
            return sp.Array(result)
        except Exception:
            # fall back if shapes inconsistent
            return result
    return result





@define(frozen=True, slots=True, kw_only=True)
class Function:
    """
    Generic (virtual) function implementation.
    """

    name: str = field(default="Function")
    args: Zstruct = field(default=Zstruct())
    definition = field(default=sympy.zeros(1, 1))

    def __call__(self):
        """Allow calling the instance to get its symbolic definition."""
        return self.definition

    def lambdify(self, modules=None):
        """Return a lambdified version of the function."""

        func = lambdify(
            self.args.get_list(),
            vectorize_constant_sympy_expressions(
                listify(self.definition), self.args.variables, self.args.aux_variables
            ),
            modules=modules,
        )
        return func
