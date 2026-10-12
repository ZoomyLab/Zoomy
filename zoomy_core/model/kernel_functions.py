"""Opaque SymPy functions (eigensystem, solve, pick, conditional, ...) that every backend
realises numerically. Each kernel must be registered in the backend's function table (NumPy,
JAX, C).
"""

import sympy as sp
import itertools
from zoomy_core.misc.misc import ZArray


class eigensystem(sp.Function):
    """Opaque full eigendecomposition ``[lambda, R, L=R^-1]`` of the row-major ``n*n`` matrix
    ``A_flat``, packed in one vector. Components are read through
    ``pick(eigensystem(*a), idx)``; used by the Roe scheme.

    Args:
        A_flat: the ``n*n`` matrix entries, row-major.
    """

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated


class spectral_radius(sp.Function):
    """Opaque estimate of ``rho(A)`` by ``k`` power-iteration steps on ``A^2`` (no
    eigendecomposition, runs on any device). Used as the wave-speed bound of the symbolic
    solvers.

    Args:
        k: number of power-iteration steps.
        safety: multiplicative safety factor on the estimate.
        A_flat: the ``n*n`` matrix entries, row-major.
    References:
        Golub & Van Loan 2013, https://doi.org/10.56021/9781421407944
    """

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated (opaque)


class eigenvalues(sp.Function):
    """Opaque eigenvalue-only spectrum of the row-major ``n*n`` matrix ``A_flat``, read through
    ``pick(eigenvalues(*a), idx)``. Cheaper companion of :class:`eigensystem` for wave-speed
    bounds when no closed-form spectrum exists.
    """

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated (opaque)


class solve(sp.Function):
    """Opaque per-cell linear solve ``A^-1 b``, called as ``solve(*A_flat, *b_flat)`` and read
    through ``pick(solve(*a), idx)``. Used by the point-implicit source treatment of the
    NumericalSystemModel.
    """

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated (opaque)


class pick(sp.Function):
    """Component ``idx`` of a packed opaque-kernel result (:class:`eigensystem`,
    :class:`eigenvalues`, :class:`solve`). Per-component backends restore ``K(idx, *a)`` through
    :func:`lower_pick_to_component`.
    """

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated (opaque)


# Names of the opaque kernels whose components are addressed positionally — the
# ones ``pick`` wraps and the per-component backends (C / UFL) unwrap.
_PER_COMPONENT_KERNELS = frozenset({"eigensystem", "eigenvalues", "solve"})


def lower_pick_to_component(expr):
    """Rewrite ``pick(K(*a), idx)`` into the per-component form ``K(idx, *a)`` required by the
    C/C++ and UFL printers.

    Returns:
        The expression (or array/matrix) with every ``pick`` of a per-component kernel swapped
        structurally.
    """
    def _rw(e):
        """Recursive rewrite over lists, arrays, matrices and expressions."""
        if isinstance(e, (list, tuple)):
            return type(e)(_rw(x) for x in e)
        if isinstance(e, sp.NDimArray):
            flat = [_rw(x) for x in sp.flatten(e)]
            rebuilt = sp.ImmutableDenseNDimArray(flat, e.shape)
            return ZArray(rebuilt) if isinstance(e, ZArray) else rebuilt
        if isinstance(e, sp.MatrixBase):
            return e.applyfunc(_rw)
        if not isinstance(e, sp.Basic):
            return e
        reps = {}
        for call in e.atoms(sp.Function):
            if call.func.__name__ != "pick":
                continue
            inner, idx = call.args
            if (isinstance(inner, sp.Function)
                    and inner.func.__name__ in _PER_COMPONENT_KERNELS):
                with sp.evaluate(False):
                    reps[call] = inner.func(idx, *inner.args)
        if not reps:
            return e
        with sp.evaluate(False):
            return e.xreplace(reps)
    return _rw(expr)


class compute_derivative(sp.Function):
    """Opaque non-local spatial derivative ``compute_derivative(field, *multi_index)``;
    ``(h, 1, 0)`` is d h / d x. The solver injects the backend's LSQ-gradient implementation; it
    is emitted by ``expose_aux`` into ``update_aux_variables``.
    """
    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        """Keep the call unevaluated."""
        return None  # always keep unevaluated (opaque)


class conditional(sp.Function):
    """Vector-aware differentiable ``conditional(c, t, f)``: a symbolic node for scalars, a
    ``ZArray`` of nodes for vector branches. Differentiation follows the active branch and
    ignores the jump at the condition.
    """

    nargs = 3
    is_commutative = (
        True  # <--- CRITICAL FIX: prevents PolynomialError in charpoly/eigenvals
    )

    def __new__(cls, condition, true_val, false_val, **kwargs):
        # 1. Helper to check for array-like inputs (excluding Symbols)
        """Build the scalar node, or broadcast over array-valued branches into a ``ZArray``."""
        def is_vec(x):
            """True for array-like (non-symbol) inputs."""
            return hasattr(x, "__getitem__") and not isinstance(
                x, (sp.Symbol, sp.Function)
            )

        is_array_t = is_vec(true_val)
        is_array_f = is_vec(false_val)

        # 2. Vector Case: Broadcast and return ZArray
        if is_array_t or is_array_f:
            # Flatten inputs
            t_flat = list(sp.flatten(true_val)) if is_array_t else [true_val]
            f_flat = list(sp.flatten(false_val)) if is_array_f else [false_val]

            # Determine shape
            if hasattr(true_val, "shape"):
                shape = true_val.shape
            elif hasattr(false_val, "shape"):
                shape = false_val.shape
            else:
                shape = (len(t_flat),)

            # Broadcast scalar '0' filler if lengths mismatch (standard python zip_longest behavior)
            result_list = []

            # Handling scalar vs vector broadcasting manually for safety
            max_len = max(len(t_flat), len(f_flat))
            if not is_array_t:
                t_flat = [true_val] * max_len
            if not is_array_f:
                f_flat = [false_val] * max_len

            for t, f in zip(t_flat, f_flat):
                # Recursively call __new__ (which will hit the Scalar Case below)
                result_list.append(cls(condition, t, f))

            # Return a ZArray containing the symbolic nodes
            return ZArray(result_list).reshape(*shape)

        # 3. Scalar Case: Create the actual Symbolic Node
        return super().__new__(cls, condition, true_val, false_val, **kwargs)

    def _eval_derivative(self, s):
        """Differentiate both branches, ignoring the jump at the condition."""
        condition, true_expr, false_expr = self.args
        return conditional(condition, true_expr.diff(s), false_expr.diff(s))


# ── The UserFunctions contract (REQ-168) ─────────────────────────────────────
# The opaque kernels every backend must SUPPLY (numpy `module` /
# `UserFunctions.H` / jax `userfunctions.py`).  A backend whose table is missing
# one of these has a silent hole — each backend ships a contract test asserting
# its table covers this set, so the next missing kernel is a RED test rather
# than a `NameError` at lambdify (python) or a link error mid-build (C++).
#
# ``conditional`` is DELIBERATELY EXCLUDED: it is printer-lowered to an inline
# ternary (``generic_c.py`` → ``((c) ? (t) : (f))``; numpy → ``np.where``), so
# it never reaches a backend UserFunctions table — a flat "all kernel classes"
# list would wrongly demand every C++ backend implement a function that must not
# exist (dmplex REQ-168 item 5).
REQUIRED_KERNELS = frozenset({
    "compute_derivative",
    "eigensystem",
    "eigenvalues",
    "solve",
})
