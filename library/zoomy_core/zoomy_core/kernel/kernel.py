"""Kernel — SymbolicRegistrar for backend-agnostic helper functions.

The Kernel holds basefunctions (safe_denominator, conditional, etc.)
that model and numerics expressions reference symbolically.  Each backend
(numpy, jax, C, UFL) provides concrete implementations during code
transformation, using the same NumpyRuntimeSymbolic / JaxRuntimeSymbolic
machinery that compiles Model and Numerics functions.

After code transformation the user gets three compiled artifacts:
    1. Model   — physics (flux, source, eigenvalues)
    2. Numerics — Riemann solver (numerical flux, fluctuations)
    3. Kernel   — helper functions shared by both

Usage:

```
    kernel = Kernel(model)
    runtime_kernel = NumpyRuntimeSymbolic(kernel)
    # merge kernel functions into the module dict used by model/numerics
```
"""

from __future__ import annotations

import sympy as sp
import param

from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import SymbolicRegistrar


class Kernel(param.Parameterized, SymbolicRegistrar):
    """Backend-agnostic registry of numerical helper functions.

    Mirrors the Model/Numerics pattern: each function is a `Function`
    basefunction registered via `register_symbolic_function`.  The
    symbolic definitions serve as blueprints; actual implementations
    are provided by the backend during compilation.
    """

    def __init__(self, model=None, **params):
        super().__init__(**params)
        self.functions = Zstruct()
        self.call = Zstruct()

        # Build convenient name → symbol maps from model
        if model is not None:
            self._var_map = {str(k): v for k, v in model.variables.items()}
            # Kernel needs symbolic parameters (Symbols), not numeric values.
            # New naming: ``model.parameters`` = Symbols,
            # ``model.parameter_values`` = floats.
            self._param_map = {
                str(k): v for k, v in model.parameters.items()
            }
        else:
            self._var_map = {}
            self._param_map = {}

        # Kernel function arguments: lightweight symbols, not full Q/Qaux/p
        self._x = sp.Symbol("x", real=True)
        self._c = sp.Symbol("c")  # condition
        self._t = sp.Symbol("t_branch", real=True)  # true branch
        self._f = sp.Symbol("f_branch", real=True)  # false branch
        self._eps = self._param_map.get("eps", sp.Symbol("eps", positive=True))

        self._initialize_functions()

    # ── helpers for model parameter access by name ─────────────────────

    def var(self, name: str) -> sp.Symbol:
        """Get a model variable symbol by name."""
        return self._var_map[name]

    def par(self, name: str) -> sp.Symbol:
        """Get a model parameter symbol by name."""
        return self._param_map[name]

    # ── function registration ──────────────────────────────────────────

    def _initialize_functions(self):
        """Register all kernel functions as basefunctions."""
        x = self._x
        c, t, f = self._c, self._t, self._f

        # Each function gets a minimal signature: just the args it needs.
        # The backend fills in the actual implementation.

        self.register_symbolic_function(
            "safe_denominator",
            self.safe_denominator,
            Zstruct(x=x, eps=self._eps),
        )
        self.register_symbolic_function(
            "conditional",
            self.conditional,
            Zstruct(condition=c, true_val=t, false_val=f),
        )

    # ── symbolic definitions (blueprints) ──────────────────────────────
    # Each returns a symbolic expression.  Backends override with
    # language-specific implementations during compilation.

    def safe_denominator(self):
        """safe_denominator(x) → x regularized to avoid 1/0."""
        return self._x + self._eps

    def conditional(self):
        """conditional(c, t, f) → t if c else f."""
        return sp.Piecewise((self._t, self._c), (self._f, True))

    # ── singularity detection + targeted regularization ───────────────

    def positive_variables(self):
        """Return list of model variable symbols declared positive."""
        return [sym for sym in self._var_map.values()
                if getattr(sym, 'is_positive', False)]

    @staticmethod
    def find_singular_denominators(expr, positive_vars):
        """Find Pow(v, negative_exp) subexpressions for positive variables.

        Walks the expression tree and collects every `Pow(v, e)` where
        `v` is a positive variable and `e < 0`.  These are the terms
        that produce NaN when `v → 0` (e.g. at dry cells).

        This is the detection function — swap this out to try different
        strategies (e.g. numerical probing, limit analysis).

        Returns
        -------
        list of (Pow_subexpr, variable) pairs
        """
        if not isinstance(expr, sp.Basic):
            return []
        hits = []
        for sub in sp.preorder_traversal(expr):
            if (isinstance(sub, sp.Pow)
                    and sub.base in positive_vars
                    and sub.exp.is_negative):
                hits.append((sub, sub.base))
        return hits

    def regularize(self, obj):
        """Regularize function definitions in-place (targeted).

        Works on any object with a `functions` Zstruct (Model, Numerics).
        For each function definition, finds `Pow(v, -n)` where `v` is a
        positive variable and replaces `v → v + eps` in that denominator only.
        Other occurrences of `v` (e.g. in `g*h`) are untouched.
        """
        eps = self._eps
        positive_vars = self.positive_variables()
        if not positive_vars:
            return

        for fn_name, fn_obj in obj.functions.items():
            expr = fn_obj.definition
            if expr is None:
                continue
            fn_obj.definition = self._regularize_expr(expr, positive_vars, eps)

    @staticmethod
    def _regularize_expr(expr, positive_vars, eps):
        """Replace Pow(v, -n) → Pow(v+eps, -n) for positive vars."""

        # ZArray / NDimArray — process element-wise, return same type
        if isinstance(expr, (ZArray, sp.tensor.array.NDimArray)):
            flat = [Kernel._regularize_expr(e, positive_vars, eps)
                    for e in sp.flatten(expr)]
            if isinstance(expr, ZArray):
                return ZArray(flat, expr.shape)
            return sp.Array(flat).reshape(*expr.shape)

        if isinstance(expr, (list, tuple)):
            return type(expr)(
                Kernel._regularize_expr(e, positive_vars, eps) for e in expr)

        if not isinstance(expr, sp.Basic):
            return expr

        # Targeted: only replace v → v+eps inside Pow(v, negative_exponent)
        for v in positive_vars:
            expr = expr.replace(
                lambda sub: (isinstance(sub, sp.Pow)
                             and sub.base == v
                             and sub.exp.is_negative),
                lambda sub: sp.Pow(v + eps, sub.exp)
            )
        return expr
