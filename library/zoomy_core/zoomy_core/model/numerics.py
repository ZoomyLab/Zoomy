"""Symbolic numerics-control wrappers — applied at runtime by the solver.

These let the model author annotate PDE terms with a *requested numerical
treatment* (gradient limiter, reconstruction scheme, ...) without
exposing the user to the discretisation layer.  The wrappers are inert
sympy Functions that the operator-form pipeline recognises and routes
into `aux_registry` entries with extra metadata fields.

The OpenFOAM analogue: `fvSchemes` exposes a scheme dictionary that
the user fills in per term (`div(phi,U)`, `grad(p)`, `laplacian(nu,U)`
each get their own scheme).  Our equivalent: the model author writes the
PDE in operator-form using these wrappers, and the runtime resolves
each one against the configured discretisation.

Currently exposed
-----------------

`limit(expr, scheme_symbol)`
    Wrap a gradient/derivative atom with a runtime TVD limiter.  The
    second argument is a `sp.Symbol` whose name names the scheme
    (`"minmod"`, `"venkatakrishnan"`, `"barth_jespersen"`).  At
    runtime, `compute_derivatives` first computes the unlimited LSQ
    gradient, then applies the named limiter per cell, and substitutes
    the limited value into the source expression.

Usage example
-------------



```
    from zoomy_core.model.numerics import limit
    import sympy as sp

    h = sp.Symbol("h")
    x = sp.Symbol("x")
    minmod = sp.Symbol("minmod")
    S[1, 0] = - 2 * P_1 * limit(sp.Derivative(b, x), minmod)
```
"""
from __future__ import annotations
import sympy as sp


class limit(sp.Function):
    """Symbolic gradient-limiter wrapper.

    Signature:

    ```
        limit(expr, scheme)
    ```

    where `expr` is the derivative atom to be limited at runtime
    (typically a `sp.Derivative` of a state/aux field), and `scheme`
    is a `sp.Symbol` whose name selects the limiter
    (`"minmod"`, `"venkatakrishnan"`, `"barth_jespersen"`).

    The operator-form pipeline picks these up in `expose_aux_atoms`
    and substitutes a fresh aux Symbol named `{target}_{axes}__{scheme}`
    (e.g. `b_x__minmod`).  The chain solver's aux refresh recomputes
    the LSQ gradient and applies the named limiter before the source
    expression is evaluated.

    Inert symbolically — `.doit()` does NOT strip the wrapper (the
    runtime needs to see it).
    """
    nargs = 2

    @property
    def inner(self):
        """The wrapped derivative atom."""
        return self.args[0]

    @property
    def scheme(self):
        """Name of the limiter scheme (string)."""
        return str(self.args[1])

    def doit(self, **hints):
        # Stay opaque under .doit() — the runtime layer needs to see
        # the limit() wrapper to install the limiter aux entry.
        return self

    def _latex(self, printer):
        inner_tex = printer._print(self.args[0])
        scheme_tex = printer._print(self.args[1])
        return rf"\operatorname{{limit}}_{{{scheme_tex}}}\!\left({inner_tex}\right)"
