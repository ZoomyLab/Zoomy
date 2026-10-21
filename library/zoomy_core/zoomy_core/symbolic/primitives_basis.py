"""Basis-projection primitives: the only place we evaluate
`∫ kernel(ζ̂) dζ̂` for ζ̂ ∈ [0, 1] with kernel containing
opaque `phi_*` calls.

Decomposes the legacy `ProjectBasisIntegrals` (ins_generator.py:2188)
3-pass operation into separable primitives:

* `canonicalize_phi_derivative_subs` — rewrite
  `Subs(Derivative(phi(ξ), ξ), ξ, val) → Derivative(phi(val), val)`.
  Pure `xreplace`; no math change, just normalisation of the
  chain-rule artifact sympy emits when basis arguments substitute.
* `project_basis_integrand` — for each
  `Integral(kernel, (ζ̂, 0, 1))` whose kernel contains a `phi_*`
  call, look up via `BasisIntegralCache` and substitute the
  polynomial result.

The legacy operation also bundled
`zoomy_core.symbolic.primitives_canonical.split_integral_over_add`
between Pass 1 and Pass 3, plus an end-of-pipeline `Subs.doit()`
walk (ins_generator.py:2319).  The redesigned recipe asks the caller
to invoke those primitives explicitly.
"""

from __future__ import annotations

import sympy as sp
from sympy import Add, Derivative, Integral, Mul, S, Subs

__all__ = [
    "canonicalize_phi_derivative_subs",
    "project_basis_integrand",
    "has_phi_call",
]


def has_phi_call(e):
    """True iff `e` contains any `phi_*` function call."""
    if (isinstance(e, sp.Function)
            and getattr(e.func, "__name__", "").startswith("phi_")):
        return True
    if e.args:
        return any(has_phi_call(a) for a in e.args)
    return False


def canonicalize_phi_derivative_subs(expr):
    """`Subs(Derivative(phi_k(ξ), ξ), ξ, val) → Derivative(phi_k(val), val)`.

    The chain-rule artefact sympy creates when a `phi_k(arg)` is
    substituted into a `Derivative` context.  Recanonicalising as a
    plain `Derivative(phi_k(val), val)` makes the pattern visible to
    downstream basis-cache lookups.

    Lifted verbatim from `ProjectBasisIntegrals._canon`
    (ins_generator.py:2236).  Pure `xreplace` — no math change.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, Subs):
            inner = e.args[0]
            subs_vars = e.args[1]
            subs_vals = e.args[2]
            if (isinstance(inner, Derivative)
                    and isinstance(inner.args[0], sp.Function)
                    and getattr(inner.args[0].func, "__name__", "")
                    .startswith("phi_")
                    and len(subs_vars) == 1
                    and len(subs_vals) == 1
                    and inner.args[0].args[0] == subs_vars[0]):
                fn = inner.args[0].func
                arg = subs_vals[0]
                # ``arg`` may be a Mul (e.g. ``ζ̂·h + b``) which sympy
                # can't differentiate — but only when the caller asks
                # to ``.doit()`` this Derivative.  Constructing it here
                # is fine; sympy holds it.
                return Derivative(fn(arg), arg)
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


def project_basis_integrand(expr, basis_cache):
    """Look up every `Integral(kernel, (ζ̂, 0, 1))` whose kernel
    contains a `phi_*` call in the given `basis_cache` and
    substitute the polynomial result.

    Single-pass: walks the expression once, replacing each matching
    Integral in place.  Outer factors are pulled out structurally:
    integrand factors that don't reference the integration variable
    become an outer coefficient; the rest is the kernel handed to the
    cache.

    Lifted from the Pass-3 `_map` of `ProjectBasisIntegrals`
    (ins_generator.py:2282), with the surrounding fixpoint loop
    (lines 2339-2344) removed — the caller is expected to compose
    this primitive with `canonicalize_phi_derivative_subs` and
    `zoomy_core.symbolic.primitives_canonical.un_subs` for any
    boundary-Subs resolution.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0])
            limits = e.args[1]
            if not (hasattr(limits, "__len__") and len(limits) == 3):
                if integrand is not e.args[0]:
                    return Integral(integrand, *e.args[1:])
                return e
            var, lo, hi = limits
            # Split integrand into var-independent const × var-dependent kernel
            if isinstance(integrand, Mul):
                consts, kern_parts = [], []
                for f in integrand.args:
                    (kern_parts if f.has(var) else consts).append(f)
                const = Mul(*consts) if consts else S.One
                kernel = Mul(*kern_parts) if kern_parts else S.One
            elif integrand.has(var):
                const, kernel = S.One, integrand
            else:
                # var-free integrand
                return integrand * (hi - lo)
            if has_phi_call(kernel):
                value = basis_cache.integrate(kernel, var, lo, hi)
                return const * value
            if integrand is not e.args[0]:
                return Integral(integrand, *e.args[1:])
            return e
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)
