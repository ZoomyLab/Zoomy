"""`canonicalise(expr)` — the structural-only normalisation pass.

Replaces the legacy `_simplify_preserve_integrals`
(ins_generator.py:4409), including its 6-iteration fixpoint loop, with
a single fixed-step pipeline of mathematically-trivial structural
rewrites that cannot alter equality.

Steps (in order):

1. `alpha_rename` — depth-keyed canonical bound-variable names so
   alpha-equivalent integrals match structurally.
2. `sp.expand` (Mul-over-Add only, kwargs-restricted; Integrals
   protected as Dummies during expand to keep their integrands
   intact).
3. `drop_zero_integrand` — `∫0 dv = 0`.
4. `kill_zero_length_integral` — `∫_a^a dv = 0` (structural
   `a == b` only — no `sp.simplify(lo - hi)`).
5. `zero_derivative_of_free_symbol` — `∂_v sym = 0` for free
   symbols distinct from `v`.
6. `drop_zero_derivative_inner` — `∂_v 0 = 0`.
7. `merge_integrals_over_add` — combine sibling integrals with
   matching `(limits, deriv-wrapper)`; required for sibling
   cancellations.
8. `split_integral_over_add` — rest state: one logical equation
   term per Integral.

Sympy calls allowed: `sp.Add` (additive group axioms — combine like
terms, drop zeros), `sp.Mul`/`Pow` constructors,
`sp.expand(_, mul=True, multinomial=False, …)` (Mul-over-Add only),
`xreplace`.

Sympy calls forbidden: `sp.simplify`, `sp.cancel`, `sp.together`,
`sp.factor`, `sp.collect`, `sp.powsimp`, `.doit()`,
`sp.diff`, `sp.integrate`.

The pass converges in one application — no fixpoint loop is needed
because every step is a contraction on a well-founded ordering
(number of Integral atoms, depth of nested integrands, count of zero
Add-summands).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic.primitives_canonical import (
    alpha_rename,
    drop_zero_derivative_inner,
    drop_zero_integrand,
    kill_zero_length_integral,
    merge_integrals_over_add,
    protect_integrals,
    split_integral_over_add,
    zero_derivative_of_free_symbol,
)

__all__ = ["canonicalise"]


def canonicalise(expr):
    """Normalise `expr` via the fixed structural pipeline.

    Pure: `canonicalise(canonicalise(x)) == canonicalise(x)`.
    Equality-preserving: `canonicalise(x - y) == 0` iff `x` and
    `y` are structurally equal up to the canonicalisation rules
    (alpha-renaming of bound variables, Mul-over-Add distribution,
    Integral linearity, zero-collapse).
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # 1. Alpha-rename bound dummies (depth-keyed canonical names).
    expr = alpha_rename(expr)

    # 2. Mul-over-Add distribution (Integrals protected to keep
    #    integrands intact — split is a separate primitive).
    protected, restore_map = protect_integrals(expr)
    expanded = sp.expand(
        protected,
        mul=True,
        multinomial=False,
        power_base=False,
        power_exp=False,
        log=False,
        complex=False,
        trig=False,
        basic=False,
    )
    expr = expanded.xreplace(restore_map)

    # 3-6. Trivial-zero collapses (each is structural pattern matching).
    expr = drop_zero_integrand(expr)
    expr = kill_zero_length_integral(expr)
    expr = zero_derivative_of_free_symbol(expr)
    expr = drop_zero_derivative_inner(expr)

    # 7. Merge sibling integrals (linearity-of-integral, with the
    #    var-independent-coefficient guard).  Required so that
    #    structurally-cancelling siblings actually cancel under
    #    sympy's Add canonicalisation.
    expr = merge_integrals_over_add(expr)

    # 8. Split integrals: rest state for the per-term view.  After
    #    merge collapsed cancellations, splitting back exposes one
    #    Integral per logical term so the user can navigate
    #    `.terms[i]` cleanly.
    expr = split_integral_over_add(expr)

    return expr
