"""Principled symbolic-derivation primitive layer for zoomy_core.

This package replaces the semi-automatic, partially sympy-driven
pipeline in `zoomy_core.model.models.ins_generator` with a closed
set of well-defined mathematical primitives.  Every primitive is
ONE textbook calculus rule that fires only when explicitly invoked.
Sympy is used for arithmetic on `Add`/`Mul`/`Pow` and for
purely-structural `xreplace`; sympy is *never* used to fire chain
rule, Leibniz, IBP, common-denominator, factor, or `.doit()`.

Public surface (Phase 0):

* `D`, `Int`, `Sub` — held-form sympy constructors
  (always `evaluate=False`).
* `PrimitiveDoesNotMatch` — exception raised when a primitive's
  preconditions fail.
* `AutoEvalGuard` — context manager that bans sympy
  auto-evaluation calls inside its block (used in tests + the slim
  walkthrough verifier).

Primitives are added incrementally during Phase 0:

* `primitives_canonical` — purely-structural rewriters
  (`AlphaRename`, `SplitIntegralOverAdd`,
  `MergeIntegralsOverAdd`, `DistributeMulOverAdd`,
  `ZeroDerivativeOfFreeSymbol`, `UnSubs`).
* `primitives_calculus` — calculus rules (`ProductRuleForward`,
  `ProductRuleInverse`, `ChainRule`, `Leibniz`,
  `FundamentalTheorem`, `IntegrationByParts`,
  `PolynomialIntegrate`, `DistributeDerivativeOverAdd`,
  `PullScalarOutOfDerivative`, `PushScalarIntoDerivative`).
* `primitives_change_of_variable` — `AffineChangeOfVariable`.
* `primitives_basis` — `CanonicalizePhiDerivativeSubs`,
  `ProjectBasisIntegrand`.
* `primitives_substitution` — `Subst`, `FunctionExpand`,
  `SubsAtPoint`, `SolveFor`.
* `canonicalise` — the fixed, finite-step `canonicalise` pass that
  replaces the legacy `_simplify_preserve_integrals` 6-iteration
  fixpoint loop.
"""

from zoomy_core.symbolic.errors import PrimitiveDoesNotMatch
from zoomy_core.symbolic.sp_safe import D, Int, Sub, held_function
from zoomy_core.symbolic.auto_eval_guard import AutoEvalGuard, AutoEvalForbidden
from zoomy_core.symbolic.canonicalise import canonicalise

# Primitive groups — re-export the named functions so callers can
# write ``from zoomy_core.symbolic import alpha_rename`` etc.
from zoomy_core.symbolic.primitives_canonical import (
    alpha_rename,
    constant_integrand,
    distribute_mul_over_add,
    drop_zero_derivative_inner,
    drop_zero_integrand,
    kill_zero_length_integral,
    merge_integrals_over_add,
    protect_integrals,
    split_integral_over_add,
    un_subs,
    zero_derivative_of_free_symbol,
)
from zoomy_core.symbolic.primitives_calculus import (
    chain_rule,
    distribute_derivative_over_add,
    fundamental_theorem,
    integration_by_parts,
    leibniz,
    leibniz_general,
    polynomial_integrate,
    product_rule_forward,
    product_rule_inverse,
    pull_scalar_out_of_derivative,
    push_scalar_into_derivative,
)
from zoomy_core.symbolic.primitives_change_of_variable import (
    affine_change_of_variable,
)
from zoomy_core.symbolic.primitives_basis import (
    canonicalize_phi_derivative_subs,
    has_phi_call,
    project_basis_integrand,
)
from zoomy_core.symbolic.primitives_substitution import (
    function_expand,
    solve_for,
    subs_at_point,
    subst,
)
from zoomy_core.symbolic.domains import (
    BoundaryIntegral,
    Box,
    BoxBoundary,
    Domain,
    Interval,
    NormalVector,
    PointSet,
    Simplex,
    SimplexBoundary,
)

__all__ = [
    # held constructors
    "D",
    "Int",
    "Sub",
    "held_function",
    # exceptions / guards
    "PrimitiveDoesNotMatch",
    "AutoEvalGuard",
    "AutoEvalForbidden",
    # the structural-normalisation pass
    "canonicalise",
    # canonical-rewrite primitives
    "alpha_rename",
    "constant_integrand",
    "distribute_mul_over_add",
    "drop_zero_derivative_inner",
    "drop_zero_integrand",
    "kill_zero_length_integral",
    "merge_integrals_over_add",
    "protect_integrals",
    "split_integral_over_add",
    "un_subs",
    "zero_derivative_of_free_symbol",
    # calculus primitives
    "chain_rule",
    "distribute_derivative_over_add",
    "fundamental_theorem",
    "integration_by_parts",
    "leibniz",
    "leibniz_general",
    "polynomial_integrate",
    "product_rule_forward",
    "product_rule_inverse",
    "pull_scalar_out_of_derivative",
    "push_scalar_into_derivative",
    # variable change
    "affine_change_of_variable",
    # basis projection
    "canonicalize_phi_derivative_subs",
    "has_phi_call",
    "project_basis_integrand",
    # substitution + solver
    "function_expand",
    "solve_for",
    "subs_at_point",
    "subst",
    # symbolic domains / boundary / normals
    "BoundaryIntegral",
    "Box",
    "BoxBoundary",
    "Domain",
    "Interval",
    "NormalVector",
    "PointSet",
    "Simplex",
    "SimplexBoundary",
]
