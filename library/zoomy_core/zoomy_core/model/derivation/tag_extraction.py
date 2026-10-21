"""Collects solver-tagged terms into operator shapes (flux, non-conservative, source) and auto-tags
raw residuals by a term classifier. Entry point is collect_solver_tag, called from a model's flux()
/ source() / nonconservative_matrix() bodies.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import sympy as sp
from sympy import Add, Derivative, Mul, S

from zoomy_core.model.derivation.tag_catalog import canonical_solver_tag


class SolverTagged:
    """Lightweight carrier of a residual plus its solver-tag groups, the return type of
    auto_solver_tag. It has the same duck-typed surface as Equation (expr, solver_tags,
    get_solver_tag, untagged_remainder), which collect_solver_tag consumes.
    """

    def __init__(self, expr, name="", solver_groups=None):
        """Store the residual, name and optional tag groups."""
        self.expr = sp.sympify(expr)
        self.name = name
        self._solver_groups = dict(solver_groups) if solver_groups else None

    def solver_tag(self, **named_groups):
        """Return a new carrier with the given tag groups merged in; names are normalized through
        the alias catalog.
        """
        merged = dict(self._solver_groups or {})
        for k, v in named_groups.items():
            merged[canonical_solver_tag(k)] = (
                v.expr if isinstance(v, SolverTagged) else sp.sympify(v))
        return SolverTagged(self.expr, self.name, solver_groups=merged)

    @property
    def solver_tags(self):
        """Dict of canonical tag to sub-expression, empty if none set."""
        return dict(self._solver_groups) if self._solver_groups else {}

    def get_solver_tag(self, name):
        """Return the sub-expression of a tag (any alias), or None if unset."""
        if not self._solver_groups:
            return None
        return self._solver_groups.get(canonical_solver_tag(name))

    def untagged_remainder(self):
        """Return the expanded residual minus all tagged parts; non-zero means untagged terms."""
        if not self._solver_groups:
            return self.expr
        tagged_sum = sum(self._solver_groups.values(), S.Zero)
        return sp.expand(self.expr - tagged_sum)


# ---------------------------------------------------------------------------
# Per-canonical extraction helpers (private)
# ---------------------------------------------------------------------------

def _iter_terms(expr):
    """Return the additive terms of the expanded expression."""
    return Add.make_args(sp.expand(expr))


def _split_coeff_and_derivative(term):
    """Split a term into (coeff, Derivative) if it is Derivative or coeff * Derivative, else (None,
    None).
    """
    if isinstance(term, Derivative):
        return S.One, term
    if isinstance(term, Mul):
        derivs = [a for a in term.args if isinstance(a, Derivative)]
        if len(derivs) != 1:
            return None, None
        deriv = derivs[0]
        coeff = Mul(*[a for a in term.args if a is not deriv])
        return coeff, deriv
    return None, None


def _first_order_direction(deriv, coords):
    """Return the index in coords of a first-order derivative w.r.t. exactly one coordinate, else
    None.
    """
    variables = deriv.variables
    if len(variables) != 1:
        return None
    v = variables[0]
    if v not in coords:
        return None
    # Order check: second-order would appear as Derivative(F, x, x) with
    # variables == (x, x), length 2. We've already filtered that out above.
    return list(coords).index(v)


def _extract_conservative(expr, coords, n_directions, context,
                          state_variables=None):
    """Return [F_0, ..., F_{n_directions-1}] from terms coeff * Derivative(F_i, x_i). Raises
    ValueError for non-conservative terms, including state variables in coeff if state_variables is
    given.
    """
    state_set = set(state_variables) if state_variables is not None else None
    result = [S.Zero] * n_directions
    for term in _iter_terms(expr):
        if term == S.Zero:
            continue
        coeff, deriv = _split_coeff_and_derivative(term)
        if deriv is None:
            raise ValueError(
                f"{context}: term {term} is not in conservative form "
                f"coeff * d/dx_i(F_i). Rewrite in conservative form or "
                f"use a different solver tag."
            )
        i = _first_order_direction(deriv, coords)
        if i is None:
            raise ValueError(
                f"{context}: term {term} — derivative {deriv} is not first-order "
                f"w.r.t. one of the spatial coords {list(coords)}."
            )
        if i >= n_directions:
            raise ValueError(
                f"{context}: direction index {i} out of range "
                f"(n_directions={n_directions})."
            )
        if state_set is not None:
            coeff_state_refs = state_set & coeff.free_symbols
            if coeff_state_refs:
                raise ValueError(
                    f"{context}: term {term} has state variables "
                    f"{coeff_state_refs} in the coefficient outside the "
                    f"derivative — that is a non-conservative product, not a "
                    f"conservative flux. Move them inside d/dx_i(...) or use "
                    f"the 'nonconservative_flux' tag."
                )
        F_i = coeff * deriv.args[0]
        result[i] = result[i] + F_i
    return result


def _extract_nc(expr, state_variables, coords, context):
    """Return (j, i, coeff) triples for terms coeff * d q_j / d x_i."""
    contributions = []
    for term in _iter_terms(expr):
        if term == S.Zero:
            continue
        coeff, deriv = _split_coeff_and_derivative(term)
        if deriv is None:
            raise ValueError(
                f"{context}: term {term} is not of form coeff * d q_j / d x_i."
            )
        i = _first_order_direction(deriv, coords)
        if i is None:
            raise ValueError(
                f"{context}: derivative {deriv} is not first-order w.r.t. one "
                f"of the spatial coords {list(coords)}."
            )
        inner = deriv.args[0]
        try:
            j = list(state_variables).index(inner)
        except ValueError:
            raise ValueError(
                f"{context}: derivative argument {inner!s} is not a state "
                f"variable. State variables: {list(state_variables)}."
            ) from None
        contributions.append((j, i, coeff))
    return contributions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def collect_solver_tag(
    system,
    tag: str,
    *,
    variable_map: dict,
    n_variables: int,
    n_directions: int = 1,
    state_variables: Optional[Sequence] = None,
    coords: Optional[Sequence] = None,
    policy: str = "strict",
):
    """Assemble the operator of one solver tag (flux, nonconservative_flux, source, time_derivative)
    from tagged equations.

    Args:
        system: Object with .equations mapping name to a tagged equation.
        tag: Canonical name or alias.
        variable_map: {equation_name: [output rows]}.
        n_variables, n_directions: Output shape.
        state_variables: Required for nonconservative_flux.
        coords: Spatial coordinates, required for flux-type tags and NCP.
        policy: "strict", "warn" or "ignore" for a non-zero untagged remainder.

    Returns:
        Matrix, N-d array or list, depending on the tag's rank; source tags carry a sign flip to the
            RHS.
    """
    canonical = canonical_solver_tag(tag)

    if canonical in ("implicit_diffusion", "explicit_diffusion"):
        raise NotImplementedError(
            f"Tag-based extraction of {canonical!r} is not implemented. "
            "Author diffusion by overriding "
            "``Model.diffusion_matrix`` (implicit) or "
            "``Model.diffusion_matrix_explicit`` (explicit) to return "
            "the rank-4 tensor A of shape (n_eq, n_state, n_dim, n_dim)."
        )

    if canonical in ("flux", "hydrostatic_pressure"):
        if coords is None:
            raise ValueError(f"tag {canonical!r} requires ``coords``.")
        out = sp.Matrix.zeros(n_variables, n_directions)
        kind = "conservative"
    elif canonical == "nonconservative_flux":
        if coords is None or state_variables is None:
            raise ValueError(
                "tag 'nonconservative_flux' requires ``coords`` and ``state_variables``."
            )
        out = sp.MutableDenseNDimArray.zeros(n_variables, n_variables, n_directions)
        kind = "nc"
    elif canonical in ("implicit_source", "explicit_source", "time_derivative"):
        out = [S.Zero] * n_variables
        kind = "raw"
    else:
        raise ValueError(f"Unsupported canonical tag: {canonical!r}")

    for eq_name, eq in system.equations.items():
        if eq_name not in variable_map:
            continue
        rows = variable_map[eq_name]

        _enforce_untagged_policy(eq, eq_name, policy)

        tag_expr = eq.get_solver_tag(canonical)
        if tag_expr is None or tag_expr == 0:
            continue

        ctx = f"tag {canonical!r} on equation {eq_name!r}"

        if kind == "conservative":
            F = _extract_conservative(tag_expr, coords, n_directions, ctx,
                                      state_variables=state_variables)
            for row in rows:
                for col, F_i in enumerate(F):
                    out[row, col] = out[row, col] + F_i
        elif kind == "nc":
            contribs = _extract_nc(tag_expr, state_variables, coords, ctx)
            for row in rows:
                for (j, i, coeff) in contribs:
                    out[row, j, i] = out[row, j, i] + coeff
        else:  # raw
            # SystemModel convention: ``∂_t Q + ∂_x F + ∂_x P + B·∂_x Q
            # = S`` (S on the RHS).  Source-tagged terms come from the
            # *LHS* of the equation (``... + source_term = 0``), so
            # they get a sign flip when moved to the RHS source slot.
            # ``time_derivative`` is NOT flipped — it goes into the
            # mass_matrix slot which sits on the LHS with the temporal
            # derivative.
            sign = -1 if canonical in (
                "implicit_source", "explicit_source") else +1
            for row in rows:
                out[row] = out[row] + sign * tag_expr

    return out


def _enforce_untagged_policy(eq, eq_name, policy):
    """Raise, warn or ignore if a tagged equation has a non-zero untagged remainder."""
    if not getattr(eq, "_solver_groups", None):
        return
    remainder = eq.untagged_remainder()
    if remainder == 0:
        return
    msg = (
        f"Equation {eq_name!r}: untagged remainder {remainder!s} "
        f"(solver_tags set: {list(eq.solver_tags.keys())}). "
        f"Either tag all terms or lower untagged_policy."
    )
    if policy == "strict":
        raise ValueError(msg)
    if policy == "warn":
        import warnings
        warnings.warn(msg, stacklevel=3)
    # policy == "ignore" → do nothing


# ---------------------------------------------------------------------------
# Heuristic auto-tagger
# ---------------------------------------------------------------------------

def _classify_term(term, *, state_funcs, gravity_param, t, coords):
    """Classify one PDE term into a canonical solver tag by priority: time derivative, flux or
    nonconservative_flux for coeff * d(state), hydrostatic_pressure, else implicit_source. The
    direction check runs over all spatial coords so cross-direction conservative fluxes are not lost
    into the source.
    """
    if any(isinstance(d, sp.Derivative) and t in d.variables
           for d in term.atoms(sp.Derivative)):
        return "time_derivative"
    coeff, deriv = _split_coeff_and_derivative(term)
    if deriv is None:
        return "implicit_source"
    if len(deriv.variables) != 1 or deriv.variables[0] not in coords:
        return "implicit_source"
    inner = deriv.args[0]
    if inner in state_funcs:
        # ∂_x(q_j) with a single bare state variable inside.  A STATE-FREE
        # coefficient makes this identical to ∂_x(coeff·q_j): a CONSERVATIVE
        # flux (it telescopes / conserves mass), so it must go to ``flux``,
        # not the non-conservative slot.  The mass flux ∂_x(q_0) is exactly
        # this case — tagging it NCP breaks the path-conservative product's
        # telescoping and leaks mass.  Only a STATE-DEPENDENT coefficient
        # (e.g. g·h·∂_x b) is a genuine non-conservative product.
        coeff_has_state = any(coeff.has(f) for f in state_funcs)
        return "nonconservative_flux" if coeff_has_state else "flux"
    inner_has_state = any(inner.has(f) for f in state_funcs)
    if not inner_has_state:
        return "implicit_source"
    coeff_has_state = any(coeff.has(f) for f in state_funcs)
    if coeff_has_state:
        return "implicit_source"
    if gravity_param is not None and inner.has(gravity_param):
        return "hydrostatic_pressure"
    return "flux"


def _is_self_diffusion_compound(term, own_func, coords):
    """True iff the term is a compound second-order coeff * d_xd(A * d_xe(.)) whose inner derivative
    carries the row's own state function. Off-diagonal terms of a foreign field (e.g. the bed slope)
    return False and stay conservative fluxes.
    """
    if own_func is None:
        return False
    _coeff, deriv = _split_coeff_and_derivative(term)
    if deriv is None:
        return False
    # Outer must be a single first-order spatial derivative ``∂_{x_d}(arg)``.
    if len(deriv.variables) != 1 or deriv.variables[0] not in coords:
        return False
    inner = deriv.args[0]
    for dd in inner.atoms(sp.Derivative):
        if (any(v in coords for v in dd.variables)
                and dd.args[0].has(own_func)):
            return True
    return False


def auto_solver_tag(expr_or_leaf, *, state_funcs, t, coords, gravity_param=None,
                    own_func=None):
    """Auto-tag the additive terms of a residual with canonical solver tags and return a
    SolverTagged.

    Args:
        expr_or_leaf: Residual expression or SolverTagged.
        state_funcs: Evolved state functions.
        t: Time symbol.
        coords: Full spatial coordinate list.
        gravity_param: Gravity symbol marking hydrostatic pressure.
        own_func: The row's own state function; self-diffusion compounds are expanded to a source.
    """
    if isinstance(expr_or_leaf, SolverTagged):
        raw, name = expr_or_leaf.expr, expr_or_leaf.name
    else:
        raw, name = sp.sympify(expr_or_leaf), ""
    groups: dict[str, sp.Expr] = {}
    # ``canon`` accumulates the residual in the SAME form that was routed into
    # the groups (self-diffusion compounds replaced by their ``.doit()`` ∂²_x
    # expansion), so ``untagged_remainder`` stays exactly zero under the strict
    # policy.  For every non-viscous residual ``canon`` is just ``expand(raw)``.
    canon_terms: list = []

    def _route(term):
        """Classify one term and add it to its tag group."""
        tag = _classify_term(
            term, state_funcs=state_funcs, gravity_param=gravity_param,
            t=t, coords=coords,
        )
        groups[tag] = groups.get(tag, sp.S.Zero) + term
        canon_terms.append(term)

    for term in sp.Add.make_args(sp.expand(raw)):
        if term == sp.S.Zero:
            continue
        if _is_self_diffusion_compound(term, own_func, coords):
            # Expand the diffusive flux ``∂_x(A·∂_x own)`` to its ∂²_x source
            # form and route the pieces (all carry the row's own gradients →
            # source); foreign-field compounds are NOT taken down this path.
            for piece in sp.Add.make_args(sp.expand(term.doit())):
                if piece != sp.S.Zero:
                    _route(piece)
            continue
        _route(term)
    canon = sp.Add(*canon_terms)
    return SolverTagged(canon, name).solver_tag(**groups)
