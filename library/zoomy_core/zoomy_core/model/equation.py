"""Equation primitives — `Term`, `_TermGroup`, `Equation`.

Direct port of the framework developed in the transparent-derivation
notebooks (``thesis/notebooks/legacy/modeling/transparent_derivations/
sme_clean.py`` etc.).  Same classes, same auto-history mechanics, same
`simplify` pipeline (sp.expand → pull var-independent factors out of
Derivative atoms → recover fluxes by pulling var-independent coeffs back
inside Derivative arguments).

Lives in its own module so the `Model` class (in `basemodel.py`) and
the derivation subclasses (`sigmaref.py`, `sme.py`, `vam.py`) can import
`Equation` without dragging in the full operations / state surface.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Expression, RelationMixin


__all__ = [
    "Term",
    "_TermGroup",
    "_TermAccessor",
    "Equation",
    "_pull_constants_out",
    "_recover_fluxes",
]


# ── canonical-form helpers used by `Equation.simplify` ────────────────


def _pull_constants_out(*deriv_args):
    """For a `Derivative(inner, var)` atom: split `inner` into the
    var-independent and var-dependent parts and lift the
    var-independent part outside.

    For `inner` of type `Add`, distribute the Derivative
    (chain-rule for sums) — otherwise `as_independent` would treat
    `const` as an additive constant and `0 * Derivative(rest, var)`
    would wrongly zero the term.
    """
    d = sp.Derivative(*deriv_args)
    if len(d.variables) != 1:
        return d
    var = d.variables[0]
    inner = d.args[0]
    if inner == 0:
        return sp.S.Zero
    if isinstance(inner, sp.Add):
        return sp.Add(*[_pull_constants_out(a, var) for a in inner.args])
    const, rest = inner.as_independent(var, as_Add=False)
    if rest == 1 or rest == 0:
        return sp.S.Zero
    if const == 1:
        return d
    return const * sp.Derivative(rest, var)


def _recover_fluxes(expr):
    """For each leaf term of shape `c · Derivative(inner, var)` where
    `c` is *var-independent*, pull `c` back inside the Derivative so
    the conservative atom `Derivative(c · inner, var)` survives — that's
    what the flux/hydrostatic tagger reads."""
    new_terms = []
    for term in sp.Add.make_args(expr):
        if isinstance(term, sp.Mul):
            derivs = [a for a in term.args
                      if isinstance(a, sp.Derivative)
                      and len(a.variables) == 1]
            if len(derivs) == 1:
                d = derivs[0]
                var = d.variables[0]
                coeff = sp.Mul(*[a for a in term.args if a is not d])
                if not coeff.has(var):
                    new_terms.append(
                        sp.Derivative(coeff * d.args[0], var))
                    continue
        new_terms.append(term)
    return sp.Add(*new_terms)


# ── Term / _TermGroup / Equation ─────────────────────────────────────


class Term:
    """One additive term of an `Equation`."""

    __slots__ = ("expr", "tag", "_parent")

    def __init__(self, expr, tag=None, parent=None):
        self.expr = expr
        self.tag = tag
        self._parent = parent

    def apply(self, *args, **kwargs):
        new_expr = Expression(self.expr, "").apply(*args, **kwargs).expr
        pieces = list(sp.Add.make_args(sp.expand(new_expr)))
        if self._parent is None or len(pieces) == 1:
            self.expr = sp.Add(*pieces)
            return self
        idx = next(i for i, t in enumerate(self._parent._terms) if t is self)
        self._parent._terms[idx:idx + 1] = [
            Term(p, parent=self._parent) for p in pieces]
        return self

    def __repr__(self):
        tag = f", tag={self.tag!r}" if self.tag else ""
        return f"Term({self.expr}{tag})"


class _TermGroup:
    """Returned by `eq.term[[i, j, k]]` / `eq.term[i, j]`.  Forwards
    `apply` to each term in the held sub-list — enables
    `eq.term[[0, 1, 7]].apply(ProductRule(...))` over a hand-picked index
    list, writing the rewrites back into the parent equation."""

    def __init__(self, terms):
        self._terms = list(terms)

    def apply(self, *args, **kwargs):
        for t in self._terms:
            t.apply(*args, **kwargs)
        return self

    @property
    def expr(self):
        """The Add of the held terms' expressions (read-only view)."""
        return sum((t.expr for t in self._terms), sp.S.Zero)


class _TermAccessor:
    """The `eq.term` accessor — ADDITIVE-TERM selection on a scalar
    `Equation`.

    Disambiguated from moment/component `[...]` access (which lives on
    `MomentFamily` / `VectorEquation`): `eq.term[i]` is a single
    additive term, `eq.term[i, j]` / `eq.term[[i, j]]` a hand-picked term
    group.  Both return a term-view whose `.apply(op)` rewrites ONLY those
    additive terms and writes the result back into the parent equation, and
    whose `.expr` reads the selected sub-expression.
    """

    __slots__ = ("_equation",)

    def __init__(self, equation):
        self._equation = equation

    def __getitem__(self, i):
        terms = self._equation._terms
        if isinstance(i, (list, tuple)):
            return _TermGroup([terms[k] for k in i])
        return terms[i]

    def __setitem__(self, i, value):
        self._equation[i] = value

    def __iter__(self):
        return iter(self._equation._terms)

    def __len__(self):
        return len(self._equation._terms)


class Equation(RelationMixin):
    """Linear list of `Term` carrying the residual of one PDE/algebraic
    equation.  `apply` delegates to `Expression.apply` (the library
    operation-dispatch layer) and auto-records a history entry on the
    parent `Model` if one is set.

    Via `RelationMixin` an Equation can be
    ORIENTED — built from `sp.Eq(lhs, rhs)` or produced by `solve_for`
    — in which case it carries `_as_relation = {lhs: rhs}` and acts as a
    substitution under `.apply` (with `.lhs` / `.rhs` / `.subs_map`).
    That is what replaces the former `Substitution` / `Relation` classes.
    """

    def __init__(self, expression, name="eq", model=None):
        self.name = name
        # ``Equation(sp.Eq(lhs, rhs))`` is an oriented relation: keep the
        # residual ``lhs - rhs`` as the term-list source of truth, record the
        # ``{lhs: rhs}`` rule so ``.apply`` consumes it as a substitution.
        if isinstance(expression, sp.Equality):
            self._as_relation = {expression.lhs: expression.rhs}
            expression = expression.lhs - expression.rhs
        self._terms = [Term(t, parent=self)
                       for t in self.to_terms(expression)]
        self._model = model
        # Solver-tag groups populated by ``Model._auto_tag_equations``
        # (or by an explicit ``solver_tag`` call).  Mirrors the
        # ``Expression._solver_groups`` interface so
        # :func:`tag_extraction.collect_solver_tag` can extract operator
        # matrices directly from this Equation.
        self._solver_groups = None

    # ── solver-tag interface (mirrors Expression) ────────────────
    @property
    def solver_tags(self):
        """Dict of solver tags (canonical names) → sub-expression, or
        empty dict if none set."""
        return dict(self._solver_groups) if self._solver_groups else {}

    def get_solver_tag(self, name):
        """Return the sp.Expr for a solver tag, or None if not set.
        `name` is normalised through `canonical_solver_tag`."""
        from zoomy_core.model.derivation.tag_catalog import canonical_solver_tag
        if not self._solver_groups:
            return None
        return self._solver_groups.get(canonical_solver_tag(name))

    def untagged_remainder(self):
        """Return `self.expr - sum(solver_tags.values())`, simplified.
        Zero ⇒ every term has been routed to a solver tag; non-zero
        drives `collect_solver_tag`'s `untagged_policy`."""
        if not self._solver_groups:
            return self.expr
        tagged_sum = sum(self._solver_groups.values(), sp.S.Zero)
        return sp.expand(self.expr - tagged_sum)

    # ── term ↔ expression bridges ─────────────────────────────────
    @staticmethod
    def to_terms(expression):
        # ``mul`` distributes products into additive terms (what we need to split
        # a leaf into Terms); ``multinomial=False`` leaves ``Pow(sum, n)``
        # UNEXPANDED — critical for rational turbulence sources, where a deep
        # expand would multinomial-blow ``sk⁴``/``(se²+ε_min)`` (work
        # GaussQuadrature does numerically later), turning the build from seconds
        # into ~hours.  Polynomial sources have no ``Pow(Add, n)`` at leaf level,
        # so the term split is unchanged for them.
        return list(sp.Add.make_args(
            sp.expand(expression, multinomial=False)))

    @staticmethod
    def to_expression(terms):
        return sum((t.expr if isinstance(t, Term) else t for t in terms),
                   sp.S.Zero)

    @property
    def expr(self):
        return self.to_expression(self._terms)

    @expr.setter
    def expr(self, value):
        self._terms = [Term(t, parent=self) for t in self.to_terms(value)]

    @property
    def terms(self):
        return self._terms

    @property
    def term(self):
        """ADDITIVE-TERM accessor — `eq.term[i]` (one term),
        `eq.term[i, j]` / `eq.term[[i, j]]` (a term group).  Each view's
        `.apply(op)` rewrites ONLY those terms back into this equation.

        Disambiguated from `[l]` MOMENT access (`MomentFamily` /
        `VectorEquation`): a scalar `Equation` is no longer term-indexable
        via bare `eq[i]` — use `eq.term[i]`."""
        return _TermAccessor(self)

    # ── indexing + iteration ─────────────────────────────────────
    def __getitem__(self, i):
        raise TypeError(
            "index a scalar Equation's terms via `.term[i]`; "
            "`[l]` is reserved for moment rows")

    def __setitem__(self, i, value):
        if isinstance(value, Term):
            value._parent = self
            self._terms[i] = value
        else:
            self._terms[i] = Term(value, parent=self)

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    # ── apply / simplify ─────────────────────────────────────────
    def apply(self, *args, **kwargs):
        # ``_no_history`` is set by ``Model.apply`` when it iterates over
        # every equation — that path adds a single ``target=*`` entry
        # instead of N per-equation duplicates.
        no_history = kwargs.pop("_no_history", False)
        level = kwargs.pop("level", "major")
        description = kwargs.pop("description", None)
        op = args[0] if args else None
        # Structural ops (e.g. ``ResolveModes``) restructure the parent model
        # rather than transform this equation's leaf expression: route them
        # through ``apply_to_equation`` and still record a history entry.
        if op is not None and hasattr(op, "apply_to_equation"):
            result = op.apply_to_equation(self)
            if not no_history and self._model is not None:
                self._model._history(
                    getattr(op, "name", None) or type(op).__name__,
                    self.name, level=level, description=description,
                    log_level=getattr(op, "log_level", 1))
            return result
        merged = Expression(self.expr, self.name)
        new = merged.apply(*args, **kwargs)
        self.expr = new.expr
        if not no_history and self._model is not None and args:
            op = args[0]
            self._model._history(
                getattr(op, "name", None) or type(op).__name__,
                self.name, level=level, description=description,
                log_level=getattr(op, "log_level", 1),
            )
        return self

    def simplify(self):
        """Canonical form (three sub-steps):

        1. `sp.expand` — flatten Mul/Add nesting and combine like terms.
        2. **Pull var-independent factors out** of every Derivative
           argument: `Derivative(g·h, x) → g·Derivative(h, x)`,
           `Derivative(0, x) → 0`.
        3. **Recover fluxes** — for each term `c · Derivative(_, x)`
           where `c` is var-independent, pull `c` back inside so the
           tagger sees the conservative atom.
        """
        expr = sp.expand(self.expr)
        expr = expr.replace(sp.Derivative, _pull_constants_out)
        expr = sp.expand(expr)
        expr = _recover_fluxes(expr)
        self.expr = expr
        return self

    # ── solve / remove (derivation-core operations) ──────────────────
    def solve_for(self, symbol):
        """Algebraically isolate `symbol` from this equation's residual
        (`self.expr == 0`); returns a NEW oriented `Equation`
        `symbol = solved-expr` (carrying `_as_relation = {symbol: sol}`).

        The result *stays an Equation* — pass it straight to `.apply` and it
        acts as the substitution `symbol -> sol` (`.lhs` / `.rhs` /
        `.subs_map` expose the rule).  This is what the old `Substitution`
        return value did, with no separate class.

        Pure algebra — `solve_for` does **not** integrate.  If `symbol`
        appears under a `Derivative` the residual is *differential* in it;
        integrate the balance first (e.g. `eq.apply(Integrate(var, lo, hi))`
        followed by a boundary-condition substitution) and then `solve_for`
        the resulting algebraic relation.
        """
        residual = sp.expand(self.expr)
        # Block only when ``symbol`` sits under a DIFFERENT/higher derivative
        # (genuinely differential).  Solving FOR a derivative (e.g. ``∂_t h``
        # from a mass balance) is fine: ``∂_t h`` is itself a ``Derivative``
        # atom, so ``der.has(symbol)`` would misfire on the ``der == symbol``
        # case — exclude it.
        if any(der != symbol and der.has(symbol)
               for der in residual.atoms(sp.Derivative)):
            raise ValueError(
                f"solve_for({symbol}) is algebraic, but {self.name!r} is "
                f"differential in {symbol} (it appears under a Derivative). "
                f"Integrate the balance first (e.g. `.apply(Integrate(var, "
                f"lo, hi))` + a boundary-condition substitution), then "
                f"solve_for the resulting algebraic relation.")
        solutions = sp.solve(residual, symbol)
        if not solutions:
            raise ValueError(f"Cannot solve {self.name!r} for {symbol}.")
        return Equation(sp.Eq(symbol, solutions[0]),
                        name=f"solve[{self.name}->{symbol}]")

    def remove(self):
        """Delete this equation from its parent model (triggers Q refresh)."""
        if self._model is None:
            raise RuntimeError(
                f"Equation {self.name!r} has no parent model to remove from.")
        self._model._remove_equation(self.name)
        return None

    # ── describe ─────────────────────────────────────────────────────
    def _ordered_tex(self, strip_args=False):
        """LaTeX of the residual with the terms in their CURRENT order — sympy's
        `Add` re-sorts to a canonical order, so an unevaluated `Add` is
        printed with `order='none'` to honour a `SortByTag` ordering."""
        ordered = (sp.Add(*[tm.expr for tm in self._terms], evaluate=False)
                   if self._terms else sp.S.Zero)
        from zoomy_core.model.operations import (
            _StripArgsLatexPrinter, _BracketLatexPrinter)
        printer = (_StripArgsLatexPrinter if strip_args
                   else _BracketLatexPrinter)(settings={"order": "none"})
        return printer.doprint(ordered)

    def _tagged_tex(self, strip_args=False):
        """Multiline `\\underbrace{…}_{tag}` render — consecutive same-tag
        terms (the rest state after `SortByTag`) grouped under one brace
        labelled by their physics tag."""
        from zoomy_core.model.operations import (
            _StripArgsLatexPrinter, _BracketLatexPrinter)
        printer = (_StripArgsLatexPrinter if strip_args
                   else _BracketLatexPrinter)(settings={"order": "none"})
        groups = []                                   # [(tag, [exprs])], in order
        for tm in self._terms:
            tag = tm.tag or "untagged"
            if groups and groups[-1][0] == tag:
                groups[-1][1].append(tm.expr)
            else:
                groups.append((tag, [tm.expr]))
        lines = []
        for i, (tag, exprs) in enumerate(groups):
            tex = printer.doprint(sp.Add(*exprs, evaluate=False))
            label = tag.replace("_", " ")
            sign = "" if (i == 0 or tex.startswith("-")) else "+ "
            lines.append(rf"  & {sign}\underbrace{{{tex}}}_{{\text{{{label}}}}}")
        return ("\\begin{aligned}\n" + " \\\\\n".join(lines)
                + "\n  &= 0\n\\end{aligned}")

    def _describe_tex(self, strip_args=False, show_tags=False):
        if show_tags and any(tm.tag for tm in self._terms):
            return self._tagged_tex(strip_args)
        return f"{self._ordered_tex(strip_args)} = 0"

    def describe(self, strip_args=False, header=True, show_tags=False):
        """Render this equation as a `Description`.

        Terms print in their current order (so a `SortByTag` shows).  An
        oriented equation (from `solve_for` / `sp.Eq`) renders `lhs = rhs`
        rather than `residual = 0`.  `show_tags=True` underbraces each
        physics-tag group."""
        from zoomy_core.misc.description import Description
        parts = []
        if header:
            parts.append(f"**{self.name}** ({len(self._terms)} terms)")
        if self._as_relation:
            from zoomy_core.model.operations import (
                _StripArgsLatexPrinter, _BracketLatexPrinter)
            pr = (_StripArgsLatexPrinter if strip_args
                  else _BracketLatexPrinter)(settings={"order": "none"})
            tex = lambda e: pr.doprint(e)
            body = " \\\\ ".join(f"{tex(l)} = {tex(r)}"
                                 for l, r in self._as_relation.items())
            parts.append(f"\n$$\n\\begin{{aligned}} {body} \\end{{aligned}}\n$$")
        else:
            parts.append(f"\n$$\n{self._describe_tex(strip_args, show_tags)}\n$$")
        return Description("\n".join(parts))

    def describe_line(self, strip_args=False, bullet=False, show_tags=False):
        """One-line markdown render (used by `Model.describe`).  An oriented
        equation renders as `lhs = rhs`; a plain residual as `… = 0` with
        terms in their current order.  `show_tags` underbraces each tag
        group (same render as `describe`)."""
        prefix = "- " if bullet else ""
        if self._as_relation:
            from zoomy_core.model.operations import (
                _StripArgsLatexPrinter, _BracketLatexPrinter)
            pr = (_StripArgsLatexPrinter if strip_args
                  else _BracketLatexPrinter)(settings={"order": "none"})
            tex = lambda e: pr.doprint(e)
            body = " \\\\ ".join(f"{tex(l)} = {tex(r)}"
                                 for l, r in self._as_relation.items())
            return f"{prefix}**{self.name}:** ${body}$"
        return f"{prefix}**{self.name}:** ${self._describe_tex(strip_args, show_tags)}$"

    def __repr__(self):
        return f"Equation({self.name!r}, {len(self._terms)} terms)"
