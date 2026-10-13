"""Composable projection operations on INS equations (`Expression`, `SigmaTransform`, `Recombine`, ...).

The user drives every step; there are no hidden assumptions.
"""

import warnings

import sympy as sp
from sympy import (
    Function, Symbol, Derivative, Integral, Add, Mul, S, Rational,
    Heaviside, DiracDelta,
)
import numpy as np


# Sentinel for optional kwargs that must distinguish "not passed" from
# "passed a falsy value" (most notably ``rhs=0``).
_UNSET = object()


# ---------------------------------------------------------------------------
# StateSpace: shared symbolic state
# ---------------------------------------------------------------------------

class StateSpace:
    """
    Shared symbolic state: coordinates, velocity fields, pressure,
    stress tensor, bathymetry, and physical parameters.

    dimension is the physical space dimension:
        2 = xz plane (1D horizontal shallow water)
        3 = xyz space (2D horizontal shallow water)

    The vertical coordinate is always z. Horizontal coordinates are
    x (and y if dimension=3). The horizontal_dim property gives the
    number of horizontal directions (1 or 2).
    """

    def __init__(self, dimension=2):
        if dimension < 2 or dimension > 3:
            raise ValueError(f"dimension must be 2 (xz) or 3 (xyz), got {dimension}")
        self.dim = dimension

        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.y = Symbol("y", real=True)
        self.z = Symbol("z", real=True)

        has_y = dimension > 2

        args_h = [self.t, self.x]
        if has_y:
            args_h.append(self.y)
        args_3d = args_h + [self.z]
        self._args_h = args_h
        self._args_3d = args_3d

        # Reference-element symbol used for ζ-integrals on [0, 1] and as the
        # post-AffineProjection coordinate; opaque ``state.zeta`` carries the
        # full (t, x, [y,] z) dependence so sympy's native chain rule fires
        # cleanly through ``phi_k(state.zeta)``.
        self.zeta_ref = Symbol("zeta", real=True)
        self.zeta = Function("zeta", real=True)(*args_3d)

        self.u = Function("u", real=True)(*args_3d)
        self.v = Function("v", real=True)(*args_3d) if has_y else S.Zero
        self.w = Function("w", real=True)(*args_3d)
        self.p = Function("p", real=True)(*args_3d)

        self.rho = Symbol("rho", positive=True)
        self.g = Symbol("g", positive=True)

        self._build_stress_tensor(has_y, args_3d)

        self.b = Function("b", real=True)(*args_h)
        self.h = Function("h", real=True)(*args_h)
        self.eta = self.b + self.h

        self.coords_h = [self.x] + ([self.y] if has_y else [])
        self.velocities_h = [self.u] + ([self.v] if has_y else [])

    def _build_stress_tensor(self, has_y, args_3d):
        from zoomy_core.misc.misc import Zstruct
        labels = ["x", "y", "z"] if has_y else ["x", "z"]
        tau_dict = {}
        for i in labels:
            for j in labels:
                tau_dict[i + j] = Function(f"tau_{i}{j}", real=True)(*args_3d)
        self.tau = Zstruct(**tau_dict)

    @property
    def horizontal_dim(self):
        return self.dim - 1

    @property
    def has_y(self):
        return self.dim > 2

    def __repr__(self):
        n_tau = self.tau.length()
        fields = "[u,v,w,p]" if self.has_y else "[u,w,p]"
        return f"StateSpace(dim={self.dim}, fields={fields}, tau={n_tau} components)"


# ---------------------------------------------------------------------------
# SymbolicBase: shared name + display
# ---------------------------------------------------------------------------

class SymbolicBase:
    """Base for all symbolic objects. Provides name and notebook display."""

    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# RelationMixin: the {lhs: rhs} orientation surface
# ---------------------------------------------------------------------------

class RelationMixin:
    """Orientation surface shared by `Expression` and
    `Equation`.

    An expression/equation OPTIONALLY carries a `{lhs: rhs}` relation in
    `_as_relation` — set either by `Expression.solve_for` (which
    isolates a symbol) or by constructing from `sp.Eq(lhs, rhs)`.  When
    present it acts as a substitution under `.apply` (the dispatcher reads
    `_as_relation`) and exposes a dict-like `.lhs` / `.rhs` /
    `.subs_map` surface.  So a SOLVED equation *is* the relation — there is
    no separate `Relation` / `Substitution` class.

    When NOT oriented, the object is a plain residual `expr == 0`: `.lhs`
    is the residual, `.rhs` is `0`, `.subs_map` is empty.
    """

    #: Default (class-level): no relation attached.  ``solve_for`` / the
    #: ``sp.Eq`` constructor set an instance-level dict.
    _as_relation = None

    @property
    def oriented(self) -> bool:
        """True when this carries a `{lhs: rhs}` relation."""
        return bool(self._as_relation)

    @property
    def subs_map(self) -> dict:
        """The `{lhs: rhs}` substitution rules (empty dict if not oriented)."""
        return dict(self._as_relation) if self._as_relation else {}

    @property
    def lhs(self):
        """Left side — the solved-for symbol when oriented, else the residual."""
        if self._as_relation and len(self._as_relation) == 1:
            return next(iter(self._as_relation))
        return self.expr

    @property
    def rhs(self):
        """Right side — the solution when oriented, else `0`."""
        if self._as_relation and len(self._as_relation) == 1:
            return next(iter(self._as_relation.values()))
        return S.Zero


# ---------------------------------------------------------------------------
# Expression: composable symbolic PDE term
# ---------------------------------------------------------------------------

class Expression(RelationMixin, SymbolicBase):
    """
    Symbolic expression for PDE terms.

    Supports:
    - Term access: .terms, [i], len(), iteration
    - Projection: .project(test, var, domain, ...)
    - Integration by parts: .ibp(var, test_weight, domain) -> IBPResult
    - Apply conditions: .apply({old: new}, ...) or .apply(relation)
    - SymPy: .subs(), .simplify(), .expand(), .doit()
    - Notebook: _repr_latex_
    """

    def __init__(self, expr, name="", term_groups=None, solver_groups=None,
                 term_tags=None, tag_order=None):
        super().__init__(name)
        if isinstance(expr, Expression):
            if expr._as_relation:
                self._as_relation = dict(expr._as_relation)
            expr = expr.expr
        # ``Expression(sp.Eq(lhs, rhs))`` builds an ORIENTED expression: the
        # residual ``lhs - rhs`` is the source of truth (every op acts on it),
        # while ``_as_relation`` records the ``{lhs: rhs}`` substitution that
        # ``.apply`` consumes.  ``Expression(expr)`` stays the plain ``expr == 0``.
        if isinstance(expr, sp.Equality):
            self._as_relation = {expr.lhs: expr.rhs}
            expr = expr.lhs - expr.rhs
        self.expr = sp.sympify(expr) if not isinstance(expr, sp.Basic) else expr
        # Physical tags: per-additive-term labels for the display view.
        # ``_term_tags`` maps each tagged term (sp.Expr) to the tag name it
        # belongs to.  Tags are propagated *follow-the-term* by ``.apply``,
        # ``.simplify``, ``.subs``, ``.expand``: when a transformation
        # replaces ``old_term`` with ``new_term(s)``, every new term
        # inherits the old term's tag.  Cross-tag cancellation in the main
        # ``self.expr`` is reflected by terms vanishing from both
        # ``Add.make_args(self.expr)`` and ``_term_tags``.
        self._term_tags = dict(term_tags) if term_tags else {}
        # Display order for tags (first-seen wins).
        self._tag_order = list(tag_order) if tag_order else []
        # Back-compat: legacy ``term_groups`` kwarg (parallel-sum dict).
        # Convert to per-term entries.  Summing is associative, so splitting
        # a tag's group into Add.make_args and recording each piece gives
        # an equivalent view.
        if term_groups:
            for name_, g in term_groups.items():
                g_sp = g if isinstance(g, sp.Basic) else sp.sympify(g)
                # ``sp.expand`` distributes scalar factors across Adds so
                # something like ``-(∂_x τ_xx + ∂_z τ_xz)/ρ`` gets split
                # into its atomic additive terms before we record each
                # under the tag.
                for term in Add.make_args(sp.expand(g_sp)):
                    if term != S.Zero:
                        self._term_tags[term] = name_
                if name_ not in self._tag_order:
                    self._tag_order.append(name_)
        # Solver tag groups for operator routing (flux, source, NC, ...).
        # {canonical_tag: sp.Expr}. Dropped per-tag whenever apply/simplify
        # changes the tagged sub-expression (structural equality check).
        self._solver_groups = dict(solver_groups) if solver_groups else None

    # ``_term_groups`` is a lazy view over ``_term_tags`` filtered by the
    # current ``self.expr`` terms.  It's always in sync with the main
    # expression (terms that cancelled out of ``self.expr`` disappear
    # from the groups automatically).  Kept as a property for any code
    # that still reads ``_term_groups``.
    @property
    def _term_groups(self):
        if not self._term_tags:
            return None
        current = set(Add.make_args(sp.expand(self.expr)))
        groups = {name: S.Zero for name in self._tag_order}
        for term in current:
            tag_name = self._term_tags.get(term)
            if tag_name is None:
                continue
            if tag_name not in groups:
                groups[tag_name] = S.Zero
            groups[tag_name] = groups[tag_name] + term
        non_empty = {k: v for k, v in groups.items() if v != S.Zero}
        return non_empty if non_empty else None

    @_term_groups.setter
    def _term_groups(self, value):
        # Setter exists only so ``__init__`` could still do the legacy
        # assignment during early-init before the dict is built.  The
        # attribute has been fully migrated to ``_term_tags`` / ``_tag_order``.
        pass

    @property
    def terms(self):
        expanded = _expand_preserve_integrals(self.expr)
        raw = Add.make_args(expanded)
        return [Expression(t, f"{self.name}[{i}]") for i, t in enumerate(raw)]

    def __getitem__(self, i):
        if isinstance(i, slice):
            ts = self.terms[i]
            combined = sum((t.expr for t in ts), S.Zero)
            return Expression(combined, self.name)
        if isinstance(i, (list, tuple)):
            # Batch read-only view of selected terms — fresh Expression
            # carrying the sum of the picked terms.  Not a live accessor.
            ts = self.terms
            combined = sum((ts[k].expr for k in i), S.Zero)
            return Expression(combined, self.name)
        # Integer index — return a live accessor whose `.apply(...)`
        # mutates this parent Expression in place.  Use ``.terms[i]``
        # if you want a detached snapshot Expression.
        return _TermAccessor(self, i)

    def apply_to_term(self, index, *operations):
        """Apply operations to a specific term and return the full expression.

        Usage:

        ```
            xmom.apply_to_term(5, ProductRule())
            xmom.apply_to_term(5, {old: new})
        ```

        .. note::
           Public callers should prefer `xmom[5].apply(...)` —
           `Expression.__getitem__` returns a live accessor whose
           `.apply` mutates the parent in place.  This method is kept
           as the internal split-apply-merge primitive that the accessor
           uses; library code may continue to call it directly.
        """
        terms = self.terms
        modified = terms[index].apply(*operations)
        new_terms = [t.expr if i != index else modified.expr
                     for i, t in enumerate(terms)]
        return Expression(sum(new_terms, S.Zero), self.name)

    def keep_groups(self, *group_names):
        """Return a new Expression keeping only the specified tags' terms."""
        if not self._term_tags:
            raise ValueError("Expression has no tags.")
        kept_terms = {t: name for t, name in self._term_tags.items()
                      if name in group_names}
        if not kept_terms:
            return Expression(S.Zero, self.name)
        kept_order = [n for n in self._tag_order if n in group_names]
        new_expr = sum(kept_terms.keys(), S.Zero)
        return Expression(new_expr, self.name,
                          term_tags=kept_terms, tag_order=kept_order)

    def drop_groups(self, *group_names):
        """Return a new Expression dropping the specified tags' terms."""
        if not self._term_tags:
            raise ValueError("Expression has no tags.")
        kept_terms = {t: name for t, name in self._term_tags.items()
                      if name not in group_names}
        if not kept_terms:
            return Expression(S.Zero, self.name)
        kept_order = [n for n in self._tag_order if n not in group_names]
        new_expr = sum(kept_terms.keys(), S.Zero)
        return Expression(new_expr, self.name,
                          term_tags=kept_terms, tag_order=kept_order)

    # ------------------------------------------------------------------
    # Tagging: physical (display) and solver (operator routing)
    # ------------------------------------------------------------------

    def tag(self, **named_groups):
        """Attach physical tags to terms for display.

        Each `name=value` kwarg is split into additive terms and each
        term is recorded as belonging to `name` in the per-term tag map.

        Tags follow the terms through `.apply` / `.subs` / `.simplify`:
        when a transformation replaces a tagged term, the new term(s)
        inherit the tag.  Terms that cancel (via `simplify`) silently
        disappear from the view — no parallel-sum math, no cross-tag
        contamination.

        Usage:

        ```
            xmom = xmom.tag(pressure=grad_p, stress=div_tau)
        ```
        """
        new_tags = dict(self._term_tags)
        new_order = list(self._tag_order)
        for name_, v in named_groups.items():
            v_sp = v.expr if isinstance(v, Expression) else sp.sympify(v)
            for term in Add.make_args(sp.expand(v_sp)):
                if term != S.Zero:
                    new_tags[term] = name_
            if name_ not in new_order:
                new_order.append(name_)
        return Expression(self.expr, self.name,
                          term_tags=new_tags, tag_order=new_order,
                          solver_groups=self._solver_groups)

    @property
    def tags(self):
        """Dict of current-expr terms grouped by physical tag.

        Computed on demand by walking `Add.make_args(self.expr)` and
        assigning each term to the tag recorded in `_term_tags`.  Terms
        not in the main expression (cancelled by simplify, etc.) do not
        appear — the view stays in sync with the single source of truth.
        """
        if not self._term_tags:
            return {}
        # Group current expression terms by tag, preserving tag order.
        groups = {name: S.Zero for name in self._tag_order}
        for term in Add.make_args(sp.expand(self.expr)):
            tag_name = self._term_tags.get(term)
            if tag_name is None:
                continue
            if tag_name not in groups:
                groups[tag_name] = S.Zero
            groups[tag_name] = groups[tag_name] + term
        # Drop empty tags (their terms all cancelled).
        return {k: v for k, v in groups.items() if v != S.Zero}

    @property
    def untagged(self):
        """Sum of current-expr terms that have no physical tag."""
        tagged_terms = set(self._term_tags)
        return sum(
            (t for t in Add.make_args(sp.expand(self.expr)) if t not in tagged_terms),
            S.Zero,
        )

    def solver_tag(self, **named_groups):
        """Attach/replace solver tag groups for operator routing.

        Tag names are normalized via the solver-tag alias catalog, so
        `.solver_tag(convection=...)` and `.solver_tag(flux=...)` both
        write to the canonical `flux` slot.

        Solver tags are dropped per-tag whenever `apply()` / `simplify()`
        changes the tagged sub-expression (structural equality check).

        Usage:

        ```
            xmom = xmom.solver_tag(flux=F_x, source=g * h * dbdx)
        ```
        """
        from zoomy_core.model.derivation.tag_catalog import canonical_solver_tag

        merged = dict(self._solver_groups or {})
        for k, v in named_groups.items():
            canonical = canonical_solver_tag(k)
            merged[canonical] = v.expr if isinstance(v, Expression) else sp.sympify(v)
        return Expression(self.expr, self.name,
                          term_groups=self._term_groups, solver_groups=merged)

    @property
    def solver_tags(self):
        """Dict of solver tags (canonical names), or empty dict if none set."""
        return dict(self._solver_groups) if self._solver_groups else {}

    def get_solver_tag(self, name):
        """Return the sp.Expr for a solver tag, or None if not set.

        `name` can be any alias; it is normalized to canonical form.
        """
        from zoomy_core.model.derivation.tag_catalog import canonical_solver_tag

        if not self._solver_groups:
            return None
        return self._solver_groups.get(canonical_solver_tag(name))

    def untagged_remainder(self):
        """Return `self.expr - sum(solver_tags.values())`, simplified.

        Zero means every term has been routed to a solver tag; a non-zero
        remainder is what drives the model's `untagged_policy` (warn/error).
        """
        if not self._solver_groups:
            return self.expr
        tagged_sum = sum(self._solver_groups.values(), S.Zero)
        return sp.expand(self.expr - tagged_sum)

    def auto_solver_tag(self, *, state_vars, time_var, coords,
                        parameters=()):
        """Algorithmically assign solver tags by structural matching.

        Each additive term of `self.expr` is classified:

        * `coeff * Derivative(q, t)` with `q` a state variable →
          `time_derivative`.
        * `coeff * Derivative(F, x_i)` where `coeff` is state-variable-
          free → `flux` (conservative form).
        * `coeff * Derivative(q_j, x_i)` with `q_j` a state variable
          and `coeff` containing state vars → `nonconservative_flux`.
        * `coeff * Derivative(p, x_i)` where `p` is a declared
          parameter and `coeff` contains state vars → also
          `nonconservative_flux` (the term couples a state coefficient
          to a parameter gradient — e.g. bathymetry-gradient forcing).
        * No derivative and purely algebraic in ``state_vars ∪
          parameters`` space → `source`.
        * Anything else (unclosed fields like `u(t, x, z)`, unmatched
          Derivative shapes) stays in the untagged remainder.

        Parameters
        ----------
        state_vars : iterable of sympy Functions / Symbols
            The evolution variables (`α_k(t, x)`, interface heights,
            etc.).
        time_var : sympy Symbol
            The time coordinate (for `time_derivative` detection).
        coords : iterable of sympy Symbols
            Spatial coordinates (`x`, `y`, …) for flux / NCP.
        parameters : iterable, optional
            Known external Function/Symbol atoms (bathymetry `b(t, x)`,
            prescribed fluxes, etc.).  These may freely appear in any
            term without the term being marked "unclosed", and they may
            appear as the inner of a spatial Derivative in the NCP
            tagging.

        Returns a new Expression with `solver_groups` populated.
        Physical tags are preserved.
        """
        state_set = set(state_vars)
        parameters = set(parameters)
        allowed_fn_set = state_set | parameters
        coords_list = list(coords)
        tags = {
            "time_derivative": S.Zero,
            "flux": S.Zero,
            "nonconservative_flux": S.Zero,
            "source": S.Zero,
        }

        def _has_unclosed_function(term):
            """`True` iff the term contains a Function call that is
            neither a declared state variable nor a declared parameter.
            Those are the terms the auto-tagger can't classify safely;
            everything else (including Function atoms of state_vars /
            parameters — they're allowed to carry `(t, x)` args) is
            fair game.
            """
            return any(a not in allowed_fn_set and a.args
                       for a in term.atoms(Function))

        for term in Add.make_args(sp.expand(self.expr)):
            if term == S.Zero:
                continue

            if _has_unclosed_function(term):
                continue

            # Try to split as `coeff * Derivative(...)`.
            coeff = S.One
            deriv = None
            if isinstance(term, Derivative):
                deriv = term
            elif isinstance(term, Mul):
                ds = [a for a in term.args if isinstance(a, Derivative)]
                if len(ds) == 1:
                    deriv = ds[0]
                    coeff = Mul(*[a for a in term.args if a is not deriv])

            if deriv is not None and len(deriv.variables) == 1:
                v = deriv.variables[0]
                inner = deriv.args[0]

                # time_derivative: coeff * ∂_t(state_var_or_parameter).
                # Time-derivatives of parameters (e.g. moving bathymetry)
                # get lumped here too — solvers typically treat them as
                # time-varying forcing alongside the genuine ∂_t q terms.
                if v == time_var and inner in allowed_fn_set:
                    tags["time_derivative"] = tags["time_derivative"] + term
                    continue

                if v in coords_list:
                    # A coeff counts as "state-bearing" iff it contains a
                    # state-variable atom.  Function state variables
                    # (``α_k(t, x)``) don't show up in ``free_symbols``,
                    # so test via ``.atoms(Function) | .free_symbols``.
                    coeff_symbols = coeff.free_symbols | coeff.atoms(Function)
                    coeff_state_refs = state_set & coeff_symbols
                    if not coeff_state_refs:
                        tags["flux"] = tags["flux"] + term
                        continue
                    # coeff has state; treat any ∂_xᵢ(state_or_parameter) as NCP.
                    if inner in allowed_fn_set:
                        tags["nonconservative_flux"] = (
                            tags["nonconservative_flux"] + term)
                        continue

            has_derivative = any(
                isinstance(a, Derivative) for a in term.atoms(Derivative)
            )
            if not has_derivative:
                tags["source"] = tags["source"] + term
                continue

        # Drop empty tag groups.
        nonempty = {t: v for t, v in tags.items() if v != 0}
        return Expression(self.expr, self.name,
                          term_groups=self._term_groups,
                          solver_groups=nonempty)

    def __len__(self):
        return len(Add.make_args(sp.expand(self.expr)))

    def __iter__(self):
        return iter(self.terms)

    def __add__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr + other_expr, self.name)

    def __radd__(self, other):
        if other == 0:
            return self
        return Expression(other + self.expr, self.name)

    def __sub__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr - other_expr, self.name)

    def __neg__(self):
        return Expression(-self.expr, self.name)

    def __mul__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr * other_expr, self.name)

    def __rmul__(self, other):
        return Expression(other * self.expr, self.name)

    def project(self, test_func, var, domain=(0, 1), weight=S.One,
                scale=S.One, numerical=False, order=4):
        """Galerkin projection: integral(expr * test * weight * scale, (var, a, b))."""
        integrand = self.expr * test_func * weight * scale
        if numerical:
            result = gauss_legendre_integrate(integrand, var, domain[0], domain[1], order)
            return Expression(result, f"project_num({self.name})")
        return Expression(
            Integral(integrand, (var, domain[0], domain[1])),
            f"project({self.name})"
        )

    def ibp(self, var, test_weight, domain=(0, 1), scale=S.One):
        """
        Integration by parts on the outermost Derivative w.r.t. var.
        Returns IBPResult(integrate, boundary_upper, boundary_lower).
        """
        inner, coeff = _extract_derivative(self.expr, var)
        if inner is None:
            raise ValueError(
                f"No Derivative w.r.t. {var} found in: {self.expr}\n"
                f"Use .project() for terms without derivatives."
            )
        a, b = domain
        tw = test_weight * scale
        return IBPResult(
            integrate=Expression(
                -Integral(coeff * inner * Derivative(test_weight, var) * scale, (var, a, b)),
                f"ibp_integrate({self.name})"
            ),
            boundary_upper=Expression(
                (coeff * inner * tw).subs(var, b),
                f"ibp_upper({self.name})"
            ),
            boundary_lower=Expression(
                (coeff * inner * tw).subs(var, a),
                f"ibp_lower({self.name})"
            ),
        )

    def apply(self, *conditions, rhs=_UNSET):
        """Apply substitutions / Relations / Operations — per term.

        The transformation is applied to every additive term of
        `self.expr` individually.  Each output term inherits the tag of
        its parent term (follow-the-term), so the per-term tag dict stays
        in sync with the evolving main expression automatically.

        Shorthand: `expr.apply(lhs, rhs=0)` is equivalent to
        `expr.apply({lhs: 0})`.  Valid only with a single positional
        argument which is a sympy expression (or `Expression`).

        After Relation/dict-style applies the result is lightly simplified
        via `_simplify_preserve_integrals` (linearity-only derivatives,
        cancellations across the full expr).  Terms that cancel vanish
        from both `self.expr` and the tag dict.

        Operations are invoked via their unified `__call__` interface;
        when the op produces multiple output pieces per input term
        (`DepthIntegrate` etc.) all pieces inherit the parent tag.
        """
        if rhs is not _UNSET:
            if len(conditions) != 1:
                raise TypeError(
                    "apply(lhs, rhs=...) shorthand requires exactly one "
                    f"positional argument (got {len(conditions)})."
                )
            lhs = conditions[0]
            lhs_sp = lhs.expr if isinstance(lhs, Expression) else lhs
            conditions = ({lhs_sp: rhs},)

        # ``single_term_only`` Operations (e.g. ProductRule) are
        # invalid on multi-term expressions — the author must pick a
        # term explicitly via ``apply_to_term(idx, op)``.  Allowing
        # them at whole-leaf level would let the operation implicitly
        # decide which terms to rewrite, which is the anti-pattern.
        # Only single-term-only ops need the (potentially expensive) term count,
        # so compute it lazily — a full ``sp.expand`` of a huge rational source
        # (ν_t = C_μ sk⁴/se², …) just to count terms otherwise dominates the
        # build (multinomial-expanding sk⁴ — work GaussQuadrature does later).
        if any(isinstance(c, Operation) and getattr(c, "single_term_only", False)
               for c in conditions):
            n_terms = len(Add.make_args(sp.expand(self.expr)))
            if n_terms > 1:
                for cond in conditions:
                    if isinstance(cond, Operation) and getattr(
                            cond, "single_term_only", False):
                        raise RuntimeError(
                            f"{type(cond).__name__} is single-term-only and "
                            f"cannot be applied to a multi-term expression "
                            f"({n_terms} terms in {self.name!r}). Use "
                            f"expression.apply_to_term(idx, {type(cond).__name__}()) "
                            f"to target a specific term by index."
                        )

        # ``whole_leaf_op`` Operations need the whole leaf at once
        # (cross-term scanning, integral merging, ...).  Route those
        # via ``Operation.__call__`` directly — same dispatch path as
        # ``System.apply`` uses on a leaf — instead of the per-term
        # iteration below, which would fragment the integrand and
        # break cross-term merges (e.g. summing two boundary atoms
        # produced by the same divergence-theorem call).
        if (len(conditions) == 1
                and isinstance(conditions[0], Operation)
                and getattr(conditions[0], "whole_leaf_op", False)):
            return conditions[0](self)

        _resolve_subs = _resolve_subs_safe  # module-level helper

        def _apply_one(expr, cond):
            if isinstance(cond, Operation):
                result = cond(Expression(expr, self.name))
                if isinstance(result, Expression):
                    # Integrate with ``upper == var`` (running integral) produces
                    # trivial ``Subs(f, var, var)`` wrappers that downstream
                    # solve/substitute steps treat as opaque.  Resolve them now.
                    return _resolve_subs(result.expr)
                raise TypeError(
                    f"{type(cond).__name__} returned {type(result).__name__} "
                    f"from per-term apply; rank-changing Operations must be "
                    f"invoked at the tree level, not through Expression.apply."
                )
            # An ORIENTED Expression / Equation (or anything carrying a
            # ``{lhs: rhs}`` relation) acts as a substitution — this is the
            # single relation path now that Relation/Substitution are gone.
            rel = getattr(cond, "_as_relation", None)
            if isinstance(rel, dict) and rel:
                expr = _resolve_subs(expr)
                # ``_apply_relation`` substitutes a FUNCTION DEFINITION
                # ``f(…, ζ) = rhs`` at the head level (so ``ω(t,x,ζ)`` also
                # rewrites the ``ω(t,x,\hat ζ)`` carried inside a bracket) and
                # any other lhs by exact match — a plain ``eq.apply(m.omega)``
                # therefore eliminates the operator with no bolt-on op.
                return _resolve_subs(_apply_relation(expr, rel))
            if isinstance(cond, dict):
                expr = _resolve_subs(expr)
                # Same relation semantics for a raw ``{lhs: rhs}`` dict.  The
                # sympify inside ``_apply_relation`` keeps a raw ``0`` rhs
                # (``{∂_t b: 0}``) from leaking a Python ``int`` and breaking
                # downstream ``.has`` checks; exact lhs go through structural
                # ``xreplace`` (basis-expansion-safe for a ``ζ``-in-rhs case).
                return _resolve_subs(_apply_relation(expr, cond))
            elif isinstance(cond, (list, tuple)):
                if len(cond) == 2 and isinstance(cond[0], sp.Basic):
                    return expr.subs(cond[0], cond[1])
                for pair in cond:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        expr = expr.subs(pair[0], pair[1])
                return expr
            return expr

        # Per-term iteration with tag inheritance.
        #
        # For non-Operation conditions we run the lightweight simplify
        # (``_simplify_preserve_integrals``, linearity-only on
        # Derivatives) **per term** before splitting into sub-pieces.
        # This fragments things like ``Derivative(g*ρ*(η − z), x) / ρ``
        # into ``g·∂_x b + g·∂_x h`` before we record each piece under
        # the parent tag — otherwise the unfragmented form becomes an
        # orphan key when a later global simplify fragments it.
        simplifying = not any(isinstance(c, Operation) for c in conditions)
        new_tags = {}
        pieces = []
        for term in Add.make_args(_expand_preserve_integrals(self.expr)):
            if term == S.Zero:
                continue
            parent_tag = self._term_tags.get(term)
            result = term
            for cond in conditions:
                result = _apply_one(result, cond)
            if simplifying:
                result = _simplify_preserve_integrals(result)
            for sub in Add.make_args(_expand_preserve_integrals(result)):
                if sub == S.Zero:
                    continue
                pieces.append(sub)
                if parent_tag is not None:
                    new_tags[sub] = parent_tag

        new_expr = sum(pieces, S.Zero)

        # Global simplify catches cross-term cancellations (e.g. the
        # ``u|_η · ∂_t b`` pieces from two different tags that sum to
        # zero after kinematic BCs).  Surviving terms keep their tags;
        # cancelled ones silently disappear.
        if simplifying:
            new_expr = _simplify_preserve_integrals(new_expr)
            current = set(Add.make_args(_expand_preserve_integrals(new_expr)))
            new_tags = {t: name for t, name in new_tags.items() if t in current}

        # Solver tags: drop any whose stored expression was touched by
        # this transformation.
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {}
            for tag, group_expr in self._solver_groups.items():
                g = group_expr
                for cond in conditions:
                    g = _apply_one(g, cond)
                if g == group_expr:
                    new_solver_groups[tag] = group_expr

        return Expression(new_expr, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def expand(self):
        """Return a new Expression with sympy expand applied."""
        expanded = sp.expand(self.expr)
        current = set(Add.make_args(expanded))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if sp.expand(g) == g}
        return Expression(expanded, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def copy(self):
        """Return an independent Expression with the same expr, tags, relation."""
        clone = Expression(self.expr, self.name,
                           term_tags=dict(self._term_tags),
                           tag_order=list(self._tag_order),
                           solver_groups=(dict(self._solver_groups)
                                          if self._solver_groups else None))
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            clone._as_relation = dict(rel)
        return clone

    def solve_for(self, variable):
        """Return a new Expression whose `_as_relation` maps `variable`
        to the solution of `self.expr == 0`.

        Pipeline-style counterpart to `_NodeProxy.solve_for`: does **not**
        mutate a system tree.  Downstream `model.apply(result)` picks up
        the relation and performs the substitution.
        """
        solutions = sp.solve(self.expr, variable)
        if not solutions:
            raise ValueError(f"Cannot solve {self.name or self.expr!r} for {variable}")
        if len(solutions) > 1:
            warnings.warn(
                f"Multiple solutions for {variable}, using first: {solutions[0]}"
            )
        solution = solutions[0]
        isolated = Expression(variable - solution, self.name,
                              term_tags=dict(self._term_tags),
                              tag_order=list(self._tag_order),
                              solver_groups=self._solver_groups)
        isolated._as_relation = {variable: solution}
        return isolated

    def depth_integrate(self, lower, upper, var, method="auto"):
        """Depth-integrate this expression over `[lower, upper]` w.r.t. `var`.

        Args:
            lower, upper: Integration bounds (typically b and b+H, both functions of t and x).
            var: The vertical coordinate z.
            method: `'auto'` detects the derivative direction; `'leibniz'` pulls a horizontal derivative outside; `'fundamental_theorem'` handles vertical derivatives; `'direct'` keeps `Integral(expr, (var, lower, upper))`.
        Returns:
            `DepthIntegralResult` with `.volume`, `.boundary_upper`, `.boundary_lower`, or a plain `Expression` for `'direct'` and `'fundamental_theorem'`.
        """
        expr = self.expr

        if method == "auto":
            # Detect: does the expression contain d/dz?
            inner_z, coeff_z = _extract_derivative(expr, var)
            if inner_z is not None:
                method = "fundamental_theorem"
            else:
                # Check for d/dx (or any horizontal derivative)
                method = "direct"
                for s in expr.free_symbols:
                    if s != var:
                        inner_h, coeff_h = _extract_derivative(expr, s)
                        if inner_h is not None:
                            method = "leibniz"
                            break

        # Running integrals have ``upper == var``.  In that case the outer
        # ``var`` is also the integration variable, which sympy accepts as
        # shadowing but which downstream tools struggle with.  Rebind the
        # integration variable to a fresh ``\hat{<var>}`` Dummy so the
        # integral is cleanly bound.  The outer ``var`` remains free,
        # available for later ζ-transform / basis expansion.
        if upper == var:
            bound_var = sp.Dummy(rf"\hat{{{var}}}", positive=True)
            expr_bound = expr.subs(var, bound_var)
        else:
            bound_var = var
            expr_bound = expr

        if method == "fundamental_theorem":
            # int df/dz dz = f(upper) - f(lower)
            inner, coeff = _extract_derivative(expr_bound, bound_var)
            if inner is None:
                raise ValueError(
                    f"No Derivative w.r.t. {var} found for fundamental theorem: {expr}"
                )
            f = coeff * inner
            f_upper = sp.Subs(f, bound_var, upper)
            f_lower = sp.Subs(f, bound_var, lower)
            return DepthIntegralResult(
                volume=Expression(S.Zero, f"ft_volume({self.name})"),
                boundary_upper=Expression(f_upper, f"ft_upper({self.name})"),
                boundary_lower=Expression(-f_lower, f"ft_lower({self.name})"),
            )

        elif method == "leibniz":
            # int df/dx dz = d/dx[int f dz] - f(upper)*d(upper)/dx + f(lower)*d(lower)/dx
            for s in list(expr_bound.free_symbols) + [Symbol("x"), Symbol("t")]:
                if s == bound_var or s == var:
                    continue
                inner, coeff = _extract_derivative(expr_bound, s)
                if inner is not None:
                    break
            else:
                raise ValueError(f"No horizontal Derivative found for Leibniz: {expr}")

            # Volume: d/dx[int f dz]
            int_f = Integral(coeff * inner, (bound_var, lower, upper))
            volume = Derivative(int_f, s)

            # Boundary terms: use Subs to keep evaluation explicit
            f = coeff * inner
            f_at_upper = sp.Subs(f, bound_var, upper)
            f_at_lower = sp.Subs(f, bound_var, lower)
            bnd_upper = -f_at_upper * Derivative(upper, s)
            bnd_lower = f_at_lower * Derivative(lower, s)

            return DepthIntegralResult(
                volume=Expression(volume, f"leibniz_volume({self.name})"),
                boundary_upper=Expression(bnd_upper, f"leibniz_upper({self.name})"),
                boundary_lower=Expression(bnd_lower, f"leibniz_lower({self.name})"),
            )

        else:  # direct
            return Expression(
                Integral(expr_bound, (bound_var, lower, upper)),
                f"integral({self.name})",
            )

    def subs(self, *args, **kwargs):
        # Per-term substitution with tag inheritance, then cleanup.
        new_tags = {}
        pieces = []
        for term in Add.make_args(_expand_preserve_integrals(self.expr)):
            if term == S.Zero:
                continue
            parent_tag = self._term_tags.get(term)
            sub = term.subs(*args, **kwargs)
            for piece in Add.make_args(_expand_preserve_integrals(sub)):
                if piece == S.Zero:
                    continue
                pieces.append(piece)
                if parent_tag is not None:
                    new_tags[piece] = parent_tag
        new_expr = sum(pieces, S.Zero)
        current = set(Add.make_args(_expand_preserve_integrals(new_expr)))
        new_tags = {t: name for t, name in new_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if g.subs(*args, **kwargs) == g}
        return Expression(new_expr, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def split_integrals(self):
        """Distribute each `Integral(Add(...), lim)` into a sum of
        single-term Integrals, recursing through any `Derivative`
        wrapper.  Returns a new Expression whose `.terms` exposes one
        logical equation term per Integral.
        """
        new_expr = _split_integrals_expr(self.expr)
        if new_expr is self.expr:
            return self
        out = Expression(new_expr, self.name,
                         term_tags=dict(self._term_tags),
                         tag_order=list(self._tag_order),
                         solver_groups=self._solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def merge_integrals(self):
        """Group sibling Integrals with matching `(limits, deriv-wrapper)`
        signature into a single `Integral(Σ c·f, lim)`.  Inverse of
        `split_integrals`; useful when cross-integrand
        cancellations need sympy's `Add` canonicalisation to see them
        in one integrand.
        """
        new_expr = _merge_integrals_expr(self.expr)
        if new_expr is self.expr:
            return self
        out = Expression(new_expr, self.name,
                         term_tags=dict(self._term_tags),
                         tag_order=list(self._tag_order),
                         solver_groups=self._solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def simplify(self):
        """Simplify: expand + cancel, preserving Integral and Derivative(Integral) terms."""
        simplified = _simplify_preserve_integrals(self.expr)
        current = set(Add.make_args(sp.expand(simplified)))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if _simplify_preserve_integrals(g) == g}
        out = Expression(simplified, self.name,
                         term_tags=new_tags, tag_order=self._tag_order,
                         solver_groups=new_solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def expand(self):
        expanded = sp.expand(self.expr)
        current = set(Add.make_args(expanded))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if sp.expand(g) == g}
        out = Expression(expanded, self.name,
                         term_tags=new_tags, tag_order=self._tag_order,
                         solver_groups=new_solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def doit(self):
        return Expression(self.expr.doit(), self.name)

    def has(self, *args):
        return self.expr.has(*args)

    def atoms(self, *types):
        return self.expr.atoms(*types)

    def xreplace(self, rule):
        return Expression(self.expr.xreplace(rule), self.name,
                          term_tags=self._term_tags,
                          tag_order=self._tag_order)

    @property
    def free_symbols(self):
        return self.expr.free_symbols

    def __repr__(self):
        short = str(self.expr)
        if len(short) > 80:
            short = short[:77] + "..."
        label = f" ({self.name})" if self.name else ""
        return f"Expression{label}: {short}"

    def _repr_latex_(self):
        return f"${self.latex(strip_args=True)}$"

    def _sympy_(self):
        return self.expr

    def __eq__(self, other):
        if isinstance(other, Expression):
            return sp.simplify(self.expr - other.expr) == 0
        return sp.simplify(self.expr - other) == 0

    # ------------------------------------------------------------------
    # Per-term operations
    # ------------------------------------------------------------------

    def map(self, fn):
        """Apply fn to each term, reassemble into a single Expression.

        fn receives an Expression (single term) and must return either
        an Expression or a DepthIntegralResult.  DepthIntegralResults are
        assembled (volume + boundaries) before summing.

        Example:
            integrated = expr.map(lambda t: t.depth_integrate(b, eta, z))
        """
        results = []
        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                results.append(r.assemble())
            elif isinstance(r, Expression):
                results.append(r)
            else:
                results.append(Expression(r, term.name))
        return sum(results, Expression(S.Zero))

    def map_with_bcs(self, fn, bcs):
        """Like map(), but collects boundary terms and applies BCs globally.

        This is the correct way to depth-integrate a full equation:
        boundary terms from ALL terms are combined first, then BCs are
        applied once (so cross-term cancellations happen properly).

        Parameters
        ----------
        fn : callable
            Applied to each term.  Must return DepthIntegralResult or Expression.
        bcs : list of Relation
            Kinematic BCs etc. applied to the combined boundary expression.

        Returns
        -------
        Expression
            The fully depth-integrated equation with BCs applied.
        """
        total_volume = Expression(S.Zero)
        total_boundary = Expression(S.Zero)

        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                total_volume = total_volume + r.volume
                total_boundary = total_boundary + r.boundary_upper + r.boundary_lower
            elif isinstance(r, Expression):
                total_volume = total_volume + r
            else:
                total_volume = total_volume + Expression(r)

        # Apply all BCs to the combined boundary
        bnd = total_boundary
        for bc in bcs:
            bnd = bnd.apply(bc)

        # Simplify: evaluate derivatives (to combine d(H+b)/dt - db/dt → dH/dt)
        # but preserve Integrals (don't re-apply Leibniz)
        result_expr = (total_volume + bnd).expr
        result_expr = _simplify_derivatives_only(result_expr)

        return Expression(result_expr, self.name)

    # ------------------------------------------------------------------
    # Term classification
    # ------------------------------------------------------------------

    def classify(self, t=None, x=None, z=None):
        """Classify each term by its role in the PDE.

        Returns a dict: {role: Expression} where role is one of:
        'temporal', 'convective', 'diffusive', 'source'.

        Detection rules:
        - Has d/dt → temporal
        - Has d/dx (first-order) of a product → convective flux
        - Has d²/dz² or d/dz of d/dz → diffusive
        - Otherwise → source (algebraic)
        """
        roles = {
            "temporal": [],
            "convective": [],
            "diffusive": [],
            "source": [],
        }

        for term in self.terms:
            e = term.expr
            classified = False

            # Check temporal
            if t is not None and e.has(Derivative) and any(
                t in d.variables for d in e.atoms(Derivative)
            ):
                roles["temporal"].append(term)
                classified = True

            # Check diffusive (second derivatives in z)
            if not classified and z is not None:
                for d in e.atoms(Derivative):
                    if d.variables.count(z) >= 2:
                        roles["diffusive"].append(term)
                        classified = True
                        break

            # Check convective (first derivative in x)
            if not classified and x is not None:
                for d in e.atoms(Derivative):
                    if x in d.variables and d.variables.count(x) == 1:
                        roles["convective"].append(term)
                        classified = True
                        break

            if not classified:
                roles["source"].append(term)

        return {k: Expression(sum((t.expr for t in v), S.Zero), k)
                for k, v in roles.items() if v}

    @property
    def temporal(self):
        """View: only temporal (d/dt) terms."""
        c = self.classify(t=Symbol("t"))
        return c.get("temporal", Expression(S.Zero))

    @property
    def convective(self):
        """View: only convective flux (d/dx) terms."""
        c = self.classify(x=Symbol("x"))
        return c.get("convective", Expression(S.Zero))

    # ------------------------------------------------------------------
    # Basis projection
    # ------------------------------------------------------------------

    def project_onto_basis(self, basis, level, field_map, z_var,
                           lower=None, upper=None, test_mode=None):
        """
        Project a depth-integrated equation onto a polynomial basis.

        Replaces every `Integral(f(u,...), (z, b, eta))` by substituting
        the basis expansion `u(z) = sum alpha_k phi_k(zeta)` and evaluating
        the resulting integrals using the `SymbolicIntegrator`.

        If `test_mode` is an integer, multiplies each integral by the test
        function phi_{test_mode}(zeta) before evaluating (Galerkin projection
        for a specific mode).  If `test_mode=None`, returns the scalar
        integral (e.g. for the mass equation where no test function is needed).

        Parameters
        ----------
        basis : Basisfunction class (e.g. Legendre_shifted)
        level : int
        field_map : dict
            Maps the original Function name to a list of SymPy Symbols
            for the basis coefficients.
            Example: {'u': [alpha_0, alpha_1, alpha_2]}
        z_var : Symbol
            The vertical coordinate (z) that appears in the integrals.
        lower, upper : sympy expressions (optional)
            The integration bounds (b, eta).  If None, detected from
            the first Integral found.
        test_mode : int or None
            If int, project onto test function phi_{test_mode}.

        Returns
        -------
        Expression
            With all depth integrals replaced by basis matrix products.
        """
        from zoomy_core.model.derivation.basis_cache import get_basis_matrices

        basis_obj = basis(level=level)
        matrices = get_basis_matrices(basis_obj, level)

        M = matrices["M"]
        A = matrices["A"]
        n = level + 1
        zeta = Symbol("zeta")
        c_mean = basis_obj.mean_coefficients()

        def _replace_integral(expr):
            """Walk the expression tree, replacing Integral nodes."""
            if not isinstance(expr, sp.Basic):
                return expr

            if isinstance(expr, Integral):
                integrand = expr.args[0]
                limits = expr.args[1]
                int_var = limits[0]

                if int_var != z_var:
                    return expr

                lo, hi = limits[1], limits[2]

                # Transform to zeta-space:
                # z = lo + (hi - lo)*zeta, dz = (hi-lo)*dzeta
                h_expr = hi - lo  # water depth H
                integrand_zeta = integrand.subs(int_var, lo + h_expr * zeta)

                # Substitute basis expansion for each field
                for fname, coeffs in field_map.items():
                    expansion = sum(
                        coeffs[k] * basis_obj.eval(k, zeta)
                        for k in range(min(len(coeffs), n))
                    )
                    # Find all applications of this function and replace
                    for atom in integrand_zeta.atoms(sp.Function):
                        if atom.func.__name__ == fname:
                            integrand_zeta = integrand_zeta.subs(atom, expansion)

                # Multiply by Jacobian H and test function
                integrand_final = h_expr * integrand_zeta
                if test_mode is not None:
                    integrand_final *= basis_obj.eval(test_mode, zeta)

                # Evaluate the integral using the integrator
                result = integrator.integrate(
                    sp.expand(integrand_final) * basis_obj.weight(zeta),
                    zeta,
                    tuple(basis_obj.bounds()),
                )
                return result

            # Recurse into Derivative, Mul, Add, etc.
            if isinstance(expr, Derivative):
                new_expr = _replace_integral(expr.args[0])
                return Derivative(new_expr, *expr.args[1:])

            if expr.args:
                new_args = [_replace_integral(a) for a in expr.args]
                return expr.func(*new_args)

            return expr

        result = _replace_integral(self.expr)
        return Expression(result, f"projected({self.name})")

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def latex(self, strip_args=False, multiline=False):
        """LaTeX representation.

        Parameters
        ----------
        strip_args : bool
            `u(t,x,z)` → `u`. Partial derivatives preserved.
        multiline : bool
            Render as `\\begin{aligned}` with one group per line
            (requires `term_groups`).
        """
        printer = _StripArgsLatexPrinter() if strip_args else None

        def _tex(expr):
            return printer.doprint(expr) if printer else sp.latex(expr)

        if multiline and self._term_tags:
            # Group current-expression terms by their tag.
            current_terms = list(Add.make_args(sp.expand(self.expr)))
            by_tag = {name: [] for name in self._tag_order}
            untagged = []
            for term in current_terms:
                name = self._term_tags.get(term)
                if name is None:
                    untagged.append(term)
                else:
                    by_tag.setdefault(name, []).append(term)
            lines = []
            first = True
            for role in list(by_tag.keys()):
                terms = by_tag.get(role, [])
                if not terms:
                    continue
                g = sum(terms, S.Zero)
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if first:
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                    first = False
                elif tex.startswith("-"):
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                else:
                    lines.append(f"  & + \\underbrace{{{tex}}}_{{{role}}}")
            if untagged:
                g = sum(untagged, S.Zero)
                if g != S.Zero:
                    tex = _tex(g)
                    prefix = "" if first or tex.startswith("-") else "+ "
                    lines.append(f"  & {prefix}\\underbrace{{{tex}}}_{{untagged}}")
            return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if multiline and not self._term_groups:
            # No term groups — split by additive terms for multiline rendering
            terms = Add.make_args(self.expr)
            if len(terms) > 1:
                lines = []
                for i, term in enumerate(terms):
                    tex = _tex(term)
                    if i == 0:
                        lines.append(f"  & {tex}")
                    elif tex.startswith("-"):
                        lines.append(f"  & {tex}")
                    else:
                        lines.append(f"  & + {tex}")
                return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if self._term_groups:
            # Render in group order (single line) — preserves physical ordering
            parts = []
            for role, g in self._term_groups.items():
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if parts and not tex.startswith("-"):
                    tex = "+ " + tex
                parts.append(tex)
            return " ".join(parts)

        return _tex(self.expr)

    def describe(self, header=True, final_equation=True, parameters=False,
                 strip_args=True):
        """Composable description of this expression.

        Returns a `Description` that renders as markdown in Jupyter.

        Parameters
        ----------
        header : bool
            Show expression name + term count.
        final_equation : bool
            Show the symbolic equation.
        parameters : bool
            List free symbols.
        strip_args : bool
            Display `u` instead of `u(t, x, z)`.
        """
        from zoomy_core.misc.description import Description

        parts = []

        if header:
            parts.append(f"**{self.name}** ({len(self)} terms)")

        if final_equation and self._as_relation:
            # ORIENTED: render the substitution rules ``lhs = rhs`` (one per
            # line) rather than the ``residual = 0`` form — this is the
            # ``{a: b}`` view of a solved equation / kinematic BC.
            _printer = _StripArgsLatexPrinter() if strip_args else None
            _tex = (lambda e: _printer.doprint(e)) if _printer else sp.latex
            rules = " \\\\ ".join(
                f"{_tex(lhs)} = {_tex(rhs)}"
                for lhs, rhs in self._as_relation.items())
            parts.append(f"\n$$\n\\begin{{aligned}} {rules} \\end{{aligned}}\n$$")
        elif final_equation:
            # Match System.describe: use multiline underbrace rendering when
            # term_groups are populated (otherwise model.describe and
            # model.<eq>.describe show the same equation differently).
            tex = self.latex(strip_args=strip_args,
                             multiline=bool(self._term_groups))
            # latex(multiline=True) already emits a trailing "&= 0" inside the
            # aligned block, so in that case we don't append another "= 0".
            if self._term_groups:
                parts.append(f"\n$$\n{tex}\n$$")
            else:
                parts.append(f"\n$$\n{tex} = 0\n$$")

        if parameters:
            from sympy import Symbol
            syms = sorted([s for s in self.expr.free_symbols
                          if isinstance(s, Symbol) and not s.is_Function],
                         key=str)
            if syms:
                sym_str = ", ".join(f"${sp.latex(s)}$" for s in syms)
                parts.append(f"\n**Parameters:** {sym_str}")

        return Description("\n".join(parts))


from sympy.printing.latex import LatexPrinter as _LatexPrinter


def _append_subscript(tex_name: str, idx_tex: str) -> str:
    """Append `_{idx_tex}` to `tex_name` without producing an invalid
    double-subscript form.

    sympy's `_deal_with_super_sub` translates a name like `"phi_p"` into
    `\\phi_{p}`.  Naively appending `_{0}` then yields `\\phi_{p}_{0}`
    — KaTeX raises *Double subscript* and the equation fails to render.
    This helper detects an existing trailing `_{...}` group on `tex_name`
    and merges the new index into it as `\\phi_{p,0}`.
    """
    if tex_name.endswith("}"):
        depth = 0
        for i in range(len(tex_name) - 1, -1, -1):
            ch = tex_name[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    if i >= 1 and tex_name[i - 1] == "_":
                        existing = tex_name[i + 1:-1]
                        base = tex_name[:i - 1]
                        return f"{base}_{{{existing},{idx_tex}}}"
                    break
    return f"{tex_name}_{{{idx_tex}}}"


# ── Galerkin-bracket predicate (single source of truth) ────────────────────


def is_bracket_body(integrand, var):
    """`True` iff `integrand` is a Galerkin BRACKET body — a pure function of
    the integration variable `var`, depending on NOTHING else (integer mode
    indices `i`/`k` excepted, since they parametrise the bracket).

    This is THE definition of "this ∫ is a bracket" and the single source of
    truth reused everywhere it is needed:

    * the LaTeX printer — render `∫_0^1 (…) dζ` as `⟨…⟩` only when this holds;
    * `ExtractBrackets` — the naming
      gate (a `t`/`x`-bearing body is not yet a bracket);
    * the bracket-quadrature step that turns `⟨…⟩` into NUMBERS — pick the
      integrals to evaluate via the basis instead of sympy.

    A factor that carries any non-index continuous symbol (`a_i(t,x)`, `h`,
    `τ̃`, `ω`, a parameter `g`) disqualifies it."""
    return all(s == var or getattr(s, "is_integer", False)
               for s in integrand.free_symbols)


def is_bracket(integral):
    """`True` iff `integral` is a Galerkin BRACKET: an `sp.Integral` over a
    CONSTANT interval whose body satisfies `is_bracket_body` — hence a pure
    NUMBER, evaluable by basis quadrature.  A running `∫_0^ζ` (variable upper
    limit) is therefore NOT a bracket; it is a ζ-function that may sit *inside*
    one."""
    if not (isinstance(integral, Integral)
            and len(integral.limits) == 1 and len(integral.limits[0]) == 3):
        return False
    var, lo, hi = integral.limits[0]
    return (getattr(lo, "is_number", False) and getattr(hi, "is_number", False)
            and is_bracket_body(integral.function, var))


class _BracketLatexPrinter(_LatexPrinter):
    """LaTeX printer that parenthesises a COMPOUND differentiation argument so a
    conservative flux `∂_x(h·b)` is visually distinct from the
    non-conservative `∂_x(h)·b`.  sympy's default prints both as `∂_x h b`
    (no parens), which is ambiguous; here `∂_x(h·b)` keeps its bracket while a
    bare `∂_x h` (single factor) stays bracket-free.  Base for the
    arg-stripping printer so the rule applies with or without `strip_args`."""

    def _print_Derivative(self, expr):
        # PRIME notation for a derivative of a UNIVARIATE function — one whose
        # only continuous dependence is the differentiation variable (mode
        # indices excepted): ``∂_ζ φ_i → φ_i'``, ``∂_ζ(c φ_k) → (c φ_k)'``,
        # ``∂_ζ² φ_i → φ_i''``.  A multivariate field (``∂_x h``, ``∂_ζ ũ``)
        # keeps the ``∂`` form — there the variable matters.
        variables = expr.variables
        if len(set(variables)) == 1:
            v = variables[0]
            arg = expr.expr
            if (v in arg.free_symbols
                    and all(s == v or getattr(s, "is_integer", False)
                            for s in arg.free_symbols)):
                inner_tex = self._print(arg)
                if isinstance(arg, (sp.Add, sp.Mul)):
                    inner_tex = r"\left(" + inner_tex + r"\right)"
                n = len(variables)
                primes = "'" * n if n <= 3 else r"^{(%d)}" % n
                return inner_tex + primes
        tex = super()._print_Derivative(expr)
        inner = expr.expr
        if isinstance(inner, (sp.Add, sp.Mul)):
            inner_tex = self._print(inner)
            if tex.endswith(inner_tex):
                return (tex[:-len(inner_tex)]
                        + r"\left(" + inner_tex + r"\right)")
        return tex

    def _is_bracket_factor(self, a):
        """A factor that renders as a Galerkin bracket `⟨…⟩` — a named
        `Gram`/`Weight` atom, a pure-ζ `∫_0^1` Integral, or a power of
        either."""
        if isinstance(a, Integral):
            return is_bracket(a)
        if isinstance(a, sp.Function) and getattr(a.func, "_is_bracket", False):
            return True
        if isinstance(a, sp.Pow):
            return self._is_bracket_factor(a.base)
        return False

    def _print_Mul(self, expr):
        # Render bracket factors (the pure-number ``⟨…⟩``) LAST in a product, so
        # a term reads "coefficients … ⟨bracket⟩" rather than burying the bracket
        # mid-term (``Σ h a_j ⟨…⟩ a_i`` → ``Σ h a_j a_i ⟨…⟩``).  Order-preserving
        # (the ``order='none'`` printer keeps ``expr.args`` order), display-only.
        if isinstance(expr, sp.Mul):
            args = list(expr.args)
            bset = [i for i, a in enumerate(args) if self._is_bracket_factor(a)]
            if bset and len(bset) < len(args):
                drop = set(bset)
                rest = [a for i, a in enumerate(args) if i not in drop]
                brk = [args[i] for i in bset]
                return super()._print_Mul(sp.Mul(*rest, *brk, evaluate=False))
        return super()._print_Mul(expr)

    def _print_Function(self, expr, exp=None):
        """An INTEGER (or integer-symbol) FIRST argument renders as a subscript
        on the name: `phi(k, ζ) → φ_k(ζ)`, `phi(1, ζ) → φ_1(ζ)`,
        `a(i, t, x) → a_i(t, x)`.  A modal index `k`/`i`/`l` carries the
        `integer` assumption; coordinates (`t`, `x`, `ζ`) do not, so
        ordinary fields like `u(t, x, z)` are unaffected.  Rendering only —
        the sympy object is unchanged."""
        args = expr.args
        if args and getattr(args[0], "is_integer", False):
            name = self._deal_with_super_sub(expr.func.__name__)
            base = _append_subscript(name, self._print(args[0]))
            rest = args[1:]
            if rest:
                base += (r"\left(" + ", ".join(self._print(a) for a in rest)
                         + r"\right)")
            return f"{base}^{{{exp}}}" if exp is not None else base
        return super()._print_Function(expr, exp=exp)

    @staticmethod
    def _is_canonical_zeta(bar_value):
        """True for the reference-element ζ-symbol, the canonicalised
        `\\hat{ζ}` integration Dummy, *or* the opaque `state.zeta(...)`
        Function-call head — all three name the σ reference coordinate and so
        mark a definite `∫_0^1` as a Galerkin bracket."""
        if bar_value == sp.Symbol("zeta", real=True):
            return True
        if isinstance(bar_value, sp.Dummy):
            return True
        if (isinstance(bar_value, sp.Function)
                and getattr(bar_value.func, "__name__", "") == "zeta"):
            return True
        return False

    def _print_Integral(self, expr):
        # A definite ``∫_0^1 (…) dζ`` over ζ / the canonicalised ``\hat{ζ}``
        # dummy is rendered as the Galerkin bracket ``⟨ … ⟩`` ONLY when its body
        # is purely ζ-dependent (a computable number).  An integrand that still
        # carries a ``t``/``x``-dependent factor is NOT a bracket and falls back
        # to the default ``∫ … d…`` — so the bracket notation never lies about a
        # term that has not yet had its coefficients pulled out.
        if (len(expr.limits) == 1 and len(expr.limits[0]) == 3
                and expr.limits[0][1] == sp.S.Zero
                and expr.limits[0][2] == sp.S.One
                and self._is_canonical_zeta(expr.limits[0][0])
                and is_bracket_body(expr.function, expr.limits[0][0])):
            body = expr.function.subs(expr.limits[0][0],
                                      sp.Symbol("zeta", real=True))
            inner = _StripArgsLatexPrinter(
                settings={"order": self._settings.get("order", "none")}
            ).doprint(body)
            return r"\langle %s \rangle" % inner
        return super()._print_Integral(expr)


class _StripArgsLatexPrinter(_BracketLatexPrinter):
    """LaTeX printer for function calls.

    One unified rule.  Each function shape has a *canonical* call —
    a tuple of "natural" arguments where the call should be displayed
    bare (just the function name, no parentheses).  When the actual
    call deviates in its **last** argument only, the deviation is
    rendered as a *restriction* `f|_{var=value}` (or, for basis
    atoms, `f|_{value}` without the variable label, since ζ vs z is
    just a pre/post-AffineProjection convention).  Anything else
    falls through to the full `f(args)` form.

    Shapes recognised:

    ============== ============== ================================
    arity          canonical      restriction-bar variable
    ============== ============== ================================
    `f(t,x)`     `(t, x)`     *none — purely horizontal, strip*
    `f(t,x,y)`   `(t, x, y)`  *none — 3D horizontal, strip*
    `f(t,x,z)`   `(t, x, z)`  `z`  — vertical / 2D vertical
    `f(t,x,y,z)` `(t,x,y,z)`  `z` — vertical / 3D vertical
    `f(t,x,ζ)`   `(t, x, ζ)`  `ζ` — **σ-mapped**, head decorated `\\tilde f`
    `f(t,x,y,ζ)` `(t,x,y,ζ)`  `ζ` — σ-mapped, 3D
    `f(ζ) / f(z)` `(ζ,)` or `(z,)` — basis test functions
                                  (bar variable elided)
    ============== ============== ================================

    Examples (in 2D):

    ```
        u(t, x, z)         →  u
        u(t, x, b+h)       →  u|_{z=b+h}
        u(t, x, zeta)      →  \\tilde{u}                            (after SigmaTransform)
        u(t, x, 0)         →  \\tilde{u}|_{ζ=0}                     (bottom value, σ coords)
        u(t, x, 1)         →  \\tilde{u}|_{ζ=1}                     (top value, σ coords)
        b(t, x)            →  b
        phi_0(zeta)        →  phi_0
        phi_0(z)           →  phi_0
        phi_0(0)           →  phi_0|_{0}
        phi_0(b+h)         →  phi_0|_{b+h}
        phi_0((z-b)/h)     →  phi_0|_{(z-b)/h}
        Subs(u(t,x,z),z,b) →  u|_{z=b}        (via sympy's Subs printer)
    ```

    The σ-mapped (tilde) detection fires when the last argument
    structurally contains the canonical `ζ` symbol (i.e. the field
    has been transported across `SigmaTransform`).  Functions
    that still depend on the physical `z` render bare-or-restricted
    as before.  Numeric `0`/`1` in the last slot is ambiguous from
    args alone; we treat them as σ-mapped boundary values when
    rendered after a σ-transform-aware derivation (the common case
    when this shape appears).  Mixed-coord derivations should make
    the bar variable explicit at the call site.
    """

    _t = sp.Symbol("t", real=True)
    _x = sp.Symbol("x", real=True)
    _y = sp.Symbol("y", real=True)
    _z = sp.Symbol("z", real=True)
    _zeta = sp.Symbol("zeta", real=True)

    def _print_Subs(self, subs):
        """KaTeX-friendly `Subs(expr, vars, vals)` rendering.

        Sympy's default uses `\\substack{ var=val }`, which KaTeX
        (Jupyter / VSCode) does not implement.  For the single-variable
        case we drop `\\substack` entirely and render
        `\\left.expr\\right|_{var=val}`; for multi-variable we fall
        back to a comma-joined subscript still without `\\substack`.
        """
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_subs = ", ".join(
            f"{self._print(o)}={self._print(n)}"
            for o, n in zip(old, new)
        )
        return r"\left. %s \right|_{%s}" % (latex_expr, latex_subs)

    def _print_Function(self, expr, exp=None):
        # Custom-rendering atoms (e.g. ``BoundaryIntegral`` /
        # ``NormalVector`` from ``zoomy_core.symbolic.domains``) attach
        # a ``_latex`` method to their manufactured Function subclass.
        # Defer to it before any of the project-specific shape rules
        # fire, otherwise generic 1-arg ``f(i)`` calls (like a normal
        # vector component ``n(0)``) would be misinterpreted as
        # basis-style restrictions ``f|_{ζ=0}``.
        custom_latex = expr.func.__dict__.get("_latex")
        if custom_latex is not None:
            return custom_latex(expr, self, exp=exp)
        # Coefficient-lookup Functions produced by ``Expand`` carry a
        # ``_coeff_table`` class attribute and a ``_user_name`` (the
        # name the user gave the family — ``U`` for ``[U_0, U_1, …]``).
        # Render them as ``U_{k}`` rather than the default 1-arg
        # basis-bar form ``coeff|_{ζ=k}``.
        if (hasattr(expr.func, "_coeff_table")
                and len(expr.args) == 1):
            user_name = getattr(expr.func, "_user_name",
                                expr.func.__name__)
            tex_name = self._deal_with_super_sub(user_name)
            tex_idx = self._print(expr.args[0])
            base = _append_subscript(tex_name, tex_idx)
            if exp is not None:
                base = f"{base}^{{{exp}}}"
            return base

        # Basis-function calls produced by Expand and Multiply carry a
        # ``_basis`` class attribute (the back-ref to the
        # ``Basisfunction`` instance).  The 2-arg signature is
        # ``phi_fn(k, ζ_arg)`` — the index ``k`` becomes a subscript on
        # the function name (``phi_k``), and the second arg follows the
        # standard ζ bar-form convention (canonical ``ζ`` → strip;
        # ``0``/``1``/``(z-b)/h`` → ``phi_k|_{ζ=…}``).  Multiplicative
        # exponent is preserved.
        if (hasattr(expr.func, "_basis")
                and len(expr.args) == 2):
            tex_name = self._deal_with_super_sub(expr.func.__name__)
            tex_idx = self._print(expr.args[0])
            subscripted = _append_subscript(tex_name, tex_idx)
            bar_value = expr.args[1]
            # ``Dummy`` args appear when ``Subs`` is binding the bar
            # variable (sympy's chain-rule output).  The outer ``Subs``
            # printer prints the binding bar; the inner basis call
            # should print bare to avoid the doubled-bar artefact
            # ``\phi_k|_{_xi}|_{_xi = …}``.
            if (self._is_canonical_zeta(bar_value)
                    or bar_value == self._z):
                base = subscripted
            else:
                value_tex = self.doprint(bar_value)
                base = (r"\left. %s \right|_{%s}"
                        % (subscripted, value_tex))
            if exp is not None:
                base = f"{base}^{{{exp}}}"
            return base

        # A plain coefficient with an INTEGER first argument (a modal
        # coefficient ``a(j, t, x)`` surfacing inside a bracket after a closure
        # substitution) subscripts the index like the base printer —
        # ``a(j, t, x) → a_j(t, x)`` — instead of falling through to the
        # unknown-shape ``a(j, t, x)`` form.  Basis ``φ`` / ``c`` were already
        # handled above by their ``_basis`` branch, so this only catches the
        # genuine coefficient families.
        if expr.args and getattr(expr.args[0], "is_integer", False):
            return super()._print_Function(expr, exp=exp)

        name = expr.func.__name__
        tex = self._deal_with_super_sub(name)
        args = expr.args

        # σ-transform detection: if the last "vertical" arg structurally
        # contains the canonical ζ symbol (i.e. the field has been
        # transported across SigmaTransform), decorate the head as
        # ``\tilde f`` and use ζ as the bar variable.  This is the
        # display-only counterpart of the truly distinct sympy object
        # ``Function("u")(t, x, ζ)`` — semantically already a different
        # function from ``Function("u")(t, x, z)``.
        def _is_sigma_mapped(vertical_arg):
            """True iff `vertical_arg` is in the ζ-domain.

            Triggers when the arg structurally contains the canonical
            ζ symbol *or* equals the canonical ζ-domain endpoint
            (`0` or `1`).  Physical-z boundary values
            (`b`, `η = b + h`, …) are functions and don't match;
            they keep the bare-or-restricted physical rendering.
            """
            if vertical_arg == sp.S.Zero or vertical_arg == sp.S.One:
                return True
            try:
                return self._zeta in vertical_arg.free_symbols
            except AttributeError:
                return False

        # Identify the shape: which symbols are "natural", and which
        # symbol — if any — is the restriction-bar variable.
        bar_var = None
        bar_value = None
        recognised = False
        if (len(args) == 2
                and args[0] == self._t and args[1] == self._x):
            # f(t, x) — horizontal, no restriction.
            recognised = True
        elif (len(args) == 3
              and args[0] == self._t and args[1] == self._x
              and args[2] == self._y):
            # f(t, x, y) — 3D horizontal, no restriction.
            recognised = True
        elif (len(args) == 3
              and args[0] == self._t and args[1] == self._x):
            # f(t, x, *) — 2D vertical.  Choose bar variable by the
            # last arg's coordinate flavour: ζ-flavoured ⇒ σ-mapped
            # (tilde), otherwise physical z.
            if _is_sigma_mapped(args[2]):
                tex = r"\tilde{%s}" % tex
                bar_var, bar_value = self._zeta, args[2]
            else:
                bar_var, bar_value = self._z, args[2]
            recognised = True
        elif (len(args) == 4
              and args[0] == self._t and args[1] == self._x
              and args[2] == self._y):
            # f(t, x, y, *) — 3D vertical.  Same coord-flavour rule.
            if _is_sigma_mapped(args[3]):
                tex = r"\tilde{%s}" % tex
                bar_var, bar_value = self._zeta, args[3]
            else:
                bar_var, bar_value = self._z, args[3]
            recognised = True
        elif len(args) == 1:
            # f(ζ) — basis-style: the single arg is the bar variable.
            # The basis is naturally on the reference interval [0, 1]
            # (we write ``phi_k((z − b)/h)`` for the physical column);
            # the reference coordinate is ``ζ``, so:
            # ``phi_k(zeta)``         → ``phi_k``
            # ``phi_k(0)`` / ``phi_k(1)`` → ``phi_k|_{ζ=0}`` / ``|_{ζ=1}``
            # ``phi_k((z-b)/h)``      → ``phi_k|_{ζ=(z-b)/h}``
            bar_var, bar_value = self._zeta, args[0]
            recognised = True

        if recognised:
            # When ``bar_var`` is the reference ζ symbol, also accept the
            # opaque ``state.zeta(...)`` Function-call head as canonical;
            # both represent the same coordinate and rendering a bar
            # ``|_{ζ=ζ}`` is a no-op artefact of sympy chain-rule output.
            zeta_canonical = (bar_var == self._zeta
                              and self._is_canonical_zeta(bar_value))
            if (bar_var is not None
                    and bar_value != bar_var
                    and not zeta_canonical):
                value_tex = self.doprint(bar_value)
                var_tex = self.doprint(bar_var)
                # KaTeX (used by Jupyter / VSCode notebook renderers)
                # does not implement ``\substack``; use a plain
                # ``\left.…\right|_{x=v}`` instead.
                tex = r"\left. %s \right|_{%s=%s}" % (
                    tex, var_tex, value_tex)
            # else: canonical call — strip args entirely.
        else:
            # Unknown shape: keep the full f(args) form so nothing
            # gets silently hidden.
            arg_tex = ", ".join(self.doprint(a) for a in args)
            tex = r"%s\!\left(%s\right)" % (tex, arg_tex)

        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex


class DepthIntegralResult:
    """
    Result of depth-integrating a term over [lower, upper].

    Attributes:
        volume:          the volume integral (Expression)
        boundary_upper:  boundary term at z=upper (Expression)
        boundary_lower:  boundary term at z=lower (Expression)

    The full integral = volume + boundary_upper - boundary_lower.
    Kinematic BCs can be applied to the boundary terms via .apply_bcs().
    """

    def __init__(self, volume, boundary_upper, boundary_lower):
        self.volume = volume
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return DepthIntegralResult(self.volume, upper, lower)

    def assemble(self):
        """Combine all terms: volume + boundary_upper + boundary_lower.

        The sign convention is that each component already carries its
        correct sign.  For Leibniz: upper = -f(eta)*d(eta)/dx,
        lower = +f(b)*db/dx.  For fundamental theorem: upper = +f(eta),
        lower = -f(b).
        """
        return self.volume + self.boundary_upper + self.boundary_lower

    def __repr__(self):
        return (f"DepthIntegralResult(\n"
                f"  volume={self.volume},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


class IBPResult:
    """
    Structured result from integration by parts.

    Attributes:
        integrate: the volume integral (Expression)
        boundary_upper: boundary term at upper limit (Expression)
        boundary_lower: boundary term at lower limit (Expression)
    """

    def __init__(self, integrate, boundary_upper, boundary_lower):
        self.integrate = integrate
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return IBPResult(self.integrate, upper, lower)

    def assemble(self):
        return self.integrate + self.boundary_upper - self.boundary_lower

    def __repr__(self):
        return (f"IBPResult(\n"
                f"  integrate={self.integrate},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


# ---------------------------------------------------------------------------
# Relation: lhs = rhs substitution rules
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Operation: callable transformation applied to all equations
# ---------------------------------------------------------------------------

class Operation(SymbolicBase):
    """A structural transformation on a tree node.

    A node is either an `Expression` (leaf) or a `Zstruct` of
    nodes (intermediate).  The unified entry point is `__call__(node)`:

    - Subclasses override `_apply_leaf` to transform a leaf
      Expression.  The default `__call__` dispatches over the tree,
      recursing into intermediate Zstructs.
    - Rank-changing Operations (`Multiply` with `outer=True` on a
      leaf, or contraction on a Zstruct) override `__call__` directly
      and may return a node of different shape than they received.

    Every Operation carries `name` + `description` so
    `System.apply` can record what was done in the derivation history.

    `log_level` (1–5) is the operation's VERBOSITY in the derivation graph:
    `1` = major (always reported — the default for every op), rising to `5`
    = minor bookkeeping (e.g. `Simplify`
    canonicalisation).  A history view can then filter, e.g. ``describe(...,
    max_log_level=1)`` shows only the major steps.  Set it per class
    (`log_level = 5`) or per instance (`Op(..., log_level=3)`).
    """

    #: Default verbosity level (1 = major / always shown).  Subclasses override.
    log_level = 1

    def __init__(self, name="", description=None, log_level=None):
        super().__init__(name)
        self.description = description or name
        if log_level is not None:
            self.log_level = log_level

    def __call__(self, node):
        """Dispatch: leaf Expression → `_apply_leaf`; Zstruct → recurse."""
        from zoomy_core.misc.misc import Zstruct
        if isinstance(node, Expression):
            return self._apply_leaf(node)
        if isinstance(node, Zstruct):
            out = Zstruct()
            for key in node._filter_dict():
                setattr(out, key, self(getattr(node, key)))
            return out
        raise TypeError(
            f"{type(self).__name__} got an unsupported node type: {type(node).__name__}"
        )

    def _apply_leaf(self, expression):
        """Transform a leaf Expression → Expression (or Zstruct for rank-changers).

        Default: delegate to `_leaf_sp` and re-wrap the result,
        inheriting the leaf's name, tags, and `_as_relation`.
        """
        new_sp = self._leaf_sp(expression.expr)
        if new_sp is expression.expr:
            return expression
        out = Expression(
            new_sp, expression.name,
            term_tags=dict(expression._term_tags),
            tag_order=list(expression._tag_order),
            solver_groups=expression._solver_groups,
        )
        rel = getattr(expression, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def _leaf_sp(self, sp_expr):
        """Transform a raw sympy expression. Override for simple ops.

        Subclasses override *either* this (for per-term sympy transforms)
        *or* `_apply_leaf` (for ops that need the Expression's
        `.terms` / tags), not both.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _leaf_sp or override _apply_leaf"
        )

    # Ops that turn a leaf Expression into an intermediate Zstruct
    # (``Multiply(outer=True)`` etc.) set this to ``True`` so tree-level
    # dispatchers (``_NodeProxy.apply`` / ``System.apply``) know to call
    # ``op(node)`` directly and replace the node, instead of routing
    # through ``Expression.apply``'s per-term loop.
    rank_changes_leaf = False

    # Ops that need to see the whole leaf at once — i.e. they look at
    # cross-term relationships rather than acting per-additive-term —
    # set this to ``True``.  ``Recombine`` is the canonical example: to
    # fold ``α·∂_v f + f·∂_v α → ∂_v(α·f)`` we need to see both terms
    # simultaneously, which the default per-term dispatch hides.
    # Dispatchers call ``op(node)`` directly when this is set, just
    # like ``rank_changes_leaf``, but the result must remain an
    # Expression (not a rank-changing Zstruct).
    whole_leaf_op = False

    # Ops that need the entire System at once — they couple multiple
    # leaf equations (``InvertMassMatrix`` is the canonical example: it
    # left-multiplies a vector of equations by ``M⁻¹`` so per-leaf
    # dispatch is meaningless).  ``System.apply`` routes these directly
    # to ``op(system)``, bypassing the leaf walk entirely.
    system_level = False

    # Ops that are invalid on a multi-term Expression — the author
    # MUST pick a single term via ``apply_to_term(idx, op)``.
    # ``ProductRule`` is the canonical example: applying it to every
    # term of a multi-term equation is the "operation deciding which
    # terms to rewrite" anti-pattern we want to forbid.
    # ``Expression.apply`` checks this flag before the per-term loop
    # and raises if the receiving expression has more than one
    # additive term.
    single_term_only = False

    def _repr_latex_(self):
        return ""


class Multiply(Operation):
    """Multiply an Expression (or tree of Expressions) by a factor.

    Default behaviour (`outer=False`) — *rank-preserving*:
      Scales every term of a leaf Expression by `factor`.  Tags follow
      the terms through the per-term dispatch in
      `Expression.apply`, so `model.apply(Multiply(1/rho))`
      scales the whole system and keeps physical tags intact.

    `outer=True` — *rank-changing* (leaf → Zstruct):
      `factor` must be a `Zstruct` of scalar factors
      (e.g. test functions `phi_0, phi_1, ...`).  The leaf Expression
      is promoted to a `Zstruct` with one child Expression per factor,
      keyed `test_<index>`.  Each child is `leaf.expr * factor_k`.

      Per-term tag inheritance: each parent tag is re-keyed through the
      multiplication (same rule as `Expression.apply`), so physical
      tags follow into every `test_<k>` child.

    Usage:

    ```
        model.apply(Multiply(1 / state.rho))                 # scalar scaling
        phis = Zstruct(**{f"phi_{k}": ... for k in range(N)})
        model.momentum.x.apply(Multiply(phis, outer=True))   # Galerkin test
    ```
    """

    def __init__(self, factor, outer=False, name=None, description=None):
        from zoomy_core.misc.misc import Zstruct
        self._outer = outer
        if outer:
            if not isinstance(factor, Zstruct):
                raise TypeError(
                    "Multiply(outer=True) requires a Zstruct of factors "
                    f"(got {type(factor).__name__})."
                )
            self._factor = factor
            default_desc = (
                f"outer-multiply by {len(factor._filter_dict())} test function(s)"
            )
        else:
            f_sp = factor.expr if isinstance(factor, Expression) else sp.sympify(factor)
            self._factor = f_sp
            default_desc = f"multiply by {f_sp}"
        super().__init__(
            name=name or "multiply",
            description=description or default_desc,
        )

    @property
    def rank_changes_leaf(self):
        return self._outer

    def _leaf_sp(self, sp_expr):
        if self._outer:
            raise NotImplementedError(
                "Multiply(outer=True) is rank-changing; invoke via "
                "node.apply() so the leaf can be replaced with a Zstruct."
            )
        return sp_expr * self._factor

    def _apply_leaf(self, expression):
        if not self._outer:
            return super()._apply_leaf(expression)
        from zoomy_core.misc.misc import Zstruct
        out = Zstruct()
        for i, key in enumerate(self._factor._filter_dict()):
            f = getattr(self._factor, key)
            f_sp = f.expr if isinstance(f, Expression) else sp.sympify(f)
            # Deep-clone the parent (expr + tags + solver_groups + relation),
            # rename to ``<parent>_<index>``, then pointwise scalar-multiply
            # by ``f_sp``.  The scalar ``Multiply`` routes through
            # ``Expression.apply``'s per-term loop, so tag re-keying against
            # the new terms is handled by the existing single-source-of-truth
            # path — no parallel logic here.
            child_name = (f"{expression.name}_{i}" if expression.name
                          else f"test_{i}")
            clone = Expression(
                expression.expr, child_name,
                term_tags=dict(expression._term_tags),
                tag_order=list(expression._tag_order),
                solver_groups=expression._solver_groups,
            )
            rel = getattr(expression, "_as_relation", None)
            if rel is not None:
                clone._as_relation = dict(rel)
            child = clone.apply(Multiply(f_sp))
            setattr(out, f"test_{i}", child)
        return out


class DepthIntegrate(Operation):
    """Depth-integrate all equations over [b, b+h] w.r.t. z.

    Applies Leibniz rule and fundamental theorem term-by-term.
    Boundary values (w at z=b, u at z=b+h, etc.) remain as `Subs`
    objects.  Apply `ApplyKinematicBCs` to see the cancellations.
    """

    def __init__(self, state):
        super().__init__(
            name="depth_integrate",
            description="Depth integration over [b, b+h] (Leibniz rule)",
        )
        self._state = state

    def _apply_leaf(self, expression):
        s = self._state
        return expression.map(lambda t: t.depth_integrate(s.b, s.eta, s.z))

    def _repr_latex_(self):
        s = self._state
        return (
            f"$\\int_{{{sp.latex(s.b)}}}^{{{sp.latex(s.eta)}}} "
            f"(\\cdot)\\, d{sp.latex(s.z)}$"
        )


class ApplyKinematicBCs(Operation):
    """Apply kinematic BCs globally to combined boundary terms.

    Evaluates `Subs` boundary values, applies kinematic BCs at
    surface and bottom, and simplifies. The Leibniz boundary u-terms
    cancel with the fundamental theorem w-terms.

    Must be applied immediately after `DepthIntegrate`.
    """

    def __init__(self, state):
        super().__init__(
            name="kinematic_bcs",
            description="Kinematic BCs (surface + bottom): w = u·∂b/∂x + ∂b/∂t",
        )
        self._kbc_s = KinematicBCSurface(state)
        self._kbc_b = KinematicBCBottom(state)

    def _leaf_sp(self, expr):
        # Evaluate only Subs objects (NOT Derivative(Integral))
        if expr.has(sp.Subs):
            subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
            expr = expr.subs(subs_map)
        # Apply both BCs (oriented Expressions → substitute their {lhs: rhs}).
        for bc in [self._kbc_s, self._kbc_b]:
            expr = expr.subs(bc.subs_map)
        # Simplify (cancels d(b+h)/dt - db/dt → dh/dt)
        return _simplify_preserve_integrals(expr)

    def _repr_latex_(self):
        parts = [self._kbc_s._repr_latex_(), self._kbc_b._repr_latex_()]
        return " \\\\ ".join(p for p in parts if p)


_integrate_cache: dict = {}
_integrate_cache_stats = {"hits": 0, "misses": 0}

# Registry of Zoomy-specific integration rules.  Each rule is a callable
# ``(integrand, limits) -> Optional[sp.Expr]``: returns the definite
# integral if the rule applies, ``None`` otherwise.  ``_cached_integrate``
# consults the registry before falling back to ``sympy.integrate``.
#
# Motivation: sympy's integration algorithm makes antiderivative choices
# that are correct but inconvenient for our downstream simplification —
# e.g. for ``∫ ∂_x f(z) dz`` with ``f`` linear in ``z`` it picks the
# chain-rule form ``f(z)²/(2·∂_z f)`` over the polynomial
# ``A·z + B·z²/2``.  The two differ by a constant, so the definite
# integral is algebraically the same, but sympy leaves the rational
# form unreduced and it survives all the way to the solver-tag
# extraction as a stubborn ``1/Derivative(state_var, x)`` residue.
# Rules in this registry give us deterministic, polynomial-shaped
# antiderivatives where we can; anything the rules don't match falls
# through to sympy's fully general (and very capable) integrator.
_INTEGRATION_RULES: list = []


def register_integration_rule(rule):
    """Register a Zoomy integration rule.

    A rule is `(integrand, limits) -> Optional[sp.Expr]`:

    * `limits` is the full `(var, lower, upper)` triple passed to
      `_cached_integrate` (matching `sympy.integrate`'s call shape).
    * Return the **definite** integral if the pattern matches — the
      caller does no further transformation.
    * Return `None` if the rule doesn't apply.  `_cached_integrate`
      tries rules in registration order; the first non-`None` wins.
    """
    _INTEGRATION_RULES.append(rule)
    return rule


def _try_integration_rules(integrand, limits):
    """Run registered rules in order, return the first non-None result."""
    for rule in _INTEGRATION_RULES:
        try:
            result = rule(integrand, limits)
        except Exception:
            # A rule that raises is treated as "does not apply" — never
            # abort the integration because a rule had a bad day.  The
            # sympy fallback catches everything the rules miss.
            continue
        if result is not None:
            return result
    return None


def _cached_integrate(integrand, limits):
    """Cached wrapper around the Zoomy-rule + `sympy.integrate` pair.

    Implements **integrand isolation**: each additive term of the
    integrand is split into a `var`-independent coefficient and a
    `var`-dependent factor; only the `var`-dependent factor is
    keyed into the integration cache.  This dramatically increases
    cache reuse across the SME / VAM derivations, where the same
    basis-product integral (e.g. `∫ φ_i(σ) · φ_j(σ) dσ`) recurs
    inside many different equations with different `c(t, x)`
    multipliers.

    Two cache layers:
      * outer cache on the whole `(integrand, limits)` — short-circuits
        identical-integrand repeats with one dict lookup.
      * inner cache on each per-term `var`-dependent factor — what
        captures the basis-product reuse.

    Resolution order for each var-dependent factor: (1) `_INTEGRATION_RULES`;
    (2) `sympy.integrate` fallback.  Returns the original Integral
    on failure (caller can detect via `isinstance(..., Integral)`).
    """
    # Outer cache — whole integrand match (most common hit).
    key = (integrand, limits)
    hit = _integrate_cache.get(key)
    if hit is not None:
        _integrate_cache_stats["hits"] += 1
        return hit

    var = limits[0]
    # Per-additive-term integrand isolation.
    if isinstance(integrand, sp.Add):
        result = sp.S.Zero
        for arg in integrand.args:
            indep, dep = arg.as_independent(var, as_Add=False)
            result = result + indep * _integrate_var_factor(dep, limits)
    else:
        indep, dep = integrand.as_independent(var, as_Add=False)
        result = indep * _integrate_var_factor(dep, limits)

    _integrate_cache[key] = result
    return result


def _integrate_var_factor(dep, limits):
    """Inner integration of a single `var`-dependent factor.

    `dep` must not contain summands with mixed `var`-dependence —
    callers split via `sp.Expr.as_independent` first.  The cache
    is keyed on `(dep, limits)` only, so identical basis-product
    shapes across different equations share the same cached scalar.
    """
    var, lo, hi = limits[0], limits[1], limits[2]
    if not dep.has(var):
        # Pure constant factor — integral is (hi − lo) · dep.
        return dep * (hi - lo)

    key = (dep, limits)
    hit = _integrate_cache.get(key)
    if hit is not None:
        _integrate_cache_stats["hits"] += 1
        return hit
    _integrate_cache_stats["misses"] += 1

    result = _try_integration_rules(dep, limits)
    if result is None:
        try:
            result = sp.integrate(dep, limits)
        except Exception:
            result = Integral(dep, limits)
    _integrate_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Default Zoomy integration rules
# ---------------------------------------------------------------------------
#
# Rules are deliberately narrow — each catches one well-understood
# shape and nothing else.  Rules compose by registration order; the
# first to match wins.


@register_integration_rule
def _rule_fundamental_theorem(integrand, limits):
    """`∫_lo^hi ∂_var g dvar = g|_hi − g|_lo`.

    Handles both single-order `∂_var g` and mixed-order
    `∂_var ∂_other g` (removing one `var` from the outer
    Derivative).  Higher-order-in-`var` like `∂²_var g` is
    handled too: `∫ ∂²_var g dvar = ∂_var g|_hi − ∂_var g|_lo`.

    Whenever `var` is one of the differentiation variables of the
    integrand, the antiderivative is the integrand with *one*
    occurrence of `var` stripped — no further computation needed.
    """
    if not isinstance(integrand, Derivative):
        return None
    var, lo, hi = limits[0], limits[1], limits[2]
    # sympy canonicalises Derivative.args[1:] to ``(var, order)`` tuples.
    diff_tuples = list(integrand.args[1:])
    match_index = None
    for i, dv in enumerate(diff_tuples):
        if len(dv) == 2 and dv[0] == var:
            match_index = i
            break
    if match_index is None:
        return None
    matched = diff_tuples[match_index]
    order = matched[1]
    if order <= 1:
        # Remove the ``(var, 1)`` entry entirely.
        remaining = diff_tuples[:match_index] + diff_tuples[match_index + 1:]
    else:
        # Reduce ``(var, n) → (var, n-1)`` — one ``var`` peels off.
        remaining = (diff_tuples[:match_index]
                     + [sp.Tuple(var, order - 1)]
                     + diff_tuples[match_index + 1:])
    inner = integrand.args[0]
    if remaining:
        antideriv = Derivative(inner, *remaining)
    else:
        antideriv = inner
    return antideriv.subs(var, hi) - antideriv.subs(var, lo)


@register_integration_rule
def _rule_derivative_of_polynomial(integrand, limits):
    """`∫_lo^hi ∂_y f(var) dvar` with `f` polynomial in `var`
    and a single differentiation variable `y ≠ var`.

    Applies the Leibniz rule directly:

    ``∫_lo^hi ∂_y f dvar = ∂_y[∫_lo^hi f dvar]
                          − f(hi)·∂_y hi + f(lo)·∂_y lo.``

    `∫ f dvar` is polynomial (sympy handles pure polynomial cases
    cleanly), so the Leibniz-swap avoids sympy's chain-rule
    antiderivative `f²/(2·∂_var f)` which introduces rational
    forms in `∂_y ∂_var f` denominators — the exact trap that
    surfaced in the SME shear-moment residual.

    Registered by default: the no-rationals invariant trumps the
    slightly-less-compact Leibniz expansion.

    Skips higher-order `∂²_y f` integrands (not a product-rule
    case) and mixed `∂_y ∂_var f` (delegated to
    `_rule_fundamental_theorem` since `var` is in the diff-var
    tuple).
    """
    if not isinstance(integrand, Derivative):
        return None
    var, lo, hi = limits[0], limits[1], limits[2]
    diff_tuples = list(integrand.args[1:])
    if len(diff_tuples) != 1:
        return None
    dv = diff_tuples[0]
    if len(dv) != 2 or dv[1] != 1:
        # Higher-order: ``∂²_y f`` — not this rule.
        return None
    y = dv[0]
    if y == var:
        # ``∂_var f``: fundamental theorem rule already fired.
        return None
    inner = integrand.args[0]
    if not inner.has(var):
        # ``var``-free integrand — sympy's default behaviour is fine
        # (and the rule would accidentally treat the integrand as a
        # zero-degree polynomial).  Let the fallback handle it.
        return None
    # Sympy's ``Derivative(var · g, y)`` stays unevaluated even though
    # ``var`` is independent of ``y`` and could be pulled out as a
    # coefficient.  ``Poly(..., var)`` fails on such Derivative atoms
    # because they "contain a generator".  Evaluate Derivative atoms
    # structurally (without running ``.doit()`` on nested Integrals /
    # Subs which we want to preserve) so ``var`` factors surface.
    def _doit_only_derivatives(e):
        if isinstance(e, Derivative):
            return e.doit()
        if e.args:
            new = tuple(_doit_only_derivatives(a) for a in e.args)
            if any(n is not o for n, o in zip(new, e.args)):
                return e.func(*new)
        return e
    inner = _doit_only_derivatives(inner)
    inner = sp.expand(inner)
    try:
        sp.Poly(inner, var)
    except (sp.PolynomialError, sp.CoercionFailed, sp.GeneratorsNeeded):
        return None
    # The polynomial-Leibniz form never builds rationals:
    #   ∫_lo^hi ∂_y f dvar  =  ∂_y[∫_lo^hi f dvar]
    #                         − f(hi)·∂_y hi + f(lo)·∂_y lo
    # ``∫ f dvar`` is pure polynomial (via ``sp.Poly``); the surface
    # terms only fire when bounds depend on ``y``.  This is always
    # the right answer when the Poly build above succeeds, so return
    # it unconditionally — sympy's ``u-sub antiderivative`` trap is
    # the default disease we're curing, and even when sympy's form is
    # clean for other shapes, this form is equally valid.
    inner_integrated = sp.Poly(inner, var).integrate().as_expr()
    inner_integrated = inner_integrated.subs(var, hi) - inner_integrated.subs(var, lo)
    main = Derivative(inner_integrated, y)
    surface = S.Zero
    if hi.has(y):
        surface -= inner.subs(var, hi) * Derivative(hi, y)
    if lo.has(y):
        surface += inner.subs(var, lo) * Derivative(lo, y)
    return main + surface


@register_integration_rule
def _rule_polynomial_integrand(integrand, limits):
    """`∫_lo^hi poly(var) dvar` via `sp.Poly.integrate` — strictly
    polynomial, no `sp.integrate` call that might invoke rational
    routines (u-sub, together, cancel).

    Applies to any integrand that, after a structural Derivative
    evaluation pre-pass, is a polynomial in `var` with coefficients
    that may contain unevaluated `Derivative` / `Subs` / other
    symbolic atoms.  Registered *after* the Derivative-of-polynomial
    rule so the outer-derivative shape wins when both apply.
    """
    var, lo, hi = limits[0], limits[1], limits[2]
    expr = integrand
    # Distribute linear derivatives (``Derivative(var·f, y) → var·∂_y f``)
    # so ``var`` surfaces as a polynomial generator.  Only walk
    # ``Derivative`` atoms — don't ``.doit()`` Integrals / Subs we
    # want preserved.
    def _doit_derivs(e):
        if isinstance(e, Derivative):
            return e.doit()
        if e.args:
            new = tuple(_doit_derivs(a) for a in e.args)
            if any(n is not o for n, o in zip(new, e.args)):
                return e.func(*new)
        return e
    expr = _doit_derivs(expr)
    expr = sp.expand(expr)
    if not expr.has(var):
        # ``var``-free integrand → result is ``(hi − lo) · expr``.
        return (hi - lo) * expr
    try:
        poly = sp.Poly(expr, var)
    except (sp.PolynomialError, sp.CoercionFailed, sp.GeneratorsNeeded):
        return None
    anti = poly.integrate().as_expr()
    return anti.subs(var, hi) - anti.subs(var, lo)


def _has_derivative_denominator(expr):
    """True iff `expr` contains `Pow(X, negative)` with a
    `Derivative` inside `X` — the signature of sympy's u-sub
    antiderivative trap."""
    for pw in expr.atoms(sp.Pow):
        exp = pw.args[1]
        if not getattr(exp, "is_negative", False):
            continue
        if pw.args[0].atoms(Derivative):
            return True
    return False


def integrate_cache_stats():
    """Return `(hits, misses)` for the cached-integrate helper."""
    return _integrate_cache_stats["hits"], _integrate_cache_stats["misses"]


def integrate_cache_clear():
    """Drop all cached integrals and reset stats (useful between runs)."""
    _integrate_cache.clear()
    _integrate_cache_stats["hits"] = 0
    _integrate_cache_stats["misses"] = 0


class EvaluateIntegrals(Operation):
    """Run `sympy.integrate` on every `Integral` node in the expression.

    After `AffineProjection` + basis expansion the volume integrals
    look like `Integral(polynomial_in_zeta, (zeta, 0, 1))`.  This op
    collapses each such integral to a scalar.

    Walks `Derivative(Integral(...), x)` too: the inner Integral is
    evaluated first, then the outer Derivative survives symbolically.

    Uses `_cached_integrate` so repeated integrands across leaves
    (every `test_k` clone after `Multiply(basis.phi, outer=True)`)
    cost one `sympy.integrate` call each, not one-per-leaf.

    Leaves unevaluatable Integrals untouched (e.g. non-polynomial ones
    or ones whose result would need a closed form sympy can't find).
    """

    # Inspecting the whole leaf at once is necessary: an outer
    # ``Derivative(Integral(...), x)`` shape has the Integral nested
    # inside an Add inside a Derivative inside the leaf, and the
    # per-term walk would miss the cross-term interactions of the
    # ``_walk`` recursion.  See the level=1 VAM closure regression.
    whole_leaf_op = True

    def __init__(self, state=None, name="evaluate_integrals",
                 description=None):
        super().__init__(
            name=name,
            description=description or "Evaluate all Integral nodes via sympy.integrate",
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative, Sum

        def _resolve_coeff_calls(e):
            """Substitute `coeff_<symbol>(k_int)` → coefficients[k_int]
            for every per-Expand coeff Function class.

            After `Sum.doit()` unrolls a Sum to `Σ_k …`, the inner
            `coeff_fn(k_int)` calls have concrete integer indices
            and the per-class `_coeff_table` provides the lookup.
            Symbolic-index calls (un-doited Sums) are left untouched.
            """
            coeff_classes = set()
            for atom in e.atoms(sp.Function):
                if hasattr(atom.func, "_coeff_table"):
                    coeff_classes.add(atom.func)
            for cls in coeff_classes:
                table = cls._coeff_table

                def _replace(*args, _t=table, _cls=cls):
                    if len(args) == 1 and getattr(args[0], "is_Integer",
                                                   False):
                        return _t[int(args[0])]
                    return _cls(*args)

                e = e.replace(cls, _replace)
            return e

        def _doit_sums(e):
            """Recursively unroll `sp.Sum` atoms via `.doit()` and
            then substitute any per-Expand `coeff_<symbol>(k_int)`
            calls with the concrete coefficient.  Other `.doit()`-able
            atoms (Derivatives, Integrals) are left alone — the outer
            integral fixpoint loop handles those.
            """
            if isinstance(e, Sum):
                # Unroll this Sum; recurse into the unrolled result so
                # nested Sums also expand.
                unrolled = _doit_sums(e.doit())
                return _resolve_coeff_calls(unrolled)
            if hasattr(e, "args") and e.args:
                new_args = [_doit_sums(a) for a in e.args]
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        def _bases_in(integrand):
            """Distinct `Basisfunction` instances referenced by opaque
            atoms in `integrand` (matching `func._basis` back-ref)."""
            found = []
            seen = set()
            for atom in integrand.atoms(sp.Function):
                basis = getattr(atom.func, "_basis", None)
                if basis is None:
                    continue
                if id(basis) not in seen:
                    seen.add(id(basis))
                    found.append(basis)
            return found

        def _resolve_integral(integrand, limits):
            """Single integration step with Sum-unroll + opaque-phi routing.

            (1) Unroll any `sp.Sum` atoms in the integrand — `Expand`
            produced unevaluated Sums to keep `ProductRule` etc.
            tractable; here at integration time we expand them so each
            term is individually integrable.  `amp_fn` auto-evaluation
            substitutes amplitudes for concrete integer indices.

            (2) If the resulting integrand contains opaque basis atoms
            (`phi_fn(k, arg)` with `func._basis` set), substitute
            each basis's atoms with its concrete polynomial form via
            `basis.resolve_atoms` and `sympy.integrate` the
            polynomial-in-`var` integrand.  Otherwise fall through to
            the cached `sympy.integrate` path.
            """
            integrand = _doit_sums(integrand)
            bases = _bases_in(integrand)
            if bases:
                resolved = integrand
                for basis in bases:
                    resolved = basis.resolve_atoms(resolved)
                # Use ``_cached_integrate`` (Zoomy's rule set) rather
                # than raw ``sp.integrate`` — the post-resolution
                # integrand often has shapes (``Derivative`` outer over
                # rational-in-h polynomial integrand) that the rule
                # set handles cleanly but raw sympy.integrate leaves
                # unevaluated.
                return _cached_integrate(resolved, limits)
            return _cached_integrate(integrand, limits)

        def _walk(e):
            if isinstance(e, Integral):
                # Recurse into nested integrands first.
                integrand = _walk(e.args[0])
                limits = e.args[1]
                evaluated = _resolve_integral(integrand, limits)
                if not isinstance(evaluated, Integral):
                    return evaluated
                return Integral(integrand, limits, *e.args[2:])
            if isinstance(e, Derivative):
                inner = _walk(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if hasattr(e, "args") and e.args:
                new_args = [_walk(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        # Alternate integrate / resolve-Subs until fixpoint: each
        # integration may uncover ``Subs`` nodes whose inner just got
        # cleaner, and each resolve may expose new ``Integral`` nodes
        # (outer integrals whose integrand was previously a ``Subs``).
        # The loop usually converges in 2 iterations but the bound is
        # generous to handle deeply nested cases.
        result = expr
        for _ in range(6):
            prev = result
            result = _walk(result)
            result = _resolve_subs_safe(result)
            if result == prev:
                break

        # Final cleanup: unroll any remaining standalone Sums (e.g. the
        # boundary-evaluation Sum produced when ``Expand`` substituted
        # ``field(t, x, b)`` or ``field(t, x, b+h)`` outside of an
        # Integral context — those Sums never went through the integral
        # path).  Then resolve any opaque basis atoms with concrete
        # polynomials so downstream tagging / quasilinear analysis sees
        # a fully closed expression.
        result = _doit_sums(result)
        bases_present = _bases_in(result)
        for basis in bases_present:
            result = basis.resolve_atoms(result)
        return result


class Expand(Operation):
    """Substitute a vertical field with its polynomial ansatz expansion
    in **unevaluated Sum form**.

    Replaces every call `field(t, x, [y,] arg_z)` with

        `Sum(coeff_fn(k) · basis.phi_fn(k, (arg_z − b)/h), (k, 0, L))`

    — a single `sp.Sum` atom rather than the unrolled Add.  Two
    placeholders make this work as one symbolic chunk:

    * `basis.phi_fn` is the basis's opaque 2-arg Function (carries
      `_basis = basis`); concrete polynomial substitution is deferred
      until `EvaluateIntegrals`.
    * `coeff_fn` is a private sympy Function class named
      `coeff_<basis.symbol>` and carrying a `_coeff_table`
      class attribute holding the pre-declared coefficient list.
      It stays opaque through `ProductRule`, `AffineProjection`,
      and `Integrate` (the Sum carries through as a single
      mathematical term).  `EvaluateIntegrals` substitutes
      `coeff_fn(k_concrete) → coefficients[k_concrete]` after the
      Sum has been `.doit()`-ed.

    Boundary evaluations of the field (`arg_z = b`, `arg_z = b+h`)
    are handled automatically: `(arg_z − b)/h` becomes 0 or 1
    respectively, the Sum is `Σ_k amp_k · phi_fn(k, 0|1)`, and
    EvaluateIntegrals' final cleanup pass resolves these via
    `basis.resolve_atoms` once the Sum has been `.doit()`-ed.

    Why Sum-form: at higher levels (L ≥ 1), the unrolled Add form has
    `L+1` terms per expanded field — products of two expanded fields
    explode quadratically, and `ProductRule` would have to be
    applied to each one individually.  The single-Sum representation
    keeps the equation compact and makes `ProductRule` a single
    apply per expansion, which is the way the math reads on paper.

    Parameters
    ----------
    field : sympy Function call
        The vertical field, e.g. `state.u` (which is `u(t, x, z)`).
    basis : `Basisfunction`
        The basis instance.  `basis.phi_fn` is the 2-arg opaque
        Function used inside the Sum.
    coefficients : sequence of sympy Function calls
        Pre-declared horizontal coefficient functions
        (`U_0(t, x), U_1(t, x), …`) — one per basis level.  Length
        must equal `basis.level + 1`.  Substituted into the Sum by
        `EvaluateIntegrals` after `Sum.doit()` unrolls the
        symbolic index to concrete integers.
    state : `StateSpace`
        The state space — used to read `state.b` and `state.h`
        for the affine ζ-map.
    """

    def __init__(self, field, basis, coefficients, state,
                 name="expand", description=None):
        coefficients = list(coefficients)
        if len(coefficients) != basis.level + 1:
            raise ValueError(
                f"Expand expects {basis.level + 1} coefficients "
                f"(basis.level + 1), got {len(coefficients)}."
            )
        super().__init__(
            name=name,
            description=description or
                f"Expand {field} via {type(basis).__name__}"
                f"(level={basis.level}, symbol={basis.symbol!r}) ansatz",
        )
        self._field_fn = field.func
        self._basis = basis
        self._coefficients = coefficients
        self._state = state

        # Per-Expand sympy Function class that carries the coefficient
        # table on the class itself.  Different bases on the same
        # model have different ``basis.symbol`` values
        # (``phi``/``eta``/``mu``/…) and therefore distinct coeff
        # classes — no collisions.  The class stays opaque through
        # the chain; ``EvaluateIntegrals`` substitutes
        # ``coeff_<symbol>(k_int) → coefficients[k_int]`` after
        # ``Sum.doit()`` produces concrete integer indices.
        #
        # ``_user_name`` is the LaTeX-display name inferred from the
        # user's coefficients (``U_0`` / ``U_1`` → ``U``).  The
        # ``_StripArgsLatexPrinter`` reads it to render the opaque
        # symbolic-index call as ``U_{k}`` rather than the default
        # 1-arg basis-bar form ``coeff|_{ζ=k}``.
        import re
        first_name = coefficients[0].func.__name__
        m = re.match(r"^(.+?)_\d+$", first_name)
        user_name = m.group(1) if m else first_name
        self._coeff_fn = type(
            f"coeff_{basis.symbol}",
            (sp.Function,),
            {
                "_coeff_table": coefficients,
                "_user_name": user_name,
            },
        )

    def _leaf_sp(self, expr):
        # Compose the ansatz with the affine ζ-map at construction
        # time: ``u(t, x, arg_z) → Σ_k U_k · phi_fn(k, (arg_z − b)/h)``.
        # The basis is defined on ``ζ ∈ [0, 1]``, so the arg has to be
        # ``(arg_z − b)/h`` for ``EvaluateIntegrals`` /
        # ``basis.resolve_atoms`` to evaluate the polynomial in the
        # right domain.  At boundary calls (``arg_z = b`` /
        # ``arg_z = b + h``) the ratio collapses to ``0`` / ``1``.
        basis = self._basis
        coeff_fn = self._coeff_fn
        L = basis.level
        b = self._state.b
        h = self._state.h

        def _rhs(*field_args):
            arg_z = field_args[-1]
            zeta = (arg_z - b) / h
            k = sp.Dummy("k", integer=True, nonnegative=True)
            return sp.Sum(
                coeff_fn(k) * basis.phi_fn(k, zeta),
                (k, 0, L),
            )

        return expr.replace(self._field_fn, _rhs)


class ProjectBasisIntegrals(Operation):
    """Resolve every `Integral(..., (ζ, a, b))` via basis-aware cache lookup.

    Replaces the old "expand polynomial basis + run `sympy.integrate`"
    chain.  Every integral whose integrand contains opaque `phi_k(arg)`
    nodes is mapped to a `BasisIntegralCache` query, with
    `ζ`-independent factors pulled out first so the kernel the cache
    sees is a clean basis-only product.

    Steps, applied per leaf expression:

    1. **Canonicalize Subs** — rewrite `Subs(Derivative(phi_k(ξ), ξ), ξ, arg)`
       (the sympy chain-rule leftover) back to
       `Derivative(phi_k(arg), arg)`.
    2. **Distribute Adds** — `Integral(Add(t1, t2), …) → Σ Integral(ti, …)`
       so each term can be factored independently.
    3. **Factor constants out + cache query** — for each
       `Integral(kernel, (var, a, b))`:

       * split the Mul integrand into `var`-independent factor `C`
         and `var`-dependent kernel `K`;
       * if `K` contains any `phi_k`, hand `K` to the cache and
         multiply its result by `C`;
       * otherwise leave the Integral alone.

    The cache returns a sympy scalar / polynomial; nested running
    integrals inside `K` get evaluated via the cache too (their
    polynomial results are substituted into the outer integrand on the
    next pass, and the fixpoint loop below handles this).

    No `sympy.integrate` is ever called on an opaque `phi_k`: the
    cache concretizes `phi_k` against the basis polynomial first, then
    does polynomial integration only.
    """

    def __init__(self, basis_cache, name="project_basis_integrals",
                 description=None):
        super().__init__(
            name=name,
            description=description or (
                "Resolve volume integrals via BasisIntegralCache lookup"),
        )
        self.basis_cache = basis_cache

    def _leaf_sp(self, expr):
        from sympy import Add, Derivative, Integral, Mul, Subs

        # ------------------ Pass 1: canonicalize Subs of phi-derivatives
        def _canon(e):
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
                    return Derivative(fn(arg), arg)
            if e.args:
                new_args = tuple(_canon(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        # ------------------ Pass 2: distribute Integral over Add
        def _distribute(e):
            if isinstance(e, Integral):
                integrand = _distribute(e.args[0])
                limits = e.args[1:]
                if isinstance(integrand, Add):
                    return Add(*(Integral(t, *limits) for t in integrand.args))
                if integrand is not e.args[0]:
                    return Integral(integrand, *limits)
                return e
            if e.args:
                new_args = tuple(_distribute(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        def _has_phi(e):
            if (isinstance(e, sp.Function)
                    and getattr(e.func, "__name__", "").startswith("phi_")):
                return True
            if e.args:
                return any(_has_phi(a) for a in e.args)
            return False

        # ------------------ Pass 3: factor + query cache
        def _map(e):
            if isinstance(e, Integral):
                integrand = _map(e.args[0])
                limits = e.args[1]
                if len(limits) != 3:
                    if integrand is not e.args[0]:
                        return Integral(integrand, *e.args[1:])
                    return e
                var, a, b = limits
                # split integrand into var-independent const × var-dependent kernel
                if isinstance(integrand, Mul):
                    consts, kern_parts = [], []
                    for f in integrand.args:
                        (kern_parts if f.has(var) else consts).append(f)
                    const = Mul(*consts) if consts else sp.S.One
                    kernel = Mul(*kern_parts) if kern_parts else sp.S.One
                elif integrand.has(var):
                    const, kernel = sp.S.One, integrand
                else:
                    return integrand * (b - a)
                if _has_phi(kernel):
                    value = self.basis_cache.integrate(kernel, var, a, b)
                    return const * value
                if integrand is not e.args[0]:
                    return Integral(integrand, *e.args[1:])
                return e
            if e.args:
                new_args = tuple(_map(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        # Resolve leftover boundary Subs ``Subs(f(z), z, b|η)`` that came
        # out of the Leibniz rule.  At this point in the pipeline there
        # are no more pending chain-rule-through-val ambiguities — any
        # inner ``Derivative(g(z), z)`` genuinely means ``g'(z)|_{z=…}``
        # and ``Subs.doit()`` handles it correctly.
        def _resolve_subs_doit(e):
            if isinstance(e, sp.Subs):
                return e.doit()
            if e.args:
                new_args = tuple(_resolve_subs_doit(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        result = _canon(expr)
        result = _distribute(result)
        # Alternate map / Subs-resolution until fixpoint.  Each map pass
        # may expose fresh ``Subs(_, z, b|η)`` boundary terms whose
        # inner has ``phi_k((z-b)/h)·…``; resolving those produces
        # ``phi_k(0)`` / ``phi_k(1)``.  Running integrals nested inside
        # outer integrands also resolve in inside-out order across
        # iterations.  Any ``phi_k(arg)`` left *outside* a volume
        # Integral after the fixpoint is a boundary evaluation
        # (``phi_k(0)``, ``phi_k(1)``, ``phi_k|_{z=b}``, …) — those
        # stay symbolic on purpose.
        for _ in range(6):
            prev = result
            result = _map(result)
            result = _resolve_subs_doit(result)
            if result == prev:
                break
        return result


class SimplifyIntegrals(Operation):
    """Evaluate integrals with constant integrand and remove zero integrals.

    - `∫ 0 dz → 0`
    - `∫ c dz → c·h` (if integrand has no z-dependence)
    - `∂/∂x ∫ 0 dz → 0`
    """

    def __init__(self, state):
        super().__init__(
            name="simplify_integrals",
            description="Evaluate constant/zero integrals",
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative

        def _simplify_int(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                var = limits[0]
                lower, upper = limits[1], limits[2]
                if integrand == S.Zero:
                    return S.Zero
                if not integrand.has(var):
                    return integrand * (upper - lower)
                return e
            if isinstance(e, Derivative):
                inner = _simplify_int(e.args[0])
                if inner == S.Zero:
                    return S.Zero
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_simplify_int(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _simplify_int(expr)


class Integrate(Operation):
    """Integrate each additive term of an equation w.r.t. `var` over [lower, upper].

    Per-term strategy is driven by `method`:

    * `"auto"` (default) — detect the outermost derivative direction
      per term and pick Leibniz / fundamental theorem / direct accordingly.
    * `"leibniz"` — force the Leibniz rule (for terms containing
      `∂/∂x_i` where `x_i != var`):
      `∫ ∂_x f dvar = ∂_x [∫ f dvar]  − f|_upper · ∂_x upper  + f|_lower · ∂_x lower`.
    * `"fundamental_theorem"` — for terms containing `∂/∂var` as the
      outermost derivative: `∫ ∂_var f dvar = f|_upper − f|_lower`.
    * `"direct"` — keep as an unevaluated `Integral(expr, (var, lower, upper))`.
    * `"analytical"` — run `sympy.integrate` on the whole expression
      (what the original small `Integrate` did; useful for z-momentum
      analytic integration when the integrand has no horizontal derivatives).

    Boundary terms produced by Leibniz / fundamental-theorem are kept as
    `Subs(f, var, bound)` expressions — no kinematic BCs are applied here.
    Resolve them later via an explicit `.apply({Subs-pattern: value})` or
    a convenience Relation (`ApplyKinematicBCs`, `StressFreeSurface`, ...).

    Partial integration (`upper == var`) is supported by the `"direct"`
    and `"analytical"` methods — sympy will produce a running integral.

    Usage:

    ```
        zmom.apply(Integrate(state.z, state.b, state.z))             # partial
        zmom.apply(Integrate(state.z, state.b, state.eta))           # full depth
        zmom.apply(Integrate(state.z, state.b, state.b + state.h,
                             method="analytical"))                    # sympy.integrate
    ```
    """

    # See ``_apply_leaf``: volume pieces produced by different input
    # terms are collapsed into a single ``Integral(sum_integrand,
    # limits)`` per ``(limits, outer-derivative-variables)`` signature,
    # so sympy's ``Add`` canonicalization can collapse like-terms inside
    # a single integrand.  That cross-term consolidation needs the
    # whole leaf at once — opt out of the per-additive-term dispatch.
    whole_leaf_op = True

    def __init__(self, var, lower, upper, method="auto"):
        super().__init__(
            name="integrate",
            description=(f"Integrate w.r.t. {var} from {lower} to {upper} "
                         f"(method={method})"),
        )
        self._var = var
        self._lower = lower
        self._upper = upper
        self._method = method

    def _apply_leaf(self, expression):
        if self._method == "analytical":
            # Whole-expression analytic integration, with Dummy substitution
            # to avoid clashes when upper == var (partial / running integral).
            dummy = sp.Dummy("_int_dummy")
            integrand = expression.expr.subs(self._var, dummy)
            result = sp.integrate(integrand, (dummy, self._lower, self._upper))
            return Expression(result, expression.name)

        # Per-term dispatch through Expression.depth_integrate.  We group
        # the volume pieces by ``(limits, outer-derivative-variable)`` so
        # every term that lands on the same signature contributes to the
        # same ``Integral(sum, limits)`` (optionally wrapped in a shared
        # ``Derivative(..., diff_var)``).  This keeps all volume
        # cancellations available to sympy's Add canonicalization inside
        # a single integrand, instead of fragmenting across many
        # ``Integral`` atoms that never recombine.  Boundary pieces are
        # not Integrals; they pass through as-is and simplify normally.
        volumes = {}   # (limits_tuple, diff_var_or_None) -> integrand sum (sympy)
        boundaries = []  # list[sympy.Expr]

        def _collect_volume(vol_expr):
            """Extract (limits, diff_var, integrand) from a volume piece and
            add to `volumes`.  `vol_expr` is either `S.Zero`,
            `Integral(f, lim)`, or `Derivative(Integral(f, lim), s)`.
            Anything else is treated as a boundary-type expression."""
            if vol_expr == S.Zero or vol_expr == 0:
                return
            if isinstance(vol_expr, Derivative) and isinstance(vol_expr.args[0], Integral):
                inner = vol_expr.args[0]
                limits = inner.args[1]
                diff_vars = tuple(vol_expr.args[1:])
                key = (tuple(limits), diff_vars)
                volumes[key] = volumes.get(key, S.Zero) + inner.args[0]
                return
            if isinstance(vol_expr, Integral):
                limits = vol_expr.args[1]
                key = (tuple(limits), None)
                volumes[key] = volumes.get(key, S.Zero) + vol_expr.args[0]
                return
            # Anything else — treat as a boundary-style piece.
            boundaries.append(vol_expr)

        for term in expression.terms:
            r = term.depth_integrate(self._lower, self._upper, self._var,
                                     method=self._method)
            if isinstance(r, DepthIntegralResult):
                _collect_volume(r.volume.expr)
                boundaries.append(r.boundary_upper.expr)
                boundaries.append(r.boundary_lower.expr)
            elif isinstance(r, Expression):
                # ``direct`` returns ``Expression(Integral(expr, lim))``.
                _collect_volume(r.expr)
            else:
                _collect_volume(r)

        pieces = []
        for (limits_tuple, diff_vars), integrand in volumes.items():
            integrand = sp.expand(integrand)
            if integrand == 0:
                continue
            integ = Integral(integrand, limits_tuple)
            if diff_vars is not None:
                integ = Derivative(integ, *diff_vars)
            pieces.append(integ)
        pieces.extend(boundaries)

        total_sp = S.Zero
        for p in pieces:
            total_sp = total_sp + p
        return Expression(total_sp, expression.name)

    def _repr_latex_(self):
        return (
            f"$\\int_{{{sp.latex(self._lower)}}}^{{{sp.latex(self._upper)}}} "
            f"(\\cdot)\\, d{sp.latex(self._var)}$"
        )


class IntegralTransform(Operation):
    """Affine change of variable applied to every `Integral` in an
    expression.  Maps each integration interval to a reference interval
    (default: the unit interval `[0, 1]`) via the affine map:

        `var = a + (b - a) · (ref - ref_lo) / (ref_hi - ref_lo)`

    with Jacobian `|dφ/dref| = (b - a) / (ref_hi - ref_lo)`.

    Unlike substitution-based approaches (`expr.subs(z, ζ·h + b)`),
    this is strictly local to each `Integral`: only the integration
    variable is replaced, only inside the integrand of the Integral
    it's bound to.  Sibling occurrences of `var` in the expression
    — boundary Subs terms, the outer `z` of a running integral after
    an outer transform — are untouched, because those aren't bound by
    *this* Integral.

    Nested Integrals are handled naturally: the walk recurses into
    integrands first (bottom-up), so the innermost Integral is
    transformed before its outer host.  After the outer transform
    replaces `var` throughout its own integrand, any inner Integral's
    now-renamed limits (in the old `var`) get rewritten in terms of
    the outer's reference variable — so a running integral
    `Integral(g(ẑ), (ẑ, b, z))` sitting inside
    `Integral(f(z), (z, b, b+h))` correctly transforms to
    `Integral(g(…), (ẑ′, 0, 1)) · (b + h·ζ − b)` after the outer
    `z → b + h·ζ` step.

    Parameters
    ----------
    ref_interval : tuple (ref_lo, ref_hi), default `(0, 1)`
        The target reference interval.  Always the unit interval in
        typical use.
    ref_name : str, optional
        Base name for the reference-variable Dummies (rendered with
        a hat by default).  Each transformed Integral gets its own
        Dummy — different Dummies never collide even with the same
        display name.
    """

    whole_leaf_op = True

    def __init__(self, ref_interval=(0, 1), ref_name=r"\hat{\zeta}",
                 name="integral_transform", description=None):
        super().__init__(
            name=name,
            description=(description or
                         f"Affine change of variable to reference "
                         f"interval {ref_interval}"),
        )
        self._ref_lo, self._ref_hi = ref_interval
        self._ref_name = ref_name

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative

        # One ``Dummy`` per nesting depth, **shared across siblings**.
        # Two top-level Integrals on the same ``Add`` live in separate
        # scopes and can safely share the same bound variable — and
        # *should*, otherwise sympy's structural equality keeps them as
        # distinct atoms and ``Add`` can't merge ``∫f dζ + ∫f dζ`` into
        # ``2·∫f dζ`` (or cancel ``+∫f dζ − ∫f dζ`` to 0).  Nested
        # Integrals (in an outer's integrand) are processed at the next
        # depth, so they never collide with their host.
        dummies_by_depth: dict[int, sp.Dummy] = {}

        def _ref(depth):
            if depth not in dummies_by_depth:
                dummies_by_depth[depth] = sp.Dummy(
                    f"{self._ref_name}_{{{depth}}}", positive=True)
            return dummies_by_depth[depth]

        def _transform(e, depth=0):
            if isinstance(e, Integral):
                # Bottom-up: transform the integrand first (one level
                # deeper) so any nested Integrals are already in
                # reference form before we substitute the outer var.
                integrand = _transform(e.args[0], depth=depth + 1)
                limits = e.args[1]
                if len(limits) != 3:
                    # unevaluated / indefinite — pass through.
                    if integrand is not e.args[0]:
                        return Integral(integrand, *e.args[1:])
                    return e
                var, a, b = limits
                ref = _ref(depth)
                span = b - a
                ref_span = self._ref_hi - self._ref_lo
                phi = a + span * (ref - self._ref_lo) / ref_span
                jac = span / ref_span
                new_integrand = integrand.subs(var, phi) * jac
                return Integral(new_integrand,
                                (ref, self._ref_lo, self._ref_hi))
            if isinstance(e, Derivative):
                inner = _transform(e.args[0], depth=depth)
                if inner is not e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = tuple(_transform(a, depth=depth) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _transform(expr)


class IsolateBasisIntegrand(Operation):
    """Rewrite each `Integral(integrand, (var, a, b))` so the
    integrand contains *only* factors that depend on `var`.

    Two passes per Integral:

    1. Distribute `Integral(Add(t1, t2, …), …) → Σ Integral(t_i, …)`,
       so each term can be analysed independently.
    2. For each (now single-term) Integral, split the multiplicative
       factors of the integrand into a `var`-independent coefficient
       `C` and a `var`-dependent kernel `K`, and rewrite as
       `C · Integral(K, (var, a, b))`.

    After this, every Integral has the form `coeff · Integral(kernel, …)`
    where `kernel` only contains `var` (and any opaque basis
    functions / derivatives evaluated at `var`).  This is the shape
    a basismatrix-lookup pattern-matches against:
    `∫_0^1 phi_k(z)·phi_l(z) dz`, `∫_0^1 z·phi_k(z)·phi_l(z) dz`,
    etc.

    The Op walks Integrals bottom-up so nested running integrals are
    isolated first.  Each Integral's integration variable `var` is
    read directly from its own limit tuple — there's no ambiguity, and
    different Integrals can carry different ``var``s (e.g. `ζ̂_0`,
    `ζ̂_1`, …).
    """

    whole_leaf_op = True

    def __init__(self, name="isolate_basis_integrand", description=None):
        super().__init__(
            name=name,
            description=(description or
                         "split each Integral into coeff · Integral("
                         "var-dependent kernel, …)"),
        )

    def _leaf_sp(self, expr):
        from sympy import Add, Integral, Mul

        def _isolate(e):
            if isinstance(e, Integral):
                # Recurse into integrand first so nested Integrals are
                # already in isolated form.
                integrand = _isolate(e.args[0])
                limits = e.args[1]
                if not (hasattr(limits, "__len__") and len(limits) == 3):
                    return Integral(integrand, *e.args[1:])
                var = limits[0]
                # Distribute Add: ∫ (a + b) dvar = ∫a dvar + ∫b dvar.
                # We do this BEFORE factoring so each term contributes
                # its own ``coeff · Integral(kernel, lim)`` piece.
                terms = Add.make_args(sp.expand(integrand))
                pieces = []
                for term in terms:
                    if term == S.Zero:
                        continue
                    # Split term into var-independent / var-dependent
                    # factors.  Single non-Mul terms are treated as
                    # one-element products.
                    factors = (Mul.make_args(term) if isinstance(term, Mul)
                               else (term,))
                    consts = [f for f in factors if not f.has(var)]
                    kern_parts = [f for f in factors if f.has(var)]
                    coeff = Mul(*consts) if consts else S.One
                    kernel = Mul(*kern_parts) if kern_parts else S.One
                    if kernel == S.One:
                        # Integrand has no var dependence at all —
                        # ∫ const dvar = const · (b − a).
                        pieces.append(coeff * (limits[2] - limits[1]))
                    else:
                        pieces.append(
                            coeff * Integral(kernel, *e.args[1:])
                        )
                return sp.Add(*pieces) if pieces else S.Zero
            if e.args:
                new_args = tuple(_isolate(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _isolate(expr)


class MapBasisToReference(Operation):
    """Rewrite `phi_k(arg) → phi_k((arg − b) / h)` in every leaf.

    With the `phi_k(state.z)` convention the basis is opaque on the
    raw coordinate `z`.  After `IntegralTransform` substitutes
    `z → ζ·h + b` inside an integrand, integrand basis evaluations
    read `phi_k(ζ·h + b)` — semantically the **physical** basis at
    point `ζ·h + b`.  This Op states the FEM convention that the
    physical basis is the *reference* basis composed with the affine
    map: `phi_k(z) := phi_k_ref((z − b)/h)`.  After the rewrite,
    sympy auto-simplifies arguments — `(b − b)/h → 0`,
    `((b + h) − b)/h → 1`, `((ζ h + b) − b)/h → ζ` — so every
    `phi_k` call lands on a single argument in `[0, 1]`: the
    canonical pattern a basismatrix lookup recognises.

    Also forces the chain rule on any `Derivative(f(t, x, arg), v)`
    whose third argument got rewritten to a (b, h)-dependent form by
    `IntegralTransform` — sympy's structural `subs` keeps a
    `Subs` wrapper around such Derivatives instead of distributing,
    so we explicitly `.doit()` them so the chain-rule metric terms
    materialise inside the integrand.

    Parameters
    ----------
    b, h : sympy expressions
        The bottom topography and column height entering the affine
        map `z = ζ·h + b`.  Typically `state.b` and `state.h`.
    """

    def __init__(self, b, h, name="map_basis_to_reference",
                 description=None):
        super().__init__(
            name=name,
            description=(description or
                         "phi_k(arg) → phi_k((arg − b)/h)"),
        )
        self._b = b
        self._h = h

    def _leaf_sp(self, expr):
        # The chain-rule contributions are already handled correctly
        # by ``ProductRule`` (which fires ``phi_k'(z)`` from the
        # vertical-convection ``∂_z`` and Leibniz boundary terms from
        # the horizontal Derivatives via ``Integrate(method='auto')``).
        # We do **not** attempt to force a second chain rule by
        # ``.doit()``-ing Derivatives — that would double-count, since
        # the moving-boundary metric has already been captured by
        # those upstream operations.
        # Step B: rewrite ``phi_k(arg) → phi_k((arg − b)/h)`` and the
        # Subs counterpart that sympy uses for the chain-rule of an
        # opaque function: ``Subs(Derivative(phi_k(ξ), ξ), ξ, value)
        # → Subs(Derivative(phi_k(ξ), ξ), ξ, (value − b)/h)``.  Both
        # forms appear after ``IntegralTransform``.  Sympy then auto-
        # simplifies the argument when ``arg`` is one of ``b``,
        # ``b + h``, or ``ζ·h + b``: ``(b − b)/h → 0``,
        # ``((b + h) − b)/h → 1``, ``((ζ·h + b) − b)/h → ζ``.
        def _is_basis_call(e):
            return (isinstance(e, sp.Function)
                    and not isinstance(e, sp.Derivative)
                    and getattr(e.func, "__name__", "")
                    .startswith("phi_"))

        def _is_basis_derivative_subs(e):
            """Detect `Subs(Derivative(phi_k(ξ), ξ), ξ, value)` —
            sympy's representation of `phi_k'(value)` for an opaque
            `phi_k`."""
            if not isinstance(e, sp.Subs):
                return False
            inner = e.args[0]
            if not isinstance(inner, Derivative):
                return False
            head = inner.args[0]
            return _is_basis_call(head) and len(head.args) == 1

        def _rewrite(e):
            if _is_basis_call(e):
                new_arg = (e.args[0] - self._b) / self._h
                return e.func(new_arg)
            if _is_basis_derivative_subs(e):
                value = e.args[2][0]
                new_value = (value - self._b) / self._h
                return sp.Subs(e.args[0], e.args[1], (new_value,))
            if e.args:
                new = tuple(_rewrite(a) for a in e.args)
                if any(n is not o for n, o in zip(new, e.args)):
                    return e.func(*new)
            return e
        return _rewrite(expr)


class PartialIntegrate(Operation):
    """Rewrite a matching `Integral(f, (var, lo, hi))` to a running integral
    with variable upper bound: `Integral(f, (var, lo, upper))`.

    The result is left symbolic — no evaluation. Downstream, the expression
    can be promoted to a `Qaux` variable via `DerivedModel.promote_to_qaux`
    and rewritten into a runtime primitive (e.g. `column_partial_integrate`).

    Matching
    --------
    Every `Integral` whose integration variable equals `var` is inspected.
    Additional matchers `lo_match` / `hi_match` restrict the rewrite to
    Integrals whose lower/upper bounds also match (structural equality); pass
    `None` (default) to leave them unrestricted.

    The new upper bound is `upper_symbol` (default: `var` itself, producing
    the classic running integral `∫_lo^{var} f dvar`).

    Usage:

    ```
        # depth integral [b, b+h] → running integral [b, z]
        eq.apply(PartialIntegrate(state.z, lo_match=state.b, hi_match=state.eta))

        # unrestricted: rewrite every ∫ dz to ∫_·^z dz
        eq.apply(PartialIntegrate(z))
    ```
    """

    def __init__(self, var, upper_symbol=None, lo_match=None, hi_match=None,
                 name="partial_integrate"):
        super().__init__(
            name=name,
            description=(f"partial integrate w.r.t. {var} "
                         f"(upper → {upper_symbol if upper_symbol is not None else var})"),
        )
        self._var = var
        self._upper = upper_symbol if upper_symbol is not None else var
        self._lo_match = lo_match
        self._hi_match = hi_match

    def _leaf_sp(self, expr):
        v = self._var
        new_upper = self._upper
        lo_m = self._lo_match
        hi_m = self._hi_match

        def _walk(e):
            if isinstance(e, Integral):
                # sympy stores limits as sp.Tuple, not Python tuple.
                limits = e.args[1] if len(e.args) >= 2 else None
                if limits is not None and len(limits) == 3:
                    int_var, lo, hi = limits[0], limits[1], limits[2]
                    if int_var == v:
                        if (lo_m is None or lo_m == lo) and (hi_m is None or hi_m == hi):
                            new_integrand = _walk(e.args[0])
                            tail = e.args[2:]
                            return Integral(new_integrand, (int_var, lo, new_upper), *tail)
            if hasattr(e, "args") and e.args:
                try:
                    return e.func(*(_walk(a) for a in e.args))
                except (TypeError, ValueError):
                    return e
            return e

        return _walk(expr)

    def _repr_latex_(self):
        return (f"$\\int_{{\\cdot}}^{{{sp.latex(self._upper)}}} "
                f"(\\cdot)\\, d{sp.latex(self._var)}$")


class AffineProjection(Operation):
    """Transform vertical coordinate: z = ζ·(upper−lower) + lower, dz = (upper−lower)·dζ.

    Transforms z-integrals whose bounds match the configured
    `(lower, upper)`:

    ```
        ∫_{lower}^{upper} f(z) dz → (upper − lower) · ∫_0^1 f(ζ·(upper−lower)+lower) dζ
    ```

    With no `lower` / `upper` given, defaults to `state.b` /
    `state.eta` — the full free-surface transform used by the
    single-layer SME walkthrough.  Pass layer-specific bounds
    (e.g. `lower=state.b, upper=z_1`) in the multi-layer case so
    each layer's integrals get mapped to their own ζ-interval.

    Only `Integral` nodes whose bounds structurally equal the
    configured pair are rewritten; other integrals (including
    already-transformed `(zeta, 0, 1)` ones) are left untouched.
    That makes it safe to apply multiple AffineProjections back-to-back
    on a branch with several integration ranges.
    """

    def __init__(self, state, lower=None, upper=None, zeta=None,
                 name=None, description=None, *,
                 rewrite_basis_args=True):
        self._z = state.z
        self._lower = lower if lower is not None else state.b
        self._upper = upper if upper is not None else state.eta
        self._h = self._upper - self._lower
        self._state = state
        self._zeta = zeta if zeta is not None else state.zeta_ref
        self._zeta_fn = state.zeta
        # Control whether to apply ``_rewrite_basis_args`` after
        # ``_transform_outer``.  Default True for backward compatibility
        # (test functions passed with z-only argument, the affine map
        # is encoded only here).  Set False when test functions were
        # already passed with explicit ``(z-b)/h`` argument upstream
        # (e.g. by ``AffinifyBasisArgs``); avoids double-application.
        self._rewrite_basis_args_flag = rewrite_basis_args
        super().__init__(
            name=name or "zeta_transform",
            description=(description or
                         f"z = ζ·({self._upper} − {self._lower}) + {self._lower}"),
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative
        z, lower, upper, h, zeta = (
            self._z, self._lower, self._upper, self._h, self._zeta,
        )

        def _transform_outer(e):
            """Pass 1: map full-layer `∫_lower^upper f(v) dv` → `h·∫_0^1 f(ζ·h+lower) dζ`.

            Matches by bounds, accepting any integration variable — after
            `Integrate(method="auto")` Leibniz can rename the bound
            variable to a `Dummy` (e.g. `\\hat{z}`), so a strict
            `v == state.z` check would silently skip those integrals.
            """
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                if (len(limits) == 3
                        and limits[1] == lower
                        and limits[2] == upper):
                    inner_var = limits[0]
                    new_integrand = integrand.subs(
                        inner_var, zeta * h + lower) * h
                    return Integral(new_integrand, (zeta, S.Zero, S.One))
                return e
            if isinstance(e, Derivative):
                inner = _transform_outer(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_transform_outer(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        # Pass 2 converts a *running* integral ``∫_lower^z f(z') dz'`` (produced
        # by e.g. the w-closure) that's ended up with its upper bound
        # substituted by Pass 1 — so it now reads ``∫_lower^{ζ·h+lower} f(z) dz``
        # — into ζ-space ``h·∫_0^ζ f(ζ'·h+lower) dζ'``.  Without this step
        # EvaluateIntegrals gets handed a mixed-space integral (z-integration
        # variable, ζ-dependent upper bound) and sympy.integrate has to chew
        # through the polynomial layer-expansion of the integrand, slowly.
        zeta_hat = sp.Dummy(rf"\hat{{{zeta}}}", positive=True)
        running_upper = zeta * h + lower

        def _transform_running(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                # Running integrals produced by ``Integrate(z, b, z)`` now
                # carry a ``\hat{z}`` Dummy as the integration variable
                # (see ``Expression.depth_integrate``).  Match any var as
                # long as ``lower`` / ``upper`` match the running pattern
                # — the integrand was built in terms of that inner var,
                # so we substitute *that* var to go into ζ-space.
                if (len(limits) == 3
                        and limits[1] == lower):
                    inner_var = limits[0]
                    bound_diff = limits[2] - running_upper
                    if (bound_diff == 0
                            or (hasattr(bound_diff, "is_zero") and bound_diff.is_zero)
                            or sp.simplify(bound_diff) == 0):
                        inner = _transform_running(integrand)
                        new_integrand = inner.subs(
                            inner_var, zeta_hat * h + lower) * h
                        return Integral(new_integrand,
                                        (zeta_hat, S.Zero, zeta))
                new_integrand = _transform_running(integrand)
                if new_integrand is not integrand:
                    return Integral(new_integrand, *e.args[1:])
                return e
            if isinstance(e, Derivative):
                inner = _transform_running(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_transform_running(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        expr = self._pre_collapse_opaque_zeta(expr)
        result = _transform_running(_transform_outer(expr))
        result = self._collapse_opaque_zeta(result)
        if self._rewrite_basis_args_flag:
            result = self._rewrite_basis_args(result)
        else:
            # Test functions are already in (z-b)/h form upstream;
            # _transform_outer's z→ζh+b substitution leaves their args
            # as ((ζh+b)-b)/h.  Canonicalise to ζ via simplification.
            result = self._simplify_basis_args(result)
        return result

    def _pre_collapse_opaque_zeta(self, expr):
        """Substitute concrete values for opaque ζ-derivatives.

        Must run **before** `_transform_outer` so that
        `Derivative(zeta_fn, v)` patterns are resolved before sympy
        turns them into `Subs` atoms.

        `_simplify_expr` calls `_canonicalize_integral_dummies` which
        alpha-renames integration bound variables (e.g. `z → _\\hat{z}`
        inside integrals).  This turns `Derivative(zeta(t,x,z), z)` into
        `Derivative(zeta(t,x,_\\hat{z}), _\\hat{z})` — a different sympy
        object that `xreplace` with a `state.z`-keyed dict misses.

        Fix: collect all `Derivative` nodes via `atoms()` and match by
        structure — `Derivative(zeta_head(*args_h, v), v)` for any `v`
        (z-like derivative), or `Derivative(zeta_head(*args_h, v), t/x)`
        for the cross-derivatives (t, x).  Use the actual last arg `v` of
        the zeta call to build `(v-b)/h` so the result is correct whether
        `v` is `state.z` or any renamed Dummy.

        No-op when `state.zeta` is a plain `Symbol` (legacy path).
        """
        zeta_fn = self._zeta_fn
        if not isinstance(zeta_fn, sp.Function):
            return expr

        zeta_head = zeta_fn.func
        state = self._state
        b, h = self._lower, self._h

        replacements = {}
        for d in expr.atoms(sp.Derivative):
            inner = d.args[0]
            if not (isinstance(inner, sp.Function) and inner.func == zeta_head):
                continue
            if len(d.variables) != 1:
                continue
            var = d.variables[0]
            # last arg of zeta(t, x[, y,] v) is the z-like coordinate
            v = inner.args[-1]
            zeta_explicit = (v - b) / h
            if var == v:
                # ∂ζ/∂v = 1/h (z-like derivative)
                replacements[d] = sp.S.One / h
            elif var == state.t:
                replacements[d] = (
                    -(sp.Derivative(b, state.t)
                      + zeta_explicit * sp.Derivative(h, state.t)) / h
                )
            elif var == state.x:
                replacements[d] = (
                    -(sp.Derivative(b, state.x)
                      + zeta_explicit * sp.Derivative(h, state.x)) / h
                )
            elif state.has_y and var == state.y:
                replacements[d] = (
                    -(sp.Derivative(b, state.y)
                      + zeta_explicit * sp.Derivative(h, state.y)) / h
                )
        if replacements:
            return expr.xreplace(replacements)
        return expr

    def _collapse_opaque_zeta(self, expr):
        """Materialise the opaque `state.zeta(t,x,[y,]z)` call-sites
        that remain **after** `_transform_outer` has substituted
        `z → ζ_ref · h + lower`.

        Derivative substitutions were already handled by
        `_pre_collapse_opaque_zeta` before the transform.  This
        pass deals with two remaining shapes:

        (ii) Boundary evaluations of the opaque head — Leibniz terms
        leave `phi(ζ(t,x,η))`, `phi(ζ(t,x,b))` in the expression;
        collapse them to `ζ_ref = 1` and `ζ_ref = 0`.

        (iii) Bulk-call collapse: `_transform_outer` rewrites the
        integration variable from `z` to `ζ_ref`, turning the
        opaque head's last arg from `z` to `ζ_ref·h+lower`; map
        the whole call to `ζ_ref` so basis atoms read
        `phi_k(ζ_ref)`.

        No-op if `state.zeta` is not a `Function` (legacy path).
        """
        zeta_fn = self._zeta_fn
        if not isinstance(zeta_fn, sp.Function):
            return expr

        zeta_ref = self._zeta
        b, h = self._lower, self._h
        state = self._state
        args_h = state._args_h

        replacements = {}

        # (ii) boundary evaluations of the opaque head
        replacements[zeta_fn.func(*args_h, self._upper)] = sp.S.One
        replacements[zeta_fn.func(*args_h, self._lower)] = sp.S.Zero

        # (iii) bulk-call collapse
        replacements[zeta_fn.func(*args_h, zeta_ref * h + b)] = zeta_ref

        return expr.xreplace(replacements)

    def _simplify_basis_args(self, expr):
        """Canonicalise basis-function arguments after `_transform_outer`.

        When test functions were passed with explicit `(z-b)/h` arg
        upstream, after `_transform_outer` substitutes `z → ζh+b` in
        integrands the basis args read `((ζh+b)-b)/h` — which sympy
        does not auto-simplify to `ζ`.  This pass walks every basis
        atom and applies `sp.cancel` to the argument to canonicalise.
        Boundary-Subs basis atoms (which still contain `z`) are left
        alone — they'll be evaluated when `Subs.doit()` fires.
        """
        basis_funcs = {
            atom.func
            for atom in expr.atoms(sp.Function)
            if hasattr(atom.func, "_basis")
        }
        for cls in basis_funcs:
            def _simplify(k_arg, z_arg, _cls=cls):
                return _cls(k_arg, sp.cancel(z_arg))
            expr = expr.replace(cls, _simplify)
        return expr

    def _rewrite_basis_args(self, expr):
        """Pass 3 — basis-arg rewrite.

        The basis is reference-defined (on [0, 1]).  Calls of
        `phi_fn(k, arg_z)` in the original physical-z integrand are
        physically interpreted as `phi_fn(k, (arg_z − b)/h)` (the
        affine composition).  After `_transform_outer` substitutes
        `z → ζ·h + lower`, basis atoms carry args like
        `ζ·h + lower` (bulk), `lower` (bottom boundary), or
        `upper = lower + h` (top boundary).  Folding
        `arg → (arg − lower)/h` and simplifying collapses these to
        reference-ζ form: `ζ`, `0`, `1` respectively — exactly
        the canonical pattern `Basisfunction.resolve_atoms` /
        `EvaluateIntegrals` expect.
        """
        lower = self._lower
        h = self._h

        basis_funcs = {
            atom.func
            for atom in expr.atoms(sp.Function)
            if hasattr(atom.func, "_basis")
        }
        for cls in basis_funcs:
            def _rewrite(k_arg, z_arg, _cls=cls):
                new_z = sp.simplify((z_arg - lower) / h)
                return _cls(k_arg, new_z)
            expr = expr.replace(cls, _rewrite)
        return expr

    def _repr_latex_(self):
        return (f"$z = \\zeta \\cdot ({sp.latex(self._upper)} - "
                f"{sp.latex(self._lower)}) + {sp.latex(self._lower)}, "
                f"\\quad dz = ({sp.latex(self._h)}) \\, d\\zeta$")


class SigmaTransform(Operation):
    """Change of variables `(t, x, [y,] z) -> (t, x, [y,] ζ)` with `ζ = (z - b) / h` on a PDE leaf, rewriting derivatives by the chain rule.

    Only first-order derivatives are supported.

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    def __init__(self, state, src=None, *, sigma=None,
                 name="sigma_transform", description=None):
        self._state = state
        self._z = state.z
        # In the new design, h and b live on MassMomentum (``src``),
        # not on StateSpace.  Accept ``src`` as the second positional;
        # fall back to ``state.h`` / ``state.b`` if `src` is None (the
        # legacy StateSpace still has them, so this keeps legacy paths
        # working).
        h_owner = src if src is not None else state
        self._h = h_owner.h
        self._b = h_owner.b
        # Register ``state.zeta_ref`` (Symbol on [0, 1]) if absent —
        # the new minimal StateSpace doesn't pre-declare it.
        if sigma is None:
            sigma = getattr(state, "zeta_ref", None)
            if sigma is None:
                state.zeta_ref = sp.Symbol(r"\hat{\zeta}", real=True)
                sigma = state.zeta_ref
        self._sigma = sigma
        # ``state.coords_h`` may not exist on the new minimal StateSpace;
        # fall back to building it from x, [y].
        coords_h = getattr(state, "coords_h", None)
        if coords_h is None:
            coords_h = [state.x] + ([state.y] if getattr(state, "has_y", False) and state.y is not None else [])
        coords_s = [state.t] + list(coords_h)
        self._chain_coords = tuple(coords_s)            # (t, x[, y])
        super().__init__(
            name=name,
            description=(description or
                         f"σ-transform: ζ = ({self._z} − {self._b}) / {self._h}, "
                         "chain-rule on ∂_t, ∂_x[, ∂_y], ∂_z"),
        )

    def _leaf_sp(self, sp_expr):
        return self._rewrite(sp_expr)

    def _rewrite(self, e):
        """Recursive chain-rule rewrite.

        Convention:
          * inside a `sympy.Function` argument list (the inner
            args of e.g. `u(t, x, z)`), substitute z → ζ — this keeps
            the mapped function `ψ̃(t, x, ζ)` opaque in ζ, so
            `Derivative(ψ̃, ζ)` cannot be sympy-chained back into
            `∂ψ/∂z · h`.
          * free atomic z outside Function arg lists is substituted by
            ζ·h + b (K&T's z = h ζ + h_b — needed for hydrostatic
            pressure  p_H = (η − z) ρ g  → (η − (ζh+b)) ρ g).
          * every `sympy.Derivative` is rewritten per the chain
            rule formulas in the class docstring.
        """
        if isinstance(e, sp.Derivative):
            inner_new = self._rewrite(e.expr)
            result = inner_new
            for var, order in e.variable_count:
                for _ in range(int(order)):
                    result = self._apply_one_derivative(result, var)
            return result
        # Function: replace z by σ in the *argument list*, recurse into
        # each arg, then rebuild.  Catches all of u, v, w, p, tau_ij.
        if isinstance(e, sp.Function):
            new_args = []
            for a in e.args:
                a_renamed = a.xreplace({self._z: self._sigma})
                a_renamed = self._rewrite(a_renamed)
                new_args.append(a_renamed)
            return e.func(*new_args)
        # Generic node with sub-args (Add, Mul, Pow, ...).
        if e.args:
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
            return e
        # Atom (Symbol/Number/...): free z in non-Function context.
        if e == self._z:
            return self._sigma * self._h + self._b
        return e

    def _apply_one_derivative(self, inner_expr, var):
        """Apply one first-order chain-rule step ∂_var to `inner_expr`."""
        sigma = self._sigma
        h, b = self._h, self._b

        if var == self._z:
            # ∂_z ψ → (1/h) · ∂_σ ψ̃
            return (sp.Integer(1) / h) * sp.Derivative(inner_expr, sigma)
        if var in self._chain_coords:
            # ∂_s ψ|_z → ∂_s ψ̃ − (∂_s(σ·h + b) / h) · ∂_σ ψ̃
            # Skip the jacobian term when ``inner_expr`` has no σ
            # dependency — its ∂_σ is identically zero and emitting it
            # would leave a sympy ``Derivative(f, ζ)`` atom that the
            # downstream display has to clean up.  Cleanest case:
            # ``∂_t b`` on a bottom topography ``b(t, x)``: the bottom
            # has no z (hence no σ) so the chain-rule branch must
            # collapse to ``∂_t b`` directly.
            if sigma not in inner_expr.free_symbols:
                return sp.Derivative(inner_expr, var)
            jacobian = sp.Derivative(sigma * h + b, var)
            return (sp.Derivative(inner_expr, var)
                    - (jacobian / h) * sp.Derivative(inner_expr, sigma))
        return sp.Derivative(inner_expr, var)

    def _repr_latex_(self):
        return (rf"$\zeta = \frac{{{sp.latex(self._z)} - {sp.latex(self._b)}}}"
                rf"{{{sp.latex(self._h)}}}, \quad "
                rf"\partial_s \psi |_z = \partial_s \tilde\psi "
                rf"- \frac{{\partial_s(\zeta h + b)}}{{h}} \partial_\zeta \tilde\psi$")


def _decompose_deriv_factor(term, var):
    """Decompose `term` as `(coeff, inner)` so that `term == coeff * Derivative(inner, var)`.

    Returns `None` if no such decomposition exists (term is not of the
    form `coeff * ∂_var(inner)` with a first-order `var`-derivative).
    Used by `_rule_anti_product_rule` to match conjugate pairs.
    """
    if isinstance(term, Derivative):
        if term.variables == (var,):
            return S.One, term.args[0]
        return None
    if isinstance(term, Mul):
        factors = list(term.args)
        ds = [f for f in factors
              if isinstance(f, Derivative) and f.variables == (var,)]
        if len(ds) != 1:
            return None
        d = ds[0]
        inner = d.args[0]
        coeff = Mul(*[f for f in factors if f is not d]) if len(factors) > 1 else S.One
        return coeff, inner
    return None


def _rule_anti_product_rule(expr, *, vars=(), **kwargs):
    """Reverse the product rule: `α·∂_v f + f·∂_v α → ∂_v(α·f)`.

    Walks the additive decomposition of `expr` once per `v` in
    `vars` and greedily pairs up conjugate terms (`α·∂_v f` with
    `f·∂_v α`), folding each matched pair into a single conservative
    `Derivative(α·f, v)` term.  Unmatched Derivative terms and
    Derivative-free terms pass through unchanged.

    The match is signed and accepts a common numerical (rational)
    prefactor on both terms: `k·α·∂_v f + k·f·∂_v α → k·∂_v(α f)`
    pairs cleanly for any common rational `k`.  This is needed
    because earlier pipeline steps (layer-weight `l_α = 1/L`, layer-
    mean closure factors, σ-Galerkin normalisations) typically leave
    a numerical prefactor on both conjugate terms; without absorbing
    it the structural comparison `α == α'` fails on `α/2 ≠ α`.
    """
    for v in vars:
        terms = list(Add.make_args(sp.expand(expr)))
        decomp = {}
        for i, term in enumerate(terms):
            d = _decompose_deriv_factor(term, v)
            if d is not None:
                ci, ii = d
                # Extract a numerical (rational) prefactor from ``ci``
                # so the structural comparison is on the symbolic part
                # only.  ``as_coeff_Mul`` returns ``(numeric_part, rest)``
                # for any Mul / atom.
                k_i, sym_i = ci.as_coeff_Mul()
                decomp[i] = (k_i, sym_i, ii)
        used = set()
        for i in list(decomp):
            if i in used:
                continue
            k_i, sym_i, ii = decomp[i]
            for j in list(decomp):
                if j <= i or j in used:
                    continue
                k_j, sym_j, ij = decomp[j]
                # Match if the symbolic parts swap (sym_i ↔ ij,
                # sym_j ↔ ii) AND the numerical prefactors agree.
                if k_i == k_j and sym_i == ij and sym_j == ii:
                    terms[i] = k_i * sp.Derivative(sym_i * ii, v)
                    terms[j] = S.Zero
                    used.update([i, j])
                    break
        if used:
            expr = Add(*terms)
    return expr


def _rule_apply_aliases(expr, *, aliases=None, **kwargs):
    """Rewrite recognised sub-expressions with human-readable names.

    `aliases` is a dict `{name: value}` — for each pair we
    replace `value` with `name` in the expression.  A direct
    `.subs` matches only if the expanded form structurally contains
    `value`; we first try that, and if nothing changes we run
    `sp.factor` on the current expression before re-trying so that
    multi-term values like `z_1 - b` fire on the factored form
    `α·(z_1 - b)` rather than the expanded `α·z_1 - α·b`.

    Recurses into Derivative / Integral / Add / Mul children so that
    aliases inside a `Derivative(..., t)` also get rewritten.
    """
    aliases = aliases or {}
    if not aliases:
        return expr

    def _try_alias_here(e):
        for name, value in aliases.items():
            e2 = e.subs(value, name)
            if e2 != e:
                e = e2
                continue
            # Fall back to factoring and retrying.  ``factor`` can be
            # slow on big expressions but fires at structural leaves
            # like ``α·z_1 - α·b`` → ``α·(z_1 - b)`` where .subs matches.
            try:
                factored = sp.factor(e)
            except Exception:
                factored = e
            e2 = factored.subs(value, name)
            if e2 != factored:
                e = e2
        return e

    def _walk(e):
        # Apply at the current node first, then **also** recurse into
        # sub-expressions — a partial match at the current level (e.g.
        # ``z_1 - b`` in one Add child) shouldn't block alias
        # substitution inside a sibling child (e.g. ``α·z_1 - α·b`` inside
        # a Derivative), which we only see after factoring the sibling.
        candidate = _try_alias_here(e)
        if candidate != e:
            e = candidate
        if hasattr(e, "args") and e.args:
            new_args = [_walk(a) for a in e.args]
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


def _rule_combine_derivatives(expr, **kwargs):
    """Combine additive `Derivative` terms sharing the same variables.

    `Derivative(f, v) + Derivative(g, v) → Derivative(f + g, v)` (linearity).
    Runs across all additive siblings; terms with other factors (e.g.
    `α(t,x)·Derivative(β, v)` where the coefficient is not a pure
    scalar) are **not** folded — that would need the product rule,
    which is `anti_product_rule`'s job.

    In practice this picks up things the pipeline emits as two
    adjacent conservative terms, e.g.
    `Derivative(α·z_1, t) + Derivative(−α·b, t)` folds to
    `Derivative(α·(z_1 − b), t)` which an `apply_aliases` rule can
    then rename to `Derivative(α·h_0, t)`.
    """
    terms = list(Add.make_args(sp.expand(expr)))
    grouped: dict = {}
    leftovers = []
    for term in terms:
        if isinstance(term, Derivative):
            key = term.args[1:]
            grouped[key] = grouped.get(key, S.Zero) + term.args[0]
            continue
        if isinstance(term, Mul):
            deriv_factors = [f for f in term.args if isinstance(f, Derivative)]
            if len(deriv_factors) == 1:
                d = deriv_factors[0]
                coeff = Mul(*[f for f in term.args if f is not d])
                # Only fold when the remaining factor is a pure scalar
                # (no free_symbols overlap with the Derivative vars).
                deriv_vars_set = set(d.variables)
                coeff_depends = bool(coeff.free_symbols & deriv_vars_set)
                if not coeff_depends and not any(
                        isinstance(a, sp.Function)
                        and any(v in a.free_symbols for v in deriv_vars_set)
                        for a in coeff.atoms(sp.Function)):
                    key = d.args[1:]
                    grouped[key] = grouped.get(key, S.Zero) + coeff * d.args[0]
                    continue
        leftovers.append(term)
    new_parts = list(leftovers)
    for key, inner in grouped.items():
        if inner == 0:
            continue
        new_parts.append(sp.Derivative(inner, *key))
    return Add(*new_parts)


def _rule_collapse_trivial_derivative(expr, **kwargs):
    """`Derivative(c, v) → 0` when `c` doesn't depend on `v`.

    A safety-net rule — `_evaluate_linear_derivatives` already
    handles this during the lightweight simplify, so in practice
    Recombine only hits it when a downstream op re-introduces a
    trivial Derivative (e.g. a manual `sp.Derivative(f, v)` with a
    `v`-free `f`).
    """
    def _walk(e):
        if isinstance(e, Derivative):
            inner = e.args[0]
            if not any(inner.has(v) for v in e.variables):
                return S.Zero
            inner_new = _walk(inner)
            if inner_new is not inner:
                return sp.Derivative(inner_new, *e.args[1:])
            return e
        if hasattr(e, "args") and e.args:
            new_args = [_walk(a) for a in e.args]
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    return _walk(expr)


class Recombine(Operation):
    """Fold expanded expressions back into compact/conservative shapes, e.g. `h·∂_t α + α·∂_t h` -> `∂_t(h α)`.

    `rules` picks from `Recombine.RULES` by name (`anti_product_rule`, `apply_aliases`, `collapse_trivial_derivative`; default all) or takes callables `fn(expr, **kwargs)`; `vars` and `aliases` are passed on as kwargs.
    """

    RULES = {
        "anti_product_rule": _rule_anti_product_rule,
        "combine_derivatives": _rule_combine_derivatives,
        "apply_aliases": _rule_apply_aliases,
        "collapse_trivial_derivative": _rule_collapse_trivial_derivative,
    }

    # Recombine needs to see the whole leaf — pair-matching rules
    # (``anti_product_rule``, ``combine_derivatives``) only work when
    # multiple additive terms are visible at once.
    whole_leaf_op = True

    def __init__(self, rules=None, vars=(), aliases=None,
                 name=None, description=None):
        if rules is None:
            rules = list(self.RULES)
        # Normalise each entry to a callable.
        self._rule_fns = []
        rule_labels = []
        for r in rules:
            if isinstance(r, str):
                if r not in self.RULES:
                    raise KeyError(
                        f"Unknown Recombine rule {r!r}. "
                        f"Available: {sorted(self.RULES)}"
                    )
                self._rule_fns.append(self.RULES[r])
                rule_labels.append(r)
            elif callable(r):
                self._rule_fns.append(r)
                rule_labels.append(getattr(r, "__name__", repr(r)))
            else:
                raise TypeError(
                    f"Recombine rules must be strings or callables "
                    f"(got {type(r).__name__})."
                )
        self._vars = tuple(vars)
        self._aliases = dict(aliases) if aliases else {}
        super().__init__(
            name=name or "recombine",
            description=(description
                         or f"recombine via {', '.join(rule_labels)}"),
        )

    def _leaf_sp(self, expr):
        for rule in self._rule_fns:
            expr = rule(expr, vars=self._vars, aliases=self._aliases)
        return expr


class ProductRule(Operation):
    """Product rule — auto-routes per term based on outermost shape.

    Default `direction="both"` — applied to a leaf, dispatches per
    additive term:

      * Bare `Derivative(Π f_i, v)` outer → **forward**:
        `∂_v(Π f_i) → Σ_i (Π_{j≠i} f_j) · ∂_v(f_i)`.
      * `Mul` carrying a first-order `Derivative(f, v)` factor → **inverse**:
        `coeff · ∂_v(f) → ∂_v(coeff · f) − ∂_v(coeff).doit() · f`
        (gets us into divergence form for the integration step).  When the term
        carries MORE THAN ONE derivative factor — e.g. a σ-metric coefficient
        `∂_x(ζh)` multiplying the field derivative `∂_x ũ` — the rule pulls
        out the **field-bearing** outermost first-order derivative and keeps the
        other derivative factors inside the conserved flux `coeff`.  (The old
        rule bailed whenever a second derivative factor was present, so the
        conservative fold no-op'd on exactly the raw-field σ-metric terms the
        pre-ansatz Galerkin projection needs.)
      * Anything else → unchanged (no-op).

    Use `direction="inverse"` / `direction="forward"` to force one
    direction.

    `variables=None` (default) — act on derivatives w.r.t. any
    `Symbol` named `t`, `x`, `y`, or `z`.  Pass
    `variables=[state.t, state.x, ...]` for an explicit whitelist
    by Symbol identity.

    `fields=None` (default) — when a term carries several allowed first-order
    derivatives, the field-bearing one (the derivative whose argument contains a
    user field application) is pulled out; pass `fields=[u, w, ...]` (heads or
    applications) to disambiguate explicitly against geometry fields (`h`,
    `b`).

    Cross-term combinations (`h·∂_t α + α·∂_t h → ∂_t(h·α)`) are
    not the operation's job — apply ProductRule once on the leaf and
    sympy's `Add` canonicalisation in the next `simplify` pulls
    sibling terms together.
    """

    _DIRECTIONS = ("inverse", "forward", "both")
    _DEFAULT_COORD_NAMES = frozenset({"t", "x", "y", "z"})
    # Apply per term automatically — no caller-side ``apply_to_term`` needed.
    single_term_only = False

    def __init__(self, variables=None, direction="inverse", fields=None):
        if direction not in self._DIRECTIONS:
            raise ValueError(
                f"ProductRule direction must be one of "
                f"{self._DIRECTIONS!r}; got {direction!r}."
            )
        self._direction = direction
        self._vars = tuple(variables) if variables is not None else None
        self._field_heads = None
        if fields is not None:
            self._field_heads = frozenset(
                f if isinstance(f, type) else getattr(f, "func", f)
                for f in fields)
        if variables is None:
            desc = (f"Product rule ({direction}) on d/d{{t,x,y,z}} "
                    "by default coord-name match")
        else:
            desc = (f"Product rule ({direction}) on "
                    f"{{{', '.join(str(v) for v in variables)}}}")
        super().__init__(name="product_rule", description=desc)

    def _var_allowed(self, var):
        if self._vars is not None:
            return var in self._vars
        return getattr(var, "name", None) in self._DEFAULT_COORD_NAMES

    def _inner_has_field(self, expr):
        """True if `expr` carries a user field application — an explicit
        `fields` head if given, else any `AppliedUndef`."""
        from sympy.core.function import AppliedUndef
        if self._field_heads is not None:
            return any(getattr(a, "func", None) in self._field_heads
                       for a in expr.atoms(sp.Function))
        return any(isinstance(a, AppliedUndef) for a in expr.atoms(sp.Function))

    def _pick_inverse_deriv(self, factors):
        """Among the term's factors, choose the ONE first-order derivative (in
        an allowed variable) to pull out: the field-bearing one, tie-broken by
        the largest inner subtree.  Returns `None` when there is no allowed
        first-order derivative factor."""
        cands = [f for f in factors
                 if isinstance(f, Derivative)
                 and len(f.variables) == 1
                 and self._var_allowed(f.variables[0])]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        field_cands = [f for f in cands if self._inner_has_field(f.args[0])]
        pool = field_cands or cands
        return max(pool, key=lambda f: sp.count_ops(f.args[0]))

    def _leaf_sp(self, expr):
        # Auto-route per additive term.  Each term's outermost shape
        # decides forward vs inverse vs no-op; the operation has no
        # cross-term logic.
        out = S.Zero
        for term in Add.make_args(sp.expand(expr)):
            out = out + self._one_term(term)
        return out

    def _one_term(self, term):
        # Forward: bare Derivative whose inner is a product or integer power.
        if (self._direction in ("forward", "both")
                and isinstance(term, Derivative)
                and len(term.variables) == 1):
            var = term.variables[0]
            if not self._var_allowed(var):
                return term
            inner = term.args[0]
            if isinstance(inner, Mul) and len(inner.args) >= 2:
                factors = inner.args
                return sum(
                    (Mul(*(factors[j] for j in range(len(factors)) if j != i))
                     * Derivative(factors[i], var)
                     for i in range(len(factors))),
                    S.Zero,
                )
            if isinstance(inner, sp.Pow):
                base, exp = inner.args
                if isinstance(exp, sp.Integer) and int(exp) >= 2:
                    n = int(exp)
                    return n * base**(n - 1) * Derivative(base, var)
            return term
        # Inverse: a Mul carrying a first-order Derivative factor in an allowed
        # variable.  Pull the OUTERMOST field-bearing such derivative out into
        # divergence form; any OTHER derivative factors (e.g. a σ-metric
        # ∂_x(ζh) coefficient) ride inside the conserved flux ``coeff``.  If
        # ``coeff`` is free of ``var`` the residual ∂_v(coeff) is zero and the
        # rewrite reduces to ∂_v(coeff · f).
        if (self._direction in ("inverse", "both")
                and isinstance(term, Mul)):
            factors = list(term.args)
            d = self._pick_inverse_deriv(factors)
            if d is None:
                return term
            var = d.variables[0]
            inner = d.args[0]
            coeff_factors = [f for f in factors if f is not d]
            coeff = Mul(*coeff_factors) if coeff_factors else S.One
            return (Derivative(coeff * inner, var)
                    - Derivative(coeff, var).doit() * inner)
        return term

    def _repr_latex_(self):
        return ""


# ---------------------------------------------------------------------------
# Material models library
# ---------------------------------------------------------------------------

def _init_relation_expr(obj, rules, name=""):
    """Initialise `obj` (an `Expression` subclass) as an ORIENTED
    MULTI-RULE relation from a `{lhs: rhs}` dict.

    The residual `Σ(lhs − rhs)` is the term-list source of truth; the
    `{lhs: rhs}` rules are recorded in `_as_relation` so `.apply`
    substitutes every one.  This is how constitutive laws and multi-component
    boundary conditions (Newtonian, HydrostaticPressure, …) are built now that
    there is no `Relation` / `Assumption` / `Material` base — a relation
    is simply an oriented Expression.
    """
    rules = {sp.sympify(k): sp.sympify(v) for k, v in dict(rules).items()}
    residual = sp.Add(*[k - v for k, v in rules.items()])
    Expression.__init__(obj, residual, name=name)
    obj._as_relation = rules


class Newtonian(Expression):
    """Newtonian fluid: tau_ij = mu * (du_i/dx_j + du_j/dx_i), mu = rho * nu."""

    def __init__(self, state: StateSpace, nu=None):
        self.nu = nu if nu is not None else Symbol("nu", positive=True)
        subs = self._build(state, self.nu)
        _init_relation_expr(self, subs, name="Newtonian")

    @staticmethod
    def _build(s, nu):
        mu = nu * s.rho
        u, w = s.u, s.w
        x, z = s.x, s.z
        subs = {
            s.tau["xx"]: 2 * mu * Derivative(u, x),
            s.tau["xz"]: mu * (Derivative(u, z) + Derivative(w, x)),
            s.tau["zx"]: mu * (Derivative(w, x) + Derivative(u, z)),
            s.tau["zz"]: 2 * mu * Derivative(w, z),
        }
        if s.has_y:
            v, y = s.v, s.y
            subs.update({
                s.tau["xy"]: mu * (Derivative(u, y) + Derivative(v, x)),
                s.tau["yx"]: mu * (Derivative(v, x) + Derivative(u, y)),
                s.tau["yy"]: 2 * mu * Derivative(v, y),
                s.tau["yz"]: mu * (Derivative(v, z) + Derivative(w, y)),
                s.tau["zy"]: mu * (Derivative(w, y) + Derivative(v, z)),
            })
        return subs


class Inviscid(Expression):
    """Inviscid fluid: all tau_ij = 0."""

    def __init__(self, state: StateSpace):
        subs = {v: S.Zero for v in state.tau.values()}
        _init_relation_expr(self, subs, name="Inviscid")


class materials:
    """Material model library. Usage: materials.newtonian(state)"""
    newtonian = Newtonian
    inviscid = Inviscid


class LayerMeanClosure(Operation):
    """Specialise a σ-resolved system to L piecewise-constant layers.

    For each leaf that depends on the σ-coordinate, project onto each
    layer `α = 1..L`: integrate symbolically over the layer's
    σ-bounds `[σ_{α-1/2}, σ_{α+1/2}]` while keeping `u(t, x, σ)`
    *opaque*, then resolve the surviving bulk integrals using the
    **layer-mean closure** `∫_{layer α} u(σ) dσ = l_α · u_α(t, x)`.
    Interior IBP rules handle `f(σ) · u^k · ∂_σ u` terms via the
    identity `u^k · ∂_σ u = (1/(k+1)) ∂_σ u^{k+1}`.

    The boundary atoms `u(t, x, σ_{α±1/2})` and `w(t, x, σ_{α±1/2})`
    are deliberately **left opaque** in the resulting per-layer leaves.
    That is the user's modeling choice — apply `KinematicBC`
    with `mass_flux=G_{α+1/2}/ρ` at each interface to resolve `w`,
    and apply an interface-velocity choice
    (donor-cell / central / sign-upwind) for the surviving `u`
    boundary atoms.  AHS26 (9) is the standard donor-cell-by-G choice;
    other closures are equally valid and now visible as separate
    substitutions rather than buried in the projection.

    Why **not** substitute a piecewise ansatz before integration:
      * `Piecewise(u_α, σ in layer α)` substituted globally would
        force sympy's integration to *pick a specific value at the
        layer boundary* (dominant-branch convention), silently
        committing the user to a specific (asymmetric) interface
        velocity.
      * `Σ u_α · 𝟙_{[σ_{α-1/2}, σ_{α+1/2}]}(σ)` (Heaviside form)
        trips sympy's Meijer-G fallback when products of Heavisides
        meet ∂_σ in the integrand.

    Keeping `u` opaque and integrating term-by-term avoids both
    traps and surfaces the modeling choice explicitly.

    Parameters
    ----------
    state : StateSpace
        The state-space the operation lives on.
    sigma : sympy.Symbol
        σ-coordinate of the σ-mapped system (typically `state.zeta_ref`).
    u_field : sympy.Function call
        The opaque `u(t, x, σ)` atom whose layer-mean closure we use.
    u_layer : sequence of sympy Function calls
        Per-layer mean velocities `u_α(t, x)`, one per layer α.
    weights : sequence of sympy expressions, optional
        Layer weights `l_α` summing to 1.  Default: uniform
        `l_α = 1/L`.
    max_u_power : int, optional
        Maximum monomial power of `u_field` to recognise in IBP
        patterns `f(σ) · u^k · ∂_σ u` (default 3 — covers
        `∂_σ u`, `u·∂_σ u` (momentum convection), `u²·∂_σ u`
        (cubic shallow-water) and `u³·∂_σ u`).
    name, description
        Standard operation labels for derivation history.
    """

    rank_changes_leaf = True

    def __init__(self, state, sigma, u_field, u_layer,
                 *, weights=None, max_u_power=3,
                 name="layer_mean_closure", description=None):
        super().__init__(
            name=name,
            description=(description or
                         f"Layer-mean closure on {len(u_layer)} layers"),
        )
        self._state = state
        self._sigma = sigma
        self._u_field = u_field
        self._u_layer = list(u_layer)
        self._L = len(u_layer)
        self._max_u_power = max_u_power
        if weights is None:
            weights = [sp.Rational(1, self._L)] * self._L
        weights = list(weights)
        if len(weights) != self._L:
            raise ValueError(
                f"weights has length {len(weights)} but u_layer has "
                f"length {self._L}"
            )
        if sum(weights) != sp.S.One:
            raise ValueError(f"Layer weights must sum to 1, got {sum(weights)}")
        self._weights = weights
        self._sigma_iface = [sum(weights[:k]) for k in range(self._L + 1)]

    def _resolve_layer_integral(self, integral_atom, u_alpha):
        """Resolve one `Integral(integrand, (σ, a, b))` atom.

        Each additive term of the integrand is matched against:
          1. `f(σ) · u^k · ∂_σ u` for `k = 0..max_u_power` — IBP
             via `u^k · ∂_σ u = (1/(k+1)) ∂_σ u^{k+1}`.  Boundary
             atoms `u_field|_{σ=a,b}` stay opaque; the bulk reduces
             to `-1/(k+1) · ∫_a^b (∂_σ f) · u_α^{k+1} dσ`.
          2. No `∂_σ u` factor — substitute `u → u_α` (layer-mean
             closure) and let sympy integrate the σ-polynomial.
        """
        var, a, bnd = integral_atom.limits[0]
        integrand = sp.expand(integral_atom.function)
        du = sp.Derivative(self._u_field, self._sigma)

        result = sp.S.Zero
        for term in sp.Add.make_args(integrand):
            if du not in term.atoms(sp.Derivative):
                bulk_term = term.subs(self._u_field, u_alpha)
                result += sp.integrate(bulk_term, (self._sigma, a, bnd))
                continue
            handled = False
            for k_test in range(self._max_u_power + 1):
                f_wild = sp.Wild("f", exclude=[du, self._u_field])
                if k_test == 0:
                    m = term.match(f_wild * du)
                else:
                    m = term.match(f_wild * self._u_field**k_test * du)
                if (m is not None and m[f_wild] != 0
                        and not m[f_wild].has(self._u_field, du)):
                    f = m[f_wild]
                    u_b = self._u_field.subs(self._sigma, bnd)
                    u_a = self._u_field.subs(self._sigma, a)
                    f_b = f.subs(self._sigma, bnd)
                    f_a = f.subs(self._sigma, a)
                    coef = sp.Rational(1, k_test + 1)
                    boundary = coef * (f_b * u_b**(k_test + 1)
                                       - f_a * u_a**(k_test + 1))
                    df = sp.diff(f, self._sigma)
                    bulk = -coef * sp.integrate(
                        df * u_alpha**(k_test + 1),
                        (self._sigma, a, bnd))
                    result += boundary + bulk
                    handled = True
                    break
            if not handled:
                # Unrecognised pattern — leave as unevaluated Integral
                # so the user sees it.
                result += sp.Integral(term, (self._sigma, a, bnd))
        return result

    def _apply_leaf(self, expression):
        """Integrate opaque u, resolve bulk integrals via layer-mean closure."""
        from zoomy_core.misc.misc import Zstruct
        out = Zstruct()
        for k in range(self._L):
            lo, hi = self._sigma_iface[k], self._sigma_iface[k + 1]
            u_alpha = self._u_layer[k]
            raw = sp.integrate(expression.expr, (self._sigma, lo, hi))
            raw = sp.simplify(raw)
            # Resolve bulk Integral atoms over [lo, hi].
            for atom in list(raw.atoms(sp.Integral)):
                if len(atom.limits) != 1:
                    continue
                var, a, bnd = atom.limits[0]
                if var != self._sigma or a != lo or bnd != hi:
                    continue
                resolved = self._resolve_layer_integral(atom, u_alpha)
                raw = raw.subs(atom, resolved)
            layer_expr = sp.simplify(raw)
            child_name = f"layer_{k + 1}"
            child = Expression(
                layer_expr, child_name,
                term_tags=dict(expression._term_tags),
                tag_order=list(expression._tag_order),
                solver_groups=expression._solver_groups,
            )
            rel = getattr(expression, "_as_relation", None)
            if rel is not None:
                child._as_relation = dict(rel)
            setattr(out, child_name, child)
        return out


# ---------------------------------------------------------------------------
# Assumptions library
# ---------------------------------------------------------------------------

class KinematicBC(Expression):
    """Kinematic boundary condition at a moving interface.

    One explicit signature covering every kinematic-BC pattern in the
    codebase — solid bed, free surface, σ-domain boundary in a
    σ-transformed system, and every internal interface in a multi-layer
    derivation.

    The interface `interface(t, x[, y])` moves with the fluid:

    .. math::

        w\\big|_\\text{at}
          \\;=\\; \\partial_t \\,\\text{interface}
          + u\\big|_\\text{at}\\, \\partial_x \\,\\text{interface}
          \\;[\\,+\\, v\\big|_\\text{at}\\, \\partial_y \\,\\text{interface}\\,]
          \\;+\\; \\frac{\\text{mass\\_flux}}{\\rho}\\,.

    Parameters
    ----------
    state : StateSpace
    interface : sympy expression
        The moving surface, a function of `(t, x[, y])`.  Typical
        choices: `state.b` (bottom), `state.eta` (free surface),
        a freshly-minted `Function("z_k")(t, x[, y])` for an
        internal multi-layer interface.
    at : sympy expression, optional
        Vertical-coordinate value at which the velocity fields are
        sampled.  Defaults to `interface` — the natural physical-z
        choice "evaluate at the interface itself".  Use an explicit
        `at` when the system has been mapped to a reference vertical
        (e.g. `at=sp.S.Zero` after `SigmaTransform` for the
        bottom; `at=sp.S.One` for the surface; per-layer σ values
        in ML).  The substitution is purely structural — replaces
        `state.z` by `at` in the velocity arg lists — so it works
        in physical `(t, x[, y], z)` or post-σ `(t, x[, y], ζ)`
        form regardless.
    mass_flux : sympy expression, optional
        Mass flux `m` per unit area per unit time crossing the
        interface.  Adds `mass_flux / ρ` to the RHS.  Default
        `None` for impermeable interfaces (bottom / free surface).
        Use a fresh `Function` for internal layer interfaces whose
        flux is an unknown of the resulting multi-layer system.
    name, description : str, optional
        History labels.

    Examples
    --------
    Single-layer in physical z (default `at=interface`):

    ```
        sys.apply(KinematicBC(state, state.b))
        sys.apply(KinematicBC(state, state.eta))
    ```

    Single-layer after `SigmaTransform` — evaluate at the σ
    endpoints rather than at the (now opaque) physical interface:

    ```
        sys.apply(KinematicBC(state, state.b,   at=sp.S.Zero))
        sys.apply(KinematicBC(state, state.eta, at=sp.S.One))
    ```

    Multi-layer (interfaces `z_k`; internals carry a mass flux
    `m_k`; each layer's state has its own σ but the formula is the
    same — only `interface`, `at`, and `mass_flux` change per
    call):

    ```
        for k, z_k in enumerate(interfaces):
            m_k = mass_fluxes[k] if 0 < k < N else None
            sys_layer_above.apply(KinematicBC(state_above, z_k, at=0, mass_flux=m_k))
            sys_layer_below.apply(KinematicBC(state_below, z_k, at=1, mass_flux=m_k))
    ```
    """

    def __init__(self, state=None, interface=None, src=None, *, at=None,
                 w=None, u=None, v=None,
                 rho=None, mass_flux=None, name=None, description=None):
        # Two construction modes, same resulting relation:
        #   • StateSpace (multilayer / production):
        #       KinematicBC(state, interface, src, at=…)
        #     — fields and coords read off the ``state``/``src`` namespaces.
        #   • field-based (derivation Model — no StateSpace needed):
        #       KinematicBC(w=w, u=u, interface=…)
        #     — the applied velocity Functions are passed directly, and the
        #     coords are INFERRED from ``w``'s argument list (t, x[, y], z =
        #     last arg).  This is what lets a derivation write
        #     ``model.add_equation("kbc", KinematicBC(w=w, u=u, interface=b))``
        #     with no adapter.
        if w is not None:
            wf, uf, vf = w, u, v
            a = wf.args
            s_t, s_x, s_z = a[0], a[1], a[-1]
            has_y = len(a) == 4
            s_y = a[2] if has_y else None
            rho_fallback = None
        else:
            s = state
            flow = src if src is not None else state
            wf, uf, vf = flow.w, flow.u, getattr(flow, "v", None)
            s_t, s_x, s_z = s.t, s.x, s.z
            has_y = getattr(s, "has_y", False)
            s_y = getattr(s, "y", None)
            rho_fallback = getattr(state, "rho", None)
        if at is None:
            at = interface
        w_at = wf.subs(s_z, at)
        u_at = uf.subs(s_z, at)
        rhs = Derivative(interface, s_t) + u_at * Derivative(interface, s_x)
        if has_y and vf is not None:
            rhs += vf.subs(s_z, at) * Derivative(interface, s_y)
        if mass_flux is not None:
            rho_sym = rho if rho is not None else rho_fallback
            if rho_sym is None:
                raise ValueError(
                    "KinematicBC: `mass_flux` requires a `rho` kwarg "
                    "(or `state.rho` on the legacy StateSpace).")
            rhs = rhs + mass_flux / rho_sym
        if name is None:
            # "evaluated at the interface itself" (default physical-z) →
            # ``kinematic_bc@interface``; "evaluated at a different
            # vertical value" (post-σ, ML per-layer σ values, ...) →
            # ``kinematic_bc@interface|_{at=...}``.
            name = (f"kinematic_bc@{interface}" if at == interface
                    else f"kinematic_bc@{interface}|_{{at={at}}}")
        # A kinematic BC IS an oriented equation ``w|_at = rhs``: build it as
        # an :class:`Expression` from ``sp.Eq`` so it carries ``_as_relation``
        # ({w_at: rhs}) and applies as a substitution — no Relation class.
        super().__init__(sp.Eq(w_at, rhs), name=name)
        # ``Expression.__init__`` doesn't carry a description; attach so
        # history-mermaid labels stay readable.
        flux_tail = f" + {mass_flux}/ρ" if mass_flux is not None else ""
        self.description = description or (
            f"w|_{{at={at}}} = ∂_t {interface} + u|·∂_x {interface}"
            f"{' + v|·∂_y ' + str(interface) if has_y else ''}"
            f"{flux_tail}"
        )


# Back-compat aliases — the canonical physical-z bottom / surface KBCs.
# Used by ``derived_chain/depth_integrated.py``, ``model_derivation.py``,
# the SME / VAM walkthroughs.  New code should call ``KinematicBC``
# directly with explicit ``interface`` (and ``at=`` when not physical-z).
def KinematicBCBottom(state: StateSpace):
    """`w|_{z=b} = ∂_t b + u_b ∂_x b [+ v_b ∂_y b]` (physical z)."""
    return KinematicBC(state, state.b, name="kinematic_bc_bottom")


def KinematicBCSurface(state: StateSpace):
    """`w|_{z=η} = ∂_t η + u_s ∂_x η [+ v_s ∂_y η]` (physical z)."""
    return KinematicBC(state, state.eta, name="kinematic_bc_surface")


# ``InterfaceKBC`` was the previous "arbitrary interface (with mass
# flux)" form; subsumed by :class:`KinematicBC` (which now accepts
# ``mass_flux`` directly).  Keep the symbol so legacy imports resolve.
InterfaceKBC = KinematicBC


# ``ContinuityClosure`` removed — superseded by the composable pipeline
# ``pointwise_continuity = model.continuity.copy()``, then
# ``pointwise_continuity.apply(Integrate(z, lower, z, "auto")).solve_for(w)``.
# The resulting Expression carries an ``_as_relation`` that ``model.apply``
# substitutes.  ``w|_lower`` is left symbolic and is closed by a separate
# ``InterfaceKBC(state, lower, mass_flux=...)`` step — the algebraic
# identity (continuity integrated from the lower interface) is decoupled
# from the kinematic boundary condition.


class NoTangentialBoundaryStress(Expression):
    """Zero tangential normal stress at both surface and bottom.

    Closes the Leibniz boundary terms `tau_xx(z=b)`, `tau_xx(z=eta)`
    (and y-components in 3D) that remain after depth-integrating
    `∂_x tau_xx`.  Uses a Newtonian-style "no flow variation" closure
    at the vertical boundaries.
    """

    def __init__(self, state: StateSpace):
        s = state
        subs = {}
        if "xx" in s.tau:
            subs[s.tau["xx"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["xx"].subs(s.z, s.eta)] = S.Zero
        if s.has_y and "yy" in s.tau:
            subs[s.tau["yy"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["yy"].subs(s.z, s.eta)] = S.Zero
            subs[s.tau["xy"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["xy"].subs(s.z, s.eta)] = S.Zero
            subs[s.tau["yx"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["yx"].subs(s.z, s.eta)] = S.Zero
        _init_relation_expr(self, subs, name="no_tangential_boundary_stress")


class HydrostaticPressure(Expression):
    """p = p_atm + rho * g * (eta - z)"""

    def __init__(self, state: StateSpace):
        s = state
        p_atm = Function("p_atm", real=True)(s.t, *s.coords_h)
        p_hydro = p_atm + s.rho * s.g * (s.eta - s.z)
        _init_relation_expr(self, {s.p: p_hydro}, name="hydrostatic_pressure")


class StressFreeSurface(Expression):
    """Stress-free surface: τ·n|_{z=η} = 0.

    For a free surface with normal n ≈ (0,0,1), this gives:
      τ_xz|_{z=η} = 0,  τ_zz|_{z=η} = 0  (and τ_yz|_{z=η} = 0 in 3D)
    """

    def __init__(self, state: StateSpace):
        s = state
        subs = {}
        # τ_xz at surface = 0
        subs[s.tau["xz"].subs(s.z, s.eta)] = S.Zero
        # τ_zx at surface = 0 (symmetric)
        if "zx" in s.tau:
            subs[s.tau["zx"].subs(s.z, s.eta)] = S.Zero
        # τ_zz at surface = 0
        subs[s.tau["zz"].subs(s.z, s.eta)] = S.Zero
        if s.has_y:
            subs[s.tau["yz"].subs(s.z, s.eta)] = S.Zero
            if "zy" in s.tau:
                subs[s.tau["zy"].subs(s.z, s.eta)] = S.Zero
        _init_relation_expr(self, subs, name="stress_free_surface")


class ZeroAtmosphericPressure(Expression):
    """p_atm = 0 (no atmospheric pressure)."""

    def __init__(self, state: StateSpace):
        p_atm = Function("p_atm", real=True)(state.t, *state.coords_h)
        _init_relation_expr(self, {p_atm: S.Zero}, name="p_atm=0")


class _FieldExpansion:
    """Function-level expansion relation — replaces every call of a
    `Function` (e.g. `u(t, x, arg)`) with a parametric RHS that
    sees the call's argument list.

    Used by `Basis.expand` to substitute the basis ansatz
    `u(t, x, arg) → Σ α_k · φ_k((arg − b)/h)` **everywhere** `u` is
    applied — including inside `Derivative(u(t,x, \\hat{z}), x)`,
    `Integral(u(t,x, ζ̂·h+b), …)`, and the boundary evaluations
    `u(t,x,b)` / `u(t,x,η)`.  A plain `{u(t,x,z): rhs}` dict
    substituted via `xreplace` / `.subs` is structural and only
    matches the exact key, which breaks the moment the argument shape
    changes.  `.replace(fn, handler)` walks the whole tree and
    rewrites every call, so this covers all the cases uniformly.
    """

    def __init__(self, field_fn, rhs_callable, name="field_expansion"):
        self.field_fn = field_fn
        self.rhs_callable = rhs_callable
        self.name = name

    def apply_to(self, expr):
        return expr.replace(self.field_fn, self.rhs_callable)


class assumptions:
    """Assumptions library."""
    kinematic_bc_bottom = KinematicBCBottom
    kinematic_bc_surface = KinematicBCSurface
    hydrostatic_pressure = HydrostaticPressure
    stress_free_surface = StressFreeSurface
    zero_atmospheric_pressure = ZeroAtmosphericPressure


# ---------------------------------------------------------------------------
# Basis: vertical polynomial basis + coefficient functions
# ---------------------------------------------------------------------------

class Basis:
    """Vertical polynomial basis for the SME expansion `u = Σ α_k φ_k(ζ)`.

    Wraps a `Basisfunction` class (`Legendre_shifted` etc.) and
    exposes symbolic pieces for the apply-based walkthrough:

    * `basis.phi` — `Zstruct(phi_0, …, phi_N)` of sympy expressions in
      `state.zeta_ref` (the reference-element symbol on `[0, 1]`).
      Drop-in factor for `Multiply(basis.phi, outer=True)` *after*
      `AffineProjection` has collapsed the opaque `state.zeta` to
      `state.zeta_ref`.
    * `basis.alpha` — `Zstruct(alpha_0, …, alpha_N)` of
      `Function` coefficients `α_k(t, x, [y])`.  Using Functions
      (not bare Symbols) keeps downstream partial derivatives honest and
      makes linear-stability analysis straightforward.
    * `basis.expand(field)` — single substitution dict covering the
      volume evaluation (in ζ-transformed form) and both boundary
      evaluations at `z=b` / `z=η`.  Feed it to `model.apply(...)`.

    Usage:

    ```
        from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
        basis = Basis(state, Legendre_shifted, level=2)
        model.momentum.x.apply(Multiply(basis.phi, outer=True))
        model.apply(AffineProjection(state))
        model.apply(basis.expand(state.u))
    ```

    The volume substitution uses the ζ-transformed key
    `field.subs(z, ζ·h + b)` — apply it **after** `AffineProjection`
    so it matches what's under the integrals.
    """

    def __init__(self, state: StateSpace, basisfunction_cls, level=0,
                 alpha_name="alpha", **basis_kwargs):
        """Wrap a `Basisfunction` subclass for the apply-based pipeline.

        `basis_kwargs` are forwarded to the `basisfunction_cls` constructor
        — use them for bases that need extra structure beyond the level
        index (e.g. `PiecewiseConstant(interfaces=[state.b, z_1, state.eta])`).
        When `basis_kwargs` is non-empty, `level` is ignored and
        re-read from the instantiated basisfunction.
        """
        from zoomy_core.misc.misc import Zstruct

        self._state = state
        if basis_kwargs:
            self._bf = basisfunction_cls(**basis_kwargs)
            self.level = self._bf.level
        else:
            self._bf = basisfunction_cls(level=level)
            self.level = level
        self._alpha_name = alpha_name

        zeta = state.zeta_ref
        self.phi = Zstruct(**{
            f"phi_{k}": self._bf.eval(k, zeta)
            for k in range(self.level + 1)
        })
        args_h = state._args_h
        self.alpha = Zstruct(**{
            f"{alpha_name}_{k}": Function(f"{alpha_name}_{k}", real=True)(*args_h)
            for k in range(self.level + 1)
        })
        # Test functions expressed as functions of z: phi_l((z-b)/h).
        # Use this BEFORE depth-integration so the test function
        # participates in the Leibniz rule and produces the W_sigma-like
        # coupling terms on ∂_t phi, ∂_x phi.  Use ``self.phi`` (in zeta)
        # AFTER AffineProjection, inside integrands.
        zeta_of_z = (state.z - state.b) / state.h
        self.phi_of_z = Zstruct(**{
            f"phi_{k}": self._bf.eval(k, zeta_of_z)
            for k in range(self.level + 1)
        })
        # Piecewise-constant bases that live in physical z-space expose
        # their breakpoints directly — this is what the layer-wise
        # walkthrough reads to pull z_i and u_i.
        self.interfaces = getattr(self._bf, "interfaces", None)

    def _sum(self, zeta_value):
        """Return `Σ α_k · φ_k(zeta_value)`."""
        return sum(
            getattr(self.alpha, f"{self._alpha_name}_{k}")
            * self._bf.eval(k, zeta_value)
            for k in range(self.level + 1)
        )

    def expand(self, field):
        """Substitution dict for every bulk / boundary evaluation of `field`.

        Four keys:

        * `field`                    → `Σ α_k(t,x) · φ_k((z−b)/h)`
          (pointwise form — matches `field(t,x,z)` in inner running
          integrals produced by the w-closure pipeline).
        * `field.subs(z, ζ·h + b)` → `Σ α_k(t,x) · φ_k(ζ)`
          (ζ-transformed form occurring inside outer ζ-integrals).
        * `field.subs(z, b)`         → `Σ α_k · φ_k(0)` (bottom eval).
        * `field.subs(z, b + h)`     → `Σ α_k · φ_k(1)` (surface eval).

        Including the pointwise key is free — it only fires where the
        expression structurally carries `field(t, x, z)` — and makes
        the w-closure's inner `∫_b^z ∂_x u dz'` close cleanly once
        `u` has been expanded.
        """
        state = self._state
        z, zeta, b, h = state.z, state.zeta_ref, state.b, state.h

        # Build a function-level rewriter.  A plain ``{u(t,x,z): rhs}``
        # dict substituted via ``xreplace`` is structural and won't match
        # ``u(t,x,ζ·h+b)``, ``u(t,x,\hat{z})`` or any other
        # argument form.  ``.replace(u, lambda *args: ...)`` rewrites
        # every call of ``u``, regardless of its third argument, so the
        # running-integral / basis / boundary evaluations all close
        # through a single rule.
        field_fn = field.func  # the raw Function (e.g. ``u``)

        def _field_rhs(*args):
            arg_z = args[-1]
            # Volume / running / boundary cases collapse into one
            # parametric rule: φ_k((arg − b)/h).  At arg=b → φ_k(0);
            # at arg=η → φ_k(1); at arg=ζ·h+b → φ_k(ζ).
            return self._sum((arg_z - b) / h)

        return _FieldExpansion(field_fn, _field_rhs)

    def layer_expand(self, field, layer_idx, zeta_transformed=False):
        """Per-layer closure substitution for a `LayeredBasis`: inside layer `layer_idx` the field is closed against that layer's coefficients.

        Args:
            field: Variable being closed (e.g. `state.u`).
            layer_idx: Index into `self._bf.interfaces` specifying the layer.
            zeta_transformed: `False`: the bulk key is `field` itself; `True` (after `AffineProjection`): `field.subs(z, ζ·h_i + z_i)`.
        Returns:
            Substitution dict with three entries (volume, lower and upper interface of the current layer).
        Raises:
            ValueError: If the wrapped basis function is not a `LayeredBasis`.
        """
        from zoomy_core.model.derivation.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError(
                "Basis.layer_expand requires a LayeredBasis "
                "(e.g. LayeredBasis(Monomials, interfaces=[b, z_1, eta]))."
            )
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(
                f"layer_idx={layer_idx} out of range for "
                f"{self._bf.n_layers} layer(s)."
            )
        state = self._state
        lower = self._bf.interfaces[layer_idx]
        upper = self._bf.interfaces[layer_idx + 1]

        def _closure(z_value):
            """Σ α_{i*m+k} · φ_inner_k(ζ_i(z_value))."""
            zeta_local = self._bf.layer_zeta(layer_idx, z_value)
            return sum(
                getattr(self.alpha,
                        f"{self._alpha_name}_"
                        f"{self._bf.flat_index(layer_idx, k)}")
                * self._bf.inner.eval(k, zeta_local)
                for k in range(self._bf.inner_n_basis)
            )

        if zeta_transformed:
            # After AffineProjection the bulk form is ``field.subs(z, ζ·h_i + lower)``.
            # ``AffineProjection`` + the lightweight simplify path store the
            # subbed argument in **expanded** form (``ζ·upper − ζ·lower + lower``),
            # while a naive ``ζ·(upper−lower)+lower`` stays factored.
            # We need the bulk key to match the expanded form produced by
            # AffineProjection, because ``xreplace`` is structural and
            # ``a·(b−c)+c`` isn't the same subtree as ``a·b − a·c + c``.
            zeta_of_z = sp.expand(state.zeta_ref * (upper - lower) + lower)
            bulk_key = field.subs(state.z, zeta_of_z)
            # RHS shortcut: after AffineProjection the bulk argument is
            # identically ``ζ``, so we evaluate the inner basis at
            # ``state.zeta_ref`` directly.  Going through ``_closure``
            # would produce ``ζ·(upper-lower)/(upper-lower)`` which sympy
            # doesn't cancel (it can't rule out ``upper == lower``),
            # leaving unsimplified ratios inside integrands and
            # blocking ``EvaluateIntegrals`` from collapsing them.
            bulk_val = sum(
                getattr(self.alpha,
                        f"{self._alpha_name}_"
                        f"{self._bf.flat_index(layer_idx, k)}")
                * self._bf.inner.eval(k, state.zeta_ref)
                for k in range(self._bf.inner_n_basis)
            )
        else:
            bulk_key = field
            bulk_val = _closure(state.z)

        return {
            bulk_key: bulk_val,
            field.subs(state.z, lower): _closure(lower),
            field.subs(state.z, upper): _closure(upper),
        }

    def layer_phi(self, layer_idx):
        """Zstruct of inner test functions in the shared `ζ` symbol.

        Equivalent to `basis.phi` but for one layer of a
        `LayeredBasis`.  Use after AffineProjection, inside
        integrands, when you need `φ_inner_k(ζ)` multiplied into an
        `EvaluateIntegrals` target.

        Raises `ValueError` on a non-layered basis.
        """
        from zoomy_core.misc.misc import Zstruct
        from zoomy_core.model.derivation.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError("layer_phi requires a LayeredBasis.")
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range.")
        zeta = self._state.zeta_ref
        return Zstruct(**{
            f"phi_{k}": self._bf.inner.eval(k, zeta)
            for k in range(self._bf.inner_n_basis)
        })

    def layer_phi_of_z(self, layer_idx):
        """Zstruct of inner test functions at `ζ_i(z) = (z − z_i)/h_i`.

        Equivalent to `basis.phi_of_z` but for one layer of a
        `LayeredBasis`.  Use **before** depth-integration —
        `Multiply(basis.layer_phi_of_z(i), outer=True)` on a layer's
        branch produces its Galerkin test equations, one per inner
        moment, each carrying the rescaled `φ_inner_k((z − z_i)/h_i)`
        factor that the Leibniz rule will dismantle.

        Raises `ValueError` on a non-layered basis.
        """
        from zoomy_core.misc.misc import Zstruct
        from zoomy_core.model.derivation.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError("layer_phi_of_z requires a LayeredBasis.")
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range.")
        zeta_i_of_z = self._bf.layer_zeta(layer_idx, self._state.z)
        return Zstruct(**{
            f"phi_{k}": self._bf.inner.eval(k, zeta_i_of_z)
            for k in range(self._bf.inner_n_basis)
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplify_derivatives_only(expr):
    """Simplify Derivative sums without touching Integrals.

    Turns `d(H+b)/dt - db/dt` into `dH/dt` while preserving
    `Derivative(Integral(...), x)` and bare `Integral(...)` nodes.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # Protect anything involving Integrals
    integral_map = {}
    counter = [0]

    def _protect(e):
        # Protect Derivative(Integral(...), ...) as a unit
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        # Protect bare Integrals
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Now safe to simplify — only pure Derivative(Function, var) remain
    simplified = protected.doit() if protected.has(Derivative) else protected
    # Restore integral-containing nodes via xreplace.  ``.subs`` would
    # wrap the result in ``Subs(_, _INT0, Integral(_, …, z))`` whenever
    # ``_INT0`` lives inside a ``Derivative(_, z)`` and the Integral has
    # ``z`` in its limits — sympy's deliberate caution.  ``xreplace`` is
    # purely structural, so the protected Dummy is replaced in place
    # without the Subs wrapping.
    return simplified.xreplace(integral_map)


def _expand_preserve_integrals(expr):
    """`sp.expand(expr)` that does NOT fragment `Integral(f + g, lim)`
    into `Integral(f, lim) + Integral(g, lim)`.

    Built-in `sp.expand` treats Integrals as transparent and distributes
    Add across the integrand — which undoes the cross-term consolidation
    that `Integrate` performs.  This helper protects every Integral as
    a Dummy, runs the normal expand, then restores — so Mul still
    distributes over Add on the outside but Integrands stay intact.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    integral_map = {}
    counter = [0]

    def _protect(e):
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = tuple(_protect(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    protected = _protect(expr)
    return sp.expand(protected).xreplace(integral_map)


def _canonicalize_integral_dummies(expr):
    """Rename every `Integral` 's bound variable to a depth-keyed
    canonical `Dummy`.

    Sympy treats Integrals with different bound-variable names as
    structurally distinct atoms, even when they are alpha-equivalent
    (`∫ f(z) dz` vs `∫ f(ẑ) dẑ`).  That blocks `Add`
    canonicalisation from cancelling duplicates and `merge_integrals`
    from grouping by `(limits, deriv-wrapper)` signature.

    Renaming each Integral's bound variable to a canonical Dummy keyed
    by *nesting depth* (one Dummy per depth, shared across siblings)
    makes alpha-equivalent integrals at the same depth structurally
    equal — exactly the rule `IntegralTransform` already enforces for
    its own outputs (see `IntegralTransform._leaf_sp`), generalised
    to every Integral in the expression.

    Nested integrals get a fresh Dummy per depth, so the inner's bound
    variable never collides with the outer's.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    canon_by_depth: dict[int, sp.Dummy] = {}

    def _hat_name(old_var):
        # The canonical bound variable is the INPUT variable, hatted:
        # ``∫ … dζ`` → ``\hat{\zeta}``, ``∫ … dz`` → ``\hat{z}``.  Already
        # hatted vars (a fresh running-integral Dummy ``\hat{\zeta}``) are
        # kept verbatim so re-canonicalisation never double-hats.
        nm = getattr(old_var, "name", "")
        if nm.startswith(r"\hat{") and nm.endswith("}"):
            return nm
        return rf"\hat{{{sp.latex(old_var)}}}"

    def _canon(depth, old_var):
        # One Dummy per nesting depth (alpha-equivalence of sibling
        # integrals is preserved); the NAME follows the input variable.
        if depth not in canon_by_depth:
            canon_by_depth[depth] = sp.Dummy(_hat_name(old_var), positive=True)
        return canon_by_depth[depth]

    def _walk(e, depth=0):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0], depth=depth + 1)
            limits = e.args[1]
            if hasattr(limits, "__len__") and len(limits) == 3:
                old_var, lo, hi = limits
                new_var = _canon(depth, old_var)
                if old_var is new_var:
                    if integrand is not e.args[0]:
                        return Integral(integrand, *e.args[1:])
                    return e
                new_int = integrand.xreplace({old_var: new_var})
                new_lo = (lo.xreplace({old_var: new_var})
                          if isinstance(lo, sp.Basic) else lo)
                new_hi = (hi.xreplace({old_var: new_var})
                          if isinstance(hi, sp.Basic) else hi)
                return Integral(new_int, (new_var, new_lo, new_hi))
            if integrand is not e.args[0]:
                return Integral(integrand, *e.args[1:])
            return e
        if isinstance(e, Derivative):
            inner = _walk(e.args[0], depth=depth)
            if inner is not e.args[0]:
                return Derivative(inner, *e.args[1:])
            return e
        if e.args:
            new_args = tuple(_walk(a, depth=depth) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


def _split_integrals_expr(expr):
    """Distribute every `Integral(Add(t1, t2, ...), lim)` into a sum of
    single-term integrals `Σ Integral(t_i, lim)`.

    Pushes through `Derivative(Integral(...), v)` by linearity.  Outer
    multiplicative factors stay attached: after `sp.Add.make_args` on
    the Mul-expanded outer expression, each summand contains at most one
    Integral, and that Integral now has a single-term integrand.

    This is the inverse of `_merge_integrals_expr` and is the
    "user-visible" rest state of an Expression — one Integral per
    logical equation term, so per-term operations
    (`apply_to_term` / `terms[i].apply(...)`) act on what the user
    reads as one term.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, Integral):
            integrand = _walk(e.args[0])
            limits = e.args[1:]
            expanded = _expand_preserve_integrals(integrand)
            terms = Add.make_args(expanded)
            if len(terms) <= 1:
                if integrand is not e.args[0]:
                    return Integral(integrand, *limits)
                return e
            return Add(*[Integral(t, *limits) for t in terms])
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            inner = _walk(e.args[0])
            wrt = e.args[1:]
            if isinstance(inner, Add):
                return Add(*[Derivative(a, *wrt) for a in inner.args])
            if inner is not e.args[0]:
                return Derivative(inner, *wrt)
            return e
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _expand_preserve_integrals(_walk(expr))


def _merge_integrals_expr(expr):
    """Combine sibling Integrals with matching `(limits, deriv-wrapper)`
    signature into a single `Integral(Σ c_i · f_i, lim)`.

    Walks every `Add` and groups summands by:
      * the limits tuple of the contained Integral, and
      * the variables of any `Derivative(...)` wrapper around it.

    Each term contributes `coeff · Integrand` to the merged integrand,
    where `coeff` is the multiplicative outer factor that does **not**
    contain the integration variable (so it can safely move inside).
    Terms whose outer factor *does* depend on the integration variable
    pass through unchanged — moving them inside would change the math.

    Inverse of `_split_integrals_expr`.  Used internally by passes
    that need cancellations between sibling integrands to fire (the
    `Integrate(method="auto")` collector already does an in-pass
    version of this); exposed publicly via
    `Expression.merge_integrals` for users who want it.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    # Canonicalise bound variables first so alpha-equivalent integrals
    # (``∫ f dz`` vs ``∫ f dẑ``) reduce to the same signature.
    expr = _canonicalize_integral_dummies(expr)

    def _classify(term):
        """Return `(sig, coeff, integrand, limits, deriv_wrt)` or None."""
        coeff_factors = []
        target = term
        if isinstance(term, sp.Mul):
            cands = []
            for f in term.args:
                if isinstance(f, Integral):
                    cands.append(f)
                elif isinstance(f, Derivative) and isinstance(f.args[0], Integral):
                    cands.append(f)
                else:
                    coeff_factors.append(f)
            if len(cands) != 1:
                return None
            target = cands[0]
        elif isinstance(term, Integral):
            pass
        elif isinstance(term, Derivative) and isinstance(term.args[0], Integral):
            pass
        else:
            return None
        if isinstance(target, Derivative):
            inner = target.args[0]
            deriv_wrt = tuple(target.args[1:])
        else:
            inner = target
            deriv_wrt = ()
        if not isinstance(inner, Integral):
            return None
        limits = inner.args[1]
        if not (hasattr(limits, "__len__") and len(limits) == 3):
            return None
        var = limits[0]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else S.One
        if coeff.has(var):
            return None
        sig = (limits, deriv_wrt)
        return sig, coeff, inner.args[0], limits, deriv_wrt

    def _walk(e):
        if isinstance(e, Add):
            new_args = [_walk(a) for a in e.args]
            groups = {}
            order = []
            passthrough = []
            for a in new_args:
                cls = _classify(a)
                if cls is None:
                    passthrough.append(a)
                    continue
                sig, coeff, integrand, limits, deriv_wrt = cls
                if sig not in groups:
                    groups[sig] = (limits, deriv_wrt, S.Zero)
                    order.append(sig)
                lim, dwrt, acc = groups[sig]
                groups[sig] = (lim, dwrt, acc + coeff * integrand)
            merged = []
            for sig in order:
                lim, dwrt, integrand_sum = groups[sig]
                integrand_sum = sp.expand(integrand_sum)
                if integrand_sum == 0:
                    continue
                term = Integral(integrand_sum, lim)
                if dwrt:
                    term = Derivative(term, *dwrt)
                merged.append(term)
            return Add(*merged, *passthrough)
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


def _kill_zero_length_integrals(expr):
    """`Integral(_, (var, a, a)) → 0`.

    Sympy doesn't auto-simplify this on construction (only via `.doit()`).
    Empty-interval integrals appear naturally in our pipeline whenever an
    outer Subs evaluates a running integral's upper bound at the same
    point as its lower bound (e.g. evaluating `∫_b^z f dẑ` at `z=b`
    leaves `∫_b^b f dẑ = 0`).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for I in expr.atoms(Integral):
        limits = I.args[1]
        if hasattr(limits, "__len__") and len(limits) == 3:
            _, lo, hi = limits
            if lo == hi or sp.simplify(lo - hi) == 0:
                mapping[I] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


def _kill_free_derivatives(expr):
    """Replace `Derivative(sym_a, sym_b)` with 0 when both are plain free
    `Symbol` s (not `Function` s).  These arise from Leibniz boundary
    terms where the upper/lower limit is a bare coordinate symbol (e.g.
    `∂_x z = 0`); sympy does not auto-reduce them and they propagate
    through the pipeline as spurious boundary residuals.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(Derivative):
        inner = d.args[0]
        if not isinstance(inner, Symbol):
            continue
        variables = []
        for v in d.args[1:]:
            variables.append(v[0] if isinstance(v, tuple) else v)
        if all(isinstance(v, Symbol) for v in variables) and all(v != inner for v in variables):
            mapping[d] = S.Zero
    return expr.xreplace(mapping) if mapping else expr


def _simplify_preserve_integrals(expr):
    """Expand + like-term collection, protecting `Integral` and
    `Derivative(Integral)` as atoms.

    No `cancel` / `together` / `ratsimp` — those build common
    denominators and turn polynomial Derivative combinations into rational
    functions.  Just ``expand + powsimp + linearity-only derivative
    distribution`` + a pre-pass that kills trivially-zero
    `Derivative(free_symbol, other_free_symbol)` boundary residuals.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    expr = _kill_free_derivatives(expr)
    expr = _kill_zero_length_integrals(expr)
    # Alpha-rename every Integral's bound variable to a depth-keyed
    # canonical Dummy so structurally-equivalent integrals (e.g. one
    # with `z`, another with `\hat z`, both spanning [b, b+h]) match
    # under sympy's ``Add`` canonicalisation.  Without this, sibling
    # integrals from different sources never cancel.
    expr = _canonicalize_integral_dummies(expr)

    integral_map = {}
    counter = [0]

    def _protect(e):
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Three-pass simplify — all internal to this one pipeline:
    #
    #   Pass 1 "expand": ``_evaluate_linear_derivatives`` distributes
    #   Derivative over Add AND pulls purely-numeric scalar factors out
    #   of Derivative inners (``Derivative(-α₁, x) → -Derivative(α₁, x)``,
    #   ``Derivative(c·f, v) → c·Derivative(f, v)``).  This canonicalizes
    #   signs / coefficients so structurally-equivalent forms actually
    #   match as sympy objects.
    #
    #   Pass 2 "simplify": ``sp.expand + sp.powsimp``.  Cancellations
    #   fire on the canonical form — e.g. ``(-∂α₁)² / ∂α₁`` reduces to
    #   ``∂α₁`` because sympy's Mul canonicalization finally sees
    #   matching bases.
    #
    #   Pass 3 "collect": ``_fold_numeric_coeffs_into_derivative`` is
    #   the inverse of Pass 1 — re-absorb numeric coefficients back into
    #   Derivative inners so fluxes read conservatively
    #   (``Derivative(g·h²/2, x)`` rather than ``(1/2)·Derivative(g·h², x)``).
    #   Safe because Pass 2 already collapsed the problematic rational
    #   forms; re-absorbing ``-1`` / ``1/2`` / etc. cannot resurrect them.
    #
    # Product-rule recombine (``h·∂_t α + α·∂_t h → ∂_t(h·α)``) is
    # **not** applied automatically — it's a deliberate user step.
    # Apply ``ProductRule()`` (default inverse) to one sibling; the
    # residual cancels the other under the next ``.simplify()`` pass
    # via sympy Add's combine-like-terms.  The automatic simplify
    # deliberately stays narrow so the user decides when the collapse
    # is the right move.
    #
    # Linearity only: no pass chain-rules through ``Derivative(u², x)``
    # or ``Derivative(u·w, z)`` — conservative shapes stay intact so
    # ``Integrate(..., method="auto")`` can still Leibniz / fund-thm.
    # ``Derivative(Integral(...))`` is already protected above.
    #
    # Fixpoint loop: one expand→simplify→collect pass may expose
    # further cancellations once Pass-3-collected conservative
    # shapes get re-expanded on the next iteration (the collected
    # form ``Derivative(c·f + c·g, x)`` expands to an Add inner that
    # Pass 1 then distributes).  Bound is generous; single-layer SME
    # reaches a fixed point in ≤ 3 iterations.
    result = protected
    for _ in range(6):
        prev = result
        evaled = _evaluate_linear_derivatives(result)
        expanded = sp.expand(evaled)
        # ``powsimp(combine="all")`` is too aggressive — it can build
        # rational-function common denominators that downstream treats
        # as unresolvable integrands (e.g. ``1/(h·∂_x α₁ − α₁·∂_x h)``
        # appearing inside a depth integral).  Plain ``expand`` already
        # collects like terms; skip powsimp entirely.
        result = _fold_numeric_coeffs_into_derivative(expanded)
        if result == prev:
            break
    # Recurse into the integrands we protected: the outer expand only
    # simplified the expression OUTSIDE Integrals.  Inside each Integral,
    # the integrand may contain its own cancellable terms (eg. the
    # ``±u·∂_t b · φ'`` pair from the Leibniz-rule mixing).  Without this
    # recursion, those cancellations never fire because each Integral is
    # one atom to the outer Add canonicalisation.
    for key, integral in list(integral_map.items()):
        if isinstance(integral, Integral):
            new_integrand = _simplify_preserve_integrals(integral.args[0])
            if new_integrand is not integral.args[0]:
                integral_map[key] = Integral(new_integrand, *integral.args[1:])
        elif isinstance(integral, Derivative) and integral.args[0].has(Integral):
            # Derivative(Integral(...), var) — simplify the inner Integral.
            new_inner = _simplify_preserve_integrals(integral.args[0])
            if new_inner is not integral.args[0]:
                integral_map[key] = Derivative(new_inner, *integral.args[1:])
    # ``xreplace`` (not ``.subs``) for the restore: ``.subs`` wraps the
    # result in ``Subs(_, _INT0, Integral(_, …, z))`` whenever the
    # protected Dummy lives inside a ``Derivative(_, z)`` and the
    # Integral has ``z`` in its limits — sympy's deliberate caution
    # against differentiating a z-dependent integral.  Structural
    # ``xreplace`` skips that wrapping.
    restored = result.xreplace(integral_map)
    # Rest-state: split Integrals so the user-visible ``.terms`` view
    # exposes one logical equation term per Integral.  Cross-term
    # cancellations have already happened above (sympy's Add
    # canonicalisation on the integrands inside each protected Integral
    # plus structural matching across siblings that share dummies);
    # splitting after that is purely a display rewrite.  Operations
    # that need merged form (``Integrate(method="auto")``, the merge
    # pass inside ``IsolateBasisIntegrand``) merge internally and don't
    # rely on input being merged.
    return _split_integrals_expr(restored)


# Mode-index letters for hygienic renaming (``l`` reserved as the test index).
_MODE_LETTERS = ["i", "j", "k", "m", "n", "o", "p", "q", "r", "s"]
# Alternative hatted dummy names for nested integration variables.
_HAT_NAMES = [r"\hat{\xi}", r"\hat{\eta}", r"\hat{\chi}", r"\hat{\vartheta}"]


def _bound_atoms(expr):
    """Summation indices + integration dummies bound anywhere in `expr`,
    together with its free symbols — the names a fresh insertion must avoid."""
    if not isinstance(expr, sp.Basic):
        return set()
    idx = {lim[0] for s in expr.atoms(sp.Sum) for lim in s.limits}
    dum = {lim[0] for ig in expr.atoms(sp.Integral) for lim in ig.limits}
    return idx | dum | set(expr.free_symbols)


def _alpha_fresh(rhs, avoid):
    """Alpha-rename `rhs`'s OWN summation indices and integration dummies that
    collide with `avoid` to fresh symbols, so substituting `rhs` inside a
    sum / integral of the target can never capture the target's index or dummy.
    Names that do NOT collide are kept verbatim."""
    if not isinstance(rhs, sp.Basic):
        return rhs
    avoid = set(avoid)
    avoid_names = {getattr(a, "name", None) for a in avoid}
    renames: dict = {}

    def _fresh_mode(idx):
        for L in _MODE_LETTERS:
            cand = sp.Symbol(L, integer=True, nonnegative=True)
            if cand not in avoid and cand not in renames.values():
                return cand
        return sp.Dummy(getattr(idx, "name", "i") + "'",
                        integer=True, nonnegative=True)

    def _fresh_hat(dv):
        taken = avoid_names | {getattr(v, "name", None)
                               for v in renames.values()}
        for nm in _HAT_NAMES:
            if nm not in taken:
                return sp.Dummy(nm, positive=True)
        return sp.Dummy(getattr(dv, "name", r"\hat{\xi}") + "'", positive=True)

    # Summation indices are named Symbols (compare by name) — an equality test
    # against ``avoid`` catches a collision.
    for s in rhs.atoms(sp.Sum):
        for lim in s.limits:
            idx = lim[0]
            if idx in avoid and idx not in renames:
                renames[idx] = _fresh_mode(idx)
    # Integration dummies compare by IDENTITY, so a distinct-object ``\hat{ζ}``
    # would print identically to the target's yet pass an ``in`` test — match on
    # NAME instead, so a nested running integral gets its own readable dummy.
    for ig in rhs.atoms(sp.Integral):
        for lim in ig.limits:
            dv = lim[0]
            if (getattr(dv, "name", None) in avoid_names
                    and dv not in renames):
                renames[dv] = _fresh_hat(dv)
    return rhs.xreplace(renames)


def _is_function_definition(lhs):
    """A relation lhs `f(a1, …, an)` is a FUNCTION DEFINITION — substitute the
    head `f` at EVERY call — iff it is an applied Function whose LAST argument
    is a bare `Symbol` (the function's variable).  A non-symbol last argument
    (`f(t,x,b+h)`, `f(t,x,0)`) is a boundary / point value, and a
    `Derivative` lhs (`∂_t b`) is a specific atom — both substitute by exact
    structural match instead.  (`sympy.Dummy` subclasses `Symbol`, so the
    canonicalised `\\hat{ζ}` integration dummy also qualifies.)

    A CONCRETE-INDEX modal coefficient (`ŵ(0, t, x)`, `a(2, t, x)`) is also a
    SPECIFIC atom, not a generalisation — a genuine function definition
    generalises over all its arguments and so carries none that are concrete
    numbers.  Excluding a numeric argument keeps a per-mode solution
    (`SolveLinearSystem` → `ŵ_j = …`) substituting by exact match instead of
    head-rewriting the whole `ŵ` family."""
    return (isinstance(lhs, sp.Function)
            and bool(lhs.args)
            and isinstance(lhs.args[-1], sp.Symbol)
            and not any(getattr(a, "is_number", False) for a in lhs.args))


def _apply_relation(expr, rel):
    """Apply a `{lhs: rhs}` relation as a substitution.

    A lhs that is a `_is_function_definition` `f(…, v)` substitutes the
    head `f` at EVERY call, binding the definition's arguments to each call's
    arguments — so `ω(t,x,ζ) = …` also rewrites `ω(t,x,\\hat ζ)` carried
    inside a Galerkin bracket (whose last arg is the bound integration dummy).
    `rhs`'s own summation indices / integration dummies are alpha-renamed
    fresh first (`_alpha_fresh`) so inserting a nonlocal closure inside a
    sum / integral never captures the target's index or dummy.

    Every other lhs (a boundary value, a derivative) substitutes by exact
    structural `xreplace` — the basis-expansion-safe path that does not refuse
    a `ζ`-in-rhs substitution the way `.subs` would."""
    out = expr
    for lhs, rhs in rel.items():
        lhs = sp.sympify(lhs)
        rhs = sp.sympify(rhs)
        if _is_function_definition(lhs):
            head = lhs.func
            rhs_fresh = _alpha_fresh(rhs, _bound_atoms(out))

            def _expand(*call_args, _a=lhs.args, _r=rhs_fresh):
                return _r.subs(dict(zip(_a, call_args)))

            out = out.replace(head, _expand)
        else:
            out = out.xreplace({lhs: rhs})
    return out


def _resolve_subs_safe(expr):
    """Unwrap every `Subs(f, var, val)` in `expr` whose inner is safe
    to `xreplace(var → val)`.

    "Safe" means **none** of the following:

    * `var` appears as a `Derivative` differentiation variable in
      `f` — would commit to chain-rule-through-val ambiguity sympy is
      deliberately conservative about.
    * `var` is the integration variable of an `Integral` nested in
      `f` — a naive `xreplace` would overwrite the integration
      binder's tuple and produce nonsense limits (``Integral(_,
      (val, lo, val))``).  This protects running integrals introduced
      by the w-closure pipeline (`Integrate(z, b, z)`).
    * `var` is the binding variable of a nested `Subs` in `f`
      — same shadowing concern.

    When all guards pass, the Subs is replaced structurally via
    `f.xreplace({var: val})` — using `xreplace` on purpose, so
    that running integrals / nested Subs / `ζ` appearances the
    caller *did* want subbed (top-level free occurrences) are resolved
    rather than refused by sympy's `.subs` "dummy dependency"
    guard.
    """
    if not expr.has(sp.Subs):
        return expr
    mapping = {}
    for s in expr.atoms(sp.Subs):
        inner = s.args[0]
        vars_tup = s.args[1]
        vals_tup = s.args[2]
        deriv_vars = set()
        for d in inner.atoms(sp.Derivative):
            deriv_vars.update(d.variables)
        if any(v in deriv_vars for v in vars_tup):
            continue
        bound_vars = set()
        for I in inner.atoms(sp.Integral):
            for lim in I.args[1:]:
                if hasattr(lim, "__getitem__") and len(lim) >= 1:
                    bound_vars.add(lim[0])
        for nested in inner.atoms(sp.Subs):
            bound_vars.update(nested.args[1])
        if any(v in bound_vars for v in vars_tup):
            continue
        mapping[s] = inner.xreplace(dict(zip(vars_tup, vals_tup)))
    return expr.xreplace(mapping) if mapping else expr


def _evaluate_linear_derivatives(expr):
    """`Derivative(Add(...) | expandable-to-Add, var)` → sum of per-term derivatives.

    Evaluates the Derivative only when it can be resolved by linearity.
    Concretely: if `sp.expand(inner)` is an `Add`, distribute the
    Derivative across the summands (`.doit()` on the Add-inner form).
    Otherwise leave the Derivative intact so conservative forms like
    `Derivative(u**2, x)` and `Derivative(u*w, z)` survive.

    Trivial zero collapses are handled here too: `Derivative(0, var)`
    → `0` and `Subs(0, var, val)` → `0`.  Without this,
    post-`hydrostatic_scaling` z-momentum carries leftover
    `∂/∂t 0`, `∂/∂x 0`, and `Subs(∂/∂Dummy 0, Dummy, b+h)`
    residue from the analytical integrate (Dummies come from
    `sp.integrate`).  Sympy refuses to reduce these on its own —
    `Derivative.doit()` only evaluates when the inner depends on the
    diff variable, and `Subs` stays lazy for the same chain-rule
    reason that keeps `Subs(Derivative(u, x), z, b)` frozen.

    We expand the inner first so that factored forms like
    `Derivative(g*ρ*(η − z), x)` — which are algebraically
    `Derivative(−g*ρ*z + g*ρ*b + g*ρ*h, x)` — also get linearly
    distributed, producing `g*ρ*∂_x b + g*ρ*∂_x h` (and 0 for the
    z-independent-of-x part).  Without the expand-first step, the
    factored form stays as a single opaque Derivative and its tag
    cannot be follow-the-term propagated during `.apply`.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, sp.Subs):
        if expr.args[0] == S.Zero:
            return S.Zero
        inner_walked = _evaluate_linear_derivatives(expr.args[0])
        # ``Subs(f, v, p) == f`` whenever ``f`` no longer depends on ANY of
        # the point variable(s) ``v`` — true REGARDLESS of ``p``.  Collapse
        # eagerly rather than leaving the no-op wrapper in place: once the
        # self-derivative closing below reduces a bed/surface-trace body
        # (e.g. a level-1 Newtonian stress trace) to a ``v``-independent
        # constant, two such Subs at DIFFERENT points (ζ=0 vs ζ=1) become
        # identical-except-for-point.  ``Subs._hashable_content`` carries
        # ``(variables, point)`` as a bare (non-``Basic``) tuple, and
        # sympy's generic ``Basic.compare`` mishandles ordering two of those
        # during ``Add`` canonicalisation (``TypeError: BooleanAtom not
        # allowed in this context``) — collapsing here means that pair of
        # raw Subs nodes is never built in the first place.
        if not any(inner_walked.has(v) for v in expr.variables):
            return inner_walked
        return expr.func(inner_walked, *expr.args[1:])
    if isinstance(expr, Derivative):
        inner = expr.args[0]
        if inner == S.Zero:
            return S.Zero
        # Trivially zero: inner doesn't mention any of the diff vars.
        # Without this shortcut, expressions like Derivative(-g·ρ·z, x)
        # (coming out of the hydrostatic pressure solve) would survive
        # our post-distribute pass.  ``sympy.Derivative.doit()`` would
        # handle it, but we can't ``.doit()`` the Add-distributed form
        # in general (it also applies the product rule, flattening
        # conservative ``Derivative(ν·∂_z u, z)`` shapes that the
        # fundamental theorem of calculus needs).  The targeted
        # "no diff-var inside" check keeps both cases correct.
        if not any(inner.has(v) for v in expr.variables):
            return S.Zero
        # Pass-1 "expand": pull a purely-numeric scalar factor out of the
        # Derivative inner via ``Mul.as_coeff_Mul()`` (default behaviour —
        # only ``Rational`` / ``Integer`` / ``Float`` come out; free
        # symbols like ``ρ``, ``g``, ``ν`` stay inside).  Turns
        # ``Derivative(-α₁, x)`` into ``-Derivative(α₁, x)`` so that
        # downstream ``(-∂α₁)² / ∂α₁`` reduces structurally via sympy's
        # Mul canonicalization.  Conservative shapes are untouched:
        # ``Derivative(u², x)`` has a Pow inner (no Mul factor to pull),
        # and ``Derivative(u·w, z)`` has ``as_coeff_Mul() == (1, u·w)``
        # so nothing is pulled.
        if isinstance(inner, sp.Mul):
            c, rest = inner.as_coeff_Mul()
            if c != S.One:
                return c * _evaluate_linear_derivatives(
                    expr.func(rest, *expr.args[1:])
                )
        inner_expanded = sp.expand(inner) if isinstance(inner, sp.Basic) else inner
        if isinstance(inner_expanded, sp.Add):
            inner_walked = _evaluate_linear_derivatives(inner_expanded)
            # Distribute the outer Derivative over the Add via linearity
            # **without** ``.doit()``.  ``.doit()`` at this stage applies
            # the product rule inside each summand, turning conservative
            # shapes like ``Derivative(ν·∂_z u, z)`` into the expanded
            # ``∂_z ν · ∂_z u + ν · ∂²u/∂z²`` form.  The second piece
            # then has a two-variable Derivative tuple that
            # ``_extract_derivative`` refuses, and the fundamental
            # theorem of calculus can no longer integrate the term.
            # By building the distributed sum manually we keep each
            # summand in conservative ``Derivative(coeff·f, var)`` form,
            # where Leibniz / fund-thm apply cleanly.
            args_rest = expr.args[1:]
            return sp.Add(*(expr.func(t, *args_rest)
                            for t in sp.Add.make_args(inner_walked)))
        inner_walked = _evaluate_linear_derivatives(inner)
        result = expr.func(inner_walked, *expr.args[1:])
        # ``inner_walked`` may now be a BARE polynomial purely in the diff
        # variable(s) themselves — e.g. a basis function already concretised
        # by ``ResolveBasis`` to ``2*zeta - 1``, whose Mul-coefficient pull
        # above (Pass-1) peeled off the self-derivative remainder
        # ``Derivative(zeta, zeta)`` (level 1), or a higher-level
        # ``Derivative(zeta**2, zeta)``.  There is no unknown Function /
        # conservative product left to guard from the product rule (the
        # only reason this function avoids ``.doit()`` elsewhere), so
        # closing the atom here is always safe.  Left unevaluated, these
        # zombies ride into flux / nonconservative_matrix / source and
        # break ``sp.lambdify`` (NumPyPrinter has no rule for a raw,
        # un-``doit``-ed ``Derivative``).
        if (not inner_walked.atoms(Function)
                and inner_walked.free_symbols <= set(expr.variables)):
            return result.doit()
        return result
    if expr.args:
        return expr.func(*(_evaluate_linear_derivatives(a) for a in expr.args))
    return expr


def _fold_numeric_coeffs_into_derivative(expr):
    """Pass-3 "collect": inverse of the Pass-1 numeric pull-out.

    Walks `expr` and rewrites every `c · Derivative(f, v, ...)` where
    `c` is a purely-numeric scalar (`Mul.as_coeff_Mul()` default)
    back into `Derivative(c · f, v, ...)`.  This restores the
    conservative-form reading of fluxes — `Derivative(g·h²/2, x)`
    instead of `(1/2)·Derivative(g·h², x)` — after Pass 2's
    `expand + powsimp` has already fired every cancellation that
    the Pass-1-canonical form exposed.

    The two passes are exact inverses (`as_coeff_Mul` is symmetric),
    so the full expand→simplify→collect pipeline is idempotent under
    repeated `.simplify()` calls.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, sp.Mul):
        walked_args = [_fold_numeric_coeffs_into_derivative(a)
                       for a in expr.args]
        derivs = [a for a in walked_args if isinstance(a, Derivative)]
        if len(derivs) == 1:
            deriv = derivs[0]
            coeff = sp.Mul(*[a for a in walked_args if a is not deriv])
            # Only fold purely-numeric coefficients — matches Pass 1's
            # ``as_coeff_Mul`` gate.  ``coeff.is_number`` is True for
            # Rational / Integer / Float and False for any free-symbol
            # product, so ``ν · ∂_x u`` stays as-is.
            if coeff.is_number and coeff != S.One:
                return Derivative(coeff * deriv.args[0], *deriv.args[1:])
        if any(n is not o for n, o in zip(walked_args, expr.args)):
            return expr.func(*walked_args)
        return expr
    if expr.args:
        walked_args = tuple(_fold_numeric_coeffs_into_derivative(a)
                            for a in expr.args)
        if any(n is not o for n, o in zip(walked_args, expr.args)):
            return expr.func(*walked_args)
    return expr


def _extract_derivative(expr, var):
    """Decompose `expr` as `coeff * Derivative(inner, var)`.

    Returns `(inner, coeff)` when the decomposition is valid,
    `(None, None)` otherwise.  Validity requires that `coeff` is
    independent of `var`.

    Without that check, a product-rule-expanded term like
    `2u · ∂_x u` would be split as `coeff = 2u, inner = u` and the
    downstream Leibniz rule ``∫ coeff · ∂_x inner dz = ∂_x[∫ coeff ·
    inner dz] + …` would compute `∂_x[∫ 2u² dz]`` — **double** the
    correct value, since the true conservative form `∂_x(u²)` gives
    `∂_x[∫ u² dz]`.

    So we only succeed when the term is in genuine conservative form:
    either a bare `Derivative(f, var)`, or a product whose only
    var-dependent factor is the `Derivative`.  Expanded product-rule
    forms fall through to "direct" integration and stay wrapped in an
    unevaluated `Integral`.
    """
    expr = sp.sympify(expr)
    if isinstance(expr, Derivative):
        if expr.variables == (var,):
            return expr.args[0], S.One
    factors = Mul.make_args(expr)
    coeff_factors = []
    inner = None
    for f in factors:
        if isinstance(f, Derivative) and f.variables == (var,) and inner is None:
            inner = f.args[0]
        else:
            coeff_factors.append(f)
    if inner is None:
        return None, None
    coeff = Mul(*coeff_factors) if coeff_factors else S.One
    # Refuse when ``coeff`` still depends on the derivative variable —
    # the term is product-rule expanded, not conservative.
    if var in coeff.free_symbols:
        return None, None
    return inner, coeff


def integrate_by_parts(f, g, var, domain=(0, 1)):
    """
    Standalone IBP: integral d(f)/dvar * g dvar = [f*g]_a^b - integral f * dg/dvar dvar
    Returns IBPResult(integrate, boundary_upper, boundary_lower).
    """
    a, b = domain
    return IBPResult(
        integrate=Expression(-Integral(f * Derivative(g, var), (var, a, b)), "ibp_integrate"),
        boundary_upper=Expression((f * g).subs(var, b), "ibp_upper"),
        boundary_lower=Expression((f * g).subs(var, a), "ibp_lower"),
    )


def gauss_legendre_integrate(expr, var, a, b, order=4):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes_shifted = [(b - a) / 2 * xi + (b + a) / 2 for xi in nodes]
    weights_scaled = [(b - a) / 2 * wi for wi in weights]
    result = S.Zero
    for xi, wi in zip(nodes_shifted, weights_scaled):
        result += sp.Rational(wi).limit_denominator(10**8) * expr.subs(var, sp.nsimplify(xi))
    return result


# ---------------------------------------------------------------------------
# Additions for the new Model-class design (not in legacy).
# ---------------------------------------------------------------------------
# These are the operations the new path needs that don't exist in the
# legacy file: ``ResolveDummy`` (promoted from the legacy notebook
# ``Model.resolve_dummy`` convenience method to a full Operation, since
# the new Model surface is minimal — only ``add_equation /
# remove_equation / apply / describe``) and ``Symmetrize`` (was in
# ``model/models/model.py`` in the old path; lives here now because it
# IS an Operation).
# ---------------------------------------------------------------------------


class Symmetrize(Operation):
    """`a → a/2 + rule(a/2)`.

    When `rule = ProductRule()` and the term has the self-pair shape
    `c · f · ∂_v(f)`, the cancellation under `Equation.simplify`
    leaves a clean conservative atom `c/2 · ∂_v(f²)`.  Used for the
    gravity self-pair fold in SME / VAM.
    """

    def __init__(self, rule):
        self.rule = rule
        super().__init__(
            name="symmetrize",
            description=f"½·a + {getattr(rule, 'name', type(rule).__name__)}(½·a)",
        )

    def _leaf_sp(self, expr):
        half = expr / 2
        applied = Expression(half, "").apply(self.rule).expr
        return sp.expand(half + applied)


class ResolveDummy(Operation):
    """Replace every `dummy(arg)` call across every equation in a model.

    Promoted from the legacy notebook `Model.resolve_dummy` method to
    a full Operation.  The new Model surface is minimal
    (`add_equation / remove_equation / apply / describe`) and
    substitutions go via `model.apply(ResolveDummy(dummy, value))`.

    Parameters
    ----------
    dummy
        A sympy `Function` placeholder (e.g. `phi_u_fn`,
        `psi_u`, `omega`) that the derivation introduced as an
        opaque modal-ansatz / weight stand-in.
    value
        - sympy expression / scalar → single substitution
          (replace every `dummy(arg)` with `value`);
        - callable `f(*args) → sp.Expr` → call-with-args
          substitution (`dummy(a, b) → value(a, b)`);
        - list / tuple → *branch each equation containing the dummy*
          into one sub-equation per list element (outer product on
          the dummy axis).  Each sub-equation is named `"{eq_name}_{i}"`.
    """

    def __init__(self, dummy, value):
        self.dummy = dummy
        self.value = value
        # List-of-values case branches each equation → needs whole-model
        # dispatch.  Scalar / callable case can be applied per-term via
        # ``_leaf_sp``.
        self.whole_model_op = isinstance(value, (list, tuple))
        super().__init__(
            name="resolve_dummy",
            description=(f"{getattr(dummy, '__name__', repr(dummy))} → "
                         f"{'<branched>' if isinstance(value, (list, tuple)) else '<resolved>'}"),
        )

    def _leaf_sp(self, expr):
        """Substitute `dummy(*args) → value(*args)` over the leaf."""
        replace = self._make_replace(self.value)
        return replace(expr, self.dummy)

    def apply_to_model(self, model):
        # Only used for the list-branching case (whole_model_op=True).
        self._apply_branched(model)
        return model

    @staticmethod
    def _make_replace(value):
        if callable(value) and not isinstance(value, sp.Basic):
            return lambda expr, dummy: expr.replace(dummy, value)
        return lambda expr, dummy: expr.replace(
            dummy, lambda *args, _v=value: _v)

    def _apply_substitution(self, model):
        replace = self._make_replace(self.value)
        for eq in model._equations.values():
            eq.expr = replace(eq.expr, self.dummy)

    def _apply_branched(self, model):
        from zoomy_core.model.equation import Equation
        new_eqs: dict = {}
        for name, eq in model._equations.items():
            if not eq.expr.has(self.dummy):
                new_eqs[name] = eq
                continue
            for i, v in enumerate(self.value):
                new_expr = self._make_replace(v)(eq.expr, self.dummy)
                sub = f"{name}_{i}"
                new_eqs[sub] = Equation(new_expr, name=sub, model=model)
        model._equations = new_eqs


# Basisfunction re-exports — lives in `model/models/basisfunctions.py`
# (NOT under legacy/), so this is legitimate.  Expose the BASE class
# as the user-facing abstraction; concrete bases (Legendre_shifted, …)
# are commits at the resolution step.
from zoomy_core.model.derivation.basisfunctions import (
    Basisfunction,
    Legendre_shifted,
)
