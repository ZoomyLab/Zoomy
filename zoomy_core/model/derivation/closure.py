"""Concrete-level Galerkin resolve, conservative-form folding, simplification and term-tagging
operations of the derivation pipeline. They carry a modal ansatz to the K&T SME system form that
system_extract classifies into SystemModel slots.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core import coords as _coords
from zoomy_core.model.operations import Operation, Multiply, is_bracket


__all__ = [
    "Resolve",
    "ResolveIntegral",
    "InvertMassMatrix",
    "FoldConservative",
    "Split",
    "Simplify",
    "Consolidate",
    "consolidate",
    "fold_self_products",
    "leibniz_fold",
    "AutoTag",
    "SortByTag",
    "Sort",
    "TAG_ORDER",
    "fold_to_conservative_form",
    "is_conservative_diffusion",
    "project_conservative_diffusion",
]


class Split(Operation):
    """Fold a self-product flux term c·f·∂_x f into ∂_x(c·f²/2) by splitting it in two halves and
    applying the inverse product rule to one half. Value-preserving (½·I + ½·ProductRule⁻¹); apply
    it to the self-product terms only.
    """

    whole_leaf_op = True

    def __init__(self, variables=None, name="split"):
        """`variables` restricts the fold to these fields (None: all)."""
        self._vars = variables
        desc = ("half product rule  c·f·∂f → ∂(c·f²/2)"
                + (f" on {{{', '.join(str(v) for v in variables)}}}"
                   if variables is not None else ""))
        super().__init__(name=name, description=desc)

    def _leaf_sp(self, sp_expr):
        """Return ½·expr + ½·(inverse product rule of expr)."""
        from zoomy_core.model.operations import ProductRule
        pr = ProductRule(variables=self._vars, direction="inverse")
        return sp.Rational(1, 2) * sp_expr \
            + sp.Rational(1, 2) * pr._leaf_sp(sp_expr)


class InvertMassMatrix(Operation):
    """Divide a projected moment row by the coefficient of its ∂_t term, normalising ∂_t Q_k to unit
    coefficient (the mass-matrix entry is read off the row). Apply it after the stray ∂_t h has been
    removed so the row carries a single diagonal ∂_t term.
    """

    whole_leaf_op = True

    def __init__(self, time=None, name="invert_mass_matrix"):
        """`time` is the time coordinate (default: the global `t`)."""
        self._t = time if time is not None else _coords.t
        super().__init__(
            name=name,
            description="divide each row by its ∂_t coefficient (mass matrix)")

    def _leaf_sp(self, sp_expr):
        """Find the t-free non-trivial coefficient of a ∂_t atom and divide each term by it,
        term-wise.
        """
        t = self._t
        # Do NOT ``sp.expand`` the whole row: post-basis it holds the rational
        # turbulence source (ν_t=C_μ sk⁴/se²) and expanding distributes N·D⁻¹ into
        # a >1M-char monster.  Work on the top-level terms as-is and divide
        # TERM-WISE; only the (small) ∂_t mass coefficient is simplify/cancel-ed.
        terms = list(sp.Add.make_args(sp.sympify(sp_expr)))
        # The mass coefficient of a ∂_t atom may be SPLIT across several terms
        # and may sit INSIDE the conservative time term (``∂_t(q_k/(2k+1))``).
        # Collect the TOTAL coefficient per ∂_t atom, then pick the (unique) one
        # that is t-free and non-trivial.
        coeffs: dict = {}
        for term in terms:
            dts = [D for D in term.atoms(sp.Derivative)
                   if D.variables == (t,)]
            if len(dts) == 1:
                D = dts[0]
                inner_c, _base = D.expr.as_independent(
                    *D.expr.atoms(sp.Function), as_Mul=True)
                coeffs[D] = coeffs.get(D, sp.S.Zero) \
                    + sp.simplify(term / D) * inner_c
        mass = None
        for _D, cand in coeffs.items():
            cand = sp.cancel(sp.simplify(cand))     # small coeff only — cheap
            if cand not in (sp.S.Zero, sp.S.One) and not cand.has(t):
                mass = cand
                break
        if mass in (None, sp.S.Zero, sp.S.One):
            return sp_expr
        # divide each term by the mass coefficient (compact per term; no global
        # expand of the huge rational source)
        out = sp.Add(*[term / mass for term in terms])

        # normalize the form: pull field-free factors out of Derivatives
        # (linearity) so ``∂_t(q/(2k+1))/(1/(2k+1))`` reads ``∂_t q``.
        def _pull(a):
            """Pull field-free factors out of one Derivative (linearity)."""
            co, rest = a.expr.as_independent(
                *a.expr.atoms(sp.Function), as_Mul=True)
            return (co * sp.Derivative(rest, *a.args[1:])
                    if co != 1 and rest != 1 else a)

        return out.replace(
            lambda a: isinstance(a, sp.Derivative), _pull)


# ── conservative fold (flux/pressure bundling, NCP-preserving) ─────────────


def _antiderivative_in(coeff, var, field_to_sym):
    """Antiderivative of `coeff` in the applied field `var`, treating every field in
    `field_to_sym` as an independent symbol.
    """
    sym_of_var = field_to_sym[var]
    inv = {s: f for f, s in field_to_sym.items()}
    integrand = coeff.xreplace(field_to_sym)
    F = sp.integrate(integrand, sym_of_var)
    return sp.expand(F.xreplace(inv))


def fold_to_conservative_form(expr, flux_fields, *, h, b, x, gravity_param):
    """Fold the spatial part of a momentum/continuity residual into conservative flux and
    hydrostatic-pressure bundles, leaving bed and cross-mode couplings as explicit nonconservative
    terms. Reproduces the production K&T decomposition, e.g. g·h·∂_x b + ∂_x(g·h²/2) + ∂_x(q_0²/h +
    q_1²/3h) + ∂_t q_0.

    Args:
        expr: residual of one row, in Function form.
        flux_fields: conserved modal fields q(k, t, x).
        h, b: depth and bed (applied Functions).
        x: spatial coordinate.

    Returns:
        The re-bundled residual.

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """
    expr = sp.expand(expr.doit())
    flux_fields = list(flux_fields)

    def dx(f):
        """∂_x of `f`."""
        return sp.Derivative(f, x)

    # Fresh bare symbols for the antiderivative-in-h integration.
    field_to_sym = {h: sp.Symbol("_H_fold", positive=True)}
    for i, f in enumerate(flux_fields):
        field_to_sym[f] = sp.Symbol(f"_Qf{i}", real=True)

    # Split the residual into non-spatial (time / source) and spatial terms.
    nonspatial, spatial = [], []
    for term in sp.Add.make_args(expr):
        if any(d.variables == (x,) for d in term.atoms(sp.Derivative)):
            spatial.append(term)
        else:
            nonspatial.append(term)
    spatial = sp.expand(sp.Add(*spatial))

    # 1. Bed coupling g·h·∂_x b stays an explicit NCP term.
    cb = sp.expand(spatial.coeff(dx(b)))
    bed = cb * dx(b)
    spatial = sp.expand(spatial - bed)

    # 2. The ∂_x h coefficient integrates in h to the conservative bundle.
    ch = sp.expand(spatial.coeff(dx(h)))
    if gravity_param is not None:
        ch_grav = sp.expand(ch.coeff(gravity_param)) * gravity_param
    else:
        ch_grav = sp.S.Zero
    ch_flux = sp.expand(ch - ch_grav)
    P = _antiderivative_in(ch_grav, h, field_to_sym) if ch_grav != 0 \
        else sp.S.Zero
    F = _antiderivative_in(ch_flux, h, field_to_sym) if ch_flux != 0 \
        else sp.S.Zero

    # 3. The ∂_x q_j couplings left after subtracting ∂_x(F + P) are NCP.
    dFP = sp.expand(sp.Derivative(P + F, x).doit())
    ncp = sp.expand(spatial - dFP)

    out = sp.Add(*nonspatial) + bed + ncp
    if P != 0:
        out += sp.Derivative(P, x)
    if F != 0:
        out += sp.Derivative(F, x)
    return out


class FoldConservative(Operation):
    """Tracked op form of fold_to_conservative_form: conservative diffusion atoms ∂_x(D·∂_x q) are
    left untouched and the hyperbolic remainder is folded. Arguments mirror
    fold_to_conservative_form (`gravity` is its `gravity_param`).
    """

    whole_leaf_op = True

    def __init__(self, flux_fields, *, h, b, x, gravity=None,
                 name="fold_conservative"):
        """Store the fold arguments."""
        self._flux = list(flux_fields)
        self._h, self._b, self._x, self._g = h, b, x, gravity
        super().__init__(
            name=name, description="conservative flux/pressure fold")

    def _leaf_sp(self, sp_expr):
        """Split off diffusion atoms, fold the rest, and reattach the diffusion."""
        x = self._x
        diff_terms, base_terms = [], []
        for term in sp.Add.make_args(sp_expr):
            (diff_terms if is_conservative_diffusion(term, x)
             else base_terms).append(term)
        base = sp.expand(sp.Add(*base_terms).doit())
        folded = fold_to_conservative_form(
            base, self._flux, h=self._h, b=self._b, x=x, gravity_param=self._g)
        return folded + sp.Add(*diff_terms)


# ── Resolve (concrete-level project + in-place ζ-integration) ──────────────


class Resolve(Operation):
    """Concrete-level Galerkin project-and-resolve: multiply the row by the test weight c(ζ)·φ(l,
    ζ), substitute the concrete basis polynomials, and integrate over ζ with the ∂_x / ∂_t
    derivatives kept inside the integral. Apply after the truncation bound is concrete and the Sum
    ansatz is expanded; modal coefficients and h, b factors are left untouched.

    Args:
        test_function: Galerkin test weight, typically c(ζ)·φ(l, ζ) with l bound to the row's mode.
        basis_cls: concrete basis class providing `eval(k, ζ)` and `weight`.
        level: concrete basis level.
        var: ζ integration variable (default `zeta`).
        bounds: integration interval (default (0, 1)).

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    whole_leaf_op = True

    def __init__(self, test_function, basis_cls, level, *, var=None,
                 bounds=(0, 1), name="resolve"):
        """Build the test-weight multiplier and the concrete basis."""
        self._test = sp.sympify(test_function)
        self._basis = basis_cls(level=level)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._multiply = Multiply(self._test)
        # Concrete weight callable (1 for Legendre): the basis exposes
        # ``weight`` as a callable / value; resolve it once.
        w = getattr(self._basis, "weight", None)
        self._weight_value = w
        super().__init__(
            name=name,
            description=(f"resolve+integrate over {self._var} via "
                         f"{getattr(basis_cls, '__name__', basis_cls)}"),
        )

    def _resolve_opaque_phi(self, expr):
        """Replace opaque φ(k, ζ) heads by the concrete basis polynomials and the opaque weight c(ζ)
        by the basis weight.
        """
        z = self._var

        def _is_phi(a):
            """True for a two-argument basis head φ(k, ζ)."""
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 2)

        def _is_weight(a):
            """True for a one-argument basis weight head c(ζ)."""
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 1)

        # weight c(ζ) → basis weight (callable → value at its arg, else value).
        wv = self._weight_value

        def _weight_at(arg):
            """Basis weight at `arg` (1 when the basis has none)."""
            if callable(wv):
                return wv(arg)
            return sp.S.One if wv is None else wv

        expr = expr.replace(_is_weight, lambda a: _weight_at(a.args[0]))
        # φ(k, ζ) → eval(k, ζ) for concrete integer k.
        expr = expr.replace(
            _is_phi,
            lambda a: (self._basis.eval(int(a.args[0]), a.args[1])
                       if a.args[0].is_Integer else a),
        )
        return expr

    def _integrate_poly(self, expr):
        """Integrate `expr` over the ζ bounds term by term analytically, falling back to
        sp.integrate for non-polynomial dependence.
        """
        z = self._var
        lo, hi = self._bounds
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if dep == 1:
                out += coeff * (hi - lo)
            elif dep == z:
                out += coeff * (hi**2 - lo**2) / 2
            elif (isinstance(dep, sp.Pow) and dep.base == z
                  and dep.exp.is_Integer and dep.exp >= 0):
                n = int(dep.exp)
                out += coeff * (hi**(n + 1) - lo**(n + 1)) / (n + 1)
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _leaf_sp(self, sp_expr):
        """Multiply by the test weight, resolve the basis, and integrate over ζ."""
        scaled = self._multiply._leaf_sp(sp_expr)
        resolved = self._resolve_opaque_phi(sp.expand(scaled))
        # ``.doit()`` collapses any concrete-index basis derivatives.
        resolved = resolved.doit()
        return self._integrate_poly(resolved)


# ── ResolveIntegral (decoupled, method-per-integral resolution) ────────────


class ResolveIntegral(Operation):
    """Resolve the opaque ∫…dζ atoms left by an abstract projection, each by a selectable method:
    "auto" (classified per term by shape), "basis", "ftc" or "numerical". `classify` (Integral ->
    method or None) overrides the default `method` per integral.

    Args:
        basis_cls: concrete basis used by the "basis" and "auto" methods.
        method: default resolution method.
        weight: basis weight when no `basis_cls` is given.
        classify: optional per-integral method override.
    """

    whole_leaf_op = True

    def __init__(self, basis_cls=None, *, var=None, method="auto", level=None,
                 weight=None, classify=None, bounds=(0, 1),
                 name="resolve_integral"):
        """Configure the default method, basis, and bounds."""
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._method = method
        self._classify = classify
        self._basis = basis_cls(level=level) if basis_cls is not None else None
        self._weight_value = (getattr(self._basis, "weight", None)
                              if self._basis is not None else weight)
        self._bounds = tuple(bounds)
        super().__init__(
            name=name,
            description=f"resolve ∫d{self._var} (default method={method})")

    # ── per-integral method dispatch ─────────────────────────────────────
    def _method_for(self, integral):
        """Method for one integral: the `classify` override, else the default."""
        if self._classify is not None:
            chosen = self._classify(integral)
            if chosen:
                return chosen
        return self._method

    def _resolve_one(self, integral):
        """Resolve one definite integral over its own bound variable with the selected method."""
        integrand = integral.function
        # Resolve using the integral's OWN bound variable (var-agnostic): the
        # field-level KinematicBC alpha-renames bound ζ → fresh ``\hat{z}``
        # Dummies, so the integration variable that actually appears is read off
        # the integral itself rather than assumed to be ``self._var``.
        lim = integral.limits[0]
        var = lim[0]
        lo, hi = (lim[1], lim[2]) if len(lim) == 3 else self._bounds
        method = self._method_for(integral)
        if method == "auto":
            return self._auto_resolve(integrand, var, lo, hi)
        if method == "ftc":
            return self._ftc(integrand, var, lo, hi)
        if method == "basis":
            return self._resolve_basis(integrand, var, lo, hi)
        if method == "numerical":
            return sp.integrate(integrand, (var, lo, hi))
        raise ValueError(f"ResolveIntegral: unknown method {method!r}")

    # ── auto: classify each additive term by SHAPE (the "smart" resolution) ──
    def _auto_resolve(self, integrand, var, lo, hi):
        """Resolve `∫_lo^hi integrand d(var)` term by term via `_auto_term`."""
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(integrand)):
            out += self._auto_term(term, var, lo, hi)
        return out

    def _auto_term(self, term, var, lo, hi):
        """Resolve one term: constant, bare ∂_var (FTC), bare ∂_w with w-independent rest (commute
        out), single Sum (push inside), or φ-bracket (basis); otherwise keep an opaque Integral.
        """
        coeff, rest = term.as_independent(var, as_Add=False)
        if rest == 1:                                       # constant in var
            return coeff * (hi - lo)
        # a bare outer derivative as the whole var-dependent body
        if isinstance(rest, sp.Derivative) and len(rest.variables) == 1:
            dvar = rest.variables[0]
            g = rest.args[0]
            if dvar == var:                                 # FTC over var
                return coeff * (g.subs(var, hi) - g.subs(var, lo))
            return coeff * sp.Derivative(                   # conservative commute
                self._auto_resolve(g, var, lo, hi), dvar)
        factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
        # a bare ``∂_w`` factor (w≠var) whose remaining factors+coeff are
        # w-independent commutes OUT of the integral (conservative flux).
        for f in factors:
            if (isinstance(f, sp.Derivative) and len(f.variables) == 1
                    and f.variables[0] != var):
                dw = f.variables[0]
                others = sp.Mul(*[g for g in factors if g is not f])
                if not (coeff * others).has(dw):
                    inner = self._auto_resolve(others * f.expr, var, lo, hi)
                    return coeff * sp.Derivative(inner, dw)
                break
        # a single unexpanded Sum factor: push the integral inside.
        sums = [f for f in factors if isinstance(f, sp.Sum)]
        if len(sums) == 1:
            the_sum = sums[0]
            others = sp.Mul(*[f for f in factors if f is not the_sum])
            cc, dep = the_sum.function.as_independent(var, as_Add=False)
            oc, od = others.as_independent(var, as_Add=False)
            return coeff * oc * sp.Sum(
                cc * sp.Integral(od * dep, (var, lo, hi)), *the_sum.limits)
        # a φ-basis bracket: substitute the concrete basis and integrate.
        if self._basis is not None and any(
                getattr(a.func, "_is_basis_head", False)
                for a in rest.atoms(sp.Function)):
            return coeff * self._resolve_basis(rest, var, lo, hi)
        # otherwise leave the var-dependent body as an opaque Integral.
        return coeff * sp.Integral(rest, (var, lo, hi))

    # ── methods ──────────────────────────────────────────────────────────
    def _ftc(self, integrand, var, lo, hi):
        """Apply the fundamental theorem of calculus to bare ∂_var terms; other terms go to
        sp.integrate.
        """
        z = var
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(integrand)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if (isinstance(dep, sp.Derivative)
                    and len(dep.variables) == 1 and dep.variables[0] == z):
                g = dep.expr
                out += coeff * (g.subs(z, hi) - g.subs(z, lo))
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _resolve_basis(self, integrand, var, lo, hi):
        """Substitute the concrete basis polynomials (and weight) for the opaque φ heads and
        integrate the polynomial analytically.
        """
        z = var
        if self._basis is None:
            return sp.integrate(integrand, (z, lo, hi))
        wv = self._weight_value

        def _is_phi(a):
            """True for a two-argument basis head φ(k, ζ)."""
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 2)

        def _is_weight(a):
            """True for a one-argument basis weight head c(ζ)."""
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 1)

        def _weight_at(arg):
            """Basis weight at `arg` (1 when the basis has none)."""
            return (wv(arg) if callable(wv)
                    else (sp.S.One if wv is None else wv))

        e = sp.expand(integrand)
        e = e.replace(_is_weight, lambda a: _weight_at(a.args[0]))
        e = e.replace(
            _is_phi,
            lambda a: (self._basis.eval(int(a.args[0]), a.args[1])
                       if a.args[0].is_Integer else a))
        return self._integrate_poly(e.doit(), z, lo, hi)

    def _integrate_poly(self, expr, var, lo, hi):
        """Integrate `expr` over (lo, hi) in `var` term by term for polynomials, sp.integrate
        otherwise.
        """
        z = var
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if dep == 1:
                out += coeff * (hi - lo)
            elif dep == z:
                out += coeff * (hi**2 - lo**2) / 2
            elif (isinstance(dep, sp.Pow) and dep.base == z
                  and dep.exp.is_Integer and dep.exp >= 0):
                n = int(dep.exp)
                out += coeff * (hi**(n + 1) - lo**(n + 1)) / (n + 1)
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _resolve_basis_atoms(self, expr):
        """Evaluate the opaque φ(k, arg) and weight atoms that survive outside integrals (the FTC
        boundary traces) with the concrete basis.
        """
        if self._basis is None:
            return expr
        bas = self._basis
        wv = self._weight_value
        expr = expr.replace(
            lambda a: (isinstance(a, sp.Function)
                       and getattr(a.func, "_is_basis_head", False)
                       and len(a.args) == 2 and a.args[0].is_Integer),
            lambda a: bas.eval(int(a.args[0]), a.args[1]))
        expr = expr.replace(
            lambda a: (isinstance(a, sp.Function)
                       and getattr(a.func, "_is_basis_head", False)
                       and len(a.args) == 1),
            lambda a: (wv(a.args[0]) if callable(wv)
                       else (sp.S.One if wv is None else wv)))
        return expr

    def _leaf_sp(self, sp_expr):
        """Resolve every matching definite integral, then close the remaining boundary-trace basis
        atoms.
        """
        z = self._var

        def _has_basis_atom(a):
            """True when `a` contains an opaque basis head."""
            return any(getattr(f.func, "_is_basis_head", False)
                       for f in a.atoms(sp.Function))

        def _mine(a):
            """True for a single-variable definite integral this op resolves (own variable, Dummy
            variable, ζ bound, or basis atom).
            """
            # Resolve any single-variable definite integral over the configured
            # var, over a fresh ``\hat{<var>}`` integration Dummy (running
            # integrals ``∫_0^ζ … d\hat ζ``, whose var is the Dummy not ζ), over
            # the configured var as a BOUND (so the running integral is caught by
            # its ζ upper limit), OR carrying an opaque basis φ/weight.
            if not (isinstance(a, sp.Integral) and len(a.limits) == 1
                    and len(a.limits[0]) == 3):
                return False
            var, lo, hi = a.limits[0]
            return (var == z or isinstance(var, sp.Dummy)
                    or z in (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols)
                    or _has_basis_atom(a))

        out = sp_expr.replace(_mine, self._resolve_one)
        # close the remaining opaque φ-evals (FTC boundary traces) + weight.
        return self._resolve_basis_atoms(out)


# ── conservative second-order (diffusion) projection ───────────────────────


def is_conservative_diffusion(term, x):
    """True iff `term` is coeff·∂_x(F) with a single first-order ∂_x factor whose flux F itself
    contains a ∂_x, the ∂_x(D·∂_x q) shape that system_extract types as diffusion_matrix. Advective
    fluxes ∂_x(q²/h) and first-order couplings coeff·∂_x q do not match.
    """
    factors = term.args if isinstance(term, sp.Mul) else [term]
    dxs = [f for f in factors
           if isinstance(f, sp.Derivative) and f.variables == (x,)]
    if len(dxs) != 1:
        return False
    inner = dxs[0].args[0]
    return any(dd.variables == (x,) for dd in inner.atoms(sp.Derivative))


def project_conservative_diffusion(visc_expr, test_weight, *, basis_cls,
                                   level, var, x):
    """Galerkin-project a viscous second-order momentum block so the diffusive flux stays in
    conservative ∂_x(F^d) form (typed as diffusion_matrix), with the σ-metric residual integrated in
    place. Built from Multiply(test weight), the inverse ProductRule, abstract Integrate and
    ResolveIntegral (basis); the sum equals the in-place Resolve.

    Args:
        visc_expr: viscous term of the momentum row.
        test_weight: Galerkin test weight c·φ_k.
        basis_cls, level: concrete basis and its level.
        var: ζ integration variable.
        x: horizontal coordinate.

    Returns:
        The resolved viscous moment (conservative flux plus metric residual).
    """
    from zoomy_core.model.derivation.projection import Integrate as _AbstractIntegrate
    from zoomy_core.model.operations import Multiply, ProductRule, Expression

    scaled = Expression(visc_expr, "").apply(Multiply(test_weight)).expr
    pr = Expression(scaled, "").apply(ProductRule()).expr

    cons_terms, rest_terms = [], []
    for term in sp.Add.make_args(sp.expand(pr)):
        (cons_terms if is_conservative_diffusion(term, x)
         else rest_terms).append(term)

    ri = ResolveIntegral(var=var, method="basis", basis_cls=basis_cls,
                         level=level)
    # Conservative flux: abstract ∫ commutes the outer ∂_x out → resolve basis.
    cons_int = Expression(sp.Add(*cons_terms), "").apply(
        _AbstractIntegrate(var=var, bounds=(0, 1))).expr
    cons_res = Expression(cons_int, "").apply(ri).expr
    # σ-metric residual: resolve in place (explicit ∫dζ wrapper, basis).
    rest_res = Expression(
        sp.Integral(sp.Add(*rest_terms), (var, 0, 1)), "").apply(ri).expr
    return sp.expand(cons_res) + sp.expand(rest_res)


# ── Simplify (graph-recorded canonicalisation) ─────────────────────────────


def pull_consts(expr):
    """∂_v(c·f) -> c·∂_v f for every factor c independent of the derivative variables, to a fixpoint
    and through nested derivatives. Gentle: it never applies the product rule to a v-dependent
    product, so ∂_x(g h²/2) -> (g/2)∂_x(h²) keeps h² grouped.
    """
    def _pull(a):
        """Pull the variable-independent multiplicative factor out of one Derivative."""
        if not a.variables:
            return a
        # MULTIPLICATIVE split (``as_Add=False``): pull a v-independent FACTOR
        # out.  The additive default would, for ``∂_v(Σ v-dependent terms)``
        # (a conservative flux of a sum, e.g. the mass ``∂_x(h·Σ_i a_i⟨φ_i⟩)``),
        # return ``const=0`` and silently ZERO the whole derivative.
        const, dep = a.expr.as_independent(*a.variables, as_Add=False)
        return a if const == sp.S.One else const * sp.Derivative(dep, *a.variables)
    prev = None
    while expr != prev:
        prev = expr
        expr = expr.replace(lambda a: isinstance(a, sp.Derivative), _pull)
    return expr


def _single_deriv_factor(term):
    """`(coeff, deriv)` when `term` has exactly one Derivative factor, else None."""
    factors = sp.Mul.make_args(term)
    ds = [f for f in factors if isinstance(f, sp.Derivative)]
    if len(ds) != 1:
        return None
    coeff = sp.Mul(*[f for f in factors if f is not ds[0]])
    return coeff, ds[0]


def fold_conservatives(expr):
    """Recover conservative groupings g·∂_v f + f·∂_v g -> ∂_v(f·g) by a pairwise fold applied only
    when it is exact. Terms with Integral atoms and genuine nonconservative couplings are left
    untouched.
    """
    terms = list(sp.Add.make_args(expr))
    changed = True
    while changed:
        changed = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                # Never attempt the fold across OPAQUE ζ-integrals: the exactness
                # check below calls ``.doit()``, which would try (and fail, slowly
                # / non-terminating) to evaluate ``∫c·φ·…dζ`` with symbolic
                # basis/weight.  Integral-bearing terms are left untouched.
                if terms[i].has(sp.Integral) or terms[j].has(sp.Integral):
                    continue
                si, sj = _single_deriv_factor(terms[i]), _single_deriv_factor(terms[j])
                if not (si and sj):
                    continue
                (ci, di), (cj, dj) = si, sj
                if di.variables != dj.variables:
                    continue
                cand = sp.Derivative(ci * di.expr, *di.variables)     # ∂_v(coeff·field)
                if pull_consts(sp.expand((cand - terms[i] - terms[j]).doit())) == 0:
                    terms[i] = cand
                    del terms[j]
                    changed = True
                    break
            if changed:
                break
    return sp.Add(*terms)


# ── Consolidate: generalised Leibniz fold (incl. inside integrals) ─────────

def _deriv_factor(term):
    """`(coeff, arg, v)` when `term` is coeff·∂_v(arg) with exactly one single-variable
    Derivative factor, else None.
    """
    facs = list(term.args) if isinstance(term, sp.Mul) else [term]
    ds = [f for f in facs
          if isinstance(f, sp.Derivative) and len(f.variables) == 1]
    if len(ds) != 1:
        return None
    dv = ds[0]
    return sp.Mul(*[f for f in facs if f is not dv]), dv.expr, dv.variables[0]


def leibniz_fold(expr):
    """Pairwise Leibniz fold with a shared passive factor M: M·a·∂_v b + M·b·∂_v a -> M·∂_v(a·b).
    Accepted only when exact and M carries no derivative or integral.
    """
    terms = list(sp.Add.make_args(sp.expand(expr)))
    changed = True
    while changed:
        changed = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if terms[i].has(sp.Integral) or terms[j].has(sp.Integral):
                    continue
                si, sj = _deriv_factor(terms[i]), _deriv_factor(terms[j])
                if not (si and sj):
                    continue
                (ci, ai, vi), (cj, aj, vj) = si, sj
                if vi != vj or aj == 0:
                    continue
                M = sp.cancel(ci / aj)
                if M.has(sp.Derivative) or M.has(sp.Integral):
                    continue
                if sp.expand(M * ai - cj) == 0:
                    terms[i] = M * sp.Derivative(ai * aj, vi)
                    del terms[j]
                    changed = True
                    break
            if changed:
                break
    return sp.Add(*terms)


def fold_self_products(expr):
    """Fold a self-product M·f·∂_v f into M·∂_v(f²/2) (M moved inside when v-independent), e.g.
    g·h·∂_x h -> ∂_x(g h²/2). Fires only when M does not reintroduce f, which makes the fold
    idempotent.
    """
    out = []
    for tm in sp.Add.make_args(sp.expand(expr)):
        df = _deriv_factor(tm)
        if df:
            coeff, arg, v = df
            if arg != 0 and not arg.is_number:
                M = sp.cancel(coeff / arg)
                # IDEMPOTENCE GUARD.  A genuine self-product ``M·f·∂_v f`` has the
                # differentiated ``f`` as an explicit factor, so ``M = coeff/f``
                # carries NO trace of ``f``: neither ``f`` itself (``not M.has(arg)``
                # — rejects ``a²∂a`` where ``M=a``) NOR ``f``'s symbols in its
                # DENOMINATOR.  The latter is the decisive one: a bare flux
                # ``∂_v(F)`` has ``coeff=1 ⇒ M=1/F`` and an already-folded
                # ``M·∂_v(F²/2)`` has ``M∝1/F`` — neither literally ``.has(F)``
                # (``1/(g h²/2)`` is ``2/(g h²)``, a different node), but both put
                # ``F``'s symbols in ``denom(M)``.  Without this the fold re-fires on
                # its own output and the ``consolidate`` fixpoint never terminates.
                if (not M.has(arg)
                        and not (sp.denom(M).free_symbols & arg.free_symbols)
                        and not M.has(sp.Derivative)
                        and not M.has(sp.Integral)
                        and sp.expand(M * arg - coeff) == 0):
                    inner = (M * arg**2 / 2 if not M.has(v) else arg**2 / 2)
                    folded = sp.Derivative(inner, v)
                    out.append(folded if not M.has(v) else M * folded)
                    continue
        out.append(tm)
    return sp.Add(*out)


def _coord_symbols(expr):
    """Coordinate-like symbols of `expr`: non-integer function arguments plus derivative and
    integral binding variables (parameters such as g are excluded).
    """
    coords = set()
    for f in expr.atoms(sp.Function):
        for a in f.args:
            coords |= {s for s in a.free_symbols if not s.is_integer}
    for d in expr.atoms(sp.Derivative):
        coords |= set(d.variables)
    for ig in expr.atoms(sp.Integral):
        coords |= {lim[0] for lim in ig.limits}
    return coords


def push_consts_into_flux(expr, coords=None):
    """Push every coordinate-free factor back into a bare derivative (inverse of pull_consts), e.g.
    (g/2)·∂_x(h²)·⟨c,φ_k⟩ -> ∂_x((g/2)·h²·⟨c,φ_k⟩). Fires only on a term with one derivative factor
    whose other factors are all coordinate-free; also applied inside every Sum.
    """
    if coords is None:
        coords = _coord_symbols(expr)

    def _push_one(term):
        """Push the coordinate-free factors of one term into its single derivative, if allowed."""
        facs = list(sp.Mul.make_args(term))
        derivs = [f for f in facs
                  if isinstance(f, sp.Derivative) and f.variables]
        if len(derivs) != 1:
            return term
        d = derivs[0]
        rest = [f for f in facs if f is not d]
        if not rest or any(f.free_symbols & coords for f in rest):
            return term
        return sp.Derivative(sp.Mul(*rest) * d.expr, *d.variables)

    out = []
    for term in sp.Add.make_args(expr):
        term = term.replace(
            lambda e: isinstance(e, sp.Sum),
            lambda e: sp.Sum(push_consts_into_flux(e.function, coords),
                             *e.args[1:]))
        out.append(_push_one(term))
    return sp.Add(*out)


def _hide_brackets(expr):
    """Replace every Galerkin bracket (an is_bracket integral) by an opaque symbol carrying its free
    integer indices; returns `(hidden_expr, restore_map)`.
    """
    brackets = sorted({ig for ig in expr.atoms(sp.Integral) if is_bracket(ig)},
                      key=sp.srepr)
    hide, restore = {}, {}
    for n, ig in enumerate(brackets):
        idxs = tuple(sorted((s for s in ig.free_symbols if s.is_integer),
                            key=str))
        sym = (sp.Function(f"_Br_{n}")(*idxs) if idxs
               else sp.Symbol(f"_Br_{n}"))
        hide[ig] = sym
        restore[sym] = ig
    return expr.xreplace(hide), restore


def merge_sums_over_add(expr):
    """Merge Sum(A, lims) + Sum(B, lims) -> Sum(A + B, lims) for identical limits, pulling
    index-independent coefficients inside first. The Sum analogue of merge_integrals_over_add.
    """
    groups, rest = {}, []
    for tm in sp.Add.make_args(sp.expand(expr)):
        facs = sp.Mul.make_args(tm)
        sums = [f for f in facs if isinstance(f, sp.Sum)]
        if len(sums) == 1:
            s = sums[0]
            coeff = sp.Mul(*[f for f in facs if f is not s])
            idxs = {lim[0] for lim in s.limits}
            if not (coeff.free_symbols & idxs):
                key = tuple(sorted(s.limits, key=lambda l: str(l[0])))
                groups.setdefault(key, []).append(coeff * s.function)
                continue
        rest.append(tm)
    out = sp.Add(*rest)
    for limits, summands in groups.items():
        out += sp.Sum(sp.Add(*summands), *limits)
    return out


def consolidate(expr, fold_self=False, fold_sums=False):
    """Unify terms under a common outer derivative: leibniz_fold on the top-level terms and, after
    merging same-limit integrals, recursively on every integrand and Sum summand.

    Args:
        expr: expression to fold.
        fold_self: also fold self-products (fold_self_products) and push constants back into fluxes.
        fold_sums: hide Galerkin brackets and merge same-index Sums first.
    """
    from zoomy_core.symbolic.primitives_canonical import merge_integrals_over_add
    restore = {}
    if fold_sums:
        # Canonicalise integral dummies first so alpha-equivalent brackets
        # (same body, different ``\hat ξ`` Dummy objects) hide to the SAME
        # symbol — otherwise the conservative pair never pairs up.
        from zoomy_core.model.operations import _canonicalize_integral_dummies
        expr = _canonicalize_integral_dummies(expr)
        expr, restore = _hide_brackets(expr)
        expr = merge_sums_over_add(expr)
    expr = merge_integrals_over_add(expr)

    def _rec(e):
        """Recursively leibniz-fold integrands and Sum summands, treating NumQuad brackets as atoms."""
        # Treat the NUMERICAL ⟨…⟩^N quadrature bracket as a true ATOM: do NOT
        # recurse into its (huge rational) body — that O(n) recursion is exactly
        # what extracting the bracket avoids.  (Galerkin Gram/Weight brackets are
        # left to recurse as before — their args are just integer indices.)
        if isinstance(e, sp.Function) and getattr(e.func, "_is_numquad", False):
            return e
        if isinstance(e, sp.Integral):
            return e.func(leibniz_fold(_rec(e.function)), *e.args[1:])
        if isinstance(e, sp.Sum):
            return e.func(leibniz_fold(_rec(e.function)), *e.args[1:])
        if e.args:
            return e.func(*[_rec(a) for a in e.args])
        return e

    # Iterate leibniz + self-product folds to a FIXPOINT: recovering a fully
    # expanded conservative flux ``a²∂_x h + 2 a h ∂_x a → ∂_x(h a²)`` needs two
    # passes — fold_self first exposes ``h ∂_x(a²)``, then leibniz combines it
    # with ``a²∂_x h`` — so a single pass leaves it half-folded.
    prev = None
    while expr != prev:
        prev = expr
        expr = leibniz_fold(_rec(expr))
        if fold_self:
            expr = fold_self_products(expr)
    if fold_self:
        # Re-conservatise: push pure constants (``g``) back into a bare flux that
        # ``pull_consts`` had split — but only when a clean ``∂_v(F)`` results.
        expr = push_consts_into_flux(expr)
    return expr.xreplace(restore) if restore else expr


class Consolidate(Operation):
    """Op form of consolidate with fold_self and fold_sums enabled."""

    whole_leaf_op = True
    log_level = 5

    def __init__(self, name="consolidate"):
        """Create the op."""
        super().__init__(name=name, description="leibniz fold (terms + integrands)")

    def _leaf_sp(self, sp_expr):
        """Apply consolidate with self-product and Sum folds on."""
        return consolidate(sp_expr, fold_self=True, fold_sums=True)


class GaussQuadrature(Operation):
    """Replace every definite Integral atom with numeric bounds by an order-point Gauss-Legendre sum
    Σ_g w_g f(ζ_g), for integrands without a closed form (e.g. non-polynomial material closures).
    Exact for polynomials up to degree 2·order − 1; apply after the basis is resolved.

    Args:
        var: integration variable (default `zeta`).
        order: number of Gauss nodes (>= 1).

    References:
        Golub, Welsch 1969, https://doi.org/10.1090/s0025-5718-69-99647-1
    """

    whole_leaf_op = True
    log_level = 1

    def __init__(self, var=None, order=8, name=None):
        """Store the node count and validate it."""
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._order = int(order)
        if self._order < 1:
            raise ValueError("GaussQuadrature: order must be >= 1")
        super().__init__(
            name=name or f"gauss_quadrature[{self._order}]",
            description=(f"∫ dζ → {self._order}-point Gauss–Legendre "
                         "quadrature (numerical integration of "
                         "analytically unintegrable closure terms)"))

    def _leaf_sp(self, sp_expr):
        """Evaluate ζ-derivatives with doit, then replace each numeric-bound integral by its mapped
        Gauss-node sum.
        """
        import numpy as _np
        integrals = list(sp.sympify(sp_expr).atoms(sp.Integral))
        if not integrals:
            return sp_expr
        x_ref, w_ref = _np.polynomial.legendre.leggauss(self._order)
        repl = {}
        for I in integrals:
            if len(I.limits) != 1 or len(I.limits[0]) != 3:
                continue                      # not a definite 1-D integral
            vv, a, b = I.limits[0]
            try:
                a_f, b_f = float(a), float(b)
            except TypeError:
                continue                      # symbolic bounds — not ours
            half = (b_f - a_f) / 2.0
            mid = (b_f + a_f) / 2.0
            # evaluate the ζ-derivatives EXPLICITLY first — substituting a
            # numeric node into an unevaluated Derivative leaves Subs junk
            # that no code printer accepts.  Do NOT sp.expand: for a rational
            # closure integrand (ν_t=C_μ sk⁴/se²) expand distributes N·D⁻¹ into a
            # >1M-char monster; node substitution works fine on the compact
            # ``.doit()`` form and yields a compact per-node summand.
            fn = sp.sympify(I.function).doit()
            acc = sp.S.Zero
            for xg, wg in zip(x_ref, w_ref):
                acc += sp.Float(wg * half) * fn.subs(
                    vv, sp.Float(mid + half * xg))
            repl[I] = acc
        return sp_expr.xreplace(repl) if repl else sp_expr


# ── opaque NUMERICAL-quadrature bracket  ⟨…⟩^N  ────────────────────────────
# Same idea as the Galerkin brackets (an opaque ``_is_bracket`` Function that
# every expand/fold/Consolidate treats as an ATOM), but for the NON-analytic
# closure integrals (the rational ν_t=C_μ sk⁴/se² terms).  ``DeferQuadrature``
# hides each surviving moment-bearing ``∫…dζ`` as ``⟨integrand⟩^N`` BEFORE the
# basis is substituted — while the body is still an OPAQUE-φ sum, so it stays
# COMPACT; it then rides the WHOLE post-projection pipeline (ResolveBasis /
# Consolidate / CoV / InvertMassMatrix) as a compact atom — nothing ever expands
# the rational body — and ``ResolveNumQuad(basis=…)`` resolves the basis at the
# Gauss nodes and rewrites it as the node sum at the very end (the system is
# settled).  This is what keeps high-N_k builds from blowing up: the rational
# never exists in concrete-basis symbolic form, only as numbers at the nodes.


def _numquad_latex(expr, printer=None, exp=None):
    """LaTeX for the NumQuad bracket ⟨f⟩^N."""
    render = printer._print if printer is not None else sp.latex
    inner = render(expr.args[0])
    s = rf"\left\langle {inner} \right\rangle^{{N}}"
    return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"


# ``NumQuad(integrand)`` — opaque numerical bracket over ζ∈[0,1].  ``_is_bracket``
# makes the fold/Consolidate machinery treat it as an atom; ``_is_numquad`` lets
# ResolveNumQuad target ONLY these (not the Galerkin Gram/Weight brackets).
NumQuad = type("NumQuad", (sp.Function,),
               {"_latex": _numquad_latex, "_is_bracket": True, "_is_numquad": True})


class DeferQuadrature(Operation):
    """Hide every surviving moment-bearing ∫_0^1 f dζ as an opaque NumQuad(f) atom, so non-analytic
    rational closure integrals ride the rest of the derivation unexpanded. Apply before
    ResolveBasis; pure-ζ Galerkin brackets are left for analytic closure.
    """

    whole_leaf_op = True
    log_level = 1

    def __init__(self, var=None, name="defer_quadrature"):
        """Set the integration variable (default `zeta`)."""
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        super().__init__(name=name,
                         description="∫dζ → opaque ⟨…⟩^N numerical bracket")

    def _leaf_sp(self, sp_expr):
        """Wrap each non-bracket definite integral over (0, 1) in NumQuad, keyed by its own
        variable.
        """
        from zoomy_core.model.operations import is_bracket
        e = sp.sympify(sp_expr)
        repl = {}
        for I in e.atoms(sp.Integral):
            # match ANY definite 1-D ∫_0^1 (like GaussQuadrature, by the
            # integral's OWN variable — the projection renames ζ to a Dummy ζ̂,
            # so a hard ``== zeta`` filter would miss every real integral) and
            # store that variable as the bracket's 2nd arg for resolution.
            # SKIP pure-ζ polynomial brackets (``is_bracket``) — those resolve
            # exactly via ResolveBasis; only the moment-bearing rational closure
            # integral needs the numerical ⟨…⟩^N treatment.
            if len(I.limits) == 1 and len(I.limits[0]) == 3 and not is_bracket(I):
                vv, a, b = I.limits[0]
                try:
                    if float(a) == 0.0 and float(b) == 1.0:
                        repl[I] = NumQuad(I.function, vv)
                except TypeError:
                    pass
        return e.xreplace(repl) if repl else sp_expr


class ResolveNumQuad(Operation):
    """Resolve every opaque NumQuad(f) to its order-point Gauss-Legendre sum over ζ in [0, 1]; run
    it as the last derivation step. `basis` (optional) resolves opaque φ(k, ζ) still present in a
    bracket body at the nodes.

    References:
        Golub, Welsch 1969, https://doi.org/10.1090/s0025-5718-69-99647-1
    """

    whole_leaf_op = True
    log_level = 1

    def __init__(self, var=None, order=8, basis=None, name=None):
        """Store the node count and the optional basis."""
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._order = int(order)
        self._basis = basis
        super().__init__(name=name or f"resolve_numquad[{self._order}]",
                         description=f"⟨…⟩^N → {self._order}-pt Gauss–Legendre sum")

    def _leaf_sp(self, sp_expr):
        """Resolve the basis in each bracket body, apply doit, and sum the weighted node values."""
        import numpy as _np
        e = sp.sympify(sp_expr)
        nqs = [a for a in e.atoms(sp.Function)
               if getattr(a.func, "_is_numquad", False)]
        if not nqs:
            return sp_expr
        x_ref, w_ref = _np.polynomial.legendre.leggauss(self._order)
        repl = {}
        for nq in nqs:
            vv = nq.args[1]                         # the bracket's own ζ̂ variable
            # If the body still carries OPAQUE φ (DeferQuadrature applied BEFORE
            # ResolveBasis — the design that keeps the rational compact), resolve
            # the basis to concrete polynomials FIRST, then ``.doit()`` to collapse
            # the now-concrete φ-derivatives (``∂ζ̂ P_i``) and any nested running
            # integral ``∫_0^ζ̂`` — only THEN sum the node values.  Doing ``.doit()``
            # AFTER concretisation is essential: a ``Derivative`` of an opaque φ
            # cannot collapse, and a stray ``Derivative(·, ζ̂)`` survives into the
            # node sub (``ζ̂ → number``) and breaks lambdify.
            fn = sp.sympify(nq.args[0])
            if self._basis is not None:
                fn = self._basis.resolve(fn, vv)
            fn = fn.doit()
            acc = sp.S.Zero
            for xg, wg in zip(x_ref, w_ref):
                acc += sp.Float(0.5 * wg) * fn.subs(vv, sp.Float(0.5 * xg + 0.5))
            repl[nq] = acc
        return e.xreplace(repl) if repl else sp_expr


class Simplify(Operation):
    """Canonicalisation that keeps fluxes grouped while exposing cancellations: pull_consts followed
    by consolidate. On an Equation it also runs Sort as a final step (`sort=False` skips it).
    """

    whole_leaf_op = True
    log_level = 5            # minor canonicalisation — hidden at low verbosity

    def __init__(self, name="simplify", sort=True):
        """`sort` toggles the final Sort pass on Equations."""
        self._sort = sort
        super().__init__(name=name, description="canonical simplify")

    def _leaf_sp(self, sp_expr):
        """Apply pull_consts, then consolidate."""
        return consolidate(pull_consts(sp_expr))

    def apply_to_equation(self, eq):
        """Canonicalise the residual, then tag and sort its terms when `sort` is on."""
        # Equation-level: canonicalise the residual, then tag + sort the terms
        # (a leaf-only ``_leaf_sp`` would lose the ordering when ``eq.expr`` is
        # rebuilt, so the sort has to happen here).
        eq.expr = self._leaf_sp(eq.expr)
        if self._sort:
            Sort().apply_to_equation(eq)
        return eq


# ── term tagging + sorting ─────────────────────────────────────────────────

#: Canonical physics order — the order terms are written by hand.
TAG_ORDER = [
    "time_derivative",        # ∂_t(…)            — the local time derivative
    "flux",                   # ∂_x(…) / ∂_ζ(…)   — conservative flux
    "pressure_flux",          # ∂_x(g h²/2 …)     — pressure flux (manual re-tag)
    "diffusion",              # ∂_x(… ∂_x …)      — diffusive (2nd-order) flux
    "nonconservative_flux",   # c(Q)·∂(Q)         — NCP
    "source",                 # algebraic / ∂(known field) — source
    "untagged",
]


def _strip_brackets_for_tagging(term):
    """Replace Galerkin brackets (is_bracket integrals and bracket atoms) by opaque Dummies so the
    tagger treats them as constants.
    """
    reps = {}
    for ig in term.atoms(sp.Integral):
        if is_bracket(ig):
            reps[ig] = sp.Dummy("B")
    for fn in term.atoms(sp.Function):
        if getattr(fn.func, "_is_bracket", False):
            reps[fn] = sp.Dummy("B")
    return term.xreplace(reps) if reps else term


def _classify_term(term, q_heads, t, spatial, detect_ncp=True):
    """Heuristic physics category of one additive term (time_derivative, flux, diffusion,
    nonconservative_flux, source or untagged), with brackets opaque and a single outer moment Sum
    peeled.
    """
    term = _strip_brackets_for_tagging(term)
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    # Peel a single outer moment Sum: classify by its per-mode summand.
    sums = [f for f in factors if isinstance(f, sp.Sum)]
    if len(sums) == 1:
        rest = sp.Mul(*[f for f in factors if f is not sums[0]])
        return _classify_term(rest * sums[0].function, q_heads, t, spatial,
                              detect_ncp)
    derivs = [f for f in factors
              if isinstance(f, sp.Derivative) and len(f.variables) == 1]
    if len(derivs) != 1:
        # No single outer derivative: a plain algebraic source, or — if it is
        # a tangle of derivatives — left untagged for the user to resolve.
        if any(isinstance(f, sp.Derivative) for f in factors):
            return "untagged"
        return "source"
    dv = derivs[0]
    var = dv.variables[0]
    arg = dv.expr
    coeff = sp.Mul(*[f for f in factors if f is not dv])
    coeff_has_field = bool(coeff.atoms(sp.Function))
    if var == t:
        return "time_derivative"
    if var in spatial and any(var in a.variables
                              for a in arg.atoms(sp.Derivative)):
        return "diffusion"
    if coeff_has_field:
        return "nonconservative_flux" if detect_ncp else "source"
    return "flux"


def _tag_context(eq):
    """`(q_heads, t, spatial)` read off the equation's parent model."""
    m = getattr(eq, "_model", None)
    if m is None:
        return set(), None, set()
    q_heads = {m._head(f) for v in m._Q.values()
               for f in m._as_field_list(v)}
    coords = m._coords
    return q_heads, coords[0], set(coords[1:-1])


class AutoTag(Operation):
    """Tag each untagged additive term with its physics category, leaving existing tags alone. Pairs
    with SortByTag; Sort runs both.
    """

    log_level = 5

    def __init__(self, name="auto_tag", detect_ncp=True):
        """`detect_ncp` toggles detection of nonconservative products (off: they become source)."""
        self._detect_ncp = detect_ncp
        super().__init__(name=name, description="tag untagged terms by category")

    def apply_to_equation(self, eq):
        """Tag the untagged terms of `eq`."""
        q_heads, t, spatial = _tag_context(eq)
        for tm in eq._terms:
            if tm.tag is None:
                tm.tag = _classify_term(tm.expr, q_heads, t, spatial,
                                        detect_ncp=self._detect_ncp)
        return eq


class SortByTag(Operation):
    """Reorder an equation's terms into TAG_ORDER, stable within each group."""

    log_level = 5

    def __init__(self, name="sort_by_tag"):
        """Create the op."""
        super().__init__(name=name, description="order terms by physics tag")

    def apply_to_equation(self, eq):
        """Sort `eq`'s terms by tag rank."""
        rank = {tag: i for i, tag in enumerate(TAG_ORDER)}
        eq._terms.sort(
            key=lambda tm: rank.get(tm.tag or "untagged", len(TAG_ORDER)))
        return eq


class Sort(Operation):
    """AutoTag then SortByTag; apply it last, because a later expression-level op rebuilds the term
    list and drops the ordering.
    """

    log_level = 5

    def __init__(self, name="sort", detect_ncp=True):
        """`detect_ncp` is passed to AutoTag."""
        self._detect_ncp = detect_ncp
        super().__init__(name=name, description="auto-tag + sort by physics tag")

    def apply_to_equation(self, eq):
        """Auto-tag, then sort `eq`'s terms."""
        AutoTag(detect_ncp=self._detect_ncp).apply_to_equation(eq)
        SortByTag().apply_to_equation(eq)
        return eq
