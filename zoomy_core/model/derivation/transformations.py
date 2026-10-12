"""Coordinate transformations applied to a Model: the vertical sigma map z -> zeta and a
channel-fitted (s, n) map. Both are Model-level rewrites that sit between the 3-D Model and the
depth-projection that builds the SystemModel.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = ["PDETransformation"]


# ── PDETransformation ────────────────────────────────────────────────────


class PDETransformation(Operation):
    """Sigma map ``z = b + h*zeta`` that rewrites every field ``f(t, x, z)`` to a decorated ``f~(t,
    x, zeta)`` via the chain rule on Derivative nodes. Model-level only (``model.apply(...)``); it
    also sigma-maps stored boundary conditions and records the map on the model.
    """

    # A coordinate transformation is a change of representation → rewrite BCs.
    transforms_bcs = True
    # Model-level: routed through ``apply_to_model`` once.
    whole_model_op = True

    _ACCENTS = {
        "tilde": r"\tilde",
        "hat": r"\hat",
        "bar": r"\bar",
        "check": r"\check",
    }

    def __init__(self, coord_map, *, decorate="tilde",
                 name="pde_transformation", description=None):
        """Build the map from ``{z: (zeta, Eq(z, b + h*zeta))}``; ``decorate`` picks the accent
        (tilde, hat, bar, check).
        """
        if len(coord_map) != 1:
            raise ValueError(
                "PDETransformation supports a single coordinate replacement "
                f"z → ζ; got {len(coord_map)} entries."
            )
        ((self._z, spec),) = coord_map.items()
        self._zeta, self._b, self._h = self._parse_spec(spec)
        self._accent = self._ACCENTS.get(decorate, decorate)
        self._via_model = False
        # Head decoration cache (one decorated Function per original head).
        self._decorated: dict = {}
        # ``{head: z_slot_index}`` for every field that appears with the vertical
        # coord ``z`` SOMEWHERE in the model.  Populated by a pre-scan in
        # ``apply_to_model`` so a BOUNDARY EVALUATION ``u(t,x,b)`` (whose z-slot
        # holds an interface value, not the free ``z``) is mapped to
        # ``ũ(t,x,(b−b)/h) = ũ(t,x,0)`` — i.e. evaluated at the σ-height of that
        # interface — instead of being left untouched.
        self._vertical_heads: dict = {}

        # The coords that receive the σ chain-rule jacobian = exactly the
        # coords the geometry depends on.  The metric correction is
        # ``∂_s(ζ h + b) = ζ ∂_s h + ∂_s b``, which is identically ZERO unless
        # ``s`` is a free symbol of ``b`` or ``h``.  So we read the chain
        # coords straight off ``b``/``h`` (e.g. ``b(t,x), h(t,x)`` → ``{t,
        # x}``) — no ``horizontal=`` argument needed at the call site.
        self._chain_coords = tuple(
            (self._b.free_symbols | self._h.free_symbols) - {self._zeta}
        )

        super().__init__(
            name=name,
            description=(description or
                         f"PDE transform: {self._z} → {self._zeta}, "
                         f"decorate={decorate}, chain-rule on "
                         f"{self._chain_coords} + ∂_{self._z}"),
        )

    def _parse_spec(self, spec):
        """Read ``(zeta, b, h)`` from the coord-map spec, with ``b = z|_{zeta=0}`` and ``h =
        dz/dzeta``.
        """
        zeta, relation = spec
        if isinstance(relation, sp.Equality):
            lhs, rhs = relation.args
            # Eq(z, b+hζ)  → inverse is rhs.  Eq(ζ, (z−b)/h) → solve for z.
            if lhs == self._z:
                inverse = rhs
            elif lhs == zeta:
                sols = sp.solve(sp.Eq(zeta, rhs), self._z)
                if not sols:
                    raise ValueError(
                        f"could not solve {relation} for {self._z}.")
                inverse = sols[0]
            else:
                inverse = rhs
        else:
            inverse = sp.sympify(relation)
        # z = b + h·ζ  ⇒  h = ∂_ζ(inverse), b = inverse|_{ζ=0}.
        h = sp.diff(inverse, zeta)
        b = inverse.subs(zeta, 0)
        return zeta, b, h

    # ── geometry accessors ────────────────────────────────────────────
    @property
    def jacobian(self):
        """Vertical Jacobian ``dz/dzeta = h``."""
        return self._h

    # ── decorated head minting ────────────────────────────────────────
    def _decorate_head(self, head):
        """Mint (and cache) a distinct Function subclass for the accented head, so ``u~`` never
        cross-substitutes with ``u``.
        """
        if head in self._decorated:
            return self._decorated[head]
        orig_name = head.__name__
        new_name = f"{self._accent}{{{orig_name}}}"
        accent = self._accent

        def _latex(expr, printer=None, exp=None):
            """LaTeX printer of the accented head (bare head under the strip-args printer)."""
            render = printer._print if printer is not None else sp.latex
            inner = sp.latex(sp.Symbol(orig_name))
            head_tex = rf"{accent}{{{inner}}}"
            # Honour the strip-args printer: ``\tilde{u}`` without the
            # ``(t, x, ζ)`` argument list (the bare-head form).
            if type(printer).__name__ == "_StripArgsLatexPrinter":
                s = head_tex
            else:
                args = ", ".join(str(render(a)) for a in expr.args)
                s = rf"{head_tex}\left({args}\right)"
            return s if exp is None else f"{s}^{{{exp}}}"

        new_head = type(new_name, (sp.Function,),
                        {"_latex": _latex, "_pde_decorated_from": orig_name})
        self._decorated[head] = new_head
        return new_head

    # ── decorated-field accessor (post-transform) ─────────────────────
    def decorated(self, field):
        """Return the decorated counterpart ``u~(t, x, zeta)`` of a physical field application, or
        the field itself if its head was never decorated.
        """
        head = self._decorated.get(field.func)
        if head is None:
            return field
        new_args = [a.xreplace({self._z: self._zeta}) for a in field.args]
        return head(*new_args)

    # ── leaf guard (model-only) ───────────────────────────────────────
    def _leaf_sp(self, sp_expr):
        """Leaf hook: rewrite only when reached through ``apply_to_model``, else raise TypeError."""
        if not self._via_model:
            raise TypeError(
                "PDETransformation is a model-level operation — it rewrites "
                "the whole coordinate system (z → ζ) and every equation. "
                "Apply it with `model.apply(PDETransformation(...))`, not "
                "`eq.apply(...)` / `term.apply(...)`."
            )
        return self._rewrite(sp_expr)

    # ── model dispatch ────────────────────────────────────────────────
    def apply_to_model(self, model):
        """Rewrite coordinates, all equations and stored assumptions, then publish the decoration
        map and move the unknowns to the decorated heads.
        """
        self._via_model = True
        try:
            self._rewrite_model_coords(model)
            # Pre-scan: record every field head that appears with the vertical
            # coord ``z`` and the arg position it occupies — order-independent, so
            # a KBC's boundary evaluation ``u(t,x,b)`` is mapped even when its
            # equation is rewritten before the bulk one carrying ``u(t,x,z)``.
            for eq in model._equations.values():
                for f in eq.expr.atoms(sp.Function):
                    if isinstance(f, sp.Sum) or self._z not in f.free_symbols:
                        continue
                    for i, a in enumerate(f.args):
                        if a == self._z or self._z in a.free_symbols:
                            self._vertical_heads.setdefault(f.func, i)
                            break
            for eq in model._equations.values():
                eq.apply(self, _no_history=True)
            # Rewrite stored assumptions (KinematicBC, …) too: σ-map both sides
            # of each rule so e.g. ``{w(b): ∂_t b + u(b)∂_x b}`` becomes
            # ``{w̃(0): ∂_t b + ũ(0)∂_x b}`` — ready to apply as a substitution.
            for asmpt in getattr(model, "_assumptions", {}).values():
                new_rel = {self._rewrite(lhs): self._rewrite(rhs)
                           for lhs, rhs in asmpt.subs_map.items()}
                # Stored assumptions are oriented Expressions (KinematicBC, …):
                # rewrite the ``{lhs: rhs}`` relation AND the residual term so
                # both sides are σ-mapped (``subs_map`` is a read-only view).
                asmpt._as_relation = new_rel
                asmpt.expr = sum((lhs - rhs for lhs, rhs in new_rel.items()),
                                 sp.S.Zero)
        finally:
            self._via_model = False
        # Publish the decoration map + σ-map metadata on the model so
        # ``kinematic_bc`` / depth-projection can read it.
        model._field_decoration = dict(self._decorated)
        model._sigma_from = self._z
        model._vertical = self._zeta
        # Transfer unknown-status from each decorated Q field to its decorated
        # head.  After the σ-map the physical-z head ``u`` is gone from every
        # equation (replaced by ``ũ``), so without this Q would silently drop
        # ``u`` instead of carrying ``ũ`` forward.  ``redeclare_unknown`` keys
        # on the family head, so we map ``u → ũ`` for every Q field that was
        # decorated.
        for nm, field in list(model._Q.items()):
            new_head = self._decorated.get(model._head(field))
            if new_head is not None:
                repl = {self._z: self._zeta}
                decorated = new_head(*[a.xreplace(repl) for a in field.args])
                model.redeclare_unknown(model._head(field), decorated)
        model._refresh_unknowns()
        return model

    def _rewrite_model_coords(self, model):
        """Replace ``z`` by ``zeta`` in the model coordinates and record ``z = b + h*zeta`` in
        ``coord_relations``.
        """
        coords = getattr(model, "_coords", None)
        if coords is None or self._z not in coords:
            return
        zeta = self._zeta
        model._coords = tuple(zeta if c == self._z else c for c in coords)
        model._vertical = zeta
        model.coord_relations[self._z] = self._b + self._h * zeta

    # ── the rewrite ───────────────────────────────────────────────────
    def _rewrite(self, e):
        """Recursively map z -> zeta: decorate z-dependent fields, evaluate interface values at
        their sigma height, apply the chain rule to Derivatives.
        """
        if isinstance(e, sp.Derivative):
            inner_new = self._rewrite(e.expr)
            result = inner_new
            for var, order in e.variable_count:
                for _ in range(int(order)):
                    result = self._apply_one_derivative(result, var)
            return result
        if isinstance(e, sp.Function) and not isinstance(e, sp.Sum):
            # A known/standard sympy function head (exp, log, sin, …) applied
            # to a field expression is a COMPOSITE, not a σ-field: recurse into
            # its argument(s) and PRESERVE the head — do NOT decorate it, strip
            # its inner ζ, nor fire the boundary-evaluation branch on the
            # function node itself.  Only genuine field/unknown applications
            # (sympy ``AppliedUndef``) carry the vertical coordinate and get the
            # σ-field treatment below.  Example: ``exp(lk(z))`` →
            # ``exp(l̃k(ζ))`` (head kept, inner field σ-mapped), instead of the
            # bogus ``\tilde{exp}`` / boundary-eval of ``lk`` at ``ζ``.
            if not isinstance(e, sp.core.function.AppliedUndef):
                new_args = tuple(self._rewrite(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
                return e
            # A field application.  If it contains z, decorate the head and
            # map z → ζ in the arg list; recurse into each arg.
            if self._z in e.free_symbols:
                new_args = [self._rewrite(a.xreplace({self._z: self._zeta}))
                            for a in e.args]
                new_head = self._decorate_head(e.func)
                return new_head(*new_args)
            # BOUNDARY EVALUATION of a vertical field: ``u(t,x,Z)`` with ``Z`` an
            # interface value (no free ``z``).  Evaluate the σ-field at the height
            # of that interface, ζ = (Z − b)/h  (e.g. Z = b → ζ = 0, b+h → 1).
            if e.func in self._vertical_heads:
                slot = self._vertical_heads[e.func]
                new_args = [self._rewrite(a) for a in e.args]
                if slot < len(new_args):
                    new_args[slot] = sp.simplify(
                        (e.args[slot] - self._b) / self._h)
                return self._decorate_head(e.func)(*new_args)
            # No z dependence — leave the head, but still recurse args.
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
            return e
        if e.args:
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
            return e
        # Atom: free z in non-Function context → b + h·ζ.
        if e == self._z:
            return self._zeta * self._h + self._b
        return e

    def _apply_one_derivative(self, inner_expr, var):
        """Chain rule for one derivative: ``d_z -> (1/h) d_zeta`` and ``d_s -> d_s - (d_s(zeta h +
        b)/h) d_zeta``.
        """
        zeta, h, b = self._zeta, self._h, self._b
        if var == self._z:
            # ∂_z u → (1/h)·∂_ζ ũ
            return (sp.Integer(1) / h) * sp.Derivative(inner_expr, zeta)
        if var in self._chain_coords:
            # ∂_s u|_z → ∂_s ũ − (∂_s(ζh+b)/h)·∂_ζ ũ
            # Skip the jacobian correction when ``inner_expr`` has no ζ
            # dependency — its ∂_ζ is identically zero and emitting it would
            # leave a stray ``Derivative(f, ζ)`` atom (e.g. ``∂_t b`` on a bed
            # with no z → must collapse to ``∂_t b`` directly).
            if zeta not in inner_expr.free_symbols:
                return sp.Derivative(inner_expr, var)
            jacobian = sp.Derivative(zeta * h + b, var)
            return (sp.Derivative(inner_expr, var)
                    - (jacobian / h) * sp.Derivative(inner_expr, zeta))
        return sp.Derivative(inner_expr, var)

    def _repr_latex_(self):
        """Notebook LaTeX summary of the map."""
        return (rf"$\zeta = \frac{{{sp.latex(self._z)} - {sp.latex(self._b)}}}"
                rf"{{{sp.latex(self._h)}}}, \quad "
                rf"u({sp.latex(self._z)}) \mapsto {self._accent}{{u}}"
                rf"({sp.latex(self._zeta)}), \quad "
                rf"\partial_{sp.latex(self._z)} u \mapsto "
                rf"\tfrac{{1}}{{{sp.latex(self._h)}}}\partial_\zeta "
                rf"{self._accent}{{u}}$")

class ParametrizedGeometry(Operation):
    """Orthogonal map ``X(s, n) = r(s) + n N(s)`` of a Cartesian pair (x, y) onto coordinates fitted
    to a planar centreline ``r(s)``. Curvature, metric ``m = 1 - kappa n`` and the chain rule on
    Derivative nodes are all derived from the curve; ``kappa = 0`` is the identity.
    """

    def __init__(self, centreline, coords, new_coords, *, decorate="tilde",
                 name="parametrized_geometry", description=None):
        """Derive speed, curvature, tangent, normal and metric from ``centreline`` (2-vector in the
        new parameter) and the old/new coordinate pairs.
        """
        self._r = sp.Matrix(centreline)
        if self._r.shape != (2, 1):
            raise ValueError(f"centreline must be planar 2-vector; got {self._r.shape}")
        self._x, self._y = coords
        self._s, self._n = new_coords
        rp = sp.diff(self._r, self._s)
        rpp = sp.diff(self._r, self._s, 2)
        self._speed = sp.simplify(sp.sqrt(rp[0] ** 2 + rp[1] ** 2))
        self._kappa = sp.simplify(
            (rp[0] * rpp[1] - rp[1] * rpp[0]) / self._speed ** 3)
        self._T = sp.simplify(rp / self._speed)
        self._N = sp.Matrix([-self._T[1], self._T[0]])
        self._m = sp.simplify(1 - self._kappa * self._n)
        self._decorated: dict = {}
        super().__init__(
            name=name,
            description=(description or
                         f"parametrized geometry: ({self._x}, {self._y}) -> "
                         f"({self._s}, {self._n}), kappa = {self._kappa}"),
        )

    @property
    def kappa(self):
        """Signed curvature of the centreline."""
        return self._kappa

    @property
    def metric(self):
        """Streamwise scale factor ``m = 1 - kappa n``."""
        return self._m

    @property
    def jacobian(self):
        """Area element factor ``m`` (``dA = m dn ds``)."""
        return self._m

    def admissible(self, half_width):
        """Condition ``m > 0`` across a section of the given half width."""
        return sp.simplify(sp.Abs(self._kappa) * 2 * half_width < 2)

    def _apply_one_derivative(self, inner, var):
        """Chain rule ``d_x -> (T_x/m) d_s + N_x d_n`` (likewise ``d_y``); other variables are
        unchanged.
        """
        if var == self._x:
            return (self._T[0] / self._m) * sp.Derivative(inner, self._s) \
                 + self._N[0] * sp.Derivative(inner, self._n)
        if var == self._y:
            return (self._T[1] / self._m) * sp.Derivative(inner, self._s) \
                 + self._N[1] * sp.Derivative(inner, self._n)
        return sp.Derivative(inner, var)

    def _rewrite(self, e):
        """Apply the derivative chain rule recursively to every Derivative node."""
        if isinstance(e, sp.Derivative):
            result = self._rewrite(e.expr)
            for var, order in e.variable_count:
                for _ in range(int(order)):
                    result = self._apply_one_derivative(result, var)
            return result
        if e.args:
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    def _apply_leaf(self, expr):
        """Leaf hook: rewrite one expression."""
        return self._rewrite(expr)

