"""Non-dimensional scaling of a derivation `Model`, the order-based shallow truncation and the
closure inserted in the scaled frame (design: docs/dev/symbolic_solver/research/
nondimensional_scaling.md).

The pipeline acts on the 3-D balance laws in physical coordinates, before the sigma map:

    m.apply(NonDimensionalize(...))     # scales + groups (Buckingham-Pi checked)
    m.apply(ShallowScaling(...))        # drop every term beyond its row's eps order
    m.apply(InsertClosure(closures))    # material law in the scaled frame, consistency check
    m.apply(ShallowScaling())           # same orders: the closure's own sub-leading terms
    m.apply(Dimensionalize())           # back to the physical variables

`NonDimensionalize` rewrites every term `c * T(fields)` of a row as `Pi(groups) * T(hat fields)`:
fields are replaced by `scale * hat`, a derivative in a coordinate picks up `1 / scale`, the row is
divided by its scale and every dimensional constant is written through the repeating reference
scales and the groups, which must then cancel (dimensional homogeneity).  The coordinate symbols
are kept and read as the scaled coordinates.  Group coefficients always stand in front of the
field part, so the eps order of a term is read off its coefficient.
"""
from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef

from zoomy_core.misc.options import Option
from zoomy_core.model.operations import Operation


# ── dimensions ───────────────────────────────────────────────────────────────


class Dim:
    """Physical dimension `M^a L^b T^c` (rational exponents)."""

    __slots__ = ("exp",)

    def __init__(self, M=0, L=0, T=0):
        """Store the exponents of mass, length and time."""
        self.exp = (sp.Rational(M), sp.Rational(L), sp.Rational(T))

    def __mul__(self, other):
        """Product of two dimensions."""
        return Dim(*(a + b for a, b in zip(self.exp, other.exp)))

    def __truediv__(self, other):
        """Quotient of two dimensions."""
        return Dim(*(a - b for a, b in zip(self.exp, other.exp)))

    def __pow__(self, n):
        """Rational power of a dimension."""
        n = sp.Rational(n)
        return Dim(*(a * n for a in self.exp))

    def __eq__(self, other):
        """Equal exponents."""
        return isinstance(other, Dim) and self.exp == other.exp

    def __hash__(self):
        """Hash of the exponent tuple."""
        return hash(self.exp)

    @property
    def dimensionless(self):
        """True for `M^0 L^0 T^0`."""
        return all(e == 0 for e in self.exp)

    def __repr__(self):
        """`M^a L^b T^c`, `1` when dimensionless."""
        parts = [f"{n}^{e}" if e != 1 else n for n, e in zip("MLT", self.exp) if e != 0]
        return " ".join(parts) or "1"


ONE = Dim()
LENGTH, TIME, MASS = Dim(L=1), Dim(T=1), Dim(M=1)
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME ** 2
DENSITY = MASS / LENGTH ** 3
STRESS = MASS / (LENGTH * TIME ** 2)
KINEMATIC_VISCOSITY = LENGTH ** 2 / TIME

#: Dimensions of the model parameters, by name (closures add theirs through `dimensions()`).
PARAMETER_DIMENSIONS = {
    "g": ACCELERATION, "rho": DENSITY, "nu": KINEMATIC_VISCOSITY, "e_x": ONE,
}


class DimensionError(ValueError):
    """A scale, group or term that is not dimensionally consistent."""


class ScalingError(ValueError):
    """A closure or term that is inconsistent with the scaling (the consistency check)."""


# ── Buckingham Pi ────────────────────────────────────────────────────────────


def _exponents(expr, symbols):
    """Exponent vector of the monomial `expr` over `symbols` (a numeric factor is allowed)."""
    expr = sp.powsimp(sp.sympify(expr), force=True)
    vec = [sp.S.Zero] * len(symbols)
    index = {s: i for i, s in enumerate(symbols)}
    for f in sp.Mul.make_args(expr):
        if f.is_number:
            continue
        base, e = f.as_base_exp()
        if base not in index or not e.is_number:
            raise DimensionError(f"{expr} is not a monomial in the quantities {list(symbols)} "
                                 f"(factor {f})")
        vec[index[base]] += e
    return vec


def _monomial(symbols, vec):
    """`prod symbols**vec`."""
    return sp.Mul(*[s ** e for s, e in zip(symbols, vec) if e != 0])


def _distribute(e):
    """Distribute products over sums down to the additive terms, never inside a derivative,
    function or power (`expand(deep=False)` term by term, to a fixpoint)."""
    while True:
        out = sp.Add(*[sp.expand(t, deep=False, mul=True, multinomial=False, power_exp=False,
                                 power_base=False, log=False)
                       for t in sp.Add.make_args(e)])
        if out == e:
            return out
        e = out


class PiBasis:
    """The Buckingham-Pi check of a set of target groups over dimensional quantities.

    Verifies that every group is dimensionless, that the groups are independent, and that they
    are complete (`#groups = #quantities - rank(dimension matrix)`), then writes every quantity
    as a monomial of the repeating quantities and the groups.  The groups are never chosen here
    (the choice is not unique); `suggest()` proposes a basis.

    Args:
        quantities: `{Symbol: Dim}` of the reference scales and parameters.
        groups: `{group Symbol: definition}`, a definition a monomial of the quantities and of the
            groups defined before it.
        prefer: quantities to try first as repeating variables, in order.
    """

    def __init__(self, quantities, groups, prefer=()):
        """Run the check; raises `DimensionError` naming the offending or missing group."""
        self.quantities = dict(quantities)
        q = list(self.quantities)
        self.q = q
        D = sp.Matrix([[self.quantities[s].exp[i] for s in q] for i in range(3)])
        self.D = D
        # group definitions in the quantities (earlier groups substituted)
        self.groups, self.definition = list(groups), {}
        for gsym, gdef in groups.items():
            gdef = sp.sympify(gdef).xreplace(self.definition)
            self.definition[gsym] = gdef
            vec = _exponents(gdef, q)
            dim = Dim()
            for s, e in zip(q, vec):
                dim = dim * self.quantities[s] ** e
            if not dim.dimensionless:
                raise DimensionError(f"group {gsym} = {groups[gsym]} is not dimensionless "
                                     f"(dimension {dim})")
        G = sp.Matrix([_exponents(self.definition[g], q) for g in self.groups]) \
            if self.groups else sp.zeros(0, len(q))
        self.G = G
        if G.rows and G.rank() < G.rows:
            for i in range(G.rows):
                rest = G[[j for j in range(G.rows) if j != i], :]
                if rest.rank() == G.rank():
                    raise DimensionError(
                        f"group {self.groups[i]} = {groups[self.groups[i]]} depends on the others "
                        f"({', '.join(str(g) for g in self.groups if g != self.groups[i])}); "
                        "the groups must be independent")
        need = len(q) - D.rank()
        if G.rows < need:
            missing = self._missing()
            raise DimensionError(
                f"incomplete groups: {len(q)} quantities of rank {D.rank()} need {need} groups, "
                f"{G.rows} given; missing e.g. {missing}")
        if G.rows > need:
            raise DimensionError(f"{G.rows} groups for {need} independent ones")
        # repeating quantities: a maximal dimensionally independent subset, `prefer` first
        order = [s for s in prefer if s in self.quantities] + [s for s in q if s not in prefer]
        rep = []
        for s in order:
            trial = D[:, [q.index(r) for r in rep + [s]]]
            if trial.rank() > len(rep):
                rep.append(s)
            if len(rep) == D.rank():
                break
        self.repeating = rep
        # every quantity = repeating monomial * group monomial (unique: {e_r} + groups is a basis)
        B = sp.Matrix.hstack(*([sp.Matrix([1 if s == r else 0 for s in q]) for r in rep]
                               + [G.row(i).T for i in range(G.rows)]))
        self.express = {}
        for s in q:
            if s in rep:
                continue
            e = sp.Matrix([1 if t == s else 0 for t in q])
            coef = B.LUsolve(e)
            self.express[s] = (_monomial(rep, coef[:len(rep)])
                               * _monomial(self.groups, coef[len(rep):]))

    def _missing(self):
        """A dimensionless monomial not in the span of the given groups (the suggestion)."""
        G = self.G
        for v in self.D.nullspace():
            if G.rows == 0 or sp.Matrix.vstack(G, v.T).rank() > G.rank():
                den = sp.ilcm(*[sp.fraction(c)[1] for c in v])
                return _monomial(self.q, list(v * den))
        return None

    def suggest(self):
        """A basis of dimensionless groups over the quantities (integer exponents)."""
        out = []
        for v in self.D.nullspace():
            den = sp.ilcm(*[sp.fraction(c)[1] for c in v])
            out.append(_monomial(self.q, list(v * den)))
        return out


# ── field / scaling specification ────────────────────────────────────────────


class Role(Option):
    """The kind of a scaled row; `ShallowScaling` takes one truncation order per role."""
    #: a horizontal momentum row
    horizontal = "horizontal"
    #: the vertical momentum row
    vertical = "vertical"
    #: the mass balance
    mass = "mass"


class Affine:
    """A field scaled about a known background: `f = background + scale * hat`, e.g. the pressure
    `p = rho g (s - z) + rho U^2 q_hat` (thesis eq. (pressure-scaling)).

    Args:
        background: dimensional expression in the model's fields and coordinates.
        scale: scale of the fluctuation.
        name: name of the fluctuation's scaled field (default: the field's name).
    """

    def __init__(self, background, scale, name=None):
        """Store the background, the scale and the scaled field's name."""
        self.background, self.scale, self.name = sp.sympify(background), scale, name


class Regime:
    """The orders of the groups in the small parameter, `{group name: k}` for `group = O(eps^k)`,
    e.g. `Regime(De=1)` for `De = O(eps)`.  Groups not listed take their default order (the
    scaling's, then the closures'); the small parameter itself is order 1.
    """

    def __init__(self, **orders):
        """Store the orders by group name."""
        self.orders = {k: sp.Rational(v) for k, v in orders.items()}

    def merged(self, defaults):
        """These orders over `defaults` (`{name: k}`)."""
        out = dict(defaults)
        out.update(self.orders)
        return out

    def __repr__(self):
        """`Regime(De=1, ...)`."""
        return "Regime(" + ", ".join(f"{k}={v}" for k, v in sorted(self.orders.items())) + ")"

    def __eq__(self, other):
        """Equal orders."""
        return isinstance(other, Regime) and self.orders == other.orders

    def __hash__(self):
        """Hash of the orders."""
        return hash(tuple(sorted(self.orders.items())))


class _Field:
    """One scaled field: original head, scaled head, scale and optional background."""

    def __init__(self, head, hat, scale, dim, background=None):
        """Store the heads, the scale, the dimension and the background."""
        self.head, self.hat, self.scale, self.dim = head, hat, scale, dim
        self.background = background


def _hat_head(head, name=None):
    """The scaled-field head of `head` (`\\check{u}` for `u`), with the same assumptions."""
    nm = name or head.__name__
    return sp.Function(rf"\check{{{nm}}}", **getattr(head, "_explicit_class_assumptions", {}))


class NonDimState:
    """The scaled frame of a model: fields and coordinates with their scales, the groups and the
    Pi basis, the scaled rows (role, scale), the regime and orders once truncated, and the
    closure record of `InsertClosure`.
    """

    def __init__(self, fields, coord_scales, references, parameters, groups, rows, small,
                 default_orders, prefer=()):
        """Build the state and run the Pi check over the reference scales and parameters."""
        self.fields = fields                  # {orig head: _Field}
        self.hat_of = {f.hat: f for f in fields.values()}
        self.coord_scales = dict(coord_scales)
        self.references = dict(references)   # {Symbol: Dim}
        self.parameters = dict(parameters)    # {Symbol: Dim}
        self.group_defs = dict(groups)        # {Symbol: definition}
        self.rows = dict(rows)                # {eq name: (Role, scale)}
        self.small = small
        self.default_orders = dict(default_orders)
        self.prefer = list(prefer)
        self.regime, self.orders = None, None
        self.closure = None                   # InsertClosure record
        self.bindings = {}                    # {reference scale: value} fixed by a closure
        self.log = []                         # (operation, description) of the pipeline
        self.model_parameters = set()         # every parameter Symbol of the model
        self._pi()

    # ── Pi basis ──
    def _pi(self):
        """(Re)run the Pi check over references + parameters with the current groups."""
        quantities = {**self.references, **self.parameters}
        self.pi = PiBasis(quantities, self.group_defs, prefer=self.prefer)
        self.groups = list(self.group_defs)
        self.consts = set(quantities) | set(self.groups)

    def extend(self, parameters=None, groups=None, orders=None):
        """Add closure parameters (`{Symbol: Dim}`), their groups and default orders, and rerun
        the Pi check (a parameter without a group fails it, naming the missing group)."""
        for s, dim in (parameters or {}).items():
            if s not in self.parameters and s not in self.references:
                self.parameters[s] = dim
        for g, d in (groups or {}).items():
            self.group_defs.setdefault(g, d)
        for k, v in (orders or {}).items():
            self.default_orders.setdefault(k, sp.Rational(v))
        self._pi()

    def bind(self, reference, value, parameters):
        """Fix the free reference scale `reference` to `value`, an expression in the other
        quantities and the new `parameters` (`{Symbol: Dim}`), e.g. the stress scale
        `S = rho nu U / H` of a Newtonian closure: the groups and field scales are rewritten
        and the Pi check is rerun without `reference`."""
        if reference not in self.references:
            raise ScalingError(f"{reference} is not a free reference scale")
        value = sp.sympify(value)
        self.references.pop(reference)
        self.parameters.update(parameters)
        rep = {reference: value}
        self.group_defs = {g: sp.sympify(d).xreplace(rep) for g, d in self.group_defs.items()}
        for F in self.fields.values():
            F.scale = sp.sympify(F.scale).xreplace(rep)
        self.coord_scales = {c: sp.sympify(s).xreplace(rep) for c, s in self.coord_scales.items()}
        self.bindings[reference] = value
        self.prefer = [p for p in self.prefer if p != reference]
        self._pi()

    def scale_of(self, field):
        """The scale of a field (head or applied) or of a coordinate symbol."""
        if field in self.coord_scales:
            return self.coord_scales[field]
        head = field if isinstance(field, sp.FunctionClass) else field.func
        return self.fields[head].scale

    # ── dimensional constant -> group monomial ──
    def _const(self, e):
        """True for a factor made of numbers, reference scales, parameters and groups only."""
        return e.is_number or (e.free_symbols <= (self.consts | self.model_parameters)
                               and not e.atoms(AppliedUndef))

    def to_groups(self, coeff, what=""):
        """Write the dimensional constant `coeff` through the groups; raises `DimensionError`
        when a repeating quantity is left (the term is not dimensionally homogeneous)."""
        c = sp.sympify(coeff).xreplace(self.bindings).xreplace(self.pi.express)
        c = sp.powsimp(c, force=True)
        unknown = c.free_symbols & (self.model_parameters - self.consts)
        if unknown:
            raise DimensionError(f"{what}: parameters {sorted(map(str, unknown))} are not part of "
                                 "the scaling (no dimension / group)")
        left = c.free_symbols & set(self.pi.repeating)
        if left:
            raise DimensionError(f"{what}: not dimensionally homogeneous, {c} keeps "
                                 f"{sorted(map(str, left))}")
        return c

    def from_groups(self, coeff):
        """The dimensional value of a group coefficient (groups -> definitions)."""
        return sp.sympify(coeff).xreplace(self.pi.definition).xreplace(self.bindings)

    # ── orders ──
    def order_map(self):
        """`{group Symbol: order}` of the active regime (small parameter: 1)."""
        orders = dict(self.default_orders)
        if self.regime is not None:
            orders = self.regime.merged(orders)
        out = {}
        for g in self.groups:
            out[g] = sp.S.One if g == self.small else sp.Rational(orders.get(str(g), 0))
        return out

    def split(self, term):
        """`(coefficient, field part)` of a scaled term; the coefficient is the product of the
        constant factors.  Raises when a group is left inside the field part."""
        coeff, rest = [], []
        for f in sp.Mul.make_args(term):
            (coeff if self._const(f) else rest).append(f)
        rest = sp.Mul(*rest)
        if rest.free_symbols & set(self.groups):
            raise ScalingError(f"group inside the field part of the scaled term {term}")
        return sp.Mul(*coeff), rest

    def order(self, term):
        """eps order of a scaled term, read off its group coefficient."""
        coeff, _ = self.split(term)
        omap = self.order_map()
        out = sp.S.Zero
        for f in sp.Mul.make_args(coeff):
            if f.is_number:
                continue
            base, e = f.as_base_exp()
            if base not in omap:
                raise ScalingError(f"coefficient {coeff} is not a monomial in the groups")
            out += e * omap[base]
        return out

    # ── dimensional <-> scaled ──
    def _pull(self, D):
        """`Derivative(sum c_i T_i, v)` -> `sum c Derivative(sum T_i, v)` grouped by the
        constant factor `c`, so the constants stand in front and same-constant parts stay
        inside one derivative."""
        groups = {}
        for t in sp.Add.make_args(_distribute(D.expr)):
            c, r = [], []
            for f in sp.Mul.make_args(t):
                (c if self._const(f) else r).append(f)
            groups.setdefault(sp.Mul(*c), []).append(sp.Mul(*r))
        out = sp.S.Zero
        for c, rs in groups.items():
            arg = sp.Add(*rs)
            if arg.is_number:
                continue
            out += c * sp.Derivative(arg, *D.variable_count)
        return out

    def _coord_factor(self, D, inverse=False):
        """`prod scale_v**(-n)` of the derivative variables (`**n` for `inverse`)."""
        f = sp.S.One
        for v, n in D.variable_count:
            s = self.coord_scales.get(v)
            if s is None:
                raise DimensionError(f"derivative in {v}, which has no scale")
            f *= s ** (n if inverse else -n)
        return f

    def scaled(self, e):
        """Dimensional expression -> scaled fields with explicit dimensional constants."""
        if isinstance(e, AppliedUndef) and e.func in self.fields:
            F = self.fields[e.func]
            out = F.scale * F.hat(*e.args)
            if F.background is not None:
                out = self.scaled(F.background) + out
            return out
        if e in self.coord_scales:
            return self.coord_scales[e] * e
        if isinstance(e, sp.Derivative):
            inner = e.expr
            fac = self._coord_factor(e)
            if isinstance(inner, AppliedUndef) and inner.func in self.fields \
                    and self.fields[inner.func].background is not None:
                F = self.fields[inner.func]
                # the background is a known function: differentiate it now
                return fac * (sp.diff(self.scaled(F.background), *e.variable_count)
                              + F.scale * sp.Derivative(F.hat(*inner.args), *e.variable_count))
            return fac * self._pull(sp.Derivative(self.scaled(inner), *e.variable_count))
        if isinstance(e, (sp.Subs, sp.Integral)):
            raise ScalingError(f"cannot scale {type(e).__name__} {e}: the scaling acts on the "
                               "balance laws before any trace or integral")
        if isinstance(e, sp.Pow) and not e.exp.is_integer:
            # |U| = sqrt(U^2 u^2 + U^2 v^2): the common scale leaves the root
            return sp.Pow(sp.factor_terms(self.scaled(e.base)), e.exp)
        if e.args and not isinstance(e, AppliedUndef):
            return e.func(*[self.scaled(a) for a in e.args])
        return e

    def unscaled(self, e):
        """Scaled field part -> dimensional fields (inverse of `scaled`, constants in front)."""
        if isinstance(e, AppliedUndef) and e.func in self.hat_of:
            F = self.hat_of[e.func]
            out = F.head(*e.args)
            if F.background is not None:
                out = out - F.background
            return out / F.scale
        if e in self.coord_scales:
            return e / self.coord_scales[e]
        if isinstance(e, sp.Derivative):
            inner = e.expr
            fac = self._coord_factor(e, inverse=True)
            if isinstance(inner, AppliedUndef) and inner.func in self.hat_of \
                    and self.hat_of[inner.func].background is not None:
                F = self.hat_of[inner.func]
                return fac * (sp.Derivative(F.head(*inner.args), *e.variable_count)
                              - sp.diff(F.background, *e.variable_count)) / F.scale
            return fac * self._pull(sp.Derivative(self.unscaled(inner), *e.variable_count))
        if e.args and not isinstance(e, AppliedUndef):
            return e.func(*[self.unscaled(a) for a in e.args])
        return e

    def to_scaled_row(self, expr, name):
        """A dimensional row -> `sum Pi_k T_k(hat fields)` divided by the row scale."""
        _role, rscale = self.rows[name]
        out = sp.S.Zero
        for term in sp.Add.make_args(expr):
            ts = _distribute(self.scaled(term))
            for t in sp.Add.make_args(ts):
                c, r = self.split(t)
                out += self.to_groups(c / rscale, what=f"{name}: term {term}") * r
        return out

    def normalize(self, expr, name):
        """Re-split a scaled row whose derivatives may carry group factors inside (after a
        substitution), so every coefficient stands in front again."""
        out = sp.S.Zero

        def _norm(e):
            if isinstance(e, sp.Derivative):
                return self._pull(sp.Derivative(_norm(e.expr), *e.variable_count))
            if e.args and not isinstance(e, AppliedUndef):
                return e.func(*[_norm(a) for a in e.args])
            return e
        for t in sp.Add.make_args(_distribute(_norm(expr))):
            c, r = self.split(t)
            out += c * r
        return out

    def to_dimensional_row(self, expr, name):
        """A scaled row -> its dimensional form (times the row scale)."""
        _role, rscale = self.rows[name]
        out = sp.S.Zero
        for term in sp.Add.make_args(expr):
            c, r = self.split(term)
            dim_c = self.from_groups(c) * rscale
            out += _distribute(dim_c * self.unscaled(r))
        return out


# ── operators ────────────────────────────────────────────────────────────────


class NonDimensionalize(Operation):
    """Rewrite the listed rows of a model in scaled variables and dimensionless groups.

    Every coordinate, field, parameter and stress component gets a scale (`scales`); every
    symbol declares its dimension (`dimensions`, the model parameters default to
    `PARAMETER_DIMENSIONS`).  The target groups (`groups`) are the user's choice and are only
    validated (`PiBasis`): dimensionless, independent and complete.

    Args:
        scales: `{applied field | coordinate Symbol: scale | Affine}`.
        groups: `{group Symbol: definition}` over the reference scales and parameters.
        references: `{reference-scale Symbol: Dim}` (e.g. `L, H, U, S`).
        dimensions: `{applied field | coordinate: Dim}` of every scaled field and coordinate.
        rows: `{equation name: (Role, row scale)}`; the row is divided by its scale.
        small: the group that is the expansion parameter (order 1).
        orders: default regime `{group name: order}`.
        prefer: reference scales tried first as repeating quantities.
    """

    whole_model_op = True

    def __init__(self, scales, groups, references, dimensions, rows, small, orders=None,
                 prefer=(), name="nondimensionalize"):
        """Store the scaling specification."""
        super().__init__(name=name, description="non-dimensionalize (scales + groups)")
        self.scales, self.groups, self.references = scales, groups, references
        self.dimensions, self.rows, self.small = dimensions, rows, small
        self.orders, self.prefer = dict(orders or {}), tuple(prefer)

    def apply_to_model(self, m):
        """Check the scales and groups, rewrite the rows, record `m.nondim`."""
        if getattr(m, "nondim", None) is not None:
            raise ScalingError("the model is already in a scaled frame")
        refs = dict(self.references)
        pars = {getattr(m.parameters, n): n for n in m.parameters.keys()}
        clash = [n for s, n in pars.items()
                 if s in refs or (s in self.groups and self.groups[s] != s)]
        if clash:
            raise ScalingError(f"parameters {clash} clash with a reference scale or group")
        # the parameters the rows, scales and groups use; each must declare its dimension
        used = set()
        for name in self.rows:
            used |= m._equations[name].expr.free_symbols
        for spec in list(self.scales.values()) + list(self.groups.values()):
            for e in ((spec.scale, spec.background) if isinstance(spec, Affine) else (spec,)):
                used |= sp.sympify(e).free_symbols
        params = {}
        for s in used:
            if s in pars:
                if pars[s] not in PARAMETER_DIMENSIONS:
                    raise DimensionError(f"parameter {pars[s]} declares no dimension "
                                         "(PARAMETER_DIMENSIONS)")
                params[s] = PARAMETER_DIMENSIONS[pars[s]]
        pi = PiBasis({**refs, **params}, self.groups, prefer=self.prefer)
        fields, coord_scales = {}, {}
        for key, spec in self.scales.items():
            dim = self.dimensions[key]
            sc = sp.sympify(spec.scale if isinstance(spec, Affine) else spec)
            flat = sc.xreplace(pi.definition)
            unknown = flat.free_symbols - set(pi.quantities)
            if unknown:
                raise DimensionError(f"scale {sc} of {key} uses unknown symbols "
                                     f"{sorted(map(str, unknown))}")
            d = Dim()
            for s, k in zip(pi.q, _exponents(flat, pi.q)):
                d = d * pi.quantities[s] ** k
            if d != dim:
                raise DimensionError(f"scale {sc} of {key} has dimension {d}, the field {dim}")
            if isinstance(key, sp.Symbol):
                coord_scales[key] = flat
                continue
            head = key.func
            bg = spec.background if isinstance(spec, Affine) else None
            hat = _hat_head(head, spec.name if isinstance(spec, Affine) else None)
            fields[head] = _Field(head, hat, flat, dim, background=bg)
        state = NonDimState(fields, coord_scales, refs, params, self.groups,
                            self.rows, self.small, self.orders, prefer=self.prefer)
        state.model_parameters = set(pars)
        for name in self.rows:
            eq = m._equations[name]
            eq.expr = state.to_scaled_row(eq.expr, name)
        m.nondim = state
        state.log.append((self.name, f"scaled rows {sorted(self.rows)}; groups "
                          + ", ".join(f"{g} = {d}" for g, d in self.groups.items())))
        return m


class ShallowScaling(Operation):
    """Drop every term of a scaled row whose eps order exceeds the row's order.

    The order of a term is read off its group coefficient with the regime's group orders
    (`Regime`; default: the scaling's, e.g. `Fr, De = O(1)`).  Horizontal and vertical rows keep
    separate orders: `vertical=0` is the hydrostatic (SME-type) balance, `vertical=2` keeps the
    `eps^2` vertical inertia (VAM-type).  Without arguments (after `InsertClosure`) it repeats the
    orders of the first truncation, which drops the closure's own sub-leading terms.

    Args:
        horizontal: order of the horizontal momentum rows.
        vertical: order of the vertical momentum row.
        mass: order of the mass balance.
        regime: the `Regime`; fixed by the first truncation.
    """

    whole_model_op = True

    def __init__(self, horizontal=None, vertical=None, mass=0, regime=None,
                 name="shallow_scaling"):
        """Store the orders and the regime."""
        super().__init__(name=name, description="order-based shallow truncation")
        self.horizontal, self.vertical, self.mass = horizontal, vertical, mass
        self.regime = regime

    def __repr__(self):
        """`ShallowScaling(horizontal=.., vertical=.., regime=..)` (a derivation-cache key)."""
        return (f"ShallowScaling(horizontal={self.horizontal}, vertical={self.vertical}, "
                f"mass={self.mass}, regime={self.regime!r})")

    def __eq__(self, other):
        """Equal orders and regime."""
        return isinstance(other, ShallowScaling) and repr(self) == repr(other)

    def __hash__(self):
        """Hash of the repr."""
        return hash(repr(self))

    def orders(self):
        """`{Role: order}` of the given orders."""
        out = {}
        for role, k in ((Role.horizontal, self.horizontal), (Role.vertical, self.vertical),
                        (Role.mass, self.mass)):
            if k is not None:
                out[role] = sp.Rational(k)
        return out

    def apply_to_model(self, m):
        """Truncate the scaled rows; records the regime and orders on the first call."""
        st = getattr(m, "nondim", None)
        if st is None:
            raise ScalingError("ShallowScaling needs a scaled model (NonDimensionalize first)")
        if st.orders is None:
            st.regime = self.regime or Regime()
            st.orders = self.orders()
        else:
            if self.regime is not None and self.regime != st.regime:
                raise ScalingError(f"regime {self.regime} differs from the first truncation's "
                                   f"{st.regime}")
            given = self.orders()
            for role, k in given.items():
                if role in st.orders and st.orders[role] != k and role != Role.mass:
                    raise ScalingError(f"{role} order {k} differs from the first truncation's "
                                       f"{st.orders[role]}")
        dropped = {}
        for name, (role, _s) in st.rows.items():
            if role not in st.orders:
                continue
            keep, drop = [], []
            for t in sp.Add.make_args(m._equations[name].expr):
                (keep if st.order(t) <= st.orders[role] else drop).append(t)
            if drop:
                m._equations[name].expr = sp.Add(*keep)
                dropped[name] = drop
        st.last_dropped = dropped
        st.log.append((self.name, "drop beyond " + ", ".join(
            f"{r.value} O(eps^{k})" for r, k in st.orders.items()) + f" ({st.regime}): "
            + "; ".join(f"{n}: {len(v)} terms" for n, v in dropped.items())))
        return m


class ClosureRecord:
    """What `InsertClosure` decided, read by the model after `Dimensionalize`.

    Attributes:
        tensor: `{key: dimensional closure stress}` of the bulk closure, untruncated.
        shear: `{key: dimensional bulk shear}` of each horizontal row's own `tau_dz` that the
            row's order keeps; it stays a closure variable until the projection (its traces carry
            the bed / surface traction).
        inplane: a horizontal row keeps an in-plane / normal stress component.
        vertical_stress: the vertical row keeps a stress component.
        dropped: names of the closures whose every contribution is beyond the order.
        orders: `{(closure, component): eps order of its scaled stress}` (the consistency check).
        shear_only: name of a bulk closure that gives only the vertical shear (no `tensor`);
            it closes the horizontal rows' own `tau_dz` after the projection, as in the unclosed
            case, and is allowed only when no row keeps another stress component.
        unchecked: names of the bed / surface closures without scaling information
            (`dimensions`, `scaling_groups`): their traction enters after the projection as
            before, without the consistency check.
    """

    def __init__(self):
        """Empty record."""
        self.tensor, self.shear, self.orders = {}, {}, {}
        self.inplane = self.vertical_stress = False
        self.dropped, self.unchecked = [], []
        self.shear_only = None


class InsertClosure(Operation):
    """Insert the material law in the scaled frame and check it against the stress ansatz.

    The bulk closure's stress tensor (`tensor(m, uvel, w, horiz)`, physical coordinates) is
    written in the scaled variables, `tau_ij = scale_ij * tau_hat_ij`.  Its own scale, the scale
    of its vertical shear (`expression`), fixes the free stress scale (Newtonian:
    `S = rho nu U / H`, so `De = eps U H / nu`).  **Consistency check:** every term of every
    `tau_hat_ij` must be O(1) or smaller in eps (no negative order with the regime); a component
    larger than its ansatz raises `ScalingError` naming it.  The bed and surface closures
    (`traction`) go through the same check with their own groups (`scaling_groups`).

    The closure is substituted into every stress component of the scaled rows except the
    horizontal shear `tau_dz` of the horizontal rows, which stays a closure variable: the
    projection integrates `d_z tau_dz` by parts and its traces carry the bed / surface traction.
    For it, the bulk closure's terms that the row's order keeps are recorded (`ClosureRecord`).

    A bulk closure without a tensor (it gives only the vertical shear: Elder, Bingham, an eddy
    viscosity) is not inserted; it closes the horizontal rows' `tau_dz` after the projection, and
    the order must keep no other stress component (else `ScalingError`).  Bed / surface closures
    without scaling information (`dimensions`, `scaling_groups`, e.g. `RoughWall`) enter after the
    projection unchecked; the record and the log name them.

    Args:
        closures: the model's closure list.
    """

    whole_model_op = True

    def __init__(self, closures, name="insert_closure"):
        """Store the closures."""
        super().__init__(name=name, description="closure in the scaled frame")
        self.closures = list(closures or [])

    def apply_to_model(self, m):
        """Check, bind the stress scale, substitute; records `m.nondim.closure`."""
        st = getattr(m, "nondim", None)
        if st is None or st.orders is None:
            raise ScalingError("InsertClosure needs a scaled, truncated model "
                               "(NonDimensionalize, ShallowScaling first)")
        from zoomy_core.model.models.material import ClosureState, ProfileField
        from zoomy_core.model.models.equations import frame_slope
        rec = ClosureRecord()
        st.closure = rec
        info = st.blueprint
        uvel, w, horiz, z = info["uvel"], info["w"], info["horiz"], info["z"]
        stress = info["stress"]                            # {key: applied tau}
        names = {x_: n for x_, n in zip(horiz, ("u", "v"))}
        fields = {names[xd]: ProfileField(uvel[i], z) for i, xd in enumerate(horiz)}
        fields["w"] = ProfileField(w, z)

        def _state(at=None, tag=None):
            """Physical-coordinate closure state (vertical derivative `d_z`)."""
            return ClosureState(fields, params=m.parameters, h=sp.S.One if at is None
                                else info["depth"], x=horiz[0], zeta=z, at=None, alias=None,
                                boundary_tag=tag, horiz=list(horiz), base=info.get("base"))

        def _hat(key):
            return stress[key].func, st.fields[stress[key].func].hat(*stress[key].args)

        def _scaled_stress(expr, key, what):
            """`tau_hat` of a dimensional stress for component `key`, coefficients in groups."""
            out = sp.S.Zero
            sc = st.fields[stress[key].func].scale
            for t in sp.Add.make_args(_distribute(st.scaled(expr))):
                c, r = st.split(t)
                out += st.to_groups(c / sc, what=what) * r
            return out

        def _check(name, key, tau_hat):
            worst = min((st.order(t) for t in sp.Add.make_args(tau_hat) if t != 0),
                        default=sp.oo)
            rec.orders[(name, key)] = worst
            if worst < 0:
                raise ScalingError(
                    f"{name}: the closure's {stress[key].func.__name__} is "
                    f"O(eps^{worst}), larger than its ansatz scale "
                    f"{st.fields[stress[key].func].scale} (scaled: {tau_hat}); the closure is "
                    "inconsistent with the stress scaling")

        # ── bulk closure: bind the stress scale, scaled tensor, check ──
        for c in self.closures:
            c.register(m)
        st.model_parameters = {getattr(m.parameters, n) for n in m.parameters.keys()}
        bulk = [c for c in self.closures if getattr(c, "closes", None) == "bulk"]
        tensor_hat = {}
        if bulk and not hasattr(bulk[0], "tensor"):
            # the vertical shear alone (eddy viscosity, Bingham, ...): nothing to substitute
            # before the projection; checked against the kept stress components below
            rec.shear_only = type(bulk[0]).__name__
        elif bulk:
            c = bulk[0]
            c.register(m)
            T = c.tensor(m, uvel, w, list(horiz))
            shear = c.expression(_state())                   # rho nu d_z u: the closure's scale
            terms = sp.Add.make_args(_distribute(st.scaled(shear)))
            if len(terms) != 1:
                raise ScalingError(f"{type(c).__name__}: its vertical shear {shear} has no single "
                                   "scale")
            S_c, _ = st.split(terms[0])
            ref = info["stress_scale"]
            new = {s: PARAMETER_DIMENSIONS.get(str(s)) for s in S_c.free_symbols
                   if s not in st.consts}
            if any(d is None for d in new.values()) or len(new) != 1:
                raise ScalingError(f"{type(c).__name__}: stress scale {S_c} must add exactly one "
                                   f"parameter with a known dimension, got {sorted(map(str, new))}")
            st.bind(ref, S_c, new)
            for key, expr in T.items():
                if key not in stress:
                    continue
                th = _scaled_stress(expr, key, f"{type(c).__name__} {key}")
                _check(type(c).__name__, key, th)
                tensor_hat[key] = th
                rec.tensor[key] = expr
        # ── bed / surface closures: their groups, scaled traction, check ──
        traction_hat = []
        for c in self.closures:
            if getattr(c, "closes", None) not in ("bottom", "surface"):
                continue
            if not (hasattr(c, "dimensions") or hasattr(c, "scaling_groups")):
                rec.unchecked.append(type(c).__name__)
                continue
            c.register(m)
            dims = getattr(c, "dimensions", lambda: {})()
            new = {s: dims.get(str(s), PARAMETER_DIMENSIONS.get(str(s)))
                   for s in (getattr(m.parameters, n) for n in m.parameters.keys())
                   if str(s) in dims}
            groups, orders = getattr(c, "scaling_groups", lambda _st: ({}, {}))(st)
            st.extend(new, groups, orders)
            tag = "b" if c.closes == "bottom" else "eta"
            tang = c.traction(_state(at=0, tag=tag)).get("tangent") or []
            flat = {frame_slope(tag, n): 0 for n in ("x", "y")}   # small-slope frame
            for i, xd in enumerate(horiz):
                if i >= len(tang):
                    continue
                key = {horiz[0]: "x"}.get(xd, "y") + "z"
                P = sp.sympify(tang[i]).xreplace(flat)
                th = _scaled_stress(P, key, f"{type(c).__name__} traction {key}")
                _check(type(c).__name__, key, th)
                traction_hat.append((c, key, th))
        # ── substitute; record the deferred shear per horizontal row ──
        hats = {_hat(k)[1]: k for k in stress}
        deferred = {k for k in stress if len(k) == 2 and k[1] == "z" and k[0] != "z"}
        kept_by = {}
        for name, (role, _s) in st.rows.items():
            eq = m._equations[name]
            expr = eq.expr
            if role == Role.horizontal:
                own = [hats[h] for h in expr.atoms(AppliedUndef) if h in hats
                       and hats[h] in deferred]
                for key in own:
                    row_terms = [t for t in sp.Add.make_args(expr)
                                 if t.has(_hat(key)[1])]
                    k_row = min(st.order(t) for t in row_terms)
                    lim = st.orders[role]
                    if key in tensor_hat:
                        kept = sum((t for t in sp.Add.make_args(tensor_hat[key])
                                    if k_row + st.order(t) <= lim), sp.S.Zero)
                        rec.shear[key] = self._dimensional(st, kept, stress[key])
                    for c, k2, th in traction_hat:
                        if k2 == key:
                            ok = any(k_row + st.order(t) <= lim for t in sp.Add.make_args(th))
                            kept_by.setdefault(c, False)
                            kept_by[c] = kept_by[c] or ok
                if any(h in hats and hats[h] not in deferred for h in expr.atoms(AppliedUndef)):
                    rec.inplane = True
            elif role == Role.vertical:
                if any(h in hats for h in expr.atoms(AppliedUndef)):
                    rec.vertical_stress = True
            sub = {h: tensor_hat[k] for h, k in hats.items() if k in tensor_hat
                   and not (role == Role.horizontal and k in deferred)}
            if not sub or not expr.has(*sub):
                continue
            # the derivatives of the substituted closure are evaluated (canonical forms)
            targets = {D: D.xreplace(sub).doit() for D in expr.atoms(sp.Derivative)
                       if D.expr.has(*sub)}
            new_expr = expr.xreplace(targets).xreplace(sub)
            eq.expr = st.normalize(sp.expand(new_expr), name)
        rec.dropped = [type(c).__name__ for c, ok in kept_by.items() if not ok]
        if rec.shear_only and (rec.inplane or rec.vertical_stress):
            raise ScalingError(
                f"{rec.shear_only} gives only the vertical shear (no stress tensor in physical "
                "coordinates), but the order keeps "
                + ("the in-plane stress" if rec.inplane else "a stress in the vertical row")
                + f" ({st.regime}); it fits a regime that keeps only the horizontal rows' "
                "vertical shear (e.g. VAMWeakViscosity, SME)")
        st.log.append((self.name, "closure " + ", ".join(type(c).__name__ for c in self.closures)
                       + " in the scaled frame; component orders "
                       + ", ".join(f"{a}.{b}: {o}" for (a, b), o in rec.orders.items())
                       + (f"; vertical shear only: {rec.shear_only}" if rec.shear_only else "")
                       + (f"; unchecked (no scaling groups): {', '.join(rec.unchecked)}"
                          if rec.unchecked else "")))
        return m

    @staticmethod
    def _dimensional(st, tau_hat, field):
        """Dimensional value of a scaled stress of `field`."""
        sc = st.fields[field.func].scale
        out = sp.S.Zero
        for t in sp.Add.make_args(tau_hat):
            if t == 0:
                continue
            c, r = st.split(t)
            out += _distribute(st.from_groups(c) * sc * st.unscaled(r))
        return out


class Dimensionalize(Operation):
    """Back from the scaled frame to the physical variables (inverse of `NonDimensionalize`)."""

    whole_model_op = True

    def __init__(self, name="dimensionalize"):
        """Name the operation."""
        super().__init__(name=name, description="back to physical variables")

    def apply_to_model(self, m):
        """Rewrite the scaled rows dimensionally and leave the scaled frame."""
        st = getattr(m, "nondim", None)
        if st is None:
            raise ScalingError("Dimensionalize needs a scaled model")
        for name in st.rows:
            eq = m._equations[name]
            eq.expr = st.to_dimensional_row(eq.expr, name)
            left = {f.func for f in eq.expr.atoms(AppliedUndef)} & set(st.hat_of)
            if left:
                raise ScalingError(f"{name}: scaled fields left after Dimensionalize: {left}")
        m._ever_seen_heads -= set(st.hat_of)
        m.nondim = None
        m.scaling = st                        # the record (regime, orders, closure) stays
        m._refresh_unknowns()
        st.log.append((self.name, "dimensional rows"))
        return m


__all__ = ["Dim", "ONE", "LENGTH", "TIME", "MASS", "VELOCITY", "ACCELERATION", "DENSITY",
           "STRESS", "KINEMATIC_VISCOSITY", "PARAMETER_DIMENSIONS", "DimensionError",
           "ScalingError", "PiBasis", "Role", "Affine", "Regime", "NonDimState",
           "NonDimensionalize", "ShallowScaling", "Dimensionalize"]
