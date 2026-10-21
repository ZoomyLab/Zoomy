"""Milestone-1 tests for the clean-redesign derivation core.

Covers the model/unknown spine:

(a) build the 3 SME balances, set `model.Q=[h,u,w,p]`, assert `Qaux`
    auto-contains `b`/`tau` and excludes coords/params;
(b) hydrostatic elimination drops `p` from Q;
(c) vector access `model.momentum.x` / `.z`;
(d) term selection `eq.term[i]` / `eq.term[[i, j]].apply(...)` restricts to
    additive terms; bare `eq[i]` is no longer term-indexing (`[l]` is
    reserved for moment rows);
(e) `describe()` renders.
"""

import sympy as sp
import pytest

from zoomy_core import coords
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import Model, ChangeOfVariables
from zoomy_core.model.operations import Integrate
from zoomy_core.misc.description import Description


t, x, z = coords.t, coords.x, coords.z


def _build_sme():
    """The SME starting point: mass + vector momentum, Q = [h, u, w, p]."""
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    g = model.parameters.g
    rho = model.parameters.rho

    u = sp.Function("u", real=True)(t, x, z)
    w = sp.Function("w", real=True)(t, x, z)
    p = sp.Function("p", real=True)(t, x, z)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    tau = sp.Function("tau", real=True)(t, x, z)

    model.Q = [h, u, w, p]
    model.add_equation("mass", d.x(u) + d.z(w))
    model.add_equation(
        "momentum",
        (2,),
        [
            d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho - d.z(tau) / rho,
            d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rho + g,
        ],
    )
    return model, dict(u=u, w=w, p=p, h=h, b=b, tau=tau, g=g, rho=rho)


# ── (a) Q declared-and-present, Qaux derived ─────────────────────────────


def test_Q_setter_autonames_from_field_heads():
    model, s = _build_sme()
    assert model.Q.keys() == ["h", "u", "w", "p"]
    # GETTER returns the applied field Functions.
    assert model.Q.h == s["h"]
    assert model.Q.u == s["u"]


def test_Qaux_auto_contains_aux_fields_excludes_coords_and_params():
    model, s = _build_sme()
    qaux_keys = set(model.Qaux.keys())
    # b and tau appear in the equations but are not declared in Q.
    assert "b" not in qaux_keys  # b not yet in any equation (only h is)
    assert "tau" in qaux_keys
    # Coordinates and parameters never count as unknowns.
    assert "t" not in qaux_keys and "x" not in qaux_keys and "z" not in qaux_keys
    assert "g" not in qaux_keys and "rho" not in qaux_keys
    # Declared Q fields never leak into Qaux.
    for k in model.Q.keys():
        assert k not in qaux_keys


def test_Qaux_picks_up_b_once_it_enters_an_equation():
    model, s = _build_sme()
    b, h, g = s["b"], s["h"], s["g"]
    # Inject a bed-slope term into the mass balance.
    model.mass.apply({d.x(s["u"]): d.x(s["u"]) + g * d.x(b)})
    assert "b" in set(model.Qaux.keys())


# ── (b) hydrostatic elimination drops p from Q ───────────────────────────


def test_hydrostatic_elimination_removes_p_from_Q():
    model, s = _build_sme()
    u, w, p, h, b, g, rho = (s[k] for k in ("u", "w", "p", "h", "b", "g", "rho"))

    assert "p" in model.Q.keys()

    # Reduce the z-momentum to the hydrostatic balance, INTEGRATE over the
    # vertical (FTC: ∂_z p → p(η)−p(z)), impose the free-surface BC
    # p|_{z=b+h}=0, then solve_for the resulting ALGEBRAIC relation.
    model.momentum.z.apply(
        {d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0}
    )
    model.momentum.z.apply(Integrate(z, z, b + h, method="analytical"))
    model.momentum.z.apply({p.subs(z, b + h): 0})
    p_hydro = model.momentum.z.solve_for(p)

    # p_hydro maps the field p to its hydrostatic profile.
    assert p in p_hydro.subs_map
    sol = p_hydro.subs_map[p]
    # p = rho g (b + h - z)
    assert sp.simplify(sol - rho * g * (b + h - z)) == 0

    # Substitute into x-momentum, then remove the z-momentum row.
    model.momentum.x.apply(p_hydro)
    model.momentum.z.remove()

    # p no longer appears in any equation -> auto-dropped from Q.
    assert "p" not in model.Q.keys()
    assert "p" not in set(model.Qaux.keys())


# ── (c) vector component access ──────────────────────────────────────────


def test_vector_equation_component_access():
    model, s = _build_sme()
    # Attribute access.
    assert model.momentum.x.expr == model.momentum[0].expr
    assert model.momentum.z.expr == model.momentum[1].expr
    # The z-momentum carries the gravity source.
    assert model.momentum.z.expr.has(s["g"])
    assert not model.momentum.x.expr.has(s["g"])


def test_vector_component_apply_is_isolated():
    model, s = _build_sme()
    before_x = model.momentum.x.expr
    model.momentum.z.apply({s["g"]: 0})
    # Touching z must not change x.
    assert model.momentum.x.expr == before_x
    assert not model.momentum.z.expr.has(s["g"])


# ── (d) term selection restricts the op to those terms ───────────────────


def test_term_selection_restricts_apply():
    model, s = _build_sme()
    u, w = s["u"], s["w"]
    # mass = ∂_x u + ∂_z w  (two terms).  Zero out only term 0 via ``.term``.
    mass = model.mass
    assert len(mass) == 2
    mass.term[0].apply({d.x(u): 0})
    assert sp.expand(model.mass.expr) == sp.expand(d.z(w))


def test_term_group_selection_product_rule_term_only():
    from zoomy_core.model.operations import ProductRule

    model, s = _build_sme()
    u = s["u"]
    # Term-scoped apply: pick the ∂_x(u*u) flux term and product-rule ONLY it.
    terms = list(sp.Add.make_args(sp.expand(model.momentum.x.expr)))
    idx = next(i for i, term in enumerate(terms)
               if isinstance(term, sp.Derivative)
               and term.args[0] == u * u)
    other_terms = [t for j, t in enumerate(terms) if j != idx]
    model.momentum.x.term[idx].apply(ProductRule(direction="forward"))
    new_terms = set(sp.Add.make_args(sp.expand(model.momentum.x.expr)))
    # ∂_x(u^2) -> 2 u ∂_x u ; the other terms are untouched.
    assert d.x(u * u) not in new_terms
    assert (2 * u * d.x(u)) in new_terms
    for ot in other_terms:
        assert ot in new_terms


def test_single_term_only_op_raises_on_multiterm():
    """A genuinely single-term-only op (granularity TERM) must raise on a
    multi-term equation — the framework forbids the op deciding which terms
    to rewrite."""
    from zoomy_core.model.operations import Operation
    from zoomy_core.model.derivation.operations import granularity_of, Granularity

    class _TermOnly(Operation):
        single_term_only = True

        def _leaf_sp(self, e):
            return e

    op = _TermOnly(name="term_only")
    assert granularity_of(op) == Granularity.TERM
    model, s = _build_sme()
    with pytest.raises(RuntimeError):
        model.mass.apply(op)
    # Term-scoped is allowed via the ``.term`` accessor.
    model.mass.term[0].apply(op)


# ── (d') `.term` accessor disambiguation: `[l]` ≠ term ───────────────────


def test_term_accessor_single_and_group():
    """`eq.term[i]` is one additive term; `eq.term[[i, j]]` /
    `eq.term[i, j]` is a term group — both write rewrites back."""
    model, s = _build_sme()
    u, w = s["u"], s["w"]
    # mass = ∂_x u + ∂_z w.
    assert model.mass.term[0].expr == sp.expand(d.x(u))
    assert model.mass.term[1].expr == sp.expand(d.z(w))
    # Group view reads the Add of its terms.
    grp = model.mass.term[[0, 1]]
    assert sp.expand(grp.expr) == sp.expand(d.x(u) + d.z(w))
    # Group apply rewrites only those terms (here: zero both) back in place.
    grp.apply({d.x(u): 0, d.z(w): 0})
    assert sp.expand(model.mass.expr) == 0


def test_bare_equation_indexing_no_longer_returns_a_term():
    """A scalar `Equation` is no longer term-indexable via bare `eq[i]` —
    `[l]` is reserved for moment rows.  `eq[0]` must raise, NOT return a
    term."""
    model, s = _build_sme()
    with pytest.raises(TypeError):
        _ = model.mass[0]
    with pytest.raises(TypeError):
        model.mass[0].apply({d.x(s["u"]): 0})


# ── (e) describe renders ─────────────────────────────────────────────────


def test_describe_returns_description():
    model, s = _build_sme()
    desc = model.describe()
    assert isinstance(desc, Description)
    md = desc.to_markdown()
    assert "mass" in md and "momentum" in md
    # Header reflects model name + #eqs + #ops.
    assert "equation" in md.lower()


def test_equation_describe_renders():
    model, s = _build_sme()
    desc = model.mass.describe()
    assert isinstance(desc, Description)
    assert "mass" in desc.to_markdown()


# ── ChangeOfVariables vs a bare substitution (Q-swap semantics) ────────────


def test_change_of_variables_swaps_unknown_family():
    """`ChangeOfVariables` swaps the unknown family in Q; a bare
    A bare substitution must NOT touch Q."""
    model = Model(coords=(t, x), parameters={})
    a0 = sp.Function("a")(0, t, x)
    a1 = sp.Function("a")(1, t, x)
    h = sp.Function("h", positive=True)(t, x)

    model.Q = [h, a0, a1]
    model.add_equation("c0", d.t(a0) + d.x(a0))
    model.add_equation("c1", d.t(a1) + d.x(a1 * a0))

    a = sp.Function("a")
    q = sp.Function("q")
    # a_i -> q_i / h
    model.apply(ChangeOfVariables("a", "q", lambda q_i: q_i / h))

    keys = set(model.Q.keys())
    assert "a" not in keys
    assert "q" in keys
    assert "h" in keys
    # equations now contain q, not a.
    assert not model.c0.expr.has(a)
    assert model.c0.expr.has(q)


def test_bare_substitution_does_not_swap_Q():
    model = Model(coords=(t, x), parameters={})
    a0 = sp.Function("a")(0, t, x)
    h = sp.Function("h", positive=True)(t, x)
    model.Q = [h, a0]
    model.add_equation("c0", d.t(a0))
    # A bare value substitution does not redeclare unknowns.
    model.apply({d.t(a0): d.x(a0)})
    assert set(model.Q.keys()) == {"h", "a"}
