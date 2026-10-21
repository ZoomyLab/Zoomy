"""Milestone-2 tests: the transformation + boundary-condition layer.

Covers three building blocks:

A. `PDETransformation` — the explicit-geometry σ-map `z = b + h·ζ`
   that mints decorated heads (`ũ`) of `(t, x, ζ)` and applies the σ
   chain rule with NO leftover `Subs`.
B. `KinematicBC` — added to the model as an
   oriented relation BEFORE the σ-map, σ-mapped in place by
   `PDETransformation` (`w(b) → w̃(0)` …), then applied as a
   substitution.
C. the conservative fold — `Multiply(h)` + `ProductRule()` on a momentum
   row reaches divergence form (a `∂_x(h·…)` appears).
"""

import sympy as sp

from zoomy_core import coords
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import Model, PDETransformation
from zoomy_core.model.operations import KinematicBC


t, x, z, zeta = coords.t, coords.x, coords.z, coords.zeta


def _build():
    """A small model: scalar mass + a vector momentum (only x is loaded)."""
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    rho = model.parameters.rho

    u = sp.Function("u", real=True)(t, x, z)
    w = sp.Function("w", real=True)(t, x, z)
    p = sp.Function("p", real=True)(t, x, z)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)

    model.Q = [h, u, w, p]
    model.add_equation("mass", d.x(u) + d.z(w))
    model.add_equation(
        "momentum",
        (2,),
        [
            d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho,
            d.t(w),
        ],
    )
    return model, dict(u=u, w=w, p=p, h=h, b=b, rho=rho)


def _has_no_subs(expr):
    return not expr.atoms(sp.Subs)


# ── A. PDETransformation ─────────────────────────────────────────────────


def test_pde_decorates_heads_to_zeta_functions():
    model, s = _build()
    u, w = s["u"], s["w"]
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + s["h"] * zeta))})
    model.apply(pde)

    # The plain u/w heads no longer appear in the mass row…
    mass = model.mass.expr
    assert not mass.has(u.func)
    assert not mass.has(w.func)
    # …and the decorated heads are applied to (t, x, ζ).
    decorated_atoms = [a for a in mass.atoms(sp.Function)
                       if getattr(a.func, "_pde_decorated_from", None)]
    assert decorated_atoms
    for a in decorated_atoms:
        assert a.args[-1] == zeta
        assert zeta in a.free_symbols


def test_pde_leaves_no_subs():
    model, s = _build()
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + s["h"] * zeta))})
    model.apply(pde)
    for eq in model._equations.values():
        assert _has_no_subs(eq.expr), f"Subs leaked in {eq.name}: {eq.expr}"


def test_pde_z_derivative_becomes_zeta_over_h():
    """A ∂_z w term must become (1/h)·∂_ζ w̃."""
    model, s = _build()
    h = s["h"]
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + h * zeta))})
    model.apply(pde)

    w_tilde = pde.decorated(s["w"])
    mass = sp.expand(model.mass.expr)
    # mass was ∂_x u + ∂_z w → … + (1/h)·∂_ζ w̃
    assert mass.has(sp.Derivative(w_tilde, zeta))
    # The ∂_ζ w̃ term carries a 1/h factor.
    target = sp.Derivative(w_tilde, zeta) / h
    assert sp.simplify(mass - (_x_part(mass, zeta) + target)) == 0 or \
        any(sp.simplify(term - target) == 0
            for term in sp.Add.make_args(mass))


def _x_part(expr, zeta_sym):
    # helper: sum of the terms that do NOT contain a ζ-derivative
    return sum((tm for tm in sp.Add.make_args(expr)
                if not tm.has(sp.Derivative)
                or not any(zeta_sym in dd.variables
                           for dd in tm.atoms(sp.Derivative))),
               sp.S.Zero)


def test_pde_decorated_accessor():
    model, s = _build()
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + s["h"] * zeta))})
    model.apply(pde)
    u_tilde = pde.decorated(s["u"])
    assert isinstance(u_tilde, sp.Function)
    assert u_tilde.func._pde_decorated_from == "u"
    assert u_tilde.args == (t, x, zeta)


def test_pde_momentum_x_matches_expected_shape():
    """The transformed momentum.x must equal the chain-rule form
    (memory.md ~6196): ∂_t ũ + ∂_x ũ² + (∂_ζ(ũ w̃))/h − jacobian terms +
    (1/ρ)·(∂_x p̃-correction)."""
    model, s = _build()
    h, b, rho = s["h"], s["b"], s["rho"]
    pde = PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))})
    model.apply(pde)

    u_t = pde.decorated(s["u"])
    w_t = pde.decorated(s["w"])
    p_t = pde.decorated(s["p"])

    jac_t = sp.Derivative(zeta * h + b, t)
    jac_x = sp.Derivative(zeta * h + b, x)

    expected = (
        sp.Derivative(u_t, t) - (jac_t / h) * sp.Derivative(u_t, zeta)
        + sp.Derivative(u_t**2, x) - (jac_x / h) * sp.Derivative(u_t**2, zeta)
        + sp.Derivative(u_t * w_t, zeta) / h
        + (sp.Derivative(p_t, x) - (jac_x / h) * sp.Derivative(p_t, zeta)) / rho
    )
    got = model.momentum.x.expr
    assert sp.simplify(sp.expand(got) - sp.expand(expected)) == 0, (
        f"\n got:      {sp.expand(got)}\n expected: {sp.expand(expected)}")


def test_pde_is_model_level_only():
    """PDETransformation refuses a bare equation-level apply."""
    model, s = _build()
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + s["h"] * zeta))})
    import pytest
    with pytest.raises(TypeError):
        model.mass.apply(pde)


def test_pde_stores_decoration_map_on_model():
    model, s = _build()
    pde = PDETransformation({z: (zeta, sp.Eq(z, s["b"] + s["h"] * zeta))})
    model.apply(pde)
    deco = model._field_decoration
    assert s["u"].func in deco
    assert model._sigma_from == z
    assert model._vertical == zeta


# ── B. KinematicBC (σ-mapped in place, then applied) ──────────────────────


def test_kinematic_bc_substitutes_bed_relation():
    """A bed KBC added before the σ-map is σ-mapped in place; applying it
    replaces w̃|_{ζ=0} with ∂_t b + ũ|_{ζ=0}·∂_x b."""
    model, s = _build()
    u, w, b = s["u"], s["w"], s["b"]
    model.add_equation("kbc_bot", KinematicBC(w=w, u=u, interface=b))
    pde = PDETransformation({z: (zeta, sp.Eq(z, b + s["h"] * zeta))})
    model.apply(pde)

    # Put a w̃|_{ζ=0} term into the mass row so we can watch it transform.
    w_tilde = pde.decorated(w)
    u_tilde = pde.decorated(u)
    w_bed = w_tilde.subs(zeta, 0)
    model.mass.apply({sp.Derivative(w_tilde, zeta) / s["h"]: w_bed})

    model.apply(model.kbc_bot)            # apply the σ-mapped bed KBC

    mass = sp.expand(model.mass.expr)
    # w̃|_{ζ=0} is gone, replaced by the bed relation.
    assert not mass.has(w_bed)
    # The substituted relation's terms are present.
    assert mass.has(sp.Derivative(b, t))
    assert mass.has(u_tilde.subs(zeta, 0))
    # And the rest of the mass row (∂_x ũ) is untouched.
    assert mass.has(sp.Derivative(u_tilde, x))


def test_kinematic_bc_surface_interface():
    """`PDETransformation` σ-maps a stored surface KBC's `{lhs: rhs}` rule:
    `w(b+h) → w̃(1)`, `u(b+h) → ũ(1)`."""
    model, s = _build()
    u, w, b, h = s["u"], s["w"], s["b"], s["h"]
    model.add_equation("kbc_top", KinematicBC(w=w, u=u, interface=b + h))
    pde = PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))})
    model.apply(pde)
    w_tilde = pde.decorated(w)
    u_tilde = pde.decorated(u)
    (lhs, rhs), = model.kbc_top.subs_map.items()
    assert lhs == w_tilde.subs(zeta, 1)
    expected_rhs = (sp.Derivative(b + h, t)
                    + u_tilde.subs(zeta, 1) * sp.Derivative(b + h, x))
    assert sp.simplify(rhs - expected_rhs) == 0


# ── C. conservative fold ─────────────────────────────────────────────────


def test_conservative_fold_multiply_h_then_product_rule():
    """Multiply(h) then ProductRule() on momentum.x yields a ∂_x(h·…)
    conservative term."""
    from zoomy_core.model.operations import Multiply, ProductRule

    model, s = _build()
    h = s["h"]
    # Work on the pre-PDE x-momentum (physical z).  Multiply by h, then
    # product-rule to fold a flux into divergence form.
    model.momentum.x.apply(Multiply(h))
    model.momentum.x.apply(ProductRule())

    mx = sp.expand(model.momentum.x.expr)
    # A conservative ∂_x(h · u²) atom appears.
    u = s["u"]
    has_conservative_x = any(
        isinstance(term, sp.Derivative)
        and term.variables == (x,)
        and term.args[0].has(h)
        for term in mx.atoms(sp.Derivative)
    )
    assert has_conservative_x, f"no ∂_x(h·…) in {mx}"
