"""Milestone-3 tests: modal separation of variables + Galerkin projection.

Drives the SME mass row through the full pipeline:

  PDE-transform → separation_of_variables → ExpandSums → Project →
  ExtractBrackets → ResolveBasis

and asserts the distinct-index registry, unexpanded-Sum ansatz, Q/Qaux
bookkeeping (nothing orphaned after SoV), cross-term-preserving ExpandSums,
the opaque ⟨…⟩ bracket form after Project/ExtractBrackets (abstract Sum
intact, no premature .doit()), and the δ-closed-form Gram resolution.
"""

import sympy as sp
import pytest

from zoomy_core import coords
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model, PDETransformation, Basis,
    separation_of_variables, reset_modal_indices, modal_bound,
    ExpandSums, Project, PullConstants, ExtractBrackets, ResolveBasis,
    bracket_atoms, Gram,
)
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted


t, x, z, zeta = coords.t, coords.x, coords.z, coords.zeta


def _sme_after_pde():
    """SME (mass + x/z-momentum), Q=[h,u,w,p,τ aux], PDE-transformed to ζ."""
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    g, rho = model.parameters.g, model.parameters.rho

    u = sp.Function("u", real=True)(t, x, z)
    w = sp.Function("w", real=True)(t, x, z)
    p = sp.Function("p", real=True)(t, x, z)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    tau = sp.Function("tau", real=True)(t, x, z)

    model.Q = [h, u, w]
    model.add_equation("mass", d.x(u) + d.z(w))
    model.add_equation(
        "momentum",
        (2,),
        [
            d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho - d.z(tau) / rho,
            d.t(w) + d.z(p) / rho + g,
        ],
    )
    model.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
    return model, dict(u=u, w=w, p=p, h=h, b=b, tau=tau)


def _expand_sme():
    """The SME after SoV on u (→ a, index i) and w (→ aw, index j), N_w=N_u+1."""
    model, s = _sme_after_pde()
    basis = Basis(symbol="phi", weight="c")
    a = sp.Function("a")
    aw = sp.Function("aw")
    N_u = modal_bound("N_u")
    reset_modal_indices(model)
    model.apply(separation_of_variables(s["u"], a(t, x), basis, N_u))
    model.apply(separation_of_variables(s["w"], aw(t, x), basis, N_u + 1))
    s.update(basis=basis, a=a, aw=aw, N_u=N_u)
    return model, s


# ── (1) distinct indices, unexpanded Sum, Q/Qaux bookkeeping ──────────────


def test_separation_of_variables_distinct_indices():
    model, s = _expand_sme()
    i = model.modal_index(s["u"])
    j = model.modal_index(s["w"])
    assert i != j
    assert i.name == "i" and j.name == "j"
    # modal_index resolves by field OR by coefficient.
    assert model.modal_index(s["a"]) == i
    assert model.modal_index(s["aw"]) == j
    assert model.modal_basis(s["u"]) is s["basis"]


def test_ansatz_is_unexpanded_sympy_sum():
    model, s = _expand_sme()
    sums = model.mass.expr.atoms(sp.Sum)
    assert sums, "modal ansatz must carry an unexpanded sp.Sum"
    # The ansatz is a genuine sp.Sum (not a Python sum collapsed to an Add).
    u_sum = next(su for su in sums
                 if su.limits[0][0] == model.modal_index(s["u"]))
    assert u_sum.limits[0] == (model.modal_index(s["u"]), 0, s["N_u"])


def test_Q_holds_coeff_families_with_distinct_indices():
    model, s = _expand_sme()
    # After PDE + SoV, Q is the modal coefficients (+ h), NOT u/w heads.
    assert set(model.Q.keys()) == {"h", "a", "aw"}
    i = model.modal_index(s["u"])
    j = model.modal_index(s["w"])
    assert model.Q.a == s["a"](i, t, x)
    assert model.Q.aw == s["aw"](j, t, x)


def test_Qaux_has_b_and_decorated_tau_nothing_orphaned():
    model, s = _expand_sme()
    qaux = set(model.Qaux.keys())
    # b (bed) is auxiliary; the decorated stress τ̃ survives the σ-map.
    assert "b" in qaux
    assert any("tau" in k for k in qaux), f"decorated tau missing from {qaux}"
    # The opaque basis φ / weight c are NOT unknowns.
    assert "phi" not in qaux and "c" not in qaux
    # Nothing in "neither": every field atom is in Q or Qaux.
    q_heads = {Model._head(f) for f in model._Q.values()}
    qaux_names = qaux
    for f in model._collect_fields():
        nm = Model._field_name(f)
        assert Model._head(f) in q_heads or nm in qaux_names, (
            f"field {f} is in neither Q nor Qaux")


# ── (2) ExpandSums keeps cross terms ──────────────────────────────────────


def test_expand_sums_square_gives_double_sum_distinct_dummies():
    from zoomy_core.model.operations import Expression
    basis = Basis(symbol="phi", weight="c")
    a = sp.Function("a")
    i = sp.Symbol("i", integer=True, nonnegative=True)
    N = sp.Symbol("N", integer=True, nonnegative=True)
    u_sum = sp.Sum(a(i, t, x) * basis.phi(i, zeta), (i, 0, N))
    out = ExpandSums()._leaf_sp(u_sum**2)
    # (Σ_i …)² → a genuine double Sum.
    assert isinstance(out, sp.Sum)
    dummies = {lim[0] for lim in out.limits}
    assert len(dummies) == 2, f"expected two distinct dummies, got {dummies}"
    # Cross terms a_i a_j present (two different coeff applications).
    coeff_indices = {at.args[0] for at in out.function.atoms(sp.Function)
                     if at.func is a}
    assert coeff_indices == dummies


# ── (3) Project + ExtractBrackets: opaque bracket, abstract Sum intact ─────


def test_project_then_extract_brackets_mass_row():
    model, s = _expand_sme()
    basis, a, N_u = s["basis"], s["a"], s["N_u"]
    l = sp.Symbol("l", integer=True, nonnegative=True)
    c = basis.weight

    model.mass.apply(Project(c(zeta) * basis.phi(l, zeta), var=zeta))
    # After Project the unexpanded ansatz Sum survives (no premature .doit()).
    assert model.mass.expr.atoms(sp.Sum)

    # PullConstants pushes the Σ out and hoists the ζ-independent a_i, leaving a
    # pure-ζ ∫ for ExtractBrackets to NAME (the extractor no longer pushes).
    model.mass.apply(PullConstants())
    model.mass.apply(ExtractBrackets(basis, var=zeta))
    expr = model.mass.expr

    # The orthogonal mass term is Σ_i a_i ⟨φ_i, c φ_l⟩ = Σ_i a_i Gram(i, l).
    brackets = bracket_atoms(expr)
    i = model.modal_index(s["u"])
    assert Gram(i, l) in brackets, f"Gram(i, l) missing from {brackets}"

    # The abstract Sum is still intact (carried through projection).
    assert expr.atoms(sp.Sum)

    # Named brackets are derivation machinery — never leak into Q/Qaux.
    assert "Gram" not in set(model.Qaux.keys())
    assert set(model.Q.keys()) == {"h", "a", "aw"}

    # The ζ-dependent metric-coupling bodies stay as opaque ⟨…⟩ Integrals;
    # the strip-args render shows the bracket token.
    md = model.mass.describe(strip_args=True).to_markdown()
    assert r"\langle" in md


# ── (4) ResolveBasis closes the Gram bracket via δ / (2l+1) ────────────────


def test_resolve_basis_closes_gram_to_delta_form():
    model, s = _expand_sme()
    basis, a, N_u = s["basis"], s["a"], s["N_u"]
    l = sp.Symbol("l", integer=True, nonnegative=True)
    c = basis.weight

    model.mass.apply(Project(c(zeta) * basis.phi(l, zeta), var=zeta))
    model.mass.apply(PullConstants())
    model.mass.apply(ExtractBrackets(basis, var=zeta))
    assert any(at.func.__name__ == "Gram" for at in bracket_atoms(model.mass.expr))

    model.mass.apply(ResolveBasis(Legendre_shifted(level=0), var=zeta))
    expr = model.mass.expr

    # Gram is gone; the orthogonal term collapsed to a(l,…)/(2l+1) (Piecewise
    # gated N_u ≥ l, with the surviving ∂_x outside).
    assert not any(at.func.__name__ == "Gram"
                   for at in bracket_atoms(expr))
    assert expr.has(sp.Piecewise)
    # The collapsed term carries the 1/(2l+1) Legendre normalisation.
    assert expr.has(1 / (2 * l + 1))
    # u's coefficient now appears at the test index l (a(l, …)).
    assert any(at.func is a and at.args[0] == l
               for at in expr.atoms(sp.Function))


def test_resolve_basis_symbolic_l_then_bind():
    """The δ-closed-form keeps l symbolic; binding l=1 → a(1,…)/3."""
    model, s = _expand_sme()
    basis, a, N_u = s["basis"], s["a"], s["N_u"]
    l = sp.Symbol("l", integer=True, nonnegative=True)
    c = basis.weight

    model.mass.apply(Project(c(zeta) * basis.phi(l, zeta), var=zeta))
    model.mass.apply(PullConstants())
    model.mass.apply(ExtractBrackets(basis, var=zeta))
    model.mass.apply(ResolveBasis(Legendre_shifted(level=0), var=zeta))

    # Bind the test mode l = 1 (N_u ≥ 1 branch) → a(1, t, x)/3 derivative.
    bound = model.mass.expr.subs(l, 1).doit()
    expected = sp.Derivative(a(1, t, x), x) / 3
    # The orthogonal (closed) piece equals ∂_x a(1)/3 for N_u ≥ 1.
    piece = bound.subs(N_u, 1)
    assert piece.has(expected) or sp.simplify(
        piece.coeff(sp.Derivative(a(1, t, x), x)) - sp.Rational(1, 3)) == 0
