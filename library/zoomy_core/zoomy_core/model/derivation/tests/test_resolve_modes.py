"""Milestone-6 tests: `resolve_modes` — the moment-row SHAPE BUMP.

`resolve_modes` takes a row carrying an ABSTRACT test index `l` (after a
Galerkin `Project(c·φ(l,ζ)) + ExtractBrackets` style projection) and expands
it, via a per-mode index specialisation `{l: k}`,
into one row per `l`, GROUPED UNDER THE PARENT as an indexed moment family.

Shape algebra:

  * scalar `mass` `(1,)` → `(N+1,)`   → `model.mass[l]` (and `.eq0`…)
  * vector `momentum` `(2,)` → `(2, N+1)` → `model.momentum.x[l]`
    (the `(x, l)` slice of the bumped tensor).

Per-row `ResolveBasis` / `Resolve` then closes the Galerkin brackets per
moment.  These tests pin the SHAPE BUMP + the family access; the bit-exact SME
acceptance lives in `test_sme_kt19` / `test_slip_sme`.
"""

import sympy as sp
import pytest

from zoomy_core import coords
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model, PDETransformation, Basis,
    separation_of_variables, reset_modal_indices, modal_bound,
    resolve_modes,
)
from zoomy_core.model.operations import Legendre_shifted


t, x, z = coords.t, coords.x, coords.z
zeta = sp.Symbol("zeta", real=True)
l = sp.Symbol("l", integer=True, nonnegative=True)


def _modal_model():
    """A minimal model: a scalar `mass` row and a 2-vector `momentum` row,
    each carrying the abstract test index `l` via an `a(l, t, x)` head — a
    stand-in for a post-`Project(c·φ(l,ζ))` Galerkin row.  No σ-map needed for
    the pure shape-bump tests."""
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    a = sp.Function("a")
    h = sp.Function("h", positive=True)(t, x)
    model.Q = [h]
    # mass row carries the abstract index l (∂_t a_l + ∂_x a_l).
    model.add_equation("mass", d.t(a(l, t, x)) + d.x(a(l, t, x)))
    # momentum is a 2-vector; each component carries l.
    model.add_equation("momentum", (2,), [
        d.t(a(l, t, x)) + d.x(a(l, t, x) ** 2),
        sp.Derivative(a(l, t, x), x),
    ])
    return model, a, h


# ── (1) scalar mass: (1,) → (N+1,) ────────────────────────────────────────


def test_resolve_modes_scalar_mass_shape_bump():
    model, a, h = _modal_model()
    N = 2
    resolve_modes(model.mass, index=l, modes=range(N + 1))

    # The mass equation's tensor shape bumped scalar (1,) → (N+1,).
    assert model._equation_shapes["mass"] == (N + 1,)

    # Indexed-family access: model.mass[l] is the l-th moment row.
    fam = model.mass
    assert fam[0].expr == (d.t(a(0, t, x)) + d.x(a(0, t, x)))
    assert fam[1].expr == (d.t(a(1, t, x)) + d.x(a(1, t, x)))
    assert fam[2].expr == (d.t(a(2, t, x)) + d.x(a(2, t, x)))

    # Attribute aliases ``.eq0 / .eq1 / …``.
    assert fam.eq0.expr == fam[0].expr
    assert fam.eq2.expr == fam[2].expr

    # The flattened scalar rows live under ``mass_0 / mass_1 / mass_2`` so the
    # structural extractor sees one scalar row per moment.
    assert {"mass_0", "mass_1", "mass_2"} <= set(model._equations)
    assert "mass" not in model._equations   # parent template consumed


# ── (2) vector momentum: (2,) → (2, N+1) ──────────────────────────────────


def test_resolve_modes_vector_momentum_shape_bump():
    model, a, h = _modal_model()
    N = 2
    resolve_modes(model.momentum.x, index=l, modes=range(N + 1))

    # The momentum equation's tensor shape bumped (2,) → (2, N+1) — the moment
    # axis appended to the component axis.
    assert model._equation_shapes["momentum"] == (2, N + 1)

    # model.momentum.x[l] is the (x, l) slice.
    xfam = model.momentum.x
    assert xfam[0].expr == (d.t(a(0, t, x)) + d.x(a(0, t, x) ** 2))
    assert xfam[1].expr == (d.t(a(1, t, x)) + d.x(a(1, t, x) ** 2))
    assert xfam[2].expr == (d.t(a(2, t, x)) + d.x(a(2, t, x) ** 2))

    # model.momentum[c, l] tensor access equals model.momentum.x[l].
    assert model.momentum["x", 1].expr == xfam[1].expr

    # Flattened scalar rows under ``momentum_x_0 / _1 / _2``.
    assert {"momentum_x_0", "momentum_x_1", "momentum_x_2"} <= set(
        model._equations)
    # The un-bumped ``momentum_x`` template is consumed; the z-component is
    # untouched (still a plain scalar row).
    assert "momentum_x" not in model._equations


# ── (2b) single-survivor collapse: `[l]` stays MOMENT-uniform ─────────────


def test_collapse_to_single_mode_keeps_moment_family():
    """When a closure consumes all but one mode, the family must NOT decay back
    to a bare scalar `Equation`: `model.mass` stays a `MomentFamily`,
    `model.mass[0]` is the surviving moment row, and `model.mass[0].term[i]`
    indexes that row's additive terms."""
    model, a, h = _modal_model()
    N = 2
    resolve_modes(model.mass, index=l, modes=range(N + 1))
    # Drop the higher moments, keep mode 0.
    for k in range(1, N + 1):
        model._remove_equation(f"mass_{k}")
    fam = model._collapse_moment_family("mass", keep=[0])

    from zoomy_core.model.derivation.model import MomentFamily
    from zoomy_core.model.equation import _TermAccessor

    # `[l]` is still moment-uniform — model.mass is a MomentFamily of one mode.
    assert isinstance(model.mass, MomentFamily)
    assert len(model.mass) == 1
    # model.mass[0] is the FULL surviving moment row ∂_t a_0 + ∂_x a_0.
    moment0 = model.mass[0]
    assert moment0.expr == (d.t(a(0, t, x)) + d.x(a(0, t, x)))
    # The flattened scalar row is re-keyed mass_0 → mass (structural extractor).
    assert "mass" in model._equations
    assert "mass_0" not in model._equations
    assert model._equations["mass"].expr == moment0.expr
    # model.mass[0].term[i] gives the moment row's additive terms.
    assert isinstance(moment0.term, _TermAccessor)
    assert moment0.term[0].expr == sp.expand(d.t(a(0, t, x)))
    assert moment0.term[1].expr == sp.expand(d.x(a(0, t, x)))


# ── (3) the per-moment Galerkin close (resolve closure) ────────────────────


def test_resolve_modes_with_galerkin_close():
    """With `resolve=` the bumped rows are each closed by the concrete-level
    Galerkin `Resolve` (φ(l,ζ) → φ(k,ζ) at the bound mode), so the opaque
    basis brackets are gone."""
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    a = sp.Function("a")
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    u = sp.Function("u", real=True)(t, x, z)
    w = sp.Function("w", real=True)(t, x, z)
    model.Q = [h, u, w]
    model.add_equation("mass", d.x(u) + d.z(w))
    model.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
    from zoomy_core.model.operations import Multiply
    model.mass.apply(Multiply(h))
    basis = Basis(symbol="phi", weight="c")
    phi, c = basis.phi, basis.weight
    a_w = sp.Function("aw")
    N_u = modal_bound("N_u")
    reset_modal_indices(model)
    model.apply(separation_of_variables(u, a(t, x), basis, N_u))
    model.apply(separation_of_variables(w, a_w(t, x), basis, N_u + 1))
    N = 2
    model.apply({N_u: N})
    for eq in model._equations.values():
        eq.expr = eq.expr.replace(lambda e: isinstance(e, sp.Sum),
                                  lambda e: e.doit())

    # Bump the mass row to (N+1,) and close each moment by Resolve(c·φ(l,ζ)).
    resolve_modes(model.mass, index=l, modes=range(N + 1),
                  test_weight=c(zeta) * phi(l, zeta),
                  basis_cls=Legendre_shifted, level=N + 1, var=zeta)

    assert model._equation_shapes["mass"] == (N + 1,)
    # No opaque basis atoms remain after the per-moment Galerkin close.
    for k in range(N + 1):
        row = model.mass[k].expr
        assert not any(getattr(at.func, "_is_basis_head", False)
                       for at in row.atoms(sp.Function)), (
            f"mass[{k}] still carries opaque basis atoms: {row}")
