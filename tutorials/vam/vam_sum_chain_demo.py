# ---
# title: "Sum-based Galerkin chain — VAM at L=0,1,2"
# author: Ingo Steldermann
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# # Sum-based Galerkin chain — VAM at L=0, L=1, L=2
#
# Demonstrates the symbolic chain after steps 1–3 of the
# infrastructure refactor.  Three things to look at:
#
# 1. **`Expand`** produces an *unevaluated* `sp.Sum` per ansatz field
#    instead of unrolling into `L+1` separate Add terms.  After
#    `Expand`, each occurrence of `state.u(t,x,arg_z)` reads as
#    `Sum(amp_fn(k) · phi_fn(k, ζ_val), (k, 0, L))` — a single atom.
#
# 2. **`EvaluateIntegrals`** unrolls those Sums and routes the
#    resolution: `phi_fn(k, ζ)` opaque atoms get substituted with the
#    concrete polynomial via the basis's `resolve_atoms`, then the
#    polynomial integrand is closed by Zoomy's integration rules
#    (`_cached_integrate`).  No raw `sp.integrate` calls.
#
# 3. **Three independent bases** (`phi`, `eta`, `mu`) coexist in the
#    same equation as distinct sympy classes via the constructor
#    `symbol=` kwarg — `_basis` back-references on the `Function`
#    classes themselves do all the routing; no registry needed.
#
# The chain runs end-to-end at L=0, L=1, L=2 — every leaf closes
# cleanly (no remaining `Sum`s, no held `Integral`s, no residual
# `state.u/w/p/ζ` atoms).

# +
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.ins_generator import (
    StateSpace,
    FullINS,
    Inviscid,
    Multiply,
    Integrate,
    InterfaceKBC,
    AffineProjection,
    Expand,
    EvaluateIntegrals,
)
from zoomy_core.model.models.basisfunctions import Legendre_shifted
# -


# ## The chain — one function, parameterised by level
#
# The same chain runs for any L.  Per-step reasoning is in the
# inline comments.
#
# ### Why `phi(k, (z-b)/h)` and not `phi(k, z)` directly
#
# The basis polynomials live on the reference element ξ ∈ [0, 1].
# Galerkin-testing against them in a depth integral over physical
# z ∈ [b, b+h] requires the relative coordinate (z-b)/h to be the
# basis argument.  The composition `phi(k, (z-b)/h)` is the explicit
# semantic bridge: "this is basis k evaluated at the relative
# coordinate, expressed in terms of physical z."
#
# `AffineProjection` later substitutes `z → ζ·h + b` everywhere
# structurally; the composition collapses cleanly:
#
#     phi(k, (z-b)/h)   z → ζ·h+b   →   phi(k, ζ)
#
# — and the integrand emerges in the reference-element form ready
# for `EvaluateIntegrals`.  Writing `phi(k, z)` directly would leave
# `phi(k, ζ·h+b)` after `AffineProjection`, which is *not* in [0, 1]
# and has no orthogonality structure there.

# +
def derive_vam(level: int):
    """Run the Sum-based Galerkin chain for VAM at the given level."""
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z

    basis_u = Legendre_shifted(level=level, symbol="phi")
    basis_w = Legendre_shifted(level=level, symbol="eta")
    basis_p = Legendre_shifted(level=level, symbol="mu")

    coeffs_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(level + 1)]
    coeffs_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(level + 1)]
    coeffs_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(level + 1)]

    test_phi_of_z = Zstruct(
        **{f"phi_{k}": basis_u.phi[k]((z - state.b) / state.H)
           for k in range(level + 1)}
    )

    # 1. Start from full INS, apply inviscid closure.
    model = FullINS(state)
    model.apply(Inviscid(state)).simplify()

    # 2. Galerkin test: multiply momentum.x and momentum.z by every
    #    test function.  Each scalar momentum equation lifts into a
    #    Zstruct of ``test_0..test_L`` sub-equations.
    model.momentum.x.apply(Multiply(test_phi_of_z, outer=True))
    model.momentum.z.apply(Multiply(test_phi_of_z, outer=True))

    # 3. Depth-integrate, apply kinematic BCs, atmospheric pressure.
    model.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.apply(InterfaceKBC(state, state.b)).simplify()
    model.apply(InterfaceKBC(state, state.eta)).simplify()
    model.apply({state.p.subs(z, state.eta): 0}).simplify()

    # 4. Reference-element map (was ZetaTransform; renamed to
    #    AffineProjection).
    model.apply(AffineProjection(state))

    # 5. Ansatz substitutions — produce unevaluated ``sp.Sum`` atoms
    #    holding opaque ``phi_fn``/``eta_fn``/``mu_fn`` Functions.
    model.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
    model.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w, state=state))
    model.apply(Expand(state.p, basis=basis_p, coefficients=coeffs_p, state=state))

    # 6. Resolve every Integral.  Internally:
    #    - Sums get ``.doit()``-ed (amp_fn auto-evaluates per index);
    #    - opaque phi atoms get ``resolve_atoms``-ed to concrete polys;
    #    - resulting polynomial integrand routes through Zoomy's
    #      integration rule cache.
    model.apply(EvaluateIntegrals(state)).simplify()

    return model


def derive_vam_intermediate(level: int):
    """Same chain but stops right after the three Expand calls so we
    can inspect the un-resolved ``sp.Sum`` form.
    """
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z
    basis_u = Legendre_shifted(level=level, symbol="phi")
    basis_w = Legendre_shifted(level=level, symbol="eta")
    basis_p = Legendre_shifted(level=level, symbol="mu")
    coeffs_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(level + 1)]
    coeffs_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(level + 1)]
    coeffs_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(level + 1)]
    test_phi_of_z = Zstruct(
        **{f"phi_{k}": basis_u.phi[k]((z - state.b) / state.H)
           for k in range(level + 1)}
    )
    model = FullINS(state)
    model.apply(Inviscid(state)).simplify()
    model.momentum.x.apply(Multiply(test_phi_of_z, outer=True))
    model.momentum.z.apply(Multiply(test_phi_of_z, outer=True))
    model.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.apply(InterfaceKBC(state, state.b)).simplify()
    model.apply(InterfaceKBC(state, state.eta)).simplify()
    model.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.apply(AffineProjection(state))
    model.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
    model.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w, state=state))
    model.apply(Expand(state.p, basis=basis_p, coefficients=coeffs_p, state=state))
    return model
# -


# ## L=0 — closed VAM equations
#
# After the full chain, every leaf is closed in the basis amplitudes
# `(U_0, W_0, P_0)`.  No remaining `Sum`s, `Integral`s, or
# residual `state.u/w/p/ζ` atoms.

model_L0 = derive_vam(0)
model_L0.describe()


# ## L=1 — Sum-form intermediates (after `Expand`, before `EvaluateIntegrals`)
#
# Each leaf carries a small number of `Sum` atoms — the ansatz is
# unevaluated, so the equation is compact.  The Sum is a *single
# term* in each leaf's Add decomposition, which is what makes
# `ProductRule`'s `apply_to_term` cheap to use even at higher
# levels.

intermediate_L1 = derive_vam_intermediate(1)
intermediate_L1.describe()


# ## L=1 — closed system after `EvaluateIntegrals`

model_L1 = derive_vam(1)
model_L1.describe()


# ## L=2 — closed system

model_L2 = derive_vam(2)
model_L2.describe()


# ## What this proves
#
# 1. The Sum-based `Expand` produces equations that close at every
#    level we've tried (L=0, L=1, L=2) without manual term-picking.
#
# 2. The opaque-phi routing in `EvaluateIntegrals` correctly handles
#    three independent bases coexisting in one equation.
#
# 3. The integration of post-resolution polynomial integrands
#    (`Derivative`-outer over rational-in-h polynomial-in-z_hat)
#    routes through Zoomy's `_cached_integrate` rule set, not raw
#    `sp.integrate` — which was the level=1 bug fix.
#
# What's still pending (bigger refactors):
#
# - Step 4: `Model` and `SystemModel` siblings sharing the operator
#   API surface; `SystemModel.from_model(m)` factory.
# - Step 6: `VAMModel.derive_model` rebuilt to use this chain
#   internally + author-side term tagging at the end.
# - Step 9: refresh `tutorials/vam/vam_get_pde_walkthrough.py` to use
#   the full pipeline — model class → SystemModel → analysis → solver.
