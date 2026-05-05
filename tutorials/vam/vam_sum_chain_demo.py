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

# +
def derive_vam(level: int):
    """Run the Sum-based Galerkin chain for VAM at the given level.

    Returns the derived ``System`` (a tree of leaves with closed
    equations) — caller can inspect any leaf's ``expr`` directly.
    """
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z

    # Three independent bases on the same model.  Distinct ``symbol``
    # values produce distinct sympy Function classes (``phi_…``,
    # ``eta_…``, ``mu_…``); ``_basis`` back-references on each class
    # carry the link to the basis instance for later resolution.
    basis_u = Legendre_shifted(level=level, symbol="phi")
    basis_w = Legendre_shifted(level=level, symbol="eta")
    basis_p = Legendre_shifted(level=level, symbol="mu")

    # Pre-declared amplitudes: state-Function calls of (t, x).  These
    # are what would be the model's evolved fields downstream — for
    # the demo we just declare them inline.
    amps_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(level + 1)]
    amps_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(level + 1)]
    amps_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(level + 1)]

    # Test functions for Galerkin testing — these go into ``Multiply``.
    # ``basis_u.phi[k](arg)`` calls the opaque 2-arg phi_fn at index k.
    test_phi_of_z = Zstruct(
        **{f"phi_{k}": basis_u.phi[k]((z - state.b) / state.H)
           for k in range(level + 1)}
    )

    # 1. Start from full INS, apply inviscid closure.
    model = FullINS(state)
    model.apply(Inviscid(state)).simplify()

    # 2. Galerkin test: multiply momentum.x and momentum.z by every
    #    test function.  This lifts each scalar momentum equation
    #    into a Zstruct of ``test_0..test_L`` sub-equations.
    model.momentum.x.apply(Multiply(test_phi_of_z, outer=True))
    model.momentum.z.apply(Multiply(test_phi_of_z, outer=True))

    # 3. Depth-integrate, apply kinematic BCs, atmospheric pressure.
    model.apply(Integrate(z, state.b, state.eta, method="auto"))
    model.apply(InterfaceKBC(state, state.b)).simplify()
    model.apply(InterfaceKBC(state, state.eta)).simplify()
    model.apply({state.p.subs(z, state.eta): 0}).simplify()

    # 4. Reference-element map (was ZetaTransform; renamed).
    model.apply(AffineProjection(state))

    # 5. Ansatz substitutions — produce unevaluated ``sp.Sum`` atoms
    #    holding opaque ``phi_fn``/``eta_fn``/``mu_fn`` Functions.
    model.apply(Expand(state.u, basis=basis_u, amplitudes=amps_u, state=state))
    model.apply(Expand(state.w, basis=basis_w, amplitudes=amps_w, state=state))
    model.apply(Expand(state.p, basis=basis_p, amplitudes=amps_p, state=state))

    # 6. Resolve every Integral.  Internally:
    #    - Sums get ``.doit()``-ed (amp_fn auto-evaluates per index);
    #    - opaque phi atoms get ``resolve_atoms``-ed to concrete polys;
    #    - resulting polynomial integrand routes through Zoomy's
    #      integration rule cache.
    model.apply(EvaluateIntegrals(state)).simplify()

    return model, state
# -


# ## Inspect intermediate Sum-form (before EvaluateIntegrals)
#
# Cut the chain just after `Expand` to see what `Sum` atoms look like
# in the symbolic equation.  The Sum carries through `ProductRule`
# (single-term op via `apply_to_term`) and `Integrate` as a single
# atom — that's the whole point of step 2's redesign.

# +
def derive_vam_intermediate(level: int):
    """Same chain but stop right after the Expand calls."""
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z
    basis_u = Legendre_shifted(level=level, symbol="phi")
    basis_w = Legendre_shifted(level=level, symbol="eta")
    basis_p = Legendre_shifted(level=level, symbol="mu")
    amps_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(level + 1)]
    amps_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(level + 1)]
    amps_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(level + 1)]
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
    model.apply(Expand(state.u, basis=basis_u, amplitudes=amps_u, state=state))
    model.apply(Expand(state.w, basis=basis_w, amplitudes=amps_w, state=state))
    model.apply(Expand(state.p, basis=basis_p, amplitudes=amps_p, state=state))
    return model
# -


# ## Run at L=0 — print the closed equations

# +
print("=" * 72)
print("Level 0 — closed VAM(level=0) equations")
print("=" * 72)
sys_L0, _ = derive_vam(0)
for path, eq in sys_L0.leaves():
    print(f"\n[{'.'.join(path)}]")
    print(f"  {sp.expand(eq.expr)}")

print()
print("Expected (from the standard chain):")
print("  continuity:  ∂_t h + ∂_x(U_0·h) = 0")
print("  momentum.x:  ∂_t(U_0·h) + ∂_x(U_0²·h) + ∂_x(P_0·h/ρ) + P_0·∂_x b/ρ = 0")
print("  momentum.z:  g·h + ∂_t(W_0·h) + ∂_x(U_0·W_0·h) − P_0/ρ = 0")
# -


# ## Inspect intermediate Sum-form at L=1

# +
print()
print("=" * 72)
print("Level 1 — Sum-form intermediates (after Expand, before EvaluateIntegrals)")
print("=" * 72)
inter_L1 = derive_vam_intermediate(1)
for path, eq in inter_L1.leaves():
    n_sums = len(eq.expr.atoms(sp.Sum))
    n_integrals = len(eq.expr.atoms(sp.Integral))
    print(f"  [{'.'.join(path):<26}]  Sum atoms: {n_sums:>2d}   Integral atoms: {n_integrals:>2d}")
print()
print(
    "Each leaf carries Sum atoms — the ansatz is unevaluated.\n"
    "The Sum is a SINGLE term in the leaf's Add decomposition,\n"
    "which is what makes ProductRule's apply_to_term cheap to use\n"
    "(no need to apply L+1 times)."
)
# -


# ## Closure check across L=0, L=1, L=2

# +
print()
print("=" * 72)
print("Closure check — every leaf, every level")
print("=" * 72)
for L in (0, 1, 2):
    print(f"\nLevel {L}:")
    sys_L, state = derive_vam(L)
    for path, eq in sys_L.leaves():
        residuals = []
        if eq.expr.has(state.u):
            residuals.append("u")
        if eq.expr.has(state.w):
            residuals.append("w")
        if eq.expr.has(state.p):
            residuals.append("p")
        if eq.expr.has(state.zeta):
            residuals.append("ζ")
        if eq.expr.atoms(sp.Sum):
            residuals.append("Sum")
        if eq.expr.atoms(sp.Integral):
            residuals.append("Integral")
        mark = "✓" if not residuals else "✗"
        note = f"   residuals: {','.join(residuals)}" if residuals else ""
        print(f"  {mark} [{'.'.join(path):<26}]  n_terms={len(eq):>4d}{note}")
# -


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
