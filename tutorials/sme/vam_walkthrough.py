# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VAM from Euler — symbolic walkthrough
#
# **Goal.** Derive a Vertically-Averaged Model (**VAM**) directly from
# Euler.  Unlike SWE/SME, we keep **pressure as a dynamic variable**
# (no hydrostatic assumption) and expand all three fields — `u`, `w`,
# and `p` — on a polynomial basis simultaneously.
#
# The derivation uses the *same* symbolic pipeline as the SWE/SME
# walkthroughs — no hardcoded terms, no "VAM operator" class.  What
# makes it VAM instead of SWE/SME is only:
#
# 1. We skip `hydrostatic_scaling` and the analytic integration of
#    z-momentum.
# 2. We project z-momentum against the test functions just like
#    x-momentum.
# 3. We apply three independent `basis.expand(...)` dicts — one per
#    field — so each of `u`, `w`, `p` carries its own set of
#    coefficients.
#
# At the end we'll be honest about what closes, what doesn't, and
# what's a **modelling** choice vs. a **symbolic** gap.

# %% [markdown]
# ## Imports + worktree bootstrap

# %%
import sys
from pathlib import Path

_here = Path.cwd()
while _here != _here.parent and not (_here / "library" / "zoomy_core" / "zoomy_core").exists():
    _here = _here.parent
_pkg_dir = _here / "library" / "zoomy_core"
_pkg_inner = _pkg_dir / "zoomy_core"

for _k in list(sys.modules):
    if _k == "zoomy_core" or _k.startswith("zoomy_core."):
        del sys.modules[_k]
for _finder in sys.meta_path:
    _mod = sys.modules.get(getattr(_finder, "__module__", "") or "")
    _mapping = getattr(_mod, "MAPPING", None) if _mod is not None else None
    if isinstance(_mapping, dict) and "zoomy_core" in _mapping:
        _mapping["zoomy_core"] = str(_pkg_inner)
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import sympy as sp
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Integrate, Inviscid,
    InterfaceKBC, Basis, Multiply, ProductRule, ZetaTransform,
    EvaluateIntegrals,
)
from zoomy_core.model.models.basisfunctions import Legendre_shifted

print("worktree root:", _here)

# %% [markdown]
# ## Step 1 — State space + three independent bases
#
# Same `StateSpace` as before.  The three `Basis` instances share the
# same Legendre structure but have **different `alpha_name`** kwargs,
# so their coefficient Zstructs are independent namespaces: `U_0, U_1`
# for velocity, `W_0, W_1` for the vertical component, `P_0, P_1` for
# pressure.

# %%
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z

basis_u = Basis(state, Legendre_shifted, level=1, alpha_name="U")
basis_w = Basis(state, Legendre_shifted, level=1, alpha_name="W")
basis_p = Basis(state, Legendre_shifted, level=1, alpha_name="P")

print("U coeffs:", [getattr(basis_u.alpha, f"U_{k}") for k in range(2)])
print("W coeffs:", [getattr(basis_w.alpha, f"W_{k}") for k in range(2)])
print("P coeffs:", [getattr(basis_p.alpha, f"P_{k}") for k in range(2)])

# %% [markdown]
# ## Step 2 — Full INS, Inviscid closure
#
# No `hydrostatic_scaling`, no solving z-momentum for `p` — we keep
# all three equations in full force.  Inviscid collapses the stress
# tensor to zero; otherwise the Newtonian closure would bring in the
# same unresolved `∂²u/∂z²` viscous integrals that show up in the
# single-layer SME walkthrough.

# %%
model = FullINS(state)
model.apply(Inviscid(state), name="Inviscid (Euler)",
            description="tau = 0").simplify()
model.describe()

# %% [markdown]
# ## Step 3 — Galerkin test of momentum (both x and z)
#
# For VAM we also project z-momentum onto the test functions — that's
# what makes it a Vertically-Averaged **Model** with its own
# `W_0, W_1` evolution equations, rather than a pure shallow-water
# formulation where `w` is eliminated via continuity.

# %%
model.momentum.x.apply(Multiply(basis_u.phi_of_z, outer=True),
                       name="Galerkin test — x-momentum",
                       description="multiply x-momentum by phi_l((z-b)/h)")
model.momentum.z.apply(Multiply(basis_u.phi_of_z, outer=True),
                       name="Galerkin test — z-momentum",
                       description="multiply z-momentum by phi_l((z-b)/h)")
model.momentum.x.apply(ProductRule())
model.momentum.z.apply(ProductRule())

# %% [markdown]
# ## Step 4 — Depth-integrate, KBCs, atmospheric pressure BC
#
# Standard shapes — one `Integrate` over the full depth, one KBC at
# each boundary, and the atmospheric-pressure condition
# `p|_eta = 0`.

# %%
model.apply(Integrate(z, state.b, state.eta, method="auto"),
            name="∫ dz over depth")
model.apply(InterfaceKBC(state, state.b)).simplify()
model.apply(InterfaceKBC(state, state.eta)).simplify()
model.apply({state.p.subs(z, state.eta): 0},
            name="atmospheric pressure",
            description="p(t, x, eta) = 0").simplify()

# %% [markdown]
# ## Step 5 — ZetaTransform + three basis expansions + EvaluateIntegrals
#
# The three `basis_{u,w,p}.expand(state.{u,w,p})` calls are
# independent substitution dicts applied back-to-back.  `xreplace` is
# structural, so there's no interaction between them.

# %%
model.apply(ZetaTransform(state))
model.apply(basis_u.expand(state.u), name="expand u").simplify()
model.apply(basis_w.expand(state.w), name="expand w").simplify()
model.apply(basis_p.expand(state.p), name="expand p").simplify()
model.apply(EvaluateIntegrals(state)).simplify()

# %% [markdown]
# ## Step 6 — What's closed

# %%
for path, eq in model.leaves():
    ur = eq.expr.has(state.u)
    wr = eq.expr.has(state.w)
    pr = eq.expr.has(state.p)
    zr = eq.expr.has(state.zeta)
    closed = not (ur or wr or pr or zr)
    mark = "✓" if closed else "✗"
    bits = [n for n, r in [("u", ur), ("w", wr), ("p", pr), ("ζ", zr)] if r]
    note = f"   residual: {','.join(bits)}" if bits else ""
    print(f"  {mark} {'.'.join(path):<26} n_terms={len(eq):>4d}{note}")

# %% [markdown]
# ## Step 7 — The equations (explicit)

# %%
for path, eq in model.leaves():
    print(f"\n{'.'.join(path)}:")
    for term in sp.Add.make_args(sp.expand(eq.expr)):
        print(f"  {term}")

# %% [markdown]
# ## Honest accounting — what the pipeline produced
#
# Five equations, all closed against the basis coefficients:
#
# * **continuity** — depth-averaged mass balance $\partial_t h +
#   \partial_x(U_0 h) = 0$.  ``W_k`` doesn't appear (orthogonality
#   eats it) and the ``U_1`` term also drops out via Legendre
#   ``∫ (1 - 2ζ) dζ = 0``.
# * **momentum.x.test_0** — depth-averaged x-momentum with mean
#   convection ``U_0²·h`` plus shear correction ``U_1²·h/3`` plus
#   pressure terms in ``P_0, P_1``.
# * **momentum.x.test_1** — first-moment x-momentum; couples
#   ``U_0 · U_1`` and ``U_1·P_k`` type terms.
# * **momentum.z.test_0** — depth-averaged z-momentum with gravity
#   ``gh``, bottom pressure ``−(P_0 + P_1)/ρ``, and advective
#   coupling ``U_0 W_0 h``, ``U_1 W_1 h/3``.
# * **momentum.z.test_1** — first-moment z-momentum.
#
# ### What's missing (honestly)
#
# 1. **The system is underdetermined.**  Unknowns are
#    ``h, U_0, U_1, W_0, W_1, P_0, P_1`` → 7 variables.  We have 5
#    equations.  Standard VAM closures pick one of:
#
#    * ``W_0`` from depth-integrated continuity (``W`` becomes a
#      dependent variable, not an independent one).
#    * A linear-pressure constraint ``P_1 = 0``.
#    * Both, reducing to 5 unknowns and 5 equations.
#
#    **This is a modelling choice**, not a symbolic gap.  The
#    pipeline is neutral on it.
#
# 2. **Continuity was only tested against ``φ_0``** (no ``test_1``
#    equation for continuity).  Adding
#    ``model.continuity.apply(Multiply(basis_u.phi_of_z, outer=True))``
#    before step 4 would add a ``continuity.test_1`` equation for a
#    6-eq system.  Closer to balanced but still 1 short.
#
# 3. **No viscous closure.**  We used ``Inviscid``.  With
#    ``Newtonian(state)`` the ``∂²u/∂z²`` term enters, which doesn't
#    reduce by the fundamental theorem (sympy's ``_extract_derivative``
#    refuses because the outer differentiation is applied twice) and
#    ends up as a residual volume integral.  This is the **same gap**
#    the single-layer SME walkthrough flagged — common to SME, VAM,
#    and layered SME.
#
# 4. **No multi-layer VAM.**  Swapping ``basis_{u,w,p}`` for
#    ``Basis(state, LayeredBasis, inner_cls=Legendre_shifted,
#    inner_level=1, interfaces=[b, z_1, eta])`` and running the
#    per-layer pipeline from the multi-layer SWE walkthrough is the
#    natural next extension.  We haven't run it end-to-end.
#
# 5. **Display is expanded form.**  ``Recombine(vars=[t, x])`` (the
#    "fold `∂_t h + u·∂_t h → ∂_t(hu)` " op) is still on the
#    wish-list; cosmetic.
#
# ### What is **not** a problem
#
# * Hard-coding: there's no VAM-specific operator anywhere in this
#   notebook.  The derivation reads as a straight sequence of the
#   same `Integrate`, `InterfaceKBC`, `Multiply`, `ProductRule`,
#   `ZetaTransform`, `basis.expand`, `EvaluateIntegrals` steps used in
#   the SWE / SME walkthroughs.  Swap `Inviscid` for `Newtonian`,
#   change the basis, swap ``phi_of_z`` for a layered variant —
#   every knob is an input to the existing ops.
# * Each of the five equations is **closed** against the
#   ``(U_k, W_k, P_k)`` coefficients (no residual ``u, w, p, ζ``).
#   Under-determinacy is about having too few equations, not about
#   the equations being malformed.
