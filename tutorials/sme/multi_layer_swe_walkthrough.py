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
# # Multi-layer SWE from Euler — symbolic walkthrough
#
# **Goal.** Derive the classical multi-layer shallow-water equations
# with interface mass flux from the Euler equations, using the
# symbolic pipeline.
#
# The two new pieces of library surface used here:
#
# * **`InterfaceKBC(state, interface, mass_flux=None)`** — generalised
#   kinematic boundary condition at an arbitrary interface height,
#   with an optional mass-flux term.  Subsumes `KinematicBCBottom` /
#   `KinematicBCSurface` as the special cases `mass_flux=None`.
# * **`LayeredBasis(inner_cls=Monomials, inner_level=0,
#   interfaces=[…])`** — composable multi-layer basis.  Rescales any
#   existing `Basisfunction` onto each of `N` sub-intervals (in
#   ζ-space or physical z-space).  With the defaults it's
#   piecewise-constant (1 per layer → multi-layer SWE).  Swap
#   `inner_cls=Legendre_shifted, inner_level=L` and the same pipeline
#   produces a multi-layer SME with ``L+1`` moments per layer.
#   Driven via `basis.layer_expand(field, layer_idx)`.
#
# The remaining ergonomic polish (a `Layers(state, N)` bundle and a
# conservative-form `Recombine` display op) is left for later.

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
    StateSpace, FullINS, Integrate, Inviscid, SimplifyIntegrals,
    InterfaceKBC, Basis, Multiply, ExpandProductRule, ZetaTransform,
    EvaluateIntegrals, ContinuityClosure,
)
from zoomy_core.model.models.basisfunctions import LayeredBasis, Monomials, Legendre_shifted
from zoomy_core.model.models.derived_system import combined_history_mermaid
from zoomy_core.model.models.sme_model import hydrostatic_scaling

print("worktree root:", _here)

# %% [markdown]
# ## Step 1 — State space + layer quantities
#
# `StateSpace(dimension=2)` gives `(t, x, z)` and the fields `u, w,
# p, tau_ij, b, h, eta`.  For two layers we additionally need an
# internal interface height `z_1(t, x)` and its mass flux `m_1(t, x)`.

# %%
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
z_1 = sp.Function("z_1", real=True)(t, x)
m_1 = sp.Function("m_1", real=True)(t, x)

# %% [markdown]
# ## Step 2 — LayeredBasis with physical interfaces
#
# `LayeredBasis` composes *any* existing `Basisfunction` with a
# partition of the vertical axis.  For multi-layer SWE we want:
#
# * **Inner basis:** one constant per layer (`Monomials(level=0)`)
#   → ``phi_inner_0 = 1``.
# * **Interfaces:** `[b, z_1, eta]` in physical z-space.
#
# Both are the **defaults** when you just pass `interfaces=[…]`
# (inner_cls defaults to `Monomials`, `inner_level=0`).  The basis
# has ``N × (inner_level + 1) = 2 × 1 = 2`` coefficients total;
# `basis.alpha` auto-mints `alpha_0(t, x)` (layer 0 velocity) and
# `alpha_1(t, x)` (layer 1 velocity).
#
# If you later want layered SME instead, swap in Legendre:
# `LayeredBasis(Legendre_shifted, inner_level=1, interfaces=[…])` —
# same pipeline, 2 layers × 2 moments = 4 coefficients.

# %%
basis = Basis(
    state,
    LayeredBasis,
    interfaces=[state.b, z_1, state.eta],   # defaults: Monomials, inner_level=0
)
print("level:      ", basis.level)
print("n_layers:   ", basis._bf.n_layers)
print("inner:      ", basis._bf.inner.name, f"(level={basis._bf.inner.level})")
print("alpha:      ", [getattr(basis.alpha, f"alpha_{k}") for k in range(basis.level + 1)])
print("interfaces: ", basis.interfaces)

# %% [markdown]
# ## Step 3 — Shared prefix: Euler with hydrostatic pressure
#
# Same as the SME walkthrough steps 2–6, except **Inviscid** so we
# stay with Euler.  This lives on the parent `model`; each layer
# branches off once we've done this.

# %%
model = FullINS(state)
model.name = "Euler"

model.momentum.z.apply(hydrostatic_scaling(state),
                       name="hydrostatic",
                       description="w=0 + tau=0 in z-mom").simplify()
model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"),
                       name="integrate z-mom",
                       description="analytic z-integration")
model.momentum.z.apply({state.p.subs(z, state.eta): 0},
                       name="atmospheric pressure",
                       description="p(t,x,eta) = 0").simplify()
model.momentum.x.apply(model.momentum.z.solve_for(state.p),
                       name="substitute p(t,x,z)").simplify()
model.momentum.z.remove()
model.apply(Inviscid(state),
            name="Inviscid (Euler)",
            description="tau = 0 everywhere").simplify()
model.describe()

# %% [markdown]
# ## Step 4 — Fork per layer

# %%
layer_0 = model.branch(name="layer 0")
layer_1 = model.branch(name="layer 1")

# %% [markdown]
# ## Step 5 — Per-layer pipeline
#
# For each layer: depth-integrate, apply `InterfaceKBC` at both
# bounding interfaces (with mass flux at internal ones, `None` at
# impermeable bed / free surface), substitute via `basis.layer_expand`
# to close against the layer's α, and collapse constant integrals.

# %%
def close_layer(branch, lower, upper, layer_idx, m_low=None, m_up=None):
    """Run the per-layer pipeline using the new library APIs."""
    branch.apply(
        Integrate(z, lower, upper, method="auto"),
        name=f"∫ dz layer {layer_idx}",
        description=f"depth-integrate [{lower}, {upper}]",
    )
    # Generalised kinematic BC — one class, two calls.
    branch.apply(InterfaceKBC(state, lower, mass_flux=m_low)).simplify()
    branch.apply(InterfaceKBC(state, upper, mass_flux=m_up)).simplify()
    # Layer closure: u(t,x,z) = α_{layer_idx}(t,x) inside + at both
    # interfaces (side-local convention — neighbour applies its own α).
    branch.apply(
        basis.layer_expand(state.u, layer_idx),
        name=f"close layer {layer_idx}",
        description=f"u -> alpha_{layer_idx}",
    ).simplify()
    branch.apply(SimplifyIntegrals(state),
                 name="collapse constant ints").simplify()


# Layer 0: [b, z_1].  Impermeable bed below, mass flux m_1 above.
close_layer(layer_0, state.b, z_1, layer_idx=0, m_low=None, m_up=m_1)
# Layer 1: [z_1, eta].  Mass flux m_1 below, free surface above.
close_layer(layer_1, z_1, state.eta, layer_idx=1, m_low=m_1, m_up=None)

# %% [markdown]
# ## Step 6 — Results

# %%
layer_0.describe()

# %%
layer_1.describe()

# %% [markdown]
# Group-by-hand / canonical form:
#
# **Layer 0 `(b, z_1)`:**
# $$
#   \partial_t h_0 + \partial_x(h_0\,\alpha_0) + m_1/\rho = 0
# $$
# $$
#   \partial_t(h_0\,\alpha_0) + \partial_x(h_0\,\alpha_0^2)
#   + g\,h_0\,\partial_x \eta + m_1 \alpha_0 / \rho = 0
# $$
#
# **Layer 1 `(z_1, \eta)`:**
# $$
#   \partial_t h_1 + \partial_x(h_1\,\alpha_1) - m_1/\rho = 0
# $$
# $$
#   \partial_t(h_1\,\alpha_1) + \partial_x(h_1\,\alpha_1^2)
#   + g\,h_1\,\partial_x \eta - m_1 \alpha_1 / \rho = 0
# $$

# %% [markdown]
# ## Step 7 — Symbolic verification
#
# `sp.expand(eq - canonical)` should be zero per equation.

# %%
a0 = basis.alpha.alpha_0
a1 = basis.alpha.alpha_1
h_0 = z_1 - state.b
h_1 = state.eta - z_1
canon = {
    ("layer 0", "continuity"):
        sp.Derivative(h_0, t) + sp.Derivative(h_0 * a0, x) + m_1 / state.rho,
    ("layer 0", "momentum.x"):
        (sp.Derivative(h_0 * a0, t) + sp.Derivative(h_0 * a0**2, x)
         + state.g * h_0 * sp.Derivative(state.eta, x) + m_1 * a0 / state.rho),
    ("layer 1", "continuity"):
        sp.Derivative(h_1, t) + sp.Derivative(h_1 * a1, x) - m_1 / state.rho,
    ("layer 1", "momentum.x"):
        (sp.Derivative(h_1 * a1, t) + sp.Derivative(h_1 * a1**2, x)
         + state.g * h_1 * sp.Derivative(state.eta, x) - m_1 * a1 / state.rho),
}
for branch, tag in [(layer_0, "layer 0"), (layer_1, "layer 1")]:
    for path, eq in branch.leaves():
        key = (tag, ".".join(path))
        residual = sp.simplify(sp.expand(eq.expr) - sp.expand(canon[key].doit()))
        status = "OK" if residual == 0 else f"RESIDUAL: {residual}"
        print(f"{tag:<8} {'.'.join(path):<12} {status}")

# %% [markdown]
# ## Step 8 — Combined derivation flowchart

# %%
combined_history_mermaid(layer_0, layer_1)

# %% [markdown]
# ## What changed vs. the hand-rolled version
#
# The earlier draft of this notebook built the per-interface KBC as a
# raw three-line dict and the layer closure as a second raw dict of
# three keys each.  With the two library additions:
#
# * The KBC is `InterfaceKBC(state, interface, mass_flux=...)` — one
#   positional, one kwarg.  The `mass_flux=None` default covers the
#   two cases that used to need separate classes
#   (`KinematicBCBottom`, `KinematicBCSurface`) so the same name now
#   describes every interface in a multi-layer stack.
# * The layer closure is `basis.layer_expand(state.u, layer_idx)` —
#   the basis already knows its interfaces, so there's nothing to
#   pass besides the layer index.
#
# Both show up in the `history_mermaid` with readable labels instead
# of `substitute 2 rules`, so a branching derivation remains legible
# even when it gets large.
#
# ## Step 9 — Same pipeline, different inner basis = multi-layer SME
#
# The composable nature of `LayeredBasis` means we don't need a new
# walkthrough to get multi-layer SME — just change the inner basis.
# The next cell *defines* (but doesn't run) a layered-Legendre basis
# to show what the swap looks like:

# %%
sme_basis = Basis(
    state,
    LayeredBasis,
    inner_cls=Legendre_shifted,
    inner_level=1,                                 # level-1 Legendre per layer
    interfaces=[state.b, z_1, state.eta],          # same interfaces
)
print("SME basis level:", sme_basis.level)        # 2 layers × 2 moments - 1 = 3
print("SME alpha:",
      [getattr(sme_basis.alpha, f"alpha_{k}") for k in range(sme_basis.level + 1)])
# Interpretation: alpha_0 = layer-0 mean, alpha_1 = layer-0 shear,
#                 alpha_2 = layer-1 mean, alpha_3 = layer-1 shear.

# %% [markdown]
# Layer-expand on this SME basis gives the full Legendre closure
# rescaled into each layer.  The multi-layer SME derivation runs the
# exact same per-layer pipeline as the SWE one — only
# `basis.layer_expand(...)` substitutes a richer sum.

# %%
print("layer 0 closure keys / values:")
for k, v in sme_basis.layer_expand(state.u, 0).items():
    print(f"  {k}")
    print(f"    -> {v}")

# %% [markdown]
# ## Step 10 — Running the multi-layer level-1 SME end-to-end
#
# Same pipeline shape as the SWE case + per-moment Galerkin testing:
# for each layer we multiply the momentum by the **layer-local**
# Legendre test functions via `basis.layer_phi_of_z(i)`, run
# `ExpandProductRule` so `φ·∂_v f` splits into conservative
# (``∂_v(φ·f)``) and non-conservative (``∂_v(φ)·f``) pieces, then the
# usual Integrate → KBCs → ZetaTransform → layer_expand →
# EvaluateIntegrals chain.
#
# Continuity stays a single scalar equation per layer (no Galerkin);
# momentum.x becomes a ``Zstruct(test_0, test_1)`` per layer.

# %%
def close_sme_layer(branch, basis, lower, upper, layer_idx, m_low=None, m_up=None):
    """Per-layer SME Galerkin pipeline."""
    # Per-moment test functions BEFORE depth-integration.
    branch.momentum.x.apply(
        Multiply(basis.layer_phi_of_z(layer_idx), outer=True),
        name=f"Galerkin test (layer {layer_idx})",
        description="multiply by layer-local φ_k((z - z_i)/h_i)",
    )
    # φ·∂_v(f) → ∂_v(φ·f) − ∂_v(φ)·f so Integrate can Leibniz.
    branch.momentum.x.apply(
        ExpandProductRule([state.t, state.x, state.z]),
        name="expand product rule",
    )
    branch.apply(Integrate(state.z, lower, upper, method="auto"),
                 name=f"∫ dz layer {layer_idx}")
    branch.apply(InterfaceKBC(state, lower, mass_flux=m_low)).simplify()
    branch.apply(InterfaceKBC(state, upper, mass_flux=m_up)).simplify()
    # Close bulk w via continuity integrated from the layer's lower
    # interface.  For an internal interface, include the mass-flux
    # contribution supplied by ``m_low``.
    branch.apply(ContinuityClosure(state, lower=lower, mass_flux=m_low),
                 name=f"continuity closure (layer {layer_idx})")
    branch.apply(ZetaTransform(state, lower=lower, upper=upper))
    # Close u twice: ζ-transformed form (outer integrals) and pointwise
    # form (inner running integral from ContinuityClosure).
    branch.apply(basis.layer_expand(state.u, layer_idx, zeta_transformed=True))
    branch.apply(basis.layer_expand(state.u, layer_idx, zeta_transformed=False))
    branch.apply(EvaluateIntegrals(state)).simplify()


layer_0_sme = model.branch(name="layer 0 SME")
layer_1_sme = model.branch(name="layer 1 SME")

close_sme_layer(layer_0_sme, sme_basis, state.b, z_1, 0,
                m_low=None, m_up=m_1)
close_sme_layer(layer_1_sme, sme_basis, z_1, state.eta, 1,
                m_low=m_1, m_up=None)

# %% [markdown]
# ## Step 11 — What closed, what didn't (honest accounting)
#
# Inspect each leaf.  "Closed" here means the expression has no
# residual ``u``, ``w``, or ``zeta`` — only ``α_k``'s,
# interface heights, bathymetry, mass fluxes, and their derivatives.

# %%
def audit(branch, tag):
    for path, eq in branch.leaves():
        ur = eq.expr.has(state.u)
        wr = eq.expr.has(state.w)
        zr = eq.expr.has(state.zeta)
        closed = not (ur or wr or zr)
        mark = "✓" if closed else "✗"
        note = ""
        if not closed:
            bits = []
            if ur: bits.append("u")
            if wr: bits.append("w")
            if zr: bits.append("ζ")
            note = f"  (residual: {', '.join(bits)})"
        print(f"  {mark} {tag:<14} {'.'.join(path):<22} {len(eq):>3d} terms{note}")

audit(layer_0_sme, "layer 0 SME")
audit(layer_1_sme, "layer 1 SME")

# %% [markdown]
# ## Step 12 — What's missing for a fully-closed multi-layer SME
#
# The shear moment (``momentum.x.test_1``) still carries
# ``∫_0^1 w(t, x, ζ·h_i + z_i) dζ`` and
# ``∫_0^1 ζ·w(t, x, ζ·h_i + z_i) dζ`` — volume moments of the
# **vertical** velocity ``w``.  The pipeline never closed ``w``.
# For the SWE / mean-moment case Legendre orthogonality eliminated
# these integrals (``∫_0^1 w dζ`` in continuity becomes a layer
# thickness tendency via the KBCs); at the shear moment the
# orthogonality doesn't save us.
#
# The standard closure is to solve continuity (``∂_x u + ∂_z w = 0``)
# for ``w(z)`` with boundary data from the layer's bottom KBC and
# substitute.  In ``σ``-SME this is the construction of ``W_σ``;
# here we'd do the analogous per-layer partial integration:
#
# * ``PartialIntegrate`` already exists (``zoomy_core.model.models.
#   ins_generator``) and can rewrite any ``∫_{z_i}^{z_{i+1}} f dz``
#   into a running integral ``∫_{z_i}^{z} f dz'`` — the right shape
#   to give us ``w(t, x, z)`` from the continuity primitive.
# * A ``ContinuityClosure(state, layer_lower, layer_lower_kbc)``
#   helper would package the steps: partially integrate the
#   depth-integrated continuity, pair with the lower-interface
#   ``w`` (from the KBC), produce the substitution
#   ``{w(t, x, z): ...}`` for the branch.
#
# **I have not implemented this closure** — it's the next honest
# piece of work before the shear-moment equation is self-contained.
# Our present walkthrough stops with the residual ``∫ w dζ`` terms
# *visible* in the output; see Step 11 above for the explicit `✗`
# markers where they appear.

# %% [markdown]
# ## Left for later
#
# * **Continuity closure for `w`** (described above) — required for
#   fully-closed multi-layer SME shear moments.
# * **`Recombine(vars=[t, x])`** display op — fold `∂_t(h·α) +
#   ∂_x(h·α²)` back into the conservative shape for readability.
# * **Explicit `sp.simplify` pass** on the final branch — the output
#   carries uncollapsed ratios like ``b² − 2bz_1 + z_1²`` (should
#   factor as ``(z_1 − b)²``); ``sp.simplify`` fixes this post-hoc
#   but we haven't wired it into the pipeline.
