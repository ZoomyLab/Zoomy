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
# # SWE and SME from scratch — symbolic walkthrough
#
# Derivation of the Shallow-Water Equations (SWE, level=0) and the Shallow
# Moment Equations (SME, level>0) directly from incompressible Navier-Stokes.
#
# Everything is mutation-based:
#
# * ``model.apply(op)``  — mutates every equation of the system, returns ``self``.
# * ``model.<eq>.apply(op).simplify()`` — chainable in-place mutation on a
#   single equation through the proxy.
# * ``model.<eq>.solve_for(var)``  — returns an Expression that ``apply``
#   consumes directly as ``{var: solution}``.
# * ``model.<eq>.remove()`` — drops an equation from the system.
#
# No ``model.equations[name] = model.equations[name].apply(...)`` pattern.
# No ``DepthIntegrate`` / ``HydrostaticPressure`` / ``ApplyKinematicBCs`` /
# ``StressFreeSurface`` / ``ZeroAtmosphericPressure`` / ``SimplifyIntegrals``
# shortcuts — everything goes through ``Integrate`` and substitution dicts.

# %% [markdown]
# ## Imports
#
# The path bootstrap below makes sure we import ``zoomy_core`` from the
# worktree we're actually sitting in, not from whatever editable install
# Python's site-packages happens to point at (different worktrees can be
# at different commits).

# %%
import sys
from pathlib import Path

# -- Pin zoomy_core to the current worktree -----------------------------------
#
# zoomy_core is installed as a PEP 660 editable package.  Its finder is
# registered on ``sys.meta_path`` with a hardcoded ``MAPPING`` that points
# at whichever worktree ``pip install -e`` was run from.  That finder
# runs BEFORE ``sys.path``, so a bare ``sys.path.insert`` is silently
# ignored.  We locate the enclosing worktree and rewrite the finder's
# MAPPING + clear any cached module so the next ``import zoomy_core``
# picks up THIS worktree's code.

_here = Path.cwd()
while _here != _here.parent and not (_here / "library" / "zoomy_core" / "zoomy_core").exists():
    _here = _here.parent
_pkg_dir   = _here / "library" / "zoomy_core"
_pkg_inner = _pkg_dir / "zoomy_core"
assert _pkg_inner.exists(), f"could not find library/zoomy_core/zoomy_core from {Path.cwd()}"

# Drop any cached modules so the next import is fresh.
for _k in list(sys.modules):
    if _k == "zoomy_core" or _k.startswith("zoomy_core."):
        del sys.modules[_k]

# Patch every editable finder on sys.meta_path whose module defines a
# MAPPING that mentions zoomy_core.  (Each zoomy_* package has its own
# finder; we only touch zoomy_core's.)
_patched = False
for _finder in sys.meta_path:
    _mod_name = getattr(_finder, "__module__", "") or ""
    _mod = sys.modules.get(_mod_name)
    _mapping = getattr(_mod, "MAPPING", None) if _mod is not None else None
    if isinstance(_mapping, dict) and "zoomy_core" in _mapping:
        _mapping["zoomy_core"] = str(_pkg_inner)
        _patched = True

# Belt-and-braces: also prepend to sys.path.
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import sympy as sp
import zoomy_core
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Integrate, Newtonian,
    Basis, Multiply, ZetaTransform, EvaluateIntegrals,
    ExpandProductRule,
)
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.sme_model import hydrostatic_scaling

print("worktree root:", _here)
print("zoomy_core.__file__:", zoomy_core.__file__)
print("editable MAPPING patched:", _patched)

# %% [markdown]
# ## Step 1 — Start from the raw Navier-Stokes system

# %%
state = StateSpace(dimension=2)          # (t, x, z)
model = FullINS(state)
model.describe()

# %% [markdown]
# ## Step 2 — Hydrostatic assumption on z-momentum
#
# $w = 0$, $\tau_{zz} = \tau_{xz} = \tau_{zx} = 0$ inside z-momentum only.
# Chained: ``apply(...)``→proxy, ``.simplify()``→proxy.

# %%
model.momentum.z.apply(hydrostatic_scaling(state),
                       name="hydrostatic scaling",
                       description="w = 0, tau_zz = tau_xz = tau_zx = 0 in z-momentum").simplify()
model.momentum.z.describe()

# %% [markdown]
# ## Step 3 — Integrate z-momentum analytically to get $p(z)$
#
# Integrate $g + \partial_z p / \rho = 0$ from the current depth $z$ up to the
# free surface $\eta$.  `method="analytical"` runs ``sympy.integrate`` on the
# whole expression (needed for partial / running integrals).

# %%
model.momentum.z.apply(
    Integrate(state.z, state.z, state.eta, method="analytical")
)
model.momentum.z.describe()

# %% [markdown]
# ## Step 4 — Atmospheric-pressure BC at the free surface
#
# $p(\eta) = 0$ (atmospheric gauge).  Plain substitution dict.

# %%
model.momentum.z.apply({state.p.subs(state.z, state.eta): 0},
                       name="atmospheric pressure",
                       description="p(t, x, eta) = 0").simplify()
model.momentum.z.describe()

# %% [markdown]
# ## Step 5 — Substitute the solved $p$ into x-momentum, then drop z-momentum

# %%
model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
model.momentum.z.remove()
model.describe()

# %% [markdown]
# ## Step 6 — Newtonian constitutive model
#
# System-level mutation: ``model.apply(op)`` runs the op on every equation
# of the system, returns ``self``, so ``.simplify()`` chains.
# The internal simplify is linearity-only (``d(b+h)/dx → db/dx + dh/dx``)
# and intentionally does **not** chain-rule-expand conservative forms like
# ``∂_x(u²)``, so they survive for the Leibniz integration in Step 7.

# %%
model.apply(Newtonian(state)).simplify()
model.describe()

# %% [markdown]
# ## Step 7 — Define the vertical basis and coefficients
#
# Pick a Legendre basis of ``LEVEL+1`` test functions.  We need the basis
# now, **before** depth-integration, so the Galerkin test function
# participates in the Leibniz rule — this is what produces the σ-SME
# coupling terms (the ``W_σ·∂_σ u`` equivalents).
#
# * ``basis.phi``     — test functions as sympy expressions in $\zeta$
#   (used after ``ZetaTransform``, inside integrands).
# * ``basis.phi_of_z``— same test functions with $\zeta = (z-b)/h$ so
#   they're explicit functions of ``z`` through ``b(t,x)``, ``h(t,x)``.
#   Used now to multiply the pointwise z-equation so the z-dependence
#   survives into the Leibniz rule.
# * ``basis.alpha``   — coefficient **Functions** $\alpha_k(t, x)$.

# %%
LEVEL = 1
basis = Basis(state, Legendre_shifted, level=LEVEL)
print("phi(zeta): ", [getattr(basis.phi,      f"phi_{k}") for k in range(LEVEL + 1)])
print("phi(z):    ", [getattr(basis.phi_of_z, f"phi_{k}") for k in range(LEVEL + 1)])
print("alpha:     ", [getattr(basis.alpha,    f"alpha_{k}") for k in range(LEVEL + 1)])

# %% [markdown]
# ## Step 8 — Galerkin test: $u$-momentum $\times \varphi_l((z-b)/h)$
#
# ``Multiply(basis.phi_of_z, outer=True)`` turns the single
# ``momentum.x`` leaf into a ``Zstruct(test_0, …, test_LEVEL)`` where
# each ``test_l`` is ``φ_l((z-b)/h) · (pointwise equation)``.  The test
# function is now a function of ``z`` through ``b``, ``h`` — it will
# interact with the Leibniz rule in Step 10.

# %%
model.momentum.x.apply(Multiply(basis.phi_of_z, outer=True))
model.momentum.describe()

# %% [markdown]
# ## Step 9 — Expand the product rule so Integrate can Leibniz
#
# Terms like $\varphi_l\bigl(\tfrac{z-b}{h}\bigr) \cdot \partial_x(u^2)$
# are *not* in conservative form: the coefficient depends on ``x``
# through ``b``, ``h``, so the default ``_extract_derivative`` check in
# ``Integrate(method='auto')`` refuses them (this is the same safety
# that protects us from double-counting chain rules).  ``ExpandProductRule``
# rewrites each such term as
#
# $$
#   \varphi_l \cdot \partial_v f = \partial_v(\varphi_l \cdot f) - \partial_v(\varphi_l) \cdot f
# $$
#
# The first piece is conservative (Leibniz applies); the second gives
# the non-conservative coupling — the ``∂_t b + ζ·∂_t h`` and
# ``∂_x b + ζ·∂_x h`` combinations that play the role of the
# σ-SME ``W_σ``.

# %%
model.momentum.x.apply(ExpandProductRule([state.t, state.x, state.z]),
                       name="expand product rule",
                       description="φ·∂_v(f) → ∂_v(φ·f) − ∂_v(φ)·f")

# %% [markdown]
# ## Step 9b — Snapshot the pointwise continuity
#
# Before we depth-integrate continuity (which reduces it to the scalar
# $\partial_t h + \partial_x(\alpha_0 h) = 0$), grab a detached copy of
# the pointwise form $\partial_x u + \partial_z w = 0$.  In Step 14 we
# run the running integral $\int_b^z$ on it and solve for $w$ to obtain
# the algebraic closure $w(z) = w|_b - \int_b^z \partial_x u\,dz'$ —
# without pre-baking the bottom KBC.

# %%
pointwise_continuity = model.continuity.copy()

# %% [markdown]
# ## Step 10 — Depth-integrate from $b$ to $\eta$
#
# One ``Integrate`` call at system level — per-term auto dispatch picks
# Leibniz for $\partial_x / \partial_t$ and the fundamental theorem for
# $\partial_z$.  Continuity integrates as a scalar (no test function);
# each ``momentum.x.test_l`` integrates with ``φ_l((z-b)/h)`` as its
# test weight.

# %%
model.apply(Integrate(state.z, state.b, state.eta, method="auto"))
model.continuity.describe()

# %% [markdown]
# ## Step 11 — Resolve $w$ boundary terms via the kinematic BCs

# %%
u_at_b = state.u.subs(state.z, state.b)
u_at_eta = state.u.subs(state.z, state.eta)
kinematic_bcs = {
    state.w.subs(state.z, state.b):
        sp.Derivative(state.b, state.t) + u_at_b * sp.Derivative(state.b, state.x),
    state.w.subs(state.z, state.eta):
        sp.Derivative(state.eta, state.t) + u_at_eta * sp.Derivative(state.eta, state.x),
}
model.apply(kinematic_bcs,
            name="kinematic BCs",
            description="w|_b, w|_eta via surface / bottom kinematic conditions").simplify()

# %% [markdown]
# ## Step 12 — Zero tangential stress at surface and bottom

# %%
stress_free_surface = {state.tau["xz"].subs(state.z, state.eta): 0}
no_tangential_normal_stress = {
    state.tau["xx"].subs(state.z, state.b): 0,
    state.tau["xx"].subs(state.z, state.eta): 0,
}
model.apply(stress_free_surface,
            name="stress-free surface",
            description="tau_xz|_eta = 0").apply(
    no_tangential_normal_stress,
    name="no tangential normal stress",
    description="tau_xx|_b = tau_xx|_eta = 0").simplify()

# %% [markdown]
# ## Step 13 — Bottom stress closure (Navier slip)

# %%
lamda = sp.Symbol("lamda", positive=True)
tau_c = sp.Symbol("tau_c", positive=True)
friction_closure = {
    state.tau["xz"].subs(state.z, state.b): state.rho * (lamda / tau_c) * u_at_b,
}
model.apply(friction_closure,
            name="Navier-slip friction",
            description="tau_xz|_b = rho * (lambda / tau_c) * u|_b").simplify()

# %% [markdown]
# ## Step 14 — Close the vertical velocity via continuity (pipeline)
#
# At this point the shear moment (``momentum.x.test_1``) still carries
# a bulk ``w(t, x, z)`` inside an ``Integral`` from the ``∂_z(u w)``
# non-conservative piece that ``ExpandProductRule`` left behind.
#
# The closure is built from the pointwise continuity snapshot we took
# at Step 9b — a small, readable pipeline over the existing primitives:
#
# 1. ``.apply(Integrate(z, b, z, "auto"))`` — running integral of
#    ``∂_x u + ∂_z w`` from the bottom to the current height.  The
#    ``∂_z w`` piece collapses by the fundamental theorem
#    (``w(t,x,z) − w(t,x,b)``); the ``∂_x u`` piece stays as a running
#    integral ``Integral(∂_x u, (z, b, z))``.
# 2. ``.solve_for(state.w)`` — isolate ``w(t,x,z)``, returning an
#    Expression whose ``_as_relation`` maps
#    ``w(t,x,z) → w|_b − ∫_b^z ∂_x u dz'``.
# 3. ``model.apply(w_closure)`` — replace bulk ``w(t,x,z)`` in every
#    equation with that relation.  ``w|_b`` is kept **symbolic** here,
#    decoupling the algebraic identity from the boundary condition.
# 4. Re-apply ``kinematic_bcs`` to close the newly introduced ``w|_b``.
#
# The inner running integral ``∫_b^z ∂_x u dz'`` is closed in Step 15
# once ``basis.expand`` substitutes ``u`` both pointwise and in its
# ζ-transformed form.

# %%
w_eq = pointwise_continuity.apply(
    Integrate(state.z, state.b, state.z, method="auto"),
)
w_closure = w_eq.solve_for(state.w)
model.apply(w_closure,
            name="w-closure from continuity",
            description="w(t,x,z) = w|_b - ∫_b^z ∂_x u dz'").simplify()
model.apply(kinematic_bcs,
            name="close w|_b (post w-closure)",
            description="close w|_b introduced by the w-closure").simplify()

# %% [markdown]
# ## Step 15 — Coordinate transform $z = \zeta\,h + b$
#
# Rewrites every $\int_b^{\eta} f(z)\,dz$ into
# $h\cdot\int_0^1 f(\zeta h + b)\,d\zeta$.

# %%
model.apply(ZetaTransform(state))

# %% [markdown]
# ## Step 16 — Substitute the basis expansion
#
# ``basis.expand`` now also supplies the pointwise substitution
# ``u(t, x, z) → Σ α_k · φ_k((z−b)/h)``, which closes the inner
# running integral produced by :class:`ContinuityClosure`.

# %%
model.apply(basis.expand(state.u))

# %% [markdown]
# ## Step 17 — Evaluate the $\zeta$-integrals
#
# Orthogonality kicks in because the test function $\varphi_l$ is now
# inside the ζ-integrand alongside ``u = Σ α_k φ_k(ζ)``.
# Cross-products ``∫ φ_l · φ_k dζ`` collapse to ``δ_{lk} / (2l+1)``
# (Legendre-shifted normalization), revealing the moment equations.

# %%
model.apply(EvaluateIntegrals(state)).simplify()
model.describe()

# %% [markdown]
# ## Where it lands and what's still open
#
# **Continuity — fully closed.**
# $$
#   \partial_t h + \partial_x(\alpha_0\,h) = 0
# $$
# The $\alpha_1$ volume integral vanishes by $\varphi_0, \varphi_1$
# orthogonality, so the depth-average moment is exactly the SWE
# continuity.
#
# **Momentum (test_0, test_1) — partially closed.**
# The clean SME moment equations are visible:
# temporal $\partial_t(h\alpha_k)$, convection $\alpha_i\alpha_j$-quadratic
# forms, hydrostatic pressure, and boundary viscous stress
# $\partial_x\alpha_k \cdot \partial_x\{b, h\}$ contributions.
#
# Two term classes remain unresolved:
#
# 1. **Volume $\int_0^1 \partial^2 u/\partial z^2 |_{z=\zeta h+b}\,d\zeta$**
#    — produced because ``Integrate(method="auto")`` at Step 7 couldn't
#    apply the fundamental theorem to second-order z-derivatives, so
#    the ``-\nu\,\partial^2 u/\partial z^2`` term stayed inside an
#    ``Integral``.  Closure options: integrate by parts inside the
#    integral (IBP in $z$) to reduce to ``\partial u/\partial z`` at the
#    boundaries, or express ``\partial u/\partial z`` directly in terms
#    of the basis ``\sum \alpha_k\,\varphi_k'(\zeta)/h``.
#
# 2. **Cross term $\int \partial^2 w/\partial x\partial z\,d\zeta$** —
#    arises from the Newtonian ``\tau_{xz}`` after z-integration.  Same
#    story: needs a boundary closure or an additional depth-integration
#    of the continuity equation to substitute $w(t,x,z)$.
#
# Both are *symbolic* — no numerical dispatch needed to resolve them.
# The cleanest follow-up is an ``IntegrateByParts`` operation that
# applies IBP to a specified ``Integral`` node, producing the
# ``\partial u/\partial z`` boundary traces which then close under the
# basis.
#
# At this point the tree is::
#
#     model
#     ├── continuity            (scalar — fully closed)
#     └── momentum
#         └── x
#             ├── test_0        (closed except 2 Integral classes above)
#             └── test_1        (same)
#
# so ``model.leaves()``, ``eq.tags``, ``eq.untagged`` and the usual
# ``.describe()`` all work; you can inspect and edit any leaf in place.

# %% [markdown]
# ## Derivation history — compact flowchart
#
# Every ``apply`` records a ``(name, description, target)`` entry in
# ``model.history``.  The default ``history_mermaid()`` renders a
# one-line-per-node left-to-right flowchart; hovering a node shows the
# ``target`` and ``description`` as a tooltip.

# %%
model.history_mermaid()

# %% [markdown]
# ## Derivation history — full table
#
# For the per-step detail (target / description / name) that the
# compact mermaid drops into tooltips, the companion table method
# lays it out in markdown.

# %%
model.history_table()

# %% [markdown]
# ## Optional — verbose / grouped views
#
# * ``verbose=True`` — legacy three-line node layout (name / target /
#   description).  Useful for offline diagrams when hover isn't
#   available.
# * ``group_by="target"`` — wraps consecutive same-target steps in
#   ``subgraph`` blocks so it's obvious which tree position each
#   phase acted on.
# * ``direction="TD"`` — top-to-bottom (default is ``LR``).
#
# Pick whichever fits the screen real estate.

# %%
model.history_mermaid(group_by="target", direction="TD")
