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
# # Performance walkthrough — per-step timing for four canonical derivations
#
# Four derivations side by side through the same symbolic pipeline, with a
# timer around every library operation so the per-step cost is visible.
#
# 1. **Single-layer SWE** — `LayeredBasis` with one layer `[b, η]`.
# 2. **Single-layer SME level 2** — `Legendre_shifted` level 2, Newtonian.
# 3. **Two-layer SWE** — `LayeredBasis` with two layers `[b, z_1, η]`.
# 4. **Two-layer SME level 2** — `LayeredBasis` + `Legendre_shifted` inner
#    level 2, Newtonian, per-layer Galerkin.
#
# Each cell times one library operation and prints
# ``[Δs | cumulative s] label`` so reading the notebook top-to-bottom feels
# the cost grow.  The ``PARTS`` dict at the end tabulates cumulative times
# per part.

# %% [markdown]
# ## Environment bootstrap

# %%
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_here = Path.cwd()
while _here != _here.parent and not (_here / "library" / "zoomy_core" / "zoomy_core").exists():
    _here = _here.parent
_pkg_dir = _here / "library" / "zoomy_core"
_pkg_inner = _pkg_dir / "zoomy_core"
assert _pkg_inner.exists(), f"could not find library/zoomy_core/zoomy_core from {Path.cwd()}"

for _k in list(sys.modules):
    if _k == "zoomy_core" or _k.startswith("zoomy_core."):
        del sys.modules[_k]

for _finder in sys.meta_path:
    _mod_name = getattr(_finder, "__module__", "") or ""
    _mod = sys.modules.get(_mod_name)
    _mapping = getattr(_mod, "MAPPING", None) if _mod is not None else None
    if isinstance(_mapping, dict) and "zoomy_core" in _mapping:
        _mapping["zoomy_core"] = str(_pkg_inner)
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import sympy as sp
import zoomy_core
from zoomy_core.model.models.ins_generator import (
    Basis, EvaluateIntegrals, Expression, FullINS, Integrate, InterfaceKBC,
    Inviscid, Multiply, Newtonian, ProductRule, Recombine, SimplifyIntegrals,
    StateSpace, ZetaTransform,
)
from zoomy_core.model.derivation.basisfunctions import LayeredBasis, Legendre_shifted
from zoomy_core.model.models.sme_model import hydrostatic_scaling

print("zoomy_core:", zoomy_core.__file__)

# %% [markdown]
# ## Timing helper
#
# ``with timed("label"):`` wraps one library operation.  Each call appends to
# ``LOG`` and prints ``[Δs | cumulative s] label`` so the reader can feel the
# cost of every step as they scroll through the notebook.

# %%
LOG = []           # list of (part, label, dt_s, cumulative_s)
_PART = [""]       # current part name (mutable holder so @contextmanager can read it)
_START = [time.time()]
_LAST = [time.time()]

def set_part(name):
    _PART[0] = name

@contextmanager
def timed(label):
    t0 = time.time()
    yield
    dt = time.time() - t0
    total = time.time() - _START[0]
    LOG.append((_PART[0], label, dt, total))
    _LAST[0] = time.time()
    print(f"  [{dt:6.2f}s | {total:7.2f}s total] {label}")


def reset_clock():
    _START[0] = time.time()
    _LAST[0] = time.time()


# %% [markdown]
# # Part 0 — Extended general derivation
#
# Before branching into specific models, take two shared structural steps
# that are usually buried inside each per-model pipeline:
#
# 1. **Eliminate `w` from `x`-momentum** via the w-closure.  Integrate
#    pointwise continuity `∂_x u + ∂_z w = 0` from `b` to `z` to get
#    `w(z) = w|_b − ∫_b^z ∂_x u dz'`; substitute that into x-momentum so
#    the bulk `w` disappears.  The pointwise continuity itself is kept as a
#    separate `Expression` (in case later closures need the full constraint).
# 2. **Derive the h-evolution (mass balance) equation** from continuity.
#    Depth-integrate continuity from `b` to `η` via Leibniz / FTC, apply
#    the kinematic BCs at both interfaces — the `u·∂_x(…)` boundary terms
#    telescope against the KBC-produced surface contributions and what's
#    left is the clean `∂_t h + ∂_x ∫_b^η u dz = 0`.  This equation then
#    *replaces* the in-place `continuity` leaf on the system; the
#    pointwise form is held aside as ``pointwise_continuity``.
#
# Subsequent per-model parts below each restart with a fresh `FullINS` for
# comparability — they don't consume this shared state.  But the timing
# below is what a model-agnostic pre-processor would cost once, and is
# directly subtractable from each per-model Part's cost.

# %%
set_part("0. Extended general prefix")
reset_clock()
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z

with timed("FullINS + hydrostatic + p-sub + drop z-momentum + Newtonian"):
    model = FullINS(state)
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
    model.momentum.z.remove()
    model.apply(Newtonian(state)).simplify()

# %% [markdown]
# ### Eliminate `w` via the w-closure (shared)

# %%
with timed("snapshot pointwise continuity"):
    pointwise_continuity = model.continuity.copy()
with timed("∫_{b}^{z} on continuity + solve_for(w)"):
    w_eq = pointwise_continuity.apply(Integrate(z, state.b, z, method="auto"))
    w_closure = w_eq.solve_for(state.w)
with timed("apply(w_closure) on x-momentum + simplify"):
    model.momentum.x.apply(w_closure).simplify()

# %% [markdown]
# ### Derive h-evolution from mass balance (shared)

# %%
with timed("∫_{b}^{η} on continuity"):
    model.continuity.apply(Integrate(z, state.b, state.eta, method="auto"))

u_at_b = state.u.subs(z, state.b)
u_at_eta = state.u.subs(z, state.eta)
kinematic_bcs = {
    state.w.subs(z, state.b):
        sp.Derivative(state.b, t) + u_at_b * sp.Derivative(state.b, x),
    state.w.subs(z, state.eta):
        sp.Derivative(state.eta, t) + u_at_eta * sp.Derivative(state.eta, x),
}
with timed("apply kinematic BCs on continuity → h-evolution"):
    model.continuity.apply(kinematic_bcs).simplify()
with timed("apply kinematic BCs on x-momentum → close w|_b"):
    model.momentum.x.apply(kinematic_bcs).simplify()

print()
print("=== Part 0 results ===")
print("pointwise continuity (kept aside):")
print("   ", pointwise_continuity.expr)
print()
print("h-evolution (now on the system's continuity leaf):")
print("   ", model.continuity._node.expr)
print()
print("x-momentum (w eliminated):")
print("   ", model.momentum.x._node.expr)


# %% [markdown]
# # Part 1 — Single-layer SWE
#
# The classical shallow-water equations via `LayeredBasis` with one layer
# `[b, η]`.  Constant vertical profile (level 0), inviscid closure.
#
# Restarts from a fresh `FullINS` for comparability with the other parts.

# %%
set_part("1. SWE single-layer")
reset_clock()
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z

with timed("FullINS(state)"):
    model = FullINS(state)

# %% [markdown]
# ### Common inviscid prefix

# %%
with timed("hydrostatic_scaling on z-momentum"):
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
with timed("∫ p analytical, z-momentum"):
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
with timed("p(η)=0 atmospheric BC"):
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
with timed("substitute p into x-momentum"):
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
with timed("drop z-momentum"):
    model.momentum.z.remove()
with timed("Inviscid(state)"):
    model.apply(Inviscid(state)).simplify()

# %% [markdown]
# ### Depth integrate + close

# %%
with timed("Basis(LayeredBasis, [b, η])"):
    swe_basis = Basis(state, LayeredBasis, interfaces=[state.b, state.eta])
with timed("∫_{b}^{η} dz on system"):
    model.apply(Integrate(z, state.b, state.eta, method="auto"))
with timed("InterfaceKBC bottom"):
    model.apply(InterfaceKBC(state, state.b)).simplify()
with timed("InterfaceKBC surface"):
    model.apply(InterfaceKBC(state, state.eta)).simplify()
with timed("layer_expand(u, layer=0)"):
    model.apply(swe_basis.layer_expand(state.u, 0)).simplify()
with timed("SimplifyIntegrals"):
    model.apply(SimplifyIntegrals(state)).simplify()

# %% [markdown]
# ### Rename `α_0` → `hu/h` and Recombine
# Pure cosmetic — gives the canonical SWE flux display.

# %%
alpha = swe_basis.alpha.alpha_0
hu_fn = sp.Function("hu", real=True)(t, x)
with timed("rename α_0 → hu/h"):
    model.apply({alpha: hu_fn / state.H})
with timed("Recombine([t, x])"):
    model.apply(Recombine(vars=[t, x]))

print()
print("=== Part 1 final shape ===")
print("continuity:", model.continuity._node.expr)
print("momentum.x:", model.momentum.x._node.expr)


# %% [markdown]
# # Part 2 — Single-layer SME level 2
#
# Viscous Newtonian, Legendre_shifted vertical basis at level 2.  Introduces
# the Galerkin test-multiply + product-rule + w-closure-via-pipeline chain.

# %%
set_part("2. SME single-layer level=2")
reset_clock()
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
with timed("FullINS(state)"):
    model = FullINS(state)

# %% [markdown]
# ### Prefix

# %%
with timed("hydrostatic_scaling on z-momentum"):
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
with timed("∫ p analytical, z-momentum"):
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
with timed("p(η)=0 atmospheric BC"):
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
with timed("substitute p into x-momentum"):
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
with timed("drop z-momentum"):
    model.momentum.z.remove()
with timed("Newtonian(state)"):
    model.apply(Newtonian(state)).simplify()

# %% [markdown]
# ### Galerkin test + product rule

# %%
with timed("Basis(Legendre_shifted, level=2)"):
    basis = Basis(state, Legendre_shifted, level=2)
with timed("Multiply(phi_of_z, outer=True)"):
    model.momentum.x.apply(Multiply(basis.phi_of_z, outer=True))
with timed("ProductRule() (inverse)"):
    model.momentum.x.apply(ProductRule())
with timed("pointwise_continuity snapshot"):
    pointwise_continuity = model.continuity.copy()

# %% [markdown]
# ### Depth-integrate + KBCs + stress closures

# %%
with timed("∫_{b}^{η} dz on system"):
    model.apply(Integrate(z, state.b, state.eta, method="auto"))

# Inline kinematic BCs (functional form — what the library _resolve_subs_safe
# produces after Integrate).
u_at_b = state.u.subs(z, state.b)
u_at_eta = state.u.subs(z, state.eta)
kinematic_bcs = {
    state.w.subs(z, state.b):
        sp.Derivative(state.b, t) + u_at_b * sp.Derivative(state.b, x),
    state.w.subs(z, state.eta):
        sp.Derivative(state.eta, t) + u_at_eta * sp.Derivative(state.eta, x),
}
with timed("apply kinematic_bcs + simplify"):
    model.apply(kinematic_bcs).simplify()

with timed("stress-free surface + no-tangential-normal-stress"):
    model.apply({state.tau["xz"].subs(z, state.eta): 0}).apply(
        {state.tau["xx"].subs(z, state.b): 0,
         state.tau["xx"].subs(z, state.eta): 0}).simplify()

lamda = sp.Symbol("lamda", positive=True)
tau_c = sp.Symbol("tau_c", positive=True)
with timed("Navier-slip friction closure"):
    model.apply({state.tau["xz"].subs(z, state.b):
                 state.rho * (lamda / tau_c) * u_at_b}).simplify()

# %% [markdown]
# ### w-closure pipeline + re-apply KBCs

# %%
with timed("w-closure: running ∫_{b}^{z}"):
    w_eq = pointwise_continuity.apply(Integrate(z, state.b, z, method="auto"))
    w_closure = w_eq.solve_for(state.w)
with timed("apply(w_closure) + simplify"):
    model.apply(w_closure).simplify()
with timed("re-apply kinematic_bcs (close w|_b)"):
    model.apply(kinematic_bcs).simplify()

# %% [markdown]
# ### ζ-transform + basis expand + EvaluateIntegrals

# %%
with timed("ZetaTransform(state)"):
    model.apply(ZetaTransform(state))
with timed("basis.expand(state.u)"):
    model.apply(basis.expand(state.u))
with timed("EvaluateIntegrals + simplify"):
    model.apply(EvaluateIntegrals(state)).simplify()

print()
print("=== Part 2 final: leaf term counts ===")
for path, eq in model.leaves():
    print(f"  {'.'.join(path):30s} {len(sp.Add.make_args(sp.expand(eq.expr)))} terms")


# %% [markdown]
# # Part 3 — Two-layer SWE
#
# Layered basis with interfaces `[b, z_1, η]`, per-layer level-0 close.

# %%
set_part("3. SWE two-layer")
reset_clock()
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
z_1 = sp.Function("z_1", real=True)(t, x)
m_1 = sp.Function("m_1", real=True)(t, x)   # interface mass flux

with timed("FullINS(state) + prefix"):
    model = FullINS(state)
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
    model.momentum.z.remove()
    model.apply(Inviscid(state)).simplify()

with timed("Basis(LayeredBasis, [b, z_1, η])"):
    swe_ml_basis = Basis(state, LayeredBasis,
                         interfaces=[state.b, z_1, state.eta])

# %% [markdown]
# ### Layer 0

# %%
with timed("branch layer 0"):
    layer_0 = model.branch(name="layer 0 SWE")

with timed("layer 0 ∫_{b}^{z_1} dz"):
    layer_0.apply(Integrate(z, state.b, z_1, method="auto"))
with timed("layer 0 InterfaceKBC(b)"):
    layer_0.apply(InterfaceKBC(state, state.b)).simplify()
with timed("layer 0 InterfaceKBC(z_1, m_1)"):
    layer_0.apply(InterfaceKBC(state, z_1, mass_flux=m_1)).simplify()
with timed("layer 0 layer_expand(u, 0)"):
    layer_0.apply(swe_ml_basis.layer_expand(state.u, 0)).simplify()
with timed("layer 0 SimplifyIntegrals"):
    layer_0.apply(SimplifyIntegrals(state)).simplify()

# %% [markdown]
# ### Layer 1

# %%
with timed("branch layer 1"):
    layer_1 = model.branch(name="layer 1 SWE")

with timed("layer 1 ∫_{z_1}^{η} dz"):
    layer_1.apply(Integrate(z, z_1, state.eta, method="auto"))
with timed("layer 1 InterfaceKBC(z_1, m_1)"):
    layer_1.apply(InterfaceKBC(state, z_1, mass_flux=m_1)).simplify()
with timed("layer 1 InterfaceKBC(η)"):
    layer_1.apply(InterfaceKBC(state, state.eta)).simplify()
with timed("layer 1 layer_expand(u, 1)"):
    layer_1.apply(swe_ml_basis.layer_expand(state.u, 1)).simplify()
with timed("layer 1 SimplifyIntegrals"):
    layer_1.apply(SimplifyIntegrals(state)).simplify()


# %% [markdown]
# # Part 4 — Two-layer SME level 2
#
# Two layers, each with a level-2 Legendre_shifted inner basis.  This is the
# heaviest case — runs the full Galerkin + product-rule + w-closure + ζ-
# transform + EvaluateIntegrals chain per layer on the largest expressions.

# %%
set_part("4. SME two-layer level=2")
reset_clock()
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
z_1 = sp.Function("z_1", real=True)(t, x)
m_1 = sp.Function("m_1", real=True)(t, x)

with timed("FullINS(state) + Newtonian prefix"):
    model = FullINS(state)
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
    model.momentum.z.remove()
    model.apply(Newtonian(state)).simplify()

with timed("Basis(LayeredBasis, Legendre inner level=2)"):
    sme_ml_basis = Basis(state, LayeredBasis,
                         inner_cls=Legendre_shifted, inner_level=2,
                         interfaces=[state.b, z_1, state.eta])

print(f"  SME basis total level: {sme_ml_basis.level}  (2 layers × 3 moments − 1)")


# %% [markdown]
# ### Layer 0 SME pipeline

# %%
set_part("4. SME two-layer level=2 (layer 0)")
with timed("layer 0 branch + Multiply(layer_phi_of_z(0))"):
    layer_0 = model.branch(name="layer 0 SME2")
    layer_0.momentum.x.apply(
        Multiply(sme_ml_basis.layer_phi_of_z(0), outer=True))
with timed("layer 0 ProductRule()"):
    layer_0.momentum.x.apply(ProductRule())
with timed("layer 0 pointwise_continuity snapshot"):
    pointwise_continuity_0 = layer_0.continuity.copy()
with timed("layer 0 ∫_{b}^{z_1} dz on system"):
    layer_0.apply(Integrate(z, state.b, z_1, method="auto"))
with timed("layer 0 InterfaceKBC(b)"):
    layer_0.apply(InterfaceKBC(state, state.b)).simplify()
with timed("layer 0 InterfaceKBC(z_1, m_1)"):
    layer_0.apply(InterfaceKBC(state, z_1, mass_flux=m_1)).simplify()
with timed("layer 0 w-closure"):
    w_eq = pointwise_continuity_0.apply(
        Integrate(z, state.b, z, method="auto"))
    w_closure = w_eq.solve_for(state.w)
    layer_0.apply(w_closure).simplify()
with timed("layer 0 re-close w|_b via InterfaceKBC(b)"):
    layer_0.apply(InterfaceKBC(state, state.b)).simplify()
with timed("layer 0 ZetaTransform(b, z_1)"):
    layer_0.apply(ZetaTransform(state, lower=state.b, upper=z_1))
with timed("layer 0 layer_expand(u, 0, ζ-transformed)"):
    layer_0.apply(sme_ml_basis.layer_expand(
        state.u, 0, zeta_transformed=True))
with timed("layer 0 layer_expand(u, 0, pointwise)"):
    layer_0.apply(sme_ml_basis.layer_expand(
        state.u, 0, zeta_transformed=False))
with timed("layer 0 EvaluateIntegrals + simplify"):
    layer_0.apply(EvaluateIntegrals(state)).simplify()

print()
print("=== Layer 0 SME leaf term counts ===")
for path, eq in layer_0.leaves():
    print(f"  {'.'.join(path):30s} {len(sp.Add.make_args(sp.expand(eq.expr)))} terms")


# %% [markdown]
# ### Layer 1 SME pipeline

# %%
set_part("4. SME two-layer level=2 (layer 1)")
with timed("layer 1 branch + Multiply(layer_phi_of_z(1))"):
    layer_1 = model.branch(name="layer 1 SME2")
    layer_1.momentum.x.apply(
        Multiply(sme_ml_basis.layer_phi_of_z(1), outer=True))
with timed("layer 1 ProductRule()"):
    layer_1.momentum.x.apply(ProductRule())
with timed("layer 1 pointwise_continuity snapshot"):
    pointwise_continuity_1 = layer_1.continuity.copy()
with timed("layer 1 ∫_{z_1}^{η} dz on system"):
    layer_1.apply(Integrate(z, z_1, state.eta, method="auto"))
with timed("layer 1 InterfaceKBC(z_1, m_1)"):
    layer_1.apply(InterfaceKBC(state, z_1, mass_flux=m_1)).simplify()
with timed("layer 1 InterfaceKBC(η)"):
    layer_1.apply(InterfaceKBC(state, state.eta)).simplify()
with timed("layer 1 w-closure"):
    w_eq = pointwise_continuity_1.apply(
        Integrate(z, z_1, z, method="auto"))
    w_closure = w_eq.solve_for(state.w)
    layer_1.apply(w_closure).simplify()
with timed("layer 1 re-close w|_{z_1} via InterfaceKBC(z_1, m_1)"):
    layer_1.apply(InterfaceKBC(state, z_1, mass_flux=m_1)).simplify()
with timed("layer 1 ZetaTransform(z_1, η)"):
    layer_1.apply(ZetaTransform(state, lower=z_1, upper=state.eta))
with timed("layer 1 layer_expand(u, 1, ζ-transformed)"):
    layer_1.apply(sme_ml_basis.layer_expand(
        state.u, 1, zeta_transformed=True))
with timed("layer 1 layer_expand(u, 1, pointwise)"):
    layer_1.apply(sme_ml_basis.layer_expand(
        state.u, 1, zeta_transformed=False))
with timed("layer 1 EvaluateIntegrals + simplify"):
    layer_1.apply(EvaluateIntegrals(state)).simplify()

print()
print("=== Layer 1 SME leaf term counts ===")
for path, eq in layer_1.leaves():
    print(f"  {'.'.join(path):30s} {len(sp.Add.make_args(sp.expand(eq.expr)))} terms")


# %% [markdown]
# ## Summary — top per-step costs across all four parts

# %%
print()
print(f"{'part':40s} {'Δs':>8s}  {'label'}")
print("-" * 90)
slow = sorted(LOG, key=lambda r: -r[2])[:20]
for part, label, dt, total in slow:
    print(f"{part:40s} {dt:8.2f}  {label}")

print()
print(f"{'part':40s} {'total_s':>10s}")
print("-" * 60)
from itertools import groupby
by_part = {}
for part, label, dt, total in LOG:
    by_part.setdefault(part, 0.0)
    by_part[part] += dt
for part, total in by_part.items():
    print(f"{part:40s} {total:10.2f}")
