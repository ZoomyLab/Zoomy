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
# # Unified symbolic-derivation walkthrough — SWE, SME, ML-SWE, and VAM
#
# One document that derives the four canonical depth-reduced models in
# Zoomy from Navier-Stokes, side by side, using only the symbolic
# pipeline in `zoomy_core.model.models.ins_generator`.
#
# The goal is to show where **the same library call** drives
# different derivations, and — just as important — to flag the places
# where a lot of code exists inside the library to catch a single
# class of mistake.  Every such trap is called out inline at the step
# that exposes it, with a pointer to the library code that handles
# it.
#
# ## Structure
#
# * **Part 0** — environment setup (PEP 660 editable-install patch).
# * **Part 1** — common prefix shared by SWE, SME, and ML-SWE:
#   hydrostatic scaling, analytic z-integration, p-substitution,
#   z-momentum removal.
# * **Part 2** — **SWE** single-layer.  Applied to the Inviscid branch
#   of the common prefix; closed with a `LayeredBasis` of one layer.
#   Reconstruction matches the canonical 1-D SWE with topography.
# * **Part 3** — **SME** single-layer, level-1 Legendre, with
#   Newtonian viscosity.  Introduces the pipeline-style w-closure
#   (`copy + Integrate + solve_for + apply + InterfaceKBC`) and
#   documents the one known residual.
# * **Part 4** — **ML-SWE** two layers, with mass flux through the
#   internal interface.  Branches off the same common prefix.
# * **Part 5** — **VAM**, derived *from scratch* because it keeps
#   z-momentum and promotes `w` to its own moment coefficients.
#
# Difficulties and open questions are embedded in the step they
# first appear in — there's no appendix.  Cross-references take the
# form `file:line` so the reader can jump to the library source.


# %% [markdown]
# ## Part 0 — Environment setup
#
# `zoomy_core` is installed as a PEP 660 editable package.  Its
# finder on `sys.meta_path` holds a `MAPPING` dict pointing at
# whichever worktree `pip install -e` was run from.  That finder
# runs **before** `sys.path`, so a bare `sys.path.insert` is
# silently ignored.  We rewrite the finder's `MAPPING` and drop any
# cached `zoomy_core` modules so the next `import zoomy_core` picks
# up **this** worktree's code.  The same bootstrap lives at the top
# of every tutorial in `tutorials/sme/`.
#
# **Difficulty caught here:** without this patch, running the
# tutorial from worktree A while a different checkout is installed
# editably would silently import the *wrong* library version — every
# subsequent `apply(...)` would dispatch to the wrong class.  The
# bug would only surface as a structural mismatch several steps
# later, nowhere near its cause.

# %%
import sys
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

_patched = False
for _finder in sys.meta_path:
    _mod_name = getattr(_finder, "__module__", "") or ""
    _mod = sys.modules.get(_mod_name)
    _mapping = getattr(_mod, "MAPPING", None) if _mod is not None else None
    if isinstance(_mapping, dict) and "zoomy_core" in _mapping:
        _mapping["zoomy_core"] = str(_pkg_inner)
        _patched = True
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

import sympy as sp
import zoomy_core
from zoomy_core.model.models.ins_generator import (
    Basis,
    EvaluateIntegrals,
    FullINS,
    Integrate,
    InterfaceKBC,
    Inviscid,
    Multiply,
    Newtonian,
    ProductRule,
    Recombine,
    SimplifyIntegrals,
    StateSpace,
    ZetaTransform,
)
from zoomy_core.model.models.basisfunctions import (
    LayeredBasis,
    Legendre_shifted,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling
from zoomy_core.model.models.system_model import SystemModel

print("worktree root:", _here)
print("zoomy_core.__file__:", zoomy_core.__file__)
print("editable MAPPING patched:", _patched)


# %% [markdown]
# ## Part 1 — Common inviscid prefix
#
# Three of the four models (SWE, SME, ML-SWE) start from **the same
# six-step prefix**: Navier-Stokes → hydrostatic z-momentum →
# analytic z-integration → `p(η)=0` surface BC → substitute `p` into
# x-momentum → drop z-momentum.  What differs between them is only
# what happens *after* the prefix.
#
# We derive the prefix **once**, then take three independent copies
# via `model.branch(name=...)` and specialize them.
#
# ### Step 1.1 — `FullINS(state)`
#
# `StateSpace(dimension=2)` gives us the `(t, x, z)` coordinate set
# plus the physical fields `u(t,x,z)`, `w(t,x,z)`, `p(t,x,z)`,
# `tau_ij(t,x,z)`, the bathymetry `b(t,x)`, the free-surface
# `eta(t,x)` and the depth `h = eta - b`.
#
# `FullINS(state)` constructs the incompressible Navier-Stokes
# system as a `System`: one `continuity` leaf and two `momentum.{x,z}`
# leaves, all in conservative form.  Nothing is evaluated yet — the
# whole thing is a symbolic AST.

# %%
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
model = FullINS(state)
model.name = "Navier-Stokes"
print("initial leaves:")
for path, eq in model.leaves():
    print(f"  {'.'.join(path):15s} {eq.expr}")


# %% [markdown]
# ### Step 1.2 — Hydrostatic scaling on z-momentum
#
# `hydrostatic_scaling(state)` is an `Assumption` (a `Relation`) that
# zeroes out `w`, `τ_zz`, `τ_xz`, `τ_zx` **inside z-momentum only**
# — so after this, z-momentum becomes `∂_z p / ρ + g = 0`.
#
# **Difficulty caught here.**  The trailing `.simplify()` uses the
# linearity-only `_simplify_preserve_integrals`
# (`ins_generator.py:~3188`).  If it were a plain `sp.simplify`
# instead, conservative shapes like `Derivative(u², x)` would
# chain-rule-expand to `2·u·∂_x u` and the later
# `Integrate(method="auto")` dispatch would refuse them (Leibniz
# wants a single outer Derivative, not a product of a field and its
# derivative).  The library's simplify distributes Derivatives over
# Adds but never calls `.doit()` on them — see
# `_evaluate_linear_derivatives` at `ins_generator.py:~3282`.

# %%
model.momentum.z.apply(
    hydrostatic_scaling(state),
    name="hydrostatic scaling",
    description="w = 0, tau_zz = tau_xz = tau_zx = 0 in z-momentum",
).simplify()
print("post-hydrostatic z-momentum:", model.momentum.z._node.expr)


# %% [markdown]
# ### Step 1.3 — Integrate z-momentum from `z` to `η`
#
# We solve for `p(z)` by integrating `∂_z p / ρ + g = 0` from the
# current depth `z` upward to the free surface `η`.
#
# **Difficulty caught here.**  `method="analytical"` means "run
# `sympy.integrate` on the whole expression", which is only safe if
# the integrand has no *external* horizontal derivatives.  The
# default `method="auto"` we use later is per-term, but for a pure
# z-momentum equation with no horizontal-derivative clutter the
# whole-expression variant is both faster and cleaner.
#
# **Sub-difficulty:** the library's `Integrate._apply_leaf` does a
# `sp.Dummy` substitution on `var` before handing to `sp.integrate`.
# That's because the `upper == var` case (a running integral,
# `∫_lo^z f dz`) would otherwise clash the bound variable with its
# own upper bound and produce nonsense.  We don't need that here
# — `upper = η ≠ z` — but the guard is always on.  See
# `ins_generator.py:~1932`.

# %%
model.momentum.z.apply(
    Integrate(z, z, state.eta, method="analytical"),
    name="integrate z-momentum",
    description="analytic z-integration to get p(z)",
)
print("post-integrate z-momentum:", model.momentum.z._node.expr)


# %% [markdown]
# ### Step 1.4 — Atmospheric pressure BC at the free surface
#
# The surface boundary condition `p(t, x, η) = 0` (atmospheric
# gauge) collapses the evaluated-at-η term.  We pass it as a raw
# Python dict — `Relation`-shaped, but Zoomy accepts both.
#
# **Difficulty caught here.**  `Expression.apply` handles plain
# dicts through `xreplace`, not `subs`.  `.subs` is aggressive about
# sympy's "dummy dependency" guard and can refuse to substitute
# `p(t, x, η)` inside a `Subs` wrapper just because some later node
# mentions the integration variable.  `xreplace` is purely
# structural — it substitutes exactly what you ask for — which is
# what we want when the key is a `Subs`-derived form and the value
# is a constant.  See `Expression.apply` at
# `ins_generator.py:~620`.

# %%
model.momentum.z.apply(
    {state.p.subs(z, state.eta): 0},
    name="atmospheric pressure",
    description="p(t, x, eta) = 0",
).simplify()
print("post-atm pressure z-momentum:", model.momentum.z._node.expr)


# %% [markdown]
# ### Step 1.5 — Substitute solved `p` into x-momentum
#
# `model.momentum.z.solve_for(state.p)` is **a proxy method** on
# `_NodeProxy` (`derived_system.py:~286`) — it mutates the
# z-momentum leaf so that its equation is now `p = <solution>`.
# That mutated leaf, when passed to `model.momentum.x.apply(...)`,
# is treated as a relation (`_as_relation = {p: ...}`), and the
# substitution flows into x-momentum's pressure terms.
#
# **Difficulty caught here.**  There are *two* `solve_for` methods
# in the library.  The proxy one here mutates a tree node.  In
# Part 3 we'll use `Expression.solve_for` (`ins_generator.py:~707`),
# which is **non-mutating** and returns a new Expression with an
# `_as_relation` attached.  That's the form the pipeline-style
# w-closure needs.

# %%
model.momentum.x.apply(
    model.momentum.z.solve_for(state.p),
    name="substitute p(t, x, z)",
).simplify()


# %% [markdown]
# ### Step 1.6 — Drop z-momentum
#
# After substituting `p` into x-momentum, z-momentum has done its
# job.  We remove it from the tree to avoid double-accounting.
#
# At this point `model` is exactly the shared prefix — one
# continuity leaf, one x-momentum leaf with the solved pressure
# folded in.  All three downstream derivations branch from **here**.

# %%
model.momentum.z.remove()
print("shared-prefix leaves:")
for path, eq in model.leaves():
    print(f"  {'.'.join(path):15s}  n_terms={len(eq)}")


# %% [markdown]
# ## Part 2 — Single-layer SWE
#
# Branch off the common prefix, apply `Inviscid` to kill the
# viscous stress tensor, and close via a one-layer `LayeredBasis`.
# The extracted flux / NC / source triple feeds back into the
# numeric solver through `SystemModel`.

# %%
swe = model.branch(name="SWE")
swe.apply(Inviscid(state), name="Inviscid (Euler)").simplify()


# %% [markdown]
# ### Step 2.1 — Define the vertical basis
#
# `LayeredBasis` with `interfaces=[b, η]` means: one layer, spanning
# the whole depth, one coefficient `α_0(t, x)` which after
# renaming will become `hu/h`.

# %%
basis = Basis(state, LayeredBasis, interfaces=[state.b, state.eta])


# %% [markdown]
# ### Step 2.2 — Depth-integrate the whole system
#
# One `Integrate` call with `method="auto"` — per-term dispatch.
#
# **Difficulty caught here.**  `method="auto"` inspects each term
# for the *outermost* derivative direction and picks one of three
# strategies:
#
# * `∂_z f` → fundamental theorem: `∫_b^η ∂_z f dz = f|_η − f|_b`.
# * `∂_x f` (or `∂_t f`) → Leibniz rule:
#   `∫_b^η ∂_x f dz = ∂_x[∫_b^η f dz] − f|_η·∂_x η + f|_b·∂_x b`.
# * no derivative → `direct`: keep as unevaluated
#   `Integral(f, (z, b, η))` for `SimplifyIntegrals` to collapse
#   later if the integrand is `z`-independent.
#
# The dispatch lives in `Expression.depth_integrate`
# (`ins_generator.py:~720`).  A uniform `method="leibniz"` applied
# to every term would be wrong whenever `∂_z` appears — we'd get
# the wrong boundary terms and miss cancellations.  The per-term
# dispatch is load-bearing.

# %%
swe.apply(Integrate(z, state.b, state.eta, method="auto"),
          name="depth-integrate [b, eta]")


# %% [markdown]
# ### Step 2.3 — Kinematic BCs at bottom and free surface
#
# `InterfaceKBC(state, interface)` is a single class
# (`ins_generator.py:~2668`) that produces the right-hand side for
# `w(t, x, interface)`.  For the bottom (`b`) and the free surface
# (`η`), the standard kinematic condition
# `w = ∂_t interface + u·∂_x interface` applies directly.  In the
# multi-layer case (Part 4) we'll also pass `mass_flux=m_i` to
# close an internal interface.

# %%
swe.apply(InterfaceKBC(state, state.b)).simplify()
swe.apply(InterfaceKBC(state, state.eta)).simplify()


# %% [markdown]
# ### Step 2.4 — Layer-expand `u` → `α_0`
#
# `basis.layer_expand(state.u, 0)` returns a substitution dict that
# rewrites every appearance of `u(t, x, z)` (pointwise, evaluated at
# interfaces, and inside any remaining integrals) into layer-0's
# coefficient `α_0(t, x)`.  Four keys fire in sequence: pointwise,
# ζ-transformed, bottom eval, surface eval — see
# `ins_generator.py:~2960`.

# %%
swe.apply(basis.layer_expand(state.u, 0),
          name="close layer 0",
          description="u -> alpha_0").simplify()


# %% [markdown]
# ### Step 2.5 — Collapse constant `z`-integrals
#
# After the layer expansion, every remaining `Integral(c, (z, b, η))`
# has a `z`-independent integrand and should collapse to `c·(η−b) =
# c·h`.  `SimplifyIntegrals` walks the tree and does exactly that.

# %%
swe.apply(SimplifyIntegrals(state)).simplify()


# %% [markdown]
# ### Step 2.6 — Rename `α_0` to `hu/h` and `Recombine`
#
# The primitive coefficient `α_0` has units of velocity; for
# conservative SWE we evolve momentum `hu = α_0 · h`.  A single
# substitution renames it throughout; `Recombine(vars=[t, x])`
# collapses leftover bound integrals and fans out any remaining
# `Derivative(f, ...)` so the final expression reads cleanly in
# `(t, x)`-only form.

# %%
alpha = basis.alpha.alpha_0
hu_fn = sp.Function("hu", real=True)(t, x)
swe.apply({alpha: hu_fn / state.H})
swe.apply(Recombine(vars=[t, x]))

print("SWE continuity :", swe.continuity._node.expr)
print("SWE momentum.x :", swe.momentum.x._node.expr)


# %% [markdown]
# ### Step 2.7 — Extract flux / NC / source via `SystemModel`
#
# `SystemModel` is a **thin reader**.  Given a closed System and a
# state-variable ordering, it calls `auto_solver_tag` per leaf
# (`ins_generator.py:~375`) and then `collect_solver_tag("flux", ...)`
# etc. to produce `ZArray`s the numerical solver consumes
# (`tag_extraction.py:~74`).  **No physics lives in the adapter** —
# the tests literally grep its source for things like `g*h**2/2`
# as a regression guard.

# %%
h_sym = sp.Symbol("h", positive=True)
hu_sym = sp.Symbol("hu", real=True)
b_sym = sp.Symbol("b", real=True)
g_sym = sp.Symbol("g", positive=True)

sm_swe = SystemModel(
    swe,
    state_substitutions={state.H: h_sym, hu_fn: hu_sym, state.b: b_sym},
    state_variables=[b_sym, h_sym, hu_sym],
    equation_variable={("continuity",): h_sym,
                       ("momentum", "x"): hu_sym},
    time_var=t, coords=[x],
)
F = sm_swe.flux()
A = sm_swe.nonconservative_matrix()
S = sm_swe.source()

print("SWE extracted operators (state order [b, h, hu]):")
print(f"  F[h,  0] = {sp.simplify(F[1, 0])}")
print(f"  F[hu, 0] = {sp.simplify(F[2, 0])}")
for i in range(3):
    for j in range(3):
        val = sp.simplify(A[i, j, 0])
        if val != 0:
            print(f"  A[{i}, {j}, 0] = {val}")
print(f"  untagged remainders: {sm_swe.untagged_remainders() or '(none)'}")


# %% [markdown]
# ### Step 2.8 — Open question: the non-canonical decomposition
#
# What the extractor prints is **algebraically equivalent** to the
# canonical SWE `F[hu, 0] = hu²/h + g·h²/2`, `source[hu] = -g·h·∂_x b`,
# but it uses a different split:
#
# ```
#     F[hu, 0] = hu²/h + g·h·(b + h)
#     A[hu, h, 0] = -g·(b + h)
# ```
#
# Expand `∂_x(hu²/h + g·h·(b+h)) + A[hu,h]·∂_x h` and you recover
# `∂_x(hu²/h + g·h²/2) + g·h·∂_x b` — the same physics.  The
# pipeline emits the η-coupled decomposition because the symbolic
# `∂_x(g·ρ·(η - z))` term in hydrostatic pressure is more naturally
# left factored this way.  **Open question:** whether the adapter
# should canonicalize back to `g·h²/2` form for ergonomics.  For
# now the solver accepts both, and the reconstruction check below
# proves correctness.

# %%
q = (b_sym, h_sym, hu_sym)
reconstructed_mom = (
    sp.diff(F[2, 0], x)
    + sum(A[2, j, 0] * sp.diff(q_j, x) for j, q_j in enumerate(q))
)
canonical_mom = sp.diff(
    hu_sym**2 / h_sym + g_sym * h_sym**2 / 2, x,
) + g_sym * h_sym * sp.diff(b_sym, x)
assert sp.simplify(reconstructed_mom - canonical_mom) == 0
print("SWE momentum reconstruction matches canonical 1-D SWE with topography ✓")


# %% [markdown]
# ## Part 3 — Single-layer SME, level-1 Legendre, Newtonian
#
# SME projects x-momentum onto a Legendre basis (not a single
# layer-mean) before depth-integrating.  Unlike SWE, we keep the
# viscous stress tensor via `Newtonian(state)` — SME's first moment
# is exactly where viscous coupling appears.
#
# We branch off the common prefix *before* any material was applied,
# then specialize with `Newtonian`.  (Part 2 specialized with
# `Inviscid` on its own branch, so this branch is fresh.)

# %%
sme = model.branch(name="SME-L1")
sme.apply(Newtonian(state), name="Newtonian").simplify()


# %% [markdown]
# ### Step 3.1 — Galerkin test functions
#
# `Basis(state, Legendre_shifted, level=1)` carries three things:
# `basis.phi` (test functions in ζ), `basis.phi_of_z` (the same
# functions with `ζ = (z-b)/h` substituted, so they're explicit
# functions of `z` through `b, h`), and `basis.alpha` (the
# coefficient Functions `α_0(t, x), α_1(t, x)`).
#
# `Multiply(basis.phi_of_z, outer=True)` fans the single
# `momentum.x` leaf into a `Zstruct(test_0, test_1)` where each
# `test_l` is `φ_l((z-b)/h) · (pointwise x-momentum)`.  The
# `outer=True` flag is rank-changing; see `Multiply.__call__` at
# `ins_generator.py:~1625`.

# %%
basis_sme = Basis(state, Legendre_shifted, level=1)
sme.momentum.x.apply(Multiply(basis_sme.phi_of_z, outer=True),
                     name="Galerkin test",
                     description="multiply by phi_l((z-b)/h)")


# %% [markdown]
# ### Step 3.2 — `ProductRule()` to restore conservative form
#
# **Difficulty caught here.**  After the Galerkin multiply, terms
# look like `φ · ∂_x(u²)`.  These are **not** in conservative form
# — the coefficient `φ` depends on `x` through `b, h`.  When
# `Integrate(method="auto")` hits them it refuses to apply Leibniz
# (the extractor wants a single outer Derivative whose inner is
# state-side-homogeneous).  `ProductRule()` (default inverse
# direction) rewrites each such term
# `φ · ∂_v f = ∂_v(φ · f) - ∂_v(φ).doit() · f`: the first piece is
# now conservative and Leibniz applies; the second piece is the NC
# coupling that will be tagged later.  The residual form is an exact
# identity — if this transformation had been applied to one of a
# matching product-rule-expanded sibling pair, the residual would
# cancel under ``.simplify()``; here there is no sibling, so both
# pieces remain and the NC piece is the real coupling.

# %%
sme.momentum.x.apply(ProductRule(),
                     name="product rule (inverse)",
                     description="phi·∂_v(f) -> ∂_v(phi·f) - ∂_v(phi)·f")


# %% [markdown]
# ### Step 3.3 — Snapshot the pointwise continuity
#
# The bulk `w(t, x, z)` inside the shear-moment expansion needs a
# closure from pointwise continuity: `∂_x u + ∂_z w = 0`.  But in a
# few steps we'll depth-integrate continuity, turning it into a
# scalar `∂_t h + ∂_x(α_0 h) = 0` which has no `w` in it.  We need
# to snapshot the *pointwise* form **before** depth-integration and
# derive the closure later from that snapshot.
#
# **`Expression.copy()`** (`ins_generator.py:~707`) does exactly
# this: it returns a fully detached Expression with the same
# `_term_tags`, `_tag_order`, `_solver_groups`, and
# `_as_relation`.  Non-mutating — the main system sees nothing.

# %%
pointwise_continuity = sme.continuity.copy()


# %% [markdown]
# ### Step 3.4 — Depth-integrate, apply kinematic / stress / friction BCs
#
# Same depth integration as SWE, same interface KBCs, but now with
# an additional stress-free surface BC and a Navier-slip bottom
# friction closure — both as raw substitution dicts to demonstrate
# that the library doesn't demand a custom class per closure.

# %%
sme.apply(Integrate(z, state.b, state.eta, method="auto"))

u_at_b = state.u.subs(z, state.b)
u_at_eta = state.u.subs(z, state.eta)
kinematic_bcs = {
    state.w.subs(z, state.b):
        sp.Derivative(state.b, t) + u_at_b * sp.Derivative(state.b, x),
    state.w.subs(z, state.eta):
        sp.Derivative(state.eta, t) + u_at_eta * sp.Derivative(state.eta, x),
}
sme.apply(kinematic_bcs,
          name="kinematic BCs",
          description="w|_b, w|_eta via surface / bottom KBCs").simplify()

stress_free_surface = {state.tau["xz"].subs(z, state.eta): 0}
no_tangential_normal_stress = {
    state.tau["xx"].subs(z, state.b): 0,
    state.tau["xx"].subs(z, state.eta): 0,
}
sme.apply(stress_free_surface,
          name="stress-free surface").apply(
    no_tangential_normal_stress,
    name="no tangential normal stress",
).simplify()

lamda = sp.Symbol("lamda", positive=True)
tau_c = sp.Symbol("tau_c", positive=True)
friction_closure = {
    state.tau["xz"].subs(z, state.b):
        state.rho * (lamda / tau_c) * u_at_b,
}
sme.apply(friction_closure, name="Navier-slip friction").simplify()


# %% [markdown]
# ### Step 3.5 — The w-closure, pipeline style
#
# This is the piece that earned Task-3 a dedicated restart.  The
# bulk `w(t, x, z)` inside the shear moment's non-conservative
# terms cannot be closed by substitution alone — we need an
# **algebraic identity** derived from the pointwise continuity
# equation we snapshotted at Step 3.3.
#
# The identity is
#
#     w(t, x, z) = w|_b − ∫_b^z ∂_x u(t, x, z') dz'
#
# obtained by integrating `∂_x u + ∂_z w = 0` from `b` to `z`.
#
# The pipeline reads as four small, composable calls — every step
# is a primitive the user can read in isolation:

# %%
w_eq = pointwise_continuity.apply(
    Integrate(z, state.b, z, method="auto"),
)
w_closure = w_eq.solve_for(state.w)
sme.apply(w_closure,
          name="w-closure from continuity",
          description="w(t,x,z) = w|_b - int_b^z d_x u dz'").simplify()
sme.apply(kinematic_bcs,
          name="close w|_b (post w-closure)",
          description="close newly introduced w|_b").simplify()


# %% [markdown]
# **Difficulty caught here (1/3): running-integral Subs residues.**
# `Integrate(z, b, z, method="auto")` is a **running integral** —
# upper bound equals the integration variable.  The per-term
# `depth_integrate` dispatch emits `Subs(w, z, z)` wrappers that
# sympy doesn't auto-reduce (it stays opaque, refusing to commit
# to the identity substitution).  `Expression.apply` therefore
# invokes `_resolve_subs_safe` (`ins_generator.py:~3240`) after
# every Operation to unwrap trivial `Subs(f, var, var)` patterns.
# Without this, the subsequent `.solve_for(state.w)` wouldn't even
# find `w(t,x,z)` in the expression.
#
# **Difficulty caught here (2/3): binder shadowing.**
# `_resolve_subs_safe` has **three** guards before it unwraps a
# `Subs`: (a) if `var` appears as a Derivative differentiation
# variable inside, the unwrap would commit to a chain-rule choice
# sympy is deliberately conservative about; (b) if `var` is a
# bound variable of a nested Integral, a naive `xreplace` would
# overwrite the binder and produce nonsense limits like
# `Integral(_, (val, lo, val))`; (c) same for nested Subs.  Each
# of these guards prevents a specific wrong answer.
#
# **Difficulty caught here (3/3): non-mutating `solve_for`.**
# The snippet above uses `Expression.solve_for`, not the proxy
# `_NodeProxy.solve_for` from Step 1.5.  The proxy mutates a tree
# node; the Expression-level one returns a detached Expression
# whose `_as_relation = {w(t,x,z): rhs}`.  `sme.apply(w_closure)`
# then finds the `_as_relation` and routes through the relation
# substitution path.  Mixing these up silently produces a
# mutated-system-plus-detached-relation combo that behaves
# unpredictably on the next step.
#
# **Decoupling the identity from the BC.**  The closure's right-
# hand side leaves `w|_b` symbolic.  The subsequent `apply(kinematic_bcs)`
# is a separate step that closes `w|_b` using the bottom KBC.  In
# multi-layer SME (below) the analogous "lower-interface" value is
# closed via `InterfaceKBC(..., mass_flux=m_low)` — decoupling the
# identity from the BC makes the multi-layer generalization
# trivial.


# %% [markdown]
# ### Step 3.6 — ζ-transform, basis expand, evaluate integrals
#
# The remaining steps are standard: rewrite depth integrals in
# ζ = (z-b)/h, substitute the basis expansion of `u` (pointwise
# and ζ-transformed — `Basis.expand` emits both keys in one go),
# and evaluate the ζ-integrals with `EvaluateIntegrals`.

# %%
sme.apply(ZetaTransform(state))
sme.apply(basis_sme.expand(state.u))
sme.apply(EvaluateIntegrals(state)).simplify()


# %% [markdown]
# **Difficulty caught here: fixpoint loops.**  Two fixpoint loops
# run during this step.  The first is **inside** `EvaluateIntegrals`
# (`ins_generator.py:~1830`): each integration can uncover a `Subs`
# whose inner just got cleaner, and each `Subs` resolve can expose
# a new integrable `Integral`.  The loop alternates until nothing
# changes (bound 6 iterations).
#
# The second is inside `_simplify_preserve_integrals` itself
# (`ins_generator.py:~3188`), which runs three passes
# (`expand → simplify → collect`) in a fixpoint loop of its own.
# Without these loops, the single-layer SME would stop 10-20% short
# of its fixed point and a downstream `.simplify()` would be
# required to catch up.  These are invisible to the user — the
# walkthrough just calls `.simplify()` once at the end.


# %% [markdown]
# ### Step 3.7 — Closure report
#
# We look at each leaf and report: how many terms, whether any
# bulk `u, w, ζ` residue remains, and whether the final leaf
# still carries unclosed `Integral(..., (ζ, 0, 1))` nodes.

# %%
def _report(model, state):
    for path, eq in model.leaves():
        expr = eq.expr
        n_terms = len(sp.Add.make_args(sp.expand(expr)))
        residues = []
        if expr.has(state.u):
            residues.append("u")
        if expr.has(state.w):
            residues.append("w")
        if expr.has(state.zeta_ref):
            residues.append("ζ")
        integrals = [I for I in expr.atoms(sp.Integral)]
        status = "✓" if not residues and not integrals else "✗"
        note_bits = []
        if residues:
            note_bits.append(f"residues={','.join(residues)}")
        if integrals:
            note_bits.append(f"unclosed Integrals={len(integrals)}")
        note = f"  ({'; '.join(note_bits)})" if note_bits else ""
        print(f"  {status} {'.'.join(path):25s} n_terms={n_terms:3}{note}")


print("SME level-1 closure report:")
_report(sme, state)


# %% [markdown]
# ### Step 3.8 — Open question: the one unclosed ζ-integral
#
# `momentum.x.test_1` still carries **one** unclosed
# `Integral(..., (ζ, 0, 1))`.  The integrand is a rational form
# whose denominator
#
#     4·h·∂_x α_1 − 4·α_1·∂_x h  =  4·h²·∂_x(α_1/h)
#
# defeats `sympy.integrate`.  This is **not** a bug in the simplify
# pipeline — the simplify change that landed as Task-3.5 shrank the
# surrounding residual from 86 terms to 53 and eliminated every
# standalone `ν·(∂(-α_1))²·∂b / ∂α_1`-shaped artefact.  The one
# surviving integral is a pure sympy limitation.
#
# **Open question:** how to close it.  Three plausible paths:
#
# 1. **IBP inside the Integral.**  Rewrite `∫ f/g dζ` by parts
#    with a known antiderivative for `g` over `(0, 1)`.  Needs a
#    pattern matcher — we'd have to recognize the
#    `h²·∂_x(α_1/h)` factor structurally, which violates the
#    no-term-specific-branches rule unless generalized carefully.
# 2. **Pre-substitute ζ-polynomials before the Integrate.**  The
#    two `h²·Derivative(...)²` terms inside the integrand are
#    degree-2 polynomials in ζ once expanded; `sp.expand` hasn't
#    distributed them because they're wrapped by an outer
#    `Derivative(..., x)`.  A targeted pre-expansion inside
#    Integrands may be safe.
# 3. **Extend `auto_solver_tag`** to recognize product-rule-
#    expanded time derivatives like `h·∂_t α_0 + α_0·∂_t h =
#    ∂_t(α_0·h)`.  That alone eliminates six of the seven
#    mean-moment remainder terms without touching simplify; the
#    shear-moment integrand may then simplify further once the
#    surrounding structure collapses.
#
# None of these are safe to commit blindly — they interact with
# every other leaf.  The pytest test
# `test_sme_l1_shear_residual_is_single_zeta_integral` pins the
# current state so any future change is loudly visible.


# %% [markdown]
# ## Part 4 — Multi-layer SWE
#
# Two layers with a mass-flux interface between them.  We re-derive
# the common prefix (we've already specialized our `model` with
# `Inviscid` / `Newtonian` branches that fan out; cleanest is to
# start fresh).

# %%
state_ml = StateSpace(dimension=2)
t_ml, x_ml, z_ml = state_ml.t, state_ml.x, state_ml.z
z_1 = sp.Function("z_1", real=True)(t_ml, x_ml)
m_1 = sp.Function("m_1", real=True)(t_ml, x_ml)

model_ml = FullINS(state_ml)
model_ml.name = "Euler"
model_ml.momentum.z.apply(hydrostatic_scaling(state_ml)).simplify()
model_ml.momentum.z.apply(Integrate(z_ml, z_ml, state_ml.eta, method="analytical"))
model_ml.momentum.z.apply({state_ml.p.subs(z_ml, state_ml.eta): 0}).simplify()
model_ml.momentum.x.apply(model_ml.momentum.z.solve_for(state_ml.p)).simplify()
model_ml.momentum.z.remove()
model_ml.apply(Inviscid(state_ml), name="Inviscid (Euler)").simplify()

basis_ml = Basis(
    state_ml,
    LayeredBasis,
    interfaces=[state_ml.b, z_1, state_ml.eta],
)
print("ML basis level      :", basis_ml.level)
print("ML basis n_layers   :", basis_ml._bf.n_layers)


# %% [markdown]
# ### Step 4.1 — Branch per layer
#
# `System.branch(name=...)` clones the entire tree under a new
# root.  The two branches now evolve independently — closing one
# layer does not affect the other.
#
# **Difficulty caught here.**  The branched tree deep-clones every
# leaf Expression (`derived_system.py:~384`).  If the clone were
# shallow, the per-term tag dicts would alias — a `simplify()` on
# layer 0 would silently wipe tags on layer 1.  The deep clone has
# to walk both `_term_tags` and `_solver_groups` explicitly; see
# `_deep_clone`.

# %%
layer_0 = model_ml.branch(name="layer 0")
layer_1 = model_ml.branch(name="layer 1")


# %% [markdown]
# ### Step 4.2 — Per-layer pipeline with mass flux
#
# The helper below is a **thin wrapper** over the same primitives
# used in Part 2: depth-integrate, apply interface KBCs, close the
# layer's velocity via `basis.layer_expand`, collapse constant
# integrals.
#
# **Difficulty caught here — mass-flux sign convention.**  The
# middle interface `z_1` is permeable: mass can cross it from layer
# 0 into layer 1.  `InterfaceKBC(state, z_1, mass_flux=m_1)`
# produces `w|_{z_1} = ∂_t z_1 + u|_{z_1}·∂_x z_1 + m_1/ρ`.
#
# * The **lower** layer (layer 0) sees `+ m_1/ρ` as mass leaves
#   through its **upper** interface: the continuity residual picks
#   up `+ m_1/ρ`.
# * The **upper** layer (layer 1) sees `- m_1/ρ` through its
#   **lower** interface — the same `m_1` enters from below.
#
# The library supplies a single `mass_flux=m_1` value per
# interface; the **layer** picks up the sign from its own `w|_{z_1}`
# algebra during depth integration.  This convention is what makes
# the two canonical forms below correctly share a single `m_1`.

# %%
def close_layer(branch, lower, upper, layer_idx, m_low=None, m_up=None):
    branch.apply(Integrate(z_ml, lower, upper, method="auto"),
                 name=f"∫ dz layer {layer_idx}")
    branch.apply(InterfaceKBC(state_ml, lower, mass_flux=m_low)).simplify()
    branch.apply(InterfaceKBC(state_ml, upper, mass_flux=m_up)).simplify()
    branch.apply(basis_ml.layer_expand(state_ml.u, layer_idx),
                 name=f"close layer {layer_idx}").simplify()
    branch.apply(SimplifyIntegrals(state_ml)).simplify()


close_layer(layer_0, state_ml.b, z_1, 0, m_low=None, m_up=m_1)
close_layer(layer_1, z_1, state_ml.eta, 1, m_low=m_1, m_up=None)


# %% [markdown]
# ### Step 4.3 — Canonical check
#
# Compare each layer's equations against the canonical two-layer
# SWE with mass flux.  Residuals should be identically zero.

# %%
a0 = basis_ml.alpha.alpha_0
a1 = basis_ml.alpha.alpha_1
h_0 = z_1 - state_ml.b
h_1 = state_ml.eta - z_1
canon = {
    ("layer 0", "continuity"):
        sp.Derivative(h_0, t_ml) + sp.Derivative(h_0 * a0, x_ml) + m_1 / state_ml.rho,
    ("layer 0", "momentum.x"):
        (sp.Derivative(h_0 * a0, t_ml) + sp.Derivative(h_0 * a0**2, x_ml)
         + state_ml.g * h_0 * sp.Derivative(state_ml.eta, x_ml)
         + m_1 * a0 / state_ml.rho),
    ("layer 1", "continuity"):
        sp.Derivative(h_1, t_ml) + sp.Derivative(h_1 * a1, x_ml) - m_1 / state_ml.rho,
    ("layer 1", "momentum.x"):
        (sp.Derivative(h_1 * a1, t_ml) + sp.Derivative(h_1 * a1**2, x_ml)
         + state_ml.g * h_1 * sp.Derivative(state_ml.eta, x_ml)
         - m_1 * a1 / state_ml.rho),
}
print("ML-SWE canonical check:")
for branch, tag in [(layer_0, "layer 0"), (layer_1, "layer 1")]:
    for path, eq in branch.leaves():
        key = (tag, ".".join(path))
        residual = sp.simplify(sp.expand(eq.expr) - sp.expand(canon[key].doit()))
        status = "OK" if residual == 0 else f"RESIDUAL: {residual}"
        print(f"  {tag:<8} {'.'.join(path):<12} {status}")


# %% [markdown]
# ### Step 4.4 — Open question: unified two-layer `SystemModel`
#
# Today the adapter is **per-branch**: one `SystemModel` per
# layer.  For a numeric solver that evolves the joint state
# `[b, h_0, hu_0, h_1, hu_1]`, the caller has to stitch two
# per-layer adapters.  A single adapter that accepts a list of
# branches and emits block-diagonal `F` / `A` / `S` with mass-flux
# cross-terms on the off-diagonals is the natural extension — not
# built.  Task-4's regression harness doesn't cover multi-layer
# extraction yet; adding it is the obvious next test.


# %% [markdown]
# ## Part 5 — VAM, full derivation from scratch
#
# VAM (Vertically-Averaged Model) is the odd one out: it keeps
# z-momentum and treats `w` as its own dynamical variable.  There's
# no hydrostatic scaling, no analytic z-integration, no `p`
# substitution.  So **VAM cannot share the common prefix** from
# Part 1.
#
# Instead, VAM applies three independent Legendre bases — one for
# `u` (coefficients `U_0, U_1`), one for `w` (`W_0, W_1`), and one
# for `p` (`P_0, P_1`) — then projects BOTH x-momentum and z-momentum
# onto the test functions and depth-integrates the whole system.

# %%
state_v = StateSpace(dimension=2)
t_v, x_v, z_v = state_v.t, state_v.x, state_v.z

basis_u = Basis(state_v, Legendre_shifted, level=1, alpha_name="U")
basis_w = Basis(state_v, Legendre_shifted, level=1, alpha_name="W")
basis_p = Basis(state_v, Legendre_shifted, level=1, alpha_name="P")

print("VAM U coeffs:", [getattr(basis_u.alpha, f"U_{k}") for k in range(2)])
print("VAM W coeffs:", [getattr(basis_w.alpha, f"W_{k}") for k in range(2)])
print("VAM P coeffs:", [getattr(basis_p.alpha, f"P_{k}") for k in range(2)])


# %% [markdown]
# ### Step 5.1 — FullINS + Inviscid closure
#
# No hydrostatic scaling.  Inviscid (Euler) kills the stress
# tensor; the Newtonian alternative would bring in `∂²u/∂z²`
# terms that don't reduce by the fundamental theorem (the
# `_extract_derivative` in `Integrate(method="auto")` refuses
# second-order z-derivatives with a two-variable Derivative tuple)
# and would end up as a residual volume integral — the same gap
# that Part 3 documented.

# %%
model_v = FullINS(state_v)
model_v.name = "Euler (VAM root)"
model_v.apply(Inviscid(state_v), name="Inviscid (Euler)",
              description="tau = 0").simplify()


# %% [markdown]
# ### Step 5.2 — Galerkin-test BOTH momentum components
#
# Unlike SME (which tests only x-momentum because z-momentum was
# dropped after substitution), VAM keeps z-momentum alive.  We
# multiply BOTH by the test functions — the Legendre test set
# here is shared across components (`basis_u.phi_of_z`) since it
# depends only on geometry (`ζ = (z-b)/h`), not the field.
#
# **Difficulty caught here — rank-changing operation.**  The
# `outer=True` Multiply fans a single leaf into a `Zstruct` of
# `test_0`, `test_1`.  Because it changes the tree shape, it has
# to be invoked at the tree level (`model.momentum.x.apply(...)`)
# rather than the leaf level.  The library enforces this at
# `Expression.apply` (`ins_generator.py:~600`) — passing a
# rank-changing Multiply directly to `Expression.apply` raises
# `TypeError("rank-changing Operations must be invoked at the tree
# level")`.  Runtime error > silently corrupted tree.

# %%
model_v.momentum.x.apply(Multiply(basis_u.phi_of_z, outer=True),
                         name="Galerkin test — x-momentum")
model_v.momentum.z.apply(Multiply(basis_u.phi_of_z, outer=True),
                         name="Galerkin test — z-momentum")
model_v.momentum.x.apply(ProductRule())
model_v.momentum.z.apply(ProductRule())


# %% [markdown]
# ### Step 5.3 — Depth-integrate, KBCs, atmospheric pressure BC
#
# Standard shapes — `Integrate(z, b, η, method="auto")` over the
# whole system, then kinematic BCs at both interfaces and the
# atmospheric pressure at the surface.
#
# **Difficulty caught here — w-momentum boundary terms.**
# Depth-integrating `∂_t(ρw) + ∂_x(ρ uw) + ∂_z(ρ w²) + ∂_z p + ρg
# = 0` produces boundary terms `φ_l·ρ·w²|_η - φ_l·ρ·w²|_b` (from
# the `∂_z(ρ w²)` fundamental-theorem), plus the `ρ g h` volume
# piece (from the `ρg` constant term times the depth), plus the
# pressure piece that's now a Subs at the interfaces.  Because
# we never dropped z-momentum, the pressure gradient is handled
# **here** (at the z-momentum boundary values and the volume
# integral) rather than algebraically substituted out in Part 1.
# Watch the sign on `∂_z p` — it's the only place the pressure's
# `ρg` contribution enters VAM's z-momentum balance.

# %%
model_v.apply(Integrate(z_v, state_v.b, state_v.eta, method="auto"),
              name="∫ dz over depth")
model_v.apply(InterfaceKBC(state_v, state_v.b)).simplify()
model_v.apply(InterfaceKBC(state_v, state_v.eta)).simplify()
model_v.apply({state_v.p.subs(z_v, state_v.eta): 0},
              name="atmospheric pressure",
              description="p(t, x, eta) = 0").simplify()


# %% [markdown]
# ### Step 5.4 — ZetaTransform + three basis expansions
#
# Rewriting the depth integral as `∫_0^1 f(ζh + b) dζ` makes the
# integrand polynomial in ζ once the three basis expansions
# (`u = Σ U_k φ_k`, `w = Σ W_k φ_k`, `p = Σ P_k φ_k`) are plugged
# in.
#
# **Difficulty caught here — three expansions, one pipeline.**
# Each `basis.expand(field)` emits a dict of four keys
# (`ins_generator.py:~2960`): pointwise, ζ-transformed, bottom
# eval, surface eval.  Applying them sequentially is safe only
# because `xreplace` is purely structural and doesn't propagate
# dummy-dependency restrictions from one dict to the next — if we
# used `.subs` here, the second expansion could fail on keys that
# the first expansion introduced.  The `xreplace`-based dispatch
# at `ins_generator.py:~620` is what makes chained multi-field
# expansions work.

# %%
model_v.apply(ZetaTransform(state_v))
model_v.apply(basis_u.expand(state_v.u), name="expand u").simplify()
model_v.apply(basis_w.expand(state_v.w), name="expand w").simplify()
model_v.apply(basis_p.expand(state_v.p), name="expand p").simplify()
model_v.apply(EvaluateIntegrals(state_v)).simplify()


# %% [markdown]
# ### Step 5.5 — Closure report
#
# Five leaves: `continuity`, `momentum.x.{test_0, test_1}`,
# `momentum.z.{test_0, test_1}`.  All should close against the
# `(U_k, W_k, P_k)` coefficients — no bulk `u, w, p, ζ` residue.

# %%
print("VAM closure report:")
for path, eq in model_v.leaves():
    expr = eq.expr
    n_terms = len(sp.Add.make_args(sp.expand(expr)))
    residues = [name for name, f in
                [("u", state_v.u), ("w", state_v.w), ("p", state_v.p),
                 ("ζ", state_v.zeta)] if expr.has(f)]
    status = "✓" if not residues else "✗"
    note = f"   residue: {','.join(residues)}" if residues else ""
    print(f"  {status} {'.'.join(path):<26} n_terms={n_terms:3}{note}")


# %% [markdown]
# ### Step 5.6 — Open questions on VAM
#
# 1. **Under-determined.**  The final system has 5 equations for
#    7 unknowns `h, U_0, U_1, W_0, W_1, P_0, P_1`.  Standard VAM
#    closures pick one of: `W_0` from depth-integrated continuity
#    (treat `W` as dependent), or `P_1 = 0` (linear-pressure
#    assumption), or both (5 unknowns, 5 equations).  **This is a
#    modelling choice**, not a pipeline gap — the pipeline emits
#    a well-formed symbolic system and leaves the closure to the
#    modeller.
#
# 2. **No `continuity.test_1` equation.**  We projected *momentum*
#    onto `(φ_0, φ_1)` but continuity went through as a scalar.
#    Adding `model_v.continuity.apply(Multiply(basis_u.phi_of_z,
#    outer=True))` before Step 5.3 would fan continuity into
#    `continuity.test_0` and `continuity.test_1`, yielding a
#    6-equation system (still 1 short of fully determined).
#    Open question: which projection set is physically
#    meaningful?
#
# 3. **Viscous VAM.**  Swapping `Inviscid` for `Newtonian` brings
#    in `∂²u/∂z²` and `∂²w/∂z²` terms.  These hit the same
#    sympy-level wall as the SME shear moment — two-variable
#    Derivative tuples, fundamental theorem refuses, residual
#    volume integral remains.  The fix space is the same as the
#    Part-3 open question.
#
# 4. **Multi-layer VAM.**  Not derived.  In principle it's just
#    Part 4's per-layer pipeline applied to the VAM root — replace
#    the `Basis(Legendre_shifted)` trio with `Basis(LayeredBasis,
#    inner_cls=Legendre_shifted, inner_level=1, interfaces=[…])`,
#    branch per layer, run `close_layer` on each.  Not yet
#    verified end-to-end.
#
# 5. **Recombine on VAM.**  The output is displayed in expanded
#    form — `∂_t h + U_0 ∂_t h` rather than `∂_t(U_0 h)`.
#    `Recombine(vars=[t_v, x_v])` would collapse these.  Cosmetic,
#    not load-bearing.


# %% [markdown]
# ## Summary — one pipeline, four models
#
# All four derivations share the same small vocabulary:
#
# * `FullINS(state)` + `hydrostatic_scaling` / `Inviscid` /
#   `Newtonian` — Material / Assumption classes.
# * `Integrate(var, lower, upper, method=...)` — one Operation,
#   five dispatch modes (`auto`, `leibniz`, `fundamental_theorem`,
#   `direct`, `analytical`).
# * `InterfaceKBC(state, interface, mass_flux=...)` — one class
#   handling bottom, surface, and internal interfaces uniformly.
# * `Basis(state, kind, level=…)` + `basis.expand(field)` /
#   `basis.layer_expand(field, idx)` — one substitution dict
#   emitter with four keys per field.
# * `Multiply(basis.phi_of_z, outer=True)` — rank-changing
#   Galerkin projection.
# * `ProductRule()` (default inverse) — restores conservative form after the
#   projection multiplies by a z-dependent `φ`.
# * `ZetaTransform(state)` — rewrites depth integrals as ζ-integrals.
# * `EvaluateIntegrals(state)` — the fixpoint-looped integrator.
# * `Expression.copy()` + `Expression.solve_for(var)` — non-
#   mutating primitives for pipeline-style closures like the
#   SME w-closure.
# * `SystemModel(...)` — the thin reader that emits `flux`,
#   `nonconservative_matrix`, `source` for the numerical solver.
#
# What changes between the four derivations is which of these
# primitives you compose, in which order, on which branch of the
# tree.  **None of the primitives are physics-specific.**  The
# regression tests in
# `tests/unit/zoomy_core/test_symbolic_pipeline.py` lock this
# contract in place.
