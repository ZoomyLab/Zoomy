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
# # Slim SME walkthrough — opaque φ, principled integral transform
#
# Derive the **level-1 shallow-moment momentum equation** from incompressible
# Navier–Stokes with a slim, inviscid model: all stresses dropped, opaque
# basis ``φ_k`` kept symbolic on the *physical* column ``z ∈ [b, b+h]``
# throughout.  Pipeline:
#
# * **Step 0** — start with the raw INS system.
# * **Step 1** — drop **all** stresses ``τ_ij = 0``.
# * **Step 2** — hydrostatic z-momentum: integrate to get ``p(z) = ρg(η − z)``;
#   substitute into x-momentum; drop the z-equation.
# * **Step 3** — partial-integrate continuity on ``[b, z]`` → closure
#   ``w(z) = w(b) − ∫_b^z ∂_x u dẑ``.
# * **Step 4** — depth-integrate continuity on ``[b, η]`` + apply kinematic
#   BCs → h-evolution ``∂_t h + ∂_x ∫_b^η u dz = 0``.
# * **Step 5** — substitute ``w(z)`` closure into x-momentum.
# * **Step 6** — multiply x-momentum by the opaque test function
#   ``φ_k(z)`` (physical-z) and apply the inverse ``ProductRule``.
# * **Step 7** — depth-integrate x-momentum on ``[b, η]`` with
#   ``Integrate(method="auto")`` — picks Leibniz / fundamental theorem /
#   direct per-term, and consolidates volume pieces into one Integral per
#   ``(limits, diff-var)`` signature.
# * **Step 8** — apply kinematic BCs to the ``Subs(w, z, b|η)`` boundary
#   forms.
# * **Step 9** — eliminate ``∂_t h`` from x-momentum via the h-evolution
#   (``continuity.solve_for(∂_t h)`` + ``apply``).  Fires only on the
#   *boundary* ``∂_t h`` atom — the conservative ``∂_t [h · ∫…dz]`` is
#   one Derivative atom and isn't touched.
# * **Step 10** — ``IntegralTransform`` — affine map of every Integral to
#   the unit reference interval ``[0, 1]``.  Sibling Integrals share their
#   bound dummy by nesting depth (one ``\hat\zeta`` per depth) so
#   duplicates merge automatically.
# * **Step 11** — ``IsolateBasisIntegrand`` — distribute ``Add`` inside
#   each Integral and factor out every ζ̂-independent factor, leaving
#   ``coeff · Integral(pure-basis kernel, dζ̂)``.
# * **Step 12** — insert the velocity ansatz
#   ``u(z) = α_0 + Σ_{k≥1} α_k · φ_k((z − b)/h)`` (K&T eq 39) via
#   ``_FieldExpansion``.
# * **Step 13** — ``ProjectBasisIntegrals`` with a Legendre
#   ``BasisIntegralCache``: every ``∫_0^1 (basis kernel) dζ̂`` collapses
#   via orthogonality (``Poly.integrate``, no generic ``sp.integrate``).
#   Boundary evaluations ``φ_k|_{ζ=0/1}`` stay symbolic.

# %%
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.ins_generator import (
    FullINS, Integrate, IntegralTransform, IsolateBasisIntegrand,
    Multiply, ProductRule, ProjectBasisIntegrals, StateSpace,
    _FieldExpansion,
)
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.sme_model import hydrostatic_scaling


LEVEL = 1

# %% [markdown]
# ## Step 0 — Raw INS

# %%
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
model = FullINS(state)
model.describe()

# %% [markdown]
# ## Step 1 — Drop all viscous stresses

# %%
model.apply({state.tau[k]: 0 for k in state.tau._filter_dict()},
            name="drop all viscous stresses",
            description="τ_ij = 0  for all i, j").simplify()
model.describe()

# %% [markdown]
# ## Step 2 — Hydrostatic z-momentum, solve for $p$, substitute into x-mom

# %%
model.momentum.z.apply(hydrostatic_scaling(state),
                       name="hydrostatic scaling").simplify()
model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
model.momentum.z.apply({state.p.subs(z, state.eta): 0},
                       name="atmospheric pressure").simplify()
model.momentum.x.apply(model.momentum.z.solve_for(state.p),
                       name="substitute p into x-momentum").simplify()
model.momentum.z.remove()
model.describe()

# %% [markdown]
# ## Step 3 — Closure for $w(z)$ from continuity
#
# Take a snapshot of the pointwise continuity, partial-integrate from
# $b$ to $z$, and solve for $w(z)$:
# $$
#   w(z) = w(b) - \int_b^z \partial_x u(t, x, \hat z)\, d\hat z.
# $$

# %%
pointwise_continuity = model.continuity.copy()
w_eq = pointwise_continuity.apply(Integrate(z, state.b, z, method="auto"))
w_closure = w_eq.solve_for(state.w)
w_eq.describe()

# %% [markdown]
# ## Step 4 — Depth-integrate continuity + KBC → h-evolution

# %%
ub = state.u.subs(z, state.b)
ue = state.u.subs(z, state.eta)
kbc = {
    state.w.subs(z, state.b):
        sp.Derivative(state.b, t) + ub * sp.Derivative(state.b, x),
    state.w.subs(z, state.eta):
        sp.Derivative(state.eta, t) + ue * sp.Derivative(state.eta, x),
}
model.continuity.apply(Integrate(z, state.b, state.eta, method="auto"))
model.continuity.apply(kbc, name="kinematic BCs (continuity)").simplify()
model.continuity.describe()

# %% [markdown]
# ## Step 5 — Substitute $w(z)$ closure into x-momentum

# %%
model.momentum.x.apply(w_closure,
                       name="w-closure into x-momentum").simplify()
model.momentum.x.describe()

# %% [markdown]
# ## Step 6 — Galerkin test + inverse product rule
#
# Multiply by the opaque ``φ_k((z − b)/h)`` test functions; rewrite each
# ``φ · ∂_v f`` as ``∂_v(φ · f) − ∂_v(φ) · f`` so the conservative form
# is Leibniz-ready.

# %%
phi_fns = [sp.Function(f"phi_{k}") for k in range(LEVEL + 1)]
# The basis lives on the physical column ``z ∈ [b, b+h]`` via the FEM
# convention ``phi_k(z) := phi_k_ref((z − b)/h)`` where ``phi_k_ref`` is
# the reference basis on [0, 1].  We write the reference-evaluated
# form ``phi_k((z − b)/h)`` directly so:
#
# * ``ProductRule``'s chain rule on ``∂_x phi_k((z−b)/h)`` fires
#   automatically and produces the moving-boundary metric terms that
#   K&T captures via their ω-coupling — packaged as a chain-rule
#   residual integrand instead of as an integral-of-∂_ζ-of-h-u-ω.
# * Leibniz boundary terms from ``Integrate(method="auto")`` and the
#   chain-rule contributions cancel exactly as required.
# * After ``IntegralTransform`` substitutes ``z → ζh + b``, the
#   argument simplifies automatically: ``((ζh+b)−b)/h → ζ``,
#   ``(b−b)/h → 0``, ``((b+h)−b)/h → 1``.  Boundary evaluations land
#   on ``phi_k(0)`` / ``phi_k(1)`` directly.
zeta_of_z = (state.z - state.b) / state.H
phi_of_z_opaque = Zstruct(**{
    f"phi_{k}": phi_fns[k](zeta_of_z) for k in range(LEVEL + 1)
})
model.momentum.x.apply(Multiply(phi_of_z_opaque, outer=True))
model.momentum.x.apply(ProductRule(),
                       name="inverse product rule",
                       description="φ · ∂_v f → ∂_v(φ · f) − ∂_v(φ) · f")
model.momentum.x.describe()

# %% [markdown]
# ## Step 7 — Depth-integrate $x$-momentum on $[b, η]$

# %%
model.momentum.x.apply(Integrate(z, state.b, state.eta, method="auto"))
model.momentum.x.describe()

# %% [markdown]
# ## Step 8 — Apply kinematic BCs to the $w|_b$, $w|_η$ boundary forms

# %%
model.momentum.x.apply(kbc, name="kinematic BCs (momentum)").simplify()
model.momentum.x.describe()

# %% [markdown]
# ## Step 9 — Use the h-evolution to eliminate $\partial_t h$ from x-momentum
#
# Continuity gives us $\partial_t h = -\partial_x \int_0^1 h\,u(\zeta h+b)\,d\hat\zeta_0$.
# ``solve_for(∂_t h)`` returns that as a substitution relation; ``apply``
# substitutes it everywhere ``∂_t h`` appears as a free atom in
# ``momentum.x``.  Importantly this fires only on the *boundary*
# ``\phi_l|_{z=b+h}\,u|_{z=b+h}\,\partial_t h`` term — not inside the
# conservative ``∂_t \int … dz`` (which holds it as a single
# ``Derivative(integral, t)`` atom).

# %%
dt_h_relation = model.continuity.solve_for(sp.Derivative(state.H, t))
model.momentum.x.apply(dt_h_relation,
                       name="eliminate ∂_t h via continuity").simplify()
model.momentum.x.describe()

# %% [markdown]
# ## Step 10 — `IntegralTransform` to the unit reference interval

# %%
model.apply(IntegralTransform(), name="affine map to [0, 1]")
model.simplify()
model.describe()

# %% [markdown]
# ## Step 11 — Isolate basis integrands: ``coeff · ∫ kernel d\hat\zeta``
#
# For each Integral, we want the integrand to contain *only* the
# integration-variable-dependent factors — i.e. the basis functions and
# any ``\hat\zeta``-power weights.  Everything else (``α_k(t,x)``,
# ``h``, ``b``, ``\partial_x h``, …) gets pulled out as an outer
# coefficient.  After this rewrite, each Integral has the canonical
# pattern that a basismatrices lookup can recognise:
#
# * ``∫_0^1 \phi_k\,\phi_l\,d\hat\zeta``  — mass matrix entries.
# * ``∫_0^1 \phi_k'\,\phi_l\,d\hat\zeta`` — first-derivative coupling.
# * ``∫_0^1 \hat\zeta\,\phi_k\,\phi_l\,d\hat\zeta`` — z-weighted
#   couplings (from the affine map's metric terms).
# * etc.

# %%
model.apply(IsolateBasisIntegrand(), name="isolate basis integrand").simplify()
model.describe()

# %% [markdown]
# ## Step 12 — Insert the velocity ansatz
#
# K&T eq (39) writes the velocity as
# $$
#   u(t, x, z) = \alpha_0(t, x) + \sum_{j=1}^{N} \alpha_j(t, x)\,\hat\phi_j\!\bigl(\tfrac{z - b}{h}\bigr).
# $$
# In our notation ``\alpha_0 = u_m`` is the depth-averaged velocity (the
# "mass moment") and ``\alpha_j`` for ``j ≥ 1`` are the higher Legendre
# coefficients.  We use a function-level ``replace`` (via
# ``_FieldExpansion``) so the substitution descends into
# ``Derivative(u(t,x,arg), x)``, ``Subs(...)`` and the running-integral
# integrands that ordinary ``.subs()`` skips.

# %%
basis_alpha = [
    sp.Function(f"alpha_{k}", real=True)(state.t, state.x)
    for k in range(LEVEL + 1)
]


def _u_ansatz(*args):
    """u(t, x, arg)  →  α_0(t, x) + Σ_{k≥1} α_k(t, x) · phi_k((arg − b)/h).

    Each newly-introduced ``phi_k`` lands directly in reference form
    (``(arg − b)/h``) so that, for ``arg`` ∈ {``b``, ``b + h``, ``ζ·h + b``},
    sympy auto-simplifies the argument to ``0`` / ``1`` / ``ζ``.  The
    test functions multiplied in step 6 used ``phi_k(state.z)`` and
    were rewritten by ``MapBasisToReference`` in step 12 — both ends
    now read in the same canonical form.
    """
    arg_z = args[-1]
    rhs = basis_alpha[0]
    for k in range(1, LEVEL + 1):
        rhs = rhs + basis_alpha[k] * phi_fns[k]((arg_z - state.b) / state.H)
    return rhs


model.apply(_FieldExpansion(state.u.func, _u_ansatz,
                            name="u ansatz"))
model.simplify()
model.describe()

# %% [markdown]
# ## Step 13 — Project basis integrals via the basismatrix cache
#
# Every ``Integral(_, (ζ̂, 0, 1))`` whose integrand is a polynomial in
# the opaque ``\hat\phi_k(\zeta)``, ``\hat\phi_k'(\zeta)`` is now a
# basismatrix entry.  ``BasisIntegralCache(Legendre_shifted)`` performs
# the lookup — substituting the basis polynomial in just the integrand
# (so ``φ_k(ζ̂)`` collapses) and integrating via ``Poly.integrate`` (no
# generic ``sp.integrate``, no ``cancel``/``together``).  Boundary
# evaluations ``φ_k|_{ζ=0}``, ``φ_k|_{ζ=1}`` are *outside* every
# Integral and therefore stay symbolic — exactly the cleaner form you
# preferred over substituting the numeric Legendre values.

# %%
cache = BasisIntegralCache(Legendre_shifted(level=LEVEL))
model.apply(ProjectBasisIntegrals(cache),
            name="project basis integrals")
model.simplify()
model.describe()
