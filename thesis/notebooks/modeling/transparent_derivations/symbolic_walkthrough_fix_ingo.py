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
# # SWE / SME end-to-end — symbolic walkthrough

# %%
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.ins_generator import (
    Integrate,
    IntegralTransform,
    IsolateBasisIntegrand,
    KinematicBC,
    Multiply,
    ProductRule,
    ProjectBasisIntegrals,
    StateSpace,
    _FieldExpansion,
)
from zoomy_core.model.models.derived_system import System, Expression
from zoomy_core.symbolic import (
    affine_change_of_variable,
    canonicalise,
    canonicalize_phi_derivative_subs,
    distribute_derivative_over_add,
    function_expand,
    product_rule_forward,
    project_basis_integrand,
    split_integral_over_add,
    subst,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling

LEVEL = 1

# %% [markdown]
# ## Step 1 — Empty model + register equations one-by-one

# %%
state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z
H = state.h

# Start from an empty system and register every equation explicitly.
# Scalar laws (bathymetry, continuity) go in as flat top-level leaves;
# the momentum *vector* lives under the ``momentum`` parent with one
# leaf per spatial component.  The ``model.apply(...)`` pattern (used
# in later steps) then dispatches per component:
#   * ``model.momentum.x.apply(op)`` mutates only the x-component.
#   * ``model.momentum.apply(op)`` broadcasts to every component.
model = System("SME-derivation", state)

# 1a. Trivial bathymetry evolution:  ∂_t b = 0  (b is static topography).
model.add_equation("bathymetry", sp.Derivative(state.b, state.t))

# 1b. Mass balance: ∂_x u + ∂_z w = 0  (incompressible continuity in 2D).
model.add_equation(
    "continuity",
    sp.Derivative(state.u, state.x) + sp.Derivative(state.w, state.z),
)

# 1c. Declare the momentum vector with empty placeholder leaves for
# components ``x`` and ``z``.  Per-component proxies
# ``model.momentum.x`` / ``model.momentum.z`` become immediately
# addressable for filling via ``.set(...)`` below.
model.add_vector_equation("momentum", ("x", "z"))

# 1d. Fill the x-component:
#   ∂_t u + ∂_x(u u) + ∂_z(u w) + (1/ρ)·∂_x p
#     − (1/ρ)·(∂_x τ_xx + ∂_z τ_xz) = 0.
model.momentum.x.set(
    sp.Derivative(state.u, t)
    + sp.Derivative(state.u * state.u, x)
    + sp.Derivative(state.u * state.w, z)
    + sp.Derivative(state.p, x) / state.rho
    - (sp.Derivative(state.tau["xx"], x)
       + sp.Derivative(state.tau["xz"], z)) / state.rho
)

# 1e. Fill the z-component:
#   ∂_t w + ∂_x(w u) + ∂_z(w w) + (1/ρ)·∂_z p
#     − (1/ρ)·(∂_x τ_zx + ∂_z τ_zz) + g = 0.
model.momentum.z.set(
    sp.Derivative(state.w, t)
    + sp.Derivative(state.w * state.u, x)
    + sp.Derivative(state.w * state.w, z)
    + sp.Derivative(state.p, z) / state.rho
    - (sp.Derivative(state.tau["zx"], x)
       + sp.Derivative(state.tau["zz"], z)) / state.rho
    + state.g
)

# Inviscid setup matching K&T 2019: zero the viscous stress tensor
# up-front.
model.apply({state.tau[k]: 0 for k in state.tau._filter_dict()}).simplify()
model.describe()

# %% [markdown]
# ## Step 2 — Hydrostatic on z-momentum

# %%
model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
model.momentum.z.describe()

# %% [markdown]
# ## Step 3 — Analytic integration of z-momentum

# %%
model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
model.momentum.z.describe()

# %% [markdown]
# ## Step 4 — Atmospheric BC at the free surface

# %%
model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
model.momentum.z.describe()

# %% [markdown]
# ## Step 5 — Substitute $p$ into x-momentum and drop z-momentum

# %%
model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
model.momentum.z.remove()
model.describe()

# %% [markdown]
# ## Step 6 — Solve continuity (running integral) for $w(z)$

# %%
pc = model.continuity.copy()
w_eq = pc.apply(Integrate(z, state.b, z, method="auto"))

# After ``Integrate(z, state.b, z)`` the integrated continuity carries
# the upper-bound boundary term as ``Subs(w(t, x, _hat), _hat, z)``
# rather than the natural bare ``w(t, x, z)``.  Sympy-equivalent, but
# ``solve_for(state.w)`` doesn't recognise the Subs as the target.
#
# Proper fix would live in ``Integrate``: the upper-bound substitution
# should collapse to bare ``state.w`` when the bound is the integration
# variable's outer symbol.  Until then we walk the expression and swap
# the offending ``Subs(w, _hat, z)`` atom for ``state.w``, then
# ``solve_for`` works cleanly.
subs_w_at_z = next(
    s for s in w_eq.expr.atoms(sp.Subs)
    if isinstance(s.expr, sp.Function)
    and s.expr.func.__name__ == "w"
    and s.point[0] == z
)
w_eq_expr_clean = w_eq.expr.xreplace({subs_w_at_z: state.w})
w_at_z_rhs = sp.solve(w_eq_expr_clean, state.w)[0].doit()
w_closure = {state.w: w_at_z_rhs}

# %% [markdown]
# ## Step 7 — Depth-integrate continuity and substitute $w(z)$ into x-momentum

# %%
model.continuity.apply(Integrate(z, state.b, state.eta, method="auto"))
model.momentum.x.apply(w_closure).simplify()

# %% [markdown]
# ## Step 8 — Kinematic BCs at bottom and surface (new unified operator)

# %%
model.apply(KinematicBC(state, state.b)).apply(KinematicBC(state, state.eta)).simplify()
model.continuity.describe()

# %% [markdown]
# ## Step 9 — Multiply x-momentum by $\varphi_j(\zeta)$ (opaque basis)

# %%
# Two opaque-basis flavours are wired in the codebase:
#
#   (1) **Single-class + ``state.zeta``** (VAM-canonical, see
#       ``vam_galerkin.py``).  ``Legendre_shifted(level=L,
#       symbol="phi_u")`` builds ONE 2-arg Function class
#       ``phi_u(k, arg)``; basis atoms ``phi_u(k, state.zeta)`` stay
#       opaque through ``AffineProjection`` + ``EvaluateIntegrals``,
#       which route via the ``_basis`` back-ref.  No bug-3 closure
#       needed *because VAM keeps ``w`` as a separate state field* —
#       no continuity-driven w-substitution interacts with the basis
#       integrals.
#
#   (2) **Per-k classes + ``(z-b)/h``** (this walkthrough's choice).
#       One Function class per index ``phi_0, phi_1, …``, each
#       carrying its own ``_basis`` back-ref.  Composes with
#       ``IntegralTransform`` + ``IsolateBasisIntegrand`` +
#       ``ProjectBasisIntegrals(BasisIntegralCache(basis))`` (which
#       reads ``k`` from the class name).  This route survives the
#       w-closure substitution (continuity solved for ``w(z)``,
#       result fed into x-momentum) — at the cost of a bug-3
#       closure step at the end to clean up the residual
#       ``Derivative(α_l·h/k, t)`` atoms.
#
# We're using flavour (2) here because the walkthrough takes the
# classical SME path "solve continuity for w, substitute into
# x-momentum" (Steps 6–7).  Flavour (1) works for VAM precisely
# because it skips that substitution.
basis = Legendre_shifted(level=LEVEL)
phi = [
    type(f"phi_{k}", (sp.Function,), {"_basis": basis, "_index": k})
    for k in range(LEVEL + 1)
]

zeta_of_z = (z - state.b) / H
test_phi = Zstruct(**{
    f"phi_{k}": phi[k](zeta_of_z) for k in range(LEVEL + 1)
})

model.momentum.x.apply(Multiply(test_phi, outer=True))
model.momentum.x.apply(ProductRule())
model.momentum.x.apply(Integrate(z, state.b, state.eta, method="auto"))
model.momentum.x.apply(KinematicBC(state, state.b))
model.momentum.x.apply(KinematicBC(state, state.eta)).simplify()

# %% [markdown]
# ## Step 10 — Capture $\partial_t h$ relation (do NOT apply yet)

# %%
# The continuity row j=0 gives ``∂_t h = − ∂_x(h α_0)``.  We capture
# the relation but **deliberately don't substitute** it into the
# x-momentum residuals: doing so would expand ``Derivative(h α_k, t)``
# via product rule and destroy the conservative time-derivative form.
# K&T eq (4.14) is in conservative ``∂_t(h α_k)`` form, and so is the
# downstream SystemModel state ``q_k = h α_k``.  The relation gets
# used only during the K&T-comparison ``normal_form`` step (Step 16),
# applied symmetrically to BOTH the pipeline residual and the
# reference equations.
dt_h_node = model.continuity.solve_for(sp.Derivative(H, t))
dt_h_relation = dict(dt_h_node._node._as_relation)

# %% [markdown]
# ## Step 11 — Substitute the Legendre ansatz for $u(t, x, z)$ (opaque ``Expand``)

# %%
# ``Expand`` is the VAM-canonical Legendre-ansatz substitution: replaces
# every call ``u(t, x, arg)`` with a single ``Sum`` atom
#
#     Sum( coeff_u(k) · basis.phi_fn(k, (arg − b)/h),  (k, 0, L) )
#
# carrying TWO opaque placeholders — the basis Function (``_basis``
# back-ref) and a private per-Expand ``coeff_<symbol>`` Function whose
# class-level ``_coeff_table`` holds the pre-declared coefficient list.
# The Sum stays unevaluated through ``AffineProjection`` /
# ``Integrate`` / ``ProductRule``; only at integration time does
# ``EvaluateIntegrals`` ``.doit()`` the Sum and substitute concrete
# coefficient atoms.  Compare with the per-Add ``_FieldExpansion``
# (used by ``kt2019_verification.py``) — same final result, but
# ``Expand`` keeps the equation tree O(1) in level instead of O(L+1).
basis_alpha = [sp.Function(f"alpha_{k}", real=True)(t, x) for k in range(LEVEL + 1)]


def u_ansatz(*args):
    """Legendre ansatz called by ``_FieldExpansion`` for every
    ``u(t, x, arg)`` Function call.  ``args[-1]`` is the vertical
    sample point — ``state.z`` in the volume integrand, ``state.b``
    at the bottom boundary, ``state.b + state.h`` at the free
    surface.  Returning the ansatz evaluated at ``arg[-1]`` handles
    all three samples automatically.

    Why ``_FieldExpansion`` rather than ``model.apply({state.u: ...})``?
    ``state.u`` is the Function call ``u(t, x, z)`` — ``xreplace``
    matches that exact arg list only.  The KBC-substituted boundary
    calls ``u(t, x, b)`` / ``u(t, x, η)`` have different arg lists
    and would slip through.  ``_FieldExpansion`` walks every
    ``u(...)`` Function call regardless of arguments.
    """
    arg_z = args[-1]
    rhs = basis_alpha[0]
    for k in range(1, LEVEL + 1):
        rhs = rhs + basis_alpha[k] * phi[k]((arg_z - state.b) / H)
    return rhs


model.apply(_FieldExpansion(state.u.func, u_ansatz)).simplify()

# %% [markdown]
# ## Step 12 — Resolve basis integrals via `ProjectBasisIntegrals`

# %%
# Every Galerkin volume integral has the form
# ``∫_b^{b+h} (polynomial in φ_k((z-b)/h)) · (rational in α, h, b, …) dz``.
# ``IntegralTransform`` normalises the integrand for the basis cache;
# ``IsolateBasisIntegrand`` factors ζ-independent terms out;
# ``ProjectBasisIntegrals`` queries the per-basis ``BasisIntegralCache``
# (closed-form Legendre inner products) via each atom's ``_basis``
# back-reference — no manual table.
model.apply(IntegralTransform()).simplify()
model.apply(IsolateBasisIntegrand()).simplify()
cache = BasisIntegralCache(basis)
model.apply(ProjectBasisIntegrals(cache)).simplify()

# %% [markdown]
# ## Step 13 — Bug-3 closure (stranded $\partial_t h$ atoms from w-closure)

# %%
# The w-closure in Step 6 substitutes ``w(t, x, z)`` symbolically
# inside the x-momentum integrand.  After Galerkin projection
# (``Multiply(test_phi, outer=True)`` + ``ProductRule`` +
# ``Integrate`` + ``KinematicBC``), held ``Derivative(α_l·h/k, t)``
# atoms can survive the integration: the outer time derivative only
# materialises ``α_l · ∂_t h`` once an explicit product-rule
# expansion fires.  ``_close_bug3`` iterates that expansion + the
# j=0 continuity substitution ``∂_t h → −∂_x(α_0·h)`` until
# convergence, putting the residual into the K&T (4.14)
# post-substitution form.
def _close_bug3(expr):
    zeta_hat = sp.Symbol(r"\hat{\zeta}", positive=True)

    def expand_held_dt(e):
        def _walk(node):
            if isinstance(node, sp.Derivative):
                wrt = [v[0] if isinstance(v, (tuple, sp.Tuple)) else v
                       for v in node.args[1:]]
                if t in wrt and isinstance(node.args[0], (sp.Mul, sp.Pow)):
                    return product_rule_forward(node, t)
            if node.args:
                new_args = tuple(_walk(a) for a in node.args)
                if any(n is not o for n, o in zip(new_args, node.args)):
                    return node.func(*new_args)
            return node
        return distribute_derivative_over_add(_walk(e))

    for _ in range(10):
        prev = expr
        expr = expand_held_dt(expr)
        expr = subst(expr, dt_h_relation)
        expr = function_expand(expr, state.u.func, u_ansatz)
        expr = affine_change_of_variable(expr, z, state.b, state.b + H, zeta_hat)
        expr = canonicalize_phi_derivative_subs(expr)
        expr = split_integral_over_add(expr)
        expr = project_basis_integrand(expr, cache)
        expr = canonicalise(expr)
        if expr == prev:
            break
    return expr


test_eqs = {}
for k in range(LEVEL + 1):
    test_eqs[k] = getattr(model.momentum.x, f"test_{k}").expr.doit()
    sub_resolve = {}
    for j in range(LEVEL + 1):
        for arg in (sp.S.Zero, sp.S.One):
            sub_resolve[phi[j](arg)] = basis.eval(j, arg)
    test_eqs[k] = test_eqs[k].xreplace(sub_resolve)
    test_eqs[k] = _close_bug3(test_eqs[k])

# %% [markdown]
# ## Step 14 — K&T 2019 reference equations

# %%
g = state.g
a = basis_alpha

if LEVEL == 0:
    ref_eqs = [
        sp.Derivative(H * a[0], t)
        + sp.Derivative(H * a[0]**2 + g * H**2 / 2, x),
    ]
elif LEVEL == 1:
    ref_eqs = [
        sp.Derivative(H * a[0], t)
        + sp.Derivative(H * a[0]**2 + H * a[1]**2 / 3 + g * H**2 / 2, x),
        sp.Derivative(H * a[1], t)
        + sp.Derivative(2 * H * a[0] * a[1], x)
        - a[0] * sp.Derivative(H * a[1], x),
    ]
else:
    raise NotImplementedError(
        f"Add the K&T reference for LEVEL={LEVEL} (eq (4.17) or higher).")

# %% [markdown]
# ## Step 15 — Project both to K&T form (flat bottom, Legendre boundary values)

# %%
def kt_form(expr):
    # K&T 2019 assumes flat bottom — substitute it.  Opaque basis atoms
    # (boundary values of ``phi_k``) have already been resolved by
    # ``basis.resolve_atoms`` in the per-row closure above.
    sub = {
        state.b: 0,
        sp.Derivative(state.b, x): 0,
        sp.Derivative(state.b, t): 0,
    }
    return sp.expand(sp.expand(expr).subs(sub))


def expand_derivatives(expr):
    def step(e):
        if isinstance(e, sp.Add):
            return sp.Add(*[step(a) for a in e.args])
        if isinstance(e, sp.Mul):
            return sp.Mul(*[step(a) for a in e.args])
        if isinstance(e, sp.Derivative):
            inner = step(e.expr)
            wrt_pairs = e.variable_count
            v, n = wrt_pairs[0]
            rest = wrt_pairs[1:]
            if n > 1:
                rest = ((v, n - 1),) + tuple(rest)
            if isinstance(inner, sp.Add):
                return step(sp.Add(*[sp.Derivative(a, v, *rest) for a in inner.args]))
            if isinstance(inner, sp.Mul):
                factors = inner.args
                out = sp.Add(*[
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v),)
                             + factors[i + 1:]))
                    for i in range(len(factors))
                ])
                if rest:
                    out = sp.Derivative(out, *rest)
                return step(out)
            if isinstance(inner, sp.Pow):
                base, exponent = inner.args
                if isinstance(exponent, sp.Integer) and int(exponent) >= 2:
                    n_pow = int(exponent)
                    return step(n_pow * base**(n_pow - 1) * sp.Derivative(base, v, *rest))
            return e
        return e

    prev = None
    cur = sp.expand(expr)
    while prev != cur:
        prev = cur
        cur = sp.expand(step(cur))
    return cur


def kill_trivially_zero_derivatives(expr):
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(sp.Derivative):
        inner = d.args[0]
        wrt = []
        for v in d.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        if inner.is_number:
            mapping[d] = sp.S.Zero
        elif not any(inner.has(v) for v in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normal_form(expr):
    cont = {sp.Derivative(H, t): -sp.Derivative(basis_alpha[0] * H, x)}
    e = expand_derivatives(expr)
    e = kill_trivially_zero_derivatives(e)
    e = e.subs(cont)
    e = expand_derivatives(e)
    e = kill_trivially_zero_derivatives(e)
    return sp.expand(e)


# %% [markdown]
# ## Step 16 — Verify term-by-term against K&T

# %%
n_mismatch = 0
for k in range(LEVEL + 1):
    m_k = sp.Rational(1, 2 * k + 1)
    pipe = kt_form(test_eqs[k] / m_k)
    ref = kt_form(ref_eqs[k])
    pipe_n = normal_form(pipe)
    ref_n = normal_form(ref)
    diff = sp.expand(pipe_n - ref_n)
    ok = (diff == 0)
    mark = "✓ MATCH" if ok else "✗ MISMATCH"
    print(f"test_{k} ↔ K&T row {k}: {mark}")
    if not ok:
        n_mismatch += 1
        sp.pretty_print(diff)

print()
print(f"=== SME L={LEVEL}: {n_mismatch} mismatch(es) ===")

# %% [markdown]
# ## Step 17 — SystemModel in conservative variables $q_k = h \alpha_k$

# %%
from zoomy_core.model.models.system_model import SystemModel

# Read K&T (4.14) in conservative form ``q_k = h · α_k``:
#
#   ∂_t h         + ∂_x q_0                                              = 0
#   ∂_t q_0       + ∂_x( q_0²/h + g h²/2 + q_1²/(3h) )                   = 0
#   ∂_t q_1       + ∂_x( 2 q_0 q_1 / h )  −  (q_0/h) · ∂_x q_1           = 0
#   ∂_t b         = 0                                                    (trivial)
#
# This is the form the rest of the Zoomy numerics stack consumes
# (Rusanov/HLL flux + path-integral NCP).  We verified above that the
# symbolic Galerkin pipeline yields exactly this physics (K&T row-by-
# row match).  Build a ``SystemModel`` directly from these terms and
# inspect it via ``.describe()``.
h_sym = sp.Symbol("h", positive=True, real=True)
q_sym_state = [sp.Symbol(f"q_{k}", real=True) for k in range(LEVEL + 1)]
b_sym = sp.Symbol("b", real=True)
g_sym = sp.Symbol("g", positive=True)
state_syms = [h_sym] + q_sym_state + [b_sym]
n_state = len(state_syms)

F = sp.zeros(n_state, 1)
B = sp.MutableDenseNDimArray.zeros(n_state, n_state, 1)
S = sp.zeros(n_state, 1)
M = sp.eye(n_state)

# Mass row.
F[0, 0] = q_sym_state[0]
# x-mom q_0.
F[1, 0] = q_sym_state[0]**2 / h_sym + g_sym * h_sym**2 / 2
if LEVEL >= 1:
    F[1, 0] += q_sym_state[1]**2 / (3 * h_sym)
    # x-mom q_1.
    F[2, 0] = 2 * q_sym_state[0] * q_sym_state[1] / h_sym
    B[2, 2, 0] = -q_sym_state[0] / h_sym

# Bathymetry is static.
M[-1, -1] = 0

sm = SystemModel(
    time=t,
    space=[x],
    state=state_syms,
    aux_state=[],
    parameters=Zstruct(g=g_sym),
    parameter_values=Zstruct(g=9.81),
    flux=F,
    hydrostatic_pressure=sp.zeros(n_state, 1),
    nonconservative_matrix=B,
    source=S,
    mass_matrix=M,
)
sm.equation_names = ["mass"] + [f"xmom_j{k}" for k in range(LEVEL + 1)] + ["b_eq"]

# %% [markdown]
# ## Step 18 — `SystemModel.describe()` (visual K&T comparison)

# %%
print(sm.describe(full=True))

