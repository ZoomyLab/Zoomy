"""VAM (1, 2) derivation in physical (t, x, z) — Escalante et al. 2024.

Reference: Escalante, Morales de Luna, Cantero-Chinchilla, Castro-Orgaz
(2024) "Polynomial vertically averaged and moment equations for shallow
flow with non-hydrostatic pressure".  We re-derive eq (4) (governing
equations) and eq (5) (closure constraints) **directly from the
physical-z Navier-Stokes system**, without ever transforming to
σ-coordinates.

This script is intentionally stripped of all model-specific helpers.
Every math step uses ONLY:

  - sympy ``Symbol``, ``Function``, ``Derivative``, ``solve``,
    ``xreplace``, ``expand``, ``simplify``;
  - the calculus primitive ``polynomial_integrate`` from
    ``zoomy_core.symbolic``.

(Leibniz / FT are not needed when the ansatz is explicit polynomial:
 the integrand is polynomial in ``z`` after the affine map.)

Polynomial ansatz:

    u(t, x, z) = u_0(t,x) + u_1(t,x)·φ_1(ζ)                    (M = 1)
    w(t, x, z) = w_0(t,x) + w_1(t,x)·φ_1(ζ) + w_2(t,x)·φ_2(ζ)  (N_w = 2)
    p(t, x, z) = p_0(t,x) + p_1(t,x)·φ_1(ζ) + p_2(t,x)·φ_2(ζ)  (N_p = 2)

with ``ζ = (z − b)/h`` and shifted Legendre on [0, 1]:
    φ_0 = 1,   φ_1 = 1 − 2ζ,   φ_2 = 1 − 6ζ + 6ζ².

Hydrostatic pressure is absorbed into ``g·∂_x η``; ``p`` is the
non-hydrostatic remainder.  Surface BC: ``p(η) = 0`` ⇒ p_0 − p_1 + p_2 = 0.

Form of NS used (caller-applied product rule + continuity, the user's
preferred way of obtaining a clean conservative form):

  cont:  ∂_x u + ∂_z w = 0
  xmom:  ∂_t u + ∂_x(u²) + ∂_z(u w) + g ∂_x η + ∂_x p = 0
  zmom:  ∂_t w + ∂_x(u w) + ∂_z(w²)             + ∂_z p = 0

The conservative form is obtained from the standard non-conservative
NS by adding ``u·(∂_x u + ∂_z w) = 0`` to xmom and
``w·(∂_x u + ∂_z w) = 0`` to zmom (each = 0 by continuity).  Algebra:
``u ∂_x u + (∂_x u² − u ∂_x u) = ∂_x u²``,  etc.

Workflow:

  1. Setup symbols + ansatz.
  2. Project continuity against φ_0, φ_1, φ_2.
  3. KBCs at z = b and z = η.
  4. Verify Escalante eq (4) row 1, eq (5) I_1, I_2, w_2 closure as
     explicit linear combinations of the projected continuity + KBCs.
  5. Project x-mom against φ_0; verify eq (4) row 2 (after surface BC).
  6. Project z-mom against φ_0; verify eq (4) row 3 (after surface BC).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic import polynomial_integrate as poly_int


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
z = sp.Symbol("z", real=True)
xi = sp.Symbol("xi", real=True)
g = sp.Symbol("g", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b

u_0 = sp.Function("u_0", real=True)(t, x)
u_1 = sp.Function("u_1", real=True)(t, x)
w_0 = sp.Function("w_0", real=True)(t, x)
w_1 = sp.Function("w_1", real=True)(t, x)
w_2 = sp.Function("w_2", real=True)(t, x)
p_0 = sp.Function("p_0", real=True)(t, x)
p_1 = sp.Function("p_1", real=True)(t, x)
p_2 = sp.Function("p_2", real=True)(t, x)

# Basis at ζ(z).
zeta = (z - b) / h
phi_0 = sp.S.One
phi_1 = 1 - 2 * zeta
phi_2 = 1 - 6 * zeta + 6 * zeta**2

u_ansatz = u_0 * phi_0 + u_1 * phi_1
w_ansatz = w_0 * phi_0 + w_1 * phi_1 + w_2 * phi_2
p_ansatz = p_0 * phi_0 + p_1 * phi_1 + p_2 * phi_2

# Conservative-form NS (caller has already applied product rule +
# continuity to get rid of opaque chain-rule terms after projection):
cont_lhs = sp.Derivative(u_ansatz, x).doit() + sp.Derivative(w_ansatz, z).doit()
xmom_lhs = (sp.Derivative(u_ansatz, t).doit()
            + sp.Derivative(u_ansatz**2, x).doit()
            + sp.Derivative(u_ansatz * w_ansatz, z).doit()
            + g * sp.Derivative(eta, x).doit()
            + sp.Derivative(p_ansatz, x).doit())
zmom_lhs = (sp.Derivative(w_ansatz, t).doit()
            + sp.Derivative(u_ansatz * w_ansatz, x).doit()
            + sp.Derivative(w_ansatz**2, z).doit()
            + sp.Derivative(p_ansatz, z).doit())


# ---------------------------------------------------------------------------
# 2. Galerkin projection helper.
# ---------------------------------------------------------------------------

def project_z(integrand, basis):
    """∫_b^η (basis · integrand) dz  via affine map z = ξh + b."""
    integrand_xi = sp.expand((basis * integrand).xreplace({z: xi * h + b}).doit())
    return poly_int(integrand_xi * h, xi, 0, 1)


# ---------------------------------------------------------------------------
# 3. Project continuity against φ_0, φ_1, φ_2.
# ---------------------------------------------------------------------------

print("=" * 70)
print("VAM (1, 2) derivation in physical (t, x, z) — Escalante 2024")
print("=" * 70)
print()
print("Step 2: project continuity.")
cont_j0 = sp.expand(project_z(cont_lhs, phi_0))
cont_j1 = sp.expand(project_z(cont_lhs, phi_1))
cont_j2 = sp.expand(project_z(cont_lhs, phi_2))


# ---------------------------------------------------------------------------
# 4. KBCs at z = b and z = η.
# ---------------------------------------------------------------------------

print("Step 3: KBCs.")
kbc_bot_lhs = sp.expand(
    w_ansatz.xreplace({z: b}) - u_ansatz.xreplace({z: b}) * sp.Derivative(b, x)
)
kbc_top_lhs = sp.expand(
    w_ansatz.xreplace({z: eta})
    - sp.Derivative(eta, t).doit()
    - u_ansatz.xreplace({z: eta}) * sp.Derivative(eta, x).doit()
)


# ---------------------------------------------------------------------------
# 5. Reference equations from Escalante 2024 eq (4) and eq (5)
# ---------------------------------------------------------------------------

ref_cont_j0 = sp.Derivative(h, t) + sp.Derivative(h * u_0, x)
ref_I_1 = (h * sp.Derivative(u_0, x)
           + sp.Rational(1, 3) * sp.Derivative(h * u_1, x)
           + sp.Rational(1, 3) * u_1 * sp.Derivative(h, x)
           + 2 * (w_0 - u_0 * sp.Derivative(b, x)))
ref_I_2 = (h * sp.Derivative(u_0, x)
           + u_1 * sp.Derivative(h, x)
           + 2 * (u_1 * sp.Derivative(b, x) - w_1))
ref_w2_rhs = -(w_0 + w_1) + (u_0 + u_1) * sp.Derivative(b, x)
ref_xmom_j0 = (sp.Derivative(h * u_0, t)
               + sp.Derivative(h * u_0**2 + sp.Rational(1, 3) * h * u_1**2 + h * p_0, x)
               + g * h * sp.Derivative(eta, x)
               + 2 * p_1 * sp.Derivative(b, x))
ref_zmom_j0 = (sp.Derivative(h * w_0, t)
               + sp.Derivative(h * u_0 * w_0 + sp.Rational(1, 3) * h * u_1 * w_1, x)
               - 2 * p_1)


# ---------------------------------------------------------------------------
# 6. Comparison machinery.
# ---------------------------------------------------------------------------

def normalise(expr, surface_bc=False, kbc_w2=False):
    """Expand all derivatives, set ∂_t b = 0, optionally apply
    surface BC ``p_2 = p_1 − p_0`` and the w_2-closure from KBC at the
    bottom (used to eliminate w_2 from xmom/zmom projections, since the
    paper writes them in a w_2-free form)."""
    e = sp.expand(expr.doit())
    e = e.subs({sp.Derivative(b, t): 0})
    if surface_bc:
        e = e.subs({p_2: p_1 - p_0})
    if kbc_w2:
        # KBC at z = b solved for w_2: w_2 = -(w_0 + w_1) + (u_0+u_1) ∂_x b.
        e = e.subs({w_2: -(w_0 + w_1) + (u_0 + u_1) * sp.Derivative(b, x)})
    return sp.expand(e.doit())


def check(label, derived, reference, surface_bc=False, kbc_w2=False):
    diff = sp.simplify(normalise(derived, surface_bc, kbc_w2)
                       - normalise(reference, surface_bc, kbc_w2))
    mark = "✓ MATCH" if diff == 0 else "✗ MISMATCH"
    print(f"{label}: {mark}")
    if diff != 0:
        print(f"  diff = {diff}")
    return diff == 0


# ---------------------------------------------------------------------------
# 7. Verify references as explicit linear combinations
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Verification — references as explicit combinations of derived eqs")
print("=" * 70)

# Eq (4) row 1 = cont_j0 − (KBC_top − KBC_bot)
# (Reasoning: cont_j0 is depth-integrated cont via affine map; KBC_top
# − KBC_bot brings in ∂_t h via w(η)/w(b) substitution.)
eq4_row1_combo = cont_j0 - (kbc_top_lhs - kbc_bot_lhs)
check("Eq (4) row 1 = cont_j0 − (KBC_top − KBC_bot)",
      eq4_row1_combo, ref_cont_j0)

# Eq (5) I_1 = cont_j0 + cont_j1 + 2·KBC_bot
I_1_combo = cont_j0 + cont_j1 + 2 * kbc_bot_lhs
check("Eq (5) I_1 = cont_j0 + cont_j1 + 2·KBC_bot",
      I_1_combo, ref_I_1)

# Eq (5) I_2: KBC_top − KBC_bot with ∂_t h substituted via eq (4) row 1.
# (Substituting -∂_t h via dt_h_rule converts -∂_t h → ∂_x(h u_0), which
# then expands and combines with the remaining terms to give I_2.)
dt_h_rule = {sp.Derivative(h, t): -sp.Derivative(h * u_0, x).doit()}
I_2_combo = sp.expand((kbc_top_lhs - kbc_bot_lhs).xreplace(dt_h_rule).doit())
check("Eq (5) I_2 = (KBC_top − KBC_bot) with ∂_t h subst",
      I_2_combo, ref_I_2)

# w_2 closure: KBC_bot solved for w_2.
w2_sol = sp.solve(kbc_bot_lhs, w_2)
assert len(w2_sol) == 1
check("Eq (5) w_2 closure (KBC_bot solved for w_2)",
      w2_sol[0], ref_w2_rhs)

# Eq (4) row 2: x-mom j=0.
# The conservative-form physical-z projection of x-mom contains a
# polynomial expansion of ∂_z(u w) that naturally evaluates to
# (u w)|_η − (u w)|_b.  The σ-coord form of the paper has KBCs built
# into its boundary atoms, so its projection produces a w-free
# expression directly.  In physical-z the equivalent is to *add* the
# explicit boundary-KBC contribution u(η)·(KBC_top − KBC_bot), which
# converts (u w)|_{b,η} into u·∂_t h + u²·∂_x η/∂_x b atoms.
u_at_eta = u_ansatz.xreplace({z: eta})
xmom_j0_raw = sp.expand(project_z(xmom_lhs, phi_0))
xmom_j0 = sp.expand(xmom_j0_raw - u_at_eta * (kbc_top_lhs - kbc_bot_lhs))
check("Eq (4) row 2 (x-mom j=0, with u(η)·(KBC_top − KBC_bot) correction)",
      xmom_j0, ref_xmom_j0, surface_bc=True, kbc_w2=True)

# Eq (4) row 3: z-mom j=0.  Same idea: the boundary correction is
# w(η)·(KBC_top − KBC_bot).
w_at_eta = w_ansatz.xreplace({z: eta})
zmom_j0_raw = sp.expand(project_z(zmom_lhs, phi_0))
zmom_j0 = sp.expand(zmom_j0_raw - w_at_eta * (kbc_top_lhs - kbc_bot_lhs))
check("Eq (4) row 3 (z-mom j=0, with w(η)·(KBC_top − KBC_bot) correction)",
      zmom_j0, ref_zmom_j0, surface_bc=True, kbc_w2=True)
