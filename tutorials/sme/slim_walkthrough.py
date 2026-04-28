"""SME L=1 walkthrough — transparent physical-z derivation.

This is a step-by-step walkthrough of the level-1 Shallow Moment
Equations (SME) derivation, matching Kowalski & Torrilhon (2019)
eq (4.14).  We start from the inviscid, hydrostatic Navier-Stokes
system in *physical* (t, x, z) coordinates and end with the projected
moment equations.

Every math step uses ONLY:

  - sympy ``Symbol``, ``Function``, ``Derivative``, ``Integral``,
    ``solve``, ``xreplace``, ``expand``, ``simplify``;
  - the calculus primitives in ``zoomy_core.symbolic``:
      ``leibniz_general``, ``fundamental_theorem``,
      ``polynomial_integrate``.

No model-specific helper does anything implicitly.  Every step is
written out so the reader can see exactly what's happening.  This is
*route A*: close ``w(z)`` via depth-integrated continuity (so the only
ansatz we need is the M=1 polynomial expansion of ``u``).

Pipeline:

  1. Setup: NS continuity ``∂_x u + ∂_z w = 0`` and inviscid /
     hydrostatic x-momentum ``∂_t u + u ∂_x u + w ∂_z u + g ∂_x η = 0``.
  2. Integrate continuity ``∫_b^z`` — apply Leibniz to the ``∂_x u``
     part and the fundamental theorem to the ``∂_z w`` part.
  3. Apply the bottom KBC ``w(b) = u(b) · ∂_x b`` and solve for
     ``w(z)``.
  4. Substitute ``w(z)`` into x-momentum.
  5. Substitute the M=1 ``u`` ansatz
     ``u = u_0 + u_1 · (1 − 2(z−b)/h)`` and Galerkin-project against
     ``φ_j(z) = P_j(1 − 2(z−b)/h)``, ``j ∈ {0, 1}``.
  6. Apply ``∂_t h`` substitution from the depth-averaged continuity
     ``∂_t h + ∂_x(h u_0) = 0``.
  7. Verify against K&T 2019 eq (4.14) row 2 (j=0) and row 3 (j=1).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic import (
    leibniz_general, fundamental_theorem, polynomial_integrate as poly_int,
)


# ---------------------------------------------------------------------------
# 1. Setup — pointwise NS in physical (t, x, z)
# ---------------------------------------------------------------------------

t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
z = sp.Symbol("z", real=True)
xi = sp.Symbol("xi", real=True)
g = sp.Symbol("g", positive=True)
h = sp.Function("h", real=True)(t, x)
b = sp.Function("b", real=True)(x)
eta = h + b

u = sp.Function("u", real=True)(t, x, z)
w = sp.Function("w", real=True)(t, x, z)

# M = 1 polynomial ansatz coefficients (functions of (t, x) only).
u_0 = sp.Function("u_0", real=True)(t, x)
u_1 = sp.Function("u_1", real=True)(t, x)
# Shifted Legendre on [0, 1] (paper convention φ_i(0) = 1).
xi_arg = (z - b) / h
phi_0_z = sp.S.One
phi_1_z = 1 - 2 * xi_arg

# x-momentum (physical-z, hydrostatic, inviscid):
xmom = (sp.Derivative(u, t)
        + u * sp.Derivative(u, x)
        + w * sp.Derivative(u, z)
        + g * sp.Derivative(eta, x))

# Continuity:
cont = sp.Derivative(u, x) + sp.Derivative(w, z)

print("=" * 70)
print("SME L=1 walkthrough — transparent physical-z derivation")
print("=" * 70)
print(f"Continuity: {cont} = 0")
print(f"x-mom    : {sp.expand(xmom)} = 0")


# ---------------------------------------------------------------------------
# 2. Integrate continuity from b to z to get an expression for w(z)
# ---------------------------------------------------------------------------
# We want w(z) so we can plug it into the x-momentum equation.  Integrate
# the pointwise continuity from b to z, using a fresh dummy z' for the
# inner integration variable.

print()
print("Step 2: integrate continuity from b to z.")

z_prime = sp.Symbol("z_prime", real=True)
u_zp = u.xreplace({z: z_prime})
w_zp = w.xreplace({z: z_prime})

# ∫_b^z ∂_x u(t, x, z') dz'  via Leibniz (var=z' is integration variable;
# y=x is the differentiation variable that the limits depend on).
int_dx_u = leibniz_general(sp.Derivative(u_zp, x), z_prime, b, z)
# ∫_b^z ∂_z' w(t, x, z') dz' = w(t, x, z) − w(t, x, b)  via FTOC.
int_dz_w = fundamental_theorem(sp.Derivative(w_zp, z_prime), z_prime, b, z)

cont_integrated = sp.expand(int_dx_u + int_dz_w)
print(f"  After Leibniz + FTOC, integrated cont = {cont_integrated} = 0")


# ---------------------------------------------------------------------------
# 3. Apply bottom KBC: w(b) = u(b) · ∂_x b
# ---------------------------------------------------------------------------

print()
print("Step 3: apply KBC at z = b: w(b) = u(b) · ∂_x b.")
kbc_bottom = {
    w.xreplace({z: b}): u.xreplace({z: b}) * sp.Derivative(b, x),
}
cont_after_kbc = sp.expand(cont_integrated.xreplace(kbc_bottom))
print(f"  After KBC: {cont_after_kbc} = 0")


# ---------------------------------------------------------------------------
# 4. Solve for w(z)
# ---------------------------------------------------------------------------

print()
print("Step 4: solve for w(z).")
w_solution = sp.solve(cont_after_kbc, w)[0]
print(f"  w(z) = {sp.simplify(w_solution)}")


# ---------------------------------------------------------------------------
# 5. Substitute w(z) into x-momentum
# ---------------------------------------------------------------------------

print()
print("Step 5: substitute w(z) into x-momentum.")
xmom_with_w = xmom.xreplace({w: w_solution})
xmom_with_w = sp.expand(xmom_with_w.doit())
print("  (expression too long to print verbatim; will project against φ_j next)")


# ---------------------------------------------------------------------------
# 6. Insert u-ansatz at every z-argument, then project against φ_j(z)
# ---------------------------------------------------------------------------
# After step 5, the x-momentum still contains ``u(t, x, ·)`` atoms with
# different z-arguments: the running integrand has ``u(t, x, z')`` and
# the boundary terms have ``u(t, x, b)`` / ``u(t, x, η)``.  Each is a
# distinct Function-call; ``xreplace({u: u_ansatz})`` doesn't catch them
# all.  Use ``replace`` on the Function class to rewrite every ``u(...)``
# call into the ansatz at *its* z-argument.

print()
print("Step 6: substitute u ansatz + project against φ_j(z).")

u_func = u.func


def _ansatz_at(e):
    """Return the M=1 u-ansatz evaluated at the z-argument of ``e``."""
    z_val = e.args[2]
    return u_0 + u_1 * (1 - 2 * (z_val - b) / h)


xmom_full = xmom_with_w.replace(
    lambda e: isinstance(e, sp.Function) and e.func == u_func,
    _ansatz_at,
)
# Now everything is explicit polynomial in z (and z' inside any held
# Integrals).  doit() evaluates the inner ∫_b^z dz' atoms.
xmom_full = sp.expand(xmom_full.doit())


def project_z(integrand, j):
    """Multiply by φ_j(z), substitute z = ξh + b, integrate dz = h dξ over [0, 1]."""
    phi_j = phi_0_z if j == 0 else phi_1_z
    integrand_xi = sp.expand((phi_j * integrand).xreplace({z: xi * h + b}).doit())
    return poly_int(integrand_xi * h, xi, 0, 1)


print(f"  Projecting against φ_0 ...")
xmom_j0 = project_z(xmom_full, 0)
print(f"  Projecting against φ_1 ...")
xmom_j1 = project_z(xmom_full, 1)


# ---------------------------------------------------------------------------
# 7. Apply ∂_t h substitution from depth-averaged continuity
# ---------------------------------------------------------------------------
# The j=0 projection of continuity gives ``∂_t h + ∂_x(h u_0) = 0``,
# so we can eliminate ``∂_t h`` from the x-momentum projections.

dt_h_atom = sp.Derivative(h, t)
dt_h_rule = {dt_h_atom: -sp.Derivative(h * u_0, x)}


def apply_dt_h(eq):
    """Iteratively substitute ``∂_t h → −∂_x(h u_0)`` to fixpoint."""
    prev = None
    cur = sp.expand(eq.doit())
    while prev != cur:
        prev = cur
        cur = sp.expand(cur.xreplace(dt_h_rule).doit())
    return cur


# K&T 2019 setup is flat-bottom; apply that here too.
flat = {sp.Derivative(b, x): sp.S.Zero}
xmom_j0_simpl = apply_dt_h(xmom_j0.xreplace(flat))
xmom_j1_simpl = apply_dt_h(xmom_j1.xreplace(flat))


# ---------------------------------------------------------------------------
# 8. Verify against K&T 2019 eq (4.14)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Verification against K&T 2019 eq (4.14)")
print("=" * 70)

# K&T row 2: ∂_t(h u_m) + ∂_x(h u_m² + h s²/3 + g h²/2) = 0.
# In our notation u_m = u_0, s = u_1.
ref_row2 = (sp.Derivative(h * u_0, t)
            + sp.Derivative(h * u_0**2 + h * u_1**2 / 3 + g * h**2 / 2, x))
ref_row2_simpl = apply_dt_h(ref_row2.xreplace(flat))

# K&T row 3: ∂_t(h s) + ∂_x(2 h u_m s) − u_m ∂_x(h s) = 0.  Multiplied
# by 1/3 (the μ_1 mass-matrix factor) since our projection has it.
ref_row3 = (sp.Derivative(h * u_1, t) / 3
            + sp.Derivative(2 * h * u_0 * u_1, x) / 3
            - u_0 * sp.Derivative(h * u_1, x) / 3)
ref_row3_simpl = apply_dt_h(ref_row3.xreplace(flat))


def _check(label, derived, reference):
    diff = sp.simplify(derived - reference)
    mark = "✓ MATCH" if diff == 0 else "✗ MISMATCH"
    print(f"{label}: {mark}")
    if diff != 0:
        print(f"  derived   = {derived}")
        print(f"  reference = {reference}")
        print(f"  diff      = {diff}")


_check("x-mom j=0 vs K&T row 2", xmom_j0_simpl, ref_row2_simpl)
_check("x-mom j=1 vs K&T row 3", xmom_j1_simpl, ref_row3_simpl)
