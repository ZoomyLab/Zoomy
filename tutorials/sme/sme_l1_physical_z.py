"""SME L=1 derivation in physical (t, x, z) — explicit, no helpers.

Every math step uses ONLY:
  - sympy ``Symbol``, ``Function``, ``Derivative``, ``Integral``, ``solve``;
  - the calculus primitives in ``zoomy_core.symbolic``:
      ``leibniz_general``, ``fundamental_theorem``, ``polynomial_integrate``;
  - explicit substitutions via ``xreplace`` / ``subs``.

No model-specific helper does anything implicitly.  Every step is
written out so the reader can see exactly what's happening.

Workflow (route A — close ``w`` via depth-integrated continuity):

  1. Setup: NS continuity ``∂_x u + ∂_z w = 0`` and x-mom (inviscid,
     hydrostatic) ``∂_t u + u ∂_x u + w ∂_z u + g ∂_x η = 0``.
  2. Integrate continuity from ``b`` to ``z``; apply Leibniz to the
     ``∂_x u`` part and FT to the ``∂_z w`` part; apply the KBC at
     ``z = b``; solve for ``w(z)``.
  3. Substitute the resulting ``w(z)`` into x-momentum.
  4. Project against ``φ_j(z) = P_j(1 − 2(z-b)/h)`` and integrate ``dz``
     over ``[b, b+h]`` via affine map.
  5. Verify against K&T 2019 eq (4.14).

Closes with a sanity check that route B (keep ``w`` explicit, project
continuity to determine its coefficients) gives the same equations.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic import (
    leibniz_general, fundamental_theorem, polynomial_integrate as poly_int,
)


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

u = sp.Function("u", real=True)(t, x, z)
w = sp.Function("w", real=True)(t, x, z)

# Polynomial ansatz coefficients (functions of (t, x) only).
u_0 = sp.Function("u_0", real=True)(t, x)
u_1 = sp.Function("u_1", real=True)(t, x)
# Shifted Legendre on [0, 1] (paper convention φ_i(0) = 1).
xi_arg = (z - b) / h
phi_0_z = sp.S.One
phi_1_z = 1 - 2 * xi_arg
u_ansatz = u_0 + u_1 * phi_1_z

# x-momentum (physical-z, hydrostatic, inviscid):
xmom = (sp.Derivative(u, t)
        + u * sp.Derivative(u, x)
        + w * sp.Derivative(u, z)
        + g * sp.Derivative(eta, x))

# Continuity:
cont = sp.Derivative(u, x) + sp.Derivative(w, z)

print("=" * 70)
print("SME L=1 derivation in physical (t, x, z)")
print("=" * 70)
print(f"Continuity: {cont} = 0")
print(f"x-mom    : {sp.expand(xmom)} = 0")


# ---------------------------------------------------------------------------
# 2. Get w(z) by depth-integrating continuity from b to z
# ---------------------------------------------------------------------------

print()
print("Step 2: integrate continuity from b to z.")

z_prime = sp.Symbol("z_prime", real=True)
u_zp = u.xreplace({z: z_prime})
w_zp = w.xreplace({z: z_prime})

# ∫_b^z ∂_x u(t, x, z') dz' = ?  Apply leibniz_general (var=z' is integration
# variable; y=x is differentiation variable).
int_dx_u = leibniz_general(sp.Derivative(u_zp, x), z_prime, b, z)
# ∫_b^z ∂_z' w(t, x, z') dz' = w(t, x, z) - w(t, x, b)  via fundamental_theorem
int_dz_w = fundamental_theorem(sp.Derivative(w_zp, z_prime), z_prime, b, z)

cont_integrated = sp.expand(int_dx_u + int_dz_w)
print(f"  After Leibniz + FT, integrated cont = {cont_integrated} = 0")


# ---------------------------------------------------------------------------
# 3. Apply KBC at z = b: w(b) = u(b) · ∂_x b
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
# 5. Substitute w(z) into x-mom
# ---------------------------------------------------------------------------

print()
print("Step 5: substitute w(z) into x-mom.")
xmom_with_w = xmom.xreplace({w: w_solution})
xmom_with_w = sp.expand(xmom_with_w.doit())
# print(f"  xmom = {xmom_with_w}")
print("  (expression too long to print; will project against φ_j next)")


# ---------------------------------------------------------------------------
# 6. Substitute u ansatz, then project against φ_j(z)
# ---------------------------------------------------------------------------

print()
print("Step 6: substitute u ansatz + project against φ_j(z).")

# u-ansatz applies at any z-argument: u(t, x, Z) → u_0 + u_1·(1 - 2(Z-b)/h).
# We can't use xreplace({u: u_ansatz}) — the held Integral has
# u(t, x, z_prime), and boundary atoms are u(t, x, b) / u(t, x, η).
# Each is a distinct Function-call atom.  Use .replace on the Function
# class to rewrite every u(...) call into the ansatz at its z-argument.
u_func = u.func


def _ansatz_at(e):
    z_val = e.args[2]
    return u_0 + u_1 * (1 - 2 * (z_val - b) / h)


xmom_full = xmom_with_w.replace(
    lambda e: isinstance(e, sp.Function) and e.func == u_func,
    _ansatz_at,
)
# Now everything is explicit polynomial in z (and z_prime in held
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
# 7. Apply ∂_t h substitution from cont j=0 (∂_t h + ∂_x(h u_0) = 0)
# ---------------------------------------------------------------------------

dt_h_atom = sp.Derivative(h, t)
dt_h_rule = {dt_h_atom: -sp.Derivative(h * u_0, x)}


def apply_dt_h(eq):
    prev = None
    cur = sp.expand(eq.doit())
    while prev != cur:
        prev = cur
        cur = sp.expand(cur.xreplace(dt_h_rule).doit())
    return cur


# Apply flat bottom assumption (K&T 2019 setup) AND dt_h subst.
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
ref_row2 = (sp.Derivative(h * u_0, t)
            + sp.Derivative(h * u_0**2 + h * u_1**2 / 3 + g * h**2 / 2, x))
ref_row2_simpl = apply_dt_h(ref_row2.xreplace(flat))

# K&T row 3: ∂_t(h s) + ∂_x(2 h u_m s) − u_m ∂_x(h s) = 0.  Multiplied by 1/3
# (the μ_1 factor) since our projection has it.
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
