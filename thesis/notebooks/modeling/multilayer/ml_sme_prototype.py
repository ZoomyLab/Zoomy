"""ML-SME prototype: per-layer Legendre expansion + Heaviside outer basis.

Combines:

* the **multilayer** structure of Aguillon-Hörnschemeyer-Sainte-Marie
  2026 (eq (5)) — Heaviside-type indicator basis in z, layer
  interfaces with kinematic BCs giving mass-exchange terms G_{α±1/2};
* the **shallow-moment** structure of Kowalski-Torrilhon 2019 — per-
  layer Legendre expansion of the velocity profile.

Velocity ansatz at order ``L`` per layer:

    u(t, x, z)  =  Σ_α  𝟙_α(z) · ( u_α(t, x)
                                    + Σ_{k=1}^L α_{k,α}(t, x) · φ_k(ζ_α) )

with ζ_α = (z − z_{α-1/2}) / h_α and φ_k the shifted Legendre basis on
[0, 1].  The Heaviside indicator 𝟙_α(z) selects layer α; integration
over the layer gives Aguillon eq (5) for ``k = 0`` (φ_0 = 1) plus the
per-layer SME closure for ``k ≥ 1``.

This script handles **L = 0** (reproduces Aguillon eq (5) exactly,
which is also what ``aguillon2026_derivation.py`` already does, used
here as a sanity check) and sketches **L = 1** as a prototype where
each layer additionally carries an ``α_{1,α}`` evolution equation.

The L = 1 derivation per layer follows the same recipe as K&T's
single-layer SME but with:

  * integration interval [z_{α-1/2}, z_{α+1/2}] (not [b, b+h]);
  * kinematic BCs at *both* interfaces (not just b and η);
  * mass-exchange terms G_{α±1/2} appearing in the boundary
    contributions.

Treat as a structural prototype — the full level-1 closure within a
layer requires careful bug-3-style fixpoint handling, which is
outside the scope of this overnight pass.  The script's value: it
exercises the per-layer affine map ``z → ζ·h_α + z_{α-1/2}`` and
shows the basis composition.
"""
from __future__ import annotations

import argparse
import sys
import sympy as sp

from zoomy_core.symbolic import (
    D,
    Int,
    affine_change_of_variable,
    canonicalise,
    canonicalize_phi_derivative_subs,
    distribute_derivative_over_add,
    fundamental_theorem,
    function_expand,
    leibniz_general,
    polynomial_integrate,
    product_rule_forward,
    project_basis_integrand,
    split_integral_over_add,
    subst,
)
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.basisfunctions import Legendre_shifted


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

t, x, z = sp.symbols("t x z", real=True)
g = sp.Symbol("g", positive=True)
rho0 = sp.Symbol(r"\rho_0", positive=True)


def build_ml_sme_state(N, L):
    """Construct symbolic state for an N-layer column with per-layer
    Legendre level ``L``.

    ``alpha_moments[α][k]`` holds ``α_{k,α}(t, x)`` for k ∈ {1..L}
    (the level-0 mean lives in ``u[α]``).
    """
    z_iface = [None] * (N + 1)
    z_iface[0] = sp.Function("z_b", real=True)(t, x)
    for k in range(1, N):
        z_iface[k] = sp.Function(f"z_{2*k+1}_over_2", real=True)(t, x)
    z_iface[N] = sp.Function(r"\eta", real=True)(t, x)

    h = {alpha: z_iface[alpha] - z_iface[alpha - 1]
         for alpha in range(1, N + 1)}
    u = {alpha: sp.Function(f"u_{alpha}", real=True)(t, x)
         for alpha in range(1, N + 1)}

    alpha_moments = {alpha: {} for alpha in range(1, N + 1)}
    for alpha in range(1, N + 1):
        for k in range(1, L + 1):
            alpha_moments[alpha][k] = sp.Function(
                f"alpha_{k}_layer{alpha}", real=True
            )(t, x)

    u_iface = {}
    G = {}
    for two_alpha in range(1, 2 * N + 2, 2):
        u_iface[two_alpha] = sp.Function(
            f"u_{two_alpha}_over_2", real=True
        )(t, x)
        G[two_alpha] = sp.Function(
            f"G_{two_alpha}_over_2", real=True
        )(t, x)

    phi_fns = [sp.Function(f"phi_{k}") for k in range(L + 1)]

    return {
        "N": N, "L": L,
        "z_iface": z_iface,
        "h": h, "u": u,
        "alpha": alpha_moments,
        "u_iface": u_iface,
        "G": G,
        "phi_fns": phi_fns,
    }


# ---------------------------------------------------------------------------
# Per-layer ansatz: u = u_α + Σ_k α_{k,α} φ_k(ζ_α)
# ---------------------------------------------------------------------------

def make_layer_ansatz(state, alpha):
    """Return a callable ``ansatz(t, x, arg)`` that evaluates the
    layer-α velocity profile at depth ``arg``.  Used with
    ``function_expand`` to substitute u → ansatz inside Galerkin
    integrals.
    """
    z_lo = state["z_iface"][alpha - 1]
    h_alpha = state["h"][alpha]
    u_alpha = state["u"][alpha]
    phi_fns = state["phi_fns"]
    L = state["L"]

    def _ansatz(*args):
        arg_z = args[-1]
        rhs = u_alpha
        for k in range(1, L + 1):
            rhs = rhs + state["alpha"][alpha][k] * phi_fns[k](
                (arg_z - z_lo) / h_alpha
            )
        return rhs

    return _ansatz


# ---------------------------------------------------------------------------
# Per-layer Galerkin projection — one mode k, one layer α
# ---------------------------------------------------------------------------

def project_layer_continuity(state, alpha, k):
    """Project the continuity equation onto φ_k(ζ_α) inside layer α.

    For ``k = 0`` (φ_0 = 1) reduces to ``∂_t h_α + ∂_x(h_α u_α) = G_{α+1/2}
    − G_{α-1/2}`` — Aguillon eq (5) row 1.  For ``k ≥ 1`` the
    moment equation involves higher-α moments and the per-layer
    affine map ``ζ_α``.
    """
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    G_lo = state["G"][2 * alpha - 1]
    G_hi = state["G"][2 * alpha + 1]
    u_lo = state["u_iface"][2 * alpha - 1]
    u_hi = state["u_iface"][2 * alpha + 1]
    h_alpha = state["h"][alpha]
    phi = state["phi_fns"][k]

    u_field = sp.Function("u", real=True)(t, x, z)
    w_field = sp.Function("w", real=True)(t, x, z)

    # Multiply by φ_k((z − z_lo) / h_α) and integrate ∂_x u + ∂_z w over
    # the layer.
    zeta = (z - z_lo) / h_alpha
    weighted_dx_u = phi(zeta) * D(u_field, x)
    weighted_dz_w = phi(zeta) * D(w_field, z)

    # ∫ φ_k · ∂_x u dz — for general φ_k this isn't a clean Leibniz
    # because the integrand has explicit z-dependence inside φ_k.  We
    # apply the inverse product rule first to expose ∂_x(φ_k·u) form
    # — same trick as K&T's slim_walkthrough step 6.  But for k=0
    # (φ_0 = 1) this reduces directly to the Aguillon case.
    if k == 0:
        # φ_0 = 1: integrand is ∂_x u + ∂_z w; the L=0 Aguillon path.
        leib_u = leibniz_general(D(u_field, x), z, z_lo, z_hi)
        ft_w = fundamental_theorem(D(w_field, z), z, z_lo, z_hi)
        full = leib_u + ft_w
    else:
        # k ≥ 1: not implemented yet — would need ProductRule on
        # weighted form + Leibniz with the weight inside, similar to
        # K&T but per-layer.  This is the stretch path.
        raise NotImplementedError(
            f"L ≥ 1 ML-SME continuity projection (k = {k}) not yet "
            f"implemented — this is the next-step extension."
        )

    # KBC at both interfaces (mass-exchange definition).
    kbc = {
        w_field.subs(z, z_hi):
            D(z_hi, t) + u_hi * D(z_hi, x) - G_hi,
        w_field.subs(z, z_lo):
            D(z_lo, t) + u_lo * D(z_lo, x) - G_lo,
    }
    full = subst(full, kbc)
    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo})

    # Replace ∫u dz over the layer with the layer-mean closure.
    u_alpha = state["u"][alpha]
    full = _replace_integral(full, u_field, z, z_lo, z_hi,
                             h_alpha * u_alpha)

    return canonicalise(full)


def _replace_integral(expr, integrand_template, var_template,
                      z_lo, z_hi, replacement):
    """Walk ``expr`` and replace each ``Integral(integrand, (var, z_lo,
    z_hi))`` whose integrand matches modulo alpha-renaming of
    ``var_template`` with ``replacement``.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, sp.Integral):
            limits = e.args[1]
            if (hasattr(limits, "__len__") and len(limits) == 3
                    and limits[1] == z_lo and limits[2] == z_hi):
                bound_var = limits[0]
                target = integrand_template.xreplace({var_template: bound_var})
                if e.args[0] == target:
                    return replacement
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    return _walk(expr)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--L", type=int, default=0,
                        help="SME level per layer (0 = MLSWE; 1+ = stretch)")
    args = parser.parse_args()

    N, L = args.N, args.L
    print(f"=== ML-SME prototype — N = {N}, L = {L} ===\n")

    state = build_ml_sme_state(N, L)

    print("Layer interfaces:")
    for k, z_iface in enumerate(state["z_iface"]):
        print(f"  z_{2*k+1}/2 = {z_iface}")
    print()

    if L == 0:
        print("Level-0 per layer = MLSWE (Aguillon eq (5) row 1).\n")
        for alpha in range(1, N + 1):
            cont = project_layer_continuity(state, alpha, k=0)
            # Apply zero-flux BCs at bottom + surface.
            flux_zero = {state["G"][1]: 0, state["G"][2 * N + 1]: 0}
            fixed_b = {D(state["z_iface"][0], t): 0}
            cont = canonicalise(
                subst(subst(distribute_derivative_over_add(cont), flux_zero),
                      fixed_b)
            )
            print(f"  layer {alpha} continuity: {cont}")
        print()
    else:
        print("Level-1+ per-layer SME — not yet wired through the per-layer "
              "ProductRule + Leibniz pipeline.")
        print("This is the next-step extension once the L=0 framework is "
              "validated.\n")

    print("=== Notes on the basis composition ===")
    print()
    print("The Heaviside-type indicator basis 𝟙_α(z) selects each layer's")
    print("integration interval [z_{α-1/2}, z_{α+1/2}].  Within a layer, the")
    print("Legendre basis φ_k(ζ_α) with ζ_α = (z − z_{α-1/2}) / h_α expands")
    print("the per-layer velocity profile at order L.  Composition:")
    print()
    print("    u(t, x, z) = Σ_α 𝟙_α(z) · (u_α + Σ_k α_{k,α} φ_k((z − z_{α-1/2})/h_α))")
    print()
    print("For L = 0 this collapses to the multilayer SWE (constant velocity")
    print("within each layer).  Extending to L ≥ 1 requires:")
    print()
    print("  1. Apply Multiply(phi_k(zeta_alpha)) to the layer's momentum.")
    print("  2. ProductRule (inverse) per term to expose ∂_v(φ·f) shape.")
    print("  3. Leibniz / FT per term over [z_{α-1/2}, z_{α+1/2}].")
    print("  4. KBC at both interfaces (gives G_{α±1/2}).")
    print("  5. Affine map z → ζ·h_α + z_{α-1/2}, ζ ∈ [0, 1] — already in")
    print("     primitives (`affine_change_of_variable`).")
    print("  6. Insert ansatz, project basis integrals, close bug-3 fixpoint.")
    print()
    print("All ingredients exist in `zoomy_core.symbolic`.  The remaining")
    print("work is the per-layer driver that composes them, paralleling")
    print("`tutorials/sme/kt2019_verification.py` but with layer-α bounds.")


if __name__ == "__main__":
    main()
