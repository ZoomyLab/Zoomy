"""Multilayer Shallow Water derivation — Aguillon, Hörnschemeyer,
Sainte-Marie (2026) "Barotropic-Baroclinic Splitting for Multilayer
Shallow Water Models with Exchanges", arXiv:2601.16709.

Derives **eq (5)** of the paper using the new
``zoomy_core.symbolic`` primitive layer:

    ∂_t h_α  +  ∂_x(h_α u_α)        = G_{α+1/2} − G_{α-1/2}
    ∂_t(h_α u_α) + ∂_x(h_α u_α²) + g h_α ∂_x h
        = −g h_α ∂_x z_b
            +  u_{α+1/2} G_{α+1/2}  −  u_{α-1/2} G_{α-1/2}
    ∂_t(h_α T_α) + ∂_x(h_α u_α T_α)
        =  T_{α+1/2} G_{α+1/2}  −  T_{α-1/2} G_{α-1/2}

starting from Aguillon eq (1) — inviscid Euler equations with a
passive tracer.  The user-specified rules:

* The basis is **Heaviside-type** in z: each layer is a constant
  region.  Initially the basis is treated as a generic indicator
  function in z.  Affine map per layer onto the reference interval
  [0, 1] when integrating.
* Use kinematic BCs at every interface.  The KBC at z_{α+1/2}
  reads (paper convention, ρ_0 = 1):

      G_{α+1/2} = ∂_t z_{α+1/2} + u_{α+1/2} · ∂_x z_{α+1/2} − w_{α+1/2}.

  At the bottom (α = 0) and surface (α = N), G ≡ 0 (no flux
  through fixed bottom; closed surface).

* Do NOT substitute ∂_t h_α via continuity (paper convention; we
  keep ∂_t h_α as a primary unknown for now).  This matches the
  "in their derivation, they did NOT replace the resulting time
  derivatives with the h_i terms" instruction.

* The projection approximation that makes the derivation closable:

      1/h_α · ∫_{z_{α-1/2}}^{z_{α+1/2}} r·s dz  ≈  r_α · s_α     ∀ r, s

  i.e. the layer-mean of a product is approximated by the product
  of layer-means.  Linear-in-z velocity profiles within a layer
  give the next-order SME correction; this script does the
  *level-0-per-layer* (constant-in-z within each layer) baseline.

Outputs the derived equations side-by-side with the Aguillon eq (5)
reference.  Exit code 0 if they match.
"""
from __future__ import annotations

import argparse
import sys
import sympy as sp

from zoomy_core.symbolic import (
    D,
    Int,
    canonicalise,
    distribute_derivative_over_add,
    fundamental_theorem,
    leibniz_general,
    polynomial_integrate,
    subst,
)


# ---------------------------------------------------------------------------
# Symbols and per-layer state
# ---------------------------------------------------------------------------

t, x, z = sp.symbols("t x z", real=True)
g = sp.Symbol("g", positive=True)
rho0 = sp.Symbol(r"\rho_0", positive=True)


def build_layer_state(N):
    """Construct symbolic state for an N-layer column.

    Returns a Zstruct-like dict with:
      * ``z_iface[α]`` for α ∈ {1/2, 3/2, …, N+1/2}:  interface heights.
        ``z_iface[0]`` = bottom z_b, ``z_iface[N]`` = surface η.
      * ``h[α]`` for α ∈ {1, …, N}: layer height = z_{α+1/2} − z_{α−1/2}.
      * ``u[α]``, ``T[α]``: layer-averaged velocity, tracer.
      * ``u_iface[α]``, ``T_iface[α]`` for α ∈ {1/2, …, N+1/2}:
        interface values (held opaque — upwind closure per Aguillon
        eq (9), but we don't pin a specific upwind here).
      * ``G[α]`` for α ∈ {1/2, …, N+1/2}: mass-exchange terms.  At
        α = 1/2 (bottom) and α = N+1/2 (surface) these are zero by
        BC; we still carry them symbolically and substitute the
        zero-flux conditions at the end.
    """
    z_iface = [None] * (N + 1)
    z_iface[0] = sp.Function("z_b", real=True)(t, x)
    for k in range(1, N):
        z_iface[k] = sp.Function(f"z_{2*k+1}_over_2", real=True)(t, x)
    z_iface[N] = sp.Function(r"\eta", real=True)(t, x)

    h = {}
    for alpha in range(1, N + 1):
        h[alpha] = z_iface[alpha] - z_iface[alpha - 1]

    u = {alpha: sp.Function(f"u_{alpha}", real=True)(t, x)
         for alpha in range(1, N + 1)}
    T = {alpha: sp.Function(f"T_{alpha}", real=True)(t, x)
         for alpha in range(1, N + 1)}

    # Interface velocities and tracers (held opaque).  Half-integer
    # indexed: stored as int 2*α so 'u_iface[1]' = u_{1/2}.
    u_iface = {}
    T_iface = {}
    G = {}
    for two_alpha in range(1, 2 * N + 2, 2):
        # half-integer α = two_alpha / 2
        u_iface[two_alpha] = sp.Function(f"u_{two_alpha}_over_2", real=True)(t, x)
        T_iface[two_alpha] = sp.Function(f"T_{two_alpha}_over_2", real=True)(t, x)
        G[two_alpha] = sp.Function(f"G_{two_alpha}_over_2", real=True)(t, x)

    return {
        "N": N,
        "z_iface": z_iface,    # 0..N (integer interfaces; interface_α index = α-1/2 with α = 1..N+1)
        "h": h,                 # 1..N
        "u": u,                 # 1..N
        "T": T,                 # 1..N
        "u_iface": u_iface,     # half-integers as 2α: 1, 3, ..., 2N+1
        "T_iface": T_iface,
        "G": G,
    }


# ---------------------------------------------------------------------------
# Per-layer projection — derive eq (5) of Aguillon et al.
# ---------------------------------------------------------------------------

def replace_integrals_structurally(expr, integrand_template, var_template,
                                   z_lo, z_hi, replacement):
    """Walk ``expr`` and replace every ``Integral(integrand, (var, z_lo,
    z_hi))`` atom with ``replacement``, where ``integrand_template`` is
    matched modulo alpha-renaming of ``var_template``.

    Used for the moment-substitution step: after ``canonicalise`` has
    alpha-renamed the Integral bound variables to canonical Dummies,
    we still need to substitute ``∫u dz`` → ``h_α u_α`` etc., and a
    plain ``xreplace`` won't match because the bound name differs.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _walk(e):
        if isinstance(e, sp.Integral):
            limits = e.args[1]
            if (hasattr(limits, "__len__") and len(limits) == 3
                    and limits[1] == z_lo and limits[2] == z_hi):
                bound_var = limits[0]
                # Compare integrand modulo alpha-renaming of var_template → bound_var
                target = integrand_template.xreplace({var_template: bound_var})
                if e.args[0] == target:
                    return replacement
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    return _walk(expr)


def derive_layer_continuity(state, alpha):
    """Layer-α mass conservation: integrate

        ∂_x u + ∂_z w = 0

    over z ∈ [z_{α−1/2}, z_{α+1/2}].

    Apply Leibniz on the ∂_x u term and FT on ∂_z w, then substitute
    the kinematic BC at each interface.  Result: eq (2) row 1.
    """
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    G_lo = state["G"][2 * alpha - 1]  # G_{α-1/2}
    G_hi = state["G"][2 * alpha + 1]  # G_{α+1/2}
    u_lo = state["u_iface"][2 * alpha - 1]
    u_hi = state["u_iface"][2 * alpha + 1]
    # Opaque ``u(t, x, z)`` and ``w(t, x, z)`` — primitives integrate
    # them without needing closed forms.
    u_field = sp.Function("u", real=True)(t, x, z)
    w_field = sp.Function("w", real=True)(t, x, z)

    # ∫_{z_lo}^{z_hi} ∂_x u dz  via leibniz_general
    leib_u = leibniz_general(D(u_field, x), z, z_lo, z_hi)
    # ∫_{z_lo}^{z_hi} ∂_z w dz  via fundamental_theorem
    ft_w = fundamental_theorem(D(w_field, z), z, z_lo, z_hi)
    full = leib_u + ft_w

    # Kinematic BCs at the two interfaces (Aguillon convention,
    # ρ_0 = 1 absorbed):
    #   G_{α+1/2} = ∂_t z_{α+1/2} + u_{α+1/2}·∂_x z_{α+1/2} − w(z_{α+1/2})
    # ⇒ w(z_{α+1/2}) = ∂_t z_{α+1/2} + u_{α+1/2}·∂_x z_{α+1/2} − G_{α+1/2}
    kbc = {
        w_field.subs(z, z_hi):
            D(z_hi, t) + u_hi * D(z_hi, x) - G_hi,
        w_field.subs(z, z_lo):
            D(z_lo, t) + u_lo * D(z_lo, x) - G_lo,
    }
    full = subst(full, kbc)

    # Replace boundary u-values with interface velocities.
    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo})

    # Definition of layer-averaged velocity:
    #   ∫_{z_lo}^{z_hi} u dz  ≈  h_α · u_α      (level-0 per-layer ansatz)
    h_alpha = state["h"][alpha]
    u_alpha = state["u"][alpha]
    full = replace_integrals_structurally(
        full, u_field, z, z_lo, z_hi, h_alpha * u_alpha,
    )

    # The result should be:
    #   ∂_t h_α + ∂_x(h_α u_α) = G_{α+1/2} − G_{α-1/2}
    # in the form  (LHS) = 0.
    return canonicalise(full)


def derive_layer_x_momentum(state, alpha):
    """Layer-α x-momentum: integrate (after τ = 0)

        ∂_t u + ∂_x u² + ∂_z(u·w) + (1/ρ₀) ∂_x p  =  0

    over z ∈ [z_{α−1/2}, z_{α+1/2}].  ``ρ₀`` constant.  Apply Leibniz
    on ∂_t and ∂_x terms, FT on ∂_z, kinematic BC at interfaces.
    Use hydrostatic pressure  p = ρ₀ g (η − z)  + atmospheric  ≈ 0.
    """
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    eta = state["z_iface"][state["N"]]
    z_b = state["z_iface"][0]
    G_lo = state["G"][2 * alpha - 1]
    G_hi = state["G"][2 * alpha + 1]
    u_lo = state["u_iface"][2 * alpha - 1]
    u_hi = state["u_iface"][2 * alpha + 1]
    u_field = sp.Function("u", real=True)(t, x, z)
    w_field = sp.Function("w", real=True)(t, x, z)
    p_field = sp.Function("p", real=True)(t, x, z)

    # ∂_t u term — Leibniz w.r.t. t (var = z, diff_var = t)
    term_dt = leibniz_general(D(u_field, t), z, z_lo, z_hi)
    # ∂_x u² term — Leibniz w.r.t. x
    term_dx_u2 = leibniz_general(D(u_field**2, x), z, z_lo, z_hi)
    # ∂_z(u·w) term — fundamental theorem
    term_dz_uw = fundamental_theorem(D(u_field * w_field, z), z, z_lo, z_hi)
    # (1/ρ₀) ∂_x p — substitute hydrostatic ``p = ρ₀ g (η − z)`` first.
    # ∂_x p = ρ₀·g·(∂_x η − ∂_x z) = ρ₀·g·∂_x η  (since ∂_x z = 0,
    # treating z as a free coordinate).  Divide by ρ₀: g·∂_x η.
    # This is z-independent so the layer integral is just g·h_α·∂_x η.
    # We compute it explicitly to avoid having to push sympy through
    # a held Derivative(ρ₀·g·(η-z), x).
    pressure_grad_per_rho = g * D(eta, x)   # = (1/ρ₀)·∂_x p
    term_dx_p_per_rho = polynomial_integrate(pressure_grad_per_rho,
                                             z, z_lo, z_hi)

    full = term_dt + term_dx_u2 + term_dz_uw + term_dx_p_per_rho

    # Kinematic BCs at interfaces.
    kbc = {
        w_field.subs(z, z_hi):
            D(z_hi, t) + u_hi * D(z_hi, x) - G_hi,
        w_field.subs(z, z_lo):
            D(z_lo, t) + u_lo * D(z_lo, x) - G_lo,
    }
    full = subst(full, kbc)

    # Replace boundary u-values with interface velocities.
    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo})

    # Definition of layer-averaged moments + level-0 closure
    # (mean of product = product of means):
    #
    #   ∫ u dz   = h_α · u_α
    #   ∫ u² dz  = h_α · u_α²       (the closure approximation)
    h_alpha = state["h"][alpha]
    u_alpha = state["u"][alpha]
    full = replace_integrals_structurally(
        full, u_field, z, z_lo, z_hi, h_alpha * u_alpha,
    )
    full = replace_integrals_structurally(
        full, u_field**2, z, z_lo, z_hi, h_alpha * u_alpha**2,
    )

    return canonicalise(full)


def derive_layer_tracer(state, alpha):
    """Layer-α tracer balance: integrate

        ∂_t T + ∂_x(u T) + ∂_z(w T) = 0

    over the layer.  Same structure as continuity / x-momentum.
    """
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    G_lo = state["G"][2 * alpha - 1]
    G_hi = state["G"][2 * alpha + 1]
    u_lo = state["u_iface"][2 * alpha - 1]
    u_hi = state["u_iface"][2 * alpha + 1]
    T_lo = state["T_iface"][2 * alpha - 1]
    T_hi = state["T_iface"][2 * alpha + 1]
    u_field = sp.Function("u", real=True)(t, x, z)
    w_field = sp.Function("w", real=True)(t, x, z)
    T_field = sp.Function("T", real=True)(t, x, z)

    term_dt = leibniz_general(D(T_field, t), z, z_lo, z_hi)
    term_dx = leibniz_general(D(u_field * T_field, x), z, z_lo, z_hi)
    term_dz = fundamental_theorem(D(w_field * T_field, z), z, z_lo, z_hi)

    full = term_dt + term_dx + term_dz

    # KBCs at interfaces.
    kbc = {
        w_field.subs(z, z_hi):
            D(z_hi, t) + u_hi * D(z_hi, x) - G_hi,
        w_field.subs(z, z_lo):
            D(z_lo, t) + u_lo * D(z_lo, x) - G_lo,
    }
    full = subst(full, kbc)

    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo,
                        T_field.subs(z, z_hi): T_hi,
                        T_field.subs(z, z_lo): T_lo})

    h_alpha = state["h"][alpha]
    u_alpha = state["u"][alpha]
    T_alpha = state["T"][alpha]
    full = replace_integrals_structurally(
        full, T_field, z, z_lo, z_hi, h_alpha * T_alpha,
    )
    full = replace_integrals_structurally(
        full, u_field * T_field, z, z_lo, z_hi, h_alpha * u_alpha * T_alpha,
    )

    return canonicalise(full)


# ---------------------------------------------------------------------------
# Aguillon eq (5) reference — written literally.
# ---------------------------------------------------------------------------

def aguillon_eq5_reference(state, alpha):
    """Return ``(continuity_lhs, x_momentum_lhs, tracer_lhs)`` for
    layer α as written in eq (5):

        ∂_t h_α + ∂_x(h_α u_α) = G_{α+1/2} − G_{α-1/2}
        ∂_t(h_α u_α) + ∂_x(h_α u_α²) + g h_α ∂_x h
            = −g h_α ∂_x z_b + u_{α+1/2} G_{α+1/2} − u_{α-1/2} G_{α-1/2}
        ∂_t(h_α T_α) + ∂_x(h_α u_α T_α)
            = T_{α+1/2} G_{α+1/2} − T_{α-1/2} G_{α-1/2}
    """
    h_alpha = state["h"][alpha]
    u_alpha = state["u"][alpha]
    T_alpha = state["T"][alpha]
    z_b = state["z_iface"][0]
    h_total = sum(state["h"][a] for a in range(1, state["N"] + 1))
    G_lo = state["G"][2 * alpha - 1]
    G_hi = state["G"][2 * alpha + 1]
    u_lo = state["u_iface"][2 * alpha - 1]
    u_hi = state["u_iface"][2 * alpha + 1]
    T_lo = state["T_iface"][2 * alpha - 1]
    T_hi = state["T_iface"][2 * alpha + 1]

    cont = (D(h_alpha, t) + D(h_alpha * u_alpha, x)
            - (G_hi - G_lo))
    momx = (D(h_alpha * u_alpha, t) + D(h_alpha * u_alpha**2, x)
            + g * h_alpha * D(h_total, x)
            + g * h_alpha * D(z_b, x)
            - (u_hi * G_hi - u_lo * G_lo))
    tracer = (D(h_alpha * T_alpha, t) + D(h_alpha * u_alpha * T_alpha, x)
              - (T_hi * G_hi - T_lo * G_lo))
    return cont, momx, tracer


# ---------------------------------------------------------------------------
# Bottom & surface BCs: G_{1/2} = G_{N+1/2} = 0; ∂_t z_b = 0 (fixed)
# ---------------------------------------------------------------------------

def apply_zero_flux_bcs(expr, state):
    """Substitute ``G_{1/2} = G_{N+1/2} = 0`` (no flux at bottom and
    surface) and ``∂_t z_b = 0`` (fixed bottom).

    Note: the reference equations contain ``∂_t h_α = ∂_t(z_{α+1/2}
    − z_{α−1/2})`` which sympy holds as a single Derivative atom.
    Distribute over Add first so the ``∂_t z_b`` piece becomes
    explicit and can be substituted to 0.
    """
    N = state["N"]
    flux_zero = {state["G"][1]: 0, state["G"][2 * N + 1]: 0}
    fixed_b = {D(state["z_iface"][0], t): 0}
    expr = distribute_derivative_over_add(expr)
    return canonicalise(subst(subst(expr, flux_zero), fixed_b))


# ---------------------------------------------------------------------------
# Comparison helpers (same shape as kt2019_verification.py)
# ---------------------------------------------------------------------------

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
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v),) + factors[i+1:]))
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


def kill_zero_derivatives(expr):
    if not isinstance(expr, sp.Basic):
        return expr
    mapping = {}
    for d in expr.atoms(sp.Derivative):
        inner = d.args[0]
        wrt = []
        for v in d.args[1:]:
            wrt.append(v[0] if isinstance(v, (tuple, sp.Tuple)) else v)
        if inner.is_number or not any(inner.has(v) for v in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normal_form(expr, state=None):
    """Expand all Derivative atoms, then (if ``state`` given) re-apply
    the fixed-bottom and zero-flux substitutions on the now-explicit
    atoms, then kill trivially-zero derivatives.
    """
    e = expand_derivatives(expr)
    if state is not None:
        N = state["N"]
        flux_zero = {state["G"][1]: 0, state["G"][2 * N + 1]: 0}
        fixed_b = {D(state["z_iface"][0], t): 0}
        e = subst(subst(e, flux_zero), fixed_b)
    e = kill_zero_derivatives(e)
    return sp.expand(e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, default=2,
                        help="Number of layers (default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on any mismatch")
    args = parser.parse_args()

    N = args.N
    print(f"=== Aguillon-Hörnschemeyer-Sainte-Marie 2026 verification — N = {N} ===\n")

    state = build_layer_state(N)
    n_mismatch = 0

    for alpha in range(1, N + 1):
        print(f"--- Layer α = {alpha} ---")
        # Pipeline derivation.
        cont_pipe = derive_layer_continuity(state, alpha)
        momx_pipe = derive_layer_x_momentum(state, alpha)
        tracer_pipe = derive_layer_tracer(state, alpha)

        # Reference equations from eq (5).
        cont_ref, momx_ref, tracer_ref = aguillon_eq5_reference(state, alpha)

        # Apply zero-flux BCs at bottom and surface, and fixed z_b in time.
        cont_pipe = apply_zero_flux_bcs(cont_pipe, state)
        momx_pipe = apply_zero_flux_bcs(momx_pipe, state)
        tracer_pipe = apply_zero_flux_bcs(tracer_pipe, state)
        cont_ref = apply_zero_flux_bcs(cont_ref, state)
        momx_ref = apply_zero_flux_bcs(momx_ref, state)
        tracer_ref = apply_zero_flux_bcs(tracer_ref, state)

        # Compare in fully-expanded normal form.
        for label, pipe, ref in [
            ("continuity", cont_pipe, cont_ref),
            ("x-momentum", momx_pipe, momx_ref),
            ("tracer",     tracer_pipe, tracer_ref),
        ]:
            diff = sp.expand(normal_form(pipe, state) - normal_form(ref, state))
            ok = (diff == 0)
            mark = "✓ MATCH" if ok else "✗ MISMATCH"
            print(f"  {label:<12} {mark}")
            if not ok:
                n_mismatch += 1
                print(f"    pipe = {sp.expand(normal_form(pipe, state))}")
                print(f"    ref  = {sp.expand(normal_form(ref, state))}")
                print(f"    diff = {diff}")
        print()

    print(f"=== Summary: {n_mismatch} mismatch(es) at N={N} ===")
    if args.strict and n_mismatch > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
