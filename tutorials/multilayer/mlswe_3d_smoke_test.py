"""3D Multilayer SWE smoke test — 2D horizontal (x, y) + N layers.

Extends Aguillon-Hörnschemeyer-Sainte-Marie 2026 eq (2) to the case
with both x and y momentum equations.  Per layer α:

    ∂_t h_α + ∂_x(h_α u_α) + ∂_y(h_α v_α) = G_{α+1/2} − G_{α-1/2}
    ∂_t(h_α u_α) + ∂_x(h_α u_α²) + ∂_y(h_α u_α v_α) + g h_α ∂_x h
        = −g h_α ∂_x z_b + u_{α+1/2} G_{α+1/2} − u_{α-1/2} G_{α-1/2}
    ∂_t(h_α v_α) + ∂_x(h_α u_α v_α) + ∂_y(h_α v_α²) + g h_α ∂_y h
        = −g h_α ∂_y z_b + v_{α+1/2} G_{α+1/2} − v_{α-1/2} G_{α-1/2}
    ∂_t(h_α T_α) + ∂_x(h_α u_α T_α) + ∂_y(h_α v_α T_α)
        =  T_{α+1/2} G_{α+1/2} − T_{α-1/2} G_{α-1/2}

Same structural derivation as the 1D version: per-layer integration
over [z_{α-1/2}, z_{α+1/2}], Leibniz on ∂_x and ∂_y, FT on ∂_z, KBC
at every interface (now with v · ∂_y z + u · ∂_x z − w form), level-0
moment closure 1/h_α ∫r·s dz ≈ r_α s_α.
"""
import sympy as sp
from zoomy_core.symbolic import (
    D, Int, canonicalise, distribute_derivative_over_add,
    fundamental_theorem, leibniz_general, polynomial_integrate, subst,
)


t, x, y, z = sp.symbols("t x y z", real=True)
g = sp.Symbol("g", positive=True)
rho0 = sp.Symbol(r"\rho_0", positive=True)


def build_3d_layer_state(N):
    z_iface = [None] * (N + 1)
    z_iface[0] = sp.Function("z_b", real=True)(t, x, y)
    for k in range(1, N):
        z_iface[k] = sp.Function(f"z_{2*k+1}_over_2", real=True)(t, x, y)
    z_iface[N] = sp.Function(r"\eta", real=True)(t, x, y)
    h = {a: z_iface[a] - z_iface[a - 1] for a in range(1, N + 1)}
    u = {a: sp.Function(f"u_{a}", real=True)(t, x, y) for a in range(1, N + 1)}
    v = {a: sp.Function(f"v_{a}", real=True)(t, x, y) for a in range(1, N + 1)}
    u_iface, v_iface, G = {}, {}, {}
    for two_a in range(1, 2 * N + 2, 2):
        u_iface[two_a] = sp.Function(f"u_{two_a}_over_2", real=True)(t, x, y)
        v_iface[two_a] = sp.Function(f"v_{two_a}_over_2", real=True)(t, x, y)
        G[two_a] = sp.Function(f"G_{two_a}_over_2", real=True)(t, x, y)
    return {"N": N, "z_iface": z_iface, "h": h, "u": u, "v": v,
            "u_iface": u_iface, "v_iface": v_iface, "G": G}


def replace_int(expr, integrand_template, var_template, lo, hi, repl):
    if not isinstance(expr, sp.Basic):
        return expr
    def _walk(e):
        if isinstance(e, sp.Integral):
            limits = e.args[1]
            if (hasattr(limits, "__len__") and len(limits) == 3
                    and limits[1] == lo and limits[2] == hi):
                bvar = limits[0]
                if e.args[0] == integrand_template.xreplace({var_template: bvar}):
                    return repl
        if e.args:
            new_args = tuple(_walk(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    return _walk(expr)


def derive_layer_continuity_3d(state, alpha):
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    G_lo, G_hi = state["G"][2 * alpha - 1], state["G"][2 * alpha + 1]
    u_lo, u_hi = state["u_iface"][2 * alpha - 1], state["u_iface"][2 * alpha + 1]
    v_lo, v_hi = state["v_iface"][2 * alpha - 1], state["v_iface"][2 * alpha + 1]
    h_a = state["h"][alpha]
    u_a, v_a = state["u"][alpha], state["v"][alpha]
    u_field = sp.Function("u", real=True)(t, x, y, z)
    v_field = sp.Function("v", real=True)(t, x, y, z)
    w_field = sp.Function("w", real=True)(t, x, y, z)

    full = (leibniz_general(D(u_field, x), z, z_lo, z_hi)
            + leibniz_general(D(v_field, y), z, z_lo, z_hi)
            + fundamental_theorem(D(w_field, z), z, z_lo, z_hi))

    # 3D KBC: w = ∂_t z + u·∂_x z + v·∂_y z − G
    kbc = {
        w_field.subs(z, z_hi): D(z_hi, t) + u_hi * D(z_hi, x)
                               + v_hi * D(z_hi, y) - G_hi,
        w_field.subs(z, z_lo): D(z_lo, t) + u_lo * D(z_lo, x)
                               + v_lo * D(z_lo, y) - G_lo,
    }
    full = subst(full, kbc)
    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo,
                        v_field.subs(z, z_hi): v_hi,
                        v_field.subs(z, z_lo): v_lo})
    full = replace_int(full, u_field, z, z_lo, z_hi, h_a * u_a)
    full = replace_int(full, v_field, z, z_lo, z_hi, h_a * v_a)
    return canonicalise(full)


def derive_layer_x_momentum_3d(state, alpha):
    z_lo = state["z_iface"][alpha - 1]
    z_hi = state["z_iface"][alpha]
    eta = state["z_iface"][state["N"]]
    G_lo, G_hi = state["G"][2 * alpha - 1], state["G"][2 * alpha + 1]
    u_lo, u_hi = state["u_iface"][2 * alpha - 1], state["u_iface"][2 * alpha + 1]
    v_lo, v_hi = state["v_iface"][2 * alpha - 1], state["v_iface"][2 * alpha + 1]
    h_a, u_a, v_a = state["h"][alpha], state["u"][alpha], state["v"][alpha]
    u_field = sp.Function("u", real=True)(t, x, y, z)
    v_field = sp.Function("v", real=True)(t, x, y, z)
    w_field = sp.Function("w", real=True)(t, x, y, z)

    full = (
        leibniz_general(D(u_field, t), z, z_lo, z_hi)
        + leibniz_general(D(u_field**2, x), z, z_lo, z_hi)
        + leibniz_general(D(u_field * v_field, y), z, z_lo, z_hi)
        + fundamental_theorem(D(u_field * w_field, z), z, z_lo, z_hi)
        + polynomial_integrate(g * D(eta, x), z, z_lo, z_hi)
    )
    kbc = {
        w_field.subs(z, z_hi): D(z_hi, t) + u_hi * D(z_hi, x)
                               + v_hi * D(z_hi, y) - G_hi,
        w_field.subs(z, z_lo): D(z_lo, t) + u_lo * D(z_lo, x)
                               + v_lo * D(z_lo, y) - G_lo,
    }
    full = subst(full, kbc)
    full = subst(full, {u_field.subs(z, z_hi): u_hi,
                        u_field.subs(z, z_lo): u_lo,
                        v_field.subs(z, z_hi): v_hi,
                        v_field.subs(z, z_lo): v_lo})
    full = replace_int(full, u_field, z, z_lo, z_hi, h_a * u_a)
    full = replace_int(full, u_field**2, z, z_lo, z_hi, h_a * u_a**2)
    full = replace_int(full, u_field * v_field, z, z_lo, z_hi, h_a * u_a * v_a)
    return canonicalise(full)


def reference_x_momentum_3d(state, alpha):
    h_a, u_a, v_a = state["h"][alpha], state["u"][alpha], state["v"][alpha]
    z_b = state["z_iface"][0]
    h_total = sum(state["h"][a] for a in range(1, state["N"] + 1))
    G_lo, G_hi = state["G"][2 * alpha - 1], state["G"][2 * alpha + 1]
    u_lo, u_hi = state["u_iface"][2 * alpha - 1], state["u_iface"][2 * alpha + 1]
    return (D(h_a * u_a, t) + D(h_a * u_a**2, x) + D(h_a * u_a * v_a, y)
            + g * h_a * D(h_total, x) + g * h_a * D(z_b, x)
            - (u_hi * G_hi - u_lo * G_lo))


def reference_continuity_3d(state, alpha):
    h_a, u_a, v_a = state["h"][alpha], state["u"][alpha], state["v"][alpha]
    G_lo, G_hi = state["G"][2 * alpha - 1], state["G"][2 * alpha + 1]
    return D(h_a, t) + D(h_a * u_a, x) + D(h_a * v_a, y) - (G_hi - G_lo)


def expand_derivatives(expr):
    def step(e):
        if isinstance(e, sp.Add):
            return sp.Add(*[step(a) for a in e.args])
        if isinstance(e, sp.Mul):
            return sp.Mul(*[step(a) for a in e.args])
        if isinstance(e, sp.Derivative):
            inner = step(e.expr)
            wrt_pairs = e.variable_count
            v_, n = wrt_pairs[0]
            rest = wrt_pairs[1:]
            if n > 1:
                rest = ((v_, n - 1),) + tuple(rest)
            if isinstance(inner, sp.Add):
                return step(sp.Add(*[sp.Derivative(a, v_, *rest) for a in inner.args]))
            if isinstance(inner, sp.Mul):
                factors = inner.args
                out = sp.Add(*[
                    sp.Mul(*(factors[:i] + (sp.Derivative(factors[i], v_),) + factors[i+1:]))
                    for i in range(len(factors))
                ])
                if rest:
                    out = sp.Derivative(out, *rest)
                return step(out)
            if isinstance(inner, sp.Pow):
                base, exponent = inner.args
                if isinstance(exponent, sp.Integer) and int(exponent) >= 2:
                    n_pow = int(exponent)
                    return step(n_pow * base**(n_pow - 1) * sp.Derivative(base, v_, *rest))
            return e
        return e
    prev = None; cur = sp.expand(expr)
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
        wrt = [v_[0] if isinstance(v_, (tuple, sp.Tuple)) else v_
               for v_ in d.args[1:]]
        if inner.is_number or not any(inner.has(v_) for v_ in wrt):
            mapping[d] = sp.S.Zero
    return expr.xreplace(mapping) if mapping else expr


def normal_form_3d(expr, state):
    e = expand_derivatives(expr)
    N = state["N"]
    flux_zero = {state["G"][1]: 0, state["G"][2 * N + 1]: 0}
    fixed_b = {D(state["z_iface"][0], t): 0}
    e = subst(subst(e, flux_zero), fixed_b)
    e = kill_zero_derivatives(e)
    return sp.expand(e)


def main(N):
    print(f"=== 3D MLSWE — N = {N} ===\n")
    state = build_3d_layer_state(N)
    n_mismatch = 0
    for alpha in range(1, N + 1):
        print(f"--- Layer α = {alpha} ---")
        for label, pipe_fn, ref_fn in [
            ("continuity", derive_layer_continuity_3d, reference_continuity_3d),
            ("x-momentum", derive_layer_x_momentum_3d, reference_x_momentum_3d),
        ]:
            pipe = pipe_fn(state, alpha)
            ref = ref_fn(state, alpha)
            diff = sp.expand(normal_form_3d(pipe, state)
                             - normal_form_3d(ref, state))
            ok = (diff == 0)
            print(f"  {label:<12} {'✓ MATCH' if ok else '✗ MISMATCH'}")
            if not ok:
                n_mismatch += 1
                print(f"    pipe = {normal_form_3d(pipe, state)}")
                print(f"    ref  = {normal_form_3d(ref, state)}")
                print(f"    diff = {diff}")
    print(f"\n=== Summary: {n_mismatch} mismatch(es) at N={N} ===")
    return n_mismatch


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(N)
