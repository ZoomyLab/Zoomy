"""Smoke test: run the K&T-style SME pipeline at StateSpace(dimension=3),
applying every step to BOTH momentum.x and momentum.y, and verify the
y-momentum result is the K&T eq (4.14) shape with x↔y swapped.
"""
import sympy as sp
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.basis_integral_cache import BasisIntegralCache
from zoomy_core.model.models.ins_generator import (
    FullINS, Integrate, IntegralTransform, IsolateBasisIntegrand,
    Multiply, ProductRule, ProjectBasisIntegrals, StateSpace, _FieldExpansion,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling
from zoomy_core.symbolic import (
    affine_change_of_variable, canonicalise, canonicalize_phi_derivative_subs,
    distribute_derivative_over_add, function_expand, product_rule_forward,
    project_basis_integrand, split_integral_over_add, subst,
)


LEVEL = 1
state = StateSpace(dimension=3)
t, x, y, z = state.t, state.x, state.y, state.z
H = state.H
print(f"state.has_y = {state.has_y}, dim = 3")
print(f"velocities: u={state.u}, v={state.v}, w={state.w}")

model = FullINS(state)
model.apply({state.tau[k]: 0 for k in state.tau._filter_dict()}).simplify()

print(f"\nStep 1 done.  Equations present:")
print(f"  continuity = {model.continuity._node.expr}")
print(f"  momentum.x = {model.momentum.x._node.expr}")
print(f"  momentum.y = {model.momentum.y._node.expr}")
print(f"  momentum.z = {model.momentum.z._node.expr}")

# Step 2: hydrostatic + p substitution + drop momentum.z
model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
p_relation = model.momentum.z.solve_for(state.p)
model.momentum.x.apply(p_relation).simplify()
model.momentum.y.apply(p_relation).simplify()
model.momentum.z.remove()
print(f"\nStep 2 done.  After p-subst:")
print(f"  momentum.x = {model.momentum.x._node.expr}")
print(f"  momentum.y = {model.momentum.y._node.expr}")

# Step 3: w-closure from continuity
pc = model.continuity.copy()
w_eq = pc.apply(Integrate(z, state.b, z, method="auto"))
w_closure = w_eq.solve_for(state.w)
print(f"\nStep 3 done.  w_closure = {w_closure._as_relation}")

# Step 4: depth-integrate continuity + KBC (3D KBC has u and v contributions)
ub = state.u.subs(z, state.b)
ue = state.u.subs(z, state.eta)
vb = state.v.subs(z, state.b)
ve = state.v.subs(z, state.eta)
kbc = {
    state.w.subs(z, state.b):
        sp.Derivative(state.b, t)
        + ub * sp.Derivative(state.b, x)
        + vb * sp.Derivative(state.b, y),
    state.w.subs(z, state.eta):
        sp.Derivative(state.eta, t)
        + ue * sp.Derivative(state.eta, x)
        + ve * sp.Derivative(state.eta, y),
}
model.continuity.apply(Integrate(z, state.b, state.eta, method="auto"))
model.continuity.apply(kbc).simplify()
print(f"\nStep 4 done.  depth-integrated continuity = {model.continuity._node.expr}")

# Step 5: plug w(z) into both x and y momentum
model.momentum.x.apply(w_closure).simplify()
model.momentum.y.apply(w_closure).simplify()

# Step 6: multiply by phi (outer=True), apply ProductRule
phi_fns = [sp.Function(f"phi_{k}") for k in range(LEVEL + 1)]
zoz = (z - state.b) / H
phi_of_z = Zstruct(**{f"phi_{k}": phi_fns[k](zoz) for k in range(LEVEL + 1)})
model.momentum.x.apply(Multiply(phi_of_z, outer=True))
model.momentum.y.apply(Multiply(phi_of_z, outer=True))
model.momentum.x.apply(ProductRule())
model.momentum.y.apply(ProductRule())
print(f"\nStep 6 done.  test_1 leaves on x and y:")
print(f"  momentum.x.test_1 has {len(sp.Add.make_args(model.momentum.x.test_1.expr))} terms")
print(f"  momentum.y.test_1 has {len(sp.Add.make_args(model.momentum.y.test_1.expr))} terms")

# Step 7-8: depth-integrate + KBC
model.momentum.x.apply(Integrate(z, state.b, state.eta, method="auto"))
model.momentum.y.apply(Integrate(z, state.b, state.eta, method="auto"))
model.momentum.x.apply(kbc).simplify()
model.momentum.y.apply(kbc).simplify()

# Step 9: snapshot dt_h_relation
dt_h_relation = model.continuity.solve_for(sp.Derivative(H, t))
dt_h_rule = dict(dt_h_relation._node._as_relation)
print(f"\nStep 9 done.  dt_h_relation = {dt_h_rule}")
model.momentum.x.apply(dt_h_relation).simplify()
model.momentum.y.apply(dt_h_relation).simplify()

# Step 10-13: ansatz + IT + IBI + PBI
basis_alpha_x = [sp.Function(f"alpha_x_{k}", real=True)(t, x, y) for k in range(LEVEL + 1)]
basis_alpha_y = [sp.Function(f"alpha_y_{k}", real=True)(t, x, y) for k in range(LEVEL + 1)]

def u_ansatz(*args):
    arg = args[-1]
    rhs = basis_alpha_x[0]
    for k in range(1, LEVEL + 1):
        rhs = rhs + basis_alpha_x[k] * phi_fns[k]((arg - state.b) / H)
    return rhs

def v_ansatz(*args):
    arg = args[-1]
    rhs = basis_alpha_y[0]
    for k in range(1, LEVEL + 1):
        rhs = rhs + basis_alpha_y[k] * phi_fns[k]((arg - state.b) / H)
    return rhs

model.apply(_FieldExpansion(state.u.func, u_ansatz)).simplify()
model.apply(_FieldExpansion(state.v.func, v_ansatz)).simplify()
model.apply(IntegralTransform()).simplify()
model.apply(IsolateBasisIntegrand()).simplify()
cache = BasisIntegralCache(Legendre_shifted(level=LEVEL))
model.apply(ProjectBasisIntegrals(cache)).simplify()
print(f"\nStep 13 done.  ProjectBasisIntegrals applied to both momentum.x.test_k and momentum.y.test_k.")

# Bug-3 closure on each test_k for both momentum.x and momentum.y
def has_free_dt_h(expr):
    for d in expr.atoms(sp.Derivative):
        if d.args[0] == H:
            wrt = [v[0] if isinstance(v, (tuple, sp.Tuple)) else v
                   for v in d.args[1:]]
            if t in wrt:
                return True
    return False


def close_bug3(expr, ansatz, max_iter=10):
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
    for _ in range(max_iter):
        prev = expr
        expr = expand_held_dt(expr)
        expr = subst(expr, dt_h_rule)
        expr = function_expand(expr, state.u.func, u_ansatz)
        expr = function_expand(expr, state.v.func, v_ansatz)
        expr = affine_change_of_variable(expr, z, state.b, state.b + H, zeta_hat)
        expr = canonicalize_phi_derivative_subs(expr)
        expr = split_integral_over_add(expr)
        expr = project_basis_integrand(expr, cache)
        expr = canonicalise(expr)
        if expr == prev or not has_free_dt_h(expr):
            break
    return expr


for axis, ansatz, alphas in [("x", u_ansatz, basis_alpha_x),
                              ("y", v_ansatz, basis_alpha_y)]:
    print(f"\n=== {axis}-momentum K&T comparison ===")
    leaf = getattr(model.momentum, axis)
    for k in range(LEVEL + 1):
        test = getattr(leaf, f"test_{k}").expr
        test = close_bug3(test, ansatz)

        # K&T-form: substitute Legendre boundary values + flat bottom.
        phi_legendre = {phi_fns[0](sp.S.Zero): 1, phi_fns[0](sp.S.One): 1,
                        phi_fns[1](sp.S.Zero): 1, phi_fns[1](sp.S.One): -1}
        flat = {state.b: 0, sp.Derivative(state.b, x): 0,
                sp.Derivative(state.b, y): 0, sp.Derivative(state.b, t): 0}
        test_kt = sp.expand(test.subs(phi_legendre).subs(flat))

        # Expand product-rule derivatives.
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
            prev = None; cur = sp.expand(expr)
            while prev != cur:
                prev = cur
                cur = sp.expand(step(cur))
            return cur

        test_exp = expand_derivatives(test_kt)

        # Pretty-print key K&T-comparable coefficients.
        a0x, a1x = basis_alpha_x[0], basis_alpha_x[1]
        a0y, a1y = basis_alpha_y[0], basis_alpha_y[1]
        dxH = sp.Derivative(H, x)
        dyH = sp.Derivative(H, y)
        g_const = state.g
        factor = 3 if k == 1 else 1
        probes = [
            ("∂_t α_x_0·h", sp.Derivative(a0x, t) * H),
            ("∂_t α_x_1·h", sp.Derivative(a1x, t) * H),
            ("∂_t α_y_0·h", sp.Derivative(a0y, t) * H),
            ("∂_t α_y_1·h", sp.Derivative(a1y, t) * H),
            # diagonal flux contributions (∂_x of x-flux, ∂_y of y-flux)
            ("α_x_0·∂_x α_x_0·h", a0x * sp.Derivative(a0x, x) * H),
            ("α_y_0·∂_y α_y_0·h", a0y * sp.Derivative(a0y, y) * H),
            ("α_x_1²·∂_x h", a1x**2 * dxH),
            ("α_y_1²·∂_y h", a1y**2 * dyH),
            # cross-axis flux contributions (∂_y of cross x-flux,
            # ∂_x of cross y-flux) — these EXIST ONLY IN 3D
            ("α_x_0·α_y_0·∂_y h (x-mom cross)", a0x * a0y * dyH),
            ("α_x_0·α_y_0·∂_x h (y-mom cross)", a0x * a0y * dxH),
            ("α_y_0·∂_y α_x_0·h", a0y * sp.Derivative(a0x, y) * H),
            ("α_x_0·∂_y α_y_0·h", a0x * sp.Derivative(a0y, y) * H),
            ("α_y_0·∂_x α_x_0·h", a0y * sp.Derivative(a0x, x) * H),
            ("α_x_0·∂_x α_y_0·h", a0x * sp.Derivative(a0y, x) * H),
            # gravity
            ("g·h·∂_x h", g_const * H * dxH),
            ("g·h·∂_y h", g_const * H * dyH),
        ]
        print(f"  test_{k} (×{factor}):")
        for label, key in probes:
            c = sp.simplify(factor * test_exp.coeff(key))
            if c != 0:
                print(f"    {label:<25} → {c}")
print("\nDone.")
