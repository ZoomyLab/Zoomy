"""Toy pointwise kernels (1-D SWE) as sympy — stand-ins for the NSM kernels.

Printed with sympy's OWN printers (lambdify / cxxcode), exactly as the real
Model/Numerics kernels already are.
"""
import sympy as sp

hL, uL_, hR, uR_, h, hu, g, hb = sp.symbols("hL huL hR huR h hu g hb")


def _f(h_, q_):
    return [q_, q_**2 / h_ + g * h_**2 / 2]


s = sp.Max(sp.Abs(uL_ / hL) + sp.sqrt(g * hL), sp.Abs(uR_ / hR) + sp.sqrt(g * hR))
RUSANOV = [(a + b) / 2 - s * (qr - ql) / 2
           for a, b, ql, qr in zip(_f(hL, uL_), _f(hR, uR_), (hL, uL_), (hR, uR_))]
WAVESPEED = sp.Abs(hu / h) + sp.sqrt(g * h)
TROUBLED = sp.Piecewise((1.0, h < hb), (0.0, True))

KERNELS = {  # name: (face|cell, args, exprs)
    "rusanov": ("face", [hL, uL_, hR, uR_, g], RUSANOV),
    "wavespeed": ("cell", [h, hu, g], [WAVESPEED]),
    "troubled": ("cell", [h, hu, hb], [TROUBLED]),
}


def jax_kernels():
    import jax.numpy as jnp
    out = {}
    for name, (_, args, ex) in KERNELS.items():
        f = sp.lambdify(args, ex, modules="jax")
        out[name] = (lambda f: lambda *a: jnp.stack(jnp.broadcast_arrays(*f(*a))))(f)
    return out


def cxx_kernels():
    lines = []
    for name, (kind, args, ex) in KERNELS.items():
        sub, red = sp.cse(ex)
        nq = 2 if kind == "face" else 1
        # q-args read from arrays: face (qL[0..1], qR[0..1], par); cell (q[0..1], par)
        if kind == "face":
            head = (f"inline std::array<double,{len(ex)}> {name}("
                    "const double* qL, const double* qR, double par) {")
            binds = [f"double {args[0]} = qL[0], {args[1]} = qL[1], "
                     f"{args[2]} = qR[0], {args[3]} = qR[1], {args[4]} = par;"]
        else:
            head = (f"inline std::array<double,{len(ex)}> {name}("
                    "const double* q, double par) {")
            binds = [f"double {args[0]} = q[0], {args[1]} = q[1], {args[2]} = par;"]
        body = [f"  double {sp.cxxcode(v)} = {sp.cxxcode(e)};" for v, e in sub]
        ret = ", ".join(sp.cxxcode(e) for e in red)
        lines += [head, *("  " + b for b in binds), *body,
                  f"  return {{{ret}}};", "}"]
    return "\n".join(lines)
