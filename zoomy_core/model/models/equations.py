"""Balance-equation blueprints (`Mass`, `Momentum`, `MomentumNonHydrostatic`,
`Transport`) that declare their state/closure variables and add their residual to a derivation
`Model`, plus the tracked scaling steps applied to them. They feed the SME/VAM derivations,
before the Model is lifted to a SystemModel.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d

# horizontal coord symbol → its directional derivative operator, velocity-field
# name, and coordinate name string
_DERIV = {C.x: d.x, C.y: d.y}
_HNAME = {C.x: "u", C.y: "v"}
_CNAME = {C.x: "x", C.y: "y"}


def evaluate_time_derivatives(expr, t):
    """`.doit()` only the `Subs` atoms and the derivatives carrying `t` of a residual,
    keeping every purely spatial flux divergence `d_x(F)` folded so it stays in the flux slot.
    """
    expr = sp.sympify(expr)
    # resolve the modal-closure Subs first (``.doit()`` substitutes ζ=0/1 etc.);
    # a Subs never carries a bare state ∂_x flux, so this cannot distribute one.
    expr = expr.xreplace({s: s.doit() for s in expr.atoms(sp.Subs)})
    # then evaluate ONLY the time derivatives (surface ∂_t h_ℓ); spatial ∂_x(F)
    # atoms are left untouched → still folded.
    return expr.xreplace({dv: dv.doit()
                          for dv in expr.atoms(sp.Derivative)
                          if t in dv.variables})


class Equation:
    """Base balance blueprint: `add_to(model)` declares variables and adds the residual, and
    `model.add_equation(blueprint)` dispatches to it.
    """

    _is_blueprint = True

    def __init__(self, model=None):
        """Store the (optional) model the blueprint is attached to."""
        self.model = model

    def add_to(self, model):
        """Declare variables and add the residual to `model`; implemented by subclasses."""
        raise NotImplementedError


class Mass(Equation):
    """Incompressible mass balance `sum_d d_d u_d + d_z w = 0`; registers the horizontal
    velocities and `w` as state.

    Args:
        suffix: scopes the field names for a multilayer column (`"_1"` gives `u_1, w_1`).
    """

    def __init__(self, model=None, suffix=""):
        """Store the model and the field-name `suffix`."""
        super().__init__(model)
        self.suffix = suffix

    def add_to(self, model):
        """Declare `u` (`v`) and `w` as state and add the `mass` equation; returns it."""
        coords = model.coords
        horiz = coords[1:-1]                       # (x,) or (x, y)
        s = self.suffix
        uvel = [sp.Function(_HNAME[xd] + s, real=True)(*coords) for xd in horiz]
        w = sp.Function("w" + s, real=True)(*coords)
        model.declare_state(*uvel, w)
        expr = sum(_DERIV[xd](uvel[i]) for i, xd in enumerate(horiz)) + d.z(w)
        model.add_equation("mass", expr)
        self.uvel, self.w = uvel, w
        return model._equations["mass"]


def gravity_components(model, coords):
    """Gravity body-force vector per spatial direction, the single gravity path of every
    momentum blueprint (flat plane is `e_d = 0`).

    Returns:
        `(g_h, g_z)`: in-plane component per horizontal coordinate (downslope `g e_x` along
            x, else 0) and the normal component `-g`.
    """
    horiz = coords[1:-1]
    g = model.parameters.g
    incline = {C.x: g * model.parameter("e_x", 0.0)}    # downslope body force
    g_h = {xd: incline.get(xd, sp.S.Zero) for xd in horiz}
    g_z = -g                                             # normal component
    return g_h, g_z


class Momentum(Equation):
    """General 3-D momentum balance with the full symmetric stress tensor `tau_ij` declared as
    closure variables and the pressure `p` as state.
    The shallow simplification is not built in; it is applied explicitly by `moment_scaling`.
    """

    def add_to(self, model):
        """Declare `p` and the stress components, and add the `momentum` equation
        (horizontal rows plus the vertical row); returns it.
        """
        coords = model.coords
        t, zc = coords[0], coords[-1]
        horiz = coords[1:-1]
        rho = model.parameters.rho
        g_h, g_z = gravity_components(model, coords)     # shared gravity vector
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
        p = sp.Function("p", real=True)(*coords)
        dname = lambda cc: "z" if cc == zc else _CNAME[cc]
        dop = lambda cc: d.z if cc == zc else _DERIV[cc]
        tau = {}                               # symmetric stress tensor, by name

        def stress(a, b):
            """Return the (symmetric) stress function `tau_ab`, creating it on first use."""
            key = "".join(sorted([dname(a), dname(b)]))   # τ_ij = τ_ji
            if key not in tau:
                tau[key] = sp.Function(f"tau_{key}", real=True)(*coords)
            return tau[key]

        comps = []
        for i, xi in enumerate(horiz):
            adv = (sum(dop(xj)(uvel[i] * uvel[j]) for j, xj in enumerate(horiz))
                   + d.z(uvel[i] * w))
            # FULL stress divergence Σ_j ∂_j τ_ij, j over horizontals + z
            sdiv = (sum(dop(xj)(stress(xi, xj)) for xj in horiz)
                    + d.z(stress(xi, zc)))
            comps.append(d.t(uvel[i]) + adv + dop(xi)(p) / rho - sdiv / rho
                         - g_h[xi])
        # vertical momentum (eliminated hydrostatically); − g_z = +g (flat)
        comps.append(d.t(w) + sum(dop(xj)(uvel[j] * w) for j, xj in enumerate(horiz))
                     + d.z(w * w) + d.z(p) / rho - g_z)
        model.declare_state(p)
        model.declare_closure(*tau.values())
        model.add_equation("momentum", (len(comps),), comps)
        self.uvel, self.w, self.p, self.tau = uvel, w, p, tau
        return model.momentum


def moment_scaling(model, momentum):
    """Shallow thin-layer scaling step: drop the in-plane stress components (both indices
    horizontal) from the full-tensor `Momentum` and record it in the model history.

    The vertical row of `Momentum` carries no stress, so the viscous normal stress
    `tau_zz` of the same order never enters; with an in-plane closure the SME adds it to the
    vertical row (`inplane_normal_stress`).
    """
    drop = {s: sp.S.Zero for k, s in momentum.tau.items() if "z" not in k}
    if drop:
        model.apply(drop)
        model._history("moment_scaling", "momentum",
                       description=f"shallow scaling: drop in-plane stresses "
                                   f"{sorted(k for k in momentum.tau if 'z' not in k)}")
    return model


def inplane_normal_stress(model, uvel, horiz, nu):
    """Viscous normal stress `tau_zz = 2 rho nu d_z w = -2 rho nu div_h u` of the
    incompressible Newtonian fluid (bulk, before the sigma-map).

    Same order as the in-plane stress `add_inplane_viscous` keeps; the hydrostatic SME takes
    it through the vertical momentum into the pressure, `p = rho g (eta - z) + tau_zz`.
    """
    rho = model.parameters.rho
    return -2 * rho * nu * sum(_DERIV[xd](uvel[i]) for i, xd in enumerate(horiz))


def add_inplane_viscous(model, momentum_eqs, uvel, horiz, nu, normal_stress=False):
    """Keep-all complement of `moment_scaling`: add the incompressible Newtonian in-plane
    deviatoric stress divergence `-(1/rho) sum_e d_e[rho nu (d_d u_e + d_e u_d)]` to each
    horizontal momentum residual, before the sigma-map.

    Args:
        normal_stress: also add `(1/rho) d_d tau_zz` (`inplane_normal_stress`): the model's
            pressure unknown is then `p - tau_zz = -sigma_zz`, zero at the free surface (VAM,
            ML-SME, ML-VAM), and the in-plane stress `tau_de - delta_de tau_zz`.
    """
    rho = model.parameters.rho
    dop = {C.x: d.x, C.y: d.y}
    tzz = inplane_normal_stress(model, uvel, horiz, nu) if normal_stress else None
    for i, xd in enumerate(horiz):
        # τ_de = ρ ν (∂_d u_e + ∂_e u_d);  sdiv_d = Σ_e ∂_e τ_de
        sdiv = sum(dop[xe](rho * nu * (dop[xd](uvel[horiz.index(xe)]) + dop[xe](uvel[i])))
                   for xe in horiz)
        if tzz is not None:
            sdiv = sdiv - dop[xd](tzz)
        eq = momentum_eqs[i]
        eq.expr = eq.expr - sdiv / rho
    model._history("add_inplane_viscous", "momentum",
                   description="keep in-plane deviatoric stress (incompressible "
                               "Newtonian τ_de = ρν(∂_d u_e + ∂_e u_d))"
                               + (" minus δ_de τ_zz" if normal_stress else ""))
    return model


def package_viscous(expr, nu, states, space):
    """Route a retained viscous residual into conservative diffusion by rewriting
    `C d_d d_e Q` as `d_d(C d_e Q) - (d_d C) d_e Q` with the diffusive flux wrapped in a
    `ViscousDiffusion` marker; non-viscous terms pass through.

    Args:
        expr: projected residual.
        nu: viscosity symbol identifying the viscous terms.
        states: state functions `Q` that may carry second derivatives.
        space: spatial coordinates `d, e`.
    Returns:
        The residual with the viscous part in divergence form.
    """
    from zoomy_core.model.derivation.system_extract import ViscousDiffusion
    expr = sp.expand(expr)
    visc = sum((tt for tt in sp.Add.make_args(expr) if tt.has(nu)), sp.S.Zero)
    rest = expr - visc
    out = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(visc.doit())):
        d2 = [dd for dd in term.atoms(sp.Derivative)
              if len(dd.variables) == 2 and all(v in space for v in dd.variables)
              and dd.args[0] in states]
        if d2:
            D2 = d2[0]; Q = D2.args[0]; v = D2.variables
            Coef = sp.cancel(term / D2); de = sp.Derivative(Q, v[1])
            out += (sp.Derivative(ViscousDiffusion(Coef * de), v[0],
                                  evaluate=False)
                    - sp.diff(Coef, v[0]) * de)
        else:
            out += term
    return rest + out


# ── opaque boundary frame ────────────────────────────────────────────────────
# A boundary stress closure prescribes the traction in the local frame
# ``{n, t_α}`` (see closures.py).  The frame is built from OPAQUE slope symbols
# — one per (boundary interface, horizontal direction) — NOT from the physical
# ``∂_d(interface)``, so resolving the frame (``small_slope_scaling`` → 0, or the
# exact ``∂_d I``) never aliases the topographic source ``g h ∂_x b``.  This is
# the inject-opaque / resolve-late pattern: a closure references the frame
# abstractly and is therefore independent of WHEN the frame is resolved.

def frame_slope(tag, direction):
    """Opaque geometric-slope symbol of a boundary interface (`tag` is `"b"` or `"eta"`)
    in horizontal `direction` `"x"` or `"y"`, distinct from the physical
    `d_d(interface)`.
    """
    return sp.Symbol(f"frameslope_{tag}_{direction}", real=True)


def small_slope_scaling(model):
    """Tracked step resolving the opaque boundary frame to its small-slope limit (every
    `frame_slope` symbol to 0, `n -> z_hat`); skip it to keep the slope-aware tractions.
    Geometric sibling of `moment_scaling`; together they give the shallow-moment form.
    """
    slopes = set()
    for eq in model._equations.values():
        slopes |= {s for s in eq.expr.free_symbols
                   if str(s).startswith("frameslope_")}
    if slopes:
        model.apply({s: sp.S.Zero for s in slopes})
        model._history("small_slope_scaling", "momentum",
                       description="resolve boundary frame to small-slope limit "
                                   "n→ẑ (drop O(slope) traction corrections)")
    return model


def slope_aware_scaling(model, interfaces):
    """Complement of `small_slope_scaling`: resolve each `frame_slope` symbol to the
    physical interface slope `d_dir(interface)`, so the geometrically exact model has no
    opaque symbol left.

    Args:
        interfaces: map from frame tag to height expression, e.g. `{"b": b, "eta": b + h}`.
    """
    from zoomy_core import coords as _C
    dirmap = {"x": _C.x, "y": _C.y}
    subs = {}
    for eq in model._equations.values():
        for s in eq.expr.free_symbols:
            name = str(s)
            if not name.startswith("frameslope_") or s in subs:
                continue
            parts = name.split("_")            # frameslope_{tag}_{dir}
            tag, dr = parts[1], parts[2]
            if tag not in interfaces or dr not in dirmap:
                raise ValueError(
                    f"slope_aware_scaling: cannot resolve frame symbol {name!r} "
                    f"(known tags {sorted(interfaces)}, dirs {sorted(dirmap)}). "
                    "Every frame_slope must map to a physical interface slope.")
            subs[s] = sp.Derivative(interfaces[tag], dirmap[dr])
    if subs:
        model.apply(subs)
        model._history("slope_aware_scaling", "momentum",
                       description="resolve boundary frame to the physical "
                                   "interface slope ∂_d(interface) (slope-aware)")
    return model


class MomentumNonHydrostatic(Equation):
    """Non-hydrostatic momentum with the hydrostatic pressure pre-absorbed: horizontal rows
    `momentum_<d>` carry `g d_d(eta)` and `p` is only the non-hydrostatic part, and the
    vertical row `momentum_z` keeps `d_z p`.

    Args:
        suffix: scopes the field names (multilayer column).
        tau_name: stress field name for one horizontal direction (default `tau_xz`).
        free_surface: overrides `eta` (default `b + h`).
    References:
        Escalante et al. 2024, https://doi.org/10.1016/j.jcp.2024.112882
    """

    def __init__(self, model=None, suffix="", tau_name="tau_xz", free_surface=None):
        """Store the field-name `suffix`, the stress name and the optional free-surface
        override.
        """
        super().__init__(model)
        self.suffix = suffix
        self.tau_name = tau_name
        self.free_surface = free_surface

    def add_to(self, model):
        """Declare `p` and one shear stress per horizontal direction and add the
        `momentum_x` (`momentum_y`) and `momentum_z` equations.
        """
        coords = model.coords
        t = coords[0]
        horiz = coords[1:-1]
        s = self.suffix
        rho = model.parameters.rho
        g_h, g_z = gravity_components(model, coords)     # shared gravity vector
        uvel = [sp.Function(_HNAME[xd] + s, real=True)(*coords) for xd in horiz]
        w = sp.Function("w" + s, real=True)(*coords)
        p = sp.Function("p" + s, real=True)(*coords)
        # one shear stress PER horizontal direction.  1 horizontal → keep the
        # ``tau_name`` (default ``tau_xz``; multilayer overrides to ``tau_<ell>``)
        # so 1-D / per-layer is byte-identical; 2 horizontals → ``tau_xz, tau_yz``
        # (+ suffix) so momentum_y closes its OWN shear, not momentum_x's.
        def _sname(xd):
            """Name of the shear-stress field for horizontal coordinate `xd`."""
            return self.tau_name if len(horiz) == 1 else f"tau_{_CNAME[xd]}z" + s
        tau = {xd: sp.Function(_sname(xd), real=True)(*coords) for xd in horiz}
        if self.free_surface is not None:
            eta = self.free_surface
        else:
            h = sp.Function("h", positive=True)(t, *horiz)
            b = sp.Function("b", real=True)(t, *horiz)
            eta = b + h
        model.declare_state(p)
        model.declare_closure(*tau.values())
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            # −g_z·∂_d η is the pre-absorbed hydrostatic gradient (flat: +g·∂_d η);
            # −g_h[xd] is the in-plane (downslope) body force — the SAME vector
            # component Momentum carries, so an inclined-plane VAM falls out with
            # no hand-built model (flat e_x=0 → no change).
            model.add_equation(
                f"momentum_{_CNAME[xd]}",
                d.t(uvel[i]) + adv + d.z(uvel[i] * w) - g_z * _DERIV[xd](eta)
                + _DERIV[xd](p) / rho - d.z(tau[xd]) / rho - g_h[xd])
        model.add_equation(
            "momentum_z",
            d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
            + d.z(w * w) + d.z(p) / rho)
        self.uvel, self.w, self.p, self.tau = uvel, w, p, tau
        self.txz = tau[horiz[0]]
        return model


class Transport(Equation):
    """Conservative transport of a scalar field `name`,
    `d_t c + sum_d d_d(u_d c) + d_z(w c) = source`, registered as a state field (TKE,
    dissipation, tracer, ...).

    Args:
        source: optional source field or expression; `None` for a passive scalar.
    """

    def __init__(self, model=None, name="c", source=None):
        """Store the field `name` and the optional `source`."""
        super().__init__(model)
        self.name = name
        self.source = source

    def add_to(self, model):
        """Declare the scalar as state and add its transport equation; returns it."""
        coords = model.coords
        horiz = coords[1:-1]
        c = sp.Function(self.name, real=True)(*coords)
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
        model.declare_state(c)
        expr = (d.t(c) + sum(_DERIV[xd](uvel[i] * c) for i, xd in enumerate(horiz))
                + d.z(w * c))
        if self.source is not None:
            expr = expr - self.source
        model.add_equation(self.name, expr)
        self.c = c
        return model._equations[self.name]


__all__ = ["Equation", "Mass", "Momentum", "MomentumNonHydrostatic", "Transport",
           "moment_scaling", "small_slope_scaling", "frame_slope",
           "add_inplane_viscous", "inplane_normal_stress", "package_viscous",
           "gravity_components"]
