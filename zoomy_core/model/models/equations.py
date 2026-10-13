"""Balance-equation blueprints (`Mass`, `Momentum`, `MomentumNonHydrostatic`,
`Transport`) that declare their state/closure variables and add their residual to a derivation
`Model`, plus the tracked scaling steps applied to them. They feed the SME/VAM derivations,
before the Model is lifted to a SystemModel.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.misc.options import Option

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
    closure variables (in every row, the vertical one included) and the pressure `p` as state.
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
        # vertical momentum (eliminated hydrostatically); − g_z = +g (flat); its stress
        # Σ_e ∂_e τ_ez + ∂_z τ_zz is one separate term (dropped by the shallow scaling)
        vterm = -(sum(dop(xj)(stress(xj, zc)) for xj in horiz) + d.z(stress(zc, zc))) / rho
        comps.append(d.t(w) + sum(dop(xj)(uvel[j] * w) for j, xj in enumerate(horiz))
                     + d.z(w * w) + d.z(p) / rho - g_z + vterm)
        model.declare_state(p)
        model.declare_closure(*tau.values())
        model.add_equation("momentum", (len(comps),), comps)
        self.uvel, self.w, self.p, self.tau = uvel, w, p, tau
        self.rows = list(model.momentum)[:-1]
        self.vertical, self.vertical_term = model.momentum.z, vterm
        self.shallow_terms = [(self.vertical, vterm)]
        return model.momentum


class Stress(Option):
    """How much of the bulk closure's deviatoric stress a moment model keeps; the rule is
    applied by `moment_scaling`, per component, in physical coordinates before the sigma map.
    """
    #: horizontal momentum: only the vertical shear `rho nu d_z u_e` of `tau_ez` (its `d_e w`
    #: part, `tau_de` and `tau_zz` dropped); vertical balance hydrostatic (no stress)
    shallow = "shallow"
    #: the whole tensor `rho nu (grad u + grad u^T)`; in the vertical row without its eps^2
    #: term `rho nu d_e d_e w` (Marche 2007 eqs. (3.2) -> (3.5),
    #: https://doi.org/10.1016/j.euromechflu.2006.04.007)
    full = "full"


def moment_scaling(model, momentum, stress=Stress.shallow):
    """Stress scaling step of the moment models: `Stress.shallow` drops, from the full-tensor
    blueprint `momentum`, the in-plane stresses `tau_de`, `tau_zz` and the stress of the
    vertical row; `Stress.full` keeps everything.  The choice is recorded as `model.stress`,
    which the bulk closure reads for the `d_e w` part of `tau_ez` (`apply_stress_closures`).
    """
    stress = Stress(stress)
    model.stress = stress
    if stress == Stress.full:
        model._history("moment_scaling", "momentum", description="full stress: nothing dropped")
        return model
    for eq, term in getattr(momentum, "shallow_terms", ()):
        eq.expr = eq.expr - term
    drop = {s: sp.S.Zero for k, s in momentum.tau.items() if "z" not in k}
    if drop and any(sp.sympify(eq.expr).has(*drop) for eq in model._equations.values()
                    if hasattr(eq, "expr")):
        model.apply(drop)
    model._history("moment_scaling", "momentum",
                   description=f"shallow scaling: drop in-plane stresses "
                               f"{sorted(k for k in momentum.tau if 'z' not in k)}, tau_zz "
                               f"and the stress of the vertical row")
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
    `momentum_<d>` carry `g d_d(eta)` and `p` is only the non-hydrostatic part of `-sigma_zz`,
    and the vertical row `momentum_z` keeps `d_z p`; the full stress tensor is declared, the
    shallow form is applied by `moment_scaling`.

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
        """Declare `p` and the stress tensor (one shear stress per horizontal direction, the
        in-plane `tau_de` and `tau_zz`) and add the `momentum_x` (`momentum_y`) and
        `momentum_z` equations.
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
        # the rest of the tensor: in-plane tau_de and tau_zz (+ suffix); p is the
        # non-hydrostatic part of -sigma_zz = p_total - tau_zz (zero at the surface), so tau_zz
        # enters the horizontal rows and not the vertical one
        full = dict(zip((_CNAME[xd] + "z" for xd in horiz), tau.values()))
        for i, xd in enumerate(horiz):
            for xe in horiz[i:]:
                key = _CNAME[xd] + _CNAME[xe]
                full[key] = sp.Function(f"tau_{key}" + s, real=True)(*coords)
        full["zz"] = sp.Function("tau_zz" + s, real=True)(*coords)
        comp = lambda a, b: full["".join(sorted(_CNAME[a] + _CNAME[b]))]
        model.declare_state(p)
        model.declare_closure(*tau.values())
        model.declare_closure(*[f for k, f in full.items() if "z" not in k or k == "zz"])
        self.rows, self.shallow_terms = [], []
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            # in-plane stress relative to tau_zz, one separate term (dropped by the shallow scaling)
            hterm = -(sum(_DERIV[xe](comp(xd, xe)) for xe in horiz) - _DERIV[xd](full["zz"])) / rho
            # −g_z·∂_d η is the pre-absorbed hydrostatic gradient (flat: +g·∂_d η);
            # −g_h[xd] is the in-plane (downslope) body force — the SAME vector
            # component Momentum carries, so an inclined-plane VAM falls out with
            # no hand-built model (flat e_x=0 → no change).
            model.add_equation(
                f"momentum_{_CNAME[xd]}",
                d.t(uvel[i]) + adv + d.z(uvel[i] * w) - g_z * _DERIV[xd](eta)
                + _DERIV[xd](p) / rho - d.z(tau[xd]) / rho - g_h[xd] + hterm)
            self.rows.append(getattr(model, f"momentum_{_CNAME[xd]}"))
            self.shallow_terms.append((self.rows[-1], hterm))
        vterm = -sum(_DERIV[xd](tau[xd]) for xd in horiz) / rho
        model.add_equation(
            "momentum_z",
            d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
            + d.z(w * w) + d.z(p) / rho + vterm)
        self.vertical, self.vertical_term = model.momentum_z, vterm
        self.shallow_terms.append((self.vertical, vterm))
        self.uvel, self.w, self.p, self.tau = uvel, w, p, full
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
           "Stress", "package_viscous",
           "gravity_components"]
