"""Composable stress closures: each `Closure` is one constitutive relation for a stress
component (bulk, bed trace, surface trace, horizontal) over a `ClosureState`.
Models list them in `closures=[...]` and perform the substitution when the Model is derived,
before the lift to a SystemModel.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.options import Option
from zoomy_core.model.operations import Operation
from zoomy_core.model.models.material import ClosureState   # the full-access state


def _solve_traction(P, s, nh, inplane=None):
    """Solve the boundary frame system `t_a . (tau n) = P_a` for the shear-stress traces
    `tau_iz` given the prescribed tangential tractions `P`; they reduce to `tau_iz = P_i`
    under small-slope scaling.

    Args:
        inplane: optional in-plane part `g_d = sum_e tau_de d_e I` per axis (`_inplane_traction`);
            then `(tau n)_d = (tau_dz - g_d) n_z` and `tau_iz -> P_i + g_i` under small-slope
            scaling.
    """
    nvec, tans = s.normal, s.tangents
    tau = [sp.Symbol(f"__tauz{i}") for i in range(nh)]
    nz = nvec[nh]
    if inplane is None:
        taudotn = sp.Matrix([tau[i] * nz for i in range(nh)]
                            + [sum(tau[i] * nvec[i] for i in range(nh))])
    else:
        # opaque g_d through the solve; the physical slope keeps it alive under small_slope
        g = [sp.Symbol(f"__tauin{i}") for i in range(nh)]
        taudotn = sp.Matrix([(tau[i] - g[i]) * nz for i in range(nh)]
                            + [sum(tau[i] * nvec[i] for i in range(nh))])
    eqs = [sp.Eq((tans[a].T * taudotn)[0], P[a]) for a in range(nh)]
    sol = sp.solve(eqs, tau, dict=True)
    if not sol:
        raise ValueError("boundary traction frame solve failed for "
                         f"P={P} (nh={nh})")
    out = [sp.simplify(sol[0][tau[i]]) for i in range(nh)]
    if inplane is not None:
        out = [o.xreplace(dict(zip(g, inplane))) for o in out]
    return out


def full_tensor_closures(closures, stress):
    """The bulk closures that supply the full stress tensor when the model keeps it (`stress`
    is `Stress.full`; a model's choice is `m.stress`, set by `moment_scaling`), else `[]`.

    Raises:
        ValueError: `Stress.full` with no bulk closure, or with one that closes only the
            vertical shear (only `Newtonian` provides the tensor).
    """
    from zoomy_core.model.models.equations import Stress
    if stress is None or Stress(stress) != Stress.full:
        return []
    bulk = [c for c in closures if c.closes == "bulk"]
    partial = [type(c).__name__ for c in bulk if not hasattr(c, "tensor")]
    if partial or not bulk:
        why = (f"{', '.join(partial)} closes only the vertical shear rho nu d_z u"
               if partial else "none is given")
        raise ValueError("stress=Stress.full needs a bulk closure that supplies the full stress "
                         f"tensor (Newtonian); {why}: use stress=Stress.shallow")
    return bulk


def inplane_stress(full, s, nh):
    """In-plane stress relative to the vertical normal stress, `tau_de - delta_de tau_zz`,
    summed over the full-tensor closures at the trace of `s` (`nh x nh` nested list).
    """
    T = [[sp.S.Zero] * nh for _ in range(nh)]
    for c in full:
        Tc, N = c.stress(s), c.normal_stress(s)
        for d_ in range(nh):
            for e in range(nh):
                T[d_][e] += Tc[d_][e] - (N if d_ == e else 0)
    return T


def _inplane_traction(full, s, nh):
    """In-plane part of the boundary traction, `g_d = sum_e (tau_de - delta_de tau_zz) d_e I`
    per axis, or `None` without the full tensor.

    `tau_de` is the in-plane stress at the trace of `s` and `d_e I` the PHYSICAL interface
    slope, the slope the sigma map gives the Leibniz terms of the depth-integrated
    `d_e tau_de` and of the pressure; with `tau_dz = P_d + g_d` those cancel and the depth
    integral is `d_e(h <tau_de - delta_de tau_zz>)` (Gerbeau, Perthame 2001,
    https://doi.org/10.3934/dcdsb.2001.1.89).  The `tau_zz` part is the surface pressure
    `p(eta) = tau_zz(eta)` on the slope, and at the bed the normal stress the tangential
    projection removes (Marche 2007, eq. (5.18),
    https://doi.org/10.1016/j.euromechflu.2006.04.007).
    """
    if not full:
        return None
    S = s.interface_slopes
    T = inplane_stress(full, s, nh)
    return [sum(T[d_][e] * S[e] for e in range(nh)) for d_ in range(nh)]


def apply_stress_closures(closures, m, axes, state, horiz):
    """Inject the bulk, bed and surface stress closures into the projected momentum equations of
    SME/VAM, frame-aware and dimension-agnostic.  With the full tensor (`Stress.full`) the bed
    and surface tractions carry the in-plane stress too (`_inplane_traction`) and the bulk
    shear its `rho nu d_d w` part.

    Args:
        closures: list of `Closure` objects.
        m: derivation model.
        axes: one `{"mx": eq, "tau": FieldHandle, "velname": "u"|"v"}` per horizontal momentum
            component.
        state: callable `state(at, *, alias=None, btag=None) -> ClosureState`.
        horiz: horizontal coordinates.
    Returns:
        True iff a bulk closure was applied.
    """
    nh = len(axes)
    for c in closures:
        c.register(m)
    for c in closures:
        c.check(m)
    surface = [c for c in closures if c.closes == "surface"]
    bottom = [c for c in closures if c.closes == "bottom"]
    bulk = [c for c in closures if c.closes == "bulk"]
    full = full_tensor_closures(closures, getattr(m, "stress", None))
    for at, btag, pieces in ((1, "eta", surface), (0, "b", bottom)):
        if not pieces:
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh,
                                 inplane=_inplane_traction(full, state(at, btag=btag), nh))
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    _apply_bulk(bulk, full, axes, state, nh)
    return bool(bulk)


def _full_shear(full, s, nh):
    """The `rho nu d_d w` part of the shear `tau_dz` of the full-tensor closures at the trace of
    `s`, per axis, or `None` without the full tensor.
    """
    if not full:
        return None
    parts = [c.shear(s) for c in full]
    return [sum(p[i] for p in parts) for i in range(nh)]


def _apply_bulk(bulk, full, axes, state, nh):
    """Substitute the bulk shear `tau_dz`: each bulk closure's vertical shear (diagonal
    constitutive) plus, with the full tensor, its `rho nu d_d w` part.
    """
    shear = _full_shear(full, state(None), nh)
    for j, c in enumerate(bulk):
        for i, ax in enumerate(axes):
            e = c.expression(state(None, alias={"u": ax["velname"]}))
            if shear is not None and j == 0:
                e = e + shear[i]
            ax["mx"].apply({ax["tau"].expr: e})


def apply_full_tensor(closures, m, momentum):
    """With `Stress.full`, substitute the full-tensor closure's stress (physical, fixed-z
    coordinates, before the sigma map) into the blueprint `momentum`: `tau_de` and `tau_zz` in
    the horizontal rows, `tau_ez` and `tau_zz` in the vertical row (its divergence evaluated).
    The horizontal shear `tau_dz` stays a closure variable, closed after the projection
    (`apply_stress_closures`: bed / surface traction, bulk).  In the vertical row the
    `rho nu d_e d_e w` term is dropped (Marche 2007, (3.2) -> (3.5)).

    Returns:
        The tensor `{key: tau}`, or `None` with `Stress.shallow` (no-op).
    """
    full = full_tensor_closures(closures, getattr(m, "stress", None))
    if not full:
        return None
    c = full[0]
    c.register(m)
    horiz = list(m.coords[1:-1])
    T = c.tensor(m, momentum.uvel, momentum.w, horiz)
    tau = momentum.tau
    hsub = {tau[k]: T[k] for k in tau if k in T and ("z" not in k or k == "zz")}
    vsub = {tau[k]: T[k] for k in tau if k in T and "z" in k}
    for eq in momentum.rows:
        eq.expr = eq.expr.xreplace(hsub)
    # vertical row: its divergence evaluated, without the eps^2 term nu d_e d_e w of Marche
    # (3.2), which (3.5) drops: it is eps^2 below every other term of the row (the
    # non-hydrostatic inertia included) and would put d^3 b (bed-kinematic w) into the w rows
    vt = momentum.vertical_term
    w2 = {sp.Derivative(momentum.w, (xe, 2)): 0 for xe in horiz}
    momentum.vertical.expr = (momentum.vertical.expr - vt
                              + sp.expand(vt.xreplace(vsub).doit()).subs(w2))
    m._history("apply_full_tensor", "momentum",
               description=f"{type(c).__name__} full stress tensor (physical coordinates)")
    return T


class Closure(Operation):
    """Base class of a one-component stress closure; subclasses set the immutable `closes` /
    `requires` and implement `expression`.
    """

    closes = None          # "bulk" | "bottom" | "surface"   (immutable)
    requires = ()          # field names the closure consumes (immutable tuple)

    def __init__(self, name=None):
        """Name the closure after its class and describe it by the component it closes."""
        super().__init__(name=name or type(self).__name__,
                         description=f"{type(self).__name__} ({self.closes})")

    def _key_content(self):
        """Result-affecting instance state of this closure, used as its derivation-cache key."""
        from zoomy_core.model.derivation.cache_keys import _state_repr
        return _state_repr(vars(self))

    def __repr__(self):
        """Class name plus `_key_content`, so the derivation-cache canonicaliser sees the
        configuration.
        """
        # The derivation-cache canonicaliser falls back to ``repr`` for callable
        # Operations, so the repr MUST carry the result-affecting config.
        return f"{type(self).__name__}({self._key_content()})"

    def register(self, model):
        """Register-or-query the model parameters this closure needs (override)."""

    def check(self, model):
        """Assert every field in `requires` is present in the model."""
        have = set(model.functions.keys())
        for f in self.requires:
            assert f in have, (
                f"{type(self).__name__} closure needs field {f!r}, which this "
                f"model does not define (has: {sorted(have)})")

    def expression(self, s):
        """Return the symbolic relation for this stress component over the `ClosureState`
        `s` (override).
        """
        raise NotImplementedError

    def friction_velocity(self, s):
        """Bed friction velocity `u_star` this closure defines, or `None`; bed drag laws
        override it so an eddy viscosity can consume it.
        """
        return None

    def traction(self, s):
        """Boundary traction in the local frame `{n, t_a}`; the default wraps the scalar
        `expression` as one tangential component.

        Returns:
            `{"normal": n.sigma.n or None, "tangent": [t_a.sigma.n, ...]}`, one tangent entry
                per horizontal direction.
        """
        return {"normal": None, "tangent": [self.expression(s)]}


# ── bulk closures ──────────────────────────────────────────────────────────


class Newtonian(Closure):
    """Incompressible Newtonian fluid, `tau = rho nu (grad u + grad u^T)`.  `expression` is
    the vertical shear `rho nu d_z u` (all `Stress.shallow` keeps); with `Stress.full` the
    model takes the whole tensor (`tensor` before the sigma map, `stress` / `normal_stress` /
    `shear` at the traces), so the depth-averaged in-plane term is `div(h (tau - tau_zz I))`,
    i.e. `div(nu h (grad u + grad u^T)) + 2 nu grad(h div u)` at level 0.

    References:
        Marche 2007, eqs. (3.2), (3.5), (5.19), https://doi.org/10.1016/j.euromechflu.2006.04.007
    """
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        """Register the kinematic viscosity `nu`."""
        m.parameter("nu", 0.0)

    def expression(self, s):
        """Return `rho nu d_z u`."""
        return s.par.rho * s.par.nu * s.dz(s.u)

    def tensor(self, m, uvel, w, horiz):
        """The tensor in physical (fixed-z) coordinates, before the sigma map, `{key: tau}` for
        the blueprint keys (`"xx"`, `"xy"`, `"yy"`, `"xz"`, `"yz"`, `"zz"`):
        `tau_de = rho nu (d_d u_e + d_e u_d)`, `tau_dz = rho nu (d_z u_d + d_d w)`,
        `tau_zz = 2 rho nu d_z w = -2 rho nu div_h u` (incompressible).
        """
        from zoomy_core import coords as _C
        import zoomy_core.derivatives as _d
        rho, nu = m.parameters.rho, m.parameters.nu
        D, N = {_C.x: _d.x, _C.y: _d.y}, {_C.x: "x", _C.y: "y"}
        T = {}
        for i, a in enumerate(horiz):
            for j in range(i, len(horiz)):
                T[N[a] + N[horiz[j]]] = rho * nu * (D[a](uvel[j]) + D[horiz[j]](uvel[i]))
            T[N[a] + "z"] = rho * nu * (_d.z(uvel[i]) + D[a](w))
        T["zz"] = -2 * rho * nu * sum(D[a](uvel[i]) for i, a in enumerate(horiz))
        return T

    def stress(self, s):
        """In-plane tensor `tau_de = rho nu (d_d u_e + d_e u_d)` at the trace of `s`, with the
        physical (fixed-z) derivatives of the sigma map; it enters the boundary traction
        (`apply_stress_closures`).
        """
        u, X = s.horizontal_velocity, s.horizontal
        return [[s.trace(s.par.rho * s.par.nu * (s.dx_phys(u[e], X[d]) + s.dx_phys(u[d], X[e])))
                 for e in range(len(X))] for d in range(len(X))]

    def normal_stress(self, s):
        """Viscous normal stress `tau_zz = 2 rho nu d_z w = -2 rho nu div_h u` at the trace of
        `s` (physical derivatives).
        """
        u, X = s.horizontal_velocity, s.horizontal
        div = sum(s.dx_phys(u[e], X[e]) for e in range(len(X)))
        return s.trace(-2 * s.par.rho * s.par.nu * div)

    def shear(self, s):
        """The `rho nu d_d w` part of the shear `tau_dz` at the trace of `s` (physical
        derivative), per axis; `expression` is the other part.
        """
        w, X = s.vertical_velocity, s.horizontal
        return [s.trace(s.par.rho * s.par.nu * s.dx_phys(w, X[d])) for d in range(len(X))]


class KEpsilonViscosity(Closure):
    """Bulk stress `tau = rho nu_t d_z u` with the k-epsilon eddy viscosity ``nu_t = C_mu
    k^2/epsilon``; needs `k` and `varepsilon` fields and `quadrature_order > 0`.

    Args:
        wall_floor: add the near-wall mixing-length viscosity `C_mu^(1/4) sqrt(k) kappa z_p`
            so the stress still damps the velocity moments at the wall.

    References:
        Launder, Spalding 1974, https://doi.org/10.1016/0045-7825(74)90029-2
    """
    closes = "bulk"; requires = ("u", "k", "varepsilon")

    def __init__(self, name=None, wall_floor=True):
        """Store the `wall_floor` switch."""
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        """Register `C_mu`, `nu` and, with `wall_floor`, `kappa` and `z_p`."""
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return `rho (nu + nu_t) d_z u` with `nu_t = C_mu k^2/epsilon` plus the optional
        wall floor.
        """
        nu_t = s.par.C_mu * s.k ** 2 / s.varepsilon
        if self.wall_floor:
            # offset log-layer mixing length ℓ=κ(z+z_p) (see QRViscosity)
            nu_t += (s.par.C_mu ** sp.Rational(1, 4) * sp.sqrt(s.k) * s.par.kappa
                     * (s.par.z_p + s.depth * s.zeta))
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


class QRViscosity(Closure):
    """Bulk stress for the positivity-preserving q-r k-epsilon model, with ``nu_t = C_mu
    sk^4/se^2`` for the transported `sk = sqrt(k)`, `se = sqrt(epsilon)`; needs
    `quadrature_order > 0`.

    Args:
        wall_floor: add the near-wall mixing-length viscosity, as in `KEpsilonViscosity`.

    References:
        Fe et al. 2009, https://doi.org/10.1002/fld.1880
    """
    closes = "bulk"; requires = ("u", "sk", "se")

    def __init__(self, name=None, wall_floor=True):
        """Store the `wall_floor` switch."""
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        """Register `C_mu`, `nu`, the floor `eps_min` and, with `wall_floor`, `kappa`
        and `z_p`.
        """
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        m.parameter("eps_min", 0.0)   # realizability floor: ε → ε + eps_min
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return `rho (nu + nu_t) d_z u` with `nu_t = C_mu sk^4/(se^2 + eps_min)` plus the
        optional wall floor.
        """
        # ν_t = C_μ k²/ε = C_μ sk⁴/(se²+ε_min)  (sk=√k, se=√ε); the ε_min
        # realizability floor keeps the denominator off zero (see QRKESME).
        nu_t = s.par.C_mu * s.sk ** 4 / (s.se ** 2 + s.par.eps_min)
        if self.wall_floor:
            # WALL-FUNCTION-consistent near-wall eddy viscosity — the
            # Prandtl–Kolmogorov form with the log-layer mixing length, OFFSET by
            # z_p so it stays finite at the wall:  ℓ = κ(z + z_p) = κ(h ζ + z_p).
            #   ν_t,wall = C_μ^{1/4}√k · κ(z+z_p)   (= κ u_⋆ (z+z_p), u_⋆=C_μ^{1/4}√k).
            # SME is valid only ABOVE the wall layer, so the wall distance is
            # offset by z_p (the reference height): ℓ→κz_p at the bed (the
            # wall-function value) and GROWS with height like the physical κz —
            # the opposite of the bed-heavy ν_t=C_μ sk⁴/se² (which is inverted
            # because ε is under-resolved near the wall), so it straightens the
            # velocity profile.  C_μ sk⁴/se² also VANISHES at the wall, so this
            # term additionally restores the stress-trace damping of the velocity
            # moments (q₁'s eddy damping ∝ the √k gradient, else too weak).
            nu_t += (s.par.C_mu ** sp.Rational(1, 4) * s.sk * s.par.kappa
                     * (s.par.z_p + s.depth * s.zeta))
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


class Bingham(Closure):
    """Regularized Bingham bulk stress `tau = (rho nu + tau_y/sqrt((d_z u)^2 + eps^2)) d_z u`,
    rigid below the yield stress `tau_y` as `eps -> 0`.
    Not analytically integrable, so build the model with `quadrature_order > 0`.
    """
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        """Register `nu`, the yield stress `tau_y` and the regularization scale `eps_reg`."""
        m.parameter("nu", 0.0); m.parameter("tau_y", 0.0); m.parameter("eps_reg", 1e-3)

    def expression(self, s):
        """Return the regularized Bingham stress."""
        return (s.par.rho * s.par.nu
                + s.par.tau_y / sp.sqrt(s.dz(s.u) ** 2 + s.par.eps_reg ** 2)) * s.dz(s.u)


class ElderViscosity(Closure):
    """Algebraic Elder eddy viscosity `nu_t = kappa u_star h zeta (1 - zeta)`, polynomial in
    zeta so the Galerkin projection closes analytically.

    Args:
        friction: optional bed closure whose live `friction_velocity` replaces the `u_star`
            parameter.

    References:
        Elder 1959, https://doi.org/10.1017/s0022112059000374
    """
    closes = "bulk"; requires = ("u",)

    def __init__(self, name=None, friction=None):
        """Store the optional bed closure supplying `u_star`."""
        super().__init__(name=name)
        self.friction = friction      # a bottom closure exposing friction_velocity

    def register(self, m):
        """Register `kappa`, `u_star`, `nu` and the parameters of the bed closure, if any."""
        m.parameter("kappa", 0.41)
        m.parameter("u_star", 0.0)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        if self.friction is not None:
            self.friction.register(m)  # ensure the bed law's params (k_s, z_p) exist

    def expression(self, s):
        """Return `rho (nu + nu_t) d_z u` with the Elder `nu_t`."""
        u_star = (self.friction.friction_velocity(s) if self.friction is not None
                  else s.par.u_star)
        nu_t = s.par.kappa * u_star * s.depth * s.zeta * (1 - s.zeta)
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


# ── bottom (bed) closures ──────────────────────────────────────────────────


class NavierSlip(Closure):
    """Linear Navier slip at the bed, `tau_b = lambda_s u_b`."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register the slip coefficient `lambda_s`."""
        m.parameter("lambda_s", 0.0)

    def expression(self, s):
        """Return `lambda_s u`."""
        return s.par.lambda_s * s.u

    def traction(self, s):
        """Tangential traction `lambda_s u_t` per axis (reduces to `lambda_s u_i` under
        small-slope scaling).
        """
        return {"normal": None,
                "tangent": [s.par.lambda_s * ut for ut in s.u_tangent]}


class RoughWall(Closure):
    """Turbulent rough-wall bed drag `tau_b = rho C_f u_b |u_b|` with ``C_f =
    (kappa/ln(z_p/z_0))^2`` and `z_0 = k_s/30`.
    Needs `k_s` (Nikuradse roughness), `z_p` and `kappa`.
    """
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register `kappa`, `k_s` and `z_p`."""
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return `rho C_f u |u|`."""
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        return s.par.rho * Cf * s.u * sp.Abs(s.u)

    def friction_velocity(self, s):
        """Recovered bed friction velocity `sqrt(C_f) |u_b|` of the horizontal bed trace,
        consumed by `ElderViscosity`.
        """
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        ub = s.velocity_at(0)                        # bed trace [u, (v,) w]
        u_b_mag = sp.sqrt(sum(ub[i] ** 2 for i in range(len(ub) - 1)))  # horizontal
        return sp.sqrt(Cf) * u_b_mag

    def traction(self, s):
        """Quadratic bed drag as a vector traction `rho C_f |U_t| u_t` opposing the slip
        velocity, coupling the x/y components through `|U_t|`.
        """
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class WallFunctionBed(Closure):
    """k-epsilon rough-wall-function bed stress `tau_b = rho C_f |U_p| U_p` with the log-law
    `C_f` of `RoughWall` but the velocity `U_p` read at the near-wall height ``zeta_p =
    z_p/h`` rather than at the bed trace.
    Needs `k_s`, `z_p` and `kappa`.
    """
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register `kappa`, `k_s` and `z_p`."""
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def _Cf(self, s):
        """Log-law drag coefficient `(kappa/ln(z_p/(k_s/30)))^2`."""
        return (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2

    def expression(self, s):
        """Scalar 1-D drag `rho C_f U_p |U_p|` from the near-wall velocity."""
        # scalar 1-D fallback: drag from the near-wall reference velocity U_p
        Up = s.velocity_at(s.par.z_p / s.depth)[0]
        return s.par.rho * self._Cf(s) * Up * sp.Abs(Up)

    def traction(self, s):
        """Vector bed traction `rho C_f |U_p| U_p` from the tangential velocity at ``zeta_p =
        z_p/h``.
        """
        Cf = self._Cf(s)
        ut = s.u_tangent_at(s.par.z_p / s.depth)
        speed = sp.sqrt(sum(u ** 2 for u in ut))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * u for u in ut]}


# ── depth-averaged SWE closures (bed friction + horizontal mixing) ──────────
# These close the DEPTH-AVERAGED shallow-water momentum, not the vertical
# moment hierarchy: a depth-averaged model has no resolved vertical profile,
# so its dissipation is (i) a bed-stress trace and (ii) a HORIZONTAL turbulent
# mixing τ_xx/τ_xy.  They are consumed by the SWE model's own closure hook
# (``apply_swe_closures``), which supplies the operator STRUCTURE (2-D vector
# source / rank-4 diffusion tensor) while the closure supplies the
# CONSTITUTIVE COEFFICIENT — same "compose the physics from named pieces"
# philosophy as the moment-hierarchy closures above.


class ManningFriction(Closure):
    """Manning bed friction as a bottom closure for depth-averaged SWE; `expression` is the
    rate `-g n^2 |u| / max(h, h_floor)^(1/3)` multiplying each velocity component.

    Args:
        h_floor: depth floor that keeps friction finite at the wet/dry interface.
    """
    closes = "bottom"; requires = ("u", "h")

    def __init__(self, name=None, h_floor=0.0):
        """Store the depth floor `h_floor`."""
        super().__init__(name=name)
        self.h_floor = float(h_floor)

    def register(self, m):
        """Register `g` and the Manning coefficient `n`."""
        m.parameter("g", 9.81); m.parameter("n", 0.0)

    def expression(self, s):
        """Return the friction rate (per unit velocity) for the depth-averaged SWE."""
        h_eff = sp.Max(s.h, sp.Float(self.h_floor)) if self.h_floor > 0 else s.h
        return -s.par.g * s.par.n ** 2 * s.speed / h_eff ** sp.Rational(1, 3)

    def traction(self, s):
        """Bed shear stress vector `rho g n^2 |U_t| u_t / h^(1/3)` for the moment hierarchy,
        the Manning analogue of `RoughWall.traction`.
        """
        h = s.depth
        h_eff = sp.Max(h, sp.Float(self.h_floor)) if self.h_floor > 0 else h
        Cf = s.par.g * s.par.n ** 2 / h_eff ** sp.Rational(1, 3)
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class EddyViscosity(Closure):
    """Horizontal eddy viscosity closure for the depth-averaged mixing `div(nu h grad u)`;
    `expression` returns the isotropic `nu` and the SWE model builds the diffusion tensor
    from it.
    """
    closes = "horizontal"; requires = ("u",)

    def register(self, m):
        """Register the kinematic viscosity `nu`."""
        m.parameter("nu", 0.0)

    def expression(self, s):
        """Return `nu`."""
        return s.par.nu


# ── surface closures ───────────────────────────────────────────────────────


class StressFree(Closure):
    """Stress-free free surface, `tau(1) = 0`."""
    closes = "surface"; requires = ()

    def expression(self, s):
        """Return zero."""
        return sp.S.Zero

    def traction(self, s):
        """Zero tangential traction on every axis; the free-surface pressure is left to the
        hydrostatic step.
        """
        return {"normal": None, "tangent": [sp.S.Zero for _ in s.tangents]}


# ── interface-transfer closures (multilayer) ───────────────────────────────
# The shared transfer velocity u* at an INTERNAL layer interface — the
# numerical scheme for the advection trace exchanged across the interface.
# A distinct closure family: ``expression(below, above, G)`` (the two one-sided
# modal interface traces + the interface mass flux G), not the stress ``(s)``.


class InterfaceQuantity(Option):
    """The interface value an `InterfaceFlux` scheme supplies."""
    #: the transfer velocity u*
    velocity = "velocity"
    #: the stress of the interface traction (interface_traction)
    stress = "stress"


class InterfaceFlux(Closure):
    """Base of the multilayer interface schemes; `expression(below, above, G)` takes the two
    one-sided interface traces and the interface mass flux `G`.

    Args:
        quantity: the `InterfaceQuantity` it supplies.
    """
    closes = "interface"; requires = ()

    def __init__(self, name=None, quantity=None):
        """Store the `quantity` this scheme supplies (default: the transfer velocity)."""
        super().__init__(name=name)
        self.quantity = InterfaceQuantity(quantity or InterfaceQuantity.velocity)

    def expression(self, below, above, G):    # noqa: D401 - distinct signature
        """Return the shared interface velocity `u*` (override)."""
        raise NotImplementedError


class MeanInterface(InterfaceFlux):
    """Central interface velocity `u* = (u_below + u_above)/2`."""

    def expression(self, below, above, G):
        """Return the arithmetic mean of the two traces."""
        return (below + above) / 2


class UpwindInterface(InterfaceFlux):
    """Donor interface velocity chosen by the sign of the interface flux, `u* = u_below` if
    `G >= 0` else `u_above`.

    References:
        Audusse et al. 2011, https://doi.org/10.1051/m2an/2010036
    """

    def expression(self, below, above, G):
        """Return the upwind trace selected by the sign of `G`."""
        return sp.Piecewise((below, G >= 0), (above, True))


def apply_layer_stress_closures(closures, m, axes, state, *, is_top, is_bottom):
    """Per-layer variant of `apply_stress_closures` for the multilayer models: the surface
    closure acts only on the top layer and the bed closure only on the bottom layer.

    With the full tensor (`Stress.full`) every trace carries the layer's own in-plane traction
    (`_inplane_traction`), so the layer's in-plane term is ``d_e(h_l <tau_de - delta_de
    tau_zz>)``, and the bulk shear its `rho nu d_d w` part; at an internal interface the trace
    is that own part only, and the model adds the shared, single-valued rest of the traction
    (`interface_traction`).

    Args:
        is_top: this layer carries the free surface.
        is_bottom: this layer sits on the bed.
    Returns:
        True iff a bulk closure was applied.
    """
    nh = len(axes)
    stress = [c for c in closures if c.closes in ("bulk", "bottom", "surface")]
    for c in stress:
        c.register(m)
    full = full_tensor_closures(closures, getattr(m, "stress", None))
    surface = [c for c in stress if c.closes == "surface"] if is_top else []
    bottom = [c for c in stress if c.closes == "bottom"] if is_bottom else []
    bulk = [c for c in stress if c.closes == "bulk"]
    for at, btag, pieces, outer in ((1, "eta", surface, is_top), (0, "b", bottom, is_bottom)):
        g = _inplane_traction(full, state(at, btag=btag), nh)
        if not pieces:
            if g is not None and not outer:          # internal interface: own in-plane part
                for i, ax in enumerate(axes):
                    ax["mx"].apply({ax["tau"].at(at): g[i]})
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh, inplane=g)
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    _apply_bulk(bulk, full, axes, state, nh)
    return bool(bulk)


def interface_closure(closures, quantity=InterfaceQuantity.velocity):
    """Return the `InterfaceFlux` closure for `quantity` in `closures`, or `None`."""
    return next((c for c in (closures or []) if c.closes == "interface"
                 and getattr(c, "quantity", InterfaceQuantity.velocity) == quantity), None)


def _shear_trace(full, s, nh):
    """The shear `tau_dz` of the full-tensor closures at the trace of `s`, per axis: the vertical
    shear (`expression` on the bulk, traced) plus the `rho nu d_d w` part.
    """
    from zoomy_core import coords as _C
    names = [{_C.x: "u", _C.y: "v"}[X] for X in s.horizontal]
    sh = _full_shear(full, s, nh)
    return [sum(s.trace(c.expression(s.bulk(alias={"u": names[i]}))) for c in full) + sh[i]
            for i in range(nh)]


def interface_traction(closures, stress, below, above, G):
    """Shared, single-valued part of the traction at an internal layer interface `z_l`, per axis,
    `sum_e sigma*_de d_e z_l - tau*_dz`, or `None` without the full tensor (`stress`).

    `sigma*` (of `tau_de - delta_de tau_zz`) and `tau*` (of the shear `tau_dz`) are the
    interface values of the two one-sided traces (`below`: top trace of the lower layer,
    `above`: bottom trace of the upper layer, both `ClosureState`) by the `InterfaceFlux`
    closure with `quantity=InterfaceQuantity.stress`, by default their mean.  Each layer's
    trace carries only its own in-plane part (`apply_layer_stress_closures`); the model adds
    this shared part like the transfer term (`+` at the top of the layer below, `-` at the
    bottom of the layer above), so the interface traction `tau*_dz - sigma*_de d_e z_l` is
    single-valued and the layer terms telescope.
    """
    full = full_tensor_closures(closures, stress)
    if not full:
        return None
    scheme = (interface_closure(closures, quantity=InterfaceQuantity.stress)
              or MeanInterface(quantity=InterfaceQuantity.stress))
    nh = len(below.horizontal)
    Sb, Sa = inplane_stress(full, below, nh), inplane_stress(full, above, nh)
    Tb, Ta = _shear_trace(full, below, nh), _shear_trace(full, above, nh)
    slope = below.interface_slopes
    return [sum(scheme.expression(Sb[d_][e], Sa[d_][e], G) * slope[e] for e in range(nh))
            - scheme.expression(Tb[d_], Ta[d_], G) for d_ in range(nh)]


# ── depth-averaged SWE closure consumption ──────────────────────────────────
# SWE states its operators directly (no moment derivation that calls
# apply_stress_closures), so it consumes closures through this hook instead:
# the closure supplies the constitutive COEFFICIENT, the model the operator
# STRUCTURE (2-D vector source / rank-4 velocity-diffusion tensor).


class _SWEField:
    """Minimal `FunctionFamily` stand-in for a depth-averaged SWE field, whose bulk and trace
    expressions coincide.
    """

    def __init__(self, expr):
        """Store the field expression."""
        self.expr = expr

    def at(self, loc):              # bottom/surface trace == the depth average
        """Return the expression itself; the bed and surface traces equal the depth average."""
        return self.expr


def swe_closure_state(model):
    """Build a `ClosureState` over a depth-averaged SWE model, exposing `s.u`, `s.w`,
    `s.h`, `s.speed` (floored horizontal magnitude) and `s.par`, with velocity
    desingularised through the `hinv` aux.
    """
    v = model.variables
    a = model.aux_variables
    hinv = a.hinv
    u = v.hu * hinv
    w = v.hv * hinv if hasattr(v, "hv") else sp.S.Zero
    speed = sp.sqrt(u * u + w * w + 1e-12)
    fields = {"u": _SWEField(u), "w": _SWEField(w),
              "h": _SWEField(v.h), "speed": _SWEField(speed)}
    return ClosureState(fields, params=model.parameters)


__all__ = ["Closure", "ClosureState", "apply_stress_closures",
           "apply_layer_stress_closures", "interface_closure",
           "interface_traction", "inplane_stress",
           "Newtonian", "KEpsilonViscosity", "QRViscosity", "NavierSlip", "RoughWall",
           "WallFunctionBed",
           "StressFree", "InterfaceFlux", "MeanInterface", "UpwindInterface",
           "ManningFriction", "EddyViscosity", "full_tensor_closures", "apply_full_tensor",
           "swe_closure_state"]
