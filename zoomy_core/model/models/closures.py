"""Composable stress closures: each ``Closure`` is one constitutive relation for a stress
component (bulk, bed trace, surface trace, horizontal or in-plane) over a ``ClosureState``.
Models list them in ``closures=[...]`` and perform the substitution when the Model is derived,
before the lift to a SystemModel.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation
from zoomy_core.model.models.material import ClosureState   # the full-access state


def _solve_traction(P, s, nh):
    """Solve the boundary frame system ``t_a . (tau n) = P_a`` for the shear-stress traces
    ``tau_iz`` given the prescribed tangential tractions ``P``; they reduce to ``tau_iz = P_i``
    under small-slope scaling.
    """
    nvec, tans = s.normal, s.tangents
    tau = [sp.Symbol(f"__tauz{i}") for i in range(nh)]
    nz = nvec[nh]
    taudotn = sp.Matrix([tau[i] * nz for i in range(nh)]
                        + [sum(tau[i] * nvec[i] for i in range(nh))])
    eqs = [sp.Eq((tans[a].T * taudotn)[0], P[a]) for a in range(nh)]
    sol = sp.solve(eqs, tau, dict=True)
    if not sol:
        raise ValueError("boundary traction frame solve failed for "
                         f"P={P} (nh={nh})")
    return [sp.simplify(sol[0][tau[i]]) for i in range(nh)]


def apply_stress_closures(closures, m, axes, state, horiz):
    """Inject the bulk, bed and surface stress closures into the projected momentum equations of
    SME/VAM, frame-aware and dimension-agnostic.

    Args:
        closures: list of ``Closure`` objects.
        m: derivation model.
        axes: one ``{"mx": eq, "tau": FieldHandle, "velname": "u"|"v"}`` per horizontal momentum
            component.
        state: callable ``state(at, *, alias=None, btag=None) -> ClosureState``.
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
    for at, btag, pieces in ((1, "eta", surface), (0, "b", bottom)):
        if not pieces:
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh)
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    for c in bulk:                                # diagonal constitutive
        for ax in axes:
            ax["mx"].apply({ax["tau"].expr:
                            c.expression(state(None, alias={"u": ax["velname"]}))})
    return bool(bulk)


class Closure(Operation):
    """Base class of a one-component stress closure; subclasses set the immutable ``closes`` /
    ``requires`` and implement ``expression``.
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
        """Class name plus ``_key_content``, so the derivation-cache canonicaliser sees the
        configuration.
        """
        # The derivation-cache canonicaliser falls back to ``repr`` for callable
        # Operations, so the repr MUST carry the result-affecting config.
        return f"{type(self).__name__}({self._key_content()})"

    def register(self, model):
        """Register-or-query the model parameters this closure needs (override)."""

    def check(self, model):
        """Assert every field in ``requires`` is present in the model."""
        have = set(model.functions.keys())
        for f in self.requires:
            assert f in have, (
                f"{type(self).__name__} closure needs field {f!r}, which this "
                f"model does not define (has: {sorted(have)})")

    def expression(self, s):
        """Return the symbolic relation for this stress component over the ``ClosureState``
        ``s`` (override).
        """
        raise NotImplementedError

    def friction_velocity(self, s):
        """Bed friction velocity ``u_star`` this closure defines, or ``None``; bed drag laws
        override it so an eddy viscosity can consume it.
        """
        return None

    def traction(self, s):
        """Boundary traction in the local frame ``{n, t_a}``; the default wraps the scalar
        ``expression`` as one tangential component.

        Returns:
            ``{"normal": n.sigma.n or None, "tangent": [t_a.sigma.n, ...]}``, one tangent entry
                per horizontal direction.
        """
        return {"normal": None, "tangent": [self.expression(s)]}


# ── bulk closures ──────────────────────────────────────────────────────────


class Newtonian(Closure):
    """Newtonian bulk stress ``tau = rho nu d_z u``."""
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        """Register the kinematic viscosity ``nu``."""
        m.parameter("nu", 0.0)

    def expression(self, s):
        """Return ``rho nu d_z u``."""
        return s.par.rho * s.par.nu * s.dz(s.u)


class KEpsilonViscosity(Closure):
    """Bulk stress ``tau = rho nu_t d_z u`` with the k-epsilon eddy viscosity ``nu_t = C_mu
    k^2/epsilon``; needs ``k`` and ``varepsilon`` fields and ``quadrature_order > 0``.

    Args:
        wall_floor: add the near-wall mixing-length viscosity ``C_mu^(1/4) sqrt(k) kappa z_p``
            so the stress still damps the velocity moments at the wall.
    """
    closes = "bulk"; requires = ("u", "k", "varepsilon")

    def __init__(self, name=None, wall_floor=True):
        """Store the ``wall_floor`` switch."""
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        """Register ``C_mu``, ``nu`` and, with ``wall_floor``, ``kappa`` and ``z_p``."""
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return ``rho (nu + nu_t) d_z u`` with ``nu_t = C_mu k^2/epsilon`` plus the optional
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
    sk^4/se^2`` for the transported ``sk = sqrt(k)``, ``se = sqrt(epsilon)``; needs
    ``quadrature_order > 0``.

    Args:
        wall_floor: add the near-wall mixing-length viscosity, as in ``KEpsilonViscosity``.
    """
    closes = "bulk"; requires = ("u", "sk", "se")

    def __init__(self, name=None, wall_floor=True):
        """Store the ``wall_floor`` switch."""
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        """Register ``C_mu``, ``nu``, the floor ``eps_min`` and, with ``wall_floor``, ``kappa``
        and ``z_p``.
        """
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        m.parameter("eps_min", 0.0)   # realizability floor: ε → ε + eps_min
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return ``rho (nu + nu_t) d_z u`` with ``nu_t = C_mu sk^4/(se^2 + eps_min)`` plus the
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
    """Regularized Bingham bulk stress ``tau = (rho nu + tau_y/sqrt((d_z u)^2 + eps^2)) d_z u``,
    rigid below the yield stress ``tau_y`` as ``eps -> 0``.
    Not analytically integrable, so build the model with ``quadrature_order > 0``.
    """
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        """Register ``nu``, the yield stress ``tau_y`` and the regularization scale ``eps_reg``."""
        m.parameter("nu", 0.0); m.parameter("tau_y", 0.0); m.parameter("eps_reg", 1e-3)

    def expression(self, s):
        """Return the regularized Bingham stress."""
        return (s.par.rho * s.par.nu
                + s.par.tau_y / sp.sqrt(s.dz(s.u) ** 2 + s.par.eps_reg ** 2)) * s.dz(s.u)


class ElderViscosity(Closure):
    """Algebraic Elder eddy viscosity ``nu_t = kappa u_star h zeta (1 - zeta)``, polynomial in
    zeta so the Galerkin projection closes analytically.

    Args:
        friction: optional bed closure whose live ``friction_velocity`` replaces the ``u_star``
            parameter.
    """
    closes = "bulk"; requires = ("u",)

    def __init__(self, name=None, friction=None):
        """Store the optional bed closure supplying ``u_star``."""
        super().__init__(name=name)
        self.friction = friction      # a bottom closure exposing friction_velocity

    def register(self, m):
        """Register ``kappa``, ``u_star``, ``nu`` and the parameters of the bed closure, if any."""
        m.parameter("kappa", 0.41)
        m.parameter("u_star", 0.0)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        if self.friction is not None:
            self.friction.register(m)  # ensure the bed law's params (k_s, z_p) exist

    def expression(self, s):
        """Return ``rho (nu + nu_t) d_z u`` with the Elder ``nu_t``."""
        u_star = (self.friction.friction_velocity(s) if self.friction is not None
                  else s.par.u_star)
        nu_t = s.par.kappa * u_star * s.depth * s.zeta * (1 - s.zeta)
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


# ── bottom (bed) closures ──────────────────────────────────────────────────


class NavierSlip(Closure):
    """Linear Navier slip at the bed, ``tau_b = lambda_s u_b``."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register the slip coefficient ``lambda_s``."""
        m.parameter("lambda_s", 0.0)

    def expression(self, s):
        """Return ``lambda_s u``."""
        return s.par.lambda_s * s.u

    def traction(self, s):
        """Tangential traction ``lambda_s u_t`` per axis (reduces to ``lambda_s u_i`` under
        small-slope scaling).
        """
        return {"normal": None,
                "tangent": [s.par.lambda_s * ut for ut in s.u_tangent]}


class RoughWall(Closure):
    """Turbulent rough-wall bed drag ``tau_b = rho C_f u_b |u_b|`` with ``C_f =
    (kappa/ln(z_p/z_0))^2`` and ``z_0 = k_s/30``.
    Needs ``k_s`` (Nikuradse roughness), ``z_p`` and ``kappa``.
    """
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register ``kappa``, ``k_s`` and ``z_p``."""
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def expression(self, s):
        """Return ``rho C_f u |u|``."""
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        return s.par.rho * Cf * s.u * sp.Abs(s.u)

    def friction_velocity(self, s):
        """Recovered bed friction velocity ``sqrt(C_f) |u_b|`` of the horizontal bed trace,
        consumed by ``ElderViscosity``.
        """
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        ub = s.velocity_at(0)                        # bed trace [u, (v,) w]
        u_b_mag = sp.sqrt(sum(ub[i] ** 2 for i in range(len(ub) - 1)))  # horizontal
        return sp.sqrt(Cf) * u_b_mag

    def traction(self, s):
        """Quadratic bed drag as a vector traction ``rho C_f |U_t| u_t`` opposing the slip
        velocity, coupling the x/y components through ``|U_t|``.
        """
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class WallFunctionBed(Closure):
    """k-epsilon rough-wall-function bed stress ``tau_b = rho C_f |U_p| U_p`` with the log-law
    ``C_f`` of ``RoughWall`` but the velocity ``U_p`` read at the near-wall height ``zeta_p =
    z_p/h`` rather than at the bed trace.
    Needs ``k_s``, ``z_p`` and ``kappa``.
    """
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        """Register ``kappa``, ``k_s`` and ``z_p``."""
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def _Cf(self, s):
        """Log-law drag coefficient ``(kappa/ln(z_p/(k_s/30)))^2``."""
        return (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2

    def expression(self, s):
        """Scalar 1-D drag ``rho C_f U_p |U_p|`` from the near-wall velocity."""
        # scalar 1-D fallback: drag from the near-wall reference velocity U_p
        Up = s.velocity_at(s.par.z_p / s.depth)[0]
        return s.par.rho * self._Cf(s) * Up * sp.Abs(Up)

    def traction(self, s):
        """Vector bed traction ``rho C_f |U_p| U_p`` from the tangential velocity at ``zeta_p =
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
    """Manning bed friction as a bottom closure for depth-averaged SWE; ``expression`` is the
    rate ``-g n^2 |u| / max(h, h_floor)^(1/3)`` multiplying each velocity component.

    Args:
        h_floor: depth floor that keeps friction finite at the wet/dry interface.
    """
    closes = "bottom"; requires = ("u", "h")

    def __init__(self, name=None, h_floor=0.0):
        """Store the depth floor ``h_floor``."""
        super().__init__(name=name)
        self.h_floor = float(h_floor)

    def register(self, m):
        """Register ``g`` and the Manning coefficient ``n``."""
        m.parameter("g", 9.81); m.parameter("n", 0.0)

    def expression(self, s):
        """Return the friction rate (per unit velocity) for the depth-averaged SWE."""
        h_eff = sp.Max(s.h, sp.Float(self.h_floor)) if self.h_floor > 0 else s.h
        return -s.par.g * s.par.n ** 2 * s.speed / h_eff ** sp.Rational(1, 3)

    def traction(self, s):
        """Bed shear stress vector ``rho g n^2 |U_t| u_t / h^(1/3)`` for the moment hierarchy,
        the Manning analogue of ``RoughWall.traction``.
        """
        h = s.depth
        h_eff = sp.Max(h, sp.Float(self.h_floor)) if self.h_floor > 0 else h
        Cf = s.par.g * s.par.n ** 2 / h_eff ** sp.Rational(1, 3)
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class EddyViscosity(Closure):
    """Horizontal eddy viscosity closure for the depth-averaged mixing ``div(nu h grad u)``;
    ``expression`` returns the isotropic ``nu`` and the SWE model builds the diffusion tensor
    from it.
    """
    closes = "horizontal"; requires = ("u",)

    def register(self, m):
        """Register the kinematic viscosity ``nu``."""
        m.parameter("nu", 0.0)

    def expression(self, s):
        """Return ``nu``."""
        return s.par.nu


class ShallowInPlane(Closure):
    """Opt-in shallow form: drop the in-plane deviatoric stress
    (``tau_xx, tau_xy, tau_yy -> 0``) of a full-stress model.
    """
    closes = "horizontal"; requires = ()

    def expression(self, s):
        """Return zero."""
        return sp.S.Zero


class NewtonianInPlane(Closure):
    """Marker closure that keeps the in-plane deviatoric stress as incompressible Newtonian,
    ``tau_de = rho nu (d_d u_e + d_e u_d)``, the complement of ``ShallowInPlane``.
    The model adds the divergence (``add_inplane_viscous``) and routes it into
    ``diffusion_matrix`` (``package_viscous``).
    """
    closes = "in_plane"; requires = ("u",)

    def register(self, m):
        """Register the kinematic viscosity ``nu``."""
        m.parameter("nu", 0.0)

    def expression(self, s):
        """Return ``rho nu d_z u``; the in-plane tensor itself is built by the model, not by
        this scalar.
        """
        # in-plane stress is tensor-valued and consumed by the model's
        # add_inplane_viscous (not the scalar apply_stress_closures path)
        return s.par.rho * s.par.nu * s.dz(s.u)


# ── surface closures ───────────────────────────────────────────────────────


class StressFree(Closure):
    """Stress-free free surface, ``tau(1) = 0``."""
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


class InterfaceFlux(Closure):
    """Base of the multilayer interface transfer-velocity schemes;
    ``expression(below, above, G)`` takes the two one-sided interface traces and the interface
    mass flux ``G``.
    """
    closes = "interface"; requires = ()

    def expression(self, below, above, G):    # noqa: D401 - distinct signature
        """Return the shared interface velocity ``u*`` (override)."""
        raise NotImplementedError


class MeanInterface(InterfaceFlux):
    """Central interface velocity ``u* = (u_below + u_above)/2``."""

    def expression(self, below, above, G):
        """Return the arithmetic mean of the two traces."""
        return (below + above) / 2


class UpwindInterface(InterfaceFlux):
    """Donor interface velocity chosen by the sign of the interface flux, ``u* = u_below`` if
    ``G >= 0`` else ``u_above``.
    """

    def expression(self, below, above, G):
        """Return the upwind trace selected by the sign of ``G``."""
        return sp.Piecewise((below, G >= 0), (above, True))


def apply_layer_stress_closures(closures, m, axes, state, *, is_top, is_bottom):
    """Per-layer variant of ``apply_stress_closures`` for the multilayer models: the surface
    closure acts only on the top layer and the bed closure only on the bottom layer.

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
    surface = [c for c in stress if c.closes == "surface"] if is_top else []
    bottom = [c for c in stress if c.closes == "bottom"] if is_bottom else []
    bulk = [c for c in stress if c.closes == "bulk"]
    for at, btag, pieces in ((1, "eta", surface), (0, "b", bottom)):
        if not pieces:
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh)
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    for c in bulk:
        for ax in axes:
            ax["mx"].apply({ax["tau"].expr:
                            c.expression(state(None, alias={"u": ax["velname"]}))})
    return bool(bulk)


def interface_closure(closures):
    """Return the ``InterfaceFlux`` closure in ``closures``, or ``None``."""
    return next((c for c in (closures or []) if c.closes == "interface"), None)


# ── depth-averaged SWE closure consumption ──────────────────────────────────
# SWE states its operators directly (no moment derivation that calls
# apply_stress_closures), so it consumes closures through this hook instead:
# the closure supplies the constitutive COEFFICIENT, the model the operator
# STRUCTURE (2-D vector source / rank-4 velocity-diffusion tensor).


class _SWEField:
    """Minimal ``FunctionFamily`` stand-in for a depth-averaged SWE field, whose bulk and trace
    expressions coincide.
    """

    def __init__(self, expr):
        """Store the field expression."""
        self.expr = expr

    def at(self, loc):              # bottom/surface trace == the depth average
        """Return the expression itself; the bed and surface traces equal the depth average."""
        return self.expr


def swe_closure_state(model):
    """Build a ``ClosureState`` over a depth-averaged SWE model, exposing ``s.u``, ``s.w``,
    ``s.h``, ``s.speed`` (floored horizontal magnitude) and ``s.par``, with velocity
    desingularised through the ``hinv`` aux.
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
           "Newtonian", "KEpsilonViscosity", "QRViscosity", "NavierSlip", "RoughWall",
           "WallFunctionBed",
           "StressFree", "InterfaceFlux", "MeanInterface", "UpwindInterface",
           "ManningFriction", "EddyViscosity", "ShallowInPlane", "NewtonianInPlane",
           "swe_closure_state"]
