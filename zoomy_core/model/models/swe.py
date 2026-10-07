"""SWE — shallow water with bed, dimension-agnostic (1D / 2D).

The level-0 member of the moment hierarchy as an OFFICIAL model class:
state ``[b, h, hu]`` (1D) or ``[b, h, hu, hv]`` (2D), pressure in the
flux, bed slope as a nonconservative product, optional Manning friction.
(The depth-averaged operators are stated directly — the declarative
vertical derivation machinery is (t, x, z) only today; when it grows a
transverse coordinate this class collapses onto ``SME(level=0)``.)

Couples through the same canonical contract as SME: ``interpolate``
lifts to the flat profile ``u(ζ) = hu/h`` (+ hydrostatic p), ``project``
depth-averages the sampled column — a constant profile round-trips
exactly.  Both ride the standard function-group slots
(:func:`zoomy_core.systemmodel.system_model._attach_function_groups`).

Boundary definitions (``FromModel(tag=…, definition=…)``):

- ``wall``: free-slip wall — reflects ONLY the normal component of the
  momentum vector, ``q → q − 2 n (n·q)``, and leaves the tangential
  component untouched.  Written in the face-normal symbols ``n_d``, which
  ``FromModel`` substitutes per face, so ONE definition serves every wall
  orientation (there are no direction-specific ``wall_x``/``wall_y``
  variants any more — they were the workaround for the missing normal),
- ``inflow``: prescribed depth/discharge via the model parameters
  ``h_in`` / ``q_in`` (transverse momentum zero, bed extrapolates).
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import merge_parameter_overrides
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import (
    SystemModel, face_normal_symbols)

__all__ = ["SWE"]


class SWE(StructuredDerivativeModel):
    """Shallow water + bed (+ Manning friction), 1D or 2D."""

    dimension = param.Integer(default=1, bounds=(1, 2))
    # ``h`` is declared NONNEGATIVE (h >= 0, not h > 0: wet/dry admits a dry
    # cell h = 0).  The assumption reaches the sympy Symbol, so the CAS pulls
    # the depth out of the radicals the face kernels are built from —
    # sqrt(g*h**5) → sqrt(g)*h**(5/2), sqrt(h**2) → h — instead of carrying
    # them as Abs/composite powers no CSE pass can share.  Measured on the
    # three face operators of PositiveNonconservativeHLL / 2-D SWE:
    # post-CSE ops 256 → 224 (-12.5%), CSE temps 68 → 52.
    variables = ["b", ("h", "nonnegative"), "hu"]   # + "hv" when dimension == 2
    parameters = {
        "g":    (9.81, "positive"),
        "n_m":  (0.0, "nonnegative"),     # Manning; 0 = frictionless
        "h_in": (0.0, "nonnegative"),     # inflow target depth
        "q_in": (0.0, "real"),            # inflow target discharge
        # Velocity-magnitude regularizer for the Manning friction: keeps the
        # source Jacobian ∂S/∂Q FINITE at rest (REQ-166).  |u|=sqrt(hu²+hv²)
        # has derivative hu/|u| → 0/0 at zero velocity, so the unregularized
        # source NaNs the implicit/IMEX Newton on step 0 of any from-rest run.
        "vel_eps": (1e-12, "nonnegative"),
        # Wet/dry depth threshold (REQ-48 canonical name — same symbol the FVM
        # Riemann solvers, jax backend and MalpassetSWE read).  Sets the depth
        # below which ``update_variables`` caps the momentum band to zero.
        "wet_dry_eps": (1e-2, "positive"),
    }

    def __init__(self, **kw):
        if int(kw.get("dimension", 1)) == 2:
            self.variables = ["b", ("h", "nonnegative"), "hu", "hv"]
        # constructor parameters override the declared defaults ONE BY ONE
        # (a partial dict must not drop g) and KEEP the declared constraint
        # (a bare ``{"g": 9.81}`` must not drop g's positivity)
        kw["parameters"] = merge_parameter_overrides(
            type(self).parameters, kw.get("parameters"))
        # FromModel BCs resolve against the SystemModel — hold them aside
        # and attach in ``system_model`` (the SME semantics), instead of
        # letting the production pipeline resolve them at construction.
        self._coupling_bcs = kw.pop("boundary_conditions", None)
        super().__init__(**kw)
        self._function_groups = self._build_function_groups()

    # ── well-balanced reconstruction (model owns it) ───────────────────
    def reconstruction_variables(self):
        """Primitive well-balanced reconstruction map ``state → primitive``.

        Limit the FREE SURFACE ``eta = b + h`` instead of the conservative
        depth ``h`` so a slope/oscillation limiter is inert at lake-at-rest
        and does not corrupt the flat surface (the wet/dry "water creeping
        up the walls" defect).  Every other field reconstructs as identity;
        the inverse ``h = eta - b`` is auto-derived by
        :meth:`state_from_reconstruction`.

        Pure sympy, resolved by FIELD NAME — no index assumptions — so it
        codegens to every backend (UFL / numpy / jax / OpenFOAM) regardless
        of where ``h`` and ``b`` sit in the state vector.  Mirrors
        ``SME``'s ``_reconstruction_rows = {h: b + h}``.
        """
        v = self.variables
        eta = v.b + v.h
        return ZArray([eta if s == v.h else s for s in v.get_list()])

    # ── operators ─────────────────────────────────────────────────────
    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.parameters.g
        F = sp.Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu
        F[2, 0] = hu * hu / h + sp.Rational(1, 2) * g * h * h
        if self.dimension == 2:
            hv = self.Q.hv
            F[3, 0] = hu * hv / h
            F[1, 1] = hv
            F[2, 1] = hu * hv / h
            F[3, 1] = hv * hv / h + sp.Rational(1, 2) * g * h * h
        return ZArray(F)

    def nonconservative_matrix(self):
        """Bed slope: the momentum rows pick up ``g·h·∂_d b``."""
        h = self.Q.h
        g = self.parameters.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        if self.dimension == 2:
            B[3][0][1] = g * h
        return ZArray(B)

    def source(self):
        """Manning friction on the momentum rows (off for n_m = 0).

        The velocity magnitude carries the ``vel_eps`` regularizer,
        ``|u| = sqrt(hu² + hv² + vel_eps)``, so the source Jacobian ∂S/∂Q is
        FINITE at rest (REQ-166).  Without it the 2-D term ``hu·sqrt(hu²+hv²)``
        has derivative ``sqrt(...) + hu²/sqrt(...) → 0/0`` at zero velocity,
        which NaNs the implicit/IMEX Newton on step 0 of any from-rest run.
        With eps ~ 1e-12 the friction is unchanged to ~1e-6 in velocity while
        the Jacobian is smooth everywhere.  1-D uses the same regularized
        magnitude (the old ``hu·|hu|`` was Jacobian-finite but non-smooth).
        """
        h, hu = self.Q.h, self.Q.hu
        g, n_m, eps = (self.parameters.g, self.parameters.n_m,
                       self.parameters.vel_eps)
        S = sp.Matrix.zeros(self.n_variables, 1)
        if self.dimension == 1:
            mag = sp.sqrt(hu * hu + eps)
            S[2, 0] = -g * n_m**2 * hu * mag / h ** sp.Rational(7, 3)
        else:
            hv = self.Q.hv
            mag = sp.sqrt(hu * hu + hv * hv + eps)
            S[2, 0] = -g * n_m**2 * hu * mag / h ** sp.Rational(7, 3)
            S[3, 0] = -g * n_m**2 * hv * mag / h ** sp.Rational(7, 3)
        return ZArray(S)

    # ── state hygiene: wet/dry momentum cap (DG(1) dry-bed dt-collapse) ──
    def update_variables(self):
        """Per-cell wet/dry momentum cap — ``h`` and ``b`` NEVER modified.

        The DG(1) shoreline dt-collapse fix, lifted from :class:`MalpassetSWE`
        into the base so every SWE consumer (numpy / jax / OpenFOAM /
        Firedrake) inherits it.  A Riemann/limiter transient can leave a large
        momentum ``hu`` sitting on a vanishing depth ``h → wet_dry_eps``; the
        derived velocity ``u = hu/h`` then blows up and collapses the explicit
        ``dt`` at the wet/dry front.  Cap each momentum COMPONENT to the
        physically admissible band ``|hu| ≤ h_wet · 4·√(g·max(h,0))`` with
        ``h_wet = max(h − wet_dry_eps, 0)``, so the bound collapses smoothly to
        zero as the cell dries (the celerity ``2√(gh)`` sets the scale; the
        factor 4 keeps the cap loose enough to stay inert on wet supercritical
        flow, where ``|hu| = h|u| ≪ 4h√(gh)`` for Froude ≲ 4).

        A plain ``h``-floor is deliberately NOT the fix: flooring ``h`` moves
        the bad ``(h, hu)`` pair up a threshold instead of bounding the
        momentum, so the ``1/h`` blow-up simply reappears at the new floor.

        ``Min``/``Max``/``sqrt`` only (no ``Heaviside``/``sign``/``Piecewise``)
        so it lowers through every backend; resolved BY FIELD NAME so it is
        dimension-agnostic — ``hv`` is capped only when present, ``h``/``b``
        pass through untouched.
        """
        v = self.variables
        g = self.parameters.g
        h = v.h
        h_wet = sp.Max(h - self.parameters.wet_dry_eps, sp.S.Zero)
        max_hu = h_wet * 4 * sp.sqrt(g * sp.Max(h, sp.S.Zero))

        def cap(c):
            return sp.Max(-max_hu, sp.Min(c, max_hu))

        momenta = {"hu", "hv"}
        return ZArray([cap(s) if str(s) in momenta else s
                       for s in v.get_list()])

    def update_variables_jacobian_wrt_variables(self):
        # The Min/Max cap differentiates to non-serialisable Piecewise/sign;
        # this Jacobian is only consumed by the IMEX implicit-update Newton
        # (which does not run on update_variables), so return explicit zeros
        # (mirrors MalpassetSWE).
        n = self.n_variables
        return ZArray.zeros(n, n)

    # ── function groups: canonical operators + boundary definitions ───
    def _build_function_groups(self):
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        g = self.parameters.g
        p = self.parameters
        zeta = sp.Symbol("zeta", real=True)
        P3b = sp.Symbol("P3_b", real=True)
        P3h = sp.Symbol("P3_h", real=True)

        # flat-profile lift to canonical [b, h, u, v, w, p](ζ)
        interp = {0: b, 1: h, 2: hu / h, 5: g * h * (1 - zeta)}
        # depth-average of the sampled column as the Integral-FREE, fixed-node
        # Galerkin reduction q = h·⟨φ_0, u⟩ over a level-0 (constant) basis —
        # the SAME ``Basisfunction.projection_rows`` machinery every moment
        # model uses (SME/ML/VAM), never a raw ``sp.Integral`` (which the C
        # printers cannot lower, and which must never be ``.doit()``-ed).  Two
        # trapezoid nodes are exact for the flat profile; the ``P3_u(ζ_j)``
        # samples bind to the column slots the printer maps.  Round-trips a
        # constant column to q = h·U exactly.
        legendre = Legendre_shifted(level=0)
        nodes = [0.0, 1.0]
        weights = [0.5, 0.5]
        P3u = sp.Function("P3_u", real=True)
        project = {b: P3b, h: P3h,
                   hu: legendre.projection_rows(
                       nodes, weights, [P3u(nd) for nd in nodes],
                       norm=lambda _k: P3h)[0]}
        # WB primitive map: limit η = b+h and the velocities
        recon = {h: b + h, hu: hu / h}
        # free-slip wall: reflect the momentum VECTOR about the face,
        # q_ghost = q - 2 n (n·q).  In 1-D n0 = ±1 ⇒ the familiar -hu; in 2-D
        # the tangential component is preserved exactly (a normal-blind
        # -hu/-hv would destroy it).  ``FromModel`` binds n_d per face.
        nrm = face_normal_symbols(self.dimension)
        qvec = [hu] + ([self.Q.hv] if self.dimension == 2 else [])
        qn = sum(nd * qd for nd, qd in zip(nrm, qvec))
        wall = {qd: qd - 2 * nd * qn for nd, qd in zip(nrm, qvec)}
        groups = {
            "interpolate": interp,
            "project": project,
            "reconstruction": recon,
            "boundary:wall":   wall,
            "boundary:inflow": {h: p.h_in, hu: p.q_in},
        }
        if self.dimension == 2:
            hv = self.Q.hv
            interp[3] = hv / h
            P3v = sp.Function("P3_v", real=True)
            project[hv] = legendre.projection_rows(
                nodes, weights, [P3v(nd) for nd in nodes],
                norm=lambda _k: P3h)[0]
            recon[hv] = hv / h
            groups["boundary:inflow"][hv] = sp.S.Zero
        return groups

    # ── declared roles (free-surface specialisation) ─────────────────
    def declare_system_roles(self, sm):
        """Declare the free-surface roles on the built system: depth ``h``,
        bed ``b`` (stationary), gravity ``g``, momentum ``hu`` (``hv``).
        Generic solver code reads these roles, never the field names."""
        from zoomy_core.systemmodel.roles import declare_free_surface_roles
        momentum = [["hu"]] + ([["hv"]] if int(self.dimension) == 2 else [])
        declare_free_surface_roles(sm, depth="h", bed="b", gravity="g",
                                   momentum=momentum,
                                   description="SWE free-surface roles")

    # ── runtime form ──────────────────────────────────────────────────
    # Build via ``SystemModel.from_model(SWE(...))`` (REQ-143); the builder in
    # ``zoomy_core.systemmodel.model_builders`` runs the low-level path.
    _system_model_kind = "swe"
