"""Shallow water equations with bed (SWE), 1D or 2D, as a Model with state [b, h, hu(, hv)].
Level-0 member of the moment hierarchy; lifted to a SystemModel via ``SystemModel.from_model``.
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
    """Shallow water + bed + optional Manning friction, written directly as depth-averaged
    operators. Declares the function groups interpolate / project / reconstruction and the
    boundaries ``wall`` (free slip) and ``inflow``.
    """

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
        """Build the model; ``dimension=2`` adds ``hv``, and ``parameters`` overrides defaults
        one by one.
        """
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
        """Well-balanced reconstruction map: replace ``h`` by the free surface ``b + h``, other
        fields unchanged.
        """
        v = self.variables
        eta = v.b + v.h
        return ZArray([eta if s == v.h else s for s in v.get_list()])

    # ── operators ─────────────────────────────────────────────────────
    def flux(self):
        """Conservative flux of mass and momentum, ``hu^2/h + g h^2/2`` on the diagonal."""
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
        """Bed-slope product: the momentum rows pick up ``g h d_d b``."""
        h = self.Q.h
        g = self.parameters.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        if self.dimension == 2:
            B[3][0][1] = g * h
        return ZArray(B)

    def source(self):
        """Manning friction on the momentum rows (zero for ``n_m = 0``), with ``vel_eps``
        regularizing ``|u|`` so the source Jacobian stays finite at rest.
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
        """Cap each momentum component to ``h_wet * 4 sqrt(g h)``
        (``h_wet = max(h - wet_dry_eps, 0)``) at the wet/dry front; ``h`` and ``b`` are never
        modified.
        """
        v = self.variables
        g = self.parameters.g
        h = v.h
        h_wet = sp.Max(h - self.parameters.wet_dry_eps, sp.S.Zero)
        max_hu = h_wet * 4 * sp.sqrt(g * sp.Max(h, sp.S.Zero))

        def cap(c):
            """Clip ``c`` to ``[-max_hu, max_hu]``."""
            return sp.Max(-max_hu, sp.Min(c, max_hu))

        momenta = {"hu", "hv"}
        return ZArray([cap(s) if str(s) in momenta else s
                       for s in v.get_list()])

    def update_variables_jacobian_wrt_variables(self):
        """Zero Jacobian of ``update_variables`` (only the IMEX Newton consumes it, and it does
        not apply the cap).
        """
        # The Min/Max cap differentiates to non-serialisable Piecewise/sign;
        # this Jacobian is only consumed by the IMEX implicit-update Newton
        # (which does not run on update_variables), so return explicit zeros
        # (mirrors MalpassetSWE).
        n = self.n_variables
        return ZArray.zeros(n, n)

    # ── function groups: canonical operators + boundary definitions ───
    def _build_function_groups(self):
        """Return the function groups interpolate, project, reconstruction and boundary:wall /
        boundary:inflow.
        """
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
        """Declare the free-surface roles (depth ``h``, bed ``b``, gravity ``g``, momentum
        ``hu``/``hv``) on the SystemModel ``sm``.
        """
        from zoomy_core.systemmodel.roles import declare_free_surface_roles
        momentum = [["hu"]] + ([["hv"]] if int(self.dimension) == 2 else [])
        declare_free_surface_roles(sm, depth="h", bed="b", gravity="g",
                                   momentum=momentum,
                                   description="SWE free-surface roles")

    # ── runtime form ──────────────────────────────────────────────────
    # Build via ``SystemModel.from_model(SWE(...))`` (REQ-143); the builder in
    # ``zoomy_core.systemmodel.model_builders`` runs the low-level path.
    _system_model_kind = "swe"
