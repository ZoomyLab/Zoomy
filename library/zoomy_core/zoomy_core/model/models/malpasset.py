"""Viscous wet/dry SWE with Manning friction (the Malpasset variant), derived from
`SWE`. Used by the Firedrake dam-break and Malpasset cases;
`SystemModel.from_model` picks its operators up by name.
"""
from __future__ import annotations

import sympy as sp
from sympy import sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.closures import (
    ManningFriction, EddyViscosity, swe_closure_state)

__all__ = ["MalpassetSWE"]


class MalpassetSWE(SWE):
    """Viscous wetting/drying SWE with state `[b, h, hu, hv]` and aux `[hinv]`, dissipation
    closed by `closures`. The pressure `g h^2/2` is a separate operator so the well-balanced
    Riemann solvers can reconstruct it.
    """

    # Own state + parameter set (overrides the SWE class defaults so the
    # SWE.__init__ parameter merge does NOT drag in n_m / h_in / q_in).
    variables = ["b", ("h", "nonnegative"), "hu", "hv"]
    parameters = {
        "g":   (9.81, "positive"),
        "n":   (0.033, "nonnegative"),   # Manning roughness
        "nu":  (1.0, "nonnegative"),     # eddy viscosity
        # Canonical wet/dry-threshold parameter name (REQ-48): the SAME
        # symbol the FVM Riemann solvers (riemann_solvers._eps_symbol)
        # and the jax backend (parameter_values.wet_dry_eps) read.
        "wet_dry_eps": (1e-2, "positive"),  # wet/dry depth threshold
    }

    #: per-cell velocity cap used by :meth:`update_variables` (m/s).
    U_MAX = 30.0

    def __init__(self, *, g=9.81, n=0.033, nu=1.0, eps=1e-2,
                 h_friction_floor=0.5, ev_gate=True, closures=None, **kw):
        """Build the model.

        Args:
            eps: wet/dry depth threshold (declared as `wet_dry_eps`).
            h_friction_floor: depth floor in the Manning `h^(-1/3)`.
            ev_gate: zero the eigenvalues of dry cells (default on).
            closures: bottom / horizontal closures; default Manning friction + eddy viscosity.
        """
        # ``eps`` stays the public constructor knob (the wet/dry depth
        # threshold value); it is declared on the parameter vector under
        # the canonical name ``wet_dry_eps`` (REQ-48), so there is exactly
        # ONE wet/dry parameter (no duplicate ``eps`` entry).
        # Numerical constants that are NOT physical parameters (kept off
        # the parameter vector so the SystemModel parameter set stays
        # exactly {g, n, nu, wet_dry_eps}).
        self._h_friction_floor = float(h_friction_floor)
        self._ev_gate = bool(ev_gate)
        # Dissipation is CLOSED by composable closures (closures.py): bed
        # friction (bottom trace) + horizontal eddy mixing (τ_xx/τ_xy).
        # source() / diffusion_matrix_explicit() are BUILT from these — no
        # hand-rolled dissipation operators.  Set before super().__init__ so
        # the init-time operator lambdification can consume them.
        self.closures = (list(closures) if closures is not None else
                         [ManningFriction(h_floor=float(h_friction_floor)),
                          EddyViscosity()])
        # SWE.__init__ pops boundary_conditions into ``_coupling_bcs`` and it is
        # only re-attached when the SystemModel is built via
        # ``SystemModel.from_model(model)`` (+ ``UFLRuntimeModel(model)``), so we
        # restore ``self.boundary_conditions`` after the base init or the
        # wall/inflow BCs are silently lost.
        super().__init__(
            dimension=2,
            aux_variables=["hinv"],
            parameters={
                "g":   (float(g),  "positive"),
                "n":   (float(n),  "nonnegative"),
                "nu":  (float(nu), "nonnegative"),
                "wet_dry_eps": (float(eps), "positive"),
            },
            eigenvalue_mode="symbolic",
            **kw,
        )
        if self._coupling_bcs is not None:
            self.boundary_conditions = self._coupling_bcs

    # The Firedrake path does not use the declarative function groups
    # (interpolate / project / boundary:<name>); BCs reach the runtime
    # model through the ``boundary_conditions`` param.  SWE's groups
    # reference inflow params (h_in / q_in) this model does not carry, so
    # build none here.  ``reconstruction_variables`` is a METHOD (inherited
    # from SWE) and is unaffected.
    def _build_function_groups(self):
        """No function groups: the Firedrake path takes boundary conditions through
        `boundary_conditions`.
        """
        return {}

    # -- convenience -----------------------------------------------------
    @property
    def _parameter_symbols(self):
        """The parameter symbol vector."""
        return self.parameters

    def _primitives(self):
        """Return the symbols `(b, h, hu, hv, hinv)`."""
        v = self.variables
        a = self.aux_variables
        return v.b, v.h, v.hu, v.hv, a.hinv

    # -- operators -------------------------------------------------------
    def flux(self):
        """Convective momentum flux `hu (x) u` via the bounded `hinv`, without pressure."""
        _, h, hu, hv, hinv = self._primitives()
        F = sp.Matrix.zeros(4, 2)
        # mass equation
        F[1, 0] = hu
        F[1, 1] = hv
        # momentum: pure convective (no pressure here — see NCP / pressure)
        F[2, 0] = hu * hu * hinv
        F[2, 1] = hu * hv * hinv
        F[3, 0] = hu * hv * hinv
        F[3, 1] = hv * hv * hinv
        return ZArray(F)

    def hydrostatic_pressure(self):
        """Hydrostatic pressure `g h^2/2` on the momentum diagonal, added by the positive
        nonconservative Riemann solvers.
        """
        # ½ g h² on the momentum diagonal, as a SEPARATE operator (not
        # folded into ``flux`` or the NCP).  This is the encoding the
        # ``PositiveNonconservative{HLL,Rusanov}`` solvers are built
        # around: the fan sees the pressure jump via
        # ``_physical_flux_n = F·n + P·n`` and the Audusse well-balancing
        # corrections at hydrostatically reconstructed faces are exactly
        # ``(P_raw − P_star)·n``.
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        P = ZArray.zeros(4, 2)
        P[2, 0] = g * h ** 2 / 2
        P[3, 1] = g * h ** 2 / 2
        return P

    def nonconservative_matrix(self):
        """Bed-slope product `g h d_d b` only; the pressure must not be folded in here."""
        # Bed-slope coupling ONLY (``g h ∂_d b``).  The pressure term must
        # NOT be encoded here as ``g h ∂_d h``: the NCP path-integral in
        # the WB Riemann solvers is evaluated over hydrostatically
        # reconstructed face states (where ``Δb* = 0`` and ``Δh*`` is
        # shoreline-clipped), so a pressure-in-NCP encoding loses the
        # bathymetry reaction force at clipped wet/dry faces.
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 2)
        N[2, 0, 0] = g * h
        N[3, 0, 1] = g * h
        return N

    def source(self):
        """Bed-friction momentum sink `rate * u_i`, with `rate` summed over the `bottom`
        closures; zero at rest.
        """
        _, h, hu, hv, hinv = self._primitives()
        u, w = hu * hinv, hv * hinv
        st = swe_closure_state(self)
        rate = sum((c.expression(st) for c in self.closures
                    if c.closes == "bottom"), sp.S.Zero)
        return ZArray([sp.S.Zero, sp.S.Zero, rate * u, rate * w])

    def diffusion_matrix_explicit(self):
        """Explicit eddy-viscosity tensor of `div(D h grad u)` (`D` summed over the
        `horizontal` closures), diagonal on the momentum rows plus a cross term on `h`.
        """
        _, h, hu, hv, hinv = self._primitives()
        u, w = hu * hinv, hv * hinv
        st = swe_closure_state(self)
        D = sum((c.expression(st) for c in self.closures
                 if c.closes == "horizontal"), sp.S.Zero)
        A = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        for d in (0, 1):
            A[2, 2, d, d] = D
            A[3, 3, d, d] = D
            A[2, 1, d, d] = -D * u
            A[3, 1, d, d] = -D * w
        return ZArray(A)

    # -- state hygiene (SystemModel update_variables / aux slots) --------
    def update_variables(self):
        """Cap `hu` and `hv` to `max(h - wet_dry_eps, 0) * U_MAX`; `b` and `h` are
        never modified.
        """
        v = self.variables
        p = self._parameter_symbols
        h, hu, hv = v.h, v.hu, v.hv
        u_max = sp.Float(self.U_MAX)
        # Wet-mask cap: in cells with ``h ≤ eps`` clamp ``hu`` to zero (a
        # nearly-dry bleeding edge must not carry u_max of momentum); ``h``
        # is never modified.  ``Max(h − eps, 0)·u_max`` collapses smoothly
        # to zero at ``h = eps`` using Min/Max only (lowers through every
        # backend; no Heaviside/sign).
        h_wet = sp.Max(h - p.wet_dry_eps, sp.S.Zero)
        max_hu = h_wet * u_max

        def cap(c):
            """Clip `c` to `[-max_hu, max_hu]`."""
            return sp.Max(-max_hu, sp.Min(c, max_hu))

        return ZArray([v.b, h, cap(hu), cap(hv)])

    def update_variables_jacobian_wrt_variables(self):
        """Zero Jacobian of `update_variables` (the cap is not differentiated)."""
        # The Min/Max cap would differentiate to non-serialisable
        # Piecewise/sign; this Jacobian is only used by IMEX implicit-update
        # Newton (which the Firedrake backend does not run on
        # update_variables), so return explicit zeros.
        n = self.n_variables
        return ZArray.zeros(n, n)

    def eigenvalues(self):
        """Eigenvalues `[0, un, un -/+ c]`, the velocity `q / max(1e-14, h)` of the spectrum
        (`wave_speed_inverse_depth`, REQ-82), not the KP `hinv` of the physical operators, with
        an optional wet/dry gate (`ev_gate`).

        The Riemann solvers evaluate this spectrum at the Audusse states `(h*, q h*/h)`, whose
        velocity is the cell's `q/h`; their positivity needs the face speed to bound it.  The KP
        `hinv(h*)` is far below `1/h*` once `h* < wet_dry_eps`: the speed then misses the
        transport velocity of the mass flux `q*` and a dry neighbour is drained negative
        (Malpasset, order 1: h_min -8.9e-3 m).

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        from zoomy_core.systemmodel.operations import wave_speed_inverse_depth
        _, h, hu, hv, _ = self._primitives()
        p = self._parameter_symbols
        n = self.normal
        inv = wave_speed_inverse_depth(h)
        u = hu * inv
        w = hv * inv
        un = u * n.n0 + w * n.n1
        c = sqrt(p.g * sp.Max(h, p.wet_dry_eps))
        raw_ev = [sp.S.Zero, un, un - c, un + c]
        # Optional wet/dry wave-speed gate.  Without it a thin cell's speed is
        # its transport speed (finite: update_variables zeroes q where
        # h <= eps), and the gate can undersize the Rusanov face dissipation at
        # wet/dry faces (breaking the Xing-Zhang cell-mean-positivity
        # decomposition), so it defaults OFF here (ev_gate from the
        # constructor).
        if not self._ev_gate:
            return ZArray(raw_ev)
        cond = sp.Function("conditional")
        gated = [cond(h > p.wet_dry_eps, e, sp.S.Zero) for e in raw_ev]
        return ZArray(gated)

    def update_aux_variables(self):
        """Desingularised inverse depth `hinv = sqrt(2) h / sqrt(h^4 + max(h, eps)^4)`: equals
        `1/h` for `h >= eps` and tends to zero as `h -> 0`.

        References:
            Kurganov & Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
        """
        v = self.variables
        p = self._parameter_symbols
        h = v.h
        h_floor = sp.Max(h, p.wet_dry_eps)
        denom = sqrt(h ** 4 + h_floor ** 4)
        return ZArray([sqrt(2) * h / denom])
