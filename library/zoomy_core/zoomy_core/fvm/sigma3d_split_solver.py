"""Barotropic / baroclinic split solver for the stay-3D σ model — with a
FACE-CONSISTENT vertical velocity ω.

This is the *reference* numerical solver for
`zoomy_core.model.models.sigma3d.Sigma3D`.  It realises the
dimensional split the model is built for:

* a 2-D (here 1-D-x) BAROTROPIC system for `h` driven by the column mass flux,
* a 3-D (x, ζ) BAROCLINIC system for `mom = h·ũ` carried as genuine layers,

coupled by the contravariant vertical velocity `ω`.

Why a dedicated structured solver (not the generic `HyperbolicSolver`)
--------------------------------------------------------------------------
`ω` is *not* a local algebraic function of the cell-centre state — it is a
vertical integral diagnosed from continuity.  A pointwise aux fill that
central-differences the cell-centre column integral (the previous prototype)
de-couples ω from the FV face fluxes: ω ∝ ∂ₓ(state) then amplifies the
grid-scale mode by 1/Δx and `∂_ζ(mom·ω)` feeds it back — a dispersive
instability that *worsens* under refinement (verified non-convergent on
inviscid BBSM13/AHS26).

The fix is to assemble ω from the SAME numerical horizontal mass fluxes the
height and momentum equations use, as a running partial sum over ζ:

    φ_{i+½,k}  = per-layer horizontal mass flux (Rusanov on `mom`,
                 with the SHARED column wave speed a_{i+½})
    M_{i+½}    = Σ_k φ_{i+½,k} Δζ   (== the Rusanov height flux — telescopes)
    G_{i,k+½}  = G_{i,k-½} − Δζ·( ∂_t h_i + (φ_{i+½,k} − φ_{i-½,k})/Δx )   ,  G_{i,-½}=0
    ω_{i,k+½}  = G_{i,k+½} / h_i

Two exact discrete properties this gives, which the central difference did not:

1.  **ω(0)=ω(1)=0 to machine precision** — because Σ_k φ Δζ = M *by
    construction*, the top partial sum telescopes to 0 on every grid.
2.  **discretely divergence-free ω** (DGCL / freestream preservation) — a
    uniform `mom` is preserved by the vertical advection, and the ω↔mom loop
    inherits the FV upwind dissipation instead of central anti-diffusion.

Result: `|ω|max` stays bounded under refinement and the error decays at
O(Δx) at the standard CFL, for all times (verified on BBSM13/AHS26).

Scope (v1): one horizontal direction (`dimension=2` ⇒ coords (t,x,ζ)),
structured grid, Rusanov horizontal flux, explicit RK1/RK2.  Handles the
inviscid case and the viscous + Navier-slip / stress-free closures.  Horizontal
BCs: `Extrapolation` (transmissive), `Wall` (reflective), `Periodic`.
"""
from __future__ import annotations

import numpy as np


class Sigma3DSplitSolver:
    """Explicit structured (x, ζ) coupled solver with face-consistent ω.

    Parameters
    ----------
    nx, ny : int
        Horizontal cells and vertical (ζ) layers.
    domain : (x0, x1, z0, z1)
        ζ runs over `[z0, z1]` (normally 0..1); only its extent is used.
    cfl : float
        Advective CFL on `max(|u| + √(gh))`.
    rk : {1, 2}
        Forward-Euler (1) or SSP-RK2 / Heun (2).
    """

    def __init__(self, nx, ny, domain=(-5.0, 5.0, 0.0, 1.0), cfl=0.4, rk=2):
        self.nx = int(nx)
        self.ny = int(ny)
        self.domain = tuple(float(v) for v in domain)
        self.cfl = float(cfl)
        self.rk = int(rk)

    # ------------------------------------------------------------------ params
    @staticmethod
    def _params(model):
        """Resolve the physical parameters exactly like `derive_model`."""
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}
        user = getattr(model, "parameter_values", None)
        if user is not None and hasattr(user, "items"):
            values.update({k: float(v) for k, v in user.items()})
        return values

    @staticmethod
    def _closure_kinds(model):
        """Return (has_bulk_viscosity, has_navier_slip, has_stress_free)."""
        clos = getattr(getattr(model, "derivation", None),
                       "closures_resolved", None)
        if not clos:
            clos = model.closures or []
        kinds = {getattr(c, "closes", None) for c in clos}
        # default closure set (empty list) ⇒ Newtonian+NavierSlip+StressFree
        if not clos:
            return True, True, True
        return ("bulk" in kinds), ("bottom" in kinds), ("surface" in kinds)

    def _x_bc_kind(self, model):
        """Map the model's horizontal BCs to a ghost rule. left/right may differ."""
        bcs = getattr(model, "boundary_conditions", None)
        kind = {"left": "extrapolation", "right": "extrapolation"}
        if not bcs:
            return kind
        lst = bcs if isinstance(bcs, list) else getattr(
            bcs, "boundary_conditions_list", [])
        for bc in lst:
            tag = str(getattr(bc, "tag", "")).lower()
            cn = type(bc).__name__.lower()
            if "periodic" in cn:
                kind["left"] = kind["right"] = "periodic"
            elif "wall" in cn or "reflect" in cn:
                if tag in kind:
                    kind[tag] = "wall"
            elif "extrapolation" in cn or "outflow" in cn or "transmissive" in cn:
                if tag in kind:
                    kind[tag] = "extrapolation"
        return kind

    # ------------------------------------------------------------------- solve
    def solve(self, model, ic, t_end, snapshots=None):
        """Run to `t_end`.

        `ic(x, zeta)` → `(b, h, mom)` per (cell-centre x, layer-centre ζ).

        `snapshots` : int | sequence | None
            If an int `n`, record `n` evenly spaced frames over `[0, t_end]`
            (including `t=0`); if a sequence, record at (just after) each listed
            time.  Captured frames land in the result under `"frames"` as a list
            of `{"t", "h", "mom", "u"}` dicts (for animations / time series).

        Returns a dict: `x, zeta, b, h, mom, u, t` (+ `frames` if requested).
        """
        NX, NY = self.nx, self.ny
        x0, x1, z0, z1 = self.domain
        dx = (x1 - x0) / NX
        dz = (z1 - z0) / NY
        xc = x0 + (np.arange(NX) + 0.5) * dx
        zc = z0 + (np.arange(NY) + 0.5) * dz

        p = self._params(model)
        g, nu, lam, e_x = p["g"], p["nu"], p["lambda_s"], p["e_x"]
        has_bulk, has_slip, has_free = self._closure_kinds(model)

        # The bed law and the bulk viscosity come FROM THE MODEL's closures, not
        # from two hardcoded scalars.  Reading only p["nu"] and p["lambda_s"]
        # meant a RoughWall bed (which has no lambda_s) silently became
        # FRICTIONLESS and an ElderViscosity collapsed to a constant: the solver
        # ran, conserved mass, and returned a profile with exactly zero shear.
        from zoomy_core.model.models.closures import (
            NavierSlip, RoughWall, Newtonian, ElderViscosity)
        _clos = getattr(getattr(model, "derivation", None),
                        "closures_resolved", None) or model.closures or []
        bed_c = next((c for c in _clos
                      if getattr(c, "closes", None) == "bottom"), None)
        bulk_c = next((c for c in _clos
                       if getattr(c, "closes", None) == "bulk"), None)
        if not _clos:                       # default stack = Newtonian+NavierSlip
            bed_c, bulk_c = NavierSlip(), Newtonian()

        def _cf():
            """C_f = (kappa/ln(z_p/z_0))^2, z_0 = k_s/30 — RoughWall's own law."""
            kap = float(p.get("kappa", 0.41))
            k_s = float(p.get("k_s", 1e-3))
            z_p = float(p.get("z_p", 0.1))
            return (kap / np.log(z_p / (k_s / 30.0))) ** 2

        # bed traction / rho as a function of the bed velocity trace u0
        if isinstance(bed_c, RoughWall):
            _Cf = _cf()
            def bed_flux(u0):                     # tau_b/rho = C_f u0|u0|
                return _Cf * u0 * np.abs(u0)
        elif isinstance(bed_c, NavierSlip):
            def bed_flux(u0, _l=lam):             # tau_b/rho = lambda_s u0
                return _l * u0
        else:
            def bed_flux(u0):
                return np.zeros_like(u0)

        # effective viscosity at the interior zeta-faces
        if isinstance(bulk_c, ElderViscosity):
            _kap = float(p.get("kappa", 0.41))
            _Cf_e = _cf() if isinstance(bed_c, RoughWall) else None
            def nu_of_zeta(zf, hcol, u0):
                """nu + nu_t,  nu_t = kappa u_* h zeta(1-zeta).

                u_* is the SAME sqrt(C_f)|u_b| the bed law recovers when Elder
                was built with friction=<that bed>, so the eddy viscosity tracks
                the computed bed shear per column instead of a frozen constant.
                """
                us = (np.sqrt(_Cf_e) * np.abs(u0) if _Cf_e is not None
                      else float(p.get("u_star", 0.0)) * np.ones_like(u0))
                return nu + _kap * us[:, None] * hcol[:, None] * zf * (1.0 - zf)
        elif isinstance(bulk_c, Newtonian) or has_bulk:
            def nu_of_zeta(zf, hcol, u0):
                return np.full((hcol.size, zf.size), nu)
        else:
            def nu_of_zeta(zf, hcol, u0):
                return np.zeros((hcol.size, zf.size))

        nu_eff = nu if has_bulk else 0.0
        lam_eff = lam if has_slip else 0.0
        xbc = self._x_bc_kind(model)

        # initial condition
        b = np.empty(NX)
        h = np.empty(NX)
        mom = np.empty((NX, NY))
        for i in range(NX):
            for k in range(NY):
                bi, hi, mi = ic(xc[i], zc[k])
                mom[i, k] = mi
                if k == 0:
                    b[i] = bi
                    h[i] = hi

        # central bed slope (one-sided at ends)
        dbdx = np.zeros(NX)
        dbdx[1:-1] = (b[2:] - b[:-2]) / (2 * dx)
        dbdx[0] = (b[1] - b[0]) / dx
        dbdx[-1] = (b[-1] - b[-2]) / dx

        self.omega_max = 0.0

        def _faces(h, mom):
            """Column states at the NX+1 x-faces, honouring the x-BC ghost rule."""
            u = mom / h[:, None]
            hL = np.empty(NX + 1); hR = np.empty(NX + 1)
            mL = np.empty((NX + 1, NY)); mR = np.empty((NX + 1, NY))
            # interior faces 1..NX-1 share cell i-1 (L) / i (R)
            hL[1:] = h; hR[:-1] = h
            mL[1:] = mom; mR[:-1] = mom
            lk, rk_ = xbc["left"], xbc["right"]
            if lk == "periodic" or rk_ == "periodic":
                hL[0] = h[-1]; hR[-1] = h[0]
                mL[0] = mom[-1]; mR[-1] = mom[0]
            else:
                # left boundary face (index 0): ghost = column 0
                hL[0] = h[0]; mL[0] = mom[0] if lk != "wall" else -mom[0]
                # right boundary face (index NX): ghost = column NX-1
                hR[-1] = h[-1]; mR[-1] = mom[-1] if rk_ != "wall" else -mom[-1]
            return hL, hR, mL, mR

        def rhs(h, mom):
            hL, hR, mL, mR = _faces(h, mom)
            uL = mL / hL[:, None]; uR = mR / hR[:, None]
            cL = np.sqrt(g * hL); cR = np.sqrt(g * hR)
            a = np.maximum(np.max(np.abs(uL), 1) + cL,
                           np.max(np.abs(uR), 1) + cR)          # (NX+1,)
            # per-layer horizontal MASS flux (Rusanov, dissipation on shared h)
            phi = 0.5 * (mL + mR) - 0.5 * a[:, None] * (hR - hL)[:, None]
            M = np.sum(phi, 1) * dz                             # = Rusanov height flux
            dh = -(M[1:] - M[:-1]) / dx
            # vertical mass flux G[i,k+½] from discrete per-ζ continuity (telescopes)
            divphi = (phi[1:] - phi[:-1]) / dx
            G = np.cumsum(-dz * (dh[:, None] + divphi), axis=1)
            om = G / h[:, None]                                 # ω at interfaces k+½
            self.omega_max = max(self.omega_max, float(np.max(np.abs(om))))
            # momentum x-flux (advective + hydrostatic pressure), Rusanov
            fxL = mL * uL + 0.5 * g * hL[:, None] ** 2
            fxR = mR * uR + 0.5 * g * hR[:, None] ** 2
            Fx = 0.5 * (fxL + fxR) - 0.5 * a[:, None] * (mR - mL)
            dmom = -(Fx[1:] - Fx[:-1]) / dx
            # momentum ζ-advective flux at interfaces k+½ (upwind by sign ω);
            # 0 at bed (k=-½) and surface (k=NY-½) since ω=0 there.
            Fz = np.zeros((NX, NY + 1))
            w = om[:, :NY - 1]
            up = np.where(w >= 0, mom[:, :NY - 1], mom[:, 1:NY])
            Fz[:, 1:NY] = w * up
            dmom += -(Fz[:, 1:] - Fz[:, :-1]) / dz
            # bed-slope NCP  −g h ∂ₓb  and the gravity source  +e_x g h
            dmom += -g * h[:, None] * dbdx[:, None] + e_x * g * h[:, None]
            # vertical viscous diffusion  ∂_ζ( ν/h² ∂_ζ mom )
            if has_bulk or has_slip:
                Fv = np.zeros((NX, NY + 1))
                u0 = mom[:, 0] / h                       # bed velocity trace
                # interior ζ-faces at ζ = (k+1)·dz, k = 0 … NY-2
                zf = (np.arange(NY - 1) + 1.0) * dz
                nuf = nu_of_zeta(zf, h, u0)              # (NX, NY-1), ζ-DEPENDENT
                Fv[:, 1:NY] = (nuf / h[:, None] ** 2) * (
                    mom[:, 1:NY] - mom[:, :NY - 1]) / dz
                # bed face (k=-½): the model's own bed law, τ_b/ρ
                Fv[:, 0] = bed_flux(u0)
                # surface (k=NY-½): stress-free ⇒ 0  (Fv[:,NY] stays 0)
                dmom += (Fv[:, 1:] - Fv[:, :-1]) / dz
            return dh, dmom

        # snapshot schedule
        if snapshots is None:
            rec_times = []
        elif np.isscalar(snapshots):
            rec_times = list(np.linspace(0.0, t_end, int(snapshots)))
        else:
            rec_times = [float(s) for s in snapshots]
        frames = []
        rec_i = 0

        def _capture(t):
            frames.append({"t": float(t), "h": h.copy(), "mom": mom.copy(),
                           "u": (mom / h[:, None]).copy()})

        t = 0.0
        while rec_i < len(rec_times) and rec_times[rec_i] <= t + 1e-14:
            _capture(t); rec_i += 1
        while t < t_end - 1e-14:
            u = mom / h[:, None]
            amax = float(np.max(np.abs(u) + np.sqrt(g * h)[:, None]))
            dt = self.cfl * dx / max(amax, 1e-12)
            dt = min(dt, t_end - t)
            if self.rk == 1:
                dh, dm = rhs(h, mom)
                h = h + dt * dh; mom = mom + dt * dm
            else:
                dh1, dm1 = rhs(h, mom)
                h1 = h + dt * dh1; m1 = mom + dt * dm1
                dh2, dm2 = rhs(h1, m1)
                h = 0.5 * (h + h1 + dt * dh2)
                mom = 0.5 * (mom + m1 + dt * dm2)
            t += dt
            while rec_i < len(rec_times) and rec_times[rec_i] <= t + 1e-14:
                _capture(t); rec_i += 1

        out = {"x": xc, "zeta": zc, "b": b, "h": h, "mom": mom,
               "u": mom / h[:, None], "t": t}
        if snapshots is not None:
            out["frames"] = frames
        return out
