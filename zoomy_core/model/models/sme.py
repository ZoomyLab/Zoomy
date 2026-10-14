"""Shallow Moment Equations (`SME`): the declaratively derived depth-resolved moment Model,
for one or two horizontal directions. `SystemModel.from_model(SME(level=2))` lifts it to the
runtime system.
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, GaussQuadrature, InvertMassMatrix, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate          # abstract ζ-integral
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ       # vertical (pressure) integral
from zoomy_core.model.models.walls import register_free_slip_wall
from zoomy_core.model.models.equations import Stress
from zoomy_core.misc.options import OptionParam
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class SME(BaseModel):
    """Shallow Moment Equations of modal order `level` (`level=2` gives h, q_0, q_1, q_2
    plus the bed `b`), derived symbolically by Galerkin projection.

    Lifted to a SystemModel by `SystemModel.from_model`; stress closures are injected through
    `closures`.

    References:
        Kowalski & Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    _finalize_lazy = True               # declarative path — skip the production tag pipeline
    _cacheable_derivation = True        # derive_model returns m; byproducts ride on m
    level = param.Integer(default=2, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension INCLUDING the vertical: 2 → coords (t, x, z), "
        "one horizontal direction (q_0, q_1, …); 3 → coords (t, x, y, z), two "
        "horizontal directions (q_x_i, q_y_i)."))
    small_slope = param.Boolean(default=True, doc=(
        "True (default) resolves the opaque boundary frame to its n→ẑ limit, "
        "the standard shallow form. False keeps the slope-aware boundary "
        "tractions."))
    quadrature_order = param.Integer(default=0, bounds=(0, None), doc=(
        "Gauss-Legendre order for NUMERICAL integration of Galerkin "
        "integrals that survive the analytic bracket machinery "
        "(non-polynomial material closures, e.g. bingham_navier_slip). "
        "0 (default) = off: an unresolvable integral then raises at "
        "extraction."))
    stress = OptionParam(Stress, default=Stress.shallow, doc=(
        "How much of the bulk closure's stress the model keeps (`moment_scaling`): "
        "`Stress.shallow` the vertical shear only, `Stress.full` the whole Newtonian tensor "
        "(Marche 2007 eq. (3.5))."))
    scaling = param.Parameter(default=None, doc=(
        "A `ShallowScaling` (derivation.nondim): derive the stress truncation by order from "
        "the non-dimensional balance laws (`FreeSurfaceNonDim` -> `ShallowScaling` -> "
        "`InsertClosure` -> `Dimensionalize`) instead of the hand rules (`moment_scaling`, the "
        "hydrostatic inertia drop, `apply_full_tensor`); it then decides `stress`.  None "
        "(default): the hand rules."))
    closures = param.List(default=[], doc=(
        "List of composable stress Closure pieces (closures.py), e.g. "
        "[Newtonian(), NavierSlip(), StressFree()]. An empty list leaves "
        "tau_xz UNCLOSED (modal moments stay free)."))
    symbolic_spectrum = param.Boolean(default=False, doc=(
        "Compute the closed-form beta-HSWME spectrum at BUILD time; OFF by "
        "default, the runtime then takes numerical eigenvalues. The symbolic "
        "eigen-decomposition is expensive."))
    project_nz = param.Integer(default=33, bounds=(2, None), doc=(
        "Number of vertical samples N_z of the trapezoid Galerkin reduction "
        "in ``project_from_3d``; set it to the VOF column height the coupling "
        "supplies."))

    def derive_model(self):
        """Build the declarative SME model (h-equation, projected horizontal momenta, closures)
        and register the vertical reconstruction; called by the base `__init__`.
        """
        Nu = int(self.level)
        # nu (kinematic viscosity) and lambda_s (Navier slip) are MODEL
        # PARAMETERS — default 0 (inviscid / free-slip); override values via
        # ``SME(level, parameters={"lambda_s": 0.5, ...})``.
        # e_x: downslope gravity component (K&T eq 4.7 "hg(e_x - ...)") -
        # the INCLINE body force; with e_x = sin(theta) and a FLAT bed the
        # model is an exact infinite incline (periodic-domain friendly)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0,
                  "e_x": 0.0}
        # the base __init__ has already split the user's parameters= dict
        # into the Zstruct ``self.parameter_values`` — merge those numeric
        # overrides over the defaults.
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, Momentum, moment_scaling, small_slope_scaling,
            slope_aware_scaling, package_viscous)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures, apply_full_tensor

        # dimension setup: dim=2 → (t,x,z), ONE horizontal direction (the
        # canonical K&T SME, q_0,q_1,…); dim=3 → (t,x,y,z), TWO horizontals
        # (q_x_i, q_y_i).  Every step below loops over the horizontal directions
        # / velocity fields, so dim=2 reduces to exactly the 1-horizontal form.
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HAXES = ("x",) if dim == 2 else ("x", "y")
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}
        # conserved-moment family per direction: 1 horizontal keeps the bare
        # `q` (byte-identical to K&T); 2 horizontals use `q_x` / `q_y`.
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # Free-surface hydrostatic pressure flux (g·h²/2); recomputed in
        # ``system_model`` from ``m`` and tagged there so the extractor routes
        # it to hydrostatic_pressure (no longer stashed on ``self``).

        # 1 — full system, assembled from balance blueprints (equations.py).
        # Mass registers the horizontal velocity(ies) + w; Momentum (full stress
        # tensor) registers p (state) + the stresses (closure) with the
        # incline body force −g·e_x; moment_scaling applies the stress setting:
        # Stress.shallow (default, byte-identical to K&T) drops the in-plane stresses and the
        # vertical-row stress; Stress.full keeps the whole tensor, which the bulk closure
        # (Newtonian) fills in physical coordinates (apply_full_tensor) and the projection
        # routes into conservative diffusion (package_viscous, §9b).
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        mom = Momentum(m); m.add_equation(mom)
        if self.scaling is not None:
            # the truncation by order of the non-dimensional balance laws (scaling.py)
            from zoomy_core.model.models.scaling import ordered_truncation, Pressure
            rec = ordered_truncation(m, mom, self.closures or [], self.scaling,
                                     heights=[h, b], free_surface=b + h,
                                     pressure=Pressure.total, mass="mass",
                                     requested=self.stress)
            tensor = rec.tensor if rec.vertical_stress else None
        else:
            moment_scaling(m, mom, self.stress)
            tensor = apply_full_tensor(self.closures or [], m, mom)
        full_stress = m.stress == Stress.full
        uvel, w, p = mom.uvel, mom.w, mom.p
        # hook: a turbulence subclass (KESME) declares its extra depth-averaged
        # state (k, ε) HERE so the bulk stress closure can read them in §5.
        self._declare_turbulence_fields(m, t, horiz)

        def _kbc(interface):
            """Kinematic boundary condition of the vertical velocity at `interface`."""
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — hydrostatic: drop the inertial vertical-momentum terms, integrate
        # for p, broadcast the solved pressure into EVERY horizontal component,
        # then drop the vertical row.
        mz = m.momentum.z
        zdrop = {d.t(w): 0, d.z(w * w): 0}
        for i, xd in enumerate(horiz):
            zdrop[DERIV[xd](uvel[i] * w)] = 0
        # with the full stress the vertical balance keeps d_e tau_ez + d_z tau_zz (Marche 2007,
        # (3.5)); the free surface carries the normal stress-free condition
        # p(eta) = tau_zz(eta), so p = rho g (eta - z) - rho nu (div u(z) + div u(eta)), (3.11)
        tzz = None
        if tensor is not None:
            tzz = tensor["zz"]
        if self.scaling is None:
            mz.apply(zdrop)
        elif mz.expr.has(*zdrop):
            raise ValueError(f"SME is hydrostatic: {self.scaling!r} keeps the vertical inertia "
                             f"({mz.expr}); use a vertical order that drops it")
        mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
        # Free-surface pressure BC.  Default p(η)=0 (atmospheric).  A SURFACE
        # closure may instead prescribe a NORMAL traction ``{"normal": n·σn}``
        # (e.g. surface tension ``−σκ``); inject it here as p(η) so it reaches
        # the horizontal momentum through the eliminated pressure (REQ-219).
        # Gated on the GENERAL path only (small_slope=False).  When nothing is
        # injected we run the ORIGINAL ``{...: 0}`` substitution VERBATIM (literal
        # 0, not sp.S.Zero) so every existing model's derivation is byte-for-byte
        # unchanged — the whole-expression identity matters downstream.
        p_surf = None
        surface_closures = [c for c in (self.closures or [])
                            if getattr(c, "closes", None) == "surface"]
        if surface_closures and not bool(self.small_slope):
            # at=None (BULK fields), NOT at=1 (surface trace): the free-surface
            # normal traction (surface tension −σκ) is a GEOMETRIC function of the
            # free surface η=b+h, and b,h are depth-averaged (no vertical
            # dependence) so their surface value IS their bulk value.  This
            # elimination runs BEFORE the σ-map, so there is no ζ yet; a surface
            # trace `.at(1)` would substitute the field's LAST coordinate — which
            # is x here — giving b(t,1)+h(t,1) (constant in x) ⇒ ∂ₓκ=0 ⇒ the
            # surface pressure silently drops out of the momentum (REQ-219 reopen).
            # Bulk fields keep x, so κ(x) survives into the eliminated pressure.
            s_eta = ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                 zeta=None, at=None, boundary_tag="eta",
                                 horiz=list(horiz))
            acc = sp.S.Zero
            for c in surface_closures:
                c.register(m)
                nrm = c.traction(s_eta).get("normal")
                if nrm is not None:
                    acc += nrm
            if acc != 0:
                p_surf = acc
        if tzz is not None:
            p_surf = (p_surf if p_surf is not None else 0) + tzz.subs(z, b + h)
        mz.apply({p.subs(z, b + h): (p_surf if p_surf is not None else 0)})
        psol = mz.solve_for(p)
        for ax in HAXES:
            getattr(m.momentum, ax).apply(psol)
        mz.remove()
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())

        # 3 — σ-map the whole model: z = b + h·ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        # the Galerkin TEST index — "l", which _INDEX_NAMES reserves and never
        # auto-mints, so it can't collide with the trial indices of the SoV
        # families (û, v̂, ŵ); a raw Symbol("k") collided with the 3rd minted
        # trial index in 2-D and silently diagonalised the ŵ bracket.
        k = test_index(); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)        # need φ_{N_u+2} for the top w-mode

        # 4 — moment-project the MASS balance + kinematic BCs (pre-SoV).  The
        # ∂_y v term rides along automatically in 2-D.
        m.mass.apply(Multiply(h)); m.mass.apply(Multiply(c(zeta) * phi_k))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1))); m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0}); m.mass.apply(Simplify())

        # 5 — project EACH horizontal momentum component, then close the stress.
        def _state(at, *, alias=None, btag=None):
            """`ClosureState` for the bulk (`at=None`) or the bed/surface trace
            (`at=0/1`).
            """
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz), base=b)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(Multiply(h)); mxi.apply(Multiply(c(zeta) * phi_k))
            mxi.apply(ProductRule(variables=[zeta]))
            mxi.apply(Integrate(zeta, bounds=(0, 1))); mxi.apply(ResolveIntegral())
            mxi.apply(m.kbc_bot); mxi.apply(m.kbc_top); mxi.apply({sp.Derivative(b, t): 0})
        # boundary closures prescribe the traction in {n,t_α} (the frame solve
        # couples the directions for vector tractions like rough-wall friction);
        # bulk closures stay diagonal (s.u aliased to each axis velocity).
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m.momentum, ax), "tau": tau_h[ax],
                 "velname": HNAME[xd]} for ax, xd in zip(HAXES, horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())
        # Boundary-frame resolution (tracked).  The opaque frame_slope symbols
        # must NEVER survive to the runtime.  small_slope=True applies the
        # small-slope assumption (n→ẑ, frame→0) → the K&T shallow traces
        # (BYTE-IDENTICAL to before).  small_slope=False resolves the frame to
        # the PHYSICAL interface slope ∂_d(interface) so the slope-aware model is
        # fully discretizable (the framework holds in general) — was previously
        # left opaque and unrunnable.
        if bool(self.small_slope):
            small_slope_scaling(m)
        else:
            slope_aware_scaling(m, {"b": b, "eta": b + h})

        # 6 — separation of variables: each u_i → û/v̂ (N_u), w → ŵ (N_u + 1)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        if not has_bulk:
            # UNCLOSED bulk stress: expand each shear stress modally — its
            # moments stay free (K&T pre-closure / boundary-only-closure form).
            for ax, xd in zip(HAXES, horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 7 — basis → h-equation (k=0) and the ŵ closure (k=1…N_u+2).  In 2-D
        # the ŵ closure couples û and v̂; the h-equation is ∂_t h = −Σ_d ∂_d(h û_d^0).
        m.mass.apply(ExpandSums()); m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta)); m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=k, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:
            row.apply(w_closure)

        # 8 — substitute the ŵ closure into each horizontal momentum, resolve.
        # (Re-fetch the component AFTER ResolveModes' in-place promotion.)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(ExpandSums()); mxi.apply(PullConstants())
            mxi.apply(ExtractBrackets(basis, var=zeta)); mxi.apply({N_u: Nu})
            mxi.apply(EvaluateSums()); mxi.apply(w_closure)
            mxi.apply(ResolveModes(index=k, modes=range(Nu + 1)))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(ResolveBasis(legendre, var=zeta))
            if int(self.quadrature_order) > 0:
                getattr(m.momentum, ax).apply(
                    GaussQuadrature(var=zeta, order=int(self.quadrature_order)))

        # 9 — kill loose ∂_t h, consolidate, conservative CoV û_d → q_d/h
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            for kk in range(Nu + 1):
                mxi[kk].apply(h_eq); mxi[kk].apply(Consolidate())
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda q_i: q_i / h))
        m.apply(InvertMassMatrix())

        # 9b — KEEP-ALL: route the retained in-plane deviatoric-stress divergence
        # into conservative diffusion.  The projected ∂_e(2ρν ∂_e u) landed as
        # bare second derivatives C·∂²(q_i/h) (+ topography-coupled ν·q·∂b/∂²b);
        # package_viscous regroups each into ∂(D·∂Q) − ∂(D)·∂Q (diffusive flux A
        # + NCP), which the extractor routes into diffusion_matrix.  b is a STATE
        # in `states`, so the topography couplings evaluate from the same stage
        # (no flat-bed / frozen-b assumption).  No-op unless full_stress.
        if full_stress:
            q_heads = [getattr(m.functions, qn).head for qn in QNAME]
            states = {h, b}
            for qh in q_heads:
                states |= {qh(i, t, *horiz) for i in range(Nu + 1)}
            for ax in HAXES:
                mxi = getattr(m.momentum, ax)
                for kk in range(Nu + 1):
                    mxi[kk].expr = package_viscous(
                        mxi[kk].expr, m.parameters.nu, states, list(horiz))

        # 10 — vertical reconstruction → interpolate (field order [b,h,u,v,w,p];
        # v at index 3 only in 2-D, ŵ_j inlined as their closure)
        q_heads = [getattr(m.functions, qn).head for qn in QNAME]
        cov = {}
        for i, qh in enumerate(q_heads):
            cov.update({coeff_heads[i](j, t, *horiz): qh(j, t, *horiz) / h
                        for j in range(Nu + 1)})
        interp = {0: b, 1: h}
        for vi, qh in enumerate(q_heads):
            interp[2 + vi] = sum((qh(i, t, *horiz) / h) * legendre.eval(i, zeta)
                                 for i in range(Nu + 1))
        interp[4] = sum(sp.expand(w_closure[j].rhs.subs(cov)) * legendre.eval(j, zeta)
                        for j in range(Nu + 2))
        interp[5] = rho * g * h * (1 - zeta)
        m.interpolate_rows = interp

        # 11 — model-derived free-slip wall: each MOMENT is a horizontal
        # VECTOR (one component per direction), so the ghost reflects only the
        # NORMAL part, q_i → q_i − 2 n (n·q_i), and keeps the tangential part.
        # h, b (and any scalar field) extrapolate.  See ``models/walls.py``.
        register_free_slip_wall(
            m, ([qh(i, t, *horiz) for qh in q_heads] for i in range(Nu + 1)))

        # 12 — WB reconstruction: limit η = b+h and the modal velocities q_d/h.
        m.reconstruction_rows = {h: b + h}
        for qh in q_heads:
            m.reconstruction_rows.update(
                {qh(i, t, *horiz): qh(i, t, *horiz) / h for i in range(Nu + 1)})

        # 13 — project (inverse of interpolate): the Integral-FREE, fixed-node
        # Galerkin reduction q_{d,i} = h·α_i of the sampled column.  N_z uniform
        # nodes on [0,1] with trapezoid weights; the per-node samples are the
        # SAME P3_<vel> heads the printer maps for the column, evaluated at the
        # fixed node.  ``Basisfunction.projection_rows`` builds the plain
        # arithmetic rows — the 1/∫φ_i² normalisation is in its denominator, so
        # we supply only the physical h factor (q_i = h·α_i, matching the old
        # (2i+1)·h·∫₀¹ u_d φ_i dζ).  Integral-free ⇒ every printer lowers it.
        N_z = int(self.project_nz)
        # CELL-CENTER nodes (j+1/2)/N_z with midpoint (uniform) weights: the
        # coupling exchange lives on the VOF's cell centres, so baking the
        # projection on the SAME grid makes the preCICE map the IDENTITY and the
        # project∘interpolate round-trip exact (no interpolation between grids).
        # (was endpoint-inclusive j/(N-1) + trapezoidal, which mismatched the
        # exchange grid and broke the round-trip → the SME↔VOF bore-arrival
        # drift; REQ-108, coupling steward's fix/sme-vof-desing-coupling@8a62aac.)
        nodes = [(float(j) + 0.5) / N_z for j in range(N_z)]
        weights = [1.0 / N_z] * N_z
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        m.project_rows = {b: P3["b"], h: P3["h"]}
        for xd, qh in zip(horiz, q_heads):
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            samples = [P3vel(nd) for nd in nodes]
            rows = legendre.projection_rows(nodes, weights, samples,
                                            norm=lambda _k: P3["h"])
            m.project_rows.update({qh(i, t, *horiz): rows[i]
                                   for i in range(Nu + 1)})

        # hook: a turbulence subclass (KESME) adds its transported-scalar
        # balances (k, ε) now, using the conserved moments q_d (= h û_d).
        self._add_turbulence_transport(m, t, horiz, h, q_heads)

        return m

    # ── turbulence hooks (no-op on plain SME; KESME overrides) ──────────────
    def _declare_turbulence_fields(self, m, t, horiz):
        """Hook: declare extra depth-averaged transported state (e.g. k, epsilon) before the
        bulk closure; no-op on `SME`.
        """

    def _add_turbulence_transport(self, m, t, horiz, h, q_heads):
        """Hook: add the transported-scalar balances after the moment system is assembled; no-op
        on `SME`.
        """

    # The runtime operator-form system is built via
    # ``SystemModel.from_model(SME(...))`` (REQ-143); the builder in
    # ``zoomy_core.systemmodel.model_builders`` holds the conservative q-state
    # selection, the hydrostatic-pressure tag, the HSWME spectrum and BC
    # resolution.  Constructor BCs are forwarded there, as before.
    _system_model_kind = "sme"

    def _register_hswme_spectrum(self, sm):
        """Publish the closed-form beta-HSWME spectrum as `sm.eigenvalues`; valid only where
        the truncation drops nothing (level <= 1), otherwise defers to `_wave_speed_cone`.

        References:
            Koellermeier, Rominger 2020, https://doi.org/10.4208/cicp.oa-2019-0065
        """
        import re
        n_eq = sm.n_equations
        normal = list(sm.normal.values())
        A = sp.zeros(n_eq, n_eq)
        for k in range(sm.n_dim):
            for i in range(n_eq):
                for j in range(n_eq):
                    A[i, j] += normal[k] * sm.quasilinear_matrix[i, j, k]
        # Full-stress (Stress.full) models: package_viscous leaves a
        # first-derivative NCP remainder (∝ ν) in the quasilinear matrix, which
        # would pollute the HYPERBOLIC wave speed with the parabolic (diffusion)
        # terms — the CFL bound must be the INVISCID characteristic speed (the
        # diffusion is sub-stepped implicitly by the IMEX solver).  Zero ν so the
        # HSWME spectrum stays the clean SWE/moment-wave form.  No-op when the
        # model carries no ν (inviscid default).
        nu_sym = getattr(sm.parameters, "nu", None)
        if nu_sym is not None:
            A = sp.Matrix(A).subs({nu_sym: 0})
        # β-HSWME truncation: drop the higher-order moment coefficients
        # (moment index ≥ 2) — the regularisation that makes the spectrum
        # computable.  Matches q_2…, q_x_2…, q_y_2… (1-D and 2-D names alike).
        drop = {}
        for s_ in sm.state:
            mtc = re.match(r"q(?:_[xy])?_(\d+)$", str(s_))
            if mtc and int(mtc.group(1)) >= 2:
                drop[s_] = sp.S.Zero
        A = sp.Matrix(A)
        # ── THE TRUNCATION MUST BE A NO-OP ON THE MARCHED MATRIX ──────────────
        # ``A.subs(α_{≥2} → 0)`` returns the spectrum of a DIFFERENT matrix
        # whenever an ``α_{≥2}`` actually appears in ``quasilinear_matrix`` —
        # which it does from level 2 up.  The runtime marches the UNtruncated
        # SWME, so publishing that closed form as ``sm.eigenvalues`` hands the
        # CFL and the HLL Davis estimates a wave speed that is not even an UPPER
        # bound: MEASURED on ``SME(level=2, dimension=3)`` (Malpasset
        # mixing-length closure) against ``numpy.linalg.eigvals`` of the
        # normal-projected matrix over 4000 states (h∈[1e-2,50], |u|≤8,
        # |α₁|,|α₂|≤3|u|, oblique normals) — BELOW ρ(A·n) on 79% of them, down
        # to 0.102× at worst, and 2.4% of the states are not hyperbolic at all
        # (|Im λ|/ρ up to 0.036).  An HLL whose Davis speeds sit inside the true
        # cone is ANTI-dissipative for the waves it missed, and the CFL it hands
        # out is up to 10× too long: measured on the 26k-cell Malpasset march
        # this took α₁/u from 2.5 (level 1) to 41, drove h to −34 m through the
        # MOOD net at t = 6.4 s and NaN'd the field at t ≈ 9.3 s, while the SAME
        # recipe at level 1 (where ``drop`` is empty and the closed form is
        # EXACT — verified to 4 decimals) ran clean to t = 2000 s.
        #
        # This is the same statement the DELETED level-≥6 splice already made —
        # "the gravity cone bounds the TRUNCATED matrix, not the FULL one … so
        # it is not a usable CFL bound", measured there on SME(level=4,
        # dimension=3) as BELOW ρ(A·n) on 192 of 288 wet states, worst 0.031× —
        # it was simply never applied to the levels where the eigensolve DOES
        # return, which is where it bites hardest (2 and 3 are the levels people
        # actually run).
        #
        # So: publish the closed form only where the truncation changes nothing
        # (level ≤ 1 — byte-identical to before).  Otherwise leave
        # ``eigenvalues = None``, which every consumer already reads as "no
        # closed-form spectrum" and routes to ``spectral_radius_bound`` (HLL →
        # local Lax-Friedrichs on a STRICT bound), and publish that same bound
        # as the dt-only :attr:`~zoomy_core.systemmodel.system_model.
        # SystemModel.eigenvalues_cfl` cone so the timestep does not fall back
        # to a per-face numerical eigensolve.
        if any(A.has(s_) for s_ in drop):
            self._wave_speed_cone(sm)
            return
        # ``drop`` is empty here (guard above), so this is the UNtruncated
        # matrix: at level ≤ 1 the SWE/SME(1) char-poly always factors in
        # radicals, and the spectrum is exact — verified numerically against
        # ``numpy.linalg.eigvals`` (ratio 1.0000 on 4000 states).
        try:
            ev = A.eigenvals()
            lams = []
            for root, mult in ev.items():
                lams += [root] * int(mult)
            if len(lams) != n_eq:
                raise ValueError(f"got {len(lams)} eigenvalues, need {n_eq}")
            sm.eigenvalues = sp.Matrix(n_eq, 1, lams)
        except Exception:
            # No radical factorization even at level ≤ 1 (should not happen) —
            # fall back to the bound, never to a spectrum of a different matrix.
            self._wave_speed_cone(sm)

    def _wave_speed_cone(self, sm):
        """Set `eigenvalues = None` and publish a strict spectral-radius-bound cone as
        `eigenvalues_cfl`, so the CFL needs no per-face eigensolve and HLL falls back to local
        Lax-Friedrichs.
        """
        import warnings
        n_eq = sm.n_equations
        sm.eigenvalues = None
        # OPT IN to the bound for the DISSIPATION too (``use_spectral_radius_bound``
        # defaults False, i.e. the exact ``numerical_eigenvalues``, because the
        # bound is extra numerical diffusion — measured 1.4-1.5x depth / 2.7-3.5x
        # velocity error on the Ritter dry dam break).  That default is right for
        # a model that CAN afford a per-face eigensolve.  This one cannot: the
        # JAX backend lowers ``numerical_eigenvalues`` to ``jnp.linalg.eigvals``,
        # which has NO GPU implementation, and the whole reason this branch
        # exists is a 26k-cell GPU march.  So the choice here is not
        # "accuracy vs speed" but "diffusive vs not runnable" — and the extra
        # diffusion is a KNOWN, bounded error, where the wave speed it replaces
        # was simply wrong.  A backend that can afford the eigensolve should set
        # this back to False on the built system.
        sm.use_spectral_radius_bound = True
        rho = sm.spectral_radius_bound
        rho = rho[0] if hasattr(rho, "__len__") else rho
        cone = [-rho, rho] + [sp.S.Zero] * (n_eq - 2)
        sm.eigenvalues_cfl = sp.Matrix(n_eq, 1, cone)
        warnings.warn(
            f"SME(level={self.level}): symbolic_spectrum is not available — the "
            f"β-HSWME closed form is the spectrum of the α≥2-TRUNCATED matrix, "
            f"which is not the one the solver marches from level 2 up (measured "
            f"below ρ(A·n) on 79% of states, worst 0.10×).  Publishing "
            f"eigenvalues=None + the strict spectral_radius_bound cone in "
            f"eigenvalues_cfl instead.")

