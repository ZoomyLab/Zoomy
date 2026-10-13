"""Multilayer shallow moment equations (ML-SME): every layer is a full SME moment column, with
interface mass fluxes closed by the layer-fraction constraint. State is
``[b, h, q_1_0 ... q_N_level]``; the model is lifted to a SystemModel via
``SystemModel.from_model``.
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
    ResolveModes, ResolveBasis, InvertMassMatrix, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.closure import GaussQuadrature
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.derivation.system_extract import HydrostaticPressure
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.models.walls import register_free_slip_wall
from zoomy_core.model.models.equations import evaluate_time_derivatives
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class MLSME(BaseModel):
    """Multilayer SME with ``n_layers`` layers that move with the free surface, each a moment
    column of order ``level``. The interface transfer velocity is upwinded or averaged through
    the ``closures`` (``UpwindInterface`` / ``MeanInterface``); the kinematic ``w`` terms stay
    untouched.

    References:
        Kowalski & Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
        Audusse et al. 2011, https://doi.org/10.1051/m2an/2010036
    """

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    n_layers = param.Integer(default=2, bounds=(1, None))
    level = param.Parameter(default=1, doc=(
        "Moment truncation order per layer: an int applies to every layer, a "
        "list ``[N_1, …, N_{n_layers}]`` sets it per layer."))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_ℓ_i); 3 → (t,x,y,z), two horizontal (q_x_ℓ_i, q_y_ℓ_i).  Each layer "
        "is derived dimension-agnostically; the interface mass-flux closure and "
        "the transfer velocity u* generalise per horizontal direction."))
    closures = param.List(default=[], doc=(
        "Composable Closure pieces (closures.py): stress (Newtonian/RoughWall/…) "
        "AND the interface transfer scheme (MeanInterface/UpwindInterface).  "
        "Empty leaves tau UNCLOSED (modal moments stay free)."))
    interface_velocity = param.Selector(
        default="upwind", objects=["upwind", "mean"],
        doc="DEPRECATED - use closures=[UpwindInterface()/MeanInterface()].  "
            "Shared transfer velocity u* at internal interfaces.")
    project_nz = param.Integer(default=33, bounds=(2, None), doc=(
        "FIXED per-layer vertical sample count for the Integral-FREE "
        "``project_from_3d`` Galerkin reduction (trapezoid, O(N_z^-2)); see "
        "SME.project_nz.  Each layer reduces its own sub-column to moments."))

    def derive_model(self):
        """Derive each layer as an SME column, close the interface mass fluxes ``G`` from the
        fraction constraint, and assemble the model. Returns the derivation model with
        continuity, one momentum equation per layer / direction / mode, and the interpolate,
        project and wall rows.
        """
        N = int(self.n_layers)
        # per-layer moment order: int → uniform (byte-identical to before);
        # list → one degree per layer (REQ-160 stress-free plug top).
        if isinstance(self.level, (list, tuple)):
            levels = [int(v) for v in self.level]
            if len(levels) != N:
                raise ValueError(
                    f"level list has {len(levels)} entries but n_layers={N}")
        else:
            levels = [int(self.level)] * N
        qorder = int(self.quadrature_order)
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        # per-direction conserved-moment / SoV family names for a layer.  1
        # horizontal keeps q_ℓ / û_ℓ (byte-identical); 2 horizontal → q_x_ℓ,
        # q_y_ℓ / û_ℓ, v̂_ℓ.
        def qname(xd, ell):
            """Name of the discharge of layer ``ell`` in direction ``xd``."""
            return f"q_{ell}" if dim == 2 else f"q_{CN[xd]}_{ell}"
        def shat(xd, ell):
            """Name of the velocity coefficient of layer ``ell`` in direction ``xd``."""
            return (rf"\hat{{u}}_{ell}" if dim == 2
                    else rf"\hat{{{HNAME[xd]}}}_{ell}")
        def sname(xd, ell):
            """Name of the shear-stress field of layer ``ell`` in direction ``xd``."""
            return f"tau_{ell}" if dim == 2 else f"tau_{CN[xd]}z_{ell}"
        # e_x (downslope gravity component) is minted on the per-layer sub-model
        # by gravity_components (mirrors SME/VAM); declare it on the assembled
        # model too so it binds with a value at codegen.
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}
        for j in range(1, N):
            values[f"l_{j}"] = 1.0 / N
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})

        b = sp.Function("b", real=True)(t, *horiz)
        hl = [sp.Function(f"h_{ell}", positive=True)(t, *horiz)
              for ell in range(1, N + 1)]
        H = sum(hl)
        ifaces = [b]
        for ell in range(N):
            ifaces.append(ifaces[-1] + hl[ell])
        Gf = [sp.S.Zero] + [sp.Function(f"G_{ell}", real=True)(t, *horiz)
                            for ell in range(1, N)] + [sp.S.Zero]
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        from zoomy_core.model.models.equations import (
            Mass, small_slope_scaling, add_inplane_viscous, package_viscous,
            gravity_components)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_layer_stress_closures

        # KEEP-ALL: a NewtonianInPlane closure RETAINS the per-layer in-plane
        # deviatoric stress τ_de = ρν(∂_d u_e + ∂_e u_d) (the streamwise normal
        # stress ml_fullsme.py keeps); routed to conservative diffusion by
        # package_viscous.  No-op by default → Gate-1 byte-identical.
        retain_inplane = any(getattr(c, "closes", None) == "in_plane"
                             for c in (self.closures or []))

        def derive_layer(ell):
            """Return the continuity and the per-direction modal momentum rows of layer ``ell``
            as ``(cont, {direction: [row per mode]})``.
            """
            Nu = levels[ell - 1]            # per-layer moment order
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=coords, parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            # mass balance from the (dimension-agnostic) blueprint — not a
            # hand-written ∂_x u + ∂_z w; it gives Σ_d ∂_d u_ℓ_d + ∂_z w_ℓ.
            ml.add_equation(Mass(ml, suffix=f"_{ell}"))
            uvel = [sp.Function(HNAME[xd] + f"_{ell}", real=True)(*coords)
                    for xd in horiz]
            w = sp.Function(f"w_{ell}", real=True)(*coords)
            tau = {xd: sp.Function(sname(xd, ell), real=True)(*coords)
                   for xd in horiz}
            MOM = [f"momentum_{CN[xd]}" for xd in horiz]
            # shared gravity vector: g·∂_d(b+H) is the hydrostatic surface
            # gradient (= −g_z·∂_d η); −g_h[xd] is the downslope incline body
            # force g·e_x on the streamwise row (0 on cross/vertical), the SAME
            # single gravity path SME/VAM read (mints e_x on ml).
            g_h, _ = gravity_components(ml, coords)
            for i, xd in enumerate(horiz):
                adv = sum(DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
                ml.add_equation(
                    f"momentum_{CN[xd]}",
                    d.t(uvel[i]) + adv + d.z(uvel[i] * w)
                    + gl * DERIV[xd](b + H) - d.z(tau[xd]) / rl - g_h[xd])
            if retain_inplane:                       # add in-plane divergence pre-σ-map
                # the hydrostatic g d(b + H) is then -sigma_zz = p - tau_zz (continuous
                # across the interfaces)
                add_inplane_viscous(ml, [getattr(ml, mn) for mn in MOM],
                                    uvel, list(horiz), nu_s, normal_stress=True)

            def _kbc(iface, G):
                """Kinematic boundary condition of layer ``ell`` at ``iface`` with interface
                mass flux ``G``.
                """
                kw = dict(w=w, u=uvel[0], interface=iface, rho=rl,
                          mass_flux=(G if G != 0 else None))
                if dim == 3:
                    kw["v"] = uvel[1]
                return KinematicBC(**kw)
            ml.add_equation("kbc_bot", _kbc(z_bot, G_bot))
            ml.add_equation("kbc_top", _kbc(z_top, G_top))
            ml.apply(PDETransformation({z: (zeta, sp.Eq(z, z_bot + h_l * zeta))}))
            basis = Basis(symbol="phi", weight="c"); c = basis.weight
            kk = test_index(); phi_k = basis.phi(kk, zeta)
            legendre = Legendre_shifted(level=Nu + 2)
            for nm in ["mass"] + MOM:
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
            # REQ-160 root cause: the per-layer closure state must read the REAL
            # model parameter namespace — the hand-built ``SimpleNamespace(rho,
            # nu, lambda_s)`` silently lacked every OTHER closure parameter
            # (``tau_y``, ``eps_reg``, …), so any non-polynomial closure died at
            # ``s.par.tau_y``.  ``apply_layer_stress_closures`` calls
            # ``c.register(ml)`` first, so ``ml.parameters`` already carries every
            # closure's params by the time a state is built — exactly as SME does
            # (``params=m.parameters``).  Register the closures up-front so the
            # namespace is populated even for a bottom/surface-only closure set.
            for c in (self.closures or []):
                c.register(ml)

            def _state(at, *, alias=None, btag=None):
                """Closure state of layer ``ell`` evaluated at the vertical location ``at``."""
                fields = {"u": getattr(ml.functions, f"u_{ell}"),
                          "w": getattr(ml.functions, f"w_{ell}")}
                if dim == 3:
                    fields["v"] = getattr(ml.functions, f"v_{ell}")
                return ClosureState(fields, params=ml.parameters, h=h_l, x=C.x,
                                    zeta=zeta, at=at, alias=alias,
                                    boundary_tag=btag, horiz=list(horiz), base=z_bot)
            axes = [{"mx": getattr(ml, f"momentum_{CN[xd]}"),
                     "tau": getattr(ml.functions, sname(xd, ell)),
                     "velname": HNAME[xd]} for xd in horiz]
            has_bulk = apply_layer_stress_closures(
                self.closures, ml, axes, _state,
                is_top=(ell == N), is_bottom=(ell == 1))
            for nm in MOM:
                getattr(ml, nm).apply(Simplify())
            small_slope_scaling(ml)
            coeff_heads = [sp.Function(shat(xd, ell), real=True) for xd in horiz]
            wh = sp.Function(rf"\hat{{w}}_{ell}", real=True)
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            for i, xd in enumerate(horiz):
                ml.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz),
                                                 basis, Nb))
            ml.apply(separation_of_variables(w, wh(t, *horiz), basis, Nb + 1))
            if not has_bulk:
                for xd in horiz:
                    tfld = sp.Function(sname(xd, ell), real=True)(*coords)
                    sig = (rf"\hat{{\sigma}}_{ell}" if dim == 2
                           else rf"\hat{{\sigma}}_{CN[xd]}_{ell}")
                    ml.apply(separation_of_variables(tfld, sp.Function(sig, real=True)(t, *horiz),
                                                     basis, Nb + 1))
            # mass: k=0 h-eq + ŵ closure rows (2-D: couples û_ℓ, v̂_ℓ)
            ml.mass.apply(ExpandSums()); ml.mass.apply(PullConstants())
            ml.mass.apply(ExtractBrackets(basis, var=zeta))
            ml.mass.apply({Nb: Nu}); ml.mass.apply(EvaluateSums())
            ml.mass.apply(ResolveModes(index=kk, modes=range(Nu + 3)))
            ml.mass.apply(ResolveBasis(legendre, var=zeta))
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for row in ml.mass[1:Nu + 3]:
                row.apply(h_eq)
            w_closure = SolveLinearSystem(
                ml.mass[1:Nu + 3],
                [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
            for row in ml.mass[1:Nu + 3]:
                row.apply(w_closure)
            for xd in horiz:
                mxi = getattr(ml, f"momentum_{CN[xd]}")
                mxi.apply(ExpandSums()); mxi.apply(PullConstants())
                mxi.apply(ExtractBrackets(basis, var=zeta))
                mxi.apply({Nb: Nu}); mxi.apply(EvaluateSums())
                mxi.apply(w_closure)
                mxi.apply(ResolveModes(index=kk, modes=range(Nu + 1)))
            for xd in horiz:
                mxi = getattr(ml, f"momentum_{CN[xd]}")
                mxi.apply(ResolveBasis(legendre, var=zeta))
                # REQ-160: numerically resolve any Galerkin integral a
                # non-polynomial closure (Bingham/HB, k-ε, rational ν_t) left
                # behind — mirrors SME(sme.py:261).  Routed into EVERY per-layer
                # reduction from the shared ``quadrature_order``.
                if qorder > 0:
                    mxi.apply(GaussQuadrature(var=zeta, order=qorder))
            for xd in horiz:
                mxi = getattr(ml, f"momentum_{CN[xd]}")
                for kmode in range(Nu + 1):
                    mxi[kmode].apply(h_eq)
                    mxi[kmode].apply(Consolidate())
            for i, xd in enumerate(horiz):
                ml.apply(ChangeOfVariables(shat(xd, ell), qname(xd, ell),
                                           lambda qi: qi / h_l))
            ml.apply(InvertMassMatrix())
            h_eq_q = ml.mass[0].solve_for(d.t(h_l))
            cont = sp.expand(h_eq_q.lhs - h_eq_q.rhs)
            mom = {xd: [sp.expand(getattr(ml, f"momentum_{CN[xd]}")[kmode].expr)
                        for kmode in range(Nu + 1)] for xd in horiz}
            # Mark this layer's hydrostatic self-pressure g·h_ℓ²/2 (folded on the
            # mode-0 momentum flux) as HydrostaticPressure at its FOLD SITE, where
            # h_l is a DISTINCT function per layer (see MLVAM.derive_layer): routes
            # the self-weight to the well-balanced P slot (SME parity at n=1;
            # lake-at-rest for n≥2), leaving inter-layer / bed slope in the NCP.
            hyd = ml.parameters.g * h_l ** 2 / 2
            mom = {xd: [row.subs({hyd: HydrostaticPressure(hyd)}) for row in rows]
                   for xd, rows in mom.items()}
            return cont, mom

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer interface closure (∇·-based, dimension-agnostic) ──
        ht = sp.Function("h", positive=True)(t, *horiz)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, sp.S.One - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = {ell: {xd: [sp.Function(qname(xd, ell), real=True)(k, t, *horiz)
                            for k in range(levels[ell - 1] + 1)] for xd in horiz}
                 for ell in range(1, N + 1)}

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1)).subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        G_sol = {}
        for a in range(1, N):
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1)).subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        # inner (per-layer) basis object — interface traces and the piecewise
        # reconstruction evaluate it at ζ_loc∈{0,1} and at interior nodes, so the
        # bed/surface values φ_k(0)=(−1)^k / φ_k(1)=1 go through the basis object.
        inner_basis = Legendre_shifted(level=max(levels) + 2)

        def _trace(ell, side, xd):
            """Modal velocity of layer ``ell`` in direction ``xd`` at local height ``side`` (0
            bottom, 1 top).
            """
            sgn = lambda i: inner_basis.eval(i, side)
            return (sum(sgn(i) * q_mod[ell][xd][i]
                        for i in range(levels[ell - 1] + 1))
                    / (l_all[ell - 1] * ht))

        from zoomy_core.model.models.closures import interface_closure
        iface = interface_closure(self.closures)

        def _ustar(a, xd):
            """Transfer velocity ``u*`` shared by the layers adjacent to internal interface
            ``a`` in direction ``xd``.
            """
            below, above = _trace(a, 1, xd), _trace(a + 1, 0, xd)
            if iface is not None:
                return iface.expression(below, above, G_sol[Gf[a]])
            if self.interface_velocity == "mean":
                return (below + above) / 2
            return sp.Piecewise((below, G_sol[Gf[a]] >= 0), (above, True))

        # include a vertical z so Model.horizontal = (x[, y]) — a coords=(t,x,y)
        # model would wrongly read y as the vertical and collapse to n_dim=1.
        # (The depth-integrated equations carry no z-derivative.)
        m = DModel(coords=(t, *horiz, z), parameters=values)
        # register the closures on the assembled model too, so their parameters
        # (tau_y, eps_reg, …) are declared and BOUND at codegen even when the
        # user did not pass an explicit value for every one (REQ-160).
        for c in (self.closures or []):
            c.register(m)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        # shared in-plane interface traction sum_e sigma*_de d_e z_a (None without an in-plane
        # closure): sigma* from the two layers' modal traces by the stress InterfaceFlux
        # closure (default: mean); each layer's own part is already in its traction
        from zoomy_core.model.models.closures import interface_inplane_traction
        from zoomy_core.model.models.material import ClosureState, ProfileField

        def _layer_state(ell, side):
            """Closure state of layer ``ell`` over its modal profile, at local height ``side``."""
            h_ell = l_all[ell - 1] * ht
            prof = {HNAME[xd]: ProfileField(
                sum(q_mod[ell][xd][k] / h_ell * inner_basis.eval(k, zeta)
                    for k in range(levels[ell - 1] + 1)), zeta) for xd in horiz}
            return ClosureState(prof, params=m.parameters, h=h_ell, x=x, zeta=zeta, at=side,
                                horiz=list(horiz), base=b + sum(l_all[:ell - 1]) * ht)
        iface_tr = {}
        for a in range(1, N):
            tr = interface_inplane_traction(self.closures, _layer_state(a, 1),
                                            _layer_state(a + 1, 0), G_sol[Gf[a]])
            if tr is not None:
                iface_tr[a] = [sp.expand(e.doit()) for e in tr]
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            for xd in horiz:
                for k in range(levels[ell - 1] + 1):
                    # frac (h_ℓ→l_ℓ·h) then evaluate ONLY the fraction's ∂_t
                    # (∂_t(l_ℓ·h)→l_ℓ·∂_t h for the global-mass sub) — keep the
                    # spatial ∂_x(F) folded so the momentum flux ∂_x(q²/h_ℓ) and
                    # the hydrostatic ∂_x(g h_ℓ²/2) stay conservative (a blanket
                    # .doit() distributed them into the NCP → non-conservative).
                    mom = evaluate_time_derivatives(
                        layer_eqs[ell][1][xd][k].subs(frac), t)
                    # u* swap in the per-direction TRANSFER trace only
                    for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                        if 1 <= a <= N - 1:
                            phik = inner_basis.eval(k, side)
                            # interface transfer is a projected coefficient
                            # ⟨δ_interface·u*, φ_k⟩ = φ_k(side)·(…); the per-layer
                            # row was already mass-inverted (:262), so this term
                            # carries the SAME 1/μ_k Gram-norm the projection
                            # divides by — read off the basis, not hardcoded
                            # (REQ-79).  Without it modes k≥1 are off by (2k+1);
                            # μ_0=1 so mode 0 is unchanged (cid 170).
                            mom = mom + (sgn * phik
                                         * (_ustar(a, xd) - _trace(ell, side, xd))
                                         * Gf[a] / rho_s / inner_basis.gram(k, k))
                            if a in iface_tr:          # shared in-plane traction
                                mom = mom + (sgn * phik * iface_tr[a][horiz.index(xd)]
                                             / rho_s / inner_basis.gram(k, k))
                    mom = sp.expand(mom).subs(G_sol)
                    enm = (f"momentum_{ell}_{k}" if dim == 2
                           else f"momentum_{CN[xd]}_{ell}_{k}")
                    m.add_equation(enm, sp.expand(
                        evaluate_time_derivatives(mom.subs(par), t)))

        # KEEP-ALL: route the retained per-layer in-plane deviatoric-stress
        # divergence into conservative diffusion (mirrors ml_fullsme.py); b is a
        # STATE in `states` so topography couplings evaluate from the same stage.
        if retain_inplane:
            states = {ht, b}
            for ell in range(1, N + 1):
                for xd in horiz:
                    states |= {q_mod[ell][xd][k] for k in range(levels[ell - 1] + 1)}
            for ell in range(1, N + 1):
                for xd in horiz:
                    for k in range(levels[ell - 1] + 1):
                        enm = (f"momentum_{ell}_{k}" if dim == 2
                               else f"momentum_{CN[xd]}_{ell}_{k}")
                        getattr(m, enm).expr = package_viscous(
                            getattr(m, enm).expr, m.parameters.nu, states, list(horiz))

        # ── basic operators: interpolate_to_3d + project_from_3d, PIECEWISE over
        # the moving layers, so ML-SME exposes the same canonical operators as
        # SME (postprocessing / coupling consume model.interpolate_to_3d).
        # Layer ℓ spans global ζ ∈ [c_{ℓ-1}, c_ℓ] (c_ℓ = Σ_{j≤ℓ} l_j); within it
        #   u_ℓ(ζ) = Σ_k (q_ℓ_k / h_ℓ)·P_k(2 ζ_loc − 1),  ζ_loc = (ζ−c_{ℓ-1})/l_ℓ,
        # h_ℓ = l_ℓ·ht.  Field order [b, h, u(, v), w, p] (SME convention).
        cum = [sp.S.Zero]
        for lf in l_all:
            cum.append(cum[-1] + lf)

        def _u_piecewise(xd):
            """Velocity profile in direction ``xd`` over global ``zeta``, piecewise over the
            layers.
            """
            pieces = []
            for ell in range(1, N + 1):
                lf, c0 = l_all[ell - 1], cum[ell - 1]
                zloc = (zeta - c0) / lf
                u_l = sum((q_mod[ell][xd][k] / (lf * ht)) * inner_basis.eval(k, zloc)
                          for k in range(levels[ell - 1] + 1))
                pieces.append((u_l, (zeta <= cum[ell]) if ell < N else True))
            return sp.Piecewise(*pieces)

        interp = {0: b, 1: ht}
        interp[2] = _u_piecewise(x).subs(par)
        if dim == 3:
            interp[3] = _u_piecewise(y).subs(par)
        interp[4] = sp.S.Zero            # TODO: per-layer kinematic w reconstruction
        interp[5] = m.parameters.rho * m.parameters.g * ht * (1 - zeta)
        m.interpolate_rows = interp

        # inverse: q_ℓ_k = h·α_ℓ_k — the Integral-FREE, fixed-node trapezoid
        # Galerkin reduction of each layer's sub-column.  Layer ℓ spans global
        # ζ ∈ [c0, c1]; sample P3_<vel> at N_z LOCAL nodes t∈[0,1] mapped to the
        # global position ζ = c0 + l_ℓ·t.  ``projection_rows`` carries the
        # 1/∫φ_k² normalisation; the physical factor is h·l_ℓ (the dζ = l_ℓ·dt
        # Jacobian times the layer height h·l_ℓ — matching the old
        # (2k+1)·h·∫_{c0}^{c1} … dζ form).
        legendre = Legendre_shifted(level=max(levels) + 2)
        N_z = int(self.project_nz)
        loc = [float(j) / (N_z - 1) for j in range(N_z)]
        weights = [1.0 / (N_z - 1)] * N_z
        weights[0] *= 0.5; weights[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        proj = {b: P3["b"], ht: P3["h"]}
        for xd in horiz:
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            for ell in range(1, N + 1):
                lf = l_all[ell - 1].subs(par)
                c0 = cum[ell - 1].subs(par)
                samples = [P3vel(c0 + lf * t) for t in loc]
                rows = legendre.projection_rows(
                    loc, weights, samples,
                    norm=lambda _k, _lf=lf: P3["h"] * _lf)
                for k in range(levels[ell - 1] + 1):
                    proj[q_mod[ell][xd][k]] = rows[k]
        m.project_rows = proj

        # model-derived free-slip wall (the SAME geometric statement as SME:
        # reflect only the NORMAL component of each layer/mode momentum
        # vector).  ML-SME had NO wall registration at all before — a
        # ``FromModel(definition="wall")`` on an ML-SME simply raised.
        register_free_slip_wall(
            m, ([q_mod[ell][xd][k] for xd in horiz]
                for ell in range(1, N + 1)
                for k in range(levels[ell - 1] + 1)))

        m.bed = b
        m.ht = ht
        # state order: layer-major, then direction, then mode (matches the
        # equation registration order above)
        m.q_flat = [q_mod[ell][xd][k]
                    for ell in range(1, N + 1)
                    for xd in horiz for k in range(levels[ell - 1] + 1)]
        m.G_closed = {str(kk_): sp.simplify(v.subs(par))
                      for kk_, v in G_sol.items()}
        return m

    # Built via ``SystemModel.from_model(MLSME(...))`` (REQ-143); see
    # ``zoomy_core.systemmodel.model_builders.build_mlsme``.
    _system_model_kind = "mlsme"
