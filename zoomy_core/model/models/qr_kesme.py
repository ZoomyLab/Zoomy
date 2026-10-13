"""k-epsilon Shallow Moment Equations in the q-r variables `sk = sqrt(k)`,
`se = sqrt(epsilon)` (QRKESME), positive by construction. A separate variant of
`KESME`; lifted to a SystemModel via
`SystemModel.from_model`.
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, GaussQuadrature, InvertMassMatrix,
    DeferQuadrature, ResolveNumQuad,
    SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.walls import register_free_slip_wall
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.models.closures import (
    QRViscosity, RoughWall, StressFree)

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class QRKESME(SME):
    """SME with `sk` and `se` transported at their own modal order `turbulence_level`, so
    `k = sk^2 >= 0` and `epsilon = se^2 >= 0` hold identically. Default closures are
    `QRViscosity`, `RoughWall` and `StressFree`; build with `quadrature_order>0` because
    `nu_t = C_mu sk^4/se^2` is rational.

    References:
        Kowalski & Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
        Fe et al. 2009, https://doi.org/10.1002/fld.1880
    """

    _cacheable_derivation = True        # derive_model returns m; byproducts on m

    turbulence_level = param.Integer(default=1, bounds=(0, None), doc=(
        "Modal order N_k of the transported sk=√k, se=√ε fields, INDEPENDENT of "
        "the velocity order ``level`` (N_u)."))

    def __init__(self, **params):
        """Build the model with the q-r default closures and `quadrature_order=6` unless
        overridden.
        """
        params.setdefault("closures",
                          [QRViscosity(), RoughWall(), StressFree()])
        params.setdefault("quadrature_order", 6)
        super().__init__(**params)

    def derive_model(self):
        """Derive the model: a copy of `SME.derive_model` with the `sk` and `se` transport
        equations added and projected through the same pipeline. Returns the derivation model
        with interpolate / project / reconstruction rows and the free-slip wall registered.
        """
        Nu = int(self.level)
        Nk = int(self.turbulence_level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0,
                  "C_mu": 0.09, "C_1": 1.44, "C_2": 1.92,
                  "sigma_k": 1.0, "sigma_e": 1.3,
                  # realizability floors (model-level, BEFORE GaussQuadrature):
                  # ε → ε+eps_min, k → k+k_min in every singular denominator, so
                  # quadrature integrates smooth integrands and the modal source
                  # has no near-zero denominators (the low-turbulence singularity
                  # cannot be guarded post-quadrature — the denominators are then
                  # high-degree polynomials no fixed floor can match).
                  "eps_min": 0.0, "k_min": 0.0,
                  # wall-function bed BC (weak/penalty) on the q-r variables:
                  # sk(0) → √(u_*²/√C_μ),  se(0) → √(u_*³/(κ z_p)); kappa & z_p
                  # the log-law constant & wall reference height, eps_wall_penalty
                  # the relaxation rate.
                  "kappa": 0.41, "z_p": 0.05, "eps_wall_penalty": 50.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, Momentum, moment_scaling, small_slope_scaling,
            slope_aware_scaling)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures

        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HAXES = ("x",) if dim == 2 else ("x", "y")
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # hydrostatic pressure flux (g·h²/2): recomputed by the SME builder
        # (``model_builders.build_sme``) from m, not stashed on self.

        # 1 — full system from blueprints (mass, momentum, moment_scaling)
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        mom = Momentum(m); m.add_equation(mom)
        moment_scaling(m, mom, self.stress)
        uvel, w, p = mom.uvel, mom.w, mom.p

        # 1b — ADD the q-r transport equations (the only structural addition).
        #      The 3-D k-ε system with k=sk², ε=se² substituted and divided by
        #      2 sk / 2 se.  J_k, J_e are the turbulent fluxes declared like the
        #      momentum stress; the diffusion's product-rule cross terms and the
        #      1/(2 sk), 1/(2 se) production/dissipation prefactors are sources.
        C_mu = m.parameters.C_mu
        C_1, C_2 = m.parameters.C_1, m.parameters.C_2
        sig_k, sig_e = m.parameters.sigma_k, m.parameters.sigma_e
        sk = sp.Function("sk", real=True)(*coords)       # sk = √k
        se = sp.Function("se", real=True)(*coords)       # se = √ε
        Jk = sp.Function("J_k", real=True)(*coords)
        Je = sp.Function("J_e", real=True)(*coords)
        m.declare_state(sk, se)
        m.declare_closure(Jk, Je)
        # realizability floors: ε=se²→se²+eps_min, k=sk²→sk²+k_min in EVERY
        # singular denominator (sign-preserving for odd powers, 1/se→se/(se²+ε_min)).
        # Done here, on the smooth 3-D balance, so GaussQuadrature integrates a
        # non-singular integrand — the only place the low-turbulence singularity
        # can be guarded cleanly (post-quadrature the denominators are high-degree).
        e_min, kk_min = m.parameters.eps_min, m.parameters.k_min
        inv_se2 = 1 / (se ** 2 + e_min)                  # 1/se²  (ν_t)
        inv_se = se / (se ** 2 + e_min)                  # 1/se   (sign-preserving)
        inv_sk2 = 1 / (sk ** 2 + kk_min)                 # 1/sk²
        inv_sk = sk / (sk ** 2 + kk_min)                 # 1/sk
        nu_t = C_mu * sk ** 4 * inv_se2                  # = C_μ k²/ε, rational in ζ
        shear = d.z(uvel[0]) ** 2                        # (∂_z u)²  (1-horiz shear)
        adv_k = sum(DERIV[xd](uvel[i] * sk) for i, xd in enumerate(horiz)) + d.z(w * sk)
        adv_e = sum(DERIV[xd](uvel[i] * se) for i, xd in enumerate(horiz)) + d.z(w * se)
        # sk-equation: ∂_t sk + adv − ∂_z J_k − cross_k − prod_k + se²/(2 sk)
        cross_k = nu_t * inv_sk / sig_k * d.z(sk) ** 2
        prod_k = nu_t * inv_sk / 2 * shear
        m.add_equation("sk", d.t(sk) + adv_k - d.z(Jk) - cross_k - prod_k
                       + se ** 2 * inv_sk / 2)
        # se-equation: ∂_t se + adv − ∂_z J_e − cross_e − prod_e + C_2 se³/(2 sk²)
        cross_e = nu_t * inv_se / sig_e * d.z(se) ** 2
        prod_e = C_1 * C_mu * sk ** 2 * inv_se / 2 * shear
        m.add_equation("se", d.t(se) + adv_e - d.z(Je) - cross_e - prod_e
                       + C_2 * se ** 3 * inv_sk2 / 2)

        def _kbc(interface):
            """Kinematic boundary condition of the vertical velocity at `interface`."""
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — hydrostatic pressure (momentum.z only; sk, se untouched)
        mz = m.momentum.z
        zdrop = {d.t(w): 0, d.z(w * w): 0}
        for i, xd in enumerate(horiz):
            zdrop[DERIV[xd](uvel[i] * w)] = 0
        mz.apply(zdrop)
        mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
        mz.apply({p.subs(z, b + h): 0})
        psol = mz.solve_for(p)
        for ax in HAXES:
            getattr(m.momentum, ax).apply(psol)
        mz.remove()
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())

        # 3 — σ-map the whole model (sk, se ride along automatically)
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        l = test_index(); phi_l = basis.phi(l, zeta)
        legendre = Legendre_shifted(level=max(Nu + 2, Nk))

        # 4 — project the MASS balance + KBCs
        m.mass.apply(Multiply(h)); m.mass.apply(Multiply(c(zeta) * phi_l))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1))); m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0}); m.mass.apply(Simplify())

        # 5 — project each horizontal momentum, then close the stress
        def _state(at, *, alias=None, btag=None):
            """Closure state evaluated at the vertical location `at` (optionally for a
            boundary tag).
            """
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz))
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(Multiply(h)); mxi.apply(Multiply(c(zeta) * phi_l))
            mxi.apply(ProductRule(variables=[zeta]))
            mxi.apply(Integrate(zeta, bounds=(0, 1))); mxi.apply(ResolveIntegral())
            mxi.apply(m.kbc_bot); mxi.apply(m.kbc_top); mxi.apply({sp.Derivative(b, t): 0})
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m.momentum, ax), "tau": tau_h[ax],
                 "velname": HNAME[xd]} for ax, xd in zip(HAXES, horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())
        if bool(self.small_slope):
            small_slope_scaling(m)
        else:
            slope_aware_scaling(m, {"b": b, "eta": b + h})

        # 5b — project the sk, se equations exactly like the momentum, then close
        #      the turbulent fluxes J (bulk ν_t/σ ∂_z·, zero-Neumann at bed+surf)
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            eq.apply(Multiply(h)); eq.apply(Multiply(c(zeta) * phi_l))
            eq.apply(ProductRule(variables=[zeta]))
            eq.apply(Integrate(zeta, bounds=(0, 1))); eq.apply(ResolveIntegral())
            eq.apply(m.kbc_bot); eq.apply(m.kbc_top); eq.apply({sp.Derivative(b, t): 0})
        Jk_f, Je_f = m.functions.J_k, m.functions.J_e
        dz = lambda e: sp.Derivative(e, zeta) / h          # σ-aware ∂_z
        # zero-Neumann turbulent-flux traces, then the bulk diffusive flux
        m.sk.apply({Jk_f.at(0): 0, Jk_f.at(1): 0})
        m.se.apply({Je_f.at(0): 0, Je_f.at(1): 0})
        m.sk.apply({Jk_f.expr: nu_t / sig_k * dz(sk)})
        m.se.apply({Je_f.expr: nu_t / sig_e * dz(se)})
        m.sk.apply(Simplify()); m.se.apply(Simplify())

        # 6 — separation of variables: u_i → û (N_u), w → ŵ (N_u+1),
        #     sk → sk̂ (N_k), se → sê (N_k)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        skhat = sp.Function(r"\hat{sk}", real=True)
        sehat = sp.Function(r"\hat{se}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        N_k = modal_bound("N_k")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        m.apply(separation_of_variables(sk, skhat(t, *horiz), basis, N_k))
        m.apply(separation_of_variables(se, sehat(t, *horiz), basis, N_k))
        if not has_bulk:
            for ax, xd in zip(HAXES, horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 7 — mass basis → h-equation (l=0) and the ŵ closure (l=1…N_u+2)
        m.mass.apply(ExpandSums()); m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta)); m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=l, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:
            row.apply(w_closure)

        # 8 — momentum basis resolution (ŵ closure substituted, GaussQuad)
        qo = int(self.quadrature_order)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(ExpandSums()); mxi.apply(PullConstants())
            mxi.apply(ExtractBrackets(basis, var=zeta))
            mxi.apply({N_u: Nu}); mxi.apply({N_k: Nk})   # eddy-visc stress carries sk,se modes
            mxi.apply(EvaluateSums()); mxi.apply(w_closure)
            mxi.apply(ResolveModes(index=l, modes=range(Nu + 1)))
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            if qo > 0:
                # DEFER quadrature BEFORE ResolveBasis: hide the rational closure
                # ∫ as an opaque ⟨…⟩^N atom while φ is STILL OPAQUE, so the basis
                # is substituted only at the Gauss nodes (step 9c) and the
                # rational never exists in concrete-basis symbolic form.  The
                # pure-ζ polynomial brackets are NOT wrapped → ResolveBasis closes
                # them analytically right after.
                mxi.apply(DeferQuadrature(var=zeta))
            mxi.apply(ResolveBasis(legendre, var=zeta))

        # 8b — sk, se basis resolution (own bound N_k, ŵ closure).  The rational
        # ν_t and the q-r cross/source terms survive ExtractBrackets and are
        # evaluated by GaussQuadrature, exactly like a non-polynomial closure.
        # NOTE: ResolveModes promotes m.sk / m.se IN PLACE → re-fetch in a SECOND
        # loop before ResolveBasis/GaussQuadrature (else they hit a stale
        # reference and the φ/weight inside the brackets stay unresolved).
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            eq.apply(ExpandSums()); eq.apply(PullConstants())
            eq.apply(ExtractBrackets(basis, var=zeta))
            eq.apply({N_u: Nu}); eq.apply({N_k: Nk})
            eq.apply(EvaluateSums()); eq.apply(w_closure)
            eq.apply(ResolveModes(index=l, modes=range(Nk + 1)))
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            if qo > 0:
                eq.apply(DeferQuadrature(var=zeta))   # ⟨…⟩^N (see momentum note)
            eq.apply(ResolveBasis(legendre, var=zeta))

        # 9 — kill ∂_t h, consolidate, conservative CoV û_d → q_d/h, and the
        #     turbulence moments SK_i = h·sk̂_i, SE_i = h·sê_i (like q = h·û)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            for kk in range(Nu + 1):
                mxi[kk].apply(h_eq); mxi[kk].apply(Consolidate())
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            for kk in range(Nk + 1):
                eq[kk].apply(h_eq); eq[kk].apply(Consolidate())
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda q_i: q_i / h))
        m.apply(ChangeOfVariables(r"\hat{sk}", "SK", lambda Si: Si / h))
        m.apply(ChangeOfVariables(r"\hat{se}", "SE", lambda Si: Si / h))
        m.apply(InvertMassMatrix())

        # 9b — wall-function bed BCs (weak/penalty) on the q-r variables, pinned
        # to the LIVE rough-wall friction velocity u_*² = C_f·U_p² (REQ-42; see
        # block below — same bed law as the momentum closure, valid for both
        # slope- and non-slope-driven flow).  φ_i(ζ=0)=P_i(−1)=(−1)^i, so the bed
        # traces are sk(0)=Σ(SK_i/h)(−1)^i, se(0)=Σ(SE_i/h)(−1)^i.  Drive
        #   sk(0) → √(u_*²/√C_μ)        (k-wall   k=u_*²/√C_μ, sk=√k)
        #   se(0) → √(u_*³/(κ z_p))     (ε-wall   ε=u_*³/(κ z_p), se=√ε)
        # via τ_p·(−1)^l·(trace − wall) on mode l.  Both wall targets are
        # POSITIVE (u_*²=C_f·U_p²≥0) so no √(negative) — and with sk,se the state,
        # k=sk²≥0 / ε=se²≥0 hold for the whole spin-up (the q-r raison d'être).
        SK_head = m.functions.SK.head
        SE_head = m.functions.SE.head
        kap, z_p = m.parameters.kappa, m.parameters.z_p
        tau_p = m.parameters.eps_wall_penalty
        # ── Live rough-wall friction velocity (REQ-42) ───────────────────────
        # The wall-function friction velocity must be the LIVE bed friction
        # velocity from the bed law (cf. REQ-12), NOT the slope value g*h*e_x.
        # g*h*e_x is ZERO for any non-slope-driven flow (dam break, sloshing,
        # tide, ...), so the wall BC then pins the bed turbulence target to zero
        # and ACTIVELY SUPPRESSES the turbulence the flow should produce — the
        # k-buildup never happens.  Replace with u_*^2 = C_f * U_p^2, the same
        # rough-wall law the momentum bed closure uses:
        #   C_f = (kappa / ln(z_p/z0))^2,  z0 = k_s/30,  U_p = u(zeta_p).
        # Recovers the old behaviour in a uniform slope-driven channel (there
        # C_f*U_p^2 == g h e_x at equilibrium) but now tracks the LOCAL flow.
        _qh_wall = [getattr(m.functions, qn).head for qn in QNAME]
        _zeta_p = z_p / h
        _Cf_wall = (kap / sp.log(z_p / (m.parameters.k_s / 30))) ** 2
        _Up2 = sum(
            (sum((qh(i, t, *horiz) / h) * legendre.eval(i, _zeta_p)
                 for i in range(Nu + 1))) ** 2
            for qh in _qh_wall)
        ustar2 = _Cf_wall * _Up2
        sk_wall = sp.sqrt(ustar2 / sp.sqrt(C_mu))
        se_wall = sp.sqrt(ustar2 ** sp.Rational(3, 2) / (kap * z_p))
        sk0 = sum((SK_head(i, t, *horiz) / h) * legendre.at0(i) for i in range(Nk + 1))
        se0 = sum((SE_head(i, t, *horiz) / h) * legendre.at0(i) for i in range(Nk + 1))
        for l in range(Nk + 1):
            m.sk[l].expr = (sp.sympify(m.sk[l].expr)
                            + tau_p * legendre.at0(l) * (sk0 - sk_wall))
            m.se[l].expr = (sp.sympify(m.se[l].expr)
                            + tau_p * legendre.at0(l) * (se0 - se_wall))

        # 9c — resolve the deferred ⟨…⟩^N numerical brackets, the VERY LAST step
        # before extraction.  The rational closure integrals rode the whole
        # post-projection pipeline (Consolidate/CoV/InvertMass) as opaque atoms;
        # only now are they expanded — ONCE, numerically — into their
        # Gauss–Legendre node sums.  Their bodies still carry OPAQUE φ (the bracket
        # was formed before ResolveBasis), so ``basis=`` resolves the basis at the
        # nodes — the only place the rational ever touches the concrete polynomials.
        if qo > 0:
            for ax in HAXES:
                getattr(m.momentum, ax).apply(
                    ResolveNumQuad(var=zeta, order=qo, basis=legendre))
            for nm in ("sk", "se"):
                getattr(m, nm).apply(
                    ResolveNumQuad(var=zeta, order=qo, basis=legendre))

        # 10 — vertical reconstruction → interpolate (velocity rows, as SME)
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

        # 12 — WB reconstruction rows
        m.reconstruction_rows = {h: b + h}
        for qh in q_heads:
            m.reconstruction_rows.update(
                {qh(i, t, *horiz): qh(i, t, *horiz) / h for i in range(Nu + 1)})

        # 13 — project rows (inverse of interpolate): Integral-FREE, fixed-node
        # trapezoid Galerkin reduction q_i = h·α_i of the sampled column (see
        # SME.derive_model §13).  N_z uniform nodes; ``projection_rows`` carries
        # the 1/∫φ_i² normalisation, we supply only the physical h.
        N_z = int(self.project_nz)
        nodes = [float(j) / (N_z - 1) for j in range(N_z)]
        weights = [1.0 / (N_z - 1)] * N_z
        weights[0] *= 0.5; weights[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        m.project_rows = {b: P3["b"], h: P3["h"]}
        for xd, qh in zip(horiz, q_heads):
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            samples = [P3vel(nd) for nd in nodes]
            rows = legendre.projection_rows(nodes, weights, samples,
                                            norm=lambda _k: P3["h"])
            m.project_rows.update({qh(i, t, *horiz): rows[i]
                                   for i in range(Nu + 1)})

        return m

    # Built via ``SystemModel.from_model(QRKESME(...))`` (REQ-143): the SME
    # build, then the sk, se moments flagged positivity-constrained so the floor
    # guards the 1/sk, 1/se, 1/sk² source prefactors as se (=√ε) nears zero.
    # See ``zoomy_core.systemmodel.model_builders.build_qrkesme``.
    _system_model_kind = "qrkesme"

    def _register_hswme_spectrum(self, sm):
        """Set the SystemModel spectrum from the plain `SME` twin and pad the sk, se slots
        with the depth-mean normal velocity. The turbulence fields travel inside the gravity
        cone, so they never set the CFL bound.
        """
        n_eq = sm.n_equations
        twin = SystemModel.from_model(SME(level=int(self.level), dimension=int(self.dimension),
                   small_slope=bool(self.small_slope),
                   closures=self.closures,
                   parameters=dict(self.parameter_values.items())))
        if twin.eigenvalues is None:
            sm.eigenvalues = None
            return
        lams = [twin.eigenvalues[i] for i in range(twin.n_equations)]
        normal = list(sm.normal.values())
        by = {str(s_): s_ for s_ in sm.state}
        names = ["q_0"] if int(self.dimension) == 2 else ["q_x_0", "q_y_0"]
        u_m = sum(normal[k] * by[nm] for k, nm in enumerate(names)) / by["h"]
        lams += [u_m] * (n_eq - len(lams))         # sk, se slots: advective, in-cone
        sm.eigenvalues = sp.Matrix(n_eq, 1, lams)


__all__ = ["QRKESME"]
