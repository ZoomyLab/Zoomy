"""k-epsilon Shallow Moment Equations (KESME): the SME derivation with transported turbulent
kinetic energy ``k`` and dissipation ``epsilon``. They are projected on the same modal basis as
mass and momentum; the model is lifted to a SystemModel via ``SystemModel.from_model``.
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
    KEpsilonViscosity, RoughWall, StressFree)

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class KESME(SME):
    """SME with ``k`` and ``epsilon`` transported at their own modal order ``turbulence_level``,
    independent of the velocity ``level``. Default closures are ``KEpsilonViscosity``,
    ``RoughWall`` and ``StressFree``; build with ``quadrature_order>0`` because
    ``nu_t = C_mu k^2/epsilon`` is rational.

    References:
        Kowalski & Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    _cacheable_derivation = True        # derive_model returns m; byproducts on m

    turbulence_level = param.Integer(default=1, bounds=(0, None), doc=(
        "Modal order N_k of the transported k, ε fields, INDEPENDENT of the "
        "velocity order ``level`` (N_u)."))

    def __init__(self, **params):
        """Build the model with the k-epsilon default closures and ``quadrature_order=6`` unless
        overridden.
        """
        params.setdefault("closures",
                          [KEpsilonViscosity(), RoughWall(), StressFree()])
        params.setdefault("quadrature_order", 6)
        super().__init__(**params)

    def derive_model(self):
        """Derive the model: a copy of ``SME.derive_model`` with the k and epsilon transport
        equations added and projected through the same pipeline. Returns the derivation model
        with interpolate / project / reconstruction rows and the free-slip wall registered.
        """
        Nu = int(self.level)
        Nk = int(self.turbulence_level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0,
                  "C_mu": 0.09, "C_1": 1.44, "C_2": 1.92,
                  "sigma_k": 1.0, "sigma_e": 1.3,
                  # wall-function ε bed BC (weak/penalty): ε(0) → C_μ^{3/4}
                  # k(0)^{3/2}/(κ z_p); kappa & z_p the log-law constant & wall
                  # reference height; eps_wall_penalty the relaxation rate.
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
        # hydrostatic pressure flux, tagged by the SME builder
        # (``model_builders.build_sme``; the extractor routes it to
        # hydrostatic_pressure rather than the conservative flux).  Recomputed
        # there from m — not stashed on self.

        # 1 — full system from blueprints (mass, momentum, moment_scaling)
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        mom = Momentum(m); m.add_equation(mom)
        moment_scaling(m, mom)
        uvel, w, p = mom.uvel, mom.w, mom.p

        # 1b — ADD the k, ε transport equations (the only structural addition).
        #      J_k, J_ε are turbulent fluxes declared like the momentum stress.
        C_mu = m.parameters.C_mu
        C_1, C_2 = m.parameters.C_1, m.parameters.C_2
        sig_k, sig_e = m.parameters.sigma_k, m.parameters.sigma_e
        kf = sp.Function("k", real=True)(*coords)
        ef = sp.Function("varepsilon", real=True)(*coords)
        Jk = sp.Function("J_k", real=True)(*coords)
        Je = sp.Function("J_e", real=True)(*coords)
        m.declare_state(kf, ef)
        m.declare_closure(Jk, Je)
        nu_t = C_mu * kf ** 2 / ef                      # rational in ζ
        Pk = nu_t * d.z(uvel[0]) ** 2                   # shear production (1-horiz)
        adv_k = sum(DERIV[xd](uvel[i] * kf) for i, xd in enumerate(horiz)) + d.z(w * kf)
        adv_e = sum(DERIV[xd](uvel[i] * ef) for i, xd in enumerate(horiz)) + d.z(w * ef)
        m.add_equation("k", d.t(kf) + adv_k - d.z(Jk) - Pk + ef)
        m.add_equation("varepsilon",
                       d.t(ef) + adv_e - d.z(Je) - (ef / kf) * (C_1 * Pk - C_2 * ef))

        def _kbc(interface):
            """Kinematic boundary condition of the vertical velocity at ``interface``."""
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — hydrostatic pressure (momentum.z only; k, ε untouched)
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

        # 3 — σ-map the whole model (k, ε ride along automatically)
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
            """Closure state evaluated at the vertical location ``at`` (optionally for a
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

        # 5b — project the k, ε equations exactly like the momentum, then close
        #      the turbulent fluxes J (bulk ν_t/σ ∂_z·, zero-Neumann at bed+surf)
        for nm in ("k", "varepsilon"):
            eq = getattr(m, nm)
            eq.apply(Multiply(h)); eq.apply(Multiply(c(zeta) * phi_l))
            eq.apply(ProductRule(variables=[zeta]))
            eq.apply(Integrate(zeta, bounds=(0, 1))); eq.apply(ResolveIntegral())
            eq.apply(m.kbc_bot); eq.apply(m.kbc_top); eq.apply({sp.Derivative(b, t): 0})
        Jk_f, Je_f = m.functions.J_k, m.functions.J_e
        dz = lambda e: sp.Derivative(e, zeta) / h          # σ-aware ∂_z
        # zero-Neumann turbulent-flux traces, then the bulk diffusive flux
        m.k.apply({Jk_f.at(0): 0, Jk_f.at(1): 0})
        m.varepsilon.apply({Je_f.at(0): 0, Je_f.at(1): 0})
        m.k.apply({Jk_f.expr: nu_t / sig_k * dz(kf)})
        m.varepsilon.apply({Je_f.expr: nu_t / sig_e * dz(ef)})
        m.k.apply(Simplify()); m.varepsilon.apply(Simplify())

        # 6 — separation of variables: u_i → û (N_u), w → ŵ (N_u+1),
        #     k → k̂ (N_k), ε → ε̂ (N_k)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        khat = sp.Function(r"\hat{k}", real=True)
        ehat = sp.Function(r"\hat{\varepsilon}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        N_k = modal_bound("N_k")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        m.apply(separation_of_variables(kf, khat(t, *horiz), basis, N_k))
        m.apply(separation_of_variables(ef, ehat(t, *horiz), basis, N_k))
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
            mxi.apply({N_u: Nu}); mxi.apply({N_k: Nk})   # eddy-visc stress carries k,ε modes
            mxi.apply(EvaluateSums()); mxi.apply(w_closure)
            mxi.apply(ResolveModes(index=l, modes=range(Nu + 1)))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(ResolveBasis(legendre, var=zeta))
            if qo > 0:
                getattr(m.momentum, ax).apply(GaussQuadrature(var=zeta, order=qo))

        # 8b — k, ε basis resolution (own bound N_k, ŵ closure).  The rational
        # ν_t brackets survive ExtractBrackets and are evaluated by
        # GaussQuadrature, exactly like a non-polynomial momentum closure.
        # NOTE: ResolveModes promotes m.k / m.varepsilon IN PLACE → re-fetch in
        # a SECOND loop before ResolveBasis/GaussQuadrature (else they hit a
        # stale reference and the φ/weight inside the brackets stay unresolved).
        for nm in ("k", "varepsilon"):
            eq = getattr(m, nm)
            eq.apply(ExpandSums()); eq.apply(PullConstants())
            eq.apply(ExtractBrackets(basis, var=zeta))
            eq.apply({N_u: Nu}); eq.apply({N_k: Nk})
            eq.apply(EvaluateSums()); eq.apply(w_closure)
            eq.apply(ResolveModes(index=l, modes=range(Nk + 1)))
        for nm in ("k", "varepsilon"):
            eq = getattr(m, nm)
            eq.apply(ResolveBasis(legendre, var=zeta))
            if qo > 0:
                eq.apply(GaussQuadrature(var=zeta, order=qo))

        # 9 — kill ∂_t h, consolidate, conservative CoV û_d → q_d/h
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            for kk in range(Nu + 1):
                mxi[kk].apply(h_eq); mxi[kk].apply(Consolidate())
        for nm in ("k", "varepsilon"):
            eq = getattr(m, nm)
            for kk in range(Nk + 1):
                eq[kk].apply(h_eq); eq[kk].apply(Consolidate())
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda q_i: q_i / h))
        # conserved turbulence moments K_i = h·k̂_i, E_i = h·ε̂_i (like q = h·û)
        m.apply(ChangeOfVariables(r"\hat{k}", "K", lambda Ki: Ki / h))
        m.apply(ChangeOfVariables(r"\hat{\varepsilon}", "E", lambda Ei: Ei / h))
        m.apply(InvertMassMatrix())

        # 9b — wall-function k AND ε bed BCs (weak/penalty), pinned to the
        # channel friction velocity u_*² = g h e_x (the uniform open-channel bed
        # shear τ_b/ρ).  φ_i(ζ=0)=P_i(−1)=(−1)^i, so the bed traces are
        # k(0)=Σ(K_i/h)(−1)^i, ε(0)=Σ(E_i/h)(−1)^i.  Drive
        #   k(0) → k_wall = u_*²/√C_μ           (kqRWallFunction: SUSTAINS k)
        #   ε(0) → ε_wall = u_*³/(κ z_p)         (epsilonWallFunction: anchors ε)
        # via τ_p·(−1)^l·(trace − wall) on mode l.  Using u_*²(=g h e_x) — NOT
        # the modal k(0)^{3/2} — keeps both wall values positive (no √(k(0)<0))
        # and makes the turbulence self-sustaining from the slope forcing.
        K_head = m.functions.K.head
        E_head = m.functions.E.head
        kap, z_p = m.parameters.kappa, m.parameters.z_p
        tau_p = m.parameters.eps_wall_penalty
        ustar2 = g * h * m.parameters.e_x
        k_wall = ustar2 / sp.sqrt(C_mu)
        eps_wall = ustar2 ** sp.Rational(3, 2) / (kap * z_p)
        k0 = sum((K_head(i, t, *horiz) / h) * legendre.at0(i) for i in range(Nk + 1))
        e0 = sum((E_head(i, t, *horiz) / h) * legendre.at0(i) for i in range(Nk + 1))
        for l in range(Nk + 1):
            m.k[l].expr = (sp.sympify(m.k[l].expr)
                           + tau_p * legendre.at0(l) * (k0 - k_wall))
            m.varepsilon[l].expr = (sp.sympify(m.varepsilon[l].expr)
                                    + tau_p * legendre.at0(l) * (e0 - eps_wall))

        # 10 — vertical reconstruction → interpolate
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

    # Built via ``SystemModel.from_model(KESME(...))`` (REQ-143): the SME build,
    # then the k, ε moments flagged positivity-constrained so
    # ``NumericalSystemModel(regularization.positivity_floor>0)`` floors their
    # singular source dependence (ν_t=C_μk²/ε, wall √k).  See
    # ``zoomy_core.systemmodel.model_builders.build_kesme``.
    _system_model_kind = "kesme"

    def _register_hswme_spectrum(self, sm):
        """Set the SystemModel spectrum from the plain ``SME`` twin and pad the k, epsilon slots
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
        lams += [u_m] * (n_eq - len(lams))         # k, ε slots: advective, in-cone
        sm.eigenvalues = sp.Matrix(n_eq, 1, lams)


__all__ = ["KESME"]
