"""Multilayer shallow water with mass exchange (ML-SWE), state ``[b, h, q_1 ... q_N]`` with
layer heights fixed as fractions of the total depth. The interface mass fluxes follow from the
layer-fraction constraint; the model is lifted to a SystemModel via ``SystemModel.from_model``.
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, InvertMassMatrix, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class MLSWE(BaseModel):
    """Multilayer SWE with ``n_layers`` layers that move with the free surface, derived layer by
    layer with kinematic interface conditions. The interface transfer velocity is upwinded or
    averaged through the ``closures`` (``UpwindInterface`` / ``MeanInterface``).

    References:
        Audusse et al. 2011, https://doi.org/10.1051/m2an/2010036
    """

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    n_layers = param.Integer(default=2, bounds=(1, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_ℓ_0); 3 → (t,x,y,z), two horizontal (q_x_ℓ_0, q_y_ℓ_0)."))
    closures = param.List(default=[], doc=(
        "Composable Closure pieces (closures.py): stress AND the interface "
        "transfer scheme (MeanInterface/UpwindInterface). Empty leaves tau "
        "UNCLOSED (modal moments stay free)."))
    interface_velocity = param.Selector(
        default="upwind", objects=["upwind", "mean"],
        doc="DEPRECATED - use closures=[UpwindInterface()/MeanInterface()].  "
            "Transfer velocity u* at internal interfaces.")

    def derive_model(self):
        """Derive the per-layer balances, close the interface mass fluxes ``G`` from the
        fraction constraint, and assemble the model. Returns the derivation model with
        continuity and one momentum equation per layer and direction.
        """
        N = int(self.n_layers)
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
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
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
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

        from zoomy_core.model.models.equations import Mass, small_slope_scaling
        from types import SimpleNamespace
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_layer_stress_closures

        def derive_layer(ell):
            """Return the depth-averaged continuity and per-direction momentum of layer ``ell``
            as ``(cont, {direction: momentum})``.
            """
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=coords, parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            ml.add_equation(Mass(ml, suffix=f"_{ell}"))   # dimension-agnostic
            uvel = [sp.Function(HNAME[xd] + f"_{ell}", real=True)(*coords)
                    for xd in horiz]
            wl = sp.Function(f"w_{ell}", real=True)(*coords)
            tau = {xd: sp.Function(sname(xd, ell), real=True)(*coords)
                   for xd in horiz}
            MOM = [f"momentum_{CN[xd]}" for xd in horiz]
            for i, xd in enumerate(horiz):
                adv = sum(DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
                ml.add_equation(
                    f"momentum_{CN[xd]}",
                    d.t(uvel[i]) + adv + d.z(uvel[i] * wl)
                    + gl * DERIV[xd](b + H) - d.z(tau[xd]) / rl)

            def _kbc(iface, G):
                """Kinematic boundary condition of layer ``ell`` at ``iface`` with interface
                mass flux ``G``.
                """
                kw = dict(w=wl, u=uvel[0], interface=iface, rho=rl,
                          mass_flux=(G if G != 0 else None))
                if dim == 3:
                    kw["v"] = uvel[1]
                return KinematicBC(**kw)
            ml.add_equation("kbc_bot", _kbc(z_bot, G_bot))
            ml.add_equation("kbc_top", _kbc(z_top, G_top))
            ml.apply(PDETransformation({z: (zeta, sp.Eq(z, z_bot + h_l * zeta))}))
            basis = Basis(symbol="phi", weight="c"); c = basis.weight
            kk = test_index(); phi_k = basis.phi(kk, zeta)
            legendre = Legendre_shifted(level=1)
            for nm in ["mass"] + MOM:
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
            par_ns = SimpleNamespace(rho=rl, nu=nu_s, lambda_s=lam_s)

            def _state(at, *, alias=None, btag=None):
                """Closure state of layer ``ell`` evaluated at the vertical location ``at``."""
                fields = {"u": getattr(ml.functions, f"u_{ell}"),
                          "w": getattr(ml.functions, f"w_{ell}")}
                if dim == 3:
                    fields["v"] = getattr(ml.functions, f"v_{ell}")
                return ClosureState(fields, params=par_ns, h=h_l, x=C.x,
                                    zeta=zeta, at=at, alias=alias,
                                    boundary_tag=btag, horiz=list(horiz))
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
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            for i, xd in enumerate(horiz):
                ml.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz),
                                                 basis, Nb))
            if not has_bulk:
                for xd in horiz:
                    tfld = sp.Function(sname(xd, ell), real=True)(*coords)
                    sig = (rf"\hat{{\sigma}}_{ell}" if dim == 2
                           else rf"\hat{{\sigma}}_{CN[xd]}_{ell}")
                    ml.apply(separation_of_variables(tfld, sp.Function(sig, real=True)(t, *horiz),
                                                     basis, Nb + 1))
            for nm in ["mass"] + MOM:
                getattr(ml, nm).apply(ExpandSums())
                getattr(ml, nm).apply(PullConstants())
                getattr(ml, nm).apply(ExtractBrackets(basis, var=zeta))
                getattr(ml, nm).apply({Nb: 0})
                getattr(ml, nm).apply(EvaluateSums())
                getattr(ml, nm).apply(ResolveModes(index=kk, modes=range(1)))
                getattr(ml, nm).apply(ResolveBasis(legendre, var=zeta))
            for i, xd in enumerate(horiz):
                ml.apply(ChangeOfVariables(shat(xd, ell), qname(xd, ell),
                                           lambda qi: qi / h_l))
            ml.apply(InvertMassMatrix())

            def _clean(e):
                """Drop zero integrals and pull constants out of derivatives, then expand."""
                e = e.replace(
                    lambda a: isinstance(a, sp.Integral) and a.function == 0,
                    lambda a: sp.S.Zero)

                def _pull(a):
                    """Pull the factor independent of the differentiation variables out of a
                    derivative.
                    """
                    vs = [v for v, _ in a.variable_count]
                    co, rest = a.expr.as_independent(*vs, as_Mul=True)
                    return (co * sp.Derivative(rest, *a.args[1:])
                            if co != 1 else a)
                e = e.replace(lambda a: isinstance(a, sp.Derivative), _pull)
                return sp.expand(e)
            cont = _clean(ml.mass[0].expr)
            mom = {xd: _clean(getattr(ml, f"momentum_{CN[xd]}")[0].expr)
                   for xd in horiz}
            return cont, mom

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer interface closure (∇·-based, dimension-agnostic) ──
        ht = sp.Function("h", positive=True)(t, *horiz)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, sp.S.One - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_l = {ell: {xd: sp.Function(qname(xd, ell), real=True)(0, t, *horiz)
                     for xd in horiz} for ell in range(1, N + 1)}
        u_lc = {ell: {xd: q_l[ell][xd] / (l_all[ell - 1] * ht) for xd in horiz}
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

        from zoomy_core.model.models.closures import interface_closure
        iface = interface_closure(self.closures)

        def _ustar(a, xd):
            """Transfer velocity ``u*`` shared by the layers adjacent to internal interface
            ``a`` in direction ``xd``.
            """
            below, above = u_lc[a][xd], u_lc[a + 1][xd]
            if iface is not None:
                return iface.expression(below, above, G_sol[Gf[a]])
            if self.interface_velocity == "mean":
                return (below + above) / 2
            return sp.Piecewise((below, G_sol[Gf[a]] >= 0), (above, True))

        # vertical-z placeholder so Model.horizontal = (x[, y]); see ml_sme
        m = DModel(coords=(t, *horiz, z), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            for xd in horiz:
                mom = layer_eqs[ell][1][xd]
                # each G_α enters once as ±G·u_ℓ(trace)/ρ — swap the layer's own
                # per-direction trace for the SHARED u*: G → G·u*_d/u_ℓ_d
                for a in (ell - 1, ell):
                    if 1 <= a <= N - 1:
                        mom = mom.subs(Gf[a], Gf[a] * _ustar(a, xd) / u_lc[ell][xd])
                mom = mom.subs(frac).subs(G_sol)
                enm = (f"momentum_{ell}" if dim == 2
                       else f"momentum_{CN[xd]}_{ell}")
                m.add_equation(enm, sp.expand(mom.subs(par).doit()))

        m.bed = b
        m.ht = ht
        m.q_l = [q_l[ell][xd] for ell in range(1, N + 1) for xd in horiz]
        m.G_closed = {str(k): sp.simplify(v.subs(par))
                      for k, v in G_sol.items()}
        return m

    # Built via ``SystemModel.from_model(MLSWE(...))`` (REQ-143); see
    # ``zoomy_core.systemmodel.model_builders.build_mlswe``.
    _system_model_kind = "mlswe"
