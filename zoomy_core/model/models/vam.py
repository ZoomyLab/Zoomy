"""VAM: the non-hydrostatic vertically-averaged moment Model, derived with the same Galerkin
recipe as `SME` but with a modal non-hydrostatic pressure.
`SystemModel.from_model(VAM(level=Nu))` gives the square DAE; `chorin_split` gives the
predictor/pressure/corrector sub-systems.
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
    ResolveModes, ResolveBasis, InvertMassMatrix, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.derivation.closure import GaussQuadrature
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.models.walls import register_free_slip_wall
from zoomy_core.model.models.equations import Stress
from zoomy_core.misc.options import OptionParam
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class VAM(BaseModel):
    """Non-hydrostatic vertically-averaged moment equations of order `level`; u, w and the
    non-hydrostatic pressure share the modal basis.

    The pressure modes are Lagrange multipliers of the divergence constraints; the predictor
    keeps the SWE hyperbolic structure.

    References:
        Escalante et al. 2024, https://doi.org/10.1016/j.jcp.2024.112882
    """

    _finalize_lazy = True               # declarative path
    _cacheable_derivation = True        # derive_model returns m, no self._X stash
    level = param.Integer(default=1, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_i, r_i, P_i); 3 → (t,x,y,z), two horizontal (q_x_i, q_y_i, r_i, "
        "P_i).  Every projection/closure loops over the horizontal directions."))
    stress = OptionParam(Stress, default=Stress.shallow, doc=(
        "How much of the bulk closure's stress the model keeps (`moment_scaling`): "
        "`Stress.shallow` the vertical shear only, `Stress.full` the whole Newtonian tensor."))
    scaling = param.Parameter(default=None, doc=(
        "A `ShallowScaling` (derivation.nondim): derive the stress truncation by order from "
        "the non-dimensional balance laws instead of `moment_scaling` / `apply_full_tensor`; "
        "it then decides `stress`.  None (default): the hand rules."))
    closures = param.List(default=[], doc=(
        "Composable stress Closure pieces (closures.py).  Empty leaves tau_xz "
        "UNCLOSED (modal moments stay free)."))
    small_slope = param.Boolean(default=True, doc=(
        "Resolve the opaque boundary frame to its n→ẑ limit (drop O(slope) "
        "traction corrections) — the shallow form.  False keeps the slope-aware "
        "tractions."))

    def derive_model(self):
        """Build the declarative VAM model (sigma-map, projected mass, horizontal and vertical
        momentum, closures) and its 3-D projection rows; called by the base `__init__`.
        """
        Nu = int(self.level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, MomentumNonHydrostatic, moment_scaling, small_slope_scaling,
            slope_aware_scaling, package_viscous)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures, apply_full_tensor

        # dimension setup (see SME): dim=2 → 1 horizontal (q_i,r_i,P_i,
        # byte-identical Escalante); dim=3 → 2 horizontal (q_x_i,q_y_i).  The
        # vertical (r) and pressure (P) moments stay scalar; only the horizontal
        # momentum doubles and the σ-mass-flux ω̃ + flux gain the ∂_y terms.
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]
        MOM = [f"momentum_{CN[xd]}" for xd in horiz]    # momentum_x[, momentum_y]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # Free-surface hydrostatic pressure flux (g·h²/2) is tagged in
        # ``system_model``; it is recomputed there from ``m`` rather than
        # stashed on ``self`` (keeps the model surface free of derivation
        # byproducts — see ``system_model``).

        # 1 — full system (hydrostatic pressure pre-absorbed) from blueprints
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        momblue = MomentumNonHydrostatic(m); m.add_equation(momblue)
        uvel, w, p = momblue.uvel, momblue.w, momblue.p

        def _kbc(interface):
            """Kinematic boundary condition of the vertical velocity at `interface`."""
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 1b — stress scaling: Stress.shallow (default, byte-identical Escalante) keeps the
        # vertical shear only; Stress.full keeps the whole tensor (the w rows get d_e tau_ez),
        # filled by the bulk closure (Newtonian) on the 3-D velocity BEFORE the σ-map and routed
        # to conservative diffusion by package_viscous after the ω̃/regroup/curvature machinery
        # (§6e).  The pressure unknown is -sigma_zz minus the hydrostatic part, zero at the
        # surface (MomentumNonHydrostatic).
        if self.scaling is not None:
            # the truncation by order of the non-dimensional balance laws (scaling.py); the
            # pressure p is already the non-hydrostatic part
            from zoomy_core.model.models.scaling import ordered_truncation, Pressure
            rec = ordered_truncation(m, momblue, self.closures or [], self.scaling,
                                     heights=[h, b], free_surface=b + h,
                                     pressure=Pressure.nonhydrostatic, mass="mass",
                                     requested=self.stress)
            vertical_stress = rec.vertical_stress
        else:
            moment_scaling(m, momblue, self.stress)
            apply_full_tensor(self.closures or [], m, momblue)
            vertical_stress = m.stress == Stress.full
        full_stress = m.stress == Stress.full

        # 2 — σ-map:  z = b + h ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        k = test_index(); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)

        def _project(eq):
            """Galerkin-project the equation `eq` onto the test function and apply the
            bed/surface kinematic conditions.
            """
            eq.apply(Multiply(h)); eq.apply(Multiply(c(zeta) * phi_k))
            eq.apply(ProductRule(variables=[zeta]))
            eq.apply(Integrate(zeta, bounds=(0, 1))); eq.apply(ResolveIntegral())
            eq.apply(m.kbc_bot); eq.apply(m.kbc_top)
            eq.apply({sp.Derivative(b, t): 0})

        # 3 — Galerkin-project mass + each horizontal momentum + vertical
        _project(m.mass); m.mass.apply(Simplify())
        pp = m.functions.p

        def _state(at, *, alias=None, btag=None):
            """`ClosureState` for the bulk (`at=None`) or the bed/surface trace
            (`at=0/1`).
            """
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz), base=b)
        for mn in MOM:
            mxi = getattr(m, mn)
            _project(mxi)
            mxi.apply({pp.at(1): 0})          # dynamic surface BC p(ζ=1)=0
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m, mn), "tau": tau_h[ax], "velname": HNAME[xd]}
                for mn, ax, xd in zip(MOM, ("x", "y"), horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for mn in MOM:
            getattr(m, mn).apply(Simplify())
        if bool(self.small_slope):
            small_slope_scaling(m)
        else:
            slope_aware_scaling(m, {"b": b, "eta": b + h})

        mz = m.momentum_z
        _project(mz); mz.apply({pp.at(1): 0}); mz.apply(Simplify())

        # 4 — modal ansatz: each u_i ∈ P_Nu; w, p ∈ P_{Nu+1} (Escalante spaces)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        ph = sp.Function(r"\hat{p}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        m.apply(separation_of_variables(p, ph(t, *horiz), basis, N_u + 1))
        if not has_bulk:
            for ax, xd in zip(("x", "y"), horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 5 — resolve: mass k=0…Nu+1; each momentum + vertical k=0…Nu
        resolve = ([("mass", range(Nu + 2))]
                   + [(mn, range(Nu + 1)) for mn in MOM]
                   + [("momentum_z", range(Nu + 1))])
        for nm, modes in resolve:
            getattr(m, nm).apply(ExpandSums())
            getattr(m, nm).apply(PullConstants())
            getattr(m, nm).apply(ExtractBrackets(basis, var=zeta))
            getattr(m, nm).apply({N_u: Nu})
            getattr(m, nm).apply(EvaluateSums())
            getattr(m, nm).apply(ResolveModes(index=k, modes=modes))
            getattr(m, nm).apply(ResolveBasis(legendre, var=zeta))
            # REQ-160: numerically resolve any Galerkin integral a
            # non-polynomial closure leaves behind (no-op when none remain).
            if int(self.quadrature_order) > 0:
                getattr(m, nm).apply(
                    GaussQuadrature(var=zeta, order=int(self.quadrature_order)))

        # 5b — top w/p mode closures (Escalante eq 6).  2-D: the bottom
        # kinematic is w(0) = Σ_d u_d(0)·∂_d b.
        top = Nu + 1
        # bottom KBC closes the top w-mode; surface BC p(ζ=1)=0 closes the top
        # p-mode.  Every basis value goes through the basis object: the bed value
        # φ_j(0) (= (−1)^j for shifted Legendre) and surface value φ_j(1) (= 1).
        u_at0 = [sum(legendre.at0(j) * coeff_heads[i](j, t, *horiz) for j in range(Nu + 1))
                 for i in range(len(horiz))]
        w_top = (sum(u_at0[i] * DERIV[xd](b) for i, xd in enumerate(horiz))
                 - sum(legendre.at0(j) * wh(j, t, *horiz) for j in range(top))) / legendre.at0(top)
        p_top = -sum(legendre.at1(j) * ph(j, t, *horiz) for j in range(top)) / legendre.at1(top)
        m.apply({wh(top, t, *horiz): w_top})
        m.apply({ph(top, t, *horiz): p_top})

        # 6 — conservative CoV (û_d→q_d/h, ŵ→r/h, p̂→P) + h-eq substitution
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda qi: qi / h))
        m.apply(ChangeOfVariables(r"\hat{w}", "r", lambda ri: ri / h))
        m.apply(ChangeOfVariables(r"\hat{p}", "P", lambda pi: pi))
        h_eq = m.mass[0].solve_for(d.t(h))
        for nm in MOM + ["momentum_z"]:
            for kk in range(Nu + 1):
                getattr(m, nm)[kk].apply(h_eq)
                getattr(m, nm)[kk].apply(Consolidate())
        for kk in range(1, Nu + 2):
            m.mass[kk].apply(h_eq)
            m.mass[kk].apply(Consolidate())
        m.apply(InvertMassMatrix())

        # 6b — σ-mass-flux ω̃ correction (Escalante).  2-D: ω̃ couples ∂_x AND
        # ∂_y (the vertical-coupling operator just gains the second-horizontal
        # divergence + the v-advection term); the correction Δ_k is applied to
        # every horizontal momentum (field = u_d_m) and the vertical (w̃_m).
        phis = [legendre.eval(j, zeta) for j in range(Nu + 2)]
        # μ_j = ⟨φ_j, φ_j⟩, the basis Gram-norm (1/(2j+1) for shifted Legendre);
        # read off the basis, not hardcoded — the projection divides by it.
        mus = [legendre.gram(j, j) for j in range(Nu + 2)]
        qf = [[getattr(m.functions, qn).head(j, t, *horiz) for j in range(Nu + 1)]
              for qn in QNAME]                       # qf[di][j]
        rf = [m.functions.r.head(j, t, *horiz) for j in range(Nu + 1)]
        uvel_m = [sum(qf[di][j] / h * phis[j] for j in range(Nu + 1))
                  for di in range(len(horiz))]
        w_top_cons = (
            sum(sum(legendre.at0(j) * qf[di][j] / h for j in range(Nu + 1)) * DERIV[xd](b)
                for di, xd in enumerate(horiz))
            - sum(legendre.at0(j) * rf[j] / h for j in range(Nu + 1))) / legendre.at0(Nu + 1)
        wt_m = (sum(rf[j] / h * phis[j] for j in range(Nu + 1))
                + w_top_cons * phis[Nu + 1])
        # ∂_t h = −Σ_d ∂_d(q_d_0); advection Σ_d u_d_m(ζ ∂_d h + ∂_d b)
        dth = -sum(DERIV[xd](qf[di][0]) for di, xd in enumerate(horiz))
        omega_def = (wt_m - zeta * dth
                     - sum(uvel_m[di] * (zeta * DERIV[xd](h) + DERIV[xd](b))
                           for di, xd in enumerate(horiz)))
        omega_closed = -sum(DERIV[xd](qf[di][j]) * legendre.eval_psi(j, zeta)
                            for di, xd in enumerate(horiz) for j in range(1, Nu + 1))
        R_om = sp.expand(omega_closed - omega_def)
        assert sp.simplify(R_om.subs(zeta, 0)) == 0   # vanishes at the bed (KBC)

        def _zint01(e):
            """Integral of the polynomial `e` in zeta over [0, 1]."""
            poly = sp.Poly(sp.expand(e.doit()), zeta)
            return sum(cc / (nn[0] + 1)
                       for nn, cc in zip(poly.monoms(), poly.coeffs()))

        for kk in range(1, Nu + 1):
            dphi = sp.diff(phis[kk], zeta)
            for di, mn in enumerate(MOM):
                getattr(m, mn)[kk].expr = sp.expand(
                    getattr(m, mn)[kk].expr - _zint01(R_om * uvel_m[di] * dphi) / mus[kk])
            m.momentum_z[kk].expr = sp.expand(
                m.momentum_z[kk].expr - _zint01(R_om * wt_m * dphi) / mus[kk])

        # 6c — conservative regroup, DERIVED from our modal flux (no hardcoding).
        # F_{row,e,k} = (2k+1) h ∫₀¹ φ_k · flux_{row,e}(ζ) dζ, the Galerkin
        # projection of OUR physical flux of momentum-component `row` in
        # direction `e`: u_row·u_e (+ p/ρ on the diagonal of a horizontal row;
        # the vertical row carries no pressure flux).  Regrouped as a compound
        # ∂_e atom so the extraction reads it as flux; bed-slope-bearing parts
        # kept in their own atom (extraction's diffusion-branch caveat).  This
        # reproduces the Escalante flux for any Nu and any dimension.
        Pf = [m.functions.P.head(j, t, *horiz) for j in range(Nu + 1)]
        # top pressure mode closed by p(ζ=1)=0: P_top = −Σ φ_j(1)·P_j / φ_top(1).
        p_top_mode = (-sum(legendre.at1(j) * Pf[j] for j in range(Nu + 1))
                      / legendre.at1(Nu + 1))
        p_zeta = (sum(Pf[j] * phis[j] for j in range(Nu + 1))
                  + p_top_mode * phis[Nu + 1])
        vel_of = {mn: uvel_m[di] for di, mn in enumerate(MOM)}
        vel_of["momentum_z"] = wt_m
        bxs = [sp.Derivative(b, xd) for xd in horiz]
        for fam, vel_row in vel_of.items():
            for kk in range(1, Nu + 1):
                eq = getattr(m, fam)[kk]
                # expand ONCE per row (the .doit() would otherwise destroy a
                # prior direction's compound when this row is hit again for the
                # next flux direction); the inner sp.expand below has NO .doit(),
                # so compounds added for earlier directions survive.
                ex = sp.expand(sp.sympify(eq.expr).doit())
                for e, xe in enumerate(horiz):       # each flux direction
                    press = (p_zeta / rho if (fam in MOM and MOM[e] == fam)
                             else sp.S.Zero)
                    integrand = vel_row * uvel_m[e] + press
                    Fk = sp.expand(h / legendre.gram(kk, kk)
                                   * _zint01(phis[kk] * integrand))
                    F_bx = sum((tm for tm in sp.Add.make_args(Fk)
                                if any(tm.has(bxd) for bxd in bxs)), sp.S.Zero)
                    for F in (Fk - F_bx, F_bx):      # bed-free, then bed-slope
                        if F == 0:
                            continue
                        dF = sp.Derivative(F, xe)
                        ex = sp.expand(ex - sp.expand(dF.doit())) + dF
                eq.expr = ex

        # 6d — bed CURVATURE: over a curved bed the non-hydrostatic vertical
        # momentum picks up bare second derivatives C·∂²_e b (physical: w|_bed =
        # u·∇b).  The extraction can only route a second derivative inside a
        # conservative compound, so rewrite each bare term by the exact identity
        #   C·∂²_e b = ∂_e(C·∂_e b) − ∂_e(C)·∂_e b
        # (curvature → ∂_e(D·∂_e b) compound; remainder → first-derivative NCP).
        # No-op on a flat/mild bed and in 1-D (the §6c regroup leaves no bare ∂²b).
        for fam in MOM + ["momentum_z"]:
            for kk in range(1, Nu + 1):
                eq = getattr(m, fam)[kk]
                expr = sp.sympify(eq.expr)
                for xe in horiz:
                    d2 = sp.Derivative(b, (xe, 2))
                    out, changed = [], False
                    for tm in sp.Add.make_args(expr):
                        C = tm / d2
                        if (tm.has(d2) and not isinstance(tm, sp.Derivative)
                                and not C.has(d2)
                                and not C.has(sp.Derivative(b, xe))):
                            comp = sp.Derivative(C * sp.Derivative(b, xe), xe)
                            rem = -sp.expand(sp.Derivative(C, xe).doit()) * sp.Derivative(b, xe)
                            out.append(comp + rem); changed = True
                        else:
                            out.append(tm)
                    if changed:
                        expr = sp.Add(*out)
                eq.expr = expr

        # 6e — KEEP-ALL: route the retained in-plane deviatoric-stress divergence
        # into conservative diffusion (mirrors SME §9b / fullvam.py).  Runs AFTER
        # the ω̃/regroup/curvature machinery so it repackages the residual bare
        # second-derivatives C·∂²(q_i/h) (+ topography-coupled ν·q·∂b) into
        # ∂(D·∂Q) − ∂(D)·∂Q; b is a STATE in `states` so those evaluate from the
        # same stage (no frozen-b).  No-op unless full_stress.
        if full_stress or vertical_stress:
            states = {h, b}
            for qn in QNAME:
                qh = getattr(m.functions, qn).head
                states |= {qh(i, t, *horiz) for i in range(Nu + 1)}
            # the w rows too: their d_e tau_ez (first derivatives; package_viscous evaluates the
            # mode-0 brackets of d_x d_zeta the projection leaves as integrals)
            states |= {m.functions.r.head(i, t, *horiz) for i in range(Nu + 1)}
            for mn in (MOM if full_stress else []) + (["momentum_z"] if vertical_stress else []):
                for kk in range(Nu + 1):
                    getattr(m, mn)[kk].expr = package_viscous(
                        getattr(m, mn)[kk].expr, m.parameters.nu, states, list(horiz))

        # 6f — model-derived free-slip wall.  A lateral wall normal is
        # HORIZONTAL (n_x, n_y, 0), so the geometric statement
        # ``q_i → q_i − 2 n (n·q_i)`` acts on the horizontal momentum VECTOR of
        # each moment i; the vertical moments r_i and the pressure modes P_i are
        # scalars with respect to that normal (n·ẑ = 0) and extrapolate, as do
        # h and b.  One registration serves every wall orientation — including
        # oblique faces — because the n_d are the canonical face-normal symbols
        # FromModel substitutes per face.  See ``models/walls.py``.
        vam_q_heads = [getattr(m.functions, qn).head for qn in QNAME]
        register_free_slip_wall(
            m, ([qh(i, t, *horiz) for qh in vam_q_heads] for i in range(Nu + 1)))

        # 6g — WB reconstruction rows (mirrors ``SME`` §12): limit the FREE
        # SURFACE η = b+h and the PRIMITIVE velocity moments q_d/h and r/h.
        # Both q and r are h-SCALED by construction (§6a: the change of
        # variables ``\hat{w} → r`` carries ``ri/h``), so their primitives stay
        # bounded as h → 0 — that is what removes the wet/dry threshold from
        # the reconstruction instead of retuning it.  The pressure modes P_i
        # are NOT h-scaled (``\hat{p} → P`` is the identity there) and PASS
        # THROUGH: unregistered slots default to the identity at extraction, so
        # they are deliberately absent here rather than listed as identity.
        # Without these rows ``reconstruction_variables`` stayed ``None``, and
        # every order-2 VAM run fell back to the index-driven legacy path that
        # needs ``free_surface_h_index``.
        m.reconstruction_rows = {h: b + h}
        for qn in QNAME:
            _qh = getattr(m.functions, qn).head
            m.reconstruction_rows.update(
                {_qh(i, t, *horiz): _qh(i, t, *horiz) / h for i in range(Nu + 1)})
        _rh = m.functions.r.head
        m.reconstruction_rows.update(
            {_rh(i, t, *horiz): _rh(i, t, *horiz) / h for i in range(Nu + 1)})

        # 7 — vertical reconstruction → interpolate (field order [b,h,u,v,w,p];
        # v at index 3 only in dim=3).  The modal profiles assembled in §6b/§6c
        # ARE the reconstruction: ``uvel_m[di]`` = Σ_{i≤Nu}(q_{di,i}/h)·φ_i is the
        # horizontal velocity; ``wt_m`` = Σ_{j≤Nu}(r_j/h)·φ_j + ŵ_top·φ_{Nu+1}
        # the vertical velocity with the bottom-KBC top mode (so w(0)=u(0)·∂_x b);
        # ``p_zeta`` = Σ_{k≤Nu} P_k·φ_k + P_top·φ_{Nu+1} the NON-hydrostatic
        # pressure with the surface p(ζ=1)=0 top mode.  Slot 5 is the TOTAL
        # pressure: the split hydrostatic part ρ g h (1−ζ) plus that modal part.
        HNAME = {x: "u", y: "v"}
        interp = {0: b, 1: h}
        for vi in range(len(horiz)):
            interp[2 + vi] = uvel_m[vi]
        interp[4] = wt_m
        interp[5] = rho * g * h * (1 - zeta) + p_zeta
        m.interpolate_rows = interp

        # 8 — project (inverse of interpolate): the Integral-FREE fixed-node
        # Galerkin reduction (see SME §13).  N_z uniform nodes, trapezoid
        # weights; ``projection_rows`` carries the 1/∫φ_k² normalisation, so we
        # supply only the physical factor.  Horizontal momenta q and the vertical
        # r are CONSERVED moments q_k = h·α_k (norm h, sampling P3_u/P3_v / P3_w);
        # the pressure modes P_k are PLAIN modal coefficients of the
        # non-hydrostatic column (norm 1), so the sampled total-pressure column
        # P3_p has its hydrostatic part ρ g h (1−ζ_j) removed before projection.
        # The closed top modes (ŵ_{Nu+1}, P_{Nu+1}) are NOT state — no row.
        N_z = 33
        nodes = [float(j) / (N_z - 1) for j in range(N_z)]
        weights = [1.0 / (N_z - 1)] * N_z
        weights[0] *= 0.5; weights[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        m.project_rows = {b: P3["b"], h: P3["h"]}
        for xd, qn in zip(horiz, QNAME):
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            samples = [P3vel(nd) for nd in nodes]
            rows = legendre.projection_rows(nodes, weights, samples,
                                            norm=lambda _k: P3["h"])
            qh = getattr(m.functions, qn).head
            m.project_rows.update({qh(i, t, *horiz): rows[i] for i in range(Nu + 1)})
        # vertical r_k = h·⟨φ_k, w⟩
        P3w = sp.Function("P3_w", real=True)
        rows_w = legendre.projection_rows(nodes, weights, [P3w(nd) for nd in nodes],
                                          norm=lambda _k: P3["h"])
        m.project_rows.update({m.functions.r.head(i, t, *horiz): rows_w[i]
                               for i in range(Nu + 1)})
        # pressure P_k = ⟨φ_k, p − ρ g h (1−ζ)⟩ (modal, no h factor)
        P3p = sp.Function("P3_p", real=True)
        samples_p = [P3p(nd) - rho * g * P3["h"] * (1 - nd) for nd in nodes]
        rows_p = legendre.projection_rows(nodes, weights, samples_p, norm=None)
        m.project_rows.update({m.functions.P.head(i, t, *horiz): rows_p[i]
                               for i in range(Nu + 1)})

        return m

    # The square DAE (state ``[b, h, q_k, r_k, P_k]``; the P rows are the
    # divergence constraints) is built via ``SystemModel.from_model(VAM(...))``
    # (REQ-143); the builder in ``zoomy_core.systemmodel.model_builders`` selects
    # Q (+ pressure modes), tags the hydrostatic flux and resolves the BCs.  A
    # FRESH SystemModel is built on every call, so callers may mutate ICs/BCs.
    _system_model_kind = "vam"

    def chorin_split(self, dt=None, *, system_model=None):
        """Split the lifted system into the Chorin predictor, pressure and corrector sub-systems
        `(SM_pred, SM_press, SM_corr)`, with row roles read structurally off the operators.

        Args:
            dt: time step symbol; defaults to a fresh positive `dt` that the solver renames.
            system_model: an already configured SystemModel; attach ICs/BCs before the split and
                re-attach BCs on `SM_pred` after.
        References:
            Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
        """
        from zoomy_core.model.splitter import split_for_pressure_structural
        sm = system_model if system_model is not None \
            else SystemModel.from_model(self)
        if dt is None:
            dt = sp.Symbol("dt", positive=True)
        P_syms = [s for s in sm.state if str(s).startswith("P_")]
        return split_for_pressure_structural(sm, P_syms, dt)
