"""ML-VAM-WD — ML-VAM with the **wet/dry (discharge-form) divergence
constraint** of 86BQYCRP Eq. 19.  EXPERIMENTAL sibling of `ml_vam.py`.

Byte-identical to `ml_vam.py` except for ONE block — the per-layer
divergence-constraint emission in `MLVAMWD.derive_model` (search
`86BQYCRP Eq. 19`).  The stock model keeps the σ-mass-flux compound
`(1/h)·∂_x(h·q_k)` folded, which leaves a `1/h` INSIDE the emitted elliptic
operator and makes it SINGULAR as `h→0` at a thinning front.  Here the row is
distributed and multiplied by `l_ℓ·h²` so the `1/h` cancels analytically and
the pressure block stays invertible (`P→0` automatically at wet/dry fronts,
no threshold).  Same constraint manifold wherever `h>0`.

Kept separate so the validated `MLVAM` is untouched; fold back
once the 2-layer Escalante bump survives the front (cid 182).

Second departure: the **LDNH constraint structure** (Fernández-Nieto/Parisot/
Penel/Sainte-Marie 2018; Escalante–Morales de Luna 2021 eq. 8/10; the
multilayer version 32K4N2P5 §3.1).  ml_vam.py SPENDS the bottom/interface
kinematic condition to eliminate the top `ŵ` mode; here `ŵ_top` stays a
STATE (with its own `momentum_z` row) and the kinematic residual is re-added
as an INDEPENDENT constraint row, paired with the extra pressure unknown that
the `N^q = N^w + 1` degree pairing prescribes.  Spaces per layer:

```
    u ∈ P_Nu   (Nu+1 free modes, q_ℓ_k)
    w ∈ P_{Nu+1}  (Nu+2 free modes, r_ℓ_k — NO closed top mode)
    p ∈ P_{Nu+2}  (Nu+2 free modes, P_ℓ_j — top mode closed by the trace)

    constraints per layer = mass[1 … Nu+1]  +  R_kbc      = Nu+2
    free pressure modes   = P_ℓ_0 … P_ℓ_{Nu+1}           = Nu+2   ✔ square
```

MEASURED (STEP-1 rank study, symbolic Jacobian on atomised modal columns, at
(N,Nu) = (1,1), (2,1), (1,2)): the per-layer relation set
`{mass[0…Nu+1], R_kbc, R_top}` has Nu+4 rows of rank Nu+3 — EXACTLY ONE
dependency, `mass[Nu+1] = ∓R_kbc − R_top`, i.e. the top mass mode is pure
boundary data.  Consequences, both honoured below:

* `R_top` (surface/interface KBC) is REDUNDANT — do not add it;
* `mass[0]` is NOT free — it is the only relation supplying `∂_t h_ℓ`/`G`
  and stays the layer depth equation (measured gain from freeing it: **0**);
* freeing `ŵ_top` and adding `R_kbc` buys **+1** independent constraint —
  which is precisely the one extra pressure degree of `N^q = N^w + 1`.

⚠ HISTORY: an earlier pass widened p to `P_{Nu+2}` and matched it with the
mode-`Nu+2` LAYER-MASS projection.  That row is non-zero but DEPENDENT (it
differs from mode `Nu+1` by twice the bottom KBC, which was then substituted
away), so the pressure block was rank `Nu+1` for `Nu+2` unknowns.  The
extra pressure degree needs the KINEMATIC row, never another mass mode.

Layers are coupled by `KinematicBC(mass_flux=G_ℓ)` with the Hörnschemeyer
fraction-multiplier closure (`h_ℓ = l_ℓ·h`).

The N pressure top-mode closures cascade DOWNWARD through the stack: surface
`p_N(1) = 0`; every lower layer's top trace equals the layer-above's bottom
trace (downward convention — the upward one makes the elliptic block singular,
see the ml_vam thesis notebook).  The traces are FULL modal traces including
the closed top modes, so the chain resolves from the surface to the bed.

The N kinematic conditions are NO LONGER closures.  ``w_ℓ(0) = ∂t z_{ℓ-½} +
u_ℓ(0)·∂x z_{ℓ-½} + G_{ℓ-½}/ρ`` (bottom KBC `w_1(0) = u_1(0)·∂x b` on the
bed) is emitted as the constraint row `constraint_kbc_{ℓ}`, whose Lagrange
multiplier is the extra per-layer pressure unknown — the bottom/interface
pressure `q_b` of Escalante 2021 eq. (10).

Per layer the advective σ-mass flux ω̃ is resolved to its CLOSED form
`ω̃(ζ) = G_{ℓ-½}/ρ − ∫₀^ζ (∂t h_ℓ + ∂x(h_ℓ ũ_ℓ))` (Escalante's
eq(3)→(4) step with the interface mass-flux offset) — the definition form
differs by {layer-mass, constraint} combinations, equivalent ON the
constraint manifold but not in the off-manifold predictor.

Interface momentum transfer: `u*` is the surgical mean-trace swap on the
x-momentum rows (u-traces are genuinely one-sided).  The w-rows get NO
swap — the kinematic closures make the w-traces of adjacent layers exactly
continuous, so a w* transfer correction is identically zero.

`SystemModel.from_model(MLVAMWD(...))` is the square DAE (state
`[b, h, q_ℓk, r_ℓk, P_ℓj]`; the per-layer divergence constraints are the
zero-mass-matrix rows); `MLVAMWD.chorin_split(dt)` returns the structural
predictor / pressure / corrector sub-systems for
`ChorinSplitVAMSolver`.
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
from zoomy_core.model.derivation.system_extract import HydrostaticPressure
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.models.walls import register_free_slip_wall
from zoomy_core.model.models.equations import evaluate_time_derivatives
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class MLVAMWD(BaseModel):
    """ML-VAM in the **LDNH constraint structure** with the **wet/dry
    (discharge-form) constraint scaling** of 86BQYCRP Eq. 19.

    Two departures from `MLVAM`:

    * the per-layer divergence constraints are emitted multiplied by `l_ℓ·h²`
      (the kinematic row by `l_ℓ·h`) with the `1/h` cancelled ANALYTICALLY,
      so the elliptic block stays invertible as `h→0`;
    * `ŵ_top` is a STATE, not a closure.  `w ∈ P_{Nu+1}` keeps all `Nu+2`
      modes (each with its own `momentum_z` row) and the bottom/interface
      kinematic condition returns as the constraint `constraint_kbc_{ℓ}`.
      That buys the +1 independent constraint (measured, see
      `derive_model`) which pairs with `p ∈ P_{Nu+2}`'s extra unknown:
      `Nu+2` constraints ↔ `Nu+2` free pressure modes per layer.

    EXPERIMENTAL sibling, not a replacement: kept as its own model so the
    validated `MLVAM` is untouched while this is proven.  Fold back into
    `ml_vam.py` only once the 2-layer Escalante bump survives the thinning
    front (cid 182)."""

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    n_layers = param.Integer(default=2, bounds=(1, None))
    level = param.Integer(default=1, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_ℓ_i, r_ℓ_i, P_ℓ_i); 3 → (t,x,y,z), two horizontal (q_x_ℓ_i, "
        "q_y_ℓ_i; r/P stay scalar per layer)."))
    closures = param.List(default=[], doc=(
        "Composable Closure pieces (closures.py): stress AND the interface "
        "transfer scheme (MeanInterface/UpwindInterface). Default interface "
        "scheme is the mean; empty stress leaves tau UNCLOSED."))

    def derive_model(self):
        N = int(self.n_layers)
        Nu = int(self.level)
        top = Nu + 1          # HIGHEST w mode — a STATE (LDNH), not a closure
        top_p = Nu + 2        # closed TOP mode of p ∈ P_{Nu+2} (trace cascade)
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        def qname(xd, ell):
            return f"q_{ell}" if dim == 2 else f"q_{CN[xd]}_{ell}"
        def shat(xd, ell):
            return (rf"\hat{{u}}_{ell}" if dim == 2
                    else rf"\hat{{{HNAME[xd]}}}_{ell}")
        def sname(xd, ell):
            return f"tau_{ell}" if dim == 2 else f"tau_{CN[xd]}z_{ell}"
        MOM = [f"momentum_{CN[xd]}" for xd in horiz]
        # e_x (downslope gravity component) is minted on the per-layer sub-models
        # by MomentumNonHydrostatic via gravity_components; declare it on the
        # assembled model too so it binds with a value (like sme/vam/ke_sme).
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
        P_heads = [sp.Function(f"P_{ell}", real=True)
                   for ell in range(1, N + 1)]
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        # inner (per-layer) basis object: the per-layer modal reconstruction,
        # top-mode closures and interface traces all go through it, so the
        # bed/surface values φ_k(0)=(−1)^k / φ_k(1)=1 and the running integral
        # ∫₀^ζ φ_j are basis primitives rather than hard-coded Legendre forms.
        # sized by the RICHEST vertical space, the pressure one (top_p = Nu+2).
        inner_basis = Legendre_shifted(level=top_p + 1)
        phis = [inner_basis.eval(j, zeta) for j in range(top_p + 1)]
        # μ_j = ⟨φ_j, φ_j⟩, the basis Gram-norm (1/(2j+1) for shifted Legendre);
        # read off the basis, not hardcoded — the projection divides by it.
        mus = [inner_basis.gram(j, j) for j in range(top_p + 1)]

        def _zint01(e):
            poly = sp.Poly(sp.expand(e.doit()), zeta)
            return sum(cc / (nn[0] + 1)
                       for nn, cc in zip(poly.monoms(), poly.coeffs()))

        def derive_layer(ell, p_top_trace_full):
            """One VAM column.  Returns (cont, constraints, momx, momz,
            p_bot_trace_full) — the bottom pressure trace (FULL, incl. the
            closed top mode) feeds the layer below."""
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=coords, parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            p = sp.Function(f"p_{ell}", real=True)(*coords)
            from zoomy_core.model.models.equations import (
                Mass, MomentumNonHydrostatic, small_slope_scaling, moment_scaling)
            from zoomy_core.model.models.material import ClosureState
            from zoomy_core.model.models.closures import apply_layer_stress_closures
            ml.add_equation(Mass(ml, suffix=f"_{ell}"))
            # tau_name → tau_<ell> in 1-D (1 horizontal uses tau_name, ignores
            # suffix); ignored in 2-D where the blueprint mints per-direction
            # tau_xz_<ell>/tau_yz_<ell> (matches sname()).
            blue = MomentumNonHydrostatic(
                ml, suffix=f"_{ell}", tau_name=f"tau_{ell}", free_surface=b + H)
            ml.add_equation(blue)
            moment_scaling(ml, blue)            # shallow stress (the vertical shear only)
            uvel = [sp.Function(HNAME[xd] + f"_{ell}", real=True)(*coords)
                    for xd in horiz]
            w = sp.Function(f"w_{ell}", real=True)(*coords)

            def _kbc(iface, G):
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
            legendre = Legendre_shifted(level=top_p + 1)
            pp = getattr(ml.functions, f"p_{ell}")

            for nm in ["mass"] + MOM + ["momentum_z"]:
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
                getattr(ml, nm).apply({pp.at(1): p_top_trace_full})
            # REQ-160 root cause (mirror MLSME): read the REAL layer parameter
            # namespace so EVERY closure parameter (tau_y, eps_reg, …) resolves —
            # the hand-built SimpleNamespace(rho, nu, lambda_s) lacked them.
            for c in (self.closures or []):
                c.register(ml)

            def _state(at, *, alias=None, btag=None):
                fields = {"u": getattr(ml.functions, f"u_{ell}"),
                          "w": getattr(ml.functions, f"w_{ell}")}
                if dim == 3:
                    fields["v"] = getattr(ml.functions, f"v_{ell}")
                return ClosureState(fields, params=ml.parameters, h=h_l, x=C.x,
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
            small_slope_scaling(ml)          # shallow boundary frame (n→ẑ)

            # modal ansatz: each u_d ∈ P_Nu;  w ∈ P_{Nu+1};  p ∈ P_{Nu+2}
            # (N^q = N^w + 1, Fernández-Nieto 2018 eq. 2.5c / Escalante 2023)
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
            ml.apply(separation_of_variables(
                p, P_heads[ell - 1](t, *horiz), basis, Nb + 2))

            # mass is projected onto modes 0 … Nu+1 ONLY: mode 0 is the layer
            # continuity (it DEFINES ∂_t h_ℓ / G), modes 1 … Nu+1 are the
            # divergence constraints.  The Nu+2'nd free pressure mode is paired
            # NOT with a further mass mode but with the kinematic row R_kbc
            # emitted below — see the ⚠ note.
            #
            # ⚠ DO NOT ADD ``mass[Nu+2]`` HERE (measured, any Nu, any layer).
            # That row is non-zero but DEPENDENT: before the modal closures it
            # differs from the mode-(Nu+1) row by exactly
            #
            #   mass[Nu+2] − mass[Nu+1] = −2·[ u(ζ=0)·∂_x z_bot + G_bot/ρ
            #                                  − w(ζ=0) ]   =  −2·R_kbc
            #
            # i.e. by twice the bottom kinematic condition.  Adding BOTH it and
            # R_kbc re-creates the singular block of the earlier attempt.
            #
            # momentum_z now spans modes 0 … Nu+1 (one per w mode) because
            # ŵ_top is a state, not a closure.
            resolve = ([("mass", range(Nu + 2))]
                       + [(mn, range(Nu + 1)) for mn in MOM]
                       + [("momentum_z", range(Nu + 2))])
            # per-equation free-mode count, READ OFF ``resolve`` so the two can
            # never drift: MOM → Nu+1, momentum_z → Nu+2 (ŵ_top is a state).
            nmode = {nm: len(modes) for nm, modes in resolve}
            for nm, modes in resolve:
                getattr(ml, nm).apply(ExpandSums())
                getattr(ml, nm).apply(PullConstants())
                getattr(ml, nm).apply(ExtractBrackets(basis, var=zeta))
                getattr(ml, nm).apply({Nb: Nu})
                getattr(ml, nm).apply(EvaluateSums())
                getattr(ml, nm).apply(ResolveModes(index=kk, modes=modes))
                getattr(ml, nm).apply(ResolveBasis(legendre, var=zeta))
                # REQ-160: numerically resolve non-polynomial-closure integrals
                # (no-op when none remain), routed from the shared order.
                if int(self.quadrature_order) > 0:
                    getattr(ml, nm).apply(
                        GaussQuadrature(var=zeta, order=int(self.quadrature_order)))

            # ── MODAL closure for the top p mode ONLY (LDNH) ──────────────
            # top-pressure trace (downward convention):
            #   Σ_j p̂_j = p_top_trace  →  closes p̂_top_p   (top_p = Nu+2)
            #
            # The bottom kinematic/interface condition (KinematicBC orientation:
            # w|_at = ∂t I + u|_at·∂x I + G/ρ;  ∂t b = 0)
            #   R_kbc := Σ_j (−1)^j ŵ_j − w_bot = 0
            # is NOT used to close ŵ_top: ŵ_top stays a state carrying its own
            # momentum_z row, and R_kbc is emitted as a constraint row in the
            # assembly below (its multiplier is the extra pressure unknown —
            # the bottom/interface pressure q_b of Escalante 2021 eq. 10).
            u_at0 = [sum(inner_basis.at0(j) * coeff_heads[i](j, t, *horiz)
                         for j in range(Nu + 1)) for i in range(len(horiz))]
            w_bot = (sp.Derivative(z_bot, t).doit().subs(sp.Derivative(b, t), 0)
                     + sum(u_at0[i] * DERIV[xd](z_bot)
                           for i, xd in enumerate(horiz))
                     + G_bot / rl)
            p_top_mode = (p_top_trace_full
                          - sum(inner_basis.at1(j) * P_heads[ell - 1](j, t, *horiz)
                                for j in range(top_p))) / inner_basis.at1(top_p)
            ml.apply({P_heads[ell - 1](top_p, t, *horiz): p_top_mode})
            p_bot_trace_full = sp.expand(
                sum(inner_basis.at0(j) * P_heads[ell - 1](j, t, *horiz)
                    for j in range(top_p))
                + inner_basis.at0(top_p) * p_top_mode)

            for i, xd in enumerate(horiz):
                ml.apply(ChangeOfVariables(shat(xd, ell), qname(xd, ell),
                                           lambda qi: qi / h_l))
            ml.apply(ChangeOfVariables(
                rf"\hat{{w}}_{ell}", f"r_{ell}", lambda ri: ri / h_l))
            for nm in MOM + ["momentum_z"]:
                for k in range(nmode[nm]):
                    eqk = getattr(ml, nm)[k]
                    # surface ∂_t h_ℓ (for the h-eq substitution) + resolve the
                    # modal-closure Subs WITHOUT distributing the folded spatial
                    # flux ∂_x(q²/h_ℓ) / ∂_x(g h_ℓ²/2) — a blanket .doit() here
                    # sent mode 0's whole momentum flux into the NCP slot.
                    eqk.expr = sp.expand(evaluate_time_derivatives(eqk.expr, t))
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for nm in MOM + ["momentum_z"]:
                for k in range(nmode[nm]):
                    getattr(ml, nm)[k].apply(h_eq)
                    getattr(ml, nm)[k].apply(Consolidate())
            for k in range(1, Nu + 2):
                ml.mass[k].apply(h_eq)
                ml.mass[k].apply(Consolidate())
            for nm in MOM + ["momentum_z"]:
                getattr(ml, nm).apply(InvertMassMatrix())

            # ── ω̃ resolution (closed σ-mass-flux form, G-offset); 2-D couples
            #    ∂_x and ∂_y ───────────────────────────────────────────────
            qfm = [[sp.Function(qname(xd, ell), real=True)(j, t, *horiz)
                    for j in range(Nu + 1)] for xd in horiz]   # qfm[di][j]
            rfm = [sp.Function(f"r_{ell}", real=True)(j, t, *horiz)
                   for j in range(nmode["momentum_z"])]
            dt_hl = sp.sympify(h_eq.rhs)
            u_at0_c = [sum(inner_basis.at0(j) * qfm[di][j] for j in range(Nu + 1)) / h_l
                       for di in range(len(horiz))]
            zb_t = sp.Derivative(z_bot, t).doit().subs(sp.Derivative(b, t), 0)
            w_bot_c = (zb_t + sum(u_at0_c[di] * DERIV[xd](z_bot)
                                  for di, xd in enumerate(horiz)) + G_bot / rl)
            uvel_m = [sum(qfm[di][j] / h_l * phis[j] for j in range(Nu + 1))
                      for di in range(len(horiz))]
            # every w mode is free now — no bottom-KBC top mode to append
            wt_m = sum(rfm[j] / h_l * phis[j] for j in range(nmode["momentum_z"]))
            # the LDNH kinematic constraint in the CONSERVED layer state, scaled
            # by h_ℓ so the 1/h of q/h, r/h cancels ANALYTICALLY (86BQYCRP Eq. 19
            # in its kinematic-row form: ONE power of h, not h² — R_kbc carries a
            # single 1/h, the divergence rows carry ∂_x(q/h) hence two).
            R_kbc_c = sp.expand(h_l * (wt_m.subs(zeta, 0) - w_bot_c))
            omega_def = (wt_m - (zb_t + zeta * dt_hl)
                         - sum(uvel_m[di] * (DERIV[xd](z_bot)
                                             + zeta * DERIV[xd](h_l))
                               for di, xd in enumerate(horiz)))
            omega_closed = (
                G_bot / rl
                - zeta * (dt_hl + sum(DERIV[xd](qfm[di][0])
                                      for di, xd in enumerate(horiz)))
                - sum(DERIV[xd](qfm[di][j]) * inner_basis.eval_psi(j, zeta)
                      for di, xd in enumerate(horiz) for j in range(1, Nu + 1)))
            R_om = sp.expand(omega_closed - omega_def)
            # ml_vam.py could assert R_om(0) == 0 because ŵ_top was CLOSED by the
            # bottom KBC, making the two ω̃ forms identical there.  With ŵ_top a
            # state the bottom-interface mismatch is exactly the new constraint
            # (both forms still agree ON the manifold) — assert THAT, which is a
            # sharper statement than the old identity, not a relaxed one.
            assert sp.simplify(
                sp.expand(h_l * R_om.subs(zeta, 0)) + R_kbc_c) == 0, (
                f"layer {ell}: ω̃ bottom-interface residual is not the "
                f"kinematic constraint R_kbc")
            for k in range(1, Nu + 1):
                dphi = sp.diff(phis[k], zeta)
                for di, mn in enumerate(MOM):
                    getattr(ml, mn)[k].expr = sp.expand(
                        getattr(ml, mn)[k].expr
                        - _zint01(R_om * uvel_m[di] * dphi) / mus[k])
            for k in range(1, nmode["momentum_z"]):
                dphi = sp.diff(phis[k], zeta)
                ml.momentum_z[k].expr = sp.expand(
                    ml.momentum_z[k].expr - _zint01(R_om * wt_m * dphi) / mus[k])

            # DERIVED conservative flux per (row, direction) for the §6c
            # regroup (no hardcoding): F = (2k+1) h_l ∫ φ_k · flux dζ on this
            # layer's modal ansatz.  Horizontal: u_d·u_e + p/ρ (diagonal).
            # Vertical: the FULL advection w·u_e with w = ``wt_m`` (incl. the
            # bottom-KBC/interface top mode), exactly as VAM's momentum_z flux
            # uses ``wt_m``.  The top mode's ∂_t (interface motion) and G
            # (interface mass flux) pieces are rendered spatial by the §6c
            # ∂_t h → dth_glob / G_sol substitutions, so ∂_e(F) is a clean
            # conservative flux — keeping only w_bulk left that part in the NCP
            # slot and broke the MLVAM(1)==VAM reduction on r_k rows.
            p_zeta = (sum(P_heads[ell - 1](kp, t, *horiz) * phis[kp]
                          for kp in range(Nu + 2)) + p_top_mode * phis[top_p])
            flux_F = {}
            for xd in horiz:
                for k2 in range(1, nmode[f"momentum_{CN[xd]}"]):
                    di = list(horiz).index(xd)
                    for e, xe in enumerate(horiz):
                        integ = uvel_m[di] * uvel_m[e]
                        if di == e:
                            integ = integ + p_zeta / rl
                        flux_F[(f"momentum_{CN[xd]}", k2, xe)] = (
                            h_l / inner_basis.gram(k2, k2)
                            * _zint01(phis[k2] * integ))
            for k2 in range(1, nmode["momentum_z"]):
                for e, xe in enumerate(horiz):
                    flux_F[("momentum_z", k2, xe)] = (
                        h_l / inner_basis.gram(k2, k2)
                        * _zint01(phis[k2] * (wt_m * uvel_m[e])))

            cont = sp.expand(ml.mass[0].expr)
            # Nu+1 divergence rows; the ONE kinematic row R_kbc_c is returned
            # separately because it carries a DIFFERENT wet/dry scaling (already
            # ×h_ℓ above vs ×l_ℓ·h² applied to these in the assembly).  Together
            # they are the Nu+2 constraints matching the Nu+2 free pressure
            # modes (LDNH square DAE).
            constraints = [sp.expand(ml.mass[k].expr) for k in range(1, Nu + 2)]
            momd = {xd: [sp.expand(getattr(ml, f"momentum_{CN[xd]}")[k].expr)
                         for k in range(nmode[f"momentum_{CN[xd]}"])]
                    for xd in horiz}
            momz = [sp.expand(ml.momentum_z[k].expr)
                    for k in range(nmode["momentum_z"])]
            # Mark this layer's hydrostatic self-pressure g·h_ℓ²/2 (folded on the
            # mode-0 momentum flux) as HydrostaticPressure at its FOLD SITE — here
            # h_l is a DISTINCT function per layer, so the mark is unambiguous; a
            # builder-side .subs on the assembled column (h_ℓ=l_ℓ·h) cannot
            # separate the bottom layer's g·l₁²h²/2 from the same monomial inside
            # the top layer's expanded g·(1−l₁)²h²/2.  Routes the self-weight to
            # the well-balanced P slot (VAM parity at n=1; lake-at-rest over a
            # bump for n≥2); the inter-layer g·h_ℓ·∂_x h_{ℓ'} and bed g·h_ℓ·∂_x b
            # stay in the NCP.
            hyd = ml.parameters.g * h_l ** 2 / 2
            momd = {xd: [row.subs({hyd: HydrostaticPressure(hyd)}) for row in rows]
                    for xd, rows in momd.items()}
            # per-layer vertical reconstruction profiles (LOCAL ζ), in the
            # conserved layer state (qfm=q_ℓ, rfm=r_ℓ, P_ℓ) + h_l + G_bot — the
            # SAME modal columns the §6c flux uses: ``uvel_m`` horizontal
            # velocity, ``wt_m`` the full w (ALL modes free — LDNH),
            # ``p_zeta`` the non-hydrostatic pressure with the cascaded top mode.
            # Assembled into the global piecewise reconstruction below (frac /
            # ω̃ / par substitutions applied there, exactly as the flux).
            return (cont, constraints, momd, momz, flux_F,
                    list(uvel_m), wt_m, p_zeta, R_kbc_c, p_bot_trace_full)

        # layers derived TOP-DOWN so the pressure-trace cascade resolves
        layer_eqs = {}
        p_trace = sp.S.Zero                      # surface: p_N(1) = 0
        for ell in range(N, 0, -1):
            *eqs_l, p_trace = derive_layer(ell, p_trace)
            layer_eqs[ell] = tuple(eqs_l)
        _layer_eqs_debug = layer_eqs             # pre-assembly rows (attached to m below)

        # ── Hörnschemeyer closure + per-direction shared u* transfer ──
        ht = sp.Function("h", positive=True)(t, *horiz)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, sp.S.One - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = {ell: {xd: [sp.Function(qname(xd, ell), real=True)(k, t, *horiz)
                            for k in range(Nu + 1)] for xd in horiz}
                 for ell in range(1, N + 1)}
        # Nu+2 FREE w modes per layer (ŵ_top is a state, LDNH) and Nu+2 FREE
        # pressure modes (p ∈ P_{Nu+2}, top mode closed by the trace cascade).
        n_w = Nu + 2
        n_p = Nu + 2
        r_mod = [[sp.Function(f"r_{ell}", real=True)(k, t, *horiz)
                  for k in range(n_w)] for ell in range(1, N + 1)]
        P_mod = [[P_heads[ell - 1](j, t, *horiz) for j in range(n_p)]
                 for ell in range(1, N + 1)]

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1)).subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        G_sol = {}
        for a in range(1, N):
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1)).subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        def _trace(ell, side, xd):
            sgn = lambda i: inner_basis.eval(i, side)
            return (sum(sgn(i) * q_mod[ell][xd][i] for i in range(Nu + 1))
                    / (l_all[ell - 1] * ht))

        from zoomy_core.model.models.closures import interface_closure
        iface = interface_closure(self.closures)

        def _ustar(a, xd):
            below, above = _trace(a, 1, xd), _trace(a + 1, 0, xd)
            if iface is not None:
                # pass the REAL internal-interface mass flux G_α (not 0): an
                # UpwindInterface donor decision (Hörnschemeyer Eq. 9) is by the
                # sign of G — feeding 0 silently degenerated it to always-below.
                # MeanInterface ignores G, so the default path is unchanged.
                # Matches MLSME._ustar so the ML-VAM→ML-SME (P=0) reduction
                # commutes under either interface scheme.
                return iface.expression(below, above, G_sol[Gf[a]])
            return (below + above) / 2

        # vertical-z placeholder so Model.horizontal = (x[, y]) (see ml_sme)
        m = DModel(coords=(t, *horiz, z), parameters=values)
        # declare closure parameters (tau_y, eps_reg, …) on the assembled model
        # so they bind at codegen even without an explicit user value (REQ-160).
        for c in (self.closures or []):
            c.register(m)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            constraints, momd, momz = (layer_eqs[ell][1], layer_eqs[ell][2],
                                       layer_eqs[ell][3])
            for xd in horiz:
                for k in range(Nu + 1):
                    # frac (h_ℓ→l_ℓ·h) then evaluate ONLY the fraction's ∂_t
                    # (so ∂_t(l_ℓ·h)→l_ℓ·∂_t h feeds the global-mass sub) —
                    # keep ∂_x(F) folded so the momentum flux stays conservative.
                    row = evaluate_time_derivatives(momd[xd][k].subs(frac), t)
                    row = sp.expand(row).subs(sp.Derivative(ht, t), dth_glob)
                    for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                        if 1 <= a <= N - 1:
                            phik = inner_basis.eval(k, side)
                            # interface transfer is a projected coefficient
                            # ⟨δ_interface·u*, φ_k⟩ = φ_k(side)·(…); the row was
                            # already mass-inverted (:297), so it carries the
                            # SAME 1/μ_k Gram-norm every other post-inversion
                            # coefficient does (ω̃ :336/:338, flux :357) — read
                            # off the basis, not hardcoded (REQ-79).  Without it
                            # modes k≥1 are off by (2k+1); μ_0=1 so mode 0 is
                            # unchanged (cid 170).
                            row = row + (sgn * phik
                                         * (_ustar(a, xd) - _trace(ell, side, xd))
                                         * Gf[a] / rho_s / mus[k])
                    row = sp.expand(row).subs(G_sol)
                    # CN[x]="x" → momentum_x_ℓ_k in 1-D (byte-identical names)
                    m.add_equation(f"momentum_{CN[xd]}_{ell}_{k}",
                                   sp.expand(evaluate_time_derivatives(
                                       row.subs(par), t)))
            for k in range(n_w):                          # vertical (no swap)
                row = evaluate_time_derivatives(momz[k].subs(frac), t)
                row = sp.expand(row).subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
                m.add_equation(f"momentum_z_{ell}_{k}",
                               sp.expand(evaluate_time_derivatives(
                                   row.subs(par), t)))
            for j, cst in enumerate(constraints):
                cst = sp.expand(evaluate_time_derivatives(cst.subs(frac), t))
                cst = sp.expand(cst.subs(sp.Derivative(ht, t), dth_glob)).subs(G_sol)
                cst = sp.expand(evaluate_time_derivatives(cst.subs(par), t))
                # ── 86BQYCRP Eq. 19 (THE wet/dry change vs ml_vam.py) ────────
                # The stock model keeps the σ-mass-flux compound (1/h)·∂_x(h·q_k)
                # FOLDED.  That 1/h sits INSIDE an unevaluated Derivative, so the
                # emitted elliptic operator carries 1/h and degenerates to
                # SINGULAR as h→0 at a thinning front (measured: exact-LU dies
                # ``LinAlgError: Singular matrix``).  Scaling the ASSEMBLED
                # operator cannot fix that (det(D·A)=det(D)·det(A), and the 1/h
                # is inside the derivative anyway) — verified, it is a no-op.
                #
                # Escalante–Fernández-Nieto–Morales de Luna–Castro (2019) write
                # the incompressibility in DISCHARGE form (their Eq. 19: ∂_x q,
                # never ∂_x(q/h)), obtained by multiplying each layer constraint
                # by l_ℓ·h².  So: DISTRIBUTE the derivatives first (product rule
                # ⇒ ∂_x(q/h) → ∂_x q/h − q·∂_x h/h²), THEN multiply by l_ℓ·h² so
                # every 1/h cancels ANALYTICALLY — leaving a polynomial-in-h row.
                # The elliptic block then degenerates to an INVERTIBLE O(1)
                # algebraic system with RHS→0 as h→0, i.e. P→0 automatically at
                # wet/dry fronts with NO threshold (their App. D.3).
                # Same constraint manifold wherever h>0 (positive factor), so the
                # physics is unchanged; only the emitted form differs.  Verified
                # symbolically: no Pow(h, negative) survives anywhere in the row.
                cst = sp.sympify(cst).doit()
                cst = sp.expand(cst).subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
                fac = sp.sympify(l_all[ell - 1] * ht ** 2).subs(par)
                cst = sp.cancel(sp.expand(cst * fac))
                m.add_equation(f"constraint_{ell}_{j}", sp.expand(cst))
            # ── the LDNH kinematic constraint (the +1 row STEP-1 measured) ──
            # Its multiplier is the Nu+2'nd pressure mode, i.e. the bottom /
            # interface pressure q_b of Escalante 2021 eq. (10).  Already scaled
            # by h_ℓ = l_ℓ·h inside ``derive_layer`` — the kinematic analogue of
            # the Eq.19 discharge form (ONE power of h clears its single 1/h;
            # a second power would be a spurious extra degeneracy at h→0).
            kbc = layer_eqs[ell][8]
            kbc = sp.expand(evaluate_time_derivatives(kbc.subs(frac), t))
            kbc = sp.expand(kbc.subs(sp.Derivative(ht, t), dth_glob)).subs(G_sol)
            kbc = sp.sympify(sp.expand(
                evaluate_time_derivatives(kbc.subs(par), t))).doit()
            kbc = sp.expand(kbc).subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
            m.add_equation(f"constraint_kbc_{ell}", sp.cancel(sp.expand(kbc)))

        m.apply(InvertMassMatrix())

        # ── 6c — conservative regroup, DERIVED from the per-layer modal flux
        # (no hardcoding, any dim/Nu).  (1) Wrap the advective + pressure flux
        # F_{row,e} = (2k+1) h ∫ φ_k flux dζ as a conservative ∂_e compound
        # (bed-slope-free part); (2) a generalized second-derivative absorption
        # routes the bed-curvature / G-geometry ∂²-terms (any spatial direction,
        # incl. ∂_x∂_y) into the diffusion operator.
        space = list(horiz)
        for ell in range(1, N + 1):
            flux_F = layer_eqs[ell][4]
            by_row = {}
            for (rowbase, k2, xe), F in flux_F.items():
                by_row.setdefault((rowbase, k2), []).append((xe, F))
            for (rowbase, k2), flist in by_row.items():
                name = f"{rowbase}_{ell}_{k2}"
                if name not in m._equations:
                    continue
                eq = getattr(m, name)
                ex = sp.expand(sp.sympify(eq.expr).doit())
                for xe, F in flist:
                    Fa = sp.expand(sp.sympify(F).subs(frac).doit()
                                   .subs(sp.Derivative(ht, t), dth_glob)
                                   .subs(G_sol).subs(par))
                    # bed-slope-bearing flux stays nonconservative; the rest is
                    # the conservative advective/pressure ∂_e flux.
                    F_bed = sum((tm for tm in sp.Add.make_args(Fa) if tm.has(b)),
                                sp.S.Zero)
                    F_free = Fa - F_bed
                    if F_free != 0:
                        dF = sp.Derivative(F_free, xe)
                        ex = sp.expand(ex - sp.expand(dF.doit())) + dF
                eq.expr = ex
        # (base, #free modes) — momentum_z carries Nu+2 rows because ŵ_top is a
        # state; the horizontal rows keep Nu+1.
        bases = ([(f"momentum_{CN[xd]}_{ell}", Nu + 1)
                  for ell in range(1, N + 1) for xd in horiz]
                 + [(f"momentum_z_{ell}", n_w) for ell in range(1, N + 1)])
        for base, n_base in bases:
            # skip mode 0 (mirror VAM §6d): it is never doited by part-1, so it
            # carries no bare ∂²-term to absorb and its folded conservative flux
            # must survive.  NO blanket .doit() here — part-1 already surfaced the
            # bare ∂²b on modes k≥1; a doit would re-distribute the just-folded
            # ∂_x(F) back into the NCP slot.
            for k in range(1, n_base):
                name = f"{base}_{k}"
                if name not in m._equations:
                    continue
                eq = getattr(m, name)
                e = sp.expand(sp.sympify(eq.expr))
                for e_dir in space:
                    F_d = sp.S.Zero
                    for a in list(e.atoms(sp.Derivative)):
                        vs = list(a.variables)
                        if len(vs) == 2 and vs[0] == e_dir and vs[1] in space:
                            F_d = F_d + e.coeff(a) * sp.Derivative(a.expr, vs[1])
                    if F_d != 0:
                        dF = sp.Derivative(sp.expand(F_d), e_dir)
                        e = sp.expand(e - sp.expand(dF.doit())) + dF
                eq.expr = e

        # ── interpolate_to_3d + project_from_3d, PIECEWISE over the moving
        # layers (same canonical operators as VAM / ML-SME).  Layer ℓ spans the
        # global ζ ∈ [c_{ℓ-1}, c_ℓ] (c_ℓ = Σ_{j≤ℓ} l_j); within it the LOCAL
        # column ζ_loc = (ζ − c_{ℓ-1})/l_ℓ carries the per-layer modal profiles
        # returned by ``derive_layer``.  Each profile gets the SAME assembly
        # substitutions the §6c flux did (frac h_ℓ=l_ℓ·h, ∂_t h → global mass,
        # interface fluxes G, parameters).  Field order [b, h, u(, v), w, p];
        # slot 5 is the TOTAL pressure ρ g h (1−ζ) + the non-hydrostatic part
        # (Escalante split uses the GLOBAL free surface η = b + H).
        cum = [sp.S.Zero]
        for lf in l_all:
            cum.append(cum[-1] + lf)

        def _xform(prof, ell):
            """Local-ζ layer profile → global-ζ, with the assembly subs."""
            lf, c0 = l_all[ell - 1], cum[ell - 1]
            e = sp.sympify(prof).subs(zeta, (zeta - c0) / lf)
            e = sp.expand(e.subs(frac).doit())
            e = e.subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
            return e

        def _piece(prof_of_ell):
            pieces = []
            for ell in range(1, N + 1):
                val = _xform(prof_of_ell(ell), ell)
                cond = (zeta <= cum[ell]) if ell < N else True
                pieces.append((val, cond))
            return sp.Piecewise(*pieces).subs(par)

        interp = {0: b, 1: ht}
        for di, xd in enumerate(horiz):
            interp[2 + di] = _piece(lambda ell, di=di: layer_eqs[ell][5][di])
        interp[4] = _piece(lambda ell: layer_eqs[ell][6])
        interp[5] = (m.parameters.rho * m.parameters.g * ht * (1 - zeta)
                     + _piece(lambda ell: layer_eqs[ell][7]))
        m.interpolate_rows = interp

        # inverse: per-layer Integral-FREE fixed-node Galerkin reduction (see
        # ML-SME).  Layer ℓ samples its profile at N_z LOCAL nodes t∈[0,1] mapped
        # to the global ζ = c0 + l_ℓ·t.  Conserved moments q_ℓ, r_ℓ carry the
        # physical layer-height factor h·l_ℓ; the pressure modes P_ℓ are plain
        # modal coefficients of the NON-hydrostatic column (norm 1, total-pressure
        # sample with its hydrostatic ρ g h (1−ζ) removed).
        # level = top_p = Nu+2 spans EXACTLY the per-layer pressure column
        # p ∈ P_{Nu+2} that ``interpolate_rows`` emits, so the discrete-Gram
        # least-squares fit inverts it to round-off (and a fortiori the lower-
        # degree u/w columns).  Do NOT raise it: the rows come from the FULL
        # G⁻¹, so a larger span would perturb the q/r rows too.
        proj_legendre = Legendre_shifted(level=top_p)
        N_z = 33
        loc = [float(j) / (N_z - 1) for j in range(N_z)]
        wq = [1.0 / (N_z - 1)] * N_z
        wq[0] *= 0.5; wq[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        rho_p, g_p = m.parameters.rho, m.parameters.g
        proj = {b: P3["b"], ht: P3["h"]}
        for xd in horiz:
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            for ell in range(1, N + 1):
                lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
                samples = [P3vel(c0 + lf * tt) for tt in loc]
                rows = proj_legendre.projection_rows(
                    loc, wq, samples, norm=lambda _k, _lf=lf: P3["h"] * _lf)
                for k in range(Nu + 1):
                    proj[q_mod[ell][xd][k]] = rows[k]
        P3w = sp.Function("P3_w", real=True)
        for ell in range(1, N + 1):
            lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
            samples = [P3w(c0 + lf * tt) for tt in loc]
            rows = proj_legendre.projection_rows(
                loc, wq, samples, norm=lambda _k, _lf=lf: P3["h"] * _lf)
            for k in range(n_w):
                proj[r_mod[ell - 1][k]] = rows[k]
        P3p = sp.Function("P3_p", real=True)
        for ell in range(1, N + 1):
            lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
            samples = [P3p(c0 + lf * tt) - rho_p * g_p * P3["h"] * (1 - (c0 + lf * tt))
                       for tt in loc]
            rows = proj_legendre.projection_rows(loc, wq, samples, norm=None)
            for k in range(n_p):
                proj[P_mod[ell - 1][k]] = rows[k]
        m.project_rows = proj

        # model-derived free-slip wall — the SAME geometric statement as VAM /
        # ML-SME: reflect only the NORMAL component of each (layer, mode)
        # horizontal momentum vector.  The vertical r and pressure P modes are
        # scalars under a horizontal wall normal and extrapolate.  ML-VAM had
        # NO wall registration at all: FromModel(definition="wall") raised.
        register_free_slip_wall(
            m, ([q_mod[ell][xd][k] for xd in horiz]
                for ell in range(1, N + 1)
                for k in range(Nu + 1)))

        m.bed = b
        m.ht = ht
        m.q_flat = [q_mod[ell][xd][k]
                    for ell in range(1, N + 1)
                    for xd in horiz for k in range(Nu + 1)]
        m.r_flat = [r for layer in r_mod for r in layer]
        m.P_flat = [p for layer in P_mod for p in layer]
        m.layer_eqs_debug = _layer_eqs_debug
        return m

    # Built via ``SystemModel.from_model(MLVAM(...))`` (REQ-143); see
    # ``zoomy_core.systemmodel.model_builders.build_mlvam``.
    _system_model_kind = "mlvam"

    def chorin_split(self, dt=None, *, system_model=None):
        """Structural Chorin split (predictor / pressure / corrector)."""
        from zoomy_core.model.splitter import split_for_pressure_structural
        sm = system_model if system_model is not None \
            else SystemModel.from_model(self)
        if dt is None:
            dt = sp.Symbol("dt", positive=True)
        P_syms = [s for s in sm.state if str(s).startswith("P_")]
        return split_for_pressure_structural(sm, P_syms, dt)
