"""Stay-3D σ-coordinate model — the general 3-D mass+momentum balance, σ-mapped,
with the divergence constraint integrated against the kinematic BCs to give the
EXACT height-evolution equation, but WITHOUT any velocity ansatz / moment
projection.  The flow stays 3-D; the single unclosed object — the vertical
integral — is substituted into an auxiliary variable whose per-step value is
supplied by a backend "integrate over z" (the same machinery `project_from_3d`
uses).

Verified derivation (`diff = 0` against the framework, 2026-06-13):

* height equation (φ₀-moment of continuity against the two KBCs):

```
      ∂_t h + ∂_x(h·U) = 0 ,        U = ∫₀¹ ũ dζ           (depth-mean velocity)

  IMPORTANT: ``U`` is the depth-MEAN (the ζ-independent ``h`` factors out of the
  column integral by itself, so the height flux is ``F[h] = U·h``).  The integral
  MUST be substituted into ``U`` BEFORE ``Simplify`` — otherwise a spurious
  ``+U·∂_x h`` term survives (``Simplify`` illegally pulls ``h`` inside
  ``∂_x(∫ũ dζ)``).
```

* the full system, in CONSERVATIVE σ-form, is a clean conservation law over
  `(x, ζ)` with the contravariant (σ-relative) vertical velocity
  `h·ω = w̃ − ∂_t z − ũ ∂_x z`, `z = b + h ζ`:

  ```
      ∂_t h     + ∂_x(h u)            + ∂_ζ(h ω)                            = 0
      ∂_t(h u)  + ∂_x(h u² + g h²/2)  + ∂_ζ(h u ω) + g h ∂_x b − e_x g h
                                                    − (1/ρ) ∂_ζ τ̃_xz       = 0
  ```

  `ω(0)=ω(1)=0` ⟺ the bed / free-surface KBCs, so the σ metric terms are
  ABSORBED into `∂_ζ(h q ω)` (zero boundary flux) — ζ is simply a third FLUX
  direction, no ALE/metric source class is needed.  `U = ∫₀¹ u dζ` (full
  column) drives the height flux; `ω` is a RUNNING column integral diagnosed
  from continuity — both are the same per-backend integrate-over-z (full vs
  running).

The aux is introduced with EXISTING machinery — no `promote_to_aux` op:

```
    m.add_equation("U", sp.Eq(U, ∫ũ dζ), group="aux")   # records the definition
    m.mass.apply({∫ũ dζ: U})                            # U auto-lands in Qaux
```

`Sigma3D(...).derive_model()` builds the declarative model;
`SystemModel.from_model(Sigma3D(...))` returns the height-reduced
`[b, h] + U` SystemModel (the
3-D momentum is stashed on `self` for the forthcoming 3-D extraction, which
needs the vertical added to the extractor's `space`).
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ
from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.equations import Stress
from zoomy_core.misc.options import OptionParam

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class Sigma3D(BaseModel):
    """3-D mass+momentum → σ-map → column-integrated height equation, keeping the
    velocity profile UNCLOSED as a per-backend vertical integral.

    State handed to the SystemModel (v1): `[b, h]`, with the depth-mean
    velocity `U` exposed as an auxiliary defined by `U = ∫₀¹ ũ dζ` (and the
    3-D field `ũ` carried as the aux it is integrated from).  The 3-D momentum
    advance becomes SystemModel state once the extractor treats ζ as a flux
    direction (attached to the derivation as `m.momentum_sigma_x` / `m.omega_def`).
    """

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension INCLUDING the vertical: 2 → coords (t,x,z), one "
        "horizontal (U=U_x); 3 → coords (t,x,y,z), two horizontals.  Only dim=2 "
        "is wired today; dim=3 needs the SME HAXES-loop treatment."))
    stress = OptionParam(Stress, default=None, extra=[None], doc=(
        "In-plane stress: `None` keeps tau_xx as an open (unclosed) field, the full-stress "
        "model; `Stress.shallow` drops it (`moment_scaling`).  The vertical balance is "
        "hydrostatic either way."))
    closures = param.List(default=[], doc=(
        "Stress closures applied to the RESOLVED 3-D σ-momentum (closures.py): the "
        "bulk (Newtonian) closure becomes the vertical viscous diffusion "
        "∂_ζ(ρν/h ∂_ζ ũ); the bottom (NavierSlip/RoughWall) closure sets the bed "
        "friction traction τ̃_xz(0); StressFree the free surface.  Empty → defaults "
        "to [Newtonian(), NavierSlip(), StressFree()] (slip + viscous; the friction "
        "magnitude is set by the parameters nu / lambda_s)."))

    def derive_model(self):
        if int(self.dimension) != 2:
            raise NotImplementedError(
                "Sigma3D: only dimension=2 (t,x,z) is wired today.")
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (Mass, Momentum, moment_scaling)

        m = DModel(coords=(t, x, z), parameters=values)
        g, rho, e_x = m.parameters.g, m.parameters.rho, m.parameters.e_x
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)
        # Free-surface hydrostatic pressure flux (g·h²/2): recomputed in
        # ``system_model`` from m and tagged there (not stashed on self).

        # 1 — general 3-D mass + momentum, FULL deviatoric stress tensor.
        # Unlike the shallow-moment models, Sigma3D does NOT drop the in-plane
        # stress by default: τ̃_xx survives as an open field (the geometrically-exact
        # full-stress model); ``stress=Stress.shallow`` drops it (``moment_scaling``,
        # recovering the shallow form).
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))                 # ∂_t b = 0
        m.add_equation(Mass(m))                          # mints u, w
        mom = Momentum(m); m.add_equation(mom)           # mints p (+ τ closures)
        uvel, w, p = mom.uvel, mom.w, mom.p
        m.add_equation("kbc_top", KinematicBC(w=w, u=uvel[0], interface=b + h))
        m.add_equation("kbc_bot", KinematicBC(w=w, u=uvel[0], interface=b))

        if self.stress is not None and Stress(self.stress) == Stress.full:
            raise NotImplementedError("Sigma3D: stress=Stress.full is not wired (SME, VAM, "
                                      "MLSME, MLVAM are); None keeps tau_xx open")
        if self.stress is not None:
            moment_scaling(m, mom, self.stress)

        # 2 — hydrostatic elimination of p (physical z): no inertia, no stress in the
        # vertical row
        mz = m.momentum.z
        if self.stress is None:
            mz.expr = mz.expr - mom.vertical_term
        mz.apply({d.t(w): 0, d.z(w * w): 0, d.x(uvel[0] * w): 0})
        mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
        mz.apply({p.subs(z, b + h): 0})                  # p(surface) = 0
        m.momentum.x.apply(mz.solve_for(p)); mz.remove()
        m.momentum.x.apply(Simplify())

        # 3 — σ-map z = b + h·ζ (decorates u→ũ, w→w̃, τ→τ̃; σ-maps the KBCs).
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        # 3b — CLOSE the material/stress on the resolved σ-momentum.  Bulk
        # (Newtonian) τ̃_xz = ρν/h ∂_ζ ũ  → the vertical viscous diffusion
        # ∂_ζ(ρν/h ∂_ζ ũ); bottom (NavierSlip/RoughWall) → the bed friction
        # traction; StressFree → the free surface.  Same machinery as SME.
        from zoomy_core.model.models.closures import (
            Newtonian, NavierSlip, StressFree, apply_stress_closures)
        from zoomy_core.model.models.material import ClosureState
        clos = self.closures or [Newtonian(), NavierSlip(), StressFree()]
        m.closures_resolved = clos          # for the closure → ζ-face-BC reduction

        def _cstate(at, *, alias=None, btag=None):
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=[x])
        axes = [{"mx": m.momentum.x, "tau": m.functions.tau_xz, "velname": "u"}]
        apply_stress_closures(clos, m, axes, _cstate, [x])
        m.momentum.x.apply(Simplify())

        def _head(name):
            pool = m.momentum.x.expr.atoms(sp.Function)
            return next(a.func for a in pool if str(a.func) == name)
        ut = _head(r"\tilde{u}")(t, x, zeta)
        wt = _head(r"\tilde{w}")(t, x, zeta)

        # 3c — the bed/surface conditions are NOT folded into the PDE (the model
        # keeps a real ζ-discretization): bed = no-penetration ω(0)=0 + Navier
        # slip τ̃·t=λ_s u_t, top = ω(1)=0 + stress-free τ̃·t=0.  They are honored
        # as ordinary, FRAME-AWARE solver boundary conditions at the ζ=0 / ζ=1
        # faces (physical {n,t}; sourced from the closures' .traction()), applied
        # at solve time — never the old dead ``_vertical_bcs`` stash.

        # 3-D conservative-σ ingredients for the conservative momentum below
        # (verified: σ-momentum(×h) ≡ ∂_t(hu)+∂_x(hu²+gh²/2)+∂_ζ(h u ω)+ghb_x−e_x g h−τ̃_z/ρ).
        zc = b + h * zeta
        momentum_sigma_x = sp.expand((h * m.momentum.x.expr).doit())
        # The surviving in-plane deviatoric stress τ̃_xx (None/0 if a horizontal
        # closure dropped it) — fed into the conservative momentum's in-plane
        # divergence term below so the verify holds in BOTH the full-stress and
        # the shallow (τ̃_xx→0) cases.
        _inplane = [a for a in momentum_sigma_x.atoms(sp.Function)
                    if "tau_xx" in str(a.func)]
        tau_xx_t = _inplane[0] if _inplane else sp.S.Zero
        # 3-D conservative-σ ingredients, attached to m for the extractor /
        # solver (off the model surface; survive the derivation cache).
        m.omega_def = wt - sp.Derivative(zc, t) - ut * sp.Derivative(zc, x)   # = h·ω
        m.momentum_sigma_x = momentum_sigma_x
        m.heads_3d = {"u": ut.func, "w": wt.func}

        # 4 — height equation: φ₀-moment of continuity against the two KBCs.
        m.mass.apply(Multiply(h))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1)))
        m.mass.apply(ResolveIntegral())
        ints = list(m.mass.expr.atoms(sp.Integral))
        assert len(ints) == 1, f"expected exactly one vertical integral, got {ints}"
        vint = ints[0]                                   # = ∫₀¹ ũ dζ
        U = sp.Function("U", real=True)(t, x)            # depth-mean velocity aux
        # introduce the aux with existing machinery (NO promote_to_aux op):
        m.add_equation("U", sp.Eq(U, vint), group="aux")  # record the definition
        m.mass.apply({vint: U})                           # substitute → U ∈ Qaux
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0})
        m.mass.apply(Simplify())
        # internal self-check: the height eq is EXACTLY ∂_t h + ∂_x(h·U).
        ref = sp.Derivative(h, t).doit() + sp.diff(h * U, x)
        assert sp.simplify((m.mass.expr - ref).doit()) == 0, (
            f"height equation mismatch: {m.mass.expr} != ∂_t h + ∂_x(h U)")
        m.U_def = vint                                    # U = ∫₀¹ ũ dζ (documented stash)

        # 5 — STAY 3-D: the conserved momentum mom = h·ũ and the contravariant
        # vertical velocity ω, both (t, x, ζ).  u = mom/h with RAW 1/h — the KP
        # desingularization of 1/h is the NumericalSystemModel's regularization,
        # never the analytical model.
        nu = m.parameters.nu
        mom = sp.Function("mom", real=True)(t, x, zeta)        # conserved h·ũ
        omega = sp.Function("omega", real=True)(t, x, zeta)    # σ vertical velocity
        u = mom / h
        # The vertical viscous term is taken FROM THE BULK CLOSURE, not assumed.
        # A bulk closure returns τ̃_xz = ρ·ν_eff(ζ)·∂_z ũ, and under the σ-map
        # ∂_z = (1/h)∂_ζ, so the σ-momentum carries −∂_ζ(τ̃_xz/ρ).  Writing it
        # this way keeps a ζ-DEPENDENT eddy viscosity (Elder: ν_t = κ u_⋆ h ζ(1−ζ))
        # exactly as the derivation produced it; hard-writing ν/h ∂_ζ u would
        # silently assume a constant scalar viscosity and the verify below would
        # (correctly) fail.  Newtonian reduces to the old form identically.
        bulk_clos = [c for c in clos if getattr(c, "closes", None) == "bulk"]
        if bulk_clos:
            tau_bulk = sum((c.expression(_cstate(at=zeta)) for c in bulk_clos),
                           sp.S.Zero)
            # rewrite in the conserved variable, exactly as the verify below
            # rewrites the derived momentum (ũ = mom/h).
            viscous = (-d.zeta(tau_bulk / rho)).subs(ut, mom / h)
        else:                                   # no bulk closure ⇒ inviscid
            viscous = sp.S.Zero
        # conservative-σ momentum (raw 1/h; viscous term from the bulk closure).
        # The IN-PLANE deviatoric stress enters as the σ-conservative divergence
        # of τ̃_xx: physical −(h/ρ)∂_x τ_xx maps (via ∂_x|_z = ∂_x|_ζ − (∂_x zc/h)∂_ζ)
        # to −(1/ρ)[h ∂_x τ̃_xx − ∂_ζ τ̃_xx · ∂_x zc].  τ̃_xx=0 (shallow closure)
        # ⇒ this term vanishes and we recover the shallow-moment momentum exactly.
        inplane = -(h * d.x(tau_xx_t) - d.zeta(tau_xx_t) * d.x(zc)) / rho
        cons_mom = (d.t(mom) + d.x(mom * u + g * h**2 / 2) + d.zeta(mom * omega)
                    + g * h * d.x(b) - e_x * g * h + viscous
                    + inplane)
        # VERIFY the clean conservative form ≡ the derived σ-momentum (ũ=mom/h,
        # w̃=h·ω+∂_t z+ũ ∂_x z) — a faithful derivation, not a hand-write.
        wt_sub = h * omega + sp.Derivative(zc, t) + (mom / h) * sp.Derivative(zc, x)
        derived = momentum_sigma_x.subs(wt, wt_sub).subs(ut, mom / h)
        assert sp.simplify(sp.expand((cons_mom - derived).doit())) == 0, (
            "stay-3D conservative-σ momentum mismatch vs the derived σ-momentum")

        # 6 — swap the σ-momentum for its conservative form; mom becomes state.
        m.remove("momentum")
        m.add_equation("momentum", cons_mom)
        m.declare_state(h, mom)

        # 7 — rewrite the integral auxes in the conserved variable u = mom/h:
        #   U = ∫₀¹ u dζ  (full depth-mean);  ω diagnosed from continuity as the
        #   RUNNING integral  h·ω = ζ ∂_x(hU) − ∂_x(h ∫₀^ζ u dζ').  Both integrals
        #   are supplied per-backend by the same integrate-over-z function.
        m.apply({ut: mom / h})                            # ũ → mom/h in the U def
        zp = sp.Symbol("zeta_p", real=True)
        W_run = sp.Integral(u.subs(zeta, zp), (zp, 0, zeta))   # ∫₀^ζ u dζ'
        omega_run = (zeta * d.x(h * U) - d.x(h * W_run)) / h
        m.add_equation("omega", sp.Eq(omega, omega_run), group="aux")

        return m

    # The full 3-D conservative-σ operator system over ``(x, ζ)`` (state
    # ``[b, h, mom]``; ζ a genuine flux direction) is built via
    # ``SystemModel.from_model(Sigma3D(...))`` (REQ-143); the builder in
    # ``zoomy_core.systemmodel.model_builders`` tags the hydrostatic flux and
    # runs the closure → ζ-face-BC reduction via ``_vertical_face_bcs``.
    _system_model_kind = "sigma3d"

    def _vertical_face_bcs(self, sm):
        """Translate the bed/surface CLOSURES into ζ-face solver BCs (the user
        never writes σ-specific strings).  Newtonian bulk ⇒ ζζ viscous diffusion
        (already in the PDE); NavierSlip bed ⇒ a Robin viscous-flux BC
        `∂_ζ mom = (h²/ν)·λ_s·u_t` (small-slope/flat: `u_t = mom/h` →
        `λ_s·h·mom/ν`; the slope-aware `u_t` comes from
        `ClosureState.get_normal_tangential`); StressFree surface ⇒ zero viscous
        flux (`gradient=0`).  Delivered through the generalized `Flux`."""
        from zoomy_core.model.models.closures import (
            NavierSlip, RoughWall, StressFree)
        from zoomy_core.model.boundary_conditions import Flux
        names = [str(s) for s in sm.state]
        if "mom" not in names:
            return []
        hS, momS = sm.state[names.index("h")], sm.state[names.index("mom")]
        nu = sm.parameters.nu
        clos = getattr(self.derivation, "closures_resolved", [])
        out = []
        for c in clos:
            kind = getattr(c, "closes", None)
            if kind == "bottom":
                if isinstance(c, NavierSlip):
                    lam = sm.parameters.lambda_s
                    # Navier slip: the OUTWARD-normal σ-gradient of mom at the bed
                    # (ζ=0 face, outward = −ζ) is −∂_ζ mom = −λ_s·h·mom/ν
                    # (from ∂_ζ mom(0)=λ_s·h·mom/ν, u_t=mom/h flat/small-slope).
                    # The diffusion op supplies the matching diffusivity ν/h²
                    # (= −A[mom,mom,ζ,ζ]), so this is the physical relation — no
                    # sign/magnitude fudge.
                    out.append(Flux(tag="bottom", on="mom",
                                    gradient=-lam * hS * momS / nu))
                elif isinstance(c, RoughWall):
                    # Turbulent rough bed: τ_b = ρ C_f u_b|u_b|, the same law
                    # ``RoughWall.expression`` states, with
                    # C_f = (κ/ln(z_p/z_0))², z_0 = k_s/30.
                    # Matching the viscous stress ρ(ν/h)∂_ζ u to τ_b and writing
                    # u_b = mom/h, the depth CANCELS:
                    #     ∂_ζ mom = h·∂_ζ u = C_f·u_b|u_b|·h²/ν = C_f·mom|mom|/ν,
                    # so the outward-normal gradient at the ζ=0 face is its
                    # negative — the same construction as the Navier-slip arm,
                    # quadratic instead of linear.
                    #
                    # ν (molecular) is the correct conversion here even under
                    # ElderViscosity: Elder's ν_t = κ u_⋆ h ζ(1−ζ) VANISHES at
                    # ζ = 0, so the wall diffusivity the diffusion operator
                    # supplies is exactly ν.
                    kappa, k_s = sm.parameters.kappa, sm.parameters.k_s
                    z_p = sm.parameters.z_p
                    Cf = (kappa / sp.log(z_p / (k_s / 30))) ** 2
                    out.append(Flux(tag="bottom", on="mom",
                                    gradient=-Cf * momS * sp.Abs(momS) / nu))
                else:
                    raise NotImplementedError(
                        f"bed closure {type(c).__name__} → ζ-face BC not wired "
                        "(NavierSlip and RoughWall are wired; extend here via "
                        "the closure's .traction).")
            elif kind == "surface" and isinstance(c, StressFree):
                out.append(Flux(tag="top", on="mom", gradient=0))
        return out
