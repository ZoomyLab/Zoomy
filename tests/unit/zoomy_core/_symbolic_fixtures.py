"""Shared build helpers for the symbolic-pipeline regression tests.

Each ``build_*`` helper runs one of the tutorial derivations and
returns a fully-closed ``System`` + the adapter + named symbols the
tests care about.  The tutorial scripts under ``tutorials/sme/`` keep
their own inline walkthrough for literate teaching; these helpers
exist so the pytest tests stay decoupled from the tutorial files and
don't need ``runpy`` to get at the closed systems.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.ins_generator import (
    AffineProjection,
    Basis,
    EvaluateIntegrals,
    FullINS,
    Integrate,
    InterfaceKBC,
    Inviscid,
    Multiply,
    Newtonian,
    ProductRule,
    Recombine,
    SimplifyIntegrals,
    StateSpace,
)
from zoomy_core.model.derivation.basisfunctions import (
    LayeredBasis,
    Legendre_shifted,
)
from zoomy_core.model.models.sme_model import hydrostatic_scaling
from zoomy_core.systemmodel.system_model import SystemModel


# ---------------------------------------------------------------------------
# SWE (single layer)
# ---------------------------------------------------------------------------

def build_swe_system():
    """Single-layer SWE closed via the symbolic pipeline.

    Returns
    -------
    sm : SystemModel
        Adapter exposing ``flux()``, ``nonconservative_matrix()``,
        ``source()``, ``untagged_remainders()``.
    state : StateSpace
    placeholders : dict[str, sympy.Symbol]
        ``{"h", "hu", "b", "g"}`` — the symbols the solver side uses.
    """
    state = StateSpace(dimension=2)
    t, x, z = state.t, state.x, state.z
    basis = Basis(state, LayeredBasis, interfaces=[state.b, state.eta])

    model = FullINS(state)
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method="analytical"))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p))
    model.momentum.z.remove()
    model.apply(Inviscid(state)).simplify()

    swe = model.branch(name="SWE")
    swe.apply(Integrate(z, state.b, state.eta, method="auto"))
    swe.apply(InterfaceKBC(state, state.b)).simplify()
    swe.apply(InterfaceKBC(state, state.eta)).simplify()
    swe.apply(basis.layer_expand(state.u, 0)).simplify()
    swe.apply(SimplifyIntegrals(state)).simplify()

    alpha = basis.alpha.alpha_0
    hu_fn = sp.Function("hu", real=True)(t, x)
    swe.apply({alpha: hu_fn / state.h})
    swe.apply(Recombine(vars=[t, x]))

    h_sym = sp.Symbol("h", positive=True)
    hu_sym = sp.Symbol("hu", real=True)
    b_sym = sp.Symbol("b", real=True)
    g_sym = sp.Symbol("g", positive=True)

    sm = SystemModel(
        swe,
        state_substitutions={state.h: h_sym, hu_fn: hu_sym, state.b: b_sym},
        state_variables=[b_sym, h_sym, hu_sym],
        equation_variable={("continuity",): h_sym,
                           ("momentum", "x"): hu_sym},
        time_var=t, coords=[x],
    )
    return sm, state, {"h": h_sym, "hu": hu_sym, "b": b_sym, "g": g_sym}


# ---------------------------------------------------------------------------
# SME level-1 (single layer, Newtonian viscosity)
# ---------------------------------------------------------------------------

def build_sme_l1_system():
    """Level-1 Legendre SME with Newtonian viscosity.

    Mirrors ``tutorials/sme/symbolic_walkthrough.py`` end-to-end.
    The returned ``model`` has the three leaves ``continuity``,
    ``momentum.x.test_0``, ``momentum.x.test_1`` — the last of which
    carries the one known unclosed ``Integral(..., (ζ, 0, 1))``
    (sympy.integrate limitation, see Task-3.5).
    """
    state = StateSpace(dimension=2)
    model = FullINS(state)

    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(
        Integrate(state.z, state.z, state.eta, method="analytical")
    )
    model.momentum.z.apply({state.p.subs(state.z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p)).simplify()
    model.momentum.z.remove()
    model.apply(Newtonian(state)).simplify()

    basis = Basis(state, Legendre_shifted, level=1)

    model.momentum.x.apply(Multiply(basis.phi_of_z, outer=True))
    model.momentum.x.apply(ProductRule())

    pointwise_continuity = model.continuity.copy()

    model.apply(Integrate(state.z, state.b, state.eta, method="auto"))

    u_at_b = state.u.subs(state.z, state.b)
    u_at_eta = state.u.subs(state.z, state.eta)
    kinematic_bcs = {
        state.w.subs(state.z, state.b):
            sp.Derivative(state.b, state.t)
            + u_at_b * sp.Derivative(state.b, state.x),
        state.w.subs(state.z, state.eta):
            sp.Derivative(state.eta, state.t)
            + u_at_eta * sp.Derivative(state.eta, state.x),
    }
    model.apply(kinematic_bcs).simplify()

    stress_free_surface = {state.tau["xz"].subs(state.z, state.eta): 0}
    no_tangential_normal_stress = {
        state.tau["xx"].subs(state.z, state.b): 0,
        state.tau["xx"].subs(state.z, state.eta): 0,
    }
    model.apply(stress_free_surface).apply(
        no_tangential_normal_stress
    ).simplify()

    lamda = sp.Symbol("lamda", positive=True)
    tau_c = sp.Symbol("tau_c", positive=True)
    friction_closure = {
        state.tau["xz"].subs(state.z, state.b):
            state.rho * (lamda / tau_c) * u_at_b,
    }
    model.apply(friction_closure).simplify()

    # w-closure pipeline
    w_eq = pointwise_continuity.apply(
        Integrate(state.z, state.b, state.z, method="auto"),
    )
    w_closure = w_eq.solve_for(state.w)
    model.apply(w_closure).simplify()
    model.apply(kinematic_bcs).simplify()

    model.apply(AffineProjection(state))
    model.apply(basis.expand(state.u))
    model.apply(EvaluateIntegrals(state)).simplify()

    return model, state, basis
