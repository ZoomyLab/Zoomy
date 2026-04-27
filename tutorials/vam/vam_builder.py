"""VAM PDESystem builder — composed from ``zoomy_core.derivation``.

Mirrors ``tutorials/sme/sme_builder.py``.  The model logic that used to
live in ``escalante2024_generic.py`` is now expressed as the
composition

  NonHydrostaticFlow  +  PolynomialAnsatz(M, N_w=N, N_p=N)
                      +  GalerkinProjection(w_mode='state')
                      +  KBC bottom + surface BC closures.

Two output modes:

* ``eliminate_closures=False`` (default) — return the full PDESystem
  with the algebraic closures (``w_N - rhs = 0``, ``p_N - rhs = 0``)
  as separate equations.  Suitable for dispersion analysis (the
  closures appear as algebraic rows in the plane-wave matrix).

* ``eliminate_closures=True`` — substitute the algebraic closures
  into every other equation, so the returned PDESystem has no
  algebraic rows and ``M_t`` is **non-singular**.  This is the form
  needed for clean hyperbolicity sampling — the principal-symbol
  pencil then reproduces the paper's eigenvalue formulas (eq 12 of
  Escalante 2024).
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp

from zoomy_core.analysis import PDESystem
from zoomy_core.derivation import (
    NonHydrostaticFlow,
    PolynomialAnsatz,
    GalerkinProjection,
    kbc_bottom_solve_w_N,
    surface_bc_solve_p_N,
    eliminate_constraints,
)


# Re-export the canonical symbols for callers.
flow = NonHydrostaticFlow.with_defaults()
t, x, xi, g = flow.t, flow.x, flow.xi, flow.g
h, b = flow.h, flow.b


def build_vam_pde_system(M: int, N: int, *,
                         eliminate_closures: bool = False,
                         solve_constraints_jointly: bool = False,
                         hyperbolic_predictor: bool = False,
                         w_N_as_input: bool = False,
                         p_N_as_input: bool = False,
                         ) -> Tuple[PDESystem, list, list, list]:
    """Build the VAM nonlinear PDESystem at degree (M, N).

    Args:
        M: degree of the velocity ansatz (u_0..u_M).
        N: degree of w and p ansatzes (w_0..w_N; p_0..p_N).
        eliminate_closures:
            False — keep ``w_N`` and ``p_N`` as state variables, with
                    their algebraic closures included as separate
                    equations (full DAE form, suitable for dispersion).
            True  — pre-substitute the closures into every equation,
                    so ``w_N`` and ``p_N`` no longer appear; the system
                    is purely differential, ``M_t`` is non-singular.
                    NOTE: gives a chain-rule-modified Jacobian; for
                    paper-eq-(12) eigenvalues use
                    ``solve_constraints_jointly=True`` instead.
        solve_constraints_jointly:
            If True, solve ALL algebraic constraints (continuity j=1..N
            + KBC bottom + surface BC) as a SINGLE linear system in
            (w_0..w_N, p_N) and substitute the solution back into the
            evolution equations.  This gives the "underlying-hydrostatic"
            reduced system used in the paper's eq (12) eigenvalue
            analysis.  Equivalent to fully eliminating w_i via
            depth-integrated continuity in the standard SME-style
            derivation.
        hyperbolic_predictor:
            If True, additionally drop the non-hydrostatic pressure
            terms ``T(P)`` (set every ``p_i`` to zero before
            projection).  This produces the **underlying hyperbolic
            system** of eq (7) of Escalante 2024 — what the
            projection-correction predictor (eq 11) actually advances.
            The full DAE form (default) has additional pressure-related
            modes which are determined algebraically at each time step
            rather than evolved.
        w_N_as_input:
            If True, treat ``w_N`` as an opaque coefficient (parameter)
            instead of a state variable: drop ``w_N`` from the field
            list AND drop the KBC-bottom algebraic equation that
            determines it.  ``w_N`` then appears only as a coefficient
            in the remaining matrices.  Combined with
            ``hyperbolic_predictor=True``, this reproduces the paper's
            eigenvalue formula (eq 12) — the paper's matrix
            ``A(U, w_2)`` treats ``w_2`` as a free input, exactly this.
        p_N_as_input:
            Same idea for ``p_N``: drop the surface-BC equation and
            treat ``p_N`` as opaque.  Only meaningful when
            ``hyperbolic_predictor=False`` (otherwise pressure is
            already absent).

    Returns ``(system, u_fields, w_fields, p_fields)``.  Fields treated
    as inputs are excluded from the lists.
    """
    flow = NonHydrostaticFlow.with_defaults()
    # Hyperbolic-predictor mode drops the non-hydrostatic pressure
    # entirely — easiest done by giving the ansatz no p coefficients
    # so ``p ≡ 0`` and the corresponding p_N closure / pressure terms
    # all collapse.
    N_p = -1 if hyperbolic_predictor else N
    ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                              M=M, N_w=N, N_p=N_p)
    proj = GalerkinProjection(flow=flow, ansatz=ansatz, w_mode="state")

    # --- Equations ---
    # Continuity j = 0.
    cont_j0 = proj.project_continuity(0)
    # x-mom j = 0..M.
    xmom = [proj.project_x_momentum(j) for j in range(M + 1)]
    # z-mom j = 0..N-1 (j=N is the algebraic closure for w_N via KBC bottom).
    zmom = [proj.project_z_momentum(j) for j in range(N)]
    # Continuity j = 1..N (algebraic-in-time constraints).  Skipped in
    # hyperbolic_predictor mode (those are enforced by the corrector
    # Poisson step, not the predictor).
    cont_constraints = ([proj.project_continuity(j) for j in range(1, N + 1)]
                        if not hyperbolic_predictor else [])

    # Algebraic closures (RHS expressions).
    w_N_rhs = kbc_bottom_solve_w_N(ansatz, flow)
    p_N_rhs = (surface_bc_solve_p_N(ansatz, flow)
               if not hyperbolic_predictor else None)

    if solve_constraints_jointly:
        # Build the algebraic constraint set + the corresponding
        # constrained fields.
        #
        # Hyperbolic-predictor mode (paper eq 7 / eq 12):
        #   - Drop p entirely (already done via N_p=-1).
        #   - Drop cont j=1..N constraints (those are corrector-only).
        #   - Eliminate ONLY w_N via KBC bottom.
        #   - Result: state = (h, u_0..u_M, w_0..w_{N-1}); w_N is
        #     algebraically closed.
        #
        # Full DAE mode (eq 6 with pressure):
        #   - Use cont j=1..N + KBC bottom to eliminate w_0..w_N (N+1
        #     equations for N+1 unknowns).
        #   - Use surface BC to eliminate p_N.
        #   - Result: state = (h, u_0..u_M, p_0..p_{N-1}); pressure
        #     drives the dynamics through the algebraic-elimination
        #     chain rule.
        constraint_eqs = []
        constrained_fields = []
        if hyperbolic_predictor:
            # Only KBC bottom; close w_N.
            constraint_eqs.append(ansatz.w_coeffs[N] - w_N_rhs)
            constrained_fields.append(ansatz.w_coeffs[N])
        else:
            # cont j=1..N + KBC bottom for w_0..w_N (N+1 each).
            constraint_eqs.extend(cont_constraints)
            constraint_eqs.append(ansatz.w_coeffs[N] - w_N_rhs)
            constrained_fields.extend(list(ansatz.w_coeffs))   # w_0..w_N
            # Surface BC for p_N.
            constraint_eqs.append(ansatz.p_coeffs[N] - p_N_rhs)
            constrained_fields.append(ansatz.p_coeffs[N])

        dt_h_rule = {sp.Derivative(flow.h, flow.t):
                     -sp.Derivative(flow.h * ansatz.u_coeffs[0], flow.x)}

        # Apply dt_h_rule to all diff eqs EXCEPT cont j=0 (which IS the
        # rule's source — substituting into it would give 0 = 0).  Keep
        # cont j=0 unchanged.
        diff_eqs_other = xmom + zmom
        diff_eqs_other_reduced = eliminate_constraints(
            differential_eqs=diff_eqs_other,
            constraint_eqs=constraint_eqs,
            constrained_fields=constrained_fields,
            apply_dt_h_rule=dt_h_rule,
        )
        # cont j=0 just needs the constraint substitution (no dt_h).
        sol_only = eliminate_constraints(
            differential_eqs=[cont_j0],
            constraint_eqs=constraint_eqs,
            constrained_fields=constrained_fields,
            apply_dt_h_rule=None,
        )
        diff_eqs_reduced = sol_only + diff_eqs_other_reduced

        u_fields = list(ansatz.u_coeffs)
        # Keep w fields that were NOT eliminated.
        w_fields = [w for w in ansatz.w_coeffs if w not in constrained_fields]
        p_fields = ([p for p in ansatz.p_coeffs if p not in constrained_fields]
                    if not hyperbolic_predictor else [])
        fields = [flow.h] + u_fields + w_fields + p_fields
        equations = diff_eqs_reduced
    elif eliminate_closures:
        repl = {ansatz.w_coeffs[N]: w_N_rhs}
        if not hyperbolic_predictor:
            repl[ansatz.p_coeffs[N]] = p_N_rhs
        # Substitute closures into every projected equation.
        diff_eqs = xmom + zmom + cont_constraints
        diff_eqs = [sp.expand(eq.xreplace(repl).doit()) for eq in diff_eqs]
        # Apply ∂_t h substitution from cont j=0 to remove residual
        # ∂_t h atoms in the projection equations (the same bug-3
        # closure pattern as SME).  Do NOT substitute it INTO cont j=0
        # itself (that's the equation we're using).
        cont_j0_subbed = sp.expand(cont_j0.xreplace(repl).doit())
        dt_h_atom = sp.Derivative(flow.h, flow.t)
        dt_h_rhs = -sp.Derivative(flow.h * ansatz.u_coeffs[0], flow.x)
        diff_eqs_subbed = []
        for eq in diff_eqs:
            prev = None
            cur = sp.expand(eq.doit())
            while prev != cur:
                prev = cur
                cur = sp.expand(cur.xreplace({dt_h_atom: dt_h_rhs}))
                cur = sp.expand(cur.doit())
            diff_eqs_subbed.append(cur)
        u_fields = list(ansatz.u_coeffs)
        w_fields = list(ansatz.w_coeffs[:N])           # drop w_N
        p_fields = (list(ansatz.p_coeffs[:N]) if not hyperbolic_predictor
                    else [])
        fields = [flow.h] + u_fields + w_fields + p_fields
        equations = [cont_j0_subbed] + diff_eqs_subbed
    else:
        u_fields = list(ansatz.u_coeffs)
        w_fields = list(ansatz.w_coeffs)
        p_fields = list(ansatz.p_coeffs)

        # When w_N or p_N are treated as inputs, replace their
        # ``Function(t, x)`` atoms by constant ``Symbol``s throughout
        # the equations.  Otherwise sympy's ``linearise`` would still
        # see them as t,x-dependent and pick up spurious ∂_x / ∂_t
        # contributions.
        input_subs = {}
        if w_N_as_input:
            sym = sp.Symbol(f"w_{N}_input", real=True)
            input_subs[ansatz.w_coeffs[N]] = sym
            w_fields = w_fields[:N]
        if not hyperbolic_predictor and p_N_as_input:
            sym = sp.Symbol(f"p_{N}_input", real=True)
            input_subs[ansatz.p_coeffs[N]] = sym
            p_fields = p_fields[:N]

        algebraic_eqs = []
        if not w_N_as_input:
            algebraic_eqs.append(ansatz.w_coeffs[N] - w_N_rhs)
        if not hyperbolic_predictor and not p_N_as_input:
            algebraic_eqs.append(ansatz.p_coeffs[N] - p_N_rhs)

        fields = [flow.h] + u_fields + w_fields + p_fields
        equations = ([cont_j0] + xmom + zmom + cont_constraints
                     + algebraic_eqs)
        if input_subs:
            equations = [sp.expand(eq.xreplace(input_subs).doit())
                         for eq in equations]

    return (PDESystem(equations=equations,
                      fields=fields,
                      time=flow.t,
                      space=[flow.x],
                      parameters={flow.g: flow.g}),
            u_fields, w_fields, p_fields)
