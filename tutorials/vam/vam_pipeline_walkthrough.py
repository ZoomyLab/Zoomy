# ---
# title: "VAM(1, 2, 2) pipeline: inline derivation → class → SystemModel"
# author: Ingo Steldermann
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# # VAM(1, 2, 2) pipeline walkthrough
#
# A clean walk through the VAM derivation pipeline, in three parts:
#
# 1. **Line-by-line derivation** of VAM(M=1, N_w=2, N_p=2) using the
#    chain primitives end-to-end (Escalante 2024 cont-projection
#    formulation, opaque-ζ test args).
# 2. **Class-based generation** via :class:`VAMModelGalerkin` — verifies
#    bit-for-bit equivalence with §1.
# 3. **Transfer to SystemModel** via :func:`SystemModel.from_model`,
#    followed by an explicit change-of-variables to Form B
#    (conservative state).
#
# **No PDESystem.**  The chain System tree feeds directly into the
# operator-form SystemModel via tag-walking on the model's
# ``flux()`` / ``source()`` / etc. accessors.  ``SystemModel.from_model``
# is the single entry point for analysis.

# +
import copy

import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection, EvaluateIntegrals, Expand, FullINS, InterfaceKBC,
    Integrate, Inviscid, Multiply, ProductRule, StateSpace,
)
from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel
# -


# ## 1. Line-by-line derivation of VAM(1, 2, 2)
#
# Twelve numbered steps mirroring
# :meth:`VAMModelGalerkin._build_chain`.  The asymmetric levels are
#
# * ``M = 1``    — two x-velocity modes ``U_0, U_1``.
# * ``N_w = 2``  — three z-velocity modes ``W_0, W_1, W_2``;
#   ``W_2`` will be eliminated by the bottom KBC.
# * ``N_p = 2``  — three non-hydrostatic pressure modes ``P_0, P_1, P_2``;
#   ``P_2`` will be eliminated by the surface BC.
#
# After all closures the system carries seven unknowns
# ``(h, U_0, U_1, W_0, W_1, P_0, P_1)`` and seven equations:
# one mass evolution, two x-momentum, two z-momentum, two algebraic
# pressure constraints.

# ### 1.1 Setup — state, bases, mode coefficients, test functions

# +
M, N_w, N_p = 1, 2, 2

state = StateSpace(dimension=2)
t, x, z = state.t, state.x, state.z

basis_u = Legendre_shifted(level=M, symbol="phi_u")
basis_w = Legendre_shifted(level=N_w, symbol="phi_w")
basis_p = Legendre_shifted(level=N_p, symbol="phi_p")

coeffs_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(M + 1)]
coeffs_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(N_w + 1)]
coeffs_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(N_p + 1)]

# Test-function arguments use the opaque ``state.zeta(t, x, z)`` so
# sympy's chain rule fires through ``ProductRule`` for ∂_t / ∂_x / ∂_z.
test_phi_u = Zstruct(**{f"phi_{k}": basis_u.phi[k](state.zeta)
                        for k in range(M + 1)})
test_phi_w = Zstruct(**{f"phi_{k}": basis_w.phi[k](state.zeta)
                        for k in range(N_w)})
test_phi_cont = Zstruct(**{f"phi_{k}": basis_p.phi[k](state.zeta)
                           for k in range(N_p + 1)})
# -


# ### 1.2 Step 1 — 3D INS, inviscid, hydrostatic pressure split
#
# Start from the full Navier–Stokes system, drop viscous stresses, and
# split the pressure into hydrostatic + non-hydrostatic remainder
# ``p = ρ g (η − z) + p_NH``.

# +
sys = FullINS(state)
sys.apply(Inviscid(state)).simplify()
p_NH = sp.Function("p_NH", real=True)(t, x, z)
sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
          ).simplify()
# -


# ### 1.3 Step 2 — Galerkin projection of continuity AND momentum
#
# Multiply each leaf equation by its test functions, **outer**-style,
# producing a tree of projected leaves: ``continuity.test_k``,
# ``momentum.x.test_k``, ``momentum.z.test_k``.

# +
sys.continuity.apply(Multiply(test_phi_cont, outer=True))
sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
sys.momentum.z.apply(Multiply(test_phi_w, outer=True))
# -


# ### 1.4 Step 3+4 — ProductRule, then depth-integrate
#
# ``ProductRule`` distributes ``φ(ζ)·∂_v F → ∂_v(φ(ζ)·F) − F·φ'(ζ)·∂_v ζ``
# for every variable ``v ∈ {t, x, z}``.  ``Integrate`` then applies
# Leibniz on ``∂_t / ∂_x`` and the Fundamental Theorem on ``∂_z``.

# +
sys.apply(ProductRule(variables=[t, x, z]))
sys.apply(Integrate(z, state.b, state.eta, method="auto"))
# -


# ### 1.5 Step 5+6 — Kinematic BCs, drop ∂_t b, surface p_NH closure
#
# After Leibniz, the residuals contain ``u(η)·w(η)``, ``u(b)·w(b)``,
# ``∂_t η``, ``∂_t b`` atoms.  ``InterfaceKBC`` substitutes
# ``w(η) → ∂_t η + u(η)·∂_x η`` and ``w(b) → u(b)·∂_x b``.  Static
# bottom: ``∂_t b → 0``.  Surface pressure: ``p_NH(η) = 0``.

# +
sys.apply(InterfaceKBC(state, state.b)).simplify()
sys.apply(InterfaceKBC(state, state.eta)).simplify()
sys.apply({sp.Derivative(state.b, t): sp.S.Zero}).simplify()
sys.apply({p_NH.subs(z, state.eta): 0}).simplify()
# -


# ### 1.6 Step 7 — affine map + ansatz expansion (Sum-form intermediate)
#
# Change of variables ``z → ζ = (z − b)/h`` on the integration
# variable, then expand ``u``, ``w``, ``p_NH`` into modal sums
# ``Σ_k coeff_k · φ_k(ζ)``.  The Sum-form intermediate (paper notation)
# is snapshotted here.

# +
sys.apply(AffineProjection(state, rewrite_basis_args=False))
sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w, state=state))
sys.apply(Expand(p_NH,    basis=basis_p, coefficients=coeffs_p, state=state))

chain_intermediate = copy.deepcopy(sys)
# -


# ### 1.7 Step 8 — EvaluateIntegrals (boundary atoms collapse here)
#
# ``EvaluateIntegrals`` does **two** things, both via the basis cache:
#
# 1. **Bulk integrals**: resolve every polynomial ζ integral
#    ``∫_0^1 φ_i(ζ)·φ_j(ζ)·… dζ`` to its concrete rational value.
# 2. **Boundary atoms**: ``φ_k(0)``, ``φ_k(1)`` atoms (introduced by
#    Leibniz boundary terms at z=b ↔ ζ=0 and z=η ↔ ζ=1) are also
#    looked up in the basis cache and replaced by their concrete
#    rational values.
#
# After this step the system has **no opaque basis atoms left** — only
# ``U_k``, ``W_k``, ``P_k``, ``h``, ``b``, and their derivatives.

# +
sys.apply(EvaluateIntegrals(state)).simplify()
sys.apply({sp.Derivative(state.b, t): sp.S.Zero}).simplify()
# -


# ### 1.8 Step 9 — Modal closures: solve KBC bot for ``W_{N_w}``,
# # surface BC for ``P_{N_p}``
#
# The bulk basis sums have ``N_w + 1`` W-modes and ``N_p + 1`` P-modes,
# but only ``N_w`` momentum projections and ``N_p`` cont-projections.
# **The system is underdetermined by exactly 1 mode for W and 1 for P.**
# We close by solving the algebraic boundary conditions at the basis
# level:
#
# * Bottom KBC: ``Σ_k W_k·φ_w_k(0) − Σ_k U_k·φ_u_k(0)·∂_x b = 0``
#   solved for ``W_{N_w}``:
#
#   $$W_{N_w} = \frac{\sum_k U_k \, \varphi_{u,k}(0) \, \partial_x b
#                     - \sum_{k < N_w} W_k \, \varphi_{w,k}(0)}
#                    {\varphi_{w,N_w}(0)}.$$
#
#   The solution is substituted **everywhere** ``W_{N_w}`` appears in
#   the system.
# * Surface BC: ``Σ_k φ_p_k(1)·P_k = 0`` solved analogously for
#   ``P_{N_p}``.
#
# **Where the ``(∂_x b)²`` cross-terms come from.**  The residual on
# the j ≥ 1 rows already carries ``∂_x b`` factors (from the
# ``∂_x η = ∂_x b + ∂_x h`` substitution in §1.5).  Substituting the
# ``∂_x b``-containing ``W_{N_w}`` cross-multiplies, producing
# ``(∂_x b)²`` terms in higher-order rows.  These are zero **on the
# constraint surface** ``cont_j1 = cont_j2 = 0``; they do not appear
# in Escalante eq (4), which is pre-reduced.

# +
u_at_b = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero) for k in range(M + 1))
w_at_b = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero) for k in range(N_w + 1))
bot_kbc = w_at_b - u_at_b * sp.Derivative(state.b, x).doit()
w_top_sol = sp.solve(bot_kbc, coeffs_w[N_w])[0]
sys.apply({coeffs_w[N_w]: w_top_sol}).simplify()

p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One) for k in range(N_p + 1))
p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()
# -


# ### 1.9 Step 10+11 — mass equation substitution → DAE structure;
# # auto-tag every row
#
# Continuity row 0 (j=0) is the **mass evolution**.  Rows j=1, …, N_p
# would otherwise carry ``∂_t h`` from compound atoms; substituting the
# mass equation ``∂_t h = −∂_x(h·U_0)`` eliminates the time derivative
# entirely, making those rows **purely algebraic constraints**.  The
# resulting system has 5 evolution rows + 2 algebraic rows = a DAE.
#
# Each row's terms are then auto-tagged with canonical solver tags
# (``time_derivative``, ``flux``, ``hydrostatic_pressure``,
# ``nonconservative_flux``, ``source``) so the downstream
# ``SystemModel.from_model`` can route them into operator matrices.
#
# The fully-closed tagged System tree is what the **class version in §2
# produces directly** — let's render it now to confirm:

sys.describe()


# ## 2. Same model via `VAMModelGalerkin`
#
# The class executes the same chain primitives in
# :meth:`VAMModelGalerkin._build_chain` and exposes:
#
# * ``m._chain_system`` — the System tree (same as our inline ``sys``).
# * ``m.equations`` — dict mapping canonical row name (``mass``,
#   ``xmom_j0``, …) to its tagged ``Expression`` leaf.
# * ``m.flux()``, ``m.hydrostatic_pressure()``,
#   ``m.nonconservative_matrix()``, ``m.source()``, ``m.mass_matrix()``
#   — operator-API methods implemented as tag-walks over
#   ``m._chain_system``.

# +
m1d = VAMModelGalerkin(level=1)
print("State:", list(m1d.variables.keys()))
print("Equation names:", m1d.equation_names)
# -


# ### 2.1 Three normal forms
#
# The model produces the system in **Form A — primitive state**:
#
# $$
# Q^{A} = (h,\; U_0,\; U_1,\; W_0,\; W_1,\; P_0,\; P_1).
# $$
#
# Two other forms appear later:
#
# * **Form B — conservative state**:
#   $Q^{B} = (h,\; hU_0,\; hU_1,\; hW_0,\; hW_1,\; P_0,\; P_1)$.
#   Same residuals, but state entries are the conserved quantities
#   $q_{U_k} = h\,U_k$, $q_{W_k} = h\,W_k$.  Reached via
#   :meth:`SystemModel.change_state_variables`.
# * **Form C — Escalante paper form** (eq (4) in his 2024 JCP paper).
#   Conservative state like Form B, but with kinematic pressure
#   $p_k = P_k/\rho$.
#
# **Form A ↔ Form B** is an algebraic identity.  The non-identity
# entries of the Form A mass matrix migrate into the operator slots
# when we change variables — but the j=1 rows retain a residual
# off-diagonal entry from the Galerkin cross-terms (see §2.4).
#
# **Form A ↔ Form C** agrees **pointwise** on rows ``mass``,
# ``xmom_j0``, ``zmom_j0``, ``cont_j1``, ``cont_j2``.  On rows
# ``xmom_j1`` and ``zmom_j1`` it is **not** pointwise equal: the chain
# residual carries non-conservative ``∂_t h`` and ``W_k`` cross-terms
# that only vanish on the constraint surface ``cont_j1 = cont_j2 = 0``.
# Step 1 tests (``test_chain_xmom_j1_constraint_equivalent_to_escalante``,
# ``test_chain_zmom_j1_constraint_equivalent_to_escalante``) lock this
# constraint-modulo equivalence in.


# ### 2.2 Transfer to SystemModel
#
# :func:`SystemModel.from_model` reads the operator-API methods on
# ``m1d`` and freezes the resulting matrices.  No PDESystem
# intermediate.  No state-rewriting between equation form and operator
# form — the tag-walk produces the matrices directly.

# +
sm = SystemModel.from_model(m1d)
print("sm.state:", sm.state)
print("sm.equation_names:", sm.equation_names)
# -


# ### 2.3 Form A mass matrix (non-identity, primitive state)
#
# The chain residual on row 1 (xmom_j0) is
# ``∂_t(h·U_0) + ∂_x(...) + ... = 0``.  Expanding the compound
# time-derivative via product rule:
# ``∂_t(h·U_0) = U_0·∂_t h + h·∂_t U_0``.  So row 1 of the mass matrix
# in Form A has ``M[1, h-col] = U_0``, ``M[1, U_0-col] = h``.  The
# higher-order rows pick up additional Galerkin cross-terms.

sm.mass_matrix


# ### 2.4 Form B via `change_state_variables`
#
# Apply the conservative transform ``U_k → q_{U_k}/h``,
# ``W_k → q_{W_k}/h``.  The Jacobian rule transforms the operator
# slots:
#
# * Mass matrix on j=0 rows (mass, xmom_j0, zmom_j0) becomes the
#   identity row ``[…, 1, …]`` — the conservative state is precisely
#   the natural state for those rows.
# * Mass matrix on j=1 rows (xmom_j1, zmom_j1) retains a non-zero
#   ``∂_t h``-column entry equal to ``(-q_{k-1} + q_k/3) / h`` — the
#   genuine Galerkin chain-rule cross-term that does NOT vanish under
#   the naive primitive→conservative change of variables.  It vanishes
#   only on the constraint surface (Step 1 tests).

# +
sm_B = SystemModel.from_model(m1d)
h_sym, U_0_sym, U_1_sym, W_0_sym, W_1_sym, P_0_sym, P_1_sym = sm_B.state
q_U0 = sp.Symbol("q_U0", real=True)
q_U1 = sp.Symbol("q_U1", real=True)
q_W0 = sp.Symbol("q_W0", real=True)
q_W1 = sp.Symbol("q_W1", real=True)
sm_B.change_state_variables(
    new_state=[h_sym, q_U0, q_U1, q_W0, q_W1, P_0_sym, P_1_sym],
    transform={U_0_sym: q_U0 / h_sym, U_1_sym: q_U1 / h_sym,
               W_0_sym: q_W0 / h_sym, W_1_sym: q_W1 / h_sym},
)
print("Form B state:", [str(s) for s in sm_B.state])
# -

sm_B.mass_matrix


# ### 2.5 Form B flux matches Escalante eq (4) — bit-for-bit on j=0 rows
#
# Row 1 (xmom_j0) flux in Form B:
# ``q_U0²/h + q_U1²/(3·h) + h·P_0/ρ`` — matches Escalante eq (4) row 2
# inviscid flux argument.

sm_B.flux[1, 0]


# Row 3 (zmom_j0) flux: ``q_U0·q_W0/h + q_U1·q_W1/(3·h)``:

sm_B.flux[3, 0]


# Row 3 (zmom_j0) source ``S[3, 0]``.  Convention reminder:
# SystemModel residual form is ``... − S(Q) = 0``, so a LHS term of
# ``−2·P_1/ρ`` becomes ``S = +2·P_1/ρ`` (the LHS equals ``−S``).

sm_B.source[3, 0]


# ### 2.6 Residual reconstruction — sanity check
#
# ``sm.reconstruct_residuals()`` rebuilds each equation's residual from
# its operator slots: ``M·∂_t Q + ∂_x F + ∂_x P + B·∂_x Q − S``.  Three
# uses across the codebase:
#
# 1. Test comparisons against Escalante's eq (4) term-by-term.
# 2. Linearisation for stability / dispersion analysis (no separate
#    equation object after the refactor; analysis pulls residuals from
#    operator slots).
# 3. Splitter input — to substitute corrector updates into the
#    cont-projection rows you need the full residual.

# +
residuals = sm.reconstruct_residuals()
for name, res in zip(sm.equation_names, residuals):
    print(f"--- {name} ---")
    print(sp.expand(res))
# -


# ## 3. Pressure splitting
#
# ``split_for_pressure(sm, pressure_vars, dt)`` decomposes the chain
# DAE into three rectangular sub-SystemModels — the Chorin-style
# projection-correction scheme of Escalante 2024 eq (11)–(13):
#
# * ``SM_pred`` — predictor (5 evolution rows on
#   ``(h, U_0, U_1, W_0, W_1)``, pressure frozen at the previous
#   time step).
# * ``SM_press`` — pressure stage (2 algebraic elliptic rows in
#   ``(P_0, P_1)``).  Mass matrix is zero.
# * ``SM_corr`` — corrector (4 algebraic update rows for the
#   velocity moments using the new pressure).  Mass matrix is zero.
#
# All three share the same 7-entry state ``Q`` and the same parameters.
# Each carries an ``equation_to_state_index`` that records which
# entries of ``Q`` it updates; other entries pass through unchanged.

# +
from zoomy_core.model.splitter import (
    split_for_pressure, build_pressure_elliptic_block, verify_p_linearity,
)

name_to_sym = {str(s): s for s in sm.state}
P_0 = name_to_sym["P_0"]
P_1 = name_to_sym["P_1"]
dt = sp.Symbol(r"\Delta t", positive=True)

split = split_for_pressure(sm, [P_0, P_1], dt)
print("SM_pred:  n_eq=", split.SM_pred.n_equations,
      "rows=", split.SM_pred.equation_names,
      "updates idx=", split.SM_pred.equation_to_state_index)
print("SM_press: n_eq=", split.SM_press.n_equations,
      "rows=", split.SM_press.equation_names,
      "updates idx=", split.SM_press.equation_to_state_index)
print("SM_corr:  n_eq=", split.SM_corr.n_equations,
      "rows=", split.SM_corr.equation_names,
      "updates idx=", split.SM_corr.equation_to_state_index)
# -


# ### 3.1 Pressure sources `T_u`, `T_w`
#
# The elliptic-block builder extracts per-conservative-variable
# pressure sources ``T_u[k]``, ``T_w[k]`` from the chain ``xmom_jk`` /
# ``zmom_jk`` rows.  These are the linear-in-P parts that drive the
# corrector update.

# +
block = build_pressure_elliptic_block(sm, [P_0, P_1], dt)
for k, T in enumerate(block["T_u"]):
    print(f"T_u[{k}] =", sp.expand(T))
for k, T in enumerate(block["T_w"]):
    print(f"T_w[{k}] =", sp.expand(T))
# -


# ### 3.2 Elliptic block rows
#
# After substituting the corrector update ``Q_k → Q_k^(k̃) - (Δt/h)·T``
# into the cont-projection rows, the result is linear in
# ``(P_l, ∂_x P_l, ∂_xx P_l)``.  ``verify_p_linearity`` confirms this
# strict linearity and reads off the coefficient matrix — the
# discretised Poisson operator.

# +
linearity = verify_p_linearity(block["rows"], block["pressure_vars"],
                               sm.space[0])
assert set(linearity["coefficients"].keys()) == {1, 2}, (
    "elliptic block should have exactly 2 rows for VAM(1, 2, 2)"
)
print("Elliptic block rows are strictly linear in (P, ∂_x P, ∂_xx P)")
print()
print("Coefficients of row 1 (elliptic_j1):")
for atom_name, coeff in linearity["coefficients"][1].items():
    if sp.simplify(coeff) != 0:
        print(f"  {atom_name}: {sp.expand(coeff)}")
# -


# ### 3.3 Three sub-SystemModels rendered
#
# Each stage carries the canonical operator slots (`flux`,
# `hydrostatic_pressure`, `nonconservative_matrix`, `source`,
# `mass_matrix`) populated by tag-walking — the same pipeline as the
# parent chain DAE.

split.SM_pred.describe(full=False)


split.SM_press.describe(full=False)


split.SM_corr.describe(full=False)
