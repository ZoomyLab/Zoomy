# ---
# title: "VAM(1, 2, 2) pipeline: inline derivation → class → SystemModel → splitter"
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
# A clean, four-section walk through the full VAM derivation pipeline:
#
# 1. **Line-by-line derivation** of VAM(M=1, N_w=2, N_p=2) using the
#    chain primitives end-to-end (System B / Escalante cont-projection
#    formulation, opaque-ζ test args).
# 2. **Class-based generation** via :class:`VAMModelGalerkin` — verifies
#    bit-for-bit equivalence with §1.
# 3. **Transfer to SystemModel** via :meth:`SystemModel.from_pdesystem`.
# 4. **Pressure splitting** via :func:`split_for_pressure` — the
#    predictor / pressure-elliptic / corrector decomposition used by the
#    splitting solver.

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
from zoomy_core.model.splitter import (
    build_pressure_elliptic_block,
    split_for_pressure,
    verify_p_linearity,
)
from zoomy_core.analysis import PDESystem
# -


# ## 1. Line-by-line derivation of VAM(1, 2, 2)
#
# Twelve numbered steps mirroring
# :meth:`VAMModelGalerkin.derive_model`.  The asymmetric levels are
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
b, h, eta = state.b, state.h, state.eta

basis_u = Legendre_shifted(level=M,   symbol="phi_u")
basis_w = Legendre_shifted(level=N_w, symbol="phi_w")
basis_p = Legendre_shifted(level=N_p, symbol="phi_p")

coeffs_u = [sp.Function(f"U_{k}", real=True)(t, x) for k in range(M + 1)]
coeffs_w = [sp.Function(f"W_{k}", real=True)(t, x) for k in range(N_w + 1)]
coeffs_p = [sp.Function(f"P_{k}", real=True)(t, x) for k in range(N_p + 1)]

# Test-function arguments use the opaque ``state.zeta(t, x, z)`` so
# sympy's chain rule fires through ``ProductRule`` for ∂_t / ∂_x / ∂_z.
# ``AffineProjection`` collapses the ζ atoms to ``(z−b)/h`` after the
# affine map.
test_phi_u = Zstruct(
    **{f"phi_{k}": basis_u.phi[k](state.zeta) for k in range(M + 1)})
test_phi_w = Zstruct(
    **{f"phi_{k}": basis_w.phi[k](state.zeta) for k in range(N_w)})
test_phi_cont = Zstruct(
    **{f"phi_{k}": basis_p.phi[k](state.zeta) for k in range(N_p + 1)})
# -


# ### 1.2 Step 1 — 3D INS, inviscid, hydrostatic pressure split
#
# Start from the full Navier–Stokes system, drop viscous stresses, and
# split the pressure into hydrostatic + non-hydrostatic remainder
# ``p = ρ g (η − z) + p_NH``.  The hydrostatic piece is closed-form;
# the chain works on ``p_NH``.

# +
sys = FullINS(state)
sys.apply(Inviscid(state)).simplify()

p_NH = sp.Function("p_NH", real=True)(t, x, z)
sys.apply({state.p: state.rho * state.g * (eta - z) + p_NH}).simplify()
# -


# ### 1.3 Step 2 — Galerkin projection of continuity AND momentum
#
# Multiply each leaf equation by its test functions, **outer**-style,
# producing a tree of projected leaves: ``continuity.test_k``,
# ``momentum.x.test_k``, ``momentum.z.test_k``.

# +
sys.continuity.apply(Multiply(test_phi_cont, outer=True))
sys.momentum.x.apply(Multiply(test_phi_u,    outer=True))
sys.momentum.z.apply(Multiply(test_phi_w,    outer=True))
# -


# ### 1.4 Step 3+4 — ProductRule, then depth-integrate
#
# ``ProductRule`` distributes ``φ(ζ)·∂_v F → ∂_v(φ(ζ)·F) − F·φ'(ζ)·∂_v ζ``
# for every variable ``v ∈ {t, x, z}``.  Adding ``t`` to the variable
# list (vs the older mixed convention) is what makes
# ``Integrate(method="auto")`` fire Leibniz on the ``∂_t u`` / ``∂_t w``
# integrands.
#
# ``Integrate(z, b, η, method="auto")`` then depth-integrates each
# leaf — Leibniz on ∂_t / ∂_x boundaries, fundamental theorem on the
# remaining ∂_z divergences.

# +
sys.apply(ProductRule(variables=[t, x, z]))
sys.apply(Integrate(z, b, eta, method="auto"))
# -


# ### 1.5 Step 5+6 — Kinematic BCs, drop ∂_t b, and surface p_NH closure
#
# The surface KBC at ``z = η`` substitutes
# ``∂_t η = ∂_t b + ∂_t h``; the bottom KBC substitutes
# ``w(b) = ∂_t b + u(b)·∂_x b``.  We immediately drop the ``∂_t b``
# atom (static-bottom assumption) — doing it here, while the atom is
# still bare, avoids a system-wide ``.doit()`` later that would
# expand every conservative ``Derivative(F, x)`` flux atom and
# prevent solver-tag extraction in conservative form.
#
# The non-hydrostatic pressure is set to zero on the free surface
# (Dirichlet BC).

# +
sys.apply(InterfaceKBC(state, b)).simplify()
sys.apply(InterfaceKBC(state, eta)).simplify()
sys.apply({sp.Derivative(b, t): sp.S.Zero}).simplify()
sys.apply({p_NH.subs(z, eta): 0}).simplify()
# -


# ### 1.6 Step 7 — affine map + ansatz expansion (Sum-form intermediate)
#
# ``AffineProjection`` turns the integration variable into the affine
# coordinate ``ζ_ref ∈ [0, 1]`` and collapses every ``ζ(t, x, z)`` atom
# from the test-function side via ``z → ζ·h + b``.  ``rewrite_basis_args``
# is off because the test-function args are already in opaque-ζ form.
#
# ``Expand`` substitutes the polynomial ansatz
# ``u = Σ_k U_k(t, x) φ_u_k(ζ)`` (and the same for ``w`` and ``p_NH``).

# +
sys.apply(AffineProjection(state, rewrite_basis_args=False))
sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u, state=state))
sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w, state=state))
sys.apply(Expand(p_NH,    basis=basis_p, coefficients=coeffs_p, state=state))

chain_intermediate = copy.deepcopy(sys)
# -

# Render the **Sum-form intermediate** — paper notation, integrals not
# yet resolved.
chain_intermediate.describe()


# ### 1.7 Step 8 — evaluate integrals (compound ∂_x atoms preserved)
#
# ``EvaluateIntegrals`` resolves the polynomial ζ integrals via the
# basis cache.  Compound ``Derivative(F(Q), x)`` atoms are preserved
# (no system-wide ``.doit()``) so downstream tag extraction can read
# them as conservative fluxes.  ``EvaluateIntegrals`` can re-introduce
# ``Derivative(b, t)`` via boundary-evaluation pull-throughs in the
# ``test_1`` (and higher) leaves, so we re-apply the static-bottom
# rule afterwards to clear them.

# +
sys.apply(EvaluateIntegrals(state)).simplify()
sys.apply({sp.Derivative(b, t): sp.S.Zero}).simplify()
# -


# ### 1.8 Step 10+11 — modal closures for W_{N_w} and P_{N_p}
#
# Bottom KBC at the basis level: ``Σ_k W_k φ_w_k(0) − (Σ_k U_k φ_u_k(0))·∂_x b = 0``
# is a single algebraic relation that we solve for the topmost mode
# ``W_2`` and substitute everywhere.
#
# Surface BC for the pressure: ``Σ_k φ_p_k(1)·P_k = 0`` solved for
# ``P_2``.

# +
u_at_b = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero) for k in range(M + 1))
w_at_b = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero) for k in range(N_w + 1))
bot_kbc = w_at_b - u_at_b * sp.Derivative(b, x).doit()
w_top_sol = sp.solve(bot_kbc, coeffs_w[N_w])[0]
sys.apply({coeffs_w[N_w]: w_top_sol}).simplify()

p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One) for k in range(N_p + 1))
p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()
# -

# The closed chain — render at this point to see all seven leaves.
sys.describe()


# ### 1.9 Step 12 — package as a PDESystem
#
# Two pieces of post-processing go into the PDESystem:
#
# 1. The ``j = 0`` continuity row is the **mass evolution**;
#    rows ``j = 1, …, N_p`` are made purely **algebraic** by
#    substituting the mass equation
#    ``∂_t h = −∂_x(h U_0)``.  This zeroes the time-derivative on the
#    pressure constraint rows so they have an all-zero row in the mass
#    matrix (DAE algebraic constraints).
# 2. Equations are reordered into the canonical chain layout:
#    ``mass, xmom_j0..M, zmom_j0..N_w−1, cont_j1..N_p``.

# +
mass_expr = sys._tree.continuity.test_0.expr
dt_h_sub = {sp.Derivative(h, t):
            -sp.Derivative(h * coeffs_u[0], x).doit()}

ordered = [("mass", mass_expr)]
for k in range(M + 1):
    ordered.append((f"xmom_j{k}",
                    getattr(sys._tree.momentum.x, f"test_{k}").expr))
for k in range(N_w):
    ordered.append((f"zmom_j{k}",
                    getattr(sys._tree.momentum.z, f"test_{k}").expr))
for k in range(1, N_p + 1):
    cont_jk = getattr(sys._tree.continuity, f"test_{k}").expr
    ordered.append((f"cont_j{k}", sp.expand(cont_jk.doit().subs(dt_h_sub))))

inline_chain_dae = PDESystem(
    equations=[expr for _, expr in ordered],
    fields=[h] + coeffs_u + coeffs_w[:N_w] + coeffs_p[:N_p],
    time=t,
    space=[x],
    parameters={state.g: state.g, state.rho: state.rho},
)
inline_chain_dae.equation_names = [n for n, _ in ordered]
# -


# ## 2. Same model via `VAMModelGalerkin`
#
# The class executes the same twelve steps internally and exposes
# ``_chain_dae`` as the closed PDESystem.  We verify
# bit-for-bit agreement with §1.

# +
class VAM1D(VAMModelGalerkin):
    ins_dimension = 2


m1d = VAM1D(level=1)
class_chain_dae = m1d._chain_dae

assert class_chain_dae.equation_names == inline_chain_dae.equation_names
for name, inline_eq, class_eq in zip(
    inline_chain_dae.equation_names,
    inline_chain_dae.equations,
    class_chain_dae.equations,
):
    inline_raw = inline_eq.expr if hasattr(inline_eq, "expr") else inline_eq
    class_raw = class_eq.expr if hasattr(class_eq, "expr") else class_eq
    diff = sp.simplify(sp.expand(inline_raw - class_raw))
    assert diff == 0, f"{name}: inline vs class disagree, diff = {diff}"
# -


# ## 3. Transfer to `SystemModel`
#
# :meth:`SystemModel.from_pdesystem` walks the chain DAE rows (which
# are tagged ``Expression`` objects after step 13 of derive_model)
# and assembles the canonical operator matrices via
# :func:`collect_solver_tag`:
#
# * **time_derivative** tags → coefficients of $\partial_t Q$ → mass
#   matrix $M(Q)$ (extracted via the linearised pencil).
# * **flux** tags → flux vector $F(Q)$ (rows of the conservative
#   $\partial_x F$ contributions).
# * **hydrostatic_pressure** tags → flux-like $P(Q)$ slot
#   (rendered separately so a solver can handle hydrostatic
#   equilibrium specially).
# * **nonconservative_flux** tags → non-conservative product matrix
#   $B(Q)$.
# * **source** tags → algebraic right-hand side $S(Q)$ (the catch-all
#   for terms that don't fit the other categories — e.g. parameter
#   spatial derivatives like $\partial_x b$, raw state coupling
#   $W_k$, etc.).

# +
sm = SystemModel.from_pdesystem(class_chain_dae)
sm.describe(full=True)
# -


# ### 3.1 Vector / pencil form
#
# Linearising the chain DAE around the symbolic state collapses it to
# the **pencil form**
#
# $$
# \mathbf{M}_t\,\partial_t \delta Q
# + \mathbf{M}_x\,\partial_x \delta Q
# + \mathbf{M}_0\,\delta Q \;=\; 0,
# $$
#
# with $Q = [h,\,U_0,\,U_1,\,W_0,\,W_1,\,P_0,\,P_1]^\top$.  Each
# pencil matrix maps to a chunk of $\sm.describe$:
#
# * $\mathbf{M}_t$ matches ``sm.mass_matrix`` — five non-zero rows
#   (the evolution rows, with state-coefficient entries from
#   $\partial_t(h\,U_k)$ etc.) plus two all-zero rows for the
#   algebraic ``cont_jk`` constraints.
# * $\mathbf{M}_x$ = $\partial F/\partial Q + B(Q)$ — the spatial
#   Jacobian, combining the flux derivative and the non-conservative
#   product matrix.
# * $\mathbf{M}_0$ = $-\partial S/\partial Q$ plus algebraic
#   couplings.

# +
from zoomy_core.analysis.linearisation import linearise
from zoomy_core.analysis.pencil import extract_quasilinear_pencil

base_state_sym = {f: f for f in class_chain_dae.fields}
linearised = linearise(class_chain_dae, base_state_sym)
M_t, M_xa, M_0 = extract_quasilinear_pencil(linearised)
M_x = M_xa[0]
# -

# State vector $Q$ (column).
Q_vec = sp.Matrix(list(class_chain_dae.fields))
Q_vec

# Mass matrix $\mathbf{M}_t$.
M_t

# Spatial coefficient $\mathbf{M}_x$ (= $\partial F/\partial Q + B(Q)$).
M_x

# Zero-derivative coefficient $\mathbf{M}_0$.
M_0


# ### 3.2 Verify VAM(1, 2, 2) — analytical dispersion at rest
#
# Linearise the chain DAE around the rest state
# $\big(h = h_0,\;U_k = W_k = P_k = 0,\;b = 0\big)$ and assemble the
# dispersion matrix
#
# $$
# \mathbf{M}(\omega, k) \;=\; -i\omega\,\mathbf{M}_t \;+\; ik\,\mathbf{M}_x \;+\; \mathbf{M}_0.
# $$
#
# The vanishing of $\det\mathbf{M}(\omega, k)$ defines the dispersion
# curves $\omega(k)$.  Sympy solves it in closed form for VAM(1, 2, 2);
# the leading-order limit recovers shallow water, and the finite-$k$
# correction is the Boussinesq-style Padé fraction the higher modes
# are designed to produce.

# +
omega = sp.Symbol("omega", real=True)
k = sp.Symbol("k", real=True, positive=True)
h0 = sp.Symbol("h0", positive=True)

rest = {f: (h0 if f.func.__name__ == "h" else sp.S.Zero)
        for f in class_chain_dae.fields}
linearised_rest = linearise(class_chain_dae, rest)
M_t_rest, M_xa_rest, M_0_rest = extract_quasilinear_pencil(linearised_rest)
M_x_rest = M_xa_rest[0]

b_zero = {sp.Derivative(b, x): sp.S.Zero,
          sp.Derivative(b, x, x): sp.S.Zero}
M_t_rest = M_t_rest.subs(b_zero)
M_x_rest = M_x_rest.subs(b_zero)
M_0_rest = M_0_rest.subs(b_zero)

g_param = next(s for s in class_chain_dae.parameters if str(s) == "g")
rho_param = next(s for s in class_chain_dae.parameters if str(s) == "rho")

M_disp = -sp.I * omega * M_t_rest + sp.I * k * M_x_rest + M_0_rest
det_disp = sp.factor(M_disp.det(method="berkowitz"))
omega_solutions = sp.solve(sp.Eq(det_disp, 0), omega)
nontrivial = [s for s in omega_solutions if s != 0]

# Long-wave limit $k \to 0$: shallow-water recovery $c^2 \to g\,h_0$.
c_sq = sp.simplify((nontrivial[0] / k) ** 2)
c_long = sp.limit(c_sq, k, 0)
assert sp.simplify(c_long - g_param * h0) == 0, (
    f"Long-wave limit failed: c² → {c_long}, expected g·h_0"
)
# -

# Characteristic polynomial $\det\mathbf{M}(\omega, k)$, factored.
det_disp

# Symbolic dispersion solutions $\omega(k)$.
sp.Matrix(omega_solutions)

# Phase speed squared $c^2(k) = (\omega/k)^2$ for the surface modes.
c_sq


# ## 4. Pressure splitting
#
# :func:`split_for_pressure` turns the chain DAE into three
# :class:`SystemModel` sub-systems sharing the same 7-state Q vector
# but updating different subsets via rectangular operators.  The
# pipeline ``Q → Q_1 → Q_2 → Q_3 = Q^{n+1}`` is implicit: each stage
# overwrites only its indexed state entries, others pass through
# verbatim.

# +
fields_by_name = {f.func.__name__: f for f in class_chain_dae.fields}
P_0 = fields_by_name["P_0"]
P_1 = fields_by_name["P_1"]
dt = sp.Symbol(r"\Delta t", positive=True)

split = split_for_pressure(class_chain_dae, [P_0, P_1], dt)
# -


# ### 4.1 Predictor stage — `SM_pred`
#
# Five evolution equations (mass + 2 xmom + 2 zmom) updating
# $Q[0..4] = (h, U_0, U_1, W_0, W_1)$ with $P_k$ frozen at the previous
# time step $P_k^n$.  The pressure entries pass through unchanged.

split.SM_pred.describe(full=True)


# ### 4.2 Pressure stage — `SM_press`
#
# Two algebraic equations (the elliptic block in $(P_0, P_1)$) updating
# $Q[5..6]$.  Mass matrix is all-zero; the elliptic structure lives in
# the source slot.  ``verify_p_linearity`` checks that the rows are
# strictly linear in $(P_l,\,\partial_x P_l,\,\partial_{xx} P_l)$.

split.SM_press.describe(full=True)

# +
elliptic_rows = {
    int(name[len("elliptic_j"):]): -split.SM_press.source[i, 0]
    for i, name in enumerate(split.SM_press.equation_names)
}
linearity = verify_p_linearity(elliptic_rows, [P_0, P_1], x)

# Strict-linearity check passed — every row is in the linear span of
# the six pressure atoms.
assert set(linearity["coefficients"].keys()) == {1, 2}
# -


# ### 4.3 Corrector stage — `SM_corr`
#
# Four algebraic update equations
# $Q_k = Q_k^{*} - (\Delta t / h)\,T_{*}[k](P^{n+1})$
# updating $Q[1..4] = (U_0, U_1, W_0, W_1)$.  The corrector source
# entries embed the $T_u$ / $T_w$ pressure-source expressions; $h$ and
# $P_k$ pass through unchanged.

split.SM_corr.describe(full=True)


# That closes the pipeline:
# **chain primitives → PDESystem → SystemModel → predictor / pressure / corrector**.
# Each sub-system is a first-class :class:`SystemModel` carrying its own
# rectangular ``mass_matrix``, ``flux``, ``source``, and an explicit
# ``equation_to_state_index`` telling the solver which Q entries the
# stage updates.
