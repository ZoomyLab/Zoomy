# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 004 — Open questions
#
# Things that don't fully match yet, or where the "correct" answer
# requires a discussion.

# %% [markdown]
# ## 4.1 VAM hyperbolicity vs Escalante 2024 eq (12)
#
# The paper claims the underlying-hyperbolic VAM system (eq 7) has
# the eigenvalues
#
# $$
# \lambda_1 = u_0,\quad
# \lambda_{2,3} = u_0 \pm \frac{u_1}{\sqrt{3}},\quad
# \lambda_{4,5} = u_0 \pm \sqrt{gH + u_1^2},\quad \lambda_6 = 0.
# $$
#
# These are eigenvalues of $A(U, w_2) = J_F + G(U, w_2)$, where the
# **paper treats $w_2$ as a free input parameter** (not a function of
# $U$ via the closure $w_2 = -(w_0+w_1) + (u_0+u_1)\,\partial_x b$).
#
# In our framework with `eliminate_closures=True`, the closure is
# substituted into the equations; the chain rule of $w_2(w_0, w_1, u_0, u_1)$
# propagates through the Jacobian, giving a **different** matrix and
# different eigenvalues.
#
# In our framework with `eliminate_closures=False`, the closure is a
# separate algebraic equation; the resulting pencil $(M_x, M_t)$ is
# **singular** — $\det(M_x - \lambda M_t) \equiv 0$ — so eigenvalues
# can't be read off directly.

# %%
import os, sys, sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(), "..", "tutorials", "vam"))
from vam_builder import build_vam_pde_system, h, b
from zoomy_core.analysis import linearise, extract_quasilinear_pencil, reduce_singular_pencil, generalised_eigenvalues

H = sp.Symbol("H", positive=True)
g = sp.Symbol("g", positive=True)
sys_vam, *_ = build_vam_pde_system(1, 2, eliminate_closures=False, hyperbolic_predictor=True)
sys_vam = sys_vam.with_substitutions({b: -H})
H_b = sp.Symbol("H_b", positive=True)
U_0, U_1, W_0, W_1, W_2 = sp.symbols("U_0 U_1 W_0 W_1 W_2", real=True)
base = {sys_vam.fields[0]: H_b}
for f, sym in zip(sys_vam.fields[1:], [U_0, U_1, W_0, W_1, W_2]):
    base[f] = sym
sys_lin = linearise(sys_vam, base)
M_t, M_xa, M_0 = extract_quasilinear_pencil(sys_lin)
print(f"Pencil shape {M_t.shape}, M_t rank = {M_t.rank()}")

lam = sp.Symbol("lam")
char = sp.expand((M_xa[0] - lam*M_t).det(method="berkowitz"))
print(f"Singular check: det(M_x − λ M_t) = {char}")

# %% [markdown]
# ### Generic pencil reduction
#
# Our `reduce_singular_pencil` algorithm iteratively eliminates
# algebraic-constraint rows: a row with `M_t row = 0` carries a
# constraint at high $k$ (in `M_x`) or at $k=0$ (in `M_0`); use it
# to solve for one field and substitute into the rest.

# %%
M_x_red, M_t_red, fields_red = reduce_singular_pencil(
    M_xa[0], M_t, sys_lin.fields, M_0=M_0, verbose=True
)
print(f"After reduction: pencil {M_x_red.shape}, rank(M_t) = {M_t_red.rank()}")
char_red = sp.expand((M_x_red - lam * M_t_red).det(method="berkowitz"))
print(f"char poly degree in λ: {sp.Poly(char_red, lam).total_degree()}")

# %% [markdown]
# **Status: partial.**  The reduction handles `M_t row = 0` cases but
# does NOT pre-apply the cont $j=0$ → $\partial_t h$ substitution that
# would expose additional algebraic rows (cont $j\ge 1$ look
# evolution-y because $\omega$ carries $\partial_t h$ inside the IBP
# integrand, but become algebraic after the substitution).
#
# Even after partial reduction, the eigenvalues don't match paper
# eq (12) — because the paper treats $w_2$ as a *free parameter*
# rather than substituting via the closure (which is what our
# reduction does).
#
# **Open question.** What's the right semantics?
#
# 1. **Paper's reading.** Build the 5×5 matrix $A(U, w_2)$ over
#    $(h, u_0, u_1, w_0, w_1)$, treating $w_2$ as an opaque coefficient
#    (like another parameter alongside $g$).  Eigenvalues are
#    independent of $w_2$ in this case (the structure of $J_F + G$
#    happens to make them depend only on $h, u_0, u_1$).
#
# 2. **Our reading (constrained).** Substitute $w_2$ via the closure;
#    the Jacobian acquires chain-rule terms.  This gives the
#    eigenvalues of the system *restricted to physical states*.
#
# Both are consistent — they answer different questions.
# Tomorrow's discussion: probably we want a `keep_as_parameter=[w_2]`
# flag to switch between the two readings, while making the paper's
# semantics the default.

# %% [markdown]
# ## 4.2 Augmented SME degeneracy
#
# Same pattern as VAM: keep $w$ as a state polynomial (instead of
# substituting via depth-integrated continuity).  The augmented pencil
# is degenerate; finite eigenvalues do not match the standard SME
# without further reduction.  Test in
# `tutorials/analysis/sme_augmented_eigenvalue_test.py`.
#
# Conclusion: **keeping a constrained variable as state vs. substituting
# its closure gives different pencils for spectral analysis.**  This is
# not a flaw of either approach — it's a genuine ambiguity in the
# setup.

# %% [markdown]
# ## 4.3 Principal symbol vs finite-k
#
# `sample_hyperbolicity` evaluates the pencil $(M_x, M_t)$ in the
# **principal-symbol limit** ($k \to \infty$, where the $M_0/(ik)$
# correction vanishes).  This is fine for asymptotic hyperbolicity but
# misses dispersive corrections at finite $k$.  For VAM the dispersion
# *is* finite-k via $M(\omega, k)$ in `plane_wave_dispersion`, so
# wave-celerity analysis is correct; but the hyperbolicity sampler is
# strictly principal-symbol.
#
# Future enhancement: add a `k_value=` parameter to `sample_hyperbolicity`
# that includes the $M_0/(ik)$ term.

# %% [markdown]
# ## 4.4 ML-VAM (multilayer × VAM) — design only
#
# Composing the Aguillon multilayer Heaviside basis with the per-layer
# VAM polynomial basis gives an "ML-VAM" model that hasn't been
# derived yet.  Architectural sketch in
# `tutorials/multilayer/ml_sme_prototype.py`.  Open question: does the
# resulting Jacobian inherit the VAM eigenvalue structure layer-by-layer,
# or do the inter-layer mass exchanges $G_{\alpha+1/2}$ couple the
# spectra non-trivially?

# %% [markdown]
# ## 4.5 Standard literature for singular pencils
#
# **Kronecker Canonical Form (KCF)** — Gantmacher, *Theory of Matrices*
# (1959).  Decomposes any pencil $A - \lambda B$ (singular or regular)
# into block-diagonal canonical form with four block types:
#
# * **Regular Jordan blocks** — finite eigenvalues.
# * **Regular nilpotent blocks** — infinite eigenvalues (rank-deficient $B$).
# * **Right minimal indices ($L_r$)** — undetermined columns; correspond
#   to free state directions.
# * **Left minimal indices ($L_c$)** — redundant rows; correspond to
#   linearly-dependent equations.
#
# The finite eigenvalues are **uniquely determined** by the pencil,
# regardless of which row/column elimination one performs.
#
# Algorithms:
# * Van Dooren (1979): "The computation of Kronecker's canonical form
#   of a singular pencil", Linear Algebra Appl.
# * Beelen & Van Dooren (1988): improved $O(m^2 n)$ algorithm.
# * GUPTRI software (Demmel & Kågström): the de-facto numerical standard.
# * SciPy's `scipy.linalg.eig(A, B)` (QZ algorithm) handles
#   rank-deficient $B$ but doesn't extract the regular part of a
#   *truly singular* pencil cleanly.
#
# **DAE index reduction** — Pantelides (1988), Pryce $\Sigma$-method,
# Mattsson & Söderlind (1993) "dummy derivatives".  Differentiates
# constraints + applies dummy derivatives to reach an index-1 form,
# then standard ODE/eigenvalue analysis applies.  Effectively the
# "substitute closures" route in our terminology.

# %% [markdown]
# ## 4.6 Combinatorial elimination experiment
#
# We tried the user's suggestion — start from the full augmented
# pencil and find a regular sub-pencil by enumerating row/column
# subsets.  Code in `tutorials/analysis/kcf_brute_subpencil.py`.
#
# **Result.** The brute-force *finds* a regular sub-pencil, but the
# eigenvalues depend on which rows/columns one chooses to drop.  For
# augmented SME L=1 it returned eigenvalues
# $U_0 - U_1, 2U_0 \pm 2\sqrt{3}U_1/3$ — different from the standard
# SME $U_0, U_0 \pm \sqrt{gH + U_1^2}$.
#
# **Why.** Different sub-pencils correspond to different *structural
# choices* of which fields are "primary state" and which are
# "constrained".  The KCF framework would extract a unique regular
# part, but a naive sub-pencil search doesn't.

# %% [markdown]
# ## 4.7 Gaussian-then-eliminate experiment
#
# Try Gaussian elimination on $M_t$ first (to expose hidden algebraic
# rows that arise only after combining evolution rows), then drop the
# resulting all-zero $M_t$ rows.  Code in
# `tutorials/analysis/kcf_gaussian_then_eliminate.py`.
#
# **Result.** Correctly identifies all 3 algebraic constraints in
# augmented SME L=1 (after Gaussian step `M_t` shows rows 3, 4, 5 with
# zero $M_t$).  But when eliminating fields, the column-elimination
# step propagates substitutions back into $M_t$ and breaks the
# structure — `rank(M_t)` drops from 3 to 1 over the iterations,
# leaving the wrong reduced pencil.
#
# Adding a `prefer_eliminate=[w_0, w_1, w_2]` hint helps but doesn't
# fully solve it: the algebraic-row coefficients on $w_i$ may be
# zero (the algebraic content is on $h, u_0, u_1$ at that step), so
# the algorithm still picks $h$ or $u_0$ to eliminate.
#
# **Conclusion.** A **proper KCF implementation** is needed for the
# generic case — implementing Beelen-Van Dooren symbolically is
# probably ~200 lines.  Alternative: for paper-matching specifically,
# specify "input variables" (option II below) — much simpler.

# %% [markdown]
# ## 4.8 Recommended path forward
#
# Given the above, two routes:
#
# 1. **Generic (slow road)**: implement symbolic KCF (Beelen-Van Dooren
#    in SymPy).  Produces unique regular-part eigenvalues for any
#    PDE system with arbitrary algebraic constraints.  Probably 1-2
#    days of careful work.
#
# 2. **Practical (fast road)**: add a `keep_as_input=[...]` parameter
#    to `vam_builder`/`sme_builder` that drops the listed fields from
#    the state vector AND drops the associated algebraic equations,
#    treating those fields as opaque coefficients in the remaining
#    matrices.  This gives the paper's eq (12) eigenvalues for VAM
#    when called with `keep_as_input=[w_2]`, and recovers standard
#    SME from augmented SME with `keep_as_input=[w_0, w_1, w_2]`.
#    ~30 lines of code; tomorrow's discussion item.
#
# Most users will care about (2); (1) is for completeness and matters
# only if/when we hit a system where the input-vs-state choice isn't
# obvious.

# %% [markdown]
# ## 4.9 Augmented vs standard SME L=2 — eigenvalue discrepancy
#
# **Setup.** Build SME L=2 in two ways:
#
# 1. **Standard** (`w_mode='from_continuity'` in the projection): w is
#    depth-integrated from continuity, no algebraic constraints,
#    pencil shape 4×4.
# 2. **Augmented** (`w_mode='state'`, w polynomial of degree N_w=3):
#    w_0..w_3 are state variables, plus continuity j=1..3 + KBC bottom
#    as algebraic equations.  Pencil shape 8×8.
#
# Then "fully eliminate" the augmented form (drop all w_i as inputs +
# drop all algebraic equations).  This SHOULD reduce to the standard
# 4×4 (h, u_0, u_1, u_2) pencil.
#
# **Experiment** (`tutorials/analysis/sme_l2_partial_elimination.py`):
#
# At rest (U=0): standard and fully-eliminated augmented eigenvalues
# agree exactly: ±√(gH).
#
# At non-rest (e.g. U_1=2, U_2=2):
# * standard:               ±3.72, +0.34, +1.48, +4.76
# * fully-eliminated aug:   ±3.78, −0.70, +1.59, +5.63
#
# **The eigenvalues differ — even though the elimination should make
# the augmented form identical to the standard.**
#
# Investigation: my σ-coord projection of x-momentum has extra
# `(u_0 − u_1/3) ∂_t h` and similar terms inside the IBP integrand
# of `+2 ∫ω u dξ` (specifically from the `-ξ ∂_t h` piece of ω).
# After full elimination at the augmented level, these survive as
# extra coefficients that the standard projection (where w is
# already substituted via depth-integrated continuity) doesn't have.
#
# The standard formulations of K&T 2019 (SME) and Escalante 2024
# (VAM, eq 4) appear to apply additional algebraic identities (likely
# continuity j=0 to substitute ∂_t h, OR some other reduction) that
# we haven't replicated in the σ-coord projection module.
#
# **Same root cause** explains why my VAM (1,2) hyperbolic-predictor
# matrix doesn't match paper eq (12) exactly — the `±u_1/√3`
# eigenvalues match (those come from the (w_0, w_1) block which is
# unaffected), but `±√(gH+u_1²)` is off by O(u_1²).
#
# **Open work item.** Identify the algebraic reduction the standard
# formulations apply and propagate it through `GalerkinProjection`.
# Likely candidates:
# * Apply cont j=0 → ∂_t h substitution INSIDE the IBP integrand
#   before projection.
# * Use IBP with ``∂_t(h φ_j u)`` instead of `h ∂_t(φ_j u)`.
# * Retain stress σ_xz and use viscous boundary condition to absorb
#   the `-ξ ∂_t h` piece.

# %% [markdown]
# ## 4.10 Paper inconsistency (Escalante 2024 eq 6)
#
# The paper writes the compact form $A(U, w_2) = J_F + G$ with
# $U = (h, hu_0, hu_1, hw_0, hw_1)^T$, but the F, G, T vectors are
# listed in the order (continuity, x-mom $j=0$, **z-mom $j=0$**,
# **x-mom $j=1$**, z-mom $j=1$) — rows 3 and 4 are swapped relative to
# the stated $U$.  The math is consistent (same ordering used
# throughout F, G, T), but the U-vector text doesn't match.
#
# Our derivation matches each $T_i$ to its physical equation by hand
# (not by row index), so we sidestepped this bug.  Documented in our
# correspondence and in the original `escalante2024_poisson.py`
# comments.
