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
# ## 4.5 Paper inconsistency (Escalante 2024 eq 6)
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
