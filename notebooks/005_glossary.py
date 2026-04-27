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
# # 005 — Glossary of technical terms
#
# Quick reference for the terminology used in 001–004.
#
# ## Models
#
# **SWE** — Shallow Water Equations.  $\partial_t h + \partial_x(hu) = 0$;
# $\partial_t(hu) + \partial_x(hu^2 + gh^2/2) = 0$.  Recovered as SME
# at level $L=0$.
#
# **SME** — Shallow Moment Equations (Kowalski–Torrilhon 2019).  Galerkin
# projection of σ-coord shallow water with a polynomial velocity ansatz
# $u = \sum_{i=0}^L u_i\, \varphi_i(\xi)$.  Hydrostatic; closes
# automatically because $w$ is determined by depth-integrated continuity.
#
# **VAM** — *Vertically Averaged and Moment* equations
# (Escalante–Morales de Luna–Cantero-Chinchilla–Castro-Orgaz 2024).
# Non-hydrostatic generalisation of SME: keeps the z-momentum equation
# and a non-hydrostatic pressure remainder $p$ split off the
# hydrostatic part $p_H = -gh(\xi-1)$.  Our ansatz: $u \in P_M[\xi]$,
# $w, p \in P_N[\xi]$.  Paper case: $(M, N) = (1, 2)$.
#
# **ML-SWE** — Multilayer Shallow Water (Aguillon–Hörnschemeyer–Sainte-Marie
# 2026).  Heaviside basis in $z$: each layer is a piecewise-constant
# region.  Mass exchanges $G_{\alpha+1/2}$ at interfaces from kinematic
# BCs.
#
# **ML-VAM** — composition of multilayer + VAM (per-layer Legendre
# basis with non-hydrostatic pressure and z-momentum).  Architectural
# sketch only; not derived yet.
#
# ## Coordinates and basis
#
# **σ-coordinates** — change of variable $\xi = (z - b)/h$ mapping
# the physical column $z \in [b, b+h]$ to the unit interval $[0, 1]$.
# Used to project onto a basis defined on $[0, 1]$.
#
# **σ-vertical velocity** — $\omega(t, x, \xi) := w - \partial_t(\xi h + b)
# - u\,\partial_x(\xi h + b)$.  Kinematic BCs at top/bottom become
# $\omega(0) = \omega(1) = 0$.
#
# **Shifted Legendre polynomials** — $\varphi_i(\xi) = P_i(1 - 2\xi)$.
# Paper convention: $\varphi_i(0) = 1$, $\varphi_i(1) = (-1)^i$;
# $\int_0^1 \varphi_i^2\, d\xi = 1/(2i+1)$.  Orthogonal basis on $[0,1]$.
#
# **Galerkin projection** — multiply a PDE by $\varphi_j(\xi)$, integrate
# over $[0, 1]$, simplify.  Gives one equation per $j$.
#
# **Kinematic boundary condition (KBC)** — the bottom is a streamline:
# $\partial_t b + u|_b\,\partial_x b = w|_b$.  Equivalently in σ:
# $\omega|_{\xi=0} = 0$.  Same at the surface ($\omega|_{\xi=1} = 0$).
#
# **Surface boundary condition (BC) for $p$** — non-hydrostatic
# pressure vanishes at the free surface: $p|_{\xi=1} = 0$.  Algebraic
# closure for $p_N$.
#
# ## Analysis
#
# **Linearisation** — $q \to q_0 + \varepsilon\, \delta q$, expand to
# first order in $\varepsilon$.  Coefficients evaluate at the chosen
# base state $q_0$.
#
# **Plane-wave ansatz** — $\delta q = \hat q \exp(i(k x - \omega t))$.
# Reduces the linearised PDE to an algebraic system $M(\omega, k)\, \hat q = 0$.
#
# **Dispersion relation** — $\det M(\omega, k) = 0$, solved for $\omega(k)$.
# Phase velocity $C(k) = \omega/k$.
#
# **Quasilinear pencil** — $(M_t, M_x, M_0)$ such that
# $M_t \partial_t \delta q + M_x \partial_x \delta q + M_0 \delta q = 0$
# is the linearised system.  Generalised eigenvalue problem
# $M_x v = \lambda M_t v$ with $\lambda = \omega/k$.
#
# **Hyperbolicity** — all generalised eigenvalues real.  Necessary for
# well-posed time evolution.  *Strict* hyperbolicity additionally
# requires distinct eigenvalues.
#
# **Singular pencil** — $\det(M_x - \lambda M_t) \equiv 0$ for all $\lambda$
# (not just specific values).  Happens when $M_t$ and $M_x$ share a
# common kernel direction.  Indicates the system has *algebraic
# constraints* that must be eliminated before spectral analysis.
#
# ## Numerical schemes
#
# **Predictor–corrector splitting** — for VAM (Escalante 2024 §3.1): a
# two-step time advance.  Predictor solves the underlying hyperbolic
# system (eq 7, drops non-hydrostatic pressure) for $U^{(\tilde k)}$;
# corrector solves a Poisson-like system (eq 12) for $P^{(k)}$ that
# projects $U^{(\tilde k)}$ back onto the constraint manifold
# $I_1 = I_2 = 0$.
#
# **Finite-volume PVM scheme** — Polynomial Viscosity Matrix
# (Castro–Pares 2008): a path-conservative finite-volume scheme for
# non-conservative hyperbolic systems.  Used for the predictor in
# Escalante 2024.
#
# ## Validation references
#
# **Airy theory** — exact dispersion relation for linear gravity waves
# on water of finite depth: $C^2/(gH) = \tanh(kH)/(kH)$.
#
# **Padé approximant** — rational function with prescribed Taylor
# coefficients up to a given order.  VAM (1, 2) is the [2/4] Padé of
# Airy in $(kH)$; VAM (2, 3) is the next-order [4/6] Padé.
#
# ## Symbolic-derivation primitives (`zoomy_core.symbolic`)
#
# **Bug-3 closure** — historical name for the recurring need to apply
# the cont $j=0$ → $\partial_t h$ substitution everywhere $\omega$
# appears inside the IBP integrand.  Surfaces because $\omega$ carries
# $\partial_t h$ inside derivatives; the substitution becomes a
# fixpoint loop using `product_rule_forward`,
# `distribute_derivative_over_add`, `subst(dt_h_relation)`.
#
# **AutoEvalGuard** — context manager that raises if sympy fires
# automatic chain rule, IBP, or $.doit()$ on Derivative/Integral atoms
# inside its block.  Used to verify that derivations are explicit
# (every math step is a named primitive).
#
# ## Pencil theory
#
# **Matrix pencil** — a parametrised matrix family $A - \lambda B$.
# The generalised eigenvalue problem $A v = \lambda B v$ is the
# pencil's spectral problem.
#
# **Regular pencil** — $\det(A - \lambda B) \not\equiv 0$.  Has finite
# eigenvalues from the regular Jordan blocks plus possibly infinite
# eigenvalues if $B$ is rank-deficient.
#
# **Singular pencil** — $\det(A - \lambda B) \equiv 0$ identically.
# $A$ and $B$ share a common kernel direction.  Indicates the system
# has algebraic constraints + redundant equations that need
# elimination before the spectrum can be read off.
#
# **Kronecker Canonical Form (KCF)** — Gantmacher 1959.  The unique
# block-diagonal canonical form of a pencil under strict equivalence:
# regular Jordan blocks (finite eigenvalues), regular nilpotent blocks
# (infinite eigenvalues), $L_r$ blocks (right minimal indices —
# undetermined columns), $L_c$ blocks (left minimal indices —
# redundant rows).
#
# **Right minimal indices ($L_r$)** — degrees of freedom in the
# solution that are not constrained by the pencil.  Correspond to
# "input variables" in DAE language.
#
# **Left minimal indices ($L_c$)** — linearly-dependent equations.
# Can be dropped without losing information.
#
# **Generalised Schur decomposition / QZ algorithm** — Moler & Stewart
# 1973.  Numerical algorithm that brings $(A, B)$ to upper-triangular
# form simultaneously via unitary transformations.  Reads off
# eigenvalues from the diagonal.  Doesn't fully handle truly singular
# pencils; needs KCF reduction first.
#
# **GUPTRI** — Demmel & Kågström.  Numerical software for KCF
# reduction.  De-facto standard for singular-pencil spectral analysis.
#
# **Beelen-Van Dooren algorithm** (1988) — $O(m^2 n)$ KCF reduction
# using only unitary transformations; numerically stable.
#
# ## DAE theory
#
# **Differential-algebraic equation (DAE)** — system of the form
# $F(x, \dot x, t) = 0$ where $\partial F / \partial \dot x$ may be
# rank-deficient.  Mixes differential and algebraic equations.
#
# **Index** — the number of differentiations needed to reduce a DAE
# to an explicit ODE.  Index 0 = ODE; index 1 = "easy" DAE; index 2+
# = harder.
#
# **Pantelides algorithm** (1988) — graph-theoretical method for
# index reduction: identifies equations to differentiate so the
# system becomes index 1.
#
# **Dummy derivatives** — Mattsson & Söderlind 1993.  After
# differentiating constraints, replace some derivatives by
# "dummy" algebraic variables to keep dimensions matched.
#
# **Pryce $\Sigma$-method** — alternative to Pantelides; uses
# signature matrix (Σ) of derivative orders.  Tends to be more
# robust on tricky systems.
