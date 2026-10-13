"""Generalised-eigenvalue (pencil) form for analysis.

A linearised PDE system can be written in the form

    M_t · ∂_t δq + M_xa · ∂_xa δq + M_0 · δq = 0     (sum over axis a)

where `M_t`, `M_xa`, `M_0` are constant matrices (coefficients
evaluated at the base state).  Plane-wave ansatz ``δq = q̂ e^(i(k·x −
ωt))`` reduces this to

    [-iω M_t + i k · (Σ_a n_a M_xa) + M_0] q̂ = 0

where `n` is the normal direction (`Σ n_a² = 1`).  Dividing by `ik`
and writing `λ = ω/k` gives the **generalised eigenvalue problem**

    [Σ_a n_a M_xa + (1 / (ik)) M_0] q̂ = λ M_t q̂.

For the principal symbol (high-k limit, or systems with M_0 = 0):

    A_n q̂ = λ M_t q̂          A_n := Σ_a n_a M_xa.

This module extracts `(M_t, [M_xa], M_0)` mechanically from a
linearised `SystemModel` and computes the generalised
eigenvalues either symbolically (`sp.solve(charpoly)`) or
numerically (`scipy.linalg.eig`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Pencil extraction
# ---------------------------------------------------------------------------

def extract_quasilinear_pencil(linear_sm
                              ) -> Tuple[sp.Matrix, List[sp.Matrix], sp.Matrix]:
    """Extract `(M_t, [M_xa], M_0)` from a *linearised* SystemModel.

    Each entry of the linearised SystemModel's operator slots is
    expected to be linear in the perturbation state.

    Returns
    -------
    M_t    : (n_eq × n_state) sp.Matrix — coefficient of ∂_t δq_j.
    M_xa   : list of (n_eq × n_state) sp.Matrix — one per spatial axis.
    M_0    : (n_eq × n_state) sp.Matrix — coefficient of δq_j (no derivative).
    """
    n_eq = linear_sm.n_equations
    n_state = linear_sm.n_state
    n_dim = linear_sm.n_dim
    state = list(linear_sm.state)

    # M_t straight from the linearised mass matrix.
    M_t = sp.Matrix(linear_sm.mass_matrix)

    # Spatial: M_xa[a][i, j] = ∂(F[i, a] + P[i, a])/∂(δq_j) + B[i, j, a].
    M_xa = [sp.zeros(n_eq, n_state) for _ in range(n_dim)]
    for a in range(n_dim):
        for i in range(n_eq):
            f_ia = sp.sympify(linear_sm.flux[i, a])
            p_ia = sp.sympify(linear_sm.hydrostatic_pressure[i, a])
            for j in range(n_state):
                jac = sp.diff(f_ia, state[j]) + sp.diff(p_ia, state[j])
                b_ija = sp.sympify(linear_sm.nonconservative_matrix[i, j, a])
                M_xa[a][i, j] = sp.expand(jac + b_ija)

    # M_0: residual sign is ``... − S``; so M_0[i, j] = −∂S[i]/∂(δq_j).
    M_0 = sp.zeros(n_eq, n_state)
    for i in range(n_eq):
        s_i = sp.sympify(linear_sm.source[i, 0])
        for j in range(n_state):
            M_0[i, j] = sp.expand(-sp.diff(s_i, state[j]))

    return M_t, M_xa, M_0


# ---------------------------------------------------------------------------
# Generalised eigenvalues — symbolic
# ---------------------------------------------------------------------------

def generalised_eigenvalues(M_x: sp.Matrix, M_t: sp.Matrix,
                            *, lam: Optional[sp.Symbol] = None,
                            simplify: bool = True) -> List[sp.Expr]:
    """Symbolic generalised eigenvalues of the pencil `(M_x, M_t)`.

    Solves `det(M_x − λ M_t) = 0` for λ.  Returns a list of solutions.

    Caveat — symbolic charpoly degree blows up fast.  For larger systems
    use `sample_generalised_eigenvalues` (numerical).
    """
    if lam is None:
        lam = sp.Symbol("lambda")
    n = M_t.shape[0]
    if M_x.shape != M_t.shape:
        raise ValueError(f"shape mismatch: M_x={M_x.shape}, M_t={M_t.shape}")
    char = (M_x - lam * M_t).det(method="berkowitz")
    if simplify:
        char = sp.expand(char)
    return sp.solve(sp.Eq(char, 0), lam)


# ---------------------------------------------------------------------------
# Generalised eigenvalues — numerical, with parameter sampling
# ---------------------------------------------------------------------------

def _eval_matrix_numerically(M: sp.Matrix, sub: Dict, dtype=complex):
    """Evaluate a sympy matrix at numeric values of all symbols, returning numpy."""
    M_sub = M.subs(sub)
    arr = np.array(M_sub.tolist(), dtype=dtype)
    return arr


def symbolic_eigenvalues_at(sm, base_state: Dict, *,
                            axis: int = 0,
                            simplify: bool = True
                            ) -> List[sp.Expr]:
    """One-shot helper: linearise `sm` (a SystemModel) at
    `base_state`, extract the principal-symbol pencil
    `(M_x_axis, M_t)`, and return the symbolic generalised
    eigenvalues.

    Args:
        sm:          a `SystemModel`.
        base_state:  dict `{state_sym: value}` for every state entry.
        axis:        which spatial direction's pencil to use (default 0).
        simplify:    apply `sp.expand` to the characteristic poly.

    Returns:
        list of symbolic eigenvalues.
    """
    from .linearisation import linearise
    sm_lin = linearise(sm, base_state)
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)
    return generalised_eigenvalues(M_xa[axis], M_t, simplify=simplify)


def sample_generalised_eigenvalues(M_x: sp.Matrix, M_t: sp.Matrix,
                                  parameter_samples: List[Dict],
                                  *, dtype=complex,
                                  drop_infinite: bool = False
                                  ) -> List[np.ndarray]:
    """For each sample (a dict of symbolic-value → numeric-value), return
    the numerical generalised eigenvalues of `(M_x, M_t)` at that
    sample.

    Uses `scipy.linalg.eig(A, B)` which solves `A v = λ B v` even
    when `B` is singular (in which case some λ are returned as
    `inf`).  `drop_infinite=True` filters those out.
    """
    try:
        from scipy.linalg import eig as scipy_eig
    except ImportError:                                # pragma: no cover
        scipy_eig = None

    out = []
    for sub in parameter_samples:
        A = _eval_matrix_numerically(M_x, sub, dtype=dtype)
        B = _eval_matrix_numerically(M_t, sub, dtype=dtype)
        if scipy_eig is not None:
            lam = scipy_eig(A, B, left=False, right=False)
        else:
            # Fallback: use np.linalg.eig on B^-1 A (fails if B singular).
            try:
                Binv = np.linalg.inv(B)
                lam = np.linalg.eigvals(Binv @ A)
            except np.linalg.LinAlgError:
                # Add small perturbation, try again.
                lam = np.linalg.eigvals(np.linalg.solve(B + 1e-12 * np.eye(B.shape[0]), A))
        if drop_infinite:
            lam = lam[np.isfinite(lam)]
        out.append(lam)
    return out
