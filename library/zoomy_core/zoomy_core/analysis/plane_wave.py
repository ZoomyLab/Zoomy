"""Plane-wave dispersion analysis.

Inserts the 1D plane-wave ansatz `δq(t, x) = q̂ exp(i(k x − ω t))`
into a linearised `SystemModel`'s operator-form residual and
reduces to an algebraic system in the amplitudes `q̂`.  Solves
`det M(ω, k) = 0` for `ω(k)` (or `k(ω)`).
"""
from __future__ import annotations

from typing import Optional

import sympy as sp


def _residuals_from_linearised(linear_sm):
    """Reconstruct the linearised residual row-by-row by re-using the
    SystemModel's operator-form sum ``M·∂_t δq + ∂_x F + ∂_x P + B·∂_x
    δq − S``.
    """
    return linear_sm.reconstruct_residuals()


def plane_wave_matrix(linear_sm, *,
                      k: Optional[sp.Symbol] = None,
                      omega: Optional[sp.Symbol] = None,
                      axis: int = 0):
    """Insert `δq → q̂ exp(i(k x_axis − ω t))` and reduce to a matrix.

    `linear_sm` must be a SystemModel returned by `linearise`
    (state entries are the perturbation symbols `δq`).

    Returns
    -------
    M : sp.Matrix
        Coefficient matrix such that `M · q̂_vector = 0`.
    amplitudes : list[sp.Symbol]
        Amplitude symbols, in the same order as `linear_sm.state`.
    """
    if k is None:
        k = sp.Symbol("k", real=True)
    if omega is None:
        omega = sp.Symbol("omega", real=True)

    t_sym = linear_sm.time
    x_axes = list(linear_sm.space)
    if not (0 <= axis < len(x_axes)):
        raise ValueError(f"axis={axis} out of range for space={x_axes}")
    x_sym = x_axes[axis]
    I = sp.I
    E = sp.exp(I * (k * x_sym - omega * t_sym))

    # Reconstruct residuals; convert state Symbols to coordinate-dependent
    # Functions so derivatives are well-defined.
    residuals = _residuals_from_linearised(linear_sm)

    # Build amplitude symbols.  Each state entry δq_i has an amplitude
    # ``q̂_i``.  We need to substitute the state's Function form into the
    # residual; reconstruct_residuals already returns Function-form.
    amplitudes = []
    repl = {}
    state = list(linear_sm.state)
    for s in state:
        name = str(s).replace(r"\delta ", "").replace("delta ", "")
        amp = sp.Symbol(name + "_hat")
        amplitudes.append(amp)
        # The Function form is what reconstruct_residuals produces.
        delta_func = sp.Function(str(s), real=True)(t_sym, *x_axes)
        repl[delta_func] = amp * E

    pw_eqs = []
    for res in residuals:
        res_sub = res.xreplace(repl).doit()
        eq_div = sp.simplify(res_sub / E)
        pw_eqs.append(eq_div)

    M, _ = sp.linear_eq_to_matrix(pw_eqs, amplitudes)
    return M, amplitudes


def plane_wave_dispersion(linear_sm, *,
                          k: Optional[sp.Symbol] = None,
                          omega: Optional[sp.Symbol] = None,
                          axis: int = 0,
                          solve_for: str = "omega",
                          simplify: bool = True,
                          factor_in_target: bool = True):
    """Full dispersion solve.

    Returns a dict with:

    ```
        matrix       — sp.Matrix M(ω, k)
        amplitudes   — list of q̂ symbols
        determinant  — det M(ω, k)
        solutions    — list of ω(k) (or k(ω)) solutions
        phase_velocity_solutions — [ω/k for ω in solutions] (omega-mode only)
    ```
    """
    if k is None:
        k = sp.Symbol("k", real=True)
    if omega is None:
        omega = sp.Symbol("omega", real=True)

    M, amps = plane_wave_matrix(linear_sm, k=k, omega=omega, axis=axis)
    det = M.det(method="berkowitz")
    if simplify:
        det = sp.simplify(det)
    if factor_in_target:
        det = sp.factor(det, omega if solve_for == "omega" else k)

    target = omega if solve_for == "omega" else k
    sols = sp.solve(sp.Eq(det, 0), target)

    out = {
        "matrix": M,
        "amplitudes": amps,
        "determinant": det,
        "solutions": sols,
    }
    if solve_for == "omega":
        out["phase_velocity_solutions"] = [sp.simplify(s / k) for s in sols]
    return out
