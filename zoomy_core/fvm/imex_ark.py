"""IMEX additive-Runge-Kutta integrator for index-1 DAEs.

Implements Ascher-Ruuth-Spiteri (1997, DOI 10.1016/S0168-9274(97)00056-1)
ARS232 and ARS343 schemes for systems of the form

    M_t · y'  =  f_E(t, y)  +  f_I(t, y),    M_t = diag(dyn_mask),

where `dyn_mask[i] = True` for evolution rows and `False` for
algebraic constraint rows.  For algebraic rows the implicit residual is
the constraint `f_I(t, y)[i] = 0` itself; the per-stage Newton
iteration enforces it exactly.

Verified at order 2 (ARS232) and order 3 (ARS343) on linear and
nonlinear index-1 toy DAEs in `tests/unit/zoomy_core/test_imex_ark.py`.
Constraint residual stays at floating-point precision.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class IMEXTableau:
    name: str
    order: int
    A_E: np.ndarray
    b_E: np.ndarray
    c_E: np.ndarray
    A_I: np.ndarray
    b_I: np.ndarray
    c_I: np.ndarray
    s: int


@dataclass
class ExplicitButcherTableau:
    """Butcher tableau for an explicit Runge-Kutta scheme.

    Used by `explicit_rk_step` for predictor / explicit-only
    substeps (e.g. the Chorin split's hyperbolic predictor).
    Lives alongside `IMEXTableau` so the time-stepping zoo is
    one module — same construction style, just no implicit half.
    """
    name: str
    order: int
    A: np.ndarray        # (s, s) — strictly lower triangular
    b: np.ndarray        # (s,)
    c: np.ndarray        # (s,)
    s: int


def euler() -> ExplicitButcherTableau:
    """Forward Euler (RK1).  Single stage, used at reconstruction order 1."""
    return ExplicitButcherTableau(
        name="Euler",
        order=1,
        A=np.array([[0.0]]),
        b=np.array([1.0]),
        c=np.array([0.0]),
        s=1,
    )


def ssprk2() -> ExplicitButcherTableau:
    """Heun's method / SSP-RK2.  Two stages, order 2.

    Standard for explicit hyperbolic transport at order 2 — preserves
    monotonicity under CFL when paired with a TVD spatial limiter.
    """
    return ExplicitButcherTableau(
        name="SSP-RK2",
        order=2,
        A=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ]),
        b=np.array([0.5, 0.5]),
        c=np.array([0.0, 1.0]),
        s=2,
    )


def ssprk3() -> ExplicitButcherTableau:
    """Shu-Osher SSP-RK3.  Three stages, order 3.

    Strong-stability-preserving — same TVD property as SSP-RK2 but
    one order higher.  Slightly more expensive (3 RHS evaluations).
    """
    return ExplicitButcherTableau(
        name="SSP-RK3",
        order=3,
        A=np.array([
            [0.0,  0.0,  0.0],
            [1.0,  0.0,  0.0],
            [0.25, 0.25, 0.0],
        ]),
        b=np.array([1.0/6.0, 1.0/6.0, 2.0/3.0]),
        c=np.array([0.0, 1.0, 0.5]),
        s=3,
    )


def explicit_rk_step(
    t: float,
    y: np.ndarray,
    dt: float,
    table: ExplicitButcherTableau,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Advance `y` one explicit RK step under any Butcher tableau.

    Shape-agnostic: `y` and `f(t, y)` can be any array shape (1-D,
    2-D state-on-mesh, etc.) so long as they match.  For partial
    updates (e.g. a Chorin predictor that only touches certain state
    rows) the caller returns zeros for the untouched slots in `f`.

    No allocation per stage beyond the stage-state arrays themselves;
    safe to call inside a time loop.
    """
    s = table.s
    K: List[np.ndarray] = [None] * s
    for i in range(s):
        Y_i = y.copy()
        for j in range(i):
            if table.A[i, j] != 0.0:
                Y_i = Y_i + dt * table.A[i, j] * K[j]
        K[i] = f(t + table.c[i] * dt, Y_i)
    y_new = y.copy()
    for i in range(s):
        if table.b[i] != 0.0:
            y_new = y_new + dt * table.b[i] * K[i]
    return y_new


def ars232() -> IMEXTableau:
    """ARS(2,3,2) — 2nd-order, stiffly accurate.

    Reference: Ascher-Ruuth-Spiteri 1997 Sec. 2.6, Table 1.
    """
    g = 1.0 - 1.0 / math.sqrt(2.0)
    d = -2.0 * math.sqrt(2.0) / 3.0
    A_E = np.array([
        [0.0, 0.0, 0.0],
        [g,   0.0, 0.0],
        [d,   1-d, 0.0],
    ])
    b_E = np.array([0.0, 1-g, g])
    c_E = np.array([0.0, g, 1.0])
    A_I = np.array([
        [0.0, 0.0, 0.0],
        [0.0, g,   0.0],
        [0.0, 1-g, g],
    ])
    b_I = np.array([0.0, 1-g, g])
    c_I = np.array([0.0, g, 1.0])
    return IMEXTableau("ARS232", 2, A_E, b_E, c_E, A_I, b_I, c_I, 3)


def ars343() -> IMEXTableau:
    """ARS(3,4,3) — 3rd-order, stiffly accurate.

    Reference: Ascher-Ruuth-Spiteri 1997 Sec. 2.7, Table 2.
    """
    a_E = np.array([
        [0.0,           0.0,           0.0,           0.0],
        [0.4358665215,  0.0,           0.0,           0.0],
        [0.3212788860,  0.3966543747,  0.0,           0.0],
        [-0.105858296,  0.5529291479,  0.5529291479,  0.0],
    ])
    b_E = np.array([0.0, 1.208496649, -0.644363171, 0.4358665215])
    c_E = np.array([0.0, 0.4358665215, 0.7179332608, 1.0])
    a_I = np.array([
        [0.0, 0.0,           0.0,           0.0],
        [0.0, 0.4358665215,  0.0,           0.0],
        [0.0, 0.2820667392,  0.4358665215,  0.0],
        [0.0, 1.208496649,  -0.644363171,   0.4358665215],
    ])
    b_I = np.array([0.0, 1.208496649, -0.644363171, 0.4358665215])
    c_I = np.array([0.0, 0.4358665215, 0.7179332608, 1.0])
    return IMEXTableau("ARS343", 3, a_E, b_E, c_E, a_I, b_I, c_I, 4)


def imex_ark_step(
    t: float,
    y: np.ndarray,
    dt: float,
    tab: IMEXTableau,
    f_E: Callable[[float, np.ndarray], np.ndarray],
    f_I: Callable[[float, np.ndarray], np.ndarray],
    J_I: Callable[[float, np.ndarray], np.ndarray],
    dyn_mask: np.ndarray,
    *,
    newton_tol: float = 1e-10,
    newton_maxit: int = 40,
) -> np.ndarray:
    """One IMEX-ARK step of `M_t y' = f_E + f_I` with `M_t = diag(dyn_mask)`.

    Stage residual:
        R[dyn] = (Y - rhs_explicit)[dyn] - dt·γ_ii·f_I(Y)[dyn]
        R[alg] = f_I(Y)[alg]
    """
    s = tab.s
    K_explicit: List[np.ndarray] = [None] * s
    K_implicit: List[np.ndarray] = [None] * s
    Y_stage: List[np.ndarray] = [None] * s

    Y_stage[0] = y.copy()
    K_explicit[0] = f_E(t + tab.c_E[0] * dt, Y_stage[0])
    K_implicit[0] = f_I(t + tab.c_I[0] * dt, Y_stage[0])

    for i in range(1, s):
        rhs_explicit = y.copy()
        for j in range(i):
            rhs_explicit += dt * tab.A_E[i, j] * K_explicit[j]
            rhs_explicit += dt * tab.A_I[i, j] * K_implicit[j]

        gii = tab.A_I[i, i]
        t_stage = t + tab.c_I[i] * dt

        def residual(Y):
            R = np.zeros_like(Y)
            fI = f_I(t_stage, Y)
            R[dyn_mask] = (Y - rhs_explicit)[dyn_mask] - dt * gii * fI[dyn_mask]
            R[~dyn_mask] = fI[~dyn_mask]
            return R

        def jacobian(Y):
            JI = J_I(t_stage, Y)
            n = len(Y)
            J = np.zeros((n, n))
            for ii in range(n):
                if dyn_mask[ii]:
                    J[ii, :] = -dt * gii * JI[ii, :]
                    J[ii, ii] += 1.0
                else:
                    J[ii, :] = JI[ii, :]
            return J

        Y = rhs_explicit.copy()
        for _ in range(newton_maxit):
            R = residual(Y)
            if np.linalg.norm(R) < newton_tol:
                break
            J = jacobian(Y)
            dY = np.linalg.solve(J, -R)
            Y = Y + dY
        else:
            raise RuntimeError(
                f"IMEX-ARK Newton failed at stage {i}, t={t}, "
                f"|R|={np.linalg.norm(R):.3e}"
            )

        Y_stage[i] = Y
        K_explicit[i] = f_E(t_stage, Y)
        K_implicit[i] = f_I(t_stage, Y)

    y_new = y.copy()
    for i in range(s):
        y_new += dt * tab.b_E[i] * K_explicit[i]
        y_new += dt * tab.b_I[i] * K_implicit[i]

    # Project algebraic rows back onto the constraint manifold at t+dt.
    for _ in range(10):
        fI = f_I(t + dt, y_new)
        if np.linalg.norm(fI[~dyn_mask]) < newton_tol:
            break
        JI = J_I(t + dt, y_new)
        idx_alg = np.where(~dyn_mask)[0]
        sub = JI[np.ix_(idx_alg, idx_alg)]
        delta = np.linalg.solve(sub, -fI[idx_alg])
        y_new[idx_alg] += delta

    return y_new


def integrate(
    y0: np.ndarray,
    t0: float,
    t_end: float,
    dt: float,
    tab: IMEXTableau,
    f_E: Callable[[float, np.ndarray], np.ndarray],
    f_I: Callable[[float, np.ndarray], np.ndarray],
    J_I: Callable[[float, np.ndarray], np.ndarray],
    dyn_mask: np.ndarray,
    *,
    newton_tol: float = 1e-10,
    newton_maxit: int = 40,
) -> List[Tuple[float, np.ndarray]]:
    """Fixed-step IMEX-ARK integration from `t0` to `t_end`.

    Returns a list of `(t, y)` snapshots including the initial state.
    """
    t = t0
    y = y0.copy()
    history = [(t, y.copy())]
    n_steps = int(round((t_end - t0) / dt))
    for _ in range(n_steps):
        y = imex_ark_step(t, y, dt, tab, f_E, f_I, J_I, dyn_mask,
                          newton_tol=newton_tol, newton_maxit=newton_maxit)
        t += dt
        history.append((t, y.copy()))
    return history
