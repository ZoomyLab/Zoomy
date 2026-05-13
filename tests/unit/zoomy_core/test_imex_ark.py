"""Order-of-accuracy verification for the IMEX-ARK DAE integrator.

Exercises ``imex_ark.ars232`` and ``imex_ark.ars343`` on two index-1 toy
DAEs:

* Linear:    y1' = -y1 + y2 + 1,   y2' = -y1 + sin t,   y1 + y3 - 2 = 0
* Nonlinear: y1' = -y1 + y2^2 + sin t,                  y1·y2 - 1   = 0

Reference solutions come from ``scipy.integrate.solve_ivp`` (Radau,
tight tolerances) on the reduced ODE that algebraic elimination yields.
Tests check both the observed convergence rate and the constraint
residual at the final time.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from zoomy_core.fvm.imex_ark import ars232, ars343, integrate


# ---------------------------------------------------------------------------
# Linear index-1 DAE
# ---------------------------------------------------------------------------

def _linear_problem():
    def f_E(t, y):
        return np.array([1.0, math.sin(t), 0.0])

    def f_I(t, y):
        return np.array([
            -y[0] + y[1],
            -y[0],
            y[0] + y[2] - 2.0,
        ])

    def J_I(t, y):
        return np.array([
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [ 1.0, 0.0, 1.0],
        ])

    dyn_mask = np.array([True, True, False])
    y0 = np.array([1.0, 0.5, 1.0])  # consistent: y1+y3-2 = 0

    def reduced_rhs(t, q):
        y1, y2 = q
        return np.array([-y1 + y2 + 1.0, -y1 + math.sin(t)])

    return f_E, f_I, J_I, dyn_mask, y0, reduced_rhs


def _linear_reference(y0, t_eval):
    _, _, _, _, _, reduced_rhs = _linear_problem()
    sol = solve_ivp(reduced_rhs, (0.0, t_eval[-1]), y0[:2],
                    t_eval=t_eval, rtol=1e-12, atol=1e-14, method="Radau")
    y1 = sol.y[0]; y2 = sol.y[1]; y3 = 2.0 - y1
    return np.stack([y1, y2, y3], axis=0)


# ---------------------------------------------------------------------------
# Nonlinear index-1 DAE
# ---------------------------------------------------------------------------

def _nonlinear_problem():
    def f_E(t, y):
        return np.array([math.sin(t), 0.0])

    def f_I(t, y):
        return np.array([
            -y[0] + y[1] ** 2,
            y[0] * y[1] - 1.0,
        ])

    def J_I(t, y):
        return np.array([
            [-1.0,     2 * y[1]],
            [ y[1],    y[0]    ],
        ])

    dyn_mask = np.array([True, False])
    y0 = np.array([1.0, 1.0])  # consistent: 1*1 - 1 = 0

    def reduced_rhs(t, q):
        return np.array([-q[0] + 1.0 / q[0] ** 2 + math.sin(t)])

    return f_E, f_I, J_I, dyn_mask, y0, reduced_rhs


def _nonlinear_reference(y0, t_eval):
    _, _, _, _, _, reduced_rhs = _nonlinear_problem()
    sol = solve_ivp(reduced_rhs, (0.0, t_eval[-1]), [y0[0]],
                    t_eval=t_eval, rtol=1e-12, atol=1e-14, method="Radau")
    y1 = sol.y[0]; y2 = 1.0 / y1
    return np.stack([y1, y2], axis=0)


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------

def _run_order_test(tab_factory, problem_factory, reference_fn,
                    expected_order, t_end=2.0,
                    dts=(0.1, 0.05, 0.025, 0.0125)):
    """Refine ``dt`` and verify the asymptotic order matches ``expected_order``."""
    f_E, f_I, J_I, dyn_mask, y0, _ = problem_factory()
    tab = tab_factory()

    errs = []
    for dt in dts:
        history = integrate(y0, 0.0, t_end, dt, tab, f_E, f_I, J_I, dyn_mask)
        t_grid = np.array([h[0] for h in history])
        y_grid = np.array([h[1] for h in history]).T
        y_ref = reference_fn(y0, t_grid)
        errs.append(float(np.max(np.abs(y_grid - y_ref))))

    # Use the last 3 levels (asymptotic regime).
    rates = [math.log2(errs[i-1] / errs[i]) for i in range(1, len(errs))]
    final_rate = rates[-1]
    assert final_rate >= expected_order - 0.15, (
        f"{tab.name} observed final order {final_rate:.2f} < "
        f"{expected_order - 0.15}; full rates = {rates}, errs = {errs}"
    )


def test_ars232_linear_order_2():
    _run_order_test(ars232, _linear_problem, _linear_reference,
                    expected_order=2)


def test_ars343_linear_order_3():
    _run_order_test(ars343, _linear_problem, _linear_reference,
                    expected_order=3)


def test_ars232_nonlinear_order_2():
    _run_order_test(ars232, _nonlinear_problem, _nonlinear_reference,
                    expected_order=2)


def test_ars343_nonlinear_order_3():
    _run_order_test(ars343, _nonlinear_problem, _nonlinear_reference,
                    expected_order=3)


def test_linear_constraint_residual_machine_precision():
    """Algebraic row y1+y3-2=0 enforced to floating-point precision."""
    f_E, f_I, J_I, dyn_mask, y0, _ = _linear_problem()
    tab = ars343()
    history = integrate(y0, 0.0, 2.0, 0.00625, tab, f_E, f_I, J_I, dyn_mask)
    max_constraint = max(abs(y[0] + y[2] - 2.0) for _, y in history)
    assert max_constraint < 1e-10, (
        f"Constraint y1+y3-2=0 max residual {max_constraint:.3e}"
    )


def test_nonlinear_constraint_residual_machine_precision():
    """Algebraic row y1·y2-1=0 enforced to floating-point precision."""
    f_E, f_I, J_I, dyn_mask, y0, _ = _nonlinear_problem()
    tab = ars343()
    history = integrate(y0, 0.0, 2.0, 0.00625, tab, f_E, f_I, J_I, dyn_mask)
    max_constraint = max(abs(y[0] * y[1] - 1.0) for _, y in history)
    assert max_constraint < 1e-10, (
        f"Constraint y1·y2-1=0 max residual {max_constraint:.3e}"
    )
