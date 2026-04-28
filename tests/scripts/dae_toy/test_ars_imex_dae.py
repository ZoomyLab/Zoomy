"""Toy DAE test for IMEX-ARK (Ascher-Ruuth-Spiteri) integrators.

Step 1 of the DAE-solver build (per `notebooks/DAE_REFERENCES.md`):
verify a self-contained ARS-style IMEX-Runge-Kutta integrator on a
small index-1 DAE that has an analytic solution.

Tableaux taken from:

  Ascher, Ruuth, Spiteri (1997), "Implicit-Explicit Runge-Kutta methods
  for time-dependent partial differential equations".  Sec. 2.5/2.7.
  Cited via DOI: 10.1016/S0168-9274(97)00056-1
  Verified against Pareschi & Russo 2005, Tables 4-6
  (DOI: 10.1007/s10915-004-4636-4) and Gardner et al. 2018, Appendix A
  (DOI: 10.5194/gmd-11-1497-2018).

Test problem — linear index-1 DAE
---------------------------------

    y1' = -y1 + y2 + 1
    y2' = -y1 + sin(t)
    0   = y1 + y3 - 2

Three unknowns ``y = (y1, y2, y3)``.  The third row is an algebraic
constraint that determines ``y3 = 2 - y1`` pointwise.  Differentiating
gives ``y3' = -y1' = y1 - y2 - 1`` — index-1.

Analytic solution (from solving the reduced 2-ODE system):

    y1(t) = 0.5*sin(t) - 0.5*cos(t) + (a + 0.5)*exp(-t) - 0.5*sin(t)
          ... [solver constructs analytically; we just check residual at end]

We cheat by integrating with scipy.solve_ivp on the reduced ODE
``(y1, y2)`` system to high precision and treating that as the
reference; ARS232 / ARS343 should match it to their respective orders.

Goal: verify the per-stage Newton-GMRES correctly enforces the
algebraic constraint AND the explicit + implicit stages compose
correctly, observing the expected convergence rate (2 for ARS232,
3 for ARS343).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# IMEX-ARK tableau dataclass
# ---------------------------------------------------------------------------

@dataclass
class IMEXTableau:
    name: str
    order: int
    A_E: np.ndarray         # explicit Butcher A (lower triangular, A[i,j] for j<i)
    b_E: np.ndarray
    c_E: np.ndarray
    A_I: np.ndarray         # implicit Butcher A (lower triangular incl. diagonal)
    b_I: np.ndarray
    c_I: np.ndarray
    # Number of stages.
    s: int


def ars232() -> IMEXTableau:
    """Ascher-Ruuth-Spiteri (2,3,2): 2 stages explicit, 3 stages implicit, order 2.

    Reference: Ascher-Ruuth-Spiteri 1997 Sec. 2.6, Table 1.
    """
    γ = 1.0 - 1.0 / math.sqrt(2.0)
    δ = -2.0 * math.sqrt(2.0) / 3.0
    A_E = np.array([
        [0,    0,   0],
        [γ,    0,   0],
        [δ,  1-δ,   0],
    ])
    b_E = np.array([δ, 1-δ, 0.0])
    # Note: Ascher's bE for ARS232 ends in 0 (only first two stages contribute);
    # equivalent to b_E = [0, 1-γ, γ] under stiff-accuracy renaming.  We use the
    # stiffly-accurate form: b_E = [0, 1-γ, γ].
    b_E = np.array([0.0, 1-γ, γ])
    c_E = np.array([0.0, γ, 1.0])
    A_I = np.array([
        [0,   0,  0],
        [0,   γ,  0],
        [0, 1-γ,  γ],
    ])
    b_I = np.array([0.0, 1-γ, γ])
    c_I = np.array([0.0, γ, 1.0])
    return IMEXTableau("ARS232", 2, A_E, b_E, c_E, A_I, b_I, c_I, 3)


def ars343() -> IMEXTableau:
    """Ascher-Ruuth-Spiteri (3,4,3): 3 implicit + 3 explicit stages (4 total), order 3.

    Reference: Ascher-Ruuth-Spiteri 1997 Sec. 2.7, Table 2.
    """
    # Coefficients (Sec. 2.7).
    a_E = np.array([
        [0,                0,                0,         0],
        [0.4358665215,     0,                0,         0],
        [0.3212788860,     0.3966543747,     0,         0],
        [-0.105858296,     0.5529291479,     0.5529291479, 0],
    ])
    b_E = np.array([0.0, 1.208496649, -0.644363171, 0.4358665215])
    c_E = np.array([0.0, 0.4358665215, 0.7179332608, 1.0])
    a_I = np.array([
        [0,                0,                0,                0],
        [0,                0.4358665215,     0,                0],
        [0,                0.2820667392,     0.4358665215,     0],
        [0,                1.208496649,     -0.644363171,      0.4358665215],
    ])
    b_I = np.array([0.0, 1.208496649, -0.644363171, 0.4358665215])
    c_I = np.array([0.0, 0.4358665215, 0.7179332608, 1.0])
    return IMEXTableau("ARS343", 3, a_E, b_E, c_E, a_I, b_I, c_I, 4)


# ---------------------------------------------------------------------------
# Generic IMEX-ARK integrator for index-1 DAE
# ---------------------------------------------------------------------------

def imex_ark_step(t, y, dt, tab: IMEXTableau,
                  f_E, f_I, J_I, dyn_mask):
    """One IMEX-ARK step for the system
        M_t · y' = f_E(t, y) + f_I(t, y),    M_t = diag(dyn_mask),
    where ``dyn_mask[i] = 1`` if row i is evolution and 0 if algebraic.

    For algebraic rows the implicit residual is just ``f_I(t, y)[i] = 0``
    (we drop the time-derivative term for those rows).  Newton iteration
    with analytical Jacobian ``J_I`` solves the implicit stage.
    """
    s = tab.s
    K_explicit = [None] * s   # f_E evaluated at each stage
    K_implicit = [None] * s   # f_I evaluated at each stage
    Y_stage   = [None] * s    # stage values

    # Stage 1: usually a_I[0,0] = 0 → trivial, Y_1 = y.
    Y_stage[0] = y.copy()
    K_explicit[0] = f_E(t + tab.c_E[0] * dt, Y_stage[0])
    K_implicit[0] = f_I(t + tab.c_I[0] * dt, Y_stage[0])

    for i in range(1, s):
        # Build "explicit accumulator": everything except f_I at stage i.
        rhs_explicit = y.copy()
        for j in range(i):
            rhs_explicit += dt * tab.A_E[i, j] * K_explicit[j]
        for j in range(i):
            rhs_explicit += dt * tab.A_I[i, j] * K_implicit[j]

        γii = tab.A_I[i, i]
        t_stage = t + tab.c_I[i] * dt

        # For dynamic rows: M*Y_i = rhs_explicit + dt*γii*f_I(Y_i)
        # For algebraic rows: f_I(Y_i) = 0 directly.
        # Combine into one nonlinear residual R(Y_i) = 0:
        #   R[dyn] = M[dyn] · (Y_i - rhs_E)[dyn]      − dt*γii*f_I(Y_i)[dyn]
        #   R[alg] =                                                 f_I(Y_i)[alg]
        # We solve R = 0 by Newton.

        def residual(Y):
            R = np.zeros_like(Y)
            fI = f_I(t_stage, Y)
            R[dyn_mask] = (Y - rhs_explicit)[dyn_mask] - dt * γii * fI[dyn_mask]
            R[~dyn_mask] = fI[~dyn_mask]
            return R

        def jacobian(Y):
            JI = J_I(t_stage, Y)
            n = len(Y)
            J = np.zeros((n, n))
            for ii in range(n):
                if dyn_mask[ii]:
                    J[ii, :] = -dt * γii * JI[ii, :]
                    J[ii, ii] += 1.0
                else:
                    J[ii, :] = JI[ii, :]
            return J

        # Newton iteration with damped step.
        Y = rhs_explicit.copy()
        for newton_it in range(40):
            R = residual(Y)
            if np.linalg.norm(R) < 1e-12:
                break
            J = jacobian(Y)
            δY = np.linalg.solve(J, -R)
            # Plain Newton step (toy size).
            Y = Y + δY
        else:
            raise RuntimeError(
                f"Newton failed to converge at stage {i}, t={t}, "
                f"residual={np.linalg.norm(R):.3e}"
            )

        Y_stage[i] = Y
        K_explicit[i] = f_E(t_stage, Y)
        K_implicit[i] = f_I(t_stage, Y)

    # Final update.  In stiffly-accurate schemes b_I = A_I[-1, :] and
    # b_E = A_E[-1, :], so the update equals Y_stage[-1].  We compute
    # explicitly to match general (non-stiffly-accurate) tableaux too.
    y_new = y.copy()
    for i in range(s):
        y_new += dt * tab.b_E[i] * K_explicit[i]
        y_new += dt * tab.b_I[i] * K_implicit[i]
    # Algebraic rows: enforce constraint exactly at t+dt.
    # The implicit-stage solve at the last stage already enforced f_I=0
    # there.  For stiffly accurate schemes y_new == Y_stage[-1] for
    # algebraic rows.  For non-SA, we project explicitly: easiest to just
    # set algebraic rows to whatever satisfies f_I(t+dt, y_new) = 0.
    # We use a Newton step on the algebraic rows alone.
    for _ in range(10):
        fI = f_I(t + dt, y_new)
        if np.linalg.norm(fI[~dyn_mask]) < 1e-12:
            break
        JI = J_I(t + dt, y_new)
        # Solve f_I[alg, alg] · δ_alg = -f_I[alg]
        idx_alg = np.where(~dyn_mask)[0]
        sub = JI[np.ix_(idx_alg, idx_alg)]
        rhs = -fI[idx_alg]
        δ = np.linalg.solve(sub, rhs)
        y_new[idx_alg] += δ

    return y_new


def integrate(y0, t0, t_end, dt, tab, f_E, f_I, J_I, dyn_mask):
    t = t0
    y = y0.copy()
    history = [(t, y.copy())]
    n_steps = int(round((t_end - t0) / dt))
    for _ in range(n_steps):
        y = imex_ark_step(t, y, dt, tab, f_E, f_I, J_I, dyn_mask)
        t += dt
        history.append((t, y.copy()))
    return history


# ---------------------------------------------------------------------------
# Test problem: linear index-1 DAE
# ---------------------------------------------------------------------------
#
#   y1' = -y1 + y2 + 1
#   y2' = -y1 + sin(t)
#   0   = y1 + y3 - 2
#
# Initial conditions consistent with the constraint: y3(0) = 2 - y1(0).
# We split the RHS into f_E (the "non-stiff" part — sin(t) source) and
# f_I (the "stiff" part — coupling matrix + constraint).  For this
# linear problem the split is somewhat artificial but exercises the
# integrator's machinery.

def make_problem():
    # f_E = explicit-treated forcing (the sin/cos/constant inputs).
    def f_E(t, y):
        return np.array([1.0, math.sin(t), 0.0])

    # f_I = implicit-treated linear coupling + algebraic constraint.
    def f_I(t, y):
        return np.array([
            -y[0] + y[1],          # row 1: dy1/dt = … + (-y1 + y2)
            -y[0],                 # row 2: dy2/dt = … + (-y1)
            y[0] + y[2] - 2.0,     # row 3: algebraic constraint
        ])

    # Jacobian of f_I w.r.t. y.
    def J_I(t, y):
        return np.array([
            [-1, 1,  0],
            [-1, 0,  0],
            [ 1, 0,  1],
        ], dtype=float)

    dyn_mask = np.array([True, True, False])

    # Reference: integrate the reduced 2-ODE system at high precision.
    def reduced_rhs(t, q):
        y1, y2 = q
        return np.array([-y1 + y2 + 1.0, -y1 + math.sin(t)])

    return f_E, f_I, J_I, dyn_mask, reduced_rhs


def reference(y0_full, t_eval):
    """Reference solution via scipy on the reduced 2-ODE."""
    f_E, f_I, J_I, dyn_mask, reduced_rhs = make_problem()
    sol = solve_ivp(reduced_rhs, (0.0, t_eval[-1]), y0_full[:2],
                    t_eval=t_eval, rtol=1e-12, atol=1e-14, method="Radau")
    y1 = sol.y[0]
    y2 = sol.y[1]
    y3 = 2.0 - y1
    return np.stack([y1, y2, y3], axis=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    f_E, f_I, J_I, dyn_mask, _ = make_problem()
    y0 = np.array([1.0, 0.5, 1.0])  # consistent: y1+y3-2 = 0 ✓

    print("=" * 70)
    print("Toy DAE: linear index-1 with algebraic constraint")
    print("  y1' = -y1 + y2 + 1")
    print("  y2' = -y1 + sin(t)")
    print("  0   = y1 + y3 - 2")
    print(f"  y0  = {y0}")
    print("=" * 70)

    t_end = 2.0
    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    for tab_factory in (ars232, ars343):
        tab = tab_factory()
        print(f"\n{tab.name} (theoretical order = {tab.order})")
        print(f"   {'dt':>10s}  {'|y - y_ref|_∞':>16s}  {'ratio':>10s}  {'rate':>8s}")
        prev_err = None
        for dt in dts:
            history = integrate(y0, 0.0, t_end, dt, tab, f_E, f_I, J_I, dyn_mask)
            t_grid = np.array([h[0] for h in history])
            y_grid = np.array([h[1] for h in history]).T   # (3, N)
            y_ref = reference(y0, t_grid)
            err = float(np.max(np.abs(y_grid - y_ref)))
            ratio = prev_err / err if prev_err else float("nan")
            rate = math.log2(ratio) if not math.isnan(ratio) else float("nan")
            print(f"   {dt:10.6f}  {err:16.4e}  {ratio:10.3f}  {rate:8.2f}")
            prev_err = err

    # Constraint residual check (last fine run).
    print("\nConstraint residual check (ARS343, dt = 0.00625):")
    tab = ars343()
    history = integrate(y0, 0.0, t_end, 0.00625, tab, f_E, f_I, J_I, dyn_mask)
    max_constr = max(abs(y[0] + y[2] - 2.0) for _, y in history)
    print(f"   max |y1 + y3 - 2| over the trajectory: {max_constr:.3e}")

    # ---- nonlinear toy: forces Newton to do real work ----
    print()
    print("=" * 70)
    print("Nonlinear toy DAE (forces real Newton work each stage)")
    print("  y1' = -y1 + y2² + sin(t)")
    print("  0   = y1·y2 - 1     ⇒  y2 = 1/y1")
    print("=" * 70)

    def f_E_nl(t, y):
        return np.array([math.sin(t), 0.0])

    def f_I_nl(t, y):
        return np.array([
            -y[0] + y[1] ** 2,
            y[0] * y[1] - 1.0,
        ])

    def J_I_nl(t, y):
        return np.array([
            [-1.0,    2 * y[1]],
            [y[1],    y[0]],
        ])

    dyn_mask_nl = np.array([True, False])
    y0_nl = np.array([1.0, 1.0])  # consistent: 1*1 - 1 = 0 ✓

    # Reference via scipy on the reduced 1-ODE  y1' = -y1 + 1/y1² + sin(t).
    def reduced_nl(t, q):
        return np.array([-q[0] + 1.0 / q[0] ** 2 + math.sin(t)])

    for tab_factory in (ars232, ars343):
        tab = tab_factory()
        print(f"\n{tab.name} on nonlinear toy")
        print(f"   {'dt':>10s}  {'|y - y_ref|_∞':>16s}  {'ratio':>10s}  {'rate':>8s}")
        prev_err = None
        for dt in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
            history = integrate(y0_nl, 0.0, 2.0, dt, tab,
                                 f_E_nl, f_I_nl, J_I_nl, dyn_mask_nl)
            t_grid = np.array([h[0] for h in history])
            t_grid = np.clip(t_grid, 0.0, 2.0)
            y_grid = np.array([h[1] for h in history]).T
            sol = solve_ivp(reduced_nl, (0.0, 2.0), [y0_nl[0]],
                            t_eval=t_grid, rtol=1e-12, atol=1e-14, method="Radau")
            y1_ref = sol.y[0]
            y2_ref = 1.0 / y1_ref
            y_ref = np.stack([y1_ref, y2_ref], axis=0)
            err = float(np.max(np.abs(y_grid - y_ref)))
            ratio = prev_err / err if prev_err else float("nan")
            rate = math.log2(ratio) if not math.isnan(ratio) else float("nan")
            print(f"   {dt:10.6f}  {err:16.4e}  {ratio:10.3f}  {rate:8.2f}")
            prev_err = err

    return 0 if max_constr < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
