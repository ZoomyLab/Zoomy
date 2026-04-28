"""VAM(1, 2, 2) direct DAE simulation — no Chorin pressure splitting.

Wires the Escalante-aligned VAM PDESystem (`build_vam_pdesystem`) onto
the IMEX-ARK time integrator (`tests/scripts/dae_toy/test_ars_imex_dae.py`)
via:
  1. partition the rows into evolution (6) + algebraic (3) using the
     bridge in `tests/scripts/dae_toy/test_dae_partition_bridge.py`;
  2. lambdify each row as a function of cell-local fields and their
     spatial derivatives (centred differences for ∂_x);
  3. assemble the global residual + Jacobian on a 1D periodic grid;
  4. step in time with **fully-implicit backward Euler with Newton**
     (the simplest IMEX special case where everything is implicit;
     ARS343 is a straight stage-loop generalisation).

This is the proof-of-concept VAM-DAE direct simulation.  Treats stress
+ source + constraints all implicit; the explicit-flux split (Pareschi
ARS343) is the next step once this works.

Tableau-pinned references in `notebooks/DAE_REFERENCES.md`.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp
from scipy.optimize import root

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from vam_pdesystem import build_vam_pdesystem    # noqa: E402

sys.path.insert(0, "tests/scripts/dae_toy")
from test_dae_partition_bridge import dae_partition  # noqa: E402


# ---------------------------------------------------------------------------
# Build symbolic residual functions per equation
# ---------------------------------------------------------------------------

def _expression_atoms(expr, fields, t, x):
    """Walk expr; return the set of `(field, n_x)` atoms that appear,
    where n_x = 0 means f, n_x = 1 means ∂_x f, etc.  Plus ∂_t f atoms.
    """
    atoms = set()
    for f in fields:
        if expr.has(f):
            atoms.add((f, 0, 0))
    for d in expr.atoms(sp.Derivative):
        target = d.args[0]
        if target not in fields:
            continue
        n_t = 0
        n_x = 0
        for v, n in d.variable_count:
            if v == t:
                n_t += int(n)
            elif v == x:
                n_x += int(n)
        atoms.add((target, n_x, n_t))
    return atoms


def lambdify_row(expr, fields, t, x, parameters):
    """Lambdify a row of the PDESystem as a function of:
        q          : (n_fields,) cell values
        q_x        : (n_fields,) ∂_x f at this cell
        q_t        : (n_fields,) ∂_t f at this cell
        param_vals : dict
    Returns a callable ``f(q, q_x, q_t, params)`` returning the residual
    value at that cell.
    """
    # Build per-cell symbol map: f -> q_i, ∂_x f -> qx_i, ∂_t f -> qt_i.
    n = len(fields)
    q_syms = sp.symbols(f"q0:{n}", real=True)
    qx_syms = sp.symbols(f"qx0:{n}", real=True)
    qt_syms = sp.symbols(f"qt0:{n}", real=True)
    repl = {}
    for i, f in enumerate(fields):
        repl[f] = q_syms[i]
        repl[sp.Derivative(f, x)] = qx_syms[i]
        repl[sp.Derivative(f, t)] = qt_syms[i]
    # Higher derivatives of fields → assume zero (centred difference is 1st order).
    for d in expr.atoms(sp.Derivative):
        if d not in repl:
            order = sum(int(n) for _, n in d.variable_count)
            if order > 1:
                repl[d] = 0
    new = expr.xreplace(repl)
    return sp.lambdify(
        list(q_syms) + list(qx_syms) + list(qt_syms) + list(parameters.keys()),
        new, modules="numpy",
    )


# ---------------------------------------------------------------------------
# Global residual on a periodic 1D grid
# ---------------------------------------------------------------------------

class VAMDirectDAEStepper:
    """Fully-implicit backward-Euler stepper for VAM(M, N_w, N_p) on a 1D
    periodic grid."""

    def __init__(self, M=1, N_w=2, N_p=2, *, Nx=64, L=10.0, g=1.0):
        self.M = M
        self.N_w = N_w
        self.N_p = N_p
        self.Nx = Nx
        self.L = L
        self.dx = L / Nx
        self.g = g

        # Build the symbolic system.
        self.sys = build_vam_pdesystem(M=M, N_w=N_w, N_p=N_p, flat_bottom=True)
        self.fields = self.sys.fields
        self.n_fields = len(self.fields)
        self.eq_names = self.sys.equation_names
        self.dyn_rows, self.alg_rows, _, _ = dae_partition(self.sys)

        t_sym = self.sys.time
        x_sym = self.sys.space[0]
        params = {sp.Symbol("g", positive=True): self.g}

        # Lambdify each equation.
        self.row_funcs = []
        for i, eq in enumerate(self.sys.equations):
            self.row_funcs.append(
                lambdify_row(eq, self.fields, t_sym, x_sym,
                             {sp.Symbol("g", positive=True): self.g})
            )

        print(f"[VAMDirectDAEStepper] VAM(M={M}, N_w={N_w}, N_p={N_p}) "
              f"on Nx={Nx} cells, L={L}")
        print(f"  fields: {[f.func.__name__ for f in self.fields]}")
        print(f"  evolution rows: "
              f"{[self.eq_names[i] for i in self.dyn_rows]}")
        print(f"  algebraic rows: "
              f"{[self.eq_names[i] for i in self.alg_rows]}")

    def _q_x_centred(self, Q):
        """Centred-difference ∂_x for periodic state Q of shape (Nx, n_fields)."""
        Qp = np.roll(Q, -1, axis=0)
        Qm = np.roll(Q, +1, axis=0)
        return (Qp - Qm) / (2.0 * self.dx)

    def residual(self, Q_flat, Q_prev, dt):
        """Backward-Euler residual:
            R[i, j] = (eq_j evaluated at cell i with q_t = (Q-Q_prev)/dt)
        ``Q_flat`` is the flattened next-step state of shape (Nx*n_fields,).
        """
        Q = Q_flat.reshape(self.Nx, self.n_fields)
        Q_x = self._q_x_centred(Q)
        Q_t = (Q - Q_prev) / dt          # backward-Euler approx of ∂_t

        R = np.zeros((self.Nx, len(self.row_funcs)), dtype=float)
        # For algebraic rows, the equation has no ∂_t; we still pass q_t
        # but the lambdified function is independent of those args, so OK.
        for j, fn in enumerate(self.row_funcs):
            for i in range(self.Nx):
                # Pass the full (q, q_x, q_t) plus params (g).
                args = (*Q[i], *Q_x[i], *Q_t[i], self.g)
                R[i, j] = fn(*args)
        return R.reshape(-1)

    def step(self, Q, dt, *, tol=1e-8, maxit=20):
        """Backward-Euler step via scipy.optimize.root (Newton-Krylov)."""
        Q_prev = Q.copy()
        Q_init = Q.flatten()
        sol = root(
            self.residual, Q_init, args=(Q_prev, dt),
            method="hybr", tol=tol,
            options={"maxfev": maxit * len(Q_init), "xtol": tol},
        )
        if not sol.success:
            print(f"  [warn] Newton non-convergent: {sol.message}; "
                  f"|R|_inf = {np.max(np.abs(sol.fun)):.3e}")
        Q_new = sol.x.reshape(self.Nx, self.n_fields)
        return Q_new, sol


# ---------------------------------------------------------------------------
# Initial conditions consistent with the algebraic constraints
# ---------------------------------------------------------------------------

def initial_consistent(stepper, *, h_amp=0.1, u0_amp=0.0):
    """Build a smooth initial state that satisfies the algebraic constraints
    pointwise: w_i = 0, p_i = 0, h = H + bump, u_0 = U_0 + small."""
    Nx = stepper.Nx
    L = stepper.L
    n = stepper.n_fields
    x = np.arange(Nx) * stepper.dx
    Q = np.zeros((Nx, n))
    # Field order: h, u_0, u_1, w_0, w_1, w_2, p_0, p_1, p_2.
    Q[:, 0] = 1.0 + h_amp * np.cos(2 * np.pi * x / L)        # h
    Q[:, 1] = u0_amp                                          # u_0
    # u_1 = w_i = p_i = 0 → constraints satisfied:
    #   kbc_bot: w_0 + w_1 + w_2 = 0  ✓ (all zero)
    #   kbc_top_alg: w(η) - u(η) ∂_x η + ∂_x(h u_0) = 0
    #     → 0 - 0 + ∂_x(h*0) = 0  ✓
    #   surface_bc: p_0 - p_1 + p_2 = 0  ✓
    return Q


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--Nx", type=int, default=32)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--h-amp", type=float, default=0.05)
    parser.add_argument("--u0-amp", type=float, default=0.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("VAM(1, 2, 2) direct DAE simulation (backward-Euler)")
    print("=" * 70)
    stepper = VAMDirectDAEStepper(M=1, N_w=2, N_p=2,
                                  Nx=args.Nx, L=args.L, g=args.g)
    Q = initial_consistent(stepper, h_amp=args.h_amp, u0_amp=args.u0_amp)
    print(f"\nInitial state: h ∈ [{Q[:, 0].min():.3f}, {Q[:, 0].max():.3f}]")
    print(f"  u_0 = {args.u0_amp},  u_1 = w_i = p_i = 0")

    # Verify initial residual.
    R0 = stepper.residual(Q.flatten(), Q.copy(), args.dt)
    print(f"  initial backward-Euler residual |R|_inf = {np.max(np.abs(R0)):.3e}")

    n_steps = int(round(args.T / args.dt))
    snapshots = [(0.0, Q.copy())]
    print(f"\nIntegrating to T = {args.T} ({n_steps} steps of dt = {args.dt})")
    for step in range(n_steps):
        Q_new, sol = stepper.step(Q, args.dt)
        Q = Q_new
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            # sol.fun is the actual Newton residual at convergence.
            print(f"  step {step+1:4d}/{n_steps:4d}  t = {(step+1)*args.dt:.3f}"
                  f"   h ∈ [{Q[:,0].min():.4f},{Q[:,0].max():.4f}]"
                  f"   max|p_1| = {np.max(np.abs(Q[:,7])):.3e}"
                  f"   Newton |R|_inf = {np.max(np.abs(sol.fun)):.3e}")
        snapshots.append(((step + 1) * args.dt, Q.copy()))

    print(f"\nFinal: h ∈ [{Q[:, 0].min():.4f}, {Q[:, 0].max():.4f}]")
    print(f"  mass = {Q[:, 0].sum() * stepper.dx:.6f}  "
          f"(should be conserved; initial mass = "
          f"{snapshots[0][1][:, 0].sum() * stepper.dx:.6f})")

    # Plot if requested.
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            field_names = [f.func.__name__ for f in stepper.fields]
            fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
            x_grid = np.arange(stepper.Nx) * stepper.dx
            t_show = [0, len(snapshots) // 4, len(snapshots) // 2,
                      3 * len(snapshots) // 4, len(snapshots) - 1]
            for k, ax in enumerate(axes.flat):
                if k >= len(field_names):
                    ax.axis("off"); continue
                for ti in t_show:
                    t, Qs = snapshots[ti]
                    ax.plot(x_grid, Qs[:, k], lw=1.0, label=f"t={t:.2f}")
                ax.set_title(field_names[k]); ax.grid(True, alpha=0.3)
                if k == 0:
                    ax.legend(fontsize=7)
            fig.suptitle(f"VAM(1, 2, 2) direct DAE — backward Euler  "
                         f"(Nx={args.Nx}, dt={args.dt}, T={args.T})", y=1.00)
            fig.tight_layout()
            out = "tutorials/vam/vam_dae_simulate.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            print(f"\nPlot saved: {out}")
        except ImportError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
