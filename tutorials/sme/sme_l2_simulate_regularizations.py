"""SME L=2 — numerical comparison of regularisations.

Linearise around a base state inside the loss-of-hyperbolicity region;
evolve a Gaussian perturbation on a 1D periodic domain by **exact
Fourier evolution** (eigendecomposition per mode).  This avoids any
finite-difference artefact and shows the pure linear dynamics of each
regularised flux Jacobian.

Compared variants:
1. **Full SME** (route A) — original; small Im(λ) at this base state.
2. **Min-entry**: zero `A[1, 2]` only (our finding).
3. **K&T drop-feedback** (= ε = 0): replace `A` with `M_t_uu^{-1} M_x_uu`.
4. **Newtonian τ_xx**: add `2ν · ∂_xx u` to x-momentum (parabolic).

Each variant gives a (k-dependent) dispersion ω(k); we time-step each
Fourier mode exactly via the matrix exponential.

Outputs:
- per-variant time-evolution of |δq|_∞ and |δq|_2 over the domain.
- snapshots of δu_1(x, t) at t ∈ {0, T/4, T/2, 3T/4, T}.
- a comparison plot showing growth/decay rate vs time.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from sme_l2_hyperbolicity_compare import (   # noqa: E402
    build_route_a, _flat_bottom, h, u_0, u_1, u_2, g,
)
from zoomy_core.analysis import linearise, extract_quasilinear_pencil


# ---------------------------------------------------------------------------
# Build the linearised pencil of route A SME L=2 once.
# ---------------------------------------------------------------------------

def build_pencil_route_a():
    sa = _flat_bottom(build_route_a())
    base = {f: sp.Symbol(f.func.__name__ + "_bar", real=True) for f in sa.fields}
    lin = linearise(sa, base)
    M_t, [M_x], _ = extract_quasilinear_pencil(lin)
    return M_t, M_x, base, sa.fields


# ---------------------------------------------------------------------------
# Variant constructors: produce numerical (M_t, M_x [, M_xx]) at a given base.
# ---------------------------------------------------------------------------

def variant_full(M_t_n, M_x_n):
    return M_t_n, M_x_n, np.zeros_like(M_t_n)


def variant_min_entry(M_t_n, M_x_n):
    """Zero A[1, 2] in the principal-symbol matrix A = M_t⁻¹ M_x.
    Implementing as an M_x perturbation that renders the same A:
        new_M_x = M_t · A_mod  where A_mod[1, 2] = 0.
    """
    A = np.linalg.solve(M_t_n, M_x_n)
    A[1, 2] = 0.0
    return M_t_n, M_t_n @ A, np.zeros_like(M_t_n)


def variant_kt_drop_feedback(M_t_n, M_x_uu_n):
    """K&T-style: discard the Schur-feedback piece — use only the direct
    M_x_uu block.  Caller must pass M_x_uu (no feedback)."""
    return M_t_n, M_x_uu_n, np.zeros_like(M_t_n)


def variant_viscous(M_t_n, M_x_n, nu, H):
    """Add Newtonian τ_xx: M_xx super-diagonal = 2ν·diag(0, h, h/3, h/5)
    in the field-major super-diagonal pattern (see
    sme_l2_viscous_regularization.py)."""
    M_xx = np.zeros_like(M_t_n)
    M_xx[0, 1] = H        # cont_h-row, u_0-col gets viscous from xmom_j0 below
    # Actually for route A field order [h, u_0, u_1, u_2] and equation
    # order [cont_h, xmom_j0, xmom_j1, xmom_j2]:
    #   ν·∂_xx u_i term lives in xmom_j=i row, u_i column with
    #   coefficient h/(2i+1).
    # cont_h has no viscous term.
    M_xx = np.zeros((4, 4))
    M_xx[1, 1] = H            # xmom_j0:  2ν·h ∂_xx δu_0
    M_xx[2, 2] = H / 3.0      # xmom_j1:  2ν·(h/3) ∂_xx δu_1
    M_xx[3, 3] = H / 5.0      # xmom_j2:  2ν·(h/5) ∂_xx δu_2
    M_xx *= 2 * nu
    return M_t_n, M_x_n, M_xx


# ---------------------------------------------------------------------------
# Exact Fourier evolution of M_t · ∂_t δq + M_x · ∂_x δq + M_xx · ∂_xx δq = 0
# ---------------------------------------------------------------------------

def evolve_fourier(delta_q0_x, M_t, M_x, M_xx, x_grid, t_eval):
    """Evolve δq(x, 0) → δq(x, t) on a periodic domain of length L = x_grid[-1]
    + dx via exact mode-by-mode evolution.

    For each Fourier mode (real wavenumber k_n = 2π n / L), solve
        M_t · (-iω) q̂  +  M_x · (ik) q̂  +  M_xx · (-k²) q̂  = 0
    ⇒ ω(k) = eigenvalues of  M_t⁻¹ · ( k · M_x − i k² · M_xx )

    Returns ``δq(x, t_eval)`` of shape ``(len(t_eval), Nx, n_fields)``.
    """
    Nx = len(x_grid)
    L = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])
    n_fields = delta_q0_x.shape[1]

    # FFT of initial condition: shape (Nx, n_fields), each column FFT'd.
    qhat0 = np.fft.fft(delta_q0_x, axis=0) / Nx

    k_modes = 2 * np.pi * np.fft.fftfreq(Nx, d=(L / Nx))

    # For each mode k, compute the (n_fields, n_fields) propagator
    # P(k, t) = exp(-i ω(k) t).  We propagate the Fourier amplitudes:
    #   q̂(k, t) = P(k, t) q̂(k, 0).
    Mt_inv = np.linalg.inv(M_t)

    out = np.zeros((len(t_eval), Nx, n_fields), dtype=complex)
    # Pre-compute eigendecomposition per mode.
    for ki, k in enumerate(k_modes):
        # Matrix M(k) such that ∂_t q̂ = -i M(k) q̂.
        # From M_t·(-iω) + ik·M_x + (ik)²·M_xx = 0 ⇒
        #   -iω = -ik · Mt⁻¹ M_x + k² · Mt⁻¹ M_xx
        # So q̂(t) = exp((-ik · Mt⁻¹ M_x + k² · Mt⁻¹ M_xx) t) · q̂(0)
        #   wait: but for diffusion, M_xx > 0 should give DECAY, so the
        #   sign should be -k² (not +k²).  Re-derive:
        #     M_t ∂_t δq + M_x ∂_x δq + M_xx ∂_xx δq = 0
        #     M_t (-iω) + M_x (ik) + M_xx (-k²) = 0
        #     -iω = -ik Mt⁻¹ M_x + k² Mt⁻¹ M_xx
        #     iω = ik Mt⁻¹ M_x - k² Mt⁻¹ M_xx
        # Time evolution: q̂(t) = exp(-iω t) q̂(0)
        #   = exp(-(ik Mt⁻¹ M_x - k² Mt⁻¹ M_xx)·(t/i)·...
        # Simpler: q̂(t) = exp(-iωt)·q̂0 where ω matrix = Mt⁻¹·(k Mx + ik² Mxx)·sign(?).
        # Let M(k) ≡ Mt⁻¹·(k Mx − i k² Mxx); then ω(k) = eigvals(M(k)).
        # Our 'M_xx' coefficient of ∂_xx is positive for τ_xx > 0 (diffusive).
        # M(k) gets imaginary diagonal-decay from M_xx — Im(ω) negative for decay.
        Mk = Mt_inv @ (k * M_x - 1j * k**2 * M_xx)
        # Propagator: P(k, t) = exp(-i M(k) t)
        # Equivalent diagonalisation: M(k) = V Λ V⁻¹ ⇒ P = V exp(-iΛt) V⁻¹.
        try:
            eigs, V = np.linalg.eig(Mk)
            Vinv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            # Fall back: matrix-exp
            from scipy.linalg import expm
            for it, t in enumerate(t_eval):
                P = expm(-1j * Mk * t)
                out[it, ki, :] = P @ qhat0[ki, :]
            continue
        for it, t in enumerate(t_eval):
            diag = np.exp(-1j * eigs * t)
            P = V @ np.diag(diag) @ Vinv
            out[it, ki, :] = P @ qhat0[ki, :]

    # Inverse FFT.
    delta_q_xt = np.zeros((len(t_eval), Nx, n_fields), dtype=complex)
    for it in range(len(t_eval)):
        delta_q_xt[it] = np.fft.ifft(out[it] * Nx, axis=0)
    return delta_q_xt.real


# ---------------------------------------------------------------------------
# Initial condition + driver
# ---------------------------------------------------------------------------

def gaussian_ic(x_grid, n_fields, *, sigma, amplitude, components):
    """Place a Gaussian in the chosen components."""
    x0 = (x_grid[0] + x_grid[-1]) / 2
    G = amplitude * np.exp(-((x_grid - x0) / sigma) ** 2)
    delta_q0 = np.zeros((len(x_grid), n_fields))
    for c in components:
        delta_q0[:, c] = G
    return delta_q0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--n-snap", type=int, default=200)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=0.0)
    parser.add_argument("--U1", type=float, default=1.5)
    parser.add_argument("--U2", type=float, default=1.8)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--ic-sigma", type=float, default=0.5)
    parser.add_argument("--ic-amp", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("SME L=2 — exact Fourier evolution of regularisation comparison")
    print("=" * 70)

    # Build symbolic pencil of route A.
    M_t_sym, M_x_sym, base, fields = build_pencil_route_a()
    syms = [g, base[h], base[u_0], base[u_1], base[u_2]]
    f_M_t = sp.lambdify(syms, M_t_sym, "numpy")
    f_M_x = sp.lambdify(syms, M_x_sym, "numpy")

    # Also need M_x_uu only (drop-feedback) — derive via route B.
    from sme_l2_hyperbolicity_compare import build_route_b
    sb = _flat_bottom(build_route_b())
    base_b = {f: sp.Symbol(f.func.__name__ + "_bar", real=True) for f in sb.fields}
    lin_b = linearise(sb, base_b)
    M_t_b, [M_x_b], M_0_b = extract_quasilinear_pencil(lin_b)
    M_t_b_uu = M_t_b[:4, :4]
    M_x_b_uu_only = M_x_b[:4, :4]   # direct; no Schur-feedback contribution
    syms_b = [g, base_b[h], base_b[u_0], base_b[u_1], base_b[u_2]]
    f_M_t_b = sp.lambdify(syms_b, M_t_b_uu, "numpy")
    f_M_x_b = sp.lambdify(syms_b, M_x_b_uu_only, "numpy")

    params = (args.g, args.H, args.U0, args.U1, args.U2)
    M_t_n = np.asarray(f_M_t(*params), dtype=float)
    M_x_n = np.asarray(f_M_x(*params), dtype=float)
    print(f"Base state: H={args.H}, U_0={args.U0}, U_1={args.U1}, "
          f"U_2={args.U2}, g={args.g}, ν={args.nu}\n")
    eigs_full = np.linalg.eigvals(np.linalg.solve(M_t_n, M_x_n))
    print(f"Full SME (linearised) eigvals: {np.sort_complex(eigs_full)}")
    print(f"  max |Im λ|: {np.max(np.abs(np.imag(eigs_full))):.4f} "
          f"  → {'UNSTABLE' if np.max(np.abs(np.imag(eigs_full)))>1e-9 else 'stable'}")

    # Build per-variant matrices.
    M_t_kt = np.asarray(f_M_t_b(*params), dtype=float)
    M_x_kt = np.asarray(f_M_x_b(*params), dtype=float)

    variants = {
        "full SME":          variant_full(M_t_n, M_x_n),
        "min-entry (A[1,2]=0)": variant_min_entry(M_t_n, M_x_n),
        "K&T drop-feedback":  variant_kt_drop_feedback(M_t_kt, M_x_kt),
        f"τ_xx (ν={args.nu})": variant_viscous(M_t_n, M_x_n, args.nu, args.H),
    }

    print()
    print("Per-variant max |Im λ| (principal-symbol):")
    for name, (Mt, Mx, _) in variants.items():
        eigs = np.linalg.eigvals(np.linalg.solve(Mt, Mx))
        print(f"  {name:32s}: {np.max(np.abs(np.imag(eigs))):.4f}")

    # Initial condition: Gaussian in (δu_1, δu_2).
    dx = args.L / args.Nx
    x_grid = np.arange(args.Nx) * dx
    delta_q0 = gaussian_ic(x_grid, n_fields=4, sigma=args.ic_sigma,
                           amplitude=args.ic_amp, components=[2, 3])

    t_eval = np.linspace(0, args.T, args.n_snap)
    print(f"\nEvolving Gaussian δ(u_1, u_2) on periodic domain "
          f"[0, {args.L}] for T={args.T} ...")

    sols = {}
    for name, (Mt, Mx, Mxx) in variants.items():
        sols[name] = evolve_fourier(delta_q0, Mt, Mx, Mxx, x_grid, t_eval)

    # Time-series of L_inf and L_2 norms.
    norms_inf = {name: np.max(np.abs(s).max(axis=2), axis=1) for name, s in sols.items()}
    norms_2 = {name: np.linalg.norm(s, axis=(1, 2)) / np.sqrt(args.Nx)
               for name, s in sols.items()}

    print()
    print(f"  | δq |_∞ at final time T = {args.T}:")
    for name, n in norms_inf.items():
        ratio = n[-1] / n[0]
        print(f"    {name:32s}: {n[-1]:.4e}   (× {ratio:.2e} from t=0)")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # (a) L_inf norm vs time, semilogy.
        ax = axes[0, 0]
        for name, n in norms_inf.items():
            ax.semilogy(t_eval, n, lw=1.5, label=name)
        ax.set_xlabel("t"); ax.set_ylabel(r"$|\delta q|_\infty$")
        ax.set_title("L_inf norm vs time")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

        # (b) L_2 norm vs time, semilogy.
        ax = axes[0, 1]
        for name, n in norms_2.items():
            ax.semilogy(t_eval, n, lw=1.5, label=name)
        ax.set_xlabel("t"); ax.set_ylabel(r"$|\delta q|_2$")
        ax.set_title("L_2 norm vs time")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

        # (c) Snapshots of δu_1(x) at the final time.
        ax = axes[1, 0]
        for name, s in sols.items():
            ax.plot(x_grid, s[-1, :, 2], lw=1.2, label=name)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\delta u_1$")
        ax.set_title(f"$\\delta u_1(x, T={args.T})$ snapshot")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # (d) Snapshots of δu_2 at the final time.
        ax = axes[1, 1]
        for name, s in sols.items():
            ax.plot(x_grid, s[-1, :, 3], lw=1.2, label=name)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\delta u_2$")
        ax.set_title(f"$\\delta u_2(x, T={args.T})$ snapshot")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"SME L=2 regularisation comparison — exact Fourier evolution.   "
            f"Base: H={args.H}, U_0={args.U0}, U_1={args.U1}, U_2={args.U2}",
            y=1.00,
        )
        fig.tight_layout()
        out = "tutorials/sme/sme_l2_simulate_regularizations.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
