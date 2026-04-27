"""Plotting demo for ``zoomy_core.analysis``.

Two figures:

  1. **VAM dispersion vs Airy** at (M, N) = (1, 2) and (2, 3).
     Plots ``C²/(gH)`` over ``kH ∈ [0, 5]`` for both VAM orders alongside
     the exact Airy curve ``tanh(kH)/(kH)``.  Should reproduce the
     style of Fig. 2 of Escalante et al. 2024.

  2. **SME L=2 hyperbolic region** in the plane of normalised second
     and third moments (``α₁/√(gH), α₂/√(gH)``) at fixed ``α₀ = 0``.
     Reproduces the K&T-style loss-of-hyperbolicity boundary.

Saves the plots to ``./vam_dispersion.png`` and
``./sme_l2_hyperbolic_region.png`` (or ``--out`` directory).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")                                      # headless
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Re-use the model builders.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vam"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sme"))
from escalante2024_dispersion import build_vam_pde_system            # noqa
from escalante2024_generic import b, g                                # noqa
from sme_builder import build_sme_pde_system, g as g_sme              # noqa

from zoomy_core.analysis import (                                     # noqa
    linearise,
    plane_wave_dispersion,
    extract_quasilinear_pencil,
    plot_dispersion,
    plot_hyperbolic_region_2d,
)


# ---------------------------------------------------------------------------
# Figure 1 — VAM dispersion vs Airy
# ---------------------------------------------------------------------------

def figure_vam_dispersion(out_dir):
    H = sp.Symbol("H", positive=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    H_val = 1.0           # normalise so kH = k.
    g_val = 1.0           # normalise so C^2/(gH) is dimensionless.

    # VAM (1, 2)
    sys_vam, *_ = build_vam_pde_system(1, 2)
    sys_vam = sys_vam.with_substitutions({b: -H})
    base = {f: sp.S.Zero for f in sys_vam.fields}
    base[sys_vam.fields[0]] = H                # h field is first
    sys_lin = linearise(sys_vam, base)
    print("Solving VAM(1,2) dispersion ...")
    disp12 = plane_wave_dispersion(sys_lin, simplify=True,
                                    factor_in_target=True)
    plot_dispersion(
        disp12, k_range=(0.05, 5.0), ax=ax,
        n_points=120, mode="phase_velocity", squared=True,
        fixed_subs={H: H_val, g: g_val},
        nondimensionalise_by=("gH", g * H),
        mode_labels=[None, "VAM(1,2)", None],
        drop_zero_modes=True,
        title=None,
    )

    # VAM (2, 3) — heavier solve; reduce k-resolution.
    sys_vam, *_ = build_vam_pde_system(2, 3)
    sys_vam = sys_vam.with_substitutions({b: -H})
    base = {f: sp.S.Zero for f in sys_vam.fields}
    base[sys_vam.fields[0]] = H
    sys_lin = linearise(sys_vam, base)
    print("Solving VAM(2,3) dispersion ...")
    disp23 = plane_wave_dispersion(sys_lin, simplify=True,
                                    factor_in_target=True)
    plot_dispersion(
        disp23, k_range=(0.05, 5.0), ax=ax,
        n_points=120, mode="phase_velocity", squared=True,
        fixed_subs={H: H_val, g: g_val},
        nondimensionalise_by=("gH", g * H),
        mode_labels=[None, "VAM(2,3)", None],
        drop_zero_modes=True,
    )

    # Airy reference:  C²/(gH) = tanh(kH)/(kH)
    k_arr = np.linspace(0.05, 5.0, 200)
    airy = np.tanh(k_arr * H_val) / (k_arr * H_val)
    ax.plot(k_arr, airy, "--k", linewidth=1.5, label="Airy (exact)")

    # Tidy up.
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$kH$")
    ax.set_ylabel(r"$C^2 / (g H)$")
    ax.set_title("VAM phase-velocity vs Airy reference")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out = os.path.join(out_dir, "vam_dispersion.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — SME L=2 hyperbolic region
# ---------------------------------------------------------------------------

def figure_sme_hyperbolic_region(out_dir, level=2, n_grid=80):
    sys_sme, h, u_coeffs = build_sme_pde_system(level)
    H = sp.Symbol("H_bar", positive=True)
    consts = []
    base = {h: H}
    for i, ui in enumerate(u_coeffs):
        sym = sp.Symbol(f"U_{i}_bar", real=True)
        base[ui] = sym
        consts.append(sym)
    sys_lin = linearise(sys_sme, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    M_x = M_xa[0]

    # Normalise by sqrt(gH): plot in (s/c, kappa/c) where c = sqrt(gH).
    # Substitute g, H = 1; vary U_1, U_2 over wide range; fix U_0 = 0
    # (frame moving with the depth-averaged flow).
    fixed = {g_sme: 1.0, H: 1.0, consts[0]: 0.0}
    if level >= 3:
        for j in range(3, level + 1):
            fixed[consts[j]] = 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_hyperbolic_region_2d(
        M_x, M_t,
        axis_a=(consts[1], -3.0, 3.0),                # α_1 / sqrt(gH)
        axis_b=(consts[2], -3.0, 3.0),                # α_2 / sqrt(gH)
        fixed_subs=fixed,
        ax=ax, n_a=n_grid, n_b=n_grid,
        show="binary",
        title=f"SME L={level} hyperbolic region "
              f"(α₀ = 0, gH = 1; {n_grid}×{n_grid} grid)",
    )
    ax.set_xlabel(r"$\alpha_1 / \sqrt{gH}$")
    ax.set_ylabel(r"$\alpha_2 / \sqrt{gH}$")

    out = os.path.join(out_dir, f"sme_l{level}_hyperbolic_region.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=str, default=".",
                        help="output directory (default: cwd)")
    parser.add_argument("--sme-level", type=int, default=2)
    parser.add_argument("--sme-grid", type=int, default=80,
                        help="resolution for SME hyperbolic region (default 80)")
    parser.add_argument("--skip-vam", action="store_true")
    parser.add_argument("--skip-sme", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.skip_vam:
        print("=== Figure 1: VAM dispersion vs Airy ===")
        figure_vam_dispersion(args.out)
    if not args.skip_sme:
        print(f"=== Figure 2: SME L={args.sme_level} hyperbolic region "
              f"({args.sme_grid}×{args.sme_grid}) ===")
        figure_sme_hyperbolic_region(args.out, level=args.sme_level,
                                     n_grid=args.sme_grid)

    print("Done.")


if __name__ == "__main__":
    main()
