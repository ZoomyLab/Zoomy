"""Figure of gate27_amrex (AMReX templates vs JAX) from its result store.

    python gate27_figure.py OUT        # OUT: gate27_amrex.py's directory
writes OUT/amrex_vs_jax.png: (a) dry dam break h along the diagonal,
(b) VAM(1) channel h + b and P_0 along the centre line, (c) |AMReX - JAX| of h
(dry dam break, 1 GPU).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def grid(Q, nx, ny):
    """Rows of a create_2d mesh (cells y fastest) as (rows, ny, nx)."""
    return Q.reshape(Q.shape[0], nx, ny).transpose(0, 2, 1)


def main(out):
    out = Path(out)
    load = lambda name: np.load(out / f"{name}.npz")       # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    r, n = load("ref_dry"), 48
    d = np.arange(n)
    ax[0].plot(d / n * np.sqrt(2), grid(r["Q"], n, n)[1][d, d], "k-", lw=2, label="JAX")
    for c, st in (("cpu1", "C0--"), ("gpu4", "C1:")):
        if (out / f"dry_{c}.npz").exists():
            ax[0].plot(d / n * np.sqrt(2), grid(load(f"dry_{c}")["Q"], n, n)[1][d, d], st,
                       lw=1.5, label=f"AMReX {c}")
    ax[0].set(title="(a) dry dam break, t = 0.02", xlabel="diagonal", ylabel="h")
    ax[0].legend(frameon=False)
    r, nx, ny = load("ref_vam"), 60, 6
    x = -1.5 + (np.arange(nx) + 0.5) * 3.0 / nx
    R = grid(r["Q"], nx, ny)[:, ny // 2]
    ax[1].plot(x, R[0] + R[1], "k-", lw=2, label="JAX h + b")
    ax[1].plot(x, R[0], "k-", lw=0.8)
    ax2 = ax[1].twinx()
    ax2.plot(x, R[8], "k-", lw=1, alpha=0.5)
    for c, st in (("cpu1", "C0--"), ("gpu4", "C1:")):
        if (out / f"vam_{c}.npz").exists():
            A = grid(load(f"vam_{c}")["Q"], nx, ny)[:, ny // 2]
            ax[1].plot(x, A[0] + A[1], st, lw=1.5, label=f"AMReX {c}")
            ax2.plot(x, A[8], st, lw=1, alpha=0.7)
    ax[1].set(title="(b) VAM(1) channel, t = 0.05", xlabel="x", ylabel="h + b")
    ax2.set_ylabel("P_0 (thin)")
    ax[1].legend(frameon=False, loc="lower left")
    if (out / "dry_gpu1.npz").exists():
        diff = np.abs(grid(load("dry_gpu1")["Q"] - load("ref_dry")["Q"], n, n)[1])
        im = ax[2].imshow(np.log10(diff + 1e-18), origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax[2], label="log10 |h AMReX - h JAX|")
    ax[2].set(title="(c) dry dam break, 1 GPU - JAX", xlabel="i", ylabel="j")
    fig.tight_layout()
    fig.savefig(out / "amrex_vs_jax.png", dpi=130)
    print(out / "amrex_vs_jax.png")


if __name__ == "__main__":
    main(sys.argv[1])
