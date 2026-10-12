"""Figure of the test matrix from its store:  python deliverable.py  -> matrix.png
(a) C1 L1(h) vs nx, (b) C4 O2 L1(h) vs the exact Bernoulli depth, (c) C10
VAM(1) Chorin self-convergence, (d) C8 VAM P_0 along the channel, (e) C18
Sod 3-D density vs the exact solution, (f) C12 triangles L1(h) vs nx."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from matrix import HERE, err_exact, err_ref, load   # noqa: E402
from cases import C10_N, C10_VARIANTS, C4_N        # noqa: E402

STYLE = {"jax": ("k", "o", "-"), "foam": ("C0", "s", "--"), "amrex": ("C3", "^", ":")}


def loglog(ax, ns, ds, err, **kw):
    if all(d is not None for d in ds):
        ax.loglog(ns, [err(d) for d in ds], **kw)


def main():
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    ref = load("jax", "mpi8", "C1", "ref_n320")
    for b, (c, m, ls) in STYLE.items():
        st = dict(color=c, marker=m, ls=ls)
        for o, fill in ((1, "none"), (2, c)):
            if ref is not None:
                loglog(ax[0], (20, 40, 80), [load(b, "serial", "C1", f"o{o}_n{n}") for n in (20, 40, 80)],
                       lambda d: err_ref(d, ref), mfc=fill, label=f"{b} O{o}", **st)
                loglog(ax[5], (10, 20, 40), [load(b, "serial", "C12", f"bump_o{o}_n{n}") for n in (10, 20, 40)],
                       lambda d: err_ref(d, ref), mfc=fill, label=f"{b} O{o}", **st)
        for v, fill in (("sub", c), ("trans", "none")):
            loglog(ax[1], C4_N, [load(b, "serial", "C4", f"{v}_o2_n{n}") for n in C4_N],
                   lambda d: err_exact(d, v), mfc=fill, label=f"{b} {v}", **st)
        for (o, to), fill in zip(C10_VARIANTS, ("none", "0.7", c)):
            e = [load(b, "serial", "C10", f"s{o}_t{to}_n{n}") for n in C10_N]
            if all(d is not None for d in e):
                ax[2].loglog(C10_N[:2], [err_ref(e[0], e[1]), err_ref(e[1], e[2])], mfc=fill,
                             label=f"{b} space O{o} time O{to}", **st)
        d = load(b, "serial", "C8", "vam")
        if d is not None:
            names = [str(n) for n in d["names"]]
            k = np.isclose(d["xc"][1], d["xc"][1].min())
            i = np.argsort(d["xc"][0][k])
            ax[3].plot(d["xc"][0][k][i], d["Q"][names.index("P_0")][k][i], color=c, ls=ls, label=b)
        d = load(b, "serial", "C18", "o2_n100_ax0")
        if d is not None:
            i = np.argsort(d["xc"][0])
            ax[4].plot(d["xc"][0][i], d["Q"][0][i], color=c, ls=ls, label=f"{b} O2")
    import euler_common as E
    x = np.linspace(0, 1, 400)
    ax[4].plot(x, E.exact_riemann(E.SOD[0], E.SOD[1], (x - 0.5) / 0.2)[0], "0.5", lw=3,
               alpha=0.5, label="exact")
    ax[0].set(title="(a) C1 bump, quads: L1(h) vs JAX 320", xlabel="nx")
    ax[1].set(title="(b) C4 channel O2: L1(h) vs Bernoulli", xlabel="nx")
    ax[2].set(title="(c) C10 VAM(1) Chorin: |h_n - h_2n|", xlabel="nx")
    ax[3].set(title="(d) C8 VAM(1) Chorin: P_0, bottom row", xlabel="x")
    ax[4].set(title="(e) C18 Euler 3-D Sod, n 100: density", xlabel="x")
    ax[5].set(title="(f) C12 bump, triangles: L1(h) vs JAX 320", xlabel="nx")
    for a in ax:
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "matrix.png", dpi=120)
    print(HERE / "matrix.png")


if __name__ == "__main__":
    main()
