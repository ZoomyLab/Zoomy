"""Figure of the test matrix from its store:  python deliverable.py  -> matrix.png
(a) C1 L1(h) vs nx, (b) C4 L1(h) vs the exact Bernoulli depth, (c) C4
transcritical O2 h + b at nx 200, (d) C8 VAM P_0 along the channel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from matrix import HERE, err_exact, err_ref, h, load   # noqa: E402
from cases import bed, bernoulli                       # noqa: E402

STYLE = {"jax": ("k", "o", "-"), "foam": ("C0", "s", "--"), "amrex": ("C3", "^", ":")}


def main():
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    ref = load("jax", "mpi8", "C1", "ref_n320")
    for b, (c, m, ls) in STYLE.items():
        for o, fill in ((1, "none"), (2, c)):
            d = [load(b, "serial", "C1", f"o{o}_n{n}") for n in (40, 80, 160)]
            if ref is not None and all(x is not None for x in d):
                ax[0].loglog((40, 80, 160), [err_ref(x, ref) for x in d], color=c, marker=m,
                             ls=ls, mfc=fill, label=f"{b} O{o}")
        for v, fill in (("sub", c), ("trans", "none")):
            d = [load(b, "serial", "C4", f"{v}_o2_n{n}") for n in (50, 100, 200)]
            if all(x is not None for x in d):
                ax[1].loglog((50, 100, 200), [err_exact(x, v) for x in d], color=c, marker=m,
                             ls=ls, mfc=fill, label=f"{b} {v} O2")
        d = load(b, "serial", "C4", "trans_o2_n200")
        if d is not None:
            k = d["xc"][1] < 0.5
            i = np.argsort(d["xc"][0][k])
            ax[2].plot(d["xc"][0][k][i], (h(d) + d["Q"][0])[k][i], color=c, ls=ls, label=b)
        d = load(b, "serial", "C8", "vam")
        if d is not None:
            names = [str(n) for n in d["names"]]
            k = np.isclose(d["xc"][1], d["xc"][1].min())
            i = np.argsort(d["xc"][0][k])
            ax[3].plot(d["xc"][0][k][i], d["Q"][names.index("P_0")][k][i], color=c, ls=ls, label=b)
    n = np.array([40, 160])
    ax[0].loglog(n, 2e-1 / n, "0.6", lw=0.8)
    ax[0].loglog(n, 1.5 / n ** 2, "0.6", lw=0.8)
    ax[0].set(title="(a) C1 smooth bump: L1(h) vs JAX 320", xlabel="nx")
    ax[1].set(title="(b) C4 channel O2: L1(h) vs Bernoulli", xlabel="nx")
    x = np.linspace(0, 25, 400)
    ax[2].plot(x, bernoulli("trans", x) + bed(x), "0.5", lw=3, alpha=0.5, label="exact")
    ax[2].plot(x, bed(x), "0.3", lw=0.8)
    ax[2].set(title="(c) C4 transcritical O2, nx 200", xlabel="x", ylabel="h + b")
    ax[3].set(title="(d) C8 VAM(1) Chorin: P_0, bottom row", xlabel="x")
    for a in ax:
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "matrix.png", dpi=130)
    print(HERE / "matrix.png")


if __name__ == "__main__":
    main()
