"""Figure of gate 43 (the C++ diffusion blocks vs JAX) from its store.

    GATE43_STORE=g43.json python gate43_cxx_diffusion.py [par]   # the store
    python gate43_figure.py g43.json                              # -> g43.png
(a) max rel |C++ - JAX| per run (foam, AMReX; explicit, implicit) with the
round-off / solver-tolerance bounds; (b) 2 ranks vs serial; (c) the 1-D
position-dependent Dirichlet case, implicit: u of JAX, foam, AMReX, exact.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

MARK = {"foam": "o", "amrex": "s"}
COLOUR = {"expl": "C0", "impl": "C3"}
FLOOR = 1e-17                      # exact zeros on the log axis


def runs(store, prefix):
    """[(label, backend, path, rel, bound)] of the entries ``[prefix] backend
    kind path case``."""
    out = []
    for k, v in store.items():
        w = k.split()
        if prefix and w[0] != prefix:
            continue
        b, kind, path, case = w[-4:]
        out.append((f"{kind} {case}", b, path, v["rel"], v["bound"]))
    return sorted(out, key=lambda r: (r[1], r[2], r[0]))


def panel(ax, rows, title):
    labels = sorted({r[0] for r in rows})
    y = {lab: i for i, lab in enumerate(labels)}
    for lab, b, path, rel, _ in rows:
        ax.scatter(max(rel, FLOOR), y[lab] + (0.15 if b == "amrex" else -0.15), marker=MARK[b],
                   facecolor="none", edgecolor=COLOUR[path], s=30)
    for path, bound in (("expl", 1e-12), ("impl", 1e-9)):
        ax.axvline(bound, color=COLOUR[path], ls="--", lw=1)
    ax.set(xscale="log", title=title, xlabel="max rel difference", xlim=(FLOOR / 3, 1e-7))
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for b, m in MARK.items():
        ax.scatter([], [], marker=m, facecolor="none", edgecolor="k", label=b)
    for p, c in COLOUR.items():
        ax.plot([], [], color=c, label=f"{p} (bound --)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def main(path):
    store = json.load(open(path))
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.2))
    serial = {k: v for k, v in store.items() if not k.startswith(("par", "profile"))}
    panel(ax[0], runs(serial, ""), "(a) C++ vs JAX, nx = 16")
    panel(ax[1], runs({k: v for k, v in store.items() if k.startswith("par")}, "par"),
          "(b) 2 ranks vs serial, nx = 32")
    for b, st in (("foam", "C0--"), ("amrex", "C1:")):
        p = store.get(f"profile {b} impl")
        if p is None:
            continue
        if b == "foam":
            ax[2].plot(p["x"], p["exact"], "k-", lw=0.8, label="exact")
            ax[2].plot(p["x"], p["jax"], "k.", ms=8, label="JAX")
        ax[2].plot(p["x"], p["cxx"], st, marker=MARK[b], mfc="none", lw=1, label=b)
    ax[2].set(title="(c) 1-D, u = 1 + 2x on the walls, CN, t = 0.05", xlabel="x", ylabel="u")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    out = Path(path).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
