"""Figure of ANALYSIS 48 from the run store beside this file (`run.py`):

    python deliverable.py   -> malpasset_wavespeed.png

(a) min h, first 60 s: the faces rule on main goes negative at t = 31 s (MOOD's first-order repair
    at the order-2 candidate's step); with each repair at its own step it stays 0;
(b) max |q_1|/h over h > 1e-3 m: the faces rule without the cells' speeds lets the first moment of
    a cell with Audusse-dry order-2 traces run away; with them it stays at the cells rule's level;
(c) wet area (h > 0.05 m): the cells rule (the session default) on main and on this branch.
"""
import os

import numpy as np

from zoomy_plotting.plot import panels, style

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {   # label: (legend, color, line style)
    "main_faces": ("faces, main", "#C8102E", "-"),
    "faces_mood_only": ("faces, repair at its own step only", "#E69F00", "-"),
    "faces": ("faces, this branch", "#0072B2", "-"),
    "cells": ("cells, this branch", "#000000", "--"),
    "main_cells": ("cells, main", "#999999", ":"),
}


def load(label):
    p = os.path.join(HERE, f"{label}.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def series(labels, y, t_max=None):
    out = []
    for lab in labels:
        d = load(lab)
        if d is None:
            continue
        name, color, ls = RUNS[lab]
        k = np.ones_like(d["t"], bool) if t_max is None else d["t"] <= t_max
        out.append(dict(x=d["t"][k], y=y(d)[k], label=name, color=color, ls=ls, marker="",
                        lw=1.2))
    return out


def main():
    fig, ax = style.subplots(1, 3, preset="thesis")
    panels.line_plot(ax[0], series(("main_faces", "faces", "cells"), lambda d: d["h_min"], 60.0),
                     xlabel="t (s)", ylabel="min h (m)", title="(a) depth minimum")
    panels.line_plot(ax[1], series(("faces_mood_only", "faces", "cells", "main_cells"),
                                   lambda d: np.maximum(d["alpha1_max"], 1e-1)),
                     xlabel="t (s)", ylabel=r"max $|q_1|/h$ (m/s)", logy=True,
                     title="(b) first moment")
    panels.line_plot(ax[2], series(("main_cells", "cells", "faces"),
                                   lambda d: d["wet_area"] / 1e6),
                     xlabel="t (s)", ylabel=r"wet area (km$^2$)", title="(c) inundation")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    panels.row_legend(fig, list(ax), ncol=3, reserve=0.13)
    out = os.path.join(HERE, "malpasset_wavespeed.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
