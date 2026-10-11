"""Figure: SWE 2-D bump O2 + MOOD on foam, 4 ranks (scotch) vs serial.
Regenerates from the gate 23 run dirs:  python gate23_foam_parallel.py bump2 4"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from zoomy_foam.symsolver import _read_dat, _read_labels
from gate9_foam import WORK

W, N = Path(WORK), 40
_, Q1 = _read_dat(W / "bump2_np1" / "out" / "final.dat")
Q4, rank = np.empty_like(Q1), np.empty(Q1.shape[1])
for r, d in enumerate(sorted((W / "bump2_np4").glob("processor*"))):
    c = _read_labels(d / "constant" / "polyMesh" / "cellProcAddressing")
    Q4[:, c], rank[c] = _read_dat(d / "out" / "final.dat")[1], r
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (v, t, cm) in zip(ax, [(Q1[1], "(a) h, serial", "viridis"),
                              (rank, "(b) rank (scotch, 4)", "tab10"),
                              (np.abs(Q4[1] - Q1[1]), "(c) |h 4 ranks - serial|", "magma")]):
    im = a.imshow(v.reshape(N, N), origin="lower", extent=(0, 10, 0, 10), cmap=cm)  # x fastest
    a.set_title(t)
    fig.colorbar(im, ax=a, shrink=0.8)
fig.tight_layout()
out = os.path.join(WORK, "foam_parallel_bump2.png")
fig.savefig(out, dpi=120)
print(out, "max |h4 - h1| =", np.abs(Q4[1] - Q1[1]).max())
