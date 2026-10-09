"""Results figure: (a) SWE bump O2 new vs existing solver, (b) radial dry dam
break eta+MOOD, (c) preCICE split dam break vs single domain."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zoomy_core.postprocessing import style
import common as C
from gate1_vs_solver_jax import old_march, eta_kw
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "symbolic_hyperbolic_jax.png")
style.use("screen")

# (a) SWE 2-D bump, order 2, walls: diagonal cut, new vs existing
mesh, nsm = C.bump_2d(2)
s = SymbolicHyperbolicSolverJax(time_end=0.5, cfl=0.9)
Q0, A0 = s.setup(mesh, nsm)
_, Qn, _ = s.run(Q0, A0)
Qo, _ = old_march(lambda: C.bump_2d(2), 0.5, 0.9)
cc = np.asarray(mesh.cell_centers)[:, :Qo.shape[1]]
cut = np.abs(cc[1] - 5.0) < 0.13
order_x = np.argsort(cc[0][cut])

# (b) radial dry dam break, eta + MOOD
mesh_b, nsm_b = C.dry_dambreak_2d()
sb = SymbolicHyperbolicSolverJax(time_end=0.02, cfl=0.45, mood=True)
Qb0, Ab0 = sb.setup(mesh_b, nsm_b)
_, Qb, _ = sb.run(Qb0, Ab0)
nx = 48
hb = np.asarray(Qb)[1].reshape(nx, nx)
ccb = np.asarray(mesh_b.cell_centers)[:, :nx * nx]

# (c) preCICE
x = (np.arange(200) + 0.5) / 200
fig, ax = plt.subplots(3, 1, figsize=(8, 11))
ax[0].plot(cc[0][cut][order_x], np.asarray(Qn)[1][cut][order_x], color=style.CYCLE[0],
           marker="", lw=2.5, label="symbolic solver")
ax[0].plot(cc[0][cut][order_x], Qo[1][cut][order_x], color=style.CYCLE[1], marker="",
           ls="--", label="existing solver_jax")
ax[0].set(xlabel="x", ylabel="h", title="SWE bump, order 2")
ax[0].text(0.02, 0.9, f"max |diff| = {np.max(np.abs(np.asarray(Qn) - Qo)):.1e}",
           transform=ax[0].transAxes)
ax[0].legend(loc="lower right")
xs = np.unique(np.round(ccb[0], 10)); ys = np.unique(np.round(ccb[1], 10))
H = np.full((len(ys), len(xs)), np.nan)
ix = np.searchsorted(xs, np.round(ccb[0], 10)); iy = np.searchsorted(ys, np.round(ccb[1], 10))
H[iy, ix] = np.asarray(Qb)[1]
im = ax[1].pcolormesh(xs, ys, H, cmap=style.CMAP_CONTINUOUS, shading="nearest")
ax[1].set(aspect="equal", xlabel="x", ylabel="y", title="Radial dry dam break, MOOD")
fig.colorbar(im, ax=ax[1], label="h")
for order, c in ((1, style.CYCLE[0]), (2, style.CYCLE[2])):
    run = os.path.join(HERE, f"run_precice_o{order}")
    Qc = np.concatenate([np.load(f"{run}/Left.npy"), np.load(f"{run}/Right.npy")], 1)
    Qm = np.load(f"{run}/single.npy")
    ax[2].plot(x, Qm[1], color=c, marker="", lw=2.5, label=f"single domain, order {order}")
    ax[2].plot(x, Qc[1], color="k", marker="", ls=":", lw=1.5,
               label="preCICE halves" if order == 1 else None)
ax[2].axvline(0.5, color="0.5", lw=1)
ax[2].set(xlabel="x", ylabel="h", title="Dam break split by preCICE")
ax[2].legend(loc="upper right")
for a, l in zip(ax, "abc"):
    a.set_title(f"({l}) {a.get_title()}")
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(OUT)
