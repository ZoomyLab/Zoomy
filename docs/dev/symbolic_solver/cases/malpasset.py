"""Malpasset dam break (thesis/cases/malpasset_jax) with the symbolic
hyperbolic solver: the case's gmsh mesh (geo_malpasset-small.msh), its
MalpassetSME(level 1) model with IC from the mesh survey data and walls, its
numerics (free-surface eta reconstruction, positive HLL, CFL 0.4, order 1).
Written in the case's HDF5 layout and plotted with its map plot.

    python malpasset.py T_END [old]       (JAX_PLATFORMS picks cpu / cuda)
"""
import os
import sys
import time as wall

import numpy as np

CASE = "/Users/adam-obbpb5az1dhsjzf/git/Zoomy-solver-template/thesis/cases/malpasset_jax"
GATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gates")
sys.path[:0] = [CASE, GATES]
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

from mesh import build_mesh  # noqa: E402
from model import build_model  # noqa: E402
from run import load_settings  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax  # noqa: E402

T_END = (float(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1
         else 2.0)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEV = jax.devices()[0].platform
OUT = os.path.join(RES, f"malpasset_symbolic_{DEV}.h5")


def build():
    settings = load_settings()
    mesh = build_mesh(settings, CASE)
    return mesh, build_model(settings, mesh, CASE)


def symbolic():
    mesh, b = build()
    s = SymbolicHyperbolicSolverJax(time_end=T_END, cfl=0.4, mood=True,
                                    dt_write=T_END / 4, output=OUT)
    Q, Qaux = s.setup(mesh, b["nsm"])
    t0 = wall.perf_counter()
    t, Q, Qaux = s.run(Q, Qaux)
    Q = np.asarray(Q)
    jax.effects_barrier()
    return float(t), Q, wall.perf_counter() - t0, s


def old():
    from gate1_vs_solver_jax import old_march
    import zoomy_core.fvm.timestepping as ts
    mesh, b = build()
    fs = b["fs_kwargs"]
    t0 = wall.perf_counter()
    import gate1_vs_solver_jax as g1
    g1.C.adaptive = lambda cfl, mesh: ts.adaptive(CFL=cfl)
    Q, n = old_march(lambda: (mesh, b["nsm"]), T_END, 0.4, **fs)
    return Q, n, wall.perf_counter() - t0


if __name__ == "__main__":
    t, Q, sec, s = symbolic()
    vol = np.asarray(s.mesh_np.cell_volumes)[:Q.shape[1]]
    h = Q[1]
    Q0 = np.asarray(s.mesh_np.cell_centers)  # noqa: F841
    print(f"symbolic JAX ({DEV}): {Q.shape[1]} cells, t = {t:.4g}, h in "
          f"[{h.min():.3g}, {h.max():.3g}], finite {np.isfinite(Q).all()}  [{sec:.1f}s incl. compile]")
    if "old" in sys.argv[2:]:
        Qo, n, sec_o = old()
        d = np.abs(Q - Qo).max()
        print(f"old solver_jax: {n} steps [{sec_o:.1f}s]; max|new-old| = {d:.2e} "
              f"(rel {d / np.abs(Qo).max():.1e}), mass new-old "
              f"{abs((Q[1] * vol).sum() - (Qo[1] * vol).sum()):.1e}")

    # the case's own post-processing: read_hdf5 + plot_topo_field
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zoomy_plotting as zp
    from zoomy_plotting import read_hdf5
    from zoomy_core.postprocessing import mesh_plots
    from postprocess import HIDE_BELOW
    TOPO, FIELD = "b", "h"          # this store carries the state names
    zp.use("thesis")
    store = read_hdf5(OUT)
    k = store.n_snapshots - 1
    fig, ax = plt.subplots(figsize=(9, 6))
    o = mesh_plots.plot_topo_field(ax, store, topo=TOPO, field=FIELD, time_step=k,
                                   hide_below=HIDE_BELOW, colorbar=True,
                                   topo_colorbar=True)
    if o.get("colorbar"):
        o["colorbar"].set_label("water depth h [m]")
    ax.set_title(f"Malpasset, SME(1), symbolic solver ({DEV})   "
                 f"t = {list(store.times)[k]:.1f} s")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    path = os.path.join(RES, f"malpasset_symbolic_{DEV}_t{T_END:g}.png")
    fig.savefig(path, dpi=130)
    print("snapshots", store.n_snapshots, "figure", path)
    print("MALPASSET_DONE")
