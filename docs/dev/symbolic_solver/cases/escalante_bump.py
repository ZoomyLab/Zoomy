"""Escalante dam break over a bump (thesis/cases/escalante_vam_bump) with the
symbolic Chorin solver: derived VAM(1,2), inflow q_0 = Q_IN on the left,
pressure pinned to 0 on the right, 60 cells, t = 20 s.  Free surface and
bottom pressure p_b/g = p(zeta=0)/g (the model's own interpolate_to_3d) at the
final time against the digitised experiment.

    python escalante_bump.py [jax] [foam] [old]
"""
import os
import sys
import time as wall

import numpy as np

CASE = "/Users/adam-obbpb5az1dhsjzf/git/Zoomy-solver-template/thesis/cases/escalante_vam_bump"
GATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gates")
sys.path[:0] = [CASE, GATES]
import common  # noqa: F401,E402  (x64)
import jax.numpy as jnp  # noqa: E402
import sympy as sp  # noqa: E402

import make_gif as M  # noqa: E402
from run import ETA_EXP_X, ETA_EXP_Y, PB_EXP_X, PB_EXP_Y  # noqa: E402
from run_derived import BUMP, DOMAIN, G, H_DRY, H_RES, NCELLS, build_model  # noqa: E402
from gate7_chorin import pred_nsm  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.initial_conditions import Constant  # noqa: E402
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL  # noqa: E402

T_END, DT, DT_WRITE = 20.0, 2.0 ** -8, 0.5          # binary-exact time sums
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
WORK = ("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-"
        "zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/foam_cases")


def split(dt_symbol):
    model = build_model()
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(dt_symbol, system_model=sm)
    return r.SM_pred, r.SM_press, r.SM_corr


def initial_state(xc, n):
    Q = np.zeros((n, len(xc)))
    b = BUMP(xc)
    Q[0] = b
    Q[1] = np.maximum(np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    return Q


def run_symbolic(backend):
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
    pred, press, corr = split(_DT_SYMBOL)
    frames = []
    s = SymbolicChorinSolverJax(
        time_end=T_END, dt=DT, cfl=0.9, dt_write=DT_WRITE,
        on_write=lambda t, Q, A: frames.append((float(t), np.asarray(Q))))
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=NCELLS)
    Q, Qaux = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(initial_state(xc, Q.shape[0]))
    Qaux = s.initial_aux(Q, Qaux)
    t0 = wall.perf_counter()
    if backend == "jax":
        t, Q, _ = s.run(Q, Qaux)
        import jax
        jax.effects_barrier()
        Q = np.asarray(Q)
    else:
        from zoomy_foam.symsolver import SymbolicChorinSolverFoam
        sf = SymbolicChorinSolverFoam(time_end=T_END, dt=DT, cfl=0.9)
        t, Q = sf.run(s.mesh_np, (s.nsm, s.press, s.corr), np.asarray(Q),
                      f"{WORK}/escalante_bump")
    return float(t), xc, Q, wall.perf_counter() - t0, frames


def run_old():
    from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
    pred, press, corr = split(sp.Symbol("dt", positive=True))
    s = ChorinSplitVAMSolverJax(pred, press, corr, pressure_tol=1e-9,
                                pressure_maxit=200)
    Q0 = np.array(s.setup_simulation(LSQMesh.create_1d(domain=DOMAIN,
                                                       n_inner_cells=NCELLS)))
    nc = s.nc
    xc = np.asarray(s._rt_mesh.cell_centers)[0, :nc]
    Q0[:, :nc] = initial_state(xc, Q0.shape[0])
    t0 = wall.perf_counter()
    Q, *_ = s.run_jit_steps(jnp.asarray(DT), int(round(T_END / DT)), jnp.asarray(Q0))
    return T_END, xc, np.asarray(Q)[:, :nc], wall.perf_counter() - t0, []


def pb_head(Q):
    names = ["b", "h", "q_0", "q_1", "r_0", "r_1", "P_0", "P_1"]
    _, p_fn = M.build_profile_fns(names, "vam")
    return np.array([float(p_fn(0.0, *Q[:, i])) / G for i in range(Q.shape[1])])


def rms_vs(xe, ye, xc, y):
    return float(np.sqrt(np.mean((np.interp(xe, xc, y) - ye) ** 2)))


if __name__ == "__main__":
    which = sys.argv[1:] or ["jax", "old"]
    runs = {}
    for w in which:
        runs[w] = run_old() if w == "old" else run_symbolic(w)
        t, xc, Q, sec, _ = runs[w]
        eta, pb = Q[0] + Q[1], pb_head(Q)
        print(f"{w:4s}: t = {t:.4g}, finite {np.isfinite(Q).all()}, h_min {Q[1].min():.4f}, "
              f"RMS vs experiment: eta {rms_vs(ETA_EXP_X, ETA_EXP_Y, xc, eta):.4f} m, "
              f"p_b/g {rms_vs(PB_EXP_X, PB_EXP_Y, xc, pb):.4f} m  [{sec:.0f}s]", flush=True)
    names = list(runs)
    for a in names:
        for b in names:
            if a < b:
                print(f"max|{a}-{b}|: {np.abs(runs[a][2] - runs[b][2]).max():.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zoomy_plotting as zp
    zp.use("thesis")
    label = {"jax": "symbolic Chorin, JAX", "foam": "symbolic Chorin, foam (C++)",
             "old": "ChorinSplitVAMSolverJax (old)"}
    style = {"jax": "-", "foam": "--", "old": ":"}
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    xc = runs[names[0]][1]
    a0.fill_between(xc, 0, BUMP(xc), color="0.75", lw=0)
    for w, (t, xc, Q, _, _) in runs.items():
        a0.plot(xc, Q[0] + Q[1], style[w], lw=1.6, label=label[w])
        a1.plot(xc, pb_head(Q), style[w], lw=1.6, label=label[w])
    a0.plot(ETA_EXP_X, ETA_EXP_Y, "ko", ms=3.5, label="experiment")
    a1.plot(PB_EXP_X, PB_EXP_Y, "ko", ms=3.5, label="experiment")
    a0.set_ylabel(r"$\eta$ [m]"); a1.set_ylabel(r"$p_b/(\rho g)$ [m]")
    a1.set_xlabel(r"$x$ [m]"); a0.set_ylim(0, 0.42); a1.set_ylim(0, 0.42)
    a0.set_title(f"(a) free surface, t = {T_END:g} s"); a1.set_title("(b) bottom pressure")
    a0.legend(fontsize=7); a1.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT, "escalante_bump_symbolic.png")
    fig.savefig(path, dpi=150)
    frames = runs.get("jax", (None,) * 5)[4]
    if frames:
        np.savez(os.path.join(OUT, "escalante_bump_symbolic_frames.npz"),
                 t=[f[0] for f in frames], Q=np.stack([f[1] for f in frames]), xc=xc)
    print("figure", path)
    print("BUMP_DONE")
