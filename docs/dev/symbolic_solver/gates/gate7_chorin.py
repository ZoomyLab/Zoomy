"""Gate 7: symbolic Chorin (JAX) vs ChorinSplitVAMSolverJax.

Escalante VAM(1) dam break over a bump, 60 cells, open ends (the config of
zoomy_jax/tests/runtime/test_vam_chorin_bump.py), constant dt = 2^-7 to
t = 0.5 (64 steps, exact time sums), time_order 1 and 2.
"""
import sys
import time as wall

import common  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

T = "/Users/adam-obbpb5az1dhsjzf/git/Zoomy-solver-template/library/zoomy_jax/tests"
sys.path[:0] = [T, T + "/runtime"]
import cases  # noqa: E402
import models  # noqa: E402

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.fvm.solver_chorin_vam_numpy import _pad_to_square
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

DT, N_STEPS = 2.0 ** -7, 64


def split(dt_symbol):
    m = models.vam(level=1, dimension=2, bc="bump")
    r = m.chorin_split(dt_symbol, system_model=SystemModel.from_model(m))
    return r.SM_pred, r.SM_press, r.SM_corr


def mesh():
    return LSQMesh.create_1d(domain=cases.ESC_DOMAIN,
                             n_inner_cells=cases.ESC_NCELLS)


def ic(nc, xc):
    return np.stack([cases.bump_ic(xc[:, j]) for j in range(nc)], axis=1)


def pred_nsm(pred, press, corr):
    """The predictor numerics of ChorinSplitVAMSolverJax (riemann 'hr'):
    Audusse rescaling on the momenta only.  Case configuration."""
    sq = _pad_to_square(pred)
    names = [str(s) for s in sq.state]
    skip = {names.index("h"), names.index("b"),
            *map(int, press.equation_to_state_index)}
    return NumericalSystemModel.from_system_model(
        sq, riemann=PositiveNonconservativeRusanov,
        reconstruction=ReconstructionSpec(), additional_systems=[press, corr],
        scaled_q_indices=[i for i in range(sq.n_variables) if i not in skip])


def old_run(time_order):
    pred, press, corr = split(sp.Symbol("dt"))
    cases.set_state_width(pred)
    s = ChorinSplitVAMSolverJax(pred, press, corr, time_order=time_order)
    Q0 = np.array(s.setup_simulation(mesh()))
    nc = s.nc
    Q0[:, :nc] = ic(nc, np.asarray(s._rt_mesh.cell_centers)[:, :nc])
    t0 = wall.perf_counter()
    Q, Qaux, *_ = s.run_jit_steps(jnp.asarray(DT), N_STEPS, jnp.asarray(Q0))
    return np.asarray(Q)[:, :nc], np.asarray(Qaux)[:, :nc], wall.perf_counter() - t0


def new_solver(time_order, **kw):
    pred, press, corr = split(_DT_SYMBOL)
    cases.set_state_width(pred)
    s = SymbolicChorinSolverJax(time_end=DT * N_STEPS, dt=DT, cfl=0.9,
                                time_order=time_order, **kw)
    Q, Qaux = s.setup(mesh(), pred_nsm(pred, press, corr), press, corr)
    nc = Q.shape[1]
    Q = jnp.asarray(ic(nc, np.asarray(s.mesh_np.cell_centers)[:, :nc]))
    return s, Q, s.initial_aux(Q, Qaux)


if __name__ == "__main__":
    for to in (1, 2):
        s, Q, Qaux = new_solver(to)
        if to == 1:
            print(s.source)
        t0 = wall.perf_counter()
        tN, Qn, An = s.run(Q, Qaux)
        Qn = np.asarray(Qn)
        dt_new = wall.perf_counter() - t0
        Qo, Ao, dt_old = old_run(to)
        d = np.abs(Qn - Qo).max()
        names = [str(v) for v in s.nsm.state]
        per = {n: float(np.abs(Qn[i] - Qo[i]).max()) for i, n in enumerate(names)}
        print(f"VAM(1) bump, time_order {to}: t = {float(tN):.6g}, "
              f"max|new-old| = {d:.2e} (rel {d / np.abs(Qo).max():.1e}), "
              f"aux {np.abs(np.asarray(An) - Ao).max():.2e}, "
              f"h_min {Qn[1].min():.3e}  [{dt_new:.0f}s / {dt_old:.0f}s]")
        print("   per row:", {k: f"{v:.1e}" for k, v in per.items()})
    print("G7_DONE")
