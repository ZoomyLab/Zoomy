"""Gate 15: Chorin VAM(1) in 2 horizontal dimensions, a dam break over a
Gaussian bump on an n x n mesh; pressure solve GMRES vs AmgX (JAX-AMG, GPU).

    JAX_PLATFORMS=cuda,cpu python gate15_vam2d_amgx.py N [T_END]
"""
import os
import sys
import time as wall

import common  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import numpy as np
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
from zoomy_core.model.initial_conditions import Constant
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
from gate7_chorin import pred_nsm
from zoomy_core.symsolver.settings import LinearSolver

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05


def split():
    model = VAM(level=1, dimension=3, closures=[Newtonian(), StressFree()],
                boundary_conditions=[Dirichlet("right", on=f"P_{i}", value=0.0)
                                     for i in range(2)]
                + [Extrapolation(tag=t) for t in ("left", "bottom", "top")])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    return r.SM_pred, r.SM_press, r.SM_corr


out = {}
for ls in map(LinearSolver, os.environ.get("SOLVERS", "gmres,amgx").split(",")):
    pred, press, corr = split()
    kw = {"pressure_amgx": eval(os.environ["AMGX_CFG"])} if os.environ.get("AMGX_CFG") else {}
    s = SymbolicChorinSolverJax(time_end=T, cfl=0.9, pressure_linear_solver=ls, **kw)
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, -1.5, 1.5), nx=N, ny=N)
    Q, A = s.setup(mesh, pred_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    xc = np.asarray(s.mesh_np.cell_centers)[:2, :Q.shape[1]]
    q = np.zeros(Q.shape)
    b = 0.2 * np.exp(-(xc[0] ** 2 + xc[1] ** 2) / (2 * 0.2 ** 2))
    q[names.index("b")] = b
    q[names.index("h")] = np.where(xc[0] < -0.5, 0.34 - b, 0.1)
    Q = jnp.asarray(q)
    A = s.initial_aux(Q, A)
    t0 = wall.perf_counter()
    jax.block_until_ready(s.run(Q, A))
    first = wall.perf_counter() - t0
    t0 = wall.perf_counter()
    t, Qe, _ = s.run(Q, A)
    jax.block_until_ready(Qe)
    out[ls] = (np.asarray(Qe), wall.perf_counter() - t0, first)
    print(f"  {ls}: {out[ls][1]:.2f}s (first {first:.1f}s), finite {np.isfinite(out[ls][0]).all()}",
          flush=True)
    if os.environ.get("SAVE"):           # one solver per process (see AMG.md)
        np.save(f"{os.environ['SAVE']}_{ls}_{N}.npy", out[ls][0])
        prev = {o: f"{os.environ['SAVE']}_{o}_{N}.npy" for o in ("gmres", "amgx") if o != ls}
        for o, f in prev.items():
            if os.path.exists(f):
                d = np.abs(np.load(f) - out[ls][0]).max()
                print(f"  max|{o}-{ls}| {d:.2e} (rel {d / np.abs(out[ls][0]).max():.1e})")
if len(out) < 2:
    sys.exit()
d = np.abs(out["gmres"][0] - out["amgx"][0]).max()
print(f"VAM(1) 2-D {N}x{N} ({2 * N * N} pressure unknowns), t={T}: gmres {out['gmres'][1]:.2f}s | "
      f"amgx {out['amgx'][1]:.2f}s | speedup {out['gmres'][1] / out['amgx'][1]:.2f}x | "
      f"max|gmres-amgx| {d:.2e} on {jax.devices()[0].platform}")
