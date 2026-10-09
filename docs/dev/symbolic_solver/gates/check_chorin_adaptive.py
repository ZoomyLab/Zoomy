"""Chorin VAM(1) bump to T: adaptive dt with wave speeds from cells vs faces,
against the case's constant dt (4x below the CFL step).  Prints the
distance of each to the constant-dt run, the step counts, h_min, finite;
also the bottom pressure RMS vs the experiment if available."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa: F401
import escalante_bump as EB
import gate7_chorin as g7
import jax.numpy as jnp
import numpy as np
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

T_END = float(sys.argv[1]) if len(sys.argv) > 1 else EB.T_END
out = {}
for label, kw in (("const dt", dict(dt=EB.DT)), ("adaptive cells", dict(cfl_wave_speed="cells")),
                  ("adaptive faces", dict(cfl_wave_speed="faces"))):
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=T_END, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, **kw)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
    t, Qe, _ = s.run(Q, s.initial_aux(Q, A))
    out[label] = np.asarray(Qe)
    names = [str(v) for v in s.nsm.state]
    h = out[label][names.index("h")]
    print(f"{label:15s} t {float(t):.4g}  h_min {h.min():.4f}  finite {np.isfinite(out[label]).all()}"
          + ("" if label == "const dt" else
             f"  max|Q - const dt| {np.abs(out[label] - out['const dt']).max():.3e}"
             f"  (h: {np.abs(h - out['const dt'][names.index('h')]).max():.3e})"), flush=True)
d = np.abs(out["adaptive cells"] - out["adaptive faces"])
print("max|cells - faces| per row:", " ".join(f"{n}={v:.2e}" for n, v in zip(names, d.max(axis=1))))
print("max|Q| per row:", " ".join(f"{n}={v:.2e}" for n, v in zip(names, np.abs(out['const dt']).max(axis=1))))
