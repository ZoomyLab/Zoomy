"""Profile 256 steps of the symbolic Chorin loop with the CFL check every
step; durations of every eig op (and the rest)."""
import os, sys, glob, gzip, json, collections, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa
import jax, jax.numpy as jnp, numpy as np
import escalante_bump as EB, gate7_chorin as g7
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
TR = "/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/trace_loop"
pred, press, corr = EB.split(_DT_SYMBOL)
s = SymbolicChorinSolverJax(time_end=256 * EB.DT, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                            pressure_maxit=200, cfl_check_every=1)
Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
               g7.pred_nsm(pred, press, corr), press, corr)
xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
Q = jnp.asarray(EB.initial_state(xc, Q.shape[0])); A = s.initial_aux(Q, A)
jax.block_until_ready(s.run(Q, A))
shutil.rmtree(TR, ignore_errors=True)
with jax.profiler.trace(TR):
    jax.block_until_ready(s.run(Q, A))
ev = json.load(gzip.open(glob.glob(f"{TR}/**/*.trace.json.gz", recursive=True)[0]))["traceEvents"]
dur, cnt = collections.Counter(), collections.Counter()
for e in ev:
    if e.get("ph") == "X" and "dur" in e:
        dur[e["name"]] += e["dur"]; cnt[e["name"]] += 1
for k in sorted(k for k in dur if k.startswith("eig")):
    print(f"{k:12s} calls {cnt[k]:5d}  mean {dur[k] / cnt[k]:8.1f} us  total {dur[k] / 1e3:8.1f} ms")
tot = dur.get("ThunkExecutor::Execute", 0)
print(f"total {tot / 1e3:.1f} ms for 256 steps; top non-eig:",
      [(k[:26], round(v / 1e3, 1)) for k, v in dur.most_common(40) if not k.startswith(("eig", "Thunk", "$", "Pjit", "Common"))][:6])
