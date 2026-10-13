"""Where does the CFL check's time go?  Profile 256 steps of the Chorin bump
(constant dt) per configuration -- (wave speed, check every) -- and compare
per-op trace durations against the 'never' run: the extra wall time of a
configuration vs the extra time inside XLA ops, op by op.

    python trace_cfl_attribution.py
"""
import collections
import glob
import gzip
import json
import os
import shutil
import sys
import time as wall

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common  # noqa: F401
import escalante_bump as EB
import gate7_chorin as g7
import jax
import jax.numpy as jnp
import numpy as np
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_core.symsolver.settings import CflWaveSpeed
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax

TR = ("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/"
      "b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/trace_attr")
N = 256
CONFIGS = [(CflWaveSpeed.cells, 0), (CflWaveSpeed.cells, 1), (CflWaveSpeed.faces, 1),
           (CflWaveSpeed.cells, 10)]


def build(ws, every):
    pred, press, corr = EB.split(_DT_SYMBOL)
    s = SymbolicChorinSolverJax(time_end=N * EB.DT, dt=EB.DT, cfl=0.9, pressure_tol=1e-9,
                                pressure_maxit=200, cfl_check_every=every, cfl_wave_speed=ws)
    Q, A = s.setup(LSQMesh.create_1d(domain=EB.DOMAIN, n_inner_cells=EB.NCELLS),
                   g7.pred_nsm(pred, press, corr), press, corr)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(EB.initial_state(xc, Q.shape[0]))
    A = s.initial_aux(Q, A)
    f = lambda: s.run(Q, A)            # noqa: E731
    jax.block_until_ready(f())
    return f


def profile(f, tag):
    d = f"{TR}/{tag}"
    shutil.rmtree(d, ignore_errors=True)
    with jax.profiler.trace(d):
        jax.block_until_ready(f())
    ev = json.load(gzip.open(glob.glob(f"{d}/**/*.trace.json.gz", recursive=True)[0]))["traceEvents"]
    dur, cnt = collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and "dur" in e:
            dur[e["name"]] += e["dur"]
            cnt[e["name"]] += 1
    return dur, cnt


fs = {c: build(*c) for c in CONFIGS}
wall_t = {c: 1e9 for c in CONFIGS}
for _ in range(3):
    for c, f in fs.items():
        t0 = wall.perf_counter()
        jax.block_until_ready(f())
        wall_t[c] = min(wall_t[c], wall.perf_counter() - t0)
prof = {c: profile(f, f"{c[0]}_{c[1]}") for c, f in fs.items()}
base = prof[CONFIGS[0]][0]
print(f"{N} steps; wall per step (min of 3, no profiler) and op time per step "
      f"(trace events, us/step) relative to {CONFIGS[0]}")
for c in CONFIGS:
    dur, cnt = prof[c]
    eig = sum(v for k, v in dur.items() if k.startswith("eig"))
    neig = sum(v for k, v in cnt.items() if k.startswith("eig"))
    ex = dur.get("ThunkExecutor::Execute", 0)
    print(f"{str(c):14s} wall {1e3 * wall_t[c] / N:6.3f} ms/step | ThunkExecutor {ex / N / 1e3:6.3f} ms/step"
          f" | eig ops {neig / N:4.1f}/step, {eig / N / 1e3:6.3f} ms/step")
    if c != CONFIGS[0]:
        diff = {k: (dur[k] - base.get(k, 0)) / N for k in set(dur) | set(base)}
        top = sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:8]
        print("      largest per-op changes (us/step):",
              ", ".join(f"{k[:28]} {v:+.0f}" for k, v in top))
print("ATTR_DONE")
