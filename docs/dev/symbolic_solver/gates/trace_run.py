"""Profiler trace of a short full run (old run_simulation vs symbolic run):
time in eig vs everything else."""
import sys, glob, gzip, json, collections, shutil
import common as C
import jax, numpy as np
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
order, nx, t_end = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
TR = "/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/trace_run"
mesh, nsm = C.bump_2d(order, nx=nx)
o = HyperbolicSolver(time_end=t_end, compute_dt=C.adaptive(0.9, mesh), march_log_interval=0)
Qo, Ao = o.setup_simulation(mesh, nsm)
mesh, nsm = C.bump_2d(order, nx=nx)
s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.9)
Q, A = s.setup(mesh, nsm)
runs = {"old": lambda: o.run_simulation(Qo, Ao, write_output=False), "new": lambda: s.run(Q, A)}
for name, f in runs.items():
    jax.block_until_ready(f())
    d = f"{TR}_{name}"; shutil.rmtree(d, ignore_errors=True)
    with jax.profiler.trace(d):
        jax.block_until_ready(f())
    ev = json.load(gzip.open(glob.glob(f"{d}/**/*.trace.json.gz", recursive=True)[0]))["traceEvents"]
    agg = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and "dur" in e:
            agg[e["name"]] += e["dur"]
    eig = sum(v for k, v in agg.items() if k.startswith("eig"))
    total = agg.get("ThunkExecutor::Execute", 0)
    top = [(k, v) for k, v in agg.most_common(40) if not k.startswith(("eig", "Thunk", "$")) ][:8]
    print(f"{name}: total {total/1e3:.0f} ms, eig {eig/1e3:.0f} ms, rest {(total-eig)/1e3:.0f} ms; top rest:",
          [(k[:30], round(v / 1e3, 1)) for k, v in top], flush=True)
