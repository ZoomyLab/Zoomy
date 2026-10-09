"""Profiler trace of one O2 step (old and new), time aggregated by op name."""
import sys, glob, gzip, json, collections, os, shutil
sys.argv = ["x", "2", "160"]
import profile_step as P   # builds old_step/new_step and prints its summary
import jax
TR = "/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/trace"
for name, f, args in (("old", P.old_step, (P.Qo, P.Ao)), ("new", P.new_step, (P.Q, P.A))):
    jf = jax.jit(f); jax.block_until_ready(jf(*args))
    d = f"{TR}_{name}"; shutil.rmtree(d, ignore_errors=True)
    with jax.profiler.trace(d):
        for _ in range(3):
            out = jf(*args)
        jax.block_until_ready(out)
    files = glob.glob(f"{d}/**/*.trace.json.gz", recursive=True)
    ev = json.load(gzip.open(files[0]))["traceEvents"]
    agg = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and "dur" in e:
            agg[e["name"]] += e["dur"]
    total = sum(v for k, v in agg.items())
    print(f"== {name}: top ops (us over 3 steps)")
    for k, v in agg.most_common(14):
        print(f"   {v:10.0f}  {k[:110]}")
